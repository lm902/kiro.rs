//! OpenAI Chat Completions 兼容端点

use std::collections::HashMap;

use axum::{
    Json as JsonExtractor,
    body::{Body, to_bytes},
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use super::{
    handlers::post_messages,
    middleware::AppState,
    types::{ErrorResponse, Message as AnthropicMessage, MessagesRequest, SystemMessage, Tool},
};

const MAX_BODY_SIZE: usize = 50 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<i32>,
    pub max_completion_tokens: Option<i32>,
    pub tools: Option<Vec<OpenAIToolDefinition>>,
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: Option<Value>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Option<OpenAIFunctionDefinition>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

pub async fn post_chat_completions(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<OpenAIChatRequest>,
) -> Response {
    let stream = payload.stream;
    let model = payload.model.clone();
    let anthropic_payload = match convert_openai_request(payload) {
        Ok(v) => v,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("invalid_request_error", msg)),
            )
                .into_response();
        }
    };

    let response = post_messages(State(state), JsonExtractor(anthropic_payload)).await;
    transform_anthropic_response_to_openai(response, stream, &model).await
}

fn convert_openai_request(payload: OpenAIChatRequest) -> Result<MessagesRequest, String> {
    let mut anthropic_messages = Vec::new();
    let mut system_messages = Vec::new();

    for msg in payload.messages {
        match msg.role.as_str() {
            "system" => {
                let text = extract_text_content(msg.content.as_ref());
                if !text.is_empty() {
                    system_messages.push(SystemMessage { text });
                }
            }
            "user" => {
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: convert_user_content(msg.content)?,
                });
            }
            "assistant" => {
                anthropic_messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: convert_assistant_content(msg.content, msg.tool_calls),
                });
            }
            "tool" => {
                let tool_use_id = msg
                    .tool_call_id
                    .ok_or_else(|| "tool 消息缺少 tool_call_id".to_string())?;
                anthropic_messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: json!([
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": extract_text_content(msg.content.as_ref())
                        }
                    ]),
                });
            }
            _ => {}
        }
    }

    if anthropic_messages.is_empty() {
        return Err("消息列表为空".to_string());
    }

    let tools = payload.tools.map(|tools| {
        tools
            .into_iter()
            .filter_map(|t| {
                if t.tool_type != "function" {
                    return None;
                }
                let function = t.function?;
                let input_schema = function
                    .parameters
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                Some(Tool {
                    tool_type: None,
                    name: function.name,
                    description: function.description.unwrap_or_default(),
                    input_schema,
                    max_uses: None,
                })
            })
            .collect()
    });

    Ok(MessagesRequest {
        model: payload.model,
        max_tokens: payload
            .max_completion_tokens
            .or(payload.max_tokens)
            .unwrap_or(4096),
        messages: anthropic_messages,
        stream: payload.stream,
        system: if system_messages.is_empty() {
            None
        } else {
            Some(system_messages)
        },
        tools,
        tool_choice: payload.tool_choice,
        thinking: None,
        output_config: None,
        metadata: None,
    })
}

fn convert_user_content(content: Option<Value>) -> Result<Value, String> {
    let Some(content) = content else {
        return Ok(Value::String(String::new()));
    };

    match content {
        Value::String(s) => Ok(Value::String(s)),
        Value::Array(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                let Some(part_type) = part.get("type").and_then(|v| v.as_str()) else {
                    continue;
                };
                if part_type == "text" {
                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                        blocks.push(json!({"type":"text","text":text}));
                    }
                } else if part_type == "image_url" {
                    let maybe_url = part
                        .get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(|v| v.as_str());
                    if let Some(url) = maybe_url {
                        if let Some(image_block) = convert_data_url_to_image_block(url) {
                            blocks.push(image_block);
                        }
                    }
                }
            }
            Ok(Value::Array(blocks))
        }
        _ => Ok(Value::String(content.to_string())),
    }
}

fn convert_assistant_content(content: Option<Value>, tool_calls: Option<Vec<OpenAIToolCall>>) -> Value {
    let mut blocks = Vec::new();
    let text = extract_text_content(content.as_ref());
    if !text.is_empty() {
        blocks.push(json!({"type":"text","text":text}));
    }

    if let Some(tool_calls) = tool_calls {
        for call in tool_calls {
            if call.call_type != "function" {
                continue;
            }
            let input = serde_json::from_str::<Value>(&call.function.arguments)
                .unwrap_or_else(|_| json!({}));
            blocks.push(json!({
                "type":"tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": input
            }));
        }
    }

    if blocks.is_empty() {
        Value::String(" ".to_string())
    } else {
        Value::Array(blocks)
    }
}

fn convert_data_url_to_image_block(url: &str) -> Option<Value> {
    if !url.starts_with("data:") {
        return None;
    }
    let (_, rest) = url.split_once(':')?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.ends_with(";base64") {
        return None;
    }
    let media_type = meta.trim_end_matches(";base64");
    Some(json!({
        "type":"image",
        "source":{
            "type":"base64",
            "media_type":media_type,
            "data":data
        }
    }))
}

fn extract_text_content(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        _ => content.to_string(),
    }
}

async fn transform_anthropic_response_to_openai(
    response: Response,
    stream: bool,
    model: &str,
) -> Response {
    let status = response.status();
    let body = response.into_body();
    let bytes = match to_bytes(body, MAX_BODY_SIZE).await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "api_error",
                    format!("读取响应失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    if !status.is_success() {
        return Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| Response::new(Body::empty()));
    }

    if stream {
        let sse = convert_stream_body(&bytes, model);
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(Body::from(sse))
            .unwrap_or_else(|_| Response::new(Body::empty()));
    }

    match serde_json::from_slice::<Value>(&bytes) {
        Ok(v) => {
            let mapped = map_non_stream_response(v, model);
            (StatusCode::OK, Json(mapped)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse::new(
                "api_error",
                format!("响应 JSON 解析失败: {}", e),
            )),
        )
            .into_response(),
    }
}

fn map_non_stream_response(anthropic: Value, model: &str) -> Value {
    let mut content_text = String::new();
    let mut tool_calls = Vec::new();

    if let Some(content) = anthropic.get("content").and_then(|v| v.as_array()) {
        for block in content {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        content_text.push_str(text);
                    }
                }
                Some("tool_use") => {
                    tool_calls.push(json!({
                        "id": block.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "arguments": block.get("input").cloned().unwrap_or_else(|| json!({})).to_string()
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let stop_reason = anthropic
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn");
    let finish_reason = map_finish_reason(stop_reason);

    let prompt_tokens = anthropic
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let completion_tokens = anthropic
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    json!({
        "id": anthropic.get("id").cloned().unwrap_or_else(|| json!(format!("chatcmpl_{}", Uuid::new_v4().to_string().replace('-', "")))),
        "object": "chat.completion",
        "created": Utc::now().timestamp(),
        "model": anthropic.get("model").and_then(|v| v.as_str()).unwrap_or(model),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": if content_text.is_empty() { Value::Null } else { json!(content_text) },
                "tool_calls": if tool_calls.is_empty() { Value::Null } else { json!(tool_calls) }
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

fn convert_stream_body(bytes: &[u8], model: &str) -> String {
    let input = String::from_utf8_lossy(bytes);
    let created = Utc::now().timestamp();
    let mut chat_id = format!("chatcmpl_{}", Uuid::new_v4().to_string().replace('-', ""));
    let mut tool_indices: HashMap<i64, usize> = HashMap::new();
    let mut next_tool_index = 0usize;
    let mut output = String::new();

    for record in input.split("\n\n") {
        if record.trim().is_empty() {
            continue;
        }
        let mut event_name: Option<&str> = None;
        let mut data_raw = String::new();

        for line in record.lines() {
            if let Some(v) = line.strip_prefix("event: ") {
                event_name = Some(v.trim());
            } else if let Some(v) = line.strip_prefix("data: ") {
                if !data_raw.is_empty() {
                    data_raw.push('\n');
                }
                data_raw.push_str(v);
            }
        }

        let Some(event) = event_name else {
            continue;
        };
        if event == "ping" || data_raw.is_empty() {
            continue;
        }

        let Ok(data) = serde_json::from_str::<Value>(&data_raw) else {
            continue;
        };

        match event {
            "message_start" => {
                if let Some(id) = data
                    .get("message")
                    .and_then(|m| m.get("id"))
                    .and_then(|v| v.as_str())
                {
                    chat_id = id.to_string();
                }
                output.push_str(&format!(
                    "data: {}\n\n",
                    json!({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role":"assistant"},
                            "finish_reason": Value::Null
                        }]
                    })
                ));
            }
            "content_block_start" => {
                if data
                    .get("content_block")
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    let idx = data.get("index").and_then(|v| v.as_i64()).unwrap_or(-1);
                    let tool_idx = *tool_indices.entry(idx).or_insert_with(|| {
                        let i = next_tool_index;
                        next_tool_index += 1;
                        i
                    });
                    output.push_str(&format!(
                        "data: {}\n\n",
                        json!({
                            "id": chat_id,
                            "object":"chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices":[
                                {
                                    "index":0,
                                    "delta":{
                                        "tool_calls":[
                                            {
                                                "index": tool_idx,
                                                "id": data.get("content_block").and_then(|v| v.get("id")).and_then(|v| v.as_str()).unwrap_or(""),
                                                "type":"function",
                                                "function":{
                                                    "name": data.get("content_block").and_then(|v| v.get("name")).and_then(|v| v.as_str()).unwrap_or(""),
                                                    "arguments":""
                                                }
                                            }
                                        ]
                                    },
                                    "finish_reason": Value::Null
                                }
                            ]
                        })
                    ));
                }
            }
            "content_block_delta" => {
                let delta_type = data
                    .get("delta")
                    .and_then(|d| d.get("type"))
                    .and_then(|v| v.as_str());
                match delta_type {
                    Some("text_delta") => {
                        if let Some(text) = data
                            .get("delta")
                            .and_then(|d| d.get("text"))
                            .and_then(|v| v.as_str())
                        {
                            output.push_str(&format!(
                                "data: {}\n\n",
                                json!({
                                    "id": chat_id,
                                    "object":"chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices":[{"index":0,"delta":{"content":text},"finish_reason":Value::Null}]
                                })
                            ));
                        }
                    }
                    Some("input_json_delta") => {
                        let idx = data.get("index").and_then(|v| v.as_i64()).unwrap_or(-1);
                        let tool_idx = *tool_indices.entry(idx).or_insert_with(|| {
                            let i = next_tool_index;
                            next_tool_index += 1;
                            i
                        });
                        let partial = data
                            .get("delta")
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        output.push_str(&format!(
                            "data: {}\n\n",
                            json!({
                                "id": chat_id,
                                "object":"chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices":[
                                    {
                                        "index":0,
                                        "delta":{
                                            "tool_calls":[
                                                {
                                                    "index":tool_idx,
                                                    "function":{"arguments":partial}
                                                }
                                            ]
                                        },
                                        "finish_reason":Value::Null
                                    }
                                ]
                            })
                        ));
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                let stop_reason = data
                    .get("delta")
                    .and_then(|v| v.get("stop_reason"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("end_turn");
                output.push_str(&format!(
                    "data: {}\n\n",
                    json!({
                        "id": chat_id,
                        "object":"chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices":[{"index":0,"delta":{},"finish_reason":map_finish_reason(stop_reason)}]
                    })
                ));
            }
            _ => {}
        }
    }

    output.push_str("data: [DONE]\n\n");
    output
}

fn map_finish_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" | "model_context_window_exceeded" => "length",
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_stream_mapping_handles_tool_calls() {
        let input = json!({
            "id":"msg_1",
            "model":"claude-sonnet-5",
            "stop_reason":"tool_use",
            "content":[
                {"type":"text","text":"hello"},
                {"type":"tool_use","id":"tool_1","name":"weather","input":{"city":"shanghai"}}
            ],
            "usage":{"input_tokens":10,"output_tokens":5}
        });
        let output = map_non_stream_response(input, "claude-sonnet-5");
        assert_eq!(output["object"], "chat.completion");
        assert_eq!(output["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            output["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
    }

    #[test]
    fn stream_mapping_outputs_done() {
        let sse = "event: message_start\ndata: {\"message\":{\"id\":\"msg_123\"}}\n\n\
event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let output = convert_stream_body(sse.as_bytes(), "claude-sonnet-5");
        assert!(output.contains("\"chat.completion.chunk\""));
        assert!(output.contains("\"content\":\"Hi\""));
        assert!(output.contains("data: [DONE]"));
    }
}
