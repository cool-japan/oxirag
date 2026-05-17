//! WASM Web Worker entry point for `OxiRAG`.
//!
//! This module provides a complete message-passing bridge that lets the RAG
//! engine run in a dedicated [`Web Worker`][mdn-worker] thread, keeping the
//! main thread free of latency spikes caused by embedding and search.
//!
//! # Message protocol
//!
//! Messages are exchanged as plain JSON objects.  The worker never uses
//! `SharedArrayBuffer` so no COOP/COEP headers are required.
//!
//! ## Request shape
//!
//! ```json
//! { "id": "<req-id>", "type": "<verb>", "payload": { ... } }
//! ```
//!
//! | `type`  | Payload fields                            | Description                |
//! |---------|-------------------------------------------|----------------------------|
//! | `index` | `content: String`, `title?: String`       | Embed and store a document |
//! | `query` | `query: String`, `top_k?: usize`          | Semantic search            |
//! | `count` | _(empty)_                                 | Return document count      |
//! | `clear` | _(empty)_                                 | Delete all documents       |
//!
//! ## Response shape
//!
//! ```json
//! { "id": "<req-id>", "ok": true,  "result": <value> }
//! { "id": "<req-id>", "ok": false, "error":  "<message>" }
//! ```
//!
//! # JS usage example
//!
//! ```js
//! // In your main thread:
//! import init, { worker_init, worker_handle_message } from '../../pkg/oxirag.js';
//!
//! const worker = new Worker(new URL('./worker.js', import.meta.url));
//! worker_init(384);
//! worker.postMessage({ id: 'r1', type: 'query', payload: { query: 'hello', top_k: 5 } });
//! worker.onmessage = (e) => console.log(e.data);
//! ```

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use std::cell::RefCell;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use crate::pipeline::{Pipeline, PipelineConfig};

// ────────────────────────────────────────────────────────────────────────────
// Engine type alias
// ────────────────────────────────────────────────────────────────────────────

/// The concrete RAG pipeline type used inside the worker.
type WasmEngine = Pipeline<
    EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
    RuleBasedSpeculator,
    JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
>;

// ────────────────────────────────────────────────────────────────────────────
// Thread-local engine storage
// ────────────────────────────────────────────────────────────────────────────

thread_local! {
    /// The singleton RAG engine stored in the worker thread.
    ///
    /// Workers do not share memory with the main thread, so `thread_local!` is
    /// the correct pattern for per-worker state in WASM.
    static ENGINE: RefCell<Option<WasmEngine>> = const { RefCell::new(None) };
}

// ────────────────────────────────────────────────────────────────────────────
// Message types
// ────────────────────────────────────────────────────────────────────────────

/// Incoming request from the main thread.
#[derive(Debug, Deserialize)]
struct WorkerRequest {
    /// Caller-supplied correlation id, echoed back in the response.
    id: String,
    /// Verb — one of `"index"`, `"query"`, `"count"`, `"clear"`.
    #[serde(rename = "type")]
    msg_type: String,
    /// Verb-specific parameters.
    #[serde(default)]
    payload: serde_json::Value,
}

/// Outgoing response sent back to the main thread.
#[derive(Debug, Serialize)]
struct WorkerResponse {
    /// Echoed request id.
    id: String,
    /// `true` on success, `false` on error.
    ok: bool,
    /// Present on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    /// Present on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl WorkerResponse {
    fn ok(id: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    fn err(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ok: false,
            result: None,
            error: Some(message.into()),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Exported WASM bindings
// ────────────────────────────────────────────────────────────────────────────

/// Initialise the worker-local RAG engine.
///
/// Call this once from your Worker's `onmessage` handler before processing
/// any messages.  Calling it again replaces the existing engine.
///
/// # Arguments
///
/// * `dimension` – Embedding dimension expected by the `MockEmbeddingProvider`.
///   Use `384` for MiniLM-L6 or `1536` for OpenAI `text-embedding-3-small`.
#[wasm_bindgen]
pub fn worker_init(dimension: usize) {
    let echo = EchoLayer::new(
        MockEmbeddingProvider::new(dimension),
        InMemoryVectorStore::new(dimension),
    );
    let speculator = RuleBasedSpeculator::default();
    let judge = JudgeImpl::new(
        AdvancedClaimExtractor::new(),
        MockSmtVerifier::default(),
        JudgeConfig::default(),
    );
    let pipeline = Pipeline::new(echo, speculator, judge, PipelineConfig::default());

    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(pipeline);
    });
}

/// Process a single JSON-encoded worker message and return a JSON response.
///
/// The input `msg` must be either:
/// - A `JsValue` containing a JS object that can be serialised to
///   [`WorkerRequest`], or
/// - A `JsValue` string containing a JSON-encoded [`WorkerRequest`].
///
/// The return value is always a `JsValue` containing a JSON string that
/// deserialises to [`WorkerResponse`].
///
/// # Errors
///
/// If the message cannot be parsed, an error response is returned with
/// `id = "unknown"`.
#[wasm_bindgen]
pub async fn worker_handle_message(msg: JsValue) -> JsValue {
    // Deserialise the request — accept both JS objects and JSON strings.
    let request: WorkerRequest = match deserialise_request(msg) {
        Ok(r) => r,
        Err(e) => {
            return json_response_value(&WorkerResponse::err("unknown", e));
        }
    };

    let req_id = request.id.clone();
    let response = dispatch(request).await;
    let response = response.unwrap_or_else(|e| WorkerResponse::err(req_id, e));

    json_response_value(&response)
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Attempt to deserialise a [`WorkerRequest`] from either a JS object or a
/// JSON string `JsValue`.
fn deserialise_request(val: JsValue) -> Result<WorkerRequest, String> {
    // Try direct object deserialisation first.
    if let Ok(req) = serde_wasm_bindgen::from_value::<WorkerRequest>(val.clone()) {
        return Ok(req);
    }
    // Fall back to treating the value as a JSON string.
    let json_str = val
        .as_string()
        .ok_or_else(|| "message is neither a JS object nor a JSON string".to_string())?;
    serde_json::from_str(&json_str).map_err(|e| format!("JSON parse error: {e}"))
}

/// Dispatch a parsed request to the appropriate engine method.
async fn dispatch(req: WorkerRequest) -> Result<WorkerResponse, String> {
    match req.msg_type.as_str() {
        "index" => handle_index(req).await,
        "query" => handle_query(req).await,
        "count" => handle_count(req).await,
        "clear" => handle_clear(req).await,
        other => Err(format!("unknown message type: {other:?}")),
    }
}

/// Serialise a [`WorkerResponse`] to a `JsValue` containing a JSON string.
fn json_response_value(response: &WorkerResponse) -> JsValue {
    match serde_json::to_string(response) {
        Ok(s) => JsValue::from_str(&s),
        Err(e) => JsValue::from_str(&format!(
            r#"{{"id":"error","ok":false,"error":"serialisation failed: {e}"}}"#
        )),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Verb handlers
// ────────────────────────────────────────────────────────────────────────────

async fn handle_index(req: WorkerRequest) -> Result<WorkerResponse, String> {
    use crate::layer1_echo::Echo;
    use crate::types::Document;

    let content = req
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or("index payload missing 'content' string")?
        .to_string();

    let title = req
        .payload
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let mut doc = Document::new(content);
    if let Some(t) = title {
        doc = doc.with_title(t);
    }

    // Take the engine out of thread-local storage so we can drive the async
    // `Echo::index` future without holding a `RefCell` borrow across an await
    // point (which Rust does not allow).
    let engine_taken: Option<WasmEngine> = ENGINE.with(|cell| cell.borrow_mut().take());

    let mut engine = engine_taken
        .ok_or_else(|| "worker not initialised — call worker_init() first".to_string())?;

    let indexed_id = Echo::index(&mut engine, doc)
        .await
        .map_err(|e| e.to_string())?;

    // Restore the engine to thread-local storage.
    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(engine);
    });

    Ok(WorkerResponse::ok(
        req.id,
        serde_json::Value::String(indexed_id.to_string()),
    ))
}

async fn handle_query(req: WorkerRequest) -> Result<WorkerResponse, String> {
    use crate::pipeline::RagPipeline;
    use crate::types::Query;

    let query_text = req
        .payload
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("query payload missing 'query' string")?
        .to_string();

    let top_k = req
        .payload
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map_or(5, |n| n as usize);

    let query = Query::new(query_text).with_top_k(top_k);

    let engine_taken: Option<WasmEngine> = ENGINE.with(|cell| cell.borrow_mut().take());
    let engine = engine_taken
        .ok_or_else(|| "worker not initialised — call worker_init() first".to_string())?;

    let output = engine.process(query).await.map_err(|e| e.to_string())?;

    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(engine);
    });

    let json_val: serde_json::Value = serde_json::to_value(&output).map_err(|e| e.to_string())?;

    Ok(WorkerResponse::ok(req.id, json_val))
}

async fn handle_count(req: WorkerRequest) -> Result<WorkerResponse, String> {
    use crate::layer1_echo::Echo;

    let engine_taken: Option<WasmEngine> = ENGINE.with(|cell| cell.borrow_mut().take());
    let engine = engine_taken
        .ok_or_else(|| "worker not initialised — call worker_init() first".to_string())?;

    // `Pipeline::echo()` returns a reference to the `EchoLayer`, on which
    // `Echo::count` is defined.
    let count = Echo::count(engine.echo()).await;

    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(engine);
    });

    Ok(WorkerResponse::ok(
        req.id,
        serde_json::Value::Number(serde_json::Number::from(count)),
    ))
}

async fn handle_clear(req: WorkerRequest) -> Result<WorkerResponse, String> {
    use crate::layer1_echo::Echo;

    let engine_taken: Option<WasmEngine> = ENGINE.with(|cell| cell.borrow_mut().take());
    let mut engine = engine_taken
        .ok_or_else(|| "worker not initialised — call worker_init() first".to_string())?;

    Echo::clear(engine.echo_mut())
        .await
        .map_err(|e| e.to_string())?;

    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(engine);
    });

    Ok(WorkerResponse::ok(req.id, serde_json::Value::Bool(true)))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_worker_init_sets_engine() {
        worker_init(32);
        ENGINE.with(|cell| {
            assert!(
                cell.borrow().is_some(),
                "engine should be set after worker_init"
            );
        });
    }

    #[wasm_bindgen_test]
    async fn test_worker_count_after_init() {
        worker_init(32);
        let req = serde_json::json!({ "id": "t1", "type": "count", "payload": {} });
        let msg = serde_wasm_bindgen::to_value(&req).expect("serialise");
        let resp_val = worker_handle_message(msg).await;
        let resp_str = resp_val.as_string().expect("response should be a string");
        let parsed: serde_json::Value =
            serde_json::from_str(&resp_str).expect("valid JSON response");
        assert_eq!(parsed["ok"], true, "count should succeed");
        assert_eq!(parsed["result"], 0, "fresh engine should have count=0");
    }

    #[wasm_bindgen_test]
    async fn test_worker_unknown_type_returns_error() {
        worker_init(32);
        let req = serde_json::json!({ "id": "t2", "type": "bogus", "payload": {} });
        let msg = serde_wasm_bindgen::to_value(&req).expect("serialise");
        let resp_val = worker_handle_message(msg).await;
        let resp_str = resp_val.as_string().expect("response should be a string");
        let parsed: serde_json::Value =
            serde_json::from_str(&resp_str).expect("valid JSON response");
        assert_eq!(parsed["ok"], false, "unknown type should return error");
    }
}
