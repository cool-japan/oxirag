//! WASM integration tests for the Web Worker message-passing bridge.
//!
//! These tests run in a real browser via `wasm-pack test --chrome --headless`.
//! They verify that the worker's JSON message protocol works end-to-end.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[cfg(feature = "wasm")]
mod worker_tests {
    use super::*;
    use oxirag::wasm_worker::{worker_handle_message, worker_init};

    /// Initialise the worker and verify that a `count` request returns 0.
    #[wasm_bindgen_test]
    async fn test_worker_init_and_count() {
        worker_init(64);

        let req = serde_json::json!({
            "id": "test-1",
            "type": "count",
            "payload": {}
        });

        let msg: JsValue = serde_wasm_bindgen::to_value(&req).expect("serialise request");
        let resp: JsValue = worker_handle_message(msg).await;

        let resp_str = resp.as_string().expect("response should be a JSON string");
        let parsed: serde_json::Value =
            serde_json::from_str(&resp_str).expect("response should be valid JSON");

        assert_eq!(
            parsed["id"], "test-1",
            "response id should match request id"
        );
        assert_eq!(parsed["ok"], true, "count should succeed");
        assert_eq!(
            parsed["result"], 0,
            "fresh engine should have zero documents"
        );
    }

    /// Index a document and verify that count increments to 1.
    #[wasm_bindgen_test]
    async fn test_worker_index_increments_count() {
        worker_init(64);

        // Clear first to isolate this test from others.
        let clear_req = serde_json::json!({ "id": "clr", "type": "clear", "payload": {} });
        let clear_msg = serde_wasm_bindgen::to_value(&clear_req).expect("serialise clear");
        worker_handle_message(clear_msg).await;

        // Index one document.
        let index_req = serde_json::json!({
            "id": "idx-1",
            "type": "index",
            "payload": { "content": "WASM worker test document" }
        });
        let index_msg = serde_wasm_bindgen::to_value(&index_req).expect("serialise index");
        let index_resp = worker_handle_message(index_msg).await;
        let index_str = index_resp
            .as_string()
            .expect("index response should be a string");
        let index_parsed: serde_json::Value = serde_json::from_str(&index_str).expect("valid JSON");
        assert_eq!(index_parsed["ok"], true, "index should succeed");

        // Count should now be 1.
        let count_req = serde_json::json!({ "id": "cnt-1", "type": "count", "payload": {} });
        let count_msg = serde_wasm_bindgen::to_value(&count_req).expect("serialise count");
        let count_resp = worker_handle_message(count_msg).await;
        let count_str = count_resp
            .as_string()
            .expect("count response should be a string");
        let count_parsed: serde_json::Value = serde_json::from_str(&count_str).expect("valid JSON");
        assert_eq!(count_parsed["ok"], true, "count should succeed");
        assert_eq!(
            count_parsed["result"], 1,
            "count should be 1 after one index"
        );
    }

    /// An unknown message type should produce an error response (ok = false).
    #[wasm_bindgen_test]
    async fn test_worker_unknown_type_is_error() {
        worker_init(64);

        let req = serde_json::json!({
            "id": "bad-1",
            "type": "nonexistent_verb",
            "payload": {}
        });

        let msg = serde_wasm_bindgen::to_value(&req).expect("serialise");
        let resp = worker_handle_message(msg).await;
        let resp_str = resp.as_string().expect("response should be a string");
        let parsed: serde_json::Value = serde_json::from_str(&resp_str).expect("valid JSON");

        assert_eq!(parsed["id"], "bad-1", "id should be echoed");
        assert_eq!(parsed["ok"], false, "unknown type should return ok=false");
        assert!(
            parsed["error"].is_string(),
            "error field should be present and a string"
        );
    }

    /// Clear should reset the document count to 0.
    #[wasm_bindgen_test]
    async fn test_worker_clear_resets_count() {
        worker_init(64);

        // Index a document so there is something to clear.
        let index_req = serde_json::json!({
            "id": "idx-2",
            "type": "index",
            "payload": { "content": "document to clear" }
        });
        let index_msg = serde_wasm_bindgen::to_value(&index_req).expect("serialise");
        worker_handle_message(index_msg).await;

        // Clear.
        let clear_req = serde_json::json!({ "id": "clr-1", "type": "clear", "payload": {} });
        let clear_msg = serde_wasm_bindgen::to_value(&clear_req).expect("serialise");
        let clear_resp = worker_handle_message(clear_msg).await;
        let clear_str = clear_resp
            .as_string()
            .expect("clear response should be a string");
        let clear_parsed: serde_json::Value = serde_json::from_str(&clear_str).expect("valid JSON");
        assert_eq!(clear_parsed["ok"], true, "clear should succeed");

        // Count should now be 0.
        let count_req = serde_json::json!({ "id": "cnt-2", "type": "count", "payload": {} });
        let count_msg = serde_wasm_bindgen::to_value(&count_req).expect("serialise");
        let count_resp = worker_handle_message(count_msg).await;
        let count_str = count_resp
            .as_string()
            .expect("count response should be a string");
        let count_parsed: serde_json::Value = serde_json::from_str(&count_str).expect("valid JSON");
        assert_eq!(count_parsed["result"], 0, "count should be 0 after clear");
    }
}
