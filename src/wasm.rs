//! WASM bindings for `OxiRAG`.
//!
//! This module provides JavaScript-compatible APIs for running `OxiRAG`
//! in browsers and edge environments.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use crate::pipeline::{Pipeline, PipelineConfig};
use crate::types::{Document, Query};

/// Initialize logging for WASM.
#[wasm_bindgen(start)]
pub fn init() {
    // Human-readable panic messages in the browser console. `panic = "abort"`
    // makes a panic an uncatchable trap, so the message is all a caller gets.
    console_error_panic_hook::set_once();
}

/// A WASM-compatible RAG engine.
#[wasm_bindgen]
pub struct WasmRagEngine {
    pipeline: Pipeline<
        EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
        RuleBasedSpeculator,
        JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
    >,
}

#[wasm_bindgen]
impl WasmRagEngine {
    /// Create a new WASM RAG engine.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(dimension: usize) -> Self {
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

        let config = PipelineConfig::default();

        Self {
            pipeline: Pipeline::new(echo, speculator, judge, config),
        }
    }

    /// Index a document.
    ///
    /// # Arguments
    /// * `content` - The document content
    /// * `title` - Optional document title
    ///
    /// # Returns
    /// The document ID as a string
    pub async fn index(
        &mut self,
        content: String,
        title: Option<String>,
    ) -> Result<String, JsValue> {
        let mut doc = Document::new(content);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }

        let id = doc.id.clone();

        crate::layer1_echo::Echo::index(self.pipeline.echo_mut(), doc)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(id.to_string())
    }

    /// Query the RAG engine through the full four-layer pipeline.
    ///
    /// Routes the query through Echo (Layer 1), `RuleBasedSpeculator` (Layer 2), and
    /// `JudgeImpl` (Layer 3) before returning a JSON-serialised [`crate::types::PipelineOutput`].
    /// All three layers are backed by WASM-compatible mock/rule-based implementations so
    /// no native ML dependencies are required.
    ///
    /// # Arguments
    /// * `query_text` - The query string to retrieve and verify
    /// * `top_k` - Maximum number of documents to retrieve from the Echo layer
    ///
    /// # Returns
    /// JSON string containing a [`crate::types::PipelineOutput`] with search results,
    /// speculation decision, verification result, final answer, and per-layer timing.
    ///
    /// # Errors
    /// Returns a [`JsValue`] error string if the pipeline fails or JSON serialisation fails.
    pub async fn query(&self, query_text: String, top_k: usize) -> Result<String, JsValue> {
        use crate::pipeline::RagPipeline;

        let query = Query::new(query_text).with_top_k(top_k);

        // Run the full pipeline (Echo → Speculator → Judge).
        // All providers are WASM-compatible mock/rule-based implementations —
        // no candle or native-only dependencies are involved.
        let output = self
            .pipeline
            .process(query)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Serialise the full pipeline output (includes search_results, draft,
        // speculation, verification, final_answer, confidence, layers_used, timing).
        let json = serde_json::to_string(&output).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(json)
    }

    /// Get the number of indexed documents.
    pub async fn count(&self) -> usize {
        crate::layer1_echo::Echo::count(self.pipeline.echo()).await
    }

    /// Clear all indexed documents.
    pub async fn clear(&mut self) -> Result<(), JsValue> {
        crate::layer1_echo::Echo::clear(self.pipeline.echo_mut())
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Query the index and return results as a JS `Array` of JSON strings.
    ///
    /// Each element in the returned array is a JSON-serialised
    /// [`crate::types::SearchResult`].  The array is ordered by descending
    /// similarity score, and contains at most `top_k` elements.
    ///
    /// This method is the streaming-friendly complement to [`WasmRagEngine::query`]:
    /// the caller can iterate over the array incrementally rather than waiting for
    /// the full pipeline to complete.
    ///
    /// # Arguments
    ///
    /// * `query_text` - The query string.
    /// * `top_k` - Maximum number of results to return.
    ///
    /// # Returns
    ///
    /// A `Promise<Array<string>>` where each string is a JSON-serialised
    /// `SearchResult`.
    ///
    /// # Errors
    ///
    /// Returns a [`JsValue`] error string if the embedding provider or vector
    /// store fails, or if JSON serialisation fails.
    ///
    /// # JS example
    ///
    /// ```js
    /// const results = await engine.query_search_array("what is Rust?", 5);
    /// for (const item of results) {
    ///     const result = JSON.parse(item);
    ///     console.log(result.document.content, result.score);
    /// }
    /// ```
    pub async fn query_search_array(
        &self,
        query_text: String,
        top_k: usize,
    ) -> Result<js_sys::Array, JsValue> {
        use crate::layer1_echo::Echo;

        let results = Echo::search(self.pipeline.echo(), &query_text, top_k, None)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let array = js_sys::Array::new();
        for result in &results {
            let json =
                serde_json::to_string(result).map_err(|e| JsValue::from_str(&e.to_string()))?;
            array.push(&JsValue::from_str(&json));
        }

        Ok(array)
    }
}

/// Compute cosine similarity between two vectors.
#[wasm_bindgen]
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::layer1_echo::similarity::cosine_similarity(a, b)
}

/// Normalize a vector to unit length.
#[wasm_bindgen]
#[must_use]
pub fn normalize_vector(v: &[f32]) -> Vec<f32> {
    crate::layer1_echo::similarity::normalize(v)
}

/// Parse configuration from JSON.
#[wasm_bindgen]
pub fn parse_config(json: &str) -> Result<JsValue, JsValue> {
    let config: crate::config::OxiRagConfig =
        serde_json::from_str(json).map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&config).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get the library version.
#[wasm_bindgen]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_wasm_engine_creation() {
        let engine = WasmRagEngine::new(64);
        assert_eq!(engine.count().await, 0);
    }

    #[wasm_bindgen_test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[wasm_bindgen_test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
