#![cfg(feature = "nodejs")]
#![allow(missing_docs)]
//! napi-rs async pipeline wrapper.
//!
//! All async `#[napi]` methods return native Promises that can be `await`ed
//! from JavaScript/TypeScript.

use std::sync::Arc;

use napi_derive::napi;

use crate::layer1_echo::{Echo, EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeImpl, MockSmtVerifier};
use crate::nodejs::types::{NapiDocument, NapiQuery, NapiSearchResult};
use crate::observability::MemoryObserver;
use crate::pipeline::{Pipeline, RagPipeline};
use crate::types::Document;

/// Concrete pipeline type alias used for the Node.js binding.
///
/// Hard-codes mock/in-memory backends so Node.js callers do not need to
/// handle Rust generics.
pub(crate) type DefaultPipeline = Pipeline<
    EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
    RuleBasedSpeculator,
    JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
>;

// ─────────────────────────────────────────────────────────────────────────────
// NapiPipelineOutput — plain-object form returned from `NapiPipeline.query()`
// ─────────────────────────────────────────────────────────────────────────────

/// A single search-result entry embedded inside [`NapiPipelineOutput`].
///
/// Uses primitive types only so that the parent `#[napi(object)]` struct can
/// be serialised cleanly into a plain JavaScript object.
#[napi(object)]
pub struct NapiSearchResultItem {
    /// Similarity score (0.0–1.0).
    pub score: f64,
    /// Zero-indexed rank within the result set.
    pub rank: u32,
    /// The matched document ID.
    pub document_id: String,
    /// The matched document content.
    pub content: String,
    /// The matched document title, or `null`.
    pub title: Option<String>,
}

/// Full pipeline output returned from `NapiPipeline.query()`.
///
/// This is a plain JavaScript object (not a class), so every property is
/// directly accessible without a getter call.
///
/// @example
/// ```js
/// const out = await pipeline.query(new Query("What is Rust?"));
/// console.log(out.finalAnswer);
/// console.log(out.confidence);
/// for (const r of out.searchResults) {
///     console.log(r.score, r.content);
/// }
/// ```
#[napi(object)]
pub struct NapiPipelineOutput {
    /// The generated answer text.
    pub final_answer: String,
    /// Overall pipeline confidence in the range \[0, 1\].
    pub confidence: f64,
    /// Ranked search results from Layer 1 (Echo).
    pub search_results: Vec<NapiSearchResultItem>,
    /// Names of pipeline layers that processed the query.
    pub layers_used: Vec<String>,
    /// Total pipeline execution time in milliseconds.
    pub total_duration_ms: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// NapiPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// The `OxiRAG` pipeline exposed to Node.js.
///
/// All async methods return native Promises and can be awaited with
/// `async/await`.  Construct via `NapiPipelineBuilder`.
///
/// @example
/// ```js
/// const p = new PipelineBuilder().withDimension(384).build();
/// await p.index(new Document("Rust is memory-safe."));
/// const out = await p.query(new Query("What is Rust?"));
/// console.log(out.finalAnswer);
/// ```
#[napi]
pub struct NapiPipeline {
    pub(crate) inner: Arc<tokio::sync::Mutex<DefaultPipeline>>,
    /// Accumulated span records from the built-in `MemoryObserver`.
    pub(crate) observer: Arc<MemoryObserver>,
}

#[napi]
impl NapiPipeline {
    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Index a single document.
    ///
    /// Returns a `Promise<string>` that resolves to the document ID.
    ///
    /// # Errors
    ///
    /// Rejects with an `Error` if the embedding or storage step fails.
    #[napi]
    pub async fn index(&self, doc: &NapiDocument) -> napi::Result<String> {
        let pipeline = Arc::clone(&self.inner);
        let doc_inner: Document = doc.inner.clone();
        let mut guard = pipeline.lock().await;
        let id = guard
            .echo_mut()
            .index(doc_inner)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(id.to_string())
    }

    /// Index a batch of documents.
    ///
    /// Returns a `Promise<string[]>` that resolves to document IDs in the same
    /// order as the input array.
    ///
    /// # Errors
    ///
    /// Rejects with an `Error` if any embedding or storage step fails.
    #[napi]
    pub async fn index_batch(&self, docs: Vec<&NapiDocument>) -> napi::Result<Vec<String>> {
        let pipeline = Arc::clone(&self.inner);
        let docs_inner: Vec<Document> = docs.into_iter().map(|d| d.inner.clone()).collect();
        let mut guard = pipeline.lock().await;
        let ids = guard
            .echo_mut()
            .index_batch(docs_inner)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(ids.iter().map(ToString::to_string).collect())
    }

    // ── Querying ──────────────────────────────────────────────────────────────

    /// Run a query through the full pipeline.
    ///
    /// Returns a `Promise<NapiPipelineOutput>` with `finalAnswer`, `confidence`,
    /// `searchResults`, `layersUsed`, and `totalDurationMs`.
    ///
    /// # Errors
    ///
    /// Rejects with an `Error` if any pipeline layer fails.
    #[napi]
    pub async fn query(&self, query: &NapiQuery) -> napi::Result<NapiPipelineOutput> {
        let pipeline = Arc::clone(&self.inner);
        let query_inner = query.inner.clone();
        let guard = pipeline.lock().await;
        let output = guard
            .process(query_inner)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;

        let search_results = output
            .search_results
            .into_iter()
            .map(|r| {
                #[allow(clippy::cast_possible_truncation)]
                NapiSearchResultItem {
                    score: f64::from(r.score),
                    rank: r.rank as u32,
                    document_id: r.document.id.to_string(),
                    content: r.document.content.clone(),
                    title: r.document.title.clone(),
                }
            })
            .collect();

        #[allow(clippy::cast_possible_truncation)]
        Ok(NapiPipelineOutput {
            final_answer: output.final_answer,
            confidence: f64::from(output.confidence),
            search_results,
            layers_used: output.layers_used,
            total_duration_ms: output.total_duration_ms as u32,
        })
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Get the number of documents currently indexed.
    ///
    /// Returns a `Promise<number>`.
    ///
    /// # Errors
    ///
    /// Rejects with an `Error` if the pipeline mutex cannot be acquired.
    #[napi]
    pub async fn count(&self) -> napi::Result<u32> {
        let pipeline = Arc::clone(&self.inner);
        let guard = pipeline.lock().await;
        let n = guard.echo().count().await;
        #[allow(clippy::cast_possible_truncation)]
        Ok(n as u32)
    }

    /// Return the number of completed pipeline spans recorded by the built-in
    /// [`MemoryObserver`].
    ///
    /// Useful for lightweight introspection without a full `OTel` setup.
    #[napi]
    #[must_use]
    pub fn span_record_count(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        {
            self.observer.records().len() as u32
        }
    }

    /// Search the Echo layer directly (Layer 1 only, no Speculator or Judge).
    ///
    /// Returns a `Promise<NapiSearchResult[]>`.
    ///
    /// # Errors
    ///
    /// Rejects with an `Error` if the search step fails.
    #[napi]
    pub async fn search(
        &self,
        query_text: String,
        top_k: u32,
        min_score: Option<f64>,
    ) -> napi::Result<Vec<NapiSearchResult>> {
        let pipeline = Arc::clone(&self.inner);
        #[allow(clippy::cast_possible_truncation)]
        let min_score_f32 = min_score.map(|s| s as f32);
        let mut guard = pipeline.lock().await;
        let results = guard
            .echo_mut()
            .search(&query_text, top_k as usize, min_score_f32)
            .await
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|r| NapiSearchResult { inner: r })
            .collect())
    }
}
