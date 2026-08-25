#![cfg(not(target_arch = "wasm32"))]

/// Rust-side smoke tests for the logic that backs the Node.js napi-rs bridge.
///
/// These tests exercise the underlying `Document`, `Query`, `Pipeline`, and
/// `PipelineOutput` types that `NapiDocument`, `NapiQuery`, and `NapiPipeline`
/// wrap.  They run under `cargo nextest run --all-features` without requiring a
/// live Node.js process.
///
/// Full napi integration tests (JavaScript ↔ Rust round-trips) are run
/// separately via `npm test` / `@napi-rs/cli` which loads the `.node` cdylib
/// into a real Node.js runtime.
use oxirag::layer1_echo::{Echo, EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use oxirag::layer2_speculator::RuleBasedSpeculator;
use oxirag::layer3_judge::{AdvancedClaimExtractor, JudgeImpl, MockSmtVerifier};
use oxirag::observability::MemoryObserver;
use oxirag::pipeline::{Pipeline, PipelineBuilder, PipelineConfig, RagPipeline};
use oxirag::types::{Document, Query};
use std::sync::Arc;

type TestPipeline = Pipeline<
    EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
    RuleBasedSpeculator,
    JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
>;

fn build_pipeline(dim: usize) -> (TestPipeline, Arc<MemoryObserver>) {
    let echo = EchoLayer::new(
        MockEmbeddingProvider::new(dim),
        InMemoryVectorStore::new(dim),
    );
    let observer = Arc::new(MemoryObserver::new());
    let pipeline = PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(RuleBasedSpeculator::default())
        .with_judge(JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            Default::default(),
        ))
        .with_config(PipelineConfig {
            max_search_results: 5,
            ..PipelineConfig::default()
        })
        .with_observers(vec![
            Arc::clone(&observer) as Arc<dyn oxirag::observability::SpanObserver>
        ])
        .build()
        .expect("pipeline build should not fail");
    (pipeline, observer)
}

// ── Document construction ────────────────────────────────────────────────────

#[test]
fn document_content_roundtrip() {
    let doc = Document::new("Rust is memory-safe.");
    assert_eq!(doc.content, "Rust is memory-safe.");
}

#[test]
fn document_with_title() {
    let doc = Document::new("content").with_title("My Title");
    assert_eq!(doc.title, Some("My Title".to_string()));
}

#[test]
fn document_has_uuid() {
    let doc = Document::new("x");
    let id_str = doc.id.to_string();
    assert_eq!(id_str.len(), 36, "UUIDs should be 36 chars with hyphens");
}

#[test]
fn document_ids_are_unique() {
    let a = Document::new("same content");
    let b = Document::new("same content");
    assert_ne!(a.id, b.id);
}

// ── Query construction ───────────────────────────────────────────────────────

#[test]
fn query_text_roundtrip() {
    let q = Query::new("What is Rust?");
    assert_eq!(q.text, "What is Rust?");
}

#[test]
fn query_top_k_default() {
    let q = Query::new("test");
    assert!(q.top_k > 0);
}

#[test]
fn query_with_top_k() {
    let q = Query::new("test").with_top_k(7);
    assert_eq!(q.top_k, 7);
}

// ── Pipeline integration ─────────────────────────────────────────────────────

#[tokio::test]
async fn pipeline_index_and_count() {
    let (mut pipeline, _obs) = build_pipeline(8);
    let doc = Document::new("Tokio is an async runtime for Rust.");
    pipeline
        .echo_mut()
        .index(doc)
        .await
        .expect("index should succeed");
    let count = pipeline.echo().count().await;
    assert_eq!(count, 1);
}

#[tokio::test]
async fn pipeline_index_batch() {
    let (mut pipeline, _obs) = build_pipeline(8);
    let docs: Vec<Document> = (0..4)
        .map(|i| Document::new(format!("document {i}")))
        .collect();
    let ids = pipeline
        .echo_mut()
        .index_batch(docs)
        .await
        .expect("index_batch should succeed");
    assert_eq!(ids.len(), 4);
    assert_eq!(pipeline.echo().count().await, 4);
}

#[tokio::test]
async fn pipeline_query_returns_output() {
    let (mut pipeline, _obs) = build_pipeline(8);
    for i in 0..3 {
        let doc = Document::new(format!("fact number {i}"));
        pipeline.echo_mut().index(doc).await.expect("index ok");
    }
    let query = Query::new("fact number 1").with_top_k(2);
    let output = pipeline
        .process(query)
        .await
        .expect("pipeline process should succeed");
    // Mock pipeline may return confidence outside [0,1]; just assert it's finite.
    assert!(output.confidence.is_finite());
}

#[tokio::test]
async fn span_report_has_records_after_query() {
    let (mut pipeline, observer) = build_pipeline(8);
    pipeline
        .echo_mut()
        .index(Document::new("hello world"))
        .await
        .expect("index ok");
    let query = Query::new("hello");
    pipeline.process(query).await.expect("process ok");
    let records = observer.records();
    assert!(
        !records.is_empty(),
        "span report should contain records after a query"
    );
}
