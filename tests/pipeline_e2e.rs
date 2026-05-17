//! End-to-end pipeline integration tests using all in-memory backends.
//! These tests exercise the full pipeline rather than individual layers.

use std::sync::Arc;

use oxirag::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use oxirag::layer2_speculator::RuleBasedSpeculator;
use oxirag::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use oxirag::observability::MemoryObserver;
use oxirag::pipeline::{PipelineBuilder, PipelineConfig, RagPipeline};
use oxirag::types::{Document, Query};

// ─── helpers ────────────────────────────────────────────────────────────────���─

fn build_pipeline() -> impl RagPipeline {
    let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));
    let speculator = RuleBasedSpeculator::default();
    let judge = JudgeImpl::new(
        AdvancedClaimExtractor::new(),
        MockSmtVerifier::default(),
        JudgeConfig::default(),
    );

    // Disable fast-path so all layers run deterministically in tests.
    let config = PipelineConfig {
        enable_fast_path: false,
        ..PipelineConfig::default()
    };

    PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(speculator)
        .with_judge(judge)
        .with_config(config)
        .build()
        .expect("pipeline build should succeed")
}

// ─── tests ────────────────────────────────────────────────────────────────────

/// Build a minimal pipeline with only the echo layer providing documents,
/// index 3 documents, query, and assert at least 1 result with score > 0.
#[tokio::test]
async fn test_pipeline_echo_only() {
    let mut pipeline = build_pipeline();

    let docs = vec![
        Document::new("Rust is a systems programming language."),
        Document::new("Python is a scripting language popular for data science."),
        Document::new("Go is a language designed for cloud-native applications."),
    ];

    for doc in docs {
        pipeline
            .index(doc)
            .await
            .expect("document index should succeed");
    }

    let query = Query::new("What is Rust?");
    let output = pipeline
        .process(query)
        .await
        .expect("pipeline process should succeed");

    assert!(
        !output.search_results.is_empty(),
        "expected at least 1 search result"
    );

    let max_score = output
        .search_results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        max_score > 0.0,
        "expected at least one result with score > 0, got max_score={max_score}"
    );
}

/// Use `process_batch` on multiple queries; assert result count equals input
/// query count.
#[tokio::test]
async fn test_pipeline_batch_query() {
    let mut pipeline = build_pipeline();

    let docs = vec![
        Document::new("The Eiffel Tower is located in Paris."),
        Document::new("The Colosseum is an ancient amphitheatre in Rome."),
    ];
    for doc in docs {
        pipeline
            .index(doc)
            .await
            .expect("document index should succeed");
    }

    let queries = vec![
        Query::new("Where is the Eiffel Tower?"),
        Query::new("Tell me about the Colosseum."),
        Query::new("What is the capital of France?"),
    ];
    let expected_len = queries.len();

    let results = pipeline.process_batch(queries).await;

    assert_eq!(
        results.len(),
        expected_len,
        "batch result count should equal query count"
    );
}

/// Query an empty store — must return an empty results list, not a panic.
#[tokio::test]
async fn test_pipeline_empty_store_returns_no_results() {
    let pipeline = build_pipeline();

    let query = Query::new("anything");
    let output = pipeline
        .process(query)
        .await
        .expect("pipeline process on empty store should succeed");

    assert_eq!(
        output.search_results.len(),
        0,
        "empty store should yield 0 search results"
    );
}

/// Add a `MemoryObserver`, run a query, assert the observer captured at least
/// one `LayerSpanRecord`.
#[tokio::test]
async fn test_pipeline_with_observer_records_spans() {
    let observer = Arc::new(MemoryObserver::new());

    let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));
    let speculator = RuleBasedSpeculator::default();
    let judge = JudgeImpl::new(
        AdvancedClaimExtractor::new(),
        MockSmtVerifier::default(),
        JudgeConfig::default(),
    );
    let config = PipelineConfig {
        enable_fast_path: false,
        ..PipelineConfig::default()
    };

    let mut pipeline = PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(speculator)
        .with_judge(judge)
        .with_config(config)
        .with_observers(vec![
            observer.clone() as Arc<dyn oxirag::observability::SpanObserver>
        ])
        .build()
        .expect("pipeline build should succeed");

    pipeline
        .index(Document::new("Observers capture pipeline spans."))
        .await
        .expect("index should succeed");

    let query = Query::new("What do observers capture?");
    pipeline
        .process(query)
        .await
        .expect("process should succeed");

    let records = observer.records();
    assert!(
        !records.is_empty(),
        "observer should have captured at least 1 LayerSpanRecord after query"
    );
}
