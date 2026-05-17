//! Integration tests verifying that SpanObserver fires during pipeline execution.

use std::sync::Arc;

use oxirag::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use oxirag::layer2_speculator::RuleBasedSpeculator;
use oxirag::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use oxirag::observability::{MemoryObserver, SpanObserver};
use oxirag::pipeline::{PipelineBuilder, PipelineConfig, RagPipeline};
use oxirag::types::{Document, Query};

// ─── helper ──────────────────────────────────────────────────────────────────

/// Build a pipeline with the given observers wired in.
fn build_pipeline_with_observers(observers: Vec<Arc<dyn SpanObserver>>) -> impl RagPipeline {
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

    PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(speculator)
        .with_judge(judge)
        .with_config(config)
        .with_observers(observers)
        .build()
        .expect("pipeline build should succeed")
}

// ─── tests ───��────────────────────────────────────────────────────────────────

/// Run a single query through the pipeline and verify the observer collected at
/// least one span record.
#[tokio::test]
async fn test_observer_fires_on_query() {
    let observer = Arc::new(MemoryObserver::new());

    let mut pipeline =
        build_pipeline_with_observers(vec![observer.clone() as Arc<dyn SpanObserver>]);

    pipeline
        .index(Document::new(
            "Observability matters for production systems.",
        ))
        .await
        .expect("index should succeed");

    pipeline
        .process(Query::new("What matters for production?"))
        .await
        .expect("process should succeed");

    assert!(
        !observer.records().is_empty(),
        "observer should have at least 1 LayerSpanRecord after query"
    );
}

/// Assert that at least one captured record has `layer_name == "echo"`.
#[tokio::test]
async fn test_observer_captures_layer_name() {
    let observer = Arc::new(MemoryObserver::new());

    let mut pipeline =
        build_pipeline_with_observers(vec![observer.clone() as Arc<dyn SpanObserver>]);

    pipeline
        .index(Document::new("Echo layer performs semantic search."))
        .await
        .expect("index should succeed");

    pipeline
        .process(Query::new("What does the echo layer do?"))
        .await
        .expect("process should succeed");

    let records = observer.records();
    let has_echo_layer = records.iter().any(|r| r.layer_name == "echo");
    assert!(
        has_echo_layer,
        "at least one record should have layer_name == \"echo\"; got: {:?}",
        records
            .iter()
            .map(|r| r.layer_name.as_str())
            .collect::<Vec<_>>()
    );
}

/// Assert every captured span record has `duration_ms >= 0` (i.e., non-negative).
#[tokio::test]
async fn test_observer_duration_positive() {
    let observer = Arc::new(MemoryObserver::new());

    let mut pipeline =
        build_pipeline_with_observers(vec![observer.clone() as Arc<dyn SpanObserver>]);

    pipeline
        .index(Document::new(
            "Duration measurements should be non-negative.",
        ))
        .await
        .expect("index should succeed");

    pipeline
        .process(Query::new("Are durations non-negative?"))
        .await
        .expect("process should succeed");

    let records = observer.records();
    assert!(
        !records.is_empty(),
        "observer should have captured at least one record"
    );

    for record in &records {
        // `duration_ms` is u64, so this is always true by type; assert here to
        // document and validate the invariant explicitly.
        assert!(
            record.duration_ms < u64::MAX,
            "duration_ms should not be u64::MAX (would indicate overflow), got {}",
            record.duration_ms
        );
    }
}

/// Run a query and assert that `pipeline_snapshots().len() >= 1`, verifying
/// that the pipeline calls `on_pipeline_complete` on all observers.
#[tokio::test]
async fn test_observer_pipeline_complete_fires() {
    let observer = Arc::new(MemoryObserver::new());

    let mut pipeline =
        build_pipeline_with_observers(vec![observer.clone() as Arc<dyn SpanObserver>]);

    pipeline
        .index(Document::new(
            "Pipeline complete event fires after every query.",
        ))
        .await
        .expect("index should succeed");

    pipeline
        .process(Query::new("Does pipeline_complete fire?"))
        .await
        .expect("process should succeed");

    let snapshots = observer.pipeline_snapshots();
    assert!(
        !snapshots.is_empty(),
        "observer.pipeline_snapshots() should have at least 1 entry; the pipeline \
         should call on_pipeline_complete after every process() invocation"
    );
}

/// Add two `MemoryObserver`s to the pipeline; run one query; assert both
/// observers captured span records.
#[tokio::test]
async fn test_multiple_observers_both_fired() {
    let observer_a = Arc::new(MemoryObserver::new());
    let observer_b = Arc::new(MemoryObserver::new());

    let mut pipeline = build_pipeline_with_observers(vec![
        observer_a.clone() as Arc<dyn SpanObserver>,
        observer_b.clone() as Arc<dyn SpanObserver>,
    ]);

    pipeline
        .index(Document::new(
            "Multiple observers should all receive events.",
        ))
        .await
        .expect("index should succeed");

    pipeline
        .process(Query::new("Do multiple observers each receive spans?"))
        .await
        .expect("process should succeed");

    assert!(
        !observer_a.records().is_empty(),
        "first observer should have captured span records"
    );
    assert!(
        !observer_b.records().is_empty(),
        "second observer should have captured span records"
    );
}
