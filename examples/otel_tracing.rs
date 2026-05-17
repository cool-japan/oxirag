//! Demonstrates OpenTelemetry tracing integration with OxiRAG.
//!
//! Run with:
//! ```shell
//! cargo run --example otel_tracing --features otel
//! ```
//!
//! This example:
//! 1. Installs the stdout OTel exporter (no external infrastructure required).
//! 2. Creates an [`OtelSpanObserver`] and a [`MemoryObserver`] to verify callbacks.
//! 3. Builds a simple pipeline with in-memory Echo + `RuleBasedSpeculator`.
//! 4. Indexes a document and runs one query.
//! 5. Verifies that the observers received span callbacks.
//! 6. Prints a confirmation message.

use std::sync::Arc;

use oxirag::{
    OtelSpanObserver,
    observability::{MemoryObserver, PipelineSpanContext},
    pipeline::PipelineBuilder,
    prelude::*,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Install the stdout OTel exporter ──────────────────────────────────
    let otel_observer =
        OtelSpanObserver::with_stdout().expect("Failed to initialise stdout OTel exporter");

    // ── 2. Also attach a MemoryObserver to verify callbacks programmatically ─
    let memory_observer = Arc::new(MemoryObserver::new());

    // ── 3. Build a simple pipeline ───────────────────────────────────────────
    let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));
    let speculator = RuleBasedSpeculator::default();
    let judge = JudgeImpl::new(
        AdvancedClaimExtractor::new(),
        MockSmtVerifier::default(),
        JudgeConfig::default(),
    );

    let mut pipeline = PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(speculator)
        .with_judge(judge)
        .with_config(PipelineConfig {
            enable_fast_path: false,
            ..Default::default()
        })
        .build()
        .expect("Failed to build pipeline");

    // ── 4. Index a document and run a query ──────────────────────────────────
    pipeline
        .index(Document::new(
            "OpenTelemetry is an observability framework for cloud-native software.",
        ))
        .await
        .expect("Failed to index document");

    // Manually create a PipelineSpanContext to demonstrate the observer wiring.
    // In production the pipeline does this internally per-query.
    let mut ctx = PipelineSpanContext::new();
    ctx.add_observer(Arc::new(otel_observer));
    ctx.add_observer(memory_observer.clone());

    {
        let mut span = ctx.begin_layer("echo");
        span.set_attribute("query", "What is OpenTelemetry?");
        span.set_item_count(1);
        span.success();
    }
    {
        let mut span = ctx.begin_layer("speculator");
        span.set_attribute("model", "rule-based");
        span.success();
    }
    {
        let span = ctx.begin_layer("judge");
        span.skip();
    }

    ctx.finalize();

    // Also run a real query through the pipeline
    let query = Query::new("What is OpenTelemetry?").with_top_k(1);
    let result = pipeline
        .process(query)
        .await
        .expect("Failed to process query");
    println!("Pipeline answer: {}", result.final_answer);
    println!("Confidence: {:.2}", result.confidence);

    // ── 5. Verify observers were called ─────────────────────────────────────
    let records = memory_observer.records();
    assert!(
        !records.is_empty(),
        "MemoryObserver should have received at least one layer record"
    );
    println!("MemoryObserver collected {} layer records:", records.len());
    for record in &records {
        println!(
            "  layer={} status={} duration_ms={}",
            record.layer_name,
            record.status.label(),
            record.duration_ms
        );
    }

    let snapshots = memory_observer.pipeline_snapshots();
    assert_eq!(
        snapshots.len(),
        1,
        "MemoryObserver should have exactly 1 pipeline snapshot"
    );
    println!(
        "\nPipeline snapshot: {} layers, {} success, {} errors",
        snapshots[0].layer_spans.len(),
        snapshots[0].success_count,
        snapshots[0].error_count
    );

    // ── 6. Confirmation ───────────────────────────────────────────────────────
    println!("\nOTel spans exported to stdout");
    println!(
        "Run 'cargo run --example otel_tracing --features otel' to see OTel JSON output above."
    );

    Ok(())
}
