//! OpenTelemetry integration for the `OxiRAG` pipeline observability system.
//!
//! This module is gated behind the `otel` feature flag and provides a
//! [`SpanObserver`] implementation that forwards span data to an
//! OpenTelemetry-compatible backend.
//!
//! # Usage with stdout (for examples/tests):
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use oxirag::observability::otel::OtelSpanObserver;
//!
//! let observer = OtelSpanObserver::with_stdout().expect("init otel");
//! // register with a PipelineSpanContext ...
//! ```
//!
//! # Usage with OTLP/gRPC (for production):
//!
//! ```rust,ignore
//! use oxirag::observability::otel::OtelSpanObserver;
//!
//! let observer = OtelSpanObserver::with_otlp_endpoint("http://otel-collector:4317")
//!     .expect("init otel otlp");
//! ```

#![cfg(feature = "otel")]

use std::fmt;

use opentelemetry::{
    KeyValue,
    global::{self},
    trace::{Span, Tracer},
};
use opentelemetry_sdk::trace::SdkTracerProvider;

use super::{LayerSpanRecord, PipelineSpanContext, SpanObserver, SpanStatus};

// ────────────────────────────────────────────────────────────────────────────
// OtelInitError
// ────────────────────────────────────────────────────────────────────────────

/// Error returned when the OpenTelemetry tracer provider cannot be initialised.
#[derive(Debug)]
pub struct OtelInitError(String);

impl OtelInitError {
    /// Create a new [`OtelInitError`] with the given description.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for OtelInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OtelInitError: {}", self.0)
    }
}

impl std::error::Error for OtelInitError {}

// ────────────────────────────────────────────────────────────────────────────
// OtelSpanObserver
// ────────────────────────────────────────────────────────────────────────────

/// A [`SpanObserver`] that exports span data via OpenTelemetry.
///
/// Holds a reference to the [`SdkTracerProvider`] so it can be cleanly
/// shut down when the observer is dropped, flushing all pending spans.
///
/// Supports two backends:
/// - **stdout** (for development/debugging): use [`with_stdout`].
/// - **OTLP/gRPC** (for production): use [`with_otlp_endpoint`].
///
/// [`with_stdout`]: OtelSpanObserver::with_stdout
/// [`with_otlp_endpoint`]: OtelSpanObserver::with_otlp_endpoint
pub struct OtelSpanObserver {
    /// The SDK tracer provider, retained for Drop-based shutdown.
    provider: SdkTracerProvider,
}

impl fmt::Debug for OtelSpanObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtelSpanObserver").finish_non_exhaustive()
    }
}

impl OtelSpanObserver {
    /// Initialise the OpenTelemetry tracing pipeline with a **stdout exporter**.
    ///
    /// The stdout exporter is suitable for development, debugging, and tests.
    /// It writes human-readable span data to standard output.
    ///
    /// # Errors
    ///
    /// Returns [`OtelInitError`] if the provider cannot be built.
    pub fn with_stdout() -> Result<Self, OtelInitError> {
        let exporter = opentelemetry_stdout::SpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build();

        // Register as the global provider so `global::tracer(...)` resolves to it.
        // The previous provider is returned but we intentionally discard it.
        global::set_tracer_provider(provider.clone());

        Ok(Self { provider })
    }

    /// Initialise the OpenTelemetry tracing pipeline with an **OTLP/gRPC exporter**.
    ///
    /// `endpoint` should be a valid gRPC endpoint URL such as
    /// `"http://otel-collector:4317"`.
    ///
    /// # Errors
    ///
    /// Returns [`OtelInitError`] if the OTLP exporter cannot be built or
    /// the provider cannot be initialised.
    #[cfg(feature = "otel")]
    pub fn with_otlp_endpoint(endpoint: &str) -> Result<Self, OtelInitError> {
        use opentelemetry_otlp::{SpanExporter, WithExportConfig};

        let exporter = SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| OtelInitError::new(format!("OTLP exporter build error: {e}")))?;

        let provider = SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();

        global::set_tracer_provider(provider.clone());

        Ok(Self { provider })
    }

    /// Emit a single OpenTelemetry span for the given [`LayerSpanRecord`].
    ///
    /// The span is named `"oxirag.layer.<layer_name>"` and carries standard
    /// attributes plus any user-defined attributes from the record.
    #[allow(clippy::unused_self)]
    fn emit_layer_span(&self, record: &LayerSpanRecord) {
        let tracer = global::tracer("oxirag");
        let span_name = format!("oxirag.layer.{}", record.layer_name);

        let mut span = tracer.start(span_name);

        // Standard attributes — durations and counts use i64 (OTel Int value).
        span.set_attribute(KeyValue::new(
            "layer.duration_ms",
            i64::try_from(record.duration_ms).unwrap_or(i64::MAX),
        ));
        // status.label() borrows &self so we must convert to owned String.
        span.set_attribute(KeyValue::new(
            "layer.status",
            opentelemetry::Value::String(record.status.label().to_owned().into()),
        ));

        if let Some(count) = record.item_count {
            span.set_attribute(KeyValue::new(
                "layer.item_count",
                i64::try_from(count).unwrap_or(i64::MAX),
            ));
        }

        if let SpanStatus::Error(ref msg) = record.status {
            // String::from ensures owned data — no lifetime escape.
            span.set_attribute(KeyValue::new(
                "layer.error",
                opentelemetry::Value::String(msg.clone().into()),
            ));
            span.set_status(opentelemetry::trace::Status::error(msg.clone()));
        }

        // User-defined attributes — keys and values are both owned Strings.
        for (k, v) in &record.attributes {
            span.set_attribute(KeyValue::new(
                k.clone(),
                opentelemetry::Value::String(v.clone().into()),
            ));
        }

        span.end();
    }

    /// Emit a single OpenTelemetry span summarising the entire pipeline execution.
    #[allow(clippy::unused_self)]
    fn emit_pipeline_span(&self, ctx: &PipelineSpanContext) {
        let tracer = global::tracer("oxirag");

        let mut span = tracer.start("oxirag.pipeline");

        span.set_attribute(KeyValue::new(
            "pipeline.execution_id",
            ctx.execution_id.clone(),
        ));
        span.set_attribute(KeyValue::new(
            "pipeline.layer_count",
            i64::try_from(ctx.layer_spans.len()).unwrap_or(i64::MAX),
        ));

        let success_count = ctx
            .layer_spans
            .iter()
            .filter(|s| s.status.is_success())
            .count();
        let error_count = ctx
            .layer_spans
            .iter()
            .filter(|s| s.status.is_error())
            .count();

        span.set_attribute(KeyValue::new(
            "pipeline.success_count",
            i64::try_from(success_count).unwrap_or(i64::MAX),
        ));
        span.set_attribute(KeyValue::new(
            "pipeline.error_count",
            i64::try_from(error_count).unwrap_or(i64::MAX),
        ));
        span.set_attribute(KeyValue::new(
            "pipeline.duration_ms",
            i64::try_from(ctx.elapsed_ms()).unwrap_or(i64::MAX),
        ));

        span.end();
    }
}

impl Drop for OtelSpanObserver {
    /// Flush all pending spans and shut down the tracer provider cleanly.
    ///
    /// This ensures no spans are lost when the observer goes out of scope.
    fn drop(&mut self) {
        // Best-effort flush; errors are ignored to keep Drop infallible.
        let _ = self.provider.force_flush();
        let _ = self.provider.shutdown();
    }
}

impl SpanObserver for OtelSpanObserver {
    fn on_layer_complete(&self, record: &LayerSpanRecord) {
        self.emit_layer_span(record);
    }

    fn on_pipeline_complete(&self, ctx: &PipelineSpanContext) {
        self.emit_pipeline_span(ctx);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_otel_init_error_display() {
        let err = OtelInitError::new("test error message");
        let displayed = format!("{err}");
        assert!(
            displayed.contains("OtelInitError"),
            "Display must include 'OtelInitError', got: {displayed}"
        );
        assert!(
            displayed.contains("test error message"),
            "Display must include the inner message, got: {displayed}"
        );
    }

    #[test]
    fn test_otel_init_error_debug() {
        let err = OtelInitError::new("debug test");
        let debug_str = format!("{err:?}");
        assert!(
            debug_str.contains("debug test"),
            "Debug output must include message, got: {debug_str}"
        );
    }

    #[test]
    fn test_otel_init_error_is_error_trait() {
        let err = OtelInitError::new("trait test");
        // Verify it satisfies std::error::Error
        let _boxed: Box<dyn std::error::Error> = Box::new(err);
    }

    #[test]
    fn test_otel_observer_with_stdout_constructs() {
        // with_stdout() initialises the global OTel provider.
        // In test environments this should succeed; if it fails we still
        // verify the error path is handled gracefully.
        let result = OtelSpanObserver::with_stdout();
        // Either success or a well-formed error — neither should panic.
        match result {
            Ok(observer) => {
                // Verify the observer implements SpanObserver by calling through it.
                let mut ctx = super::super::PipelineSpanContext::new();
                ctx.begin_layer("test-layer").success();
                observer.on_pipeline_complete(&ctx);
            }
            Err(e) => {
                // Error path must be displayable without panicking.
                let _ = format!("{e}");
            }
        }
    }

    #[test]
    fn test_otel_observer_layer_span_emission() {
        // Construct an observer and drive it through the SpanObserver interface.
        // If OTel init fails in the test env, skip gracefully.
        let Ok(observer) = OtelSpanObserver::with_stdout() else {
            return; // OTel not available in this test env — skip
        };

        let mut attrs = HashMap::new();
        attrs.insert("top_k".to_string(), "5".to_string());

        let record = LayerSpanRecord {
            layer_name: "echo".to_string(),
            duration_ms: 42,
            status: SpanStatus::Success,
            attributes: attrs,
            item_count: Some(5),
        };

        // Must not panic
        observer.on_layer_complete(&record);
    }

    #[test]
    fn test_otel_observer_error_span_emission() {
        let Ok(observer) = OtelSpanObserver::with_stdout() else {
            return;
        };

        let record = LayerSpanRecord {
            layer_name: "judge".to_string(),
            duration_ms: 7,
            status: SpanStatus::Error("smt timeout".to_string()),
            attributes: std::collections::HashMap::new(),
            item_count: None,
        };

        observer.on_layer_complete(&record);
    }

    #[test]
    fn test_otel_observer_pipeline_complete_emission() {
        let Ok(observer) = OtelSpanObserver::with_stdout() else {
            return;
        };

        let mut ctx = super::super::PipelineSpanContext::new();
        ctx.begin_layer("echo").success();
        ctx.begin_layer("speculator").success();
        ctx.begin_layer("judge").error("smt fail");

        // Must not panic
        observer.on_pipeline_complete(&ctx);
    }
}
