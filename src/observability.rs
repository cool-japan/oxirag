//! Structured span tracking for the `OxiRAG` pipeline.
//!
//! This module provides lightweight, zero-external-dependency observability
//! primitives built on top of the [`tracing`] crate (already a dependency).
//! It does **not** require `opentelemetry` or any external tracing backend.
//!
//! # Core Concepts
//!
//! - [`PipelineSpanContext`] — accumulates timing and attribute data for one
//!   complete pipeline execution identified by a UUID.
//! - [`LayerSpan`] — an RAII guard returned by
//!   [`PipelineSpanContext::begin_layer`] that records a [`LayerSpanRecord`]
//!   when it is committed or dropped.
//! - [`SpanReport`] — a snapshot of all completed layer spans from a
//!   [`PipelineSpanContext`], suitable for serialisation or display.
//! - [`SpanStatus`] — the outcome of a single layer execution.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::observability::PipelineSpanContext;
//!
//! let mut ctx = PipelineSpanContext::new();
//! ctx.set_attribute("query", "What is Rust?");
//!
//! {
//!     let mut span = ctx.begin_layer("echo");
//!     span.set_attribute("top_k", "5");
//!     span.set_item_count(5);
//!     span.success();
//! }
//!
//! let report = ctx.report();
//! println!("{}", report.format_table());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

// ────────────────────────────────────────────────────────────────────────────
// SpanStatus
// ────────────────────────────────────────────────────────────────────────────

/// The outcome of a single layer execution within the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// The operation completed successfully.
    Success,
    /// The operation failed with an error message.
    Error(String),
    /// The operation was skipped (e.g., fast-path bypass or feature disabled).
    Skipped,
    /// The operation is still in progress (internal; set during construction).
    InProgress,
}

impl SpanStatus {
    /// Returns `true` if the status is [`SpanStatus::Success`].
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if the status is [`SpanStatus::Error`].
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Returns `true` if the status is [`SpanStatus::Skipped`].
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        matches!(self, Self::Skipped)
    }

    /// Returns `true` if the status is [`SpanStatus::InProgress`].
    #[must_use]
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::InProgress)
    }

    /// Return a short string label suitable for tabular display.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Success => "SUCCESS",
            Self::Error(_) => "ERROR",
            Self::Skipped => "SKIPPED",
            Self::InProgress => "IN_PROGRESS",
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LayerSpanRecord
// ────────────────────────────────────────────────────────────────────────────

/// A completed record of a single layer's execution within the pipeline.
///
/// Instances are created internally by [`LayerSpan`] when it is committed
/// (via [`LayerSpan::success`], [`LayerSpan::error`], or [`LayerSpan::skip`])
/// or when it is dropped without explicit commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpanRecord {
    /// Human-readable name of the pipeline layer or component (e.g. `"echo"`, `"judge"`).
    pub layer_name: String,
    /// Wall-clock duration of the operation in milliseconds.
    pub duration_ms: u64,
    /// Final status of the operation.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes recorded during the span.
    ///
    /// Use these to attach diagnostic data such as `"top_k"`, `"model_id"`,
    /// or `"cache_hit"`.
    pub attributes: HashMap<String, String>,
    /// Number of items processed (documents indexed, results returned, etc.).
    ///
    /// `None` if not explicitly set via [`LayerSpan::set_item_count`].
    pub item_count: Option<usize>,
}

impl LayerSpanRecord {
    /// Construct a fresh in-progress record.
    fn new_in_progress(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            duration_ms: 0,
            status: SpanStatus::InProgress,
            attributes: HashMap::new(),
            item_count: None,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PipelineSpanContext
// ────────────────────────────────────────────────────────────────────────────

/// Accumulated observability context for one complete pipeline execution.
///
/// Tracks:
/// - A UUID v4 `execution_id` for correlation with external systems.
/// - The wall-clock start time of the execution.
/// - An ordered list of completed [`LayerSpanRecord`]s.
/// - Arbitrary top-level attributes (e.g. `"query"`, `"user_id"`).
///
/// Typical usage:
/// 1. Call [`PipelineSpanContext::new`] at the start of a pipeline invocation.
/// 2. Use [`begin_layer`] to obtain a [`LayerSpan`] guard for each pipeline step.
/// 3. Commit the guard with [`LayerSpan::success`], [`LayerSpan::error`], or
///    [`LayerSpan::skip`].
/// 4. After all steps, call [`report`] to obtain a [`SpanReport`] and
///    [`emit_traces`] to flush all events to the `tracing` subscriber.
///
/// [`begin_layer`]: PipelineSpanContext::begin_layer
/// [`report`]: PipelineSpanContext::report
/// [`emit_traces`]: PipelineSpanContext::emit_traces
#[derive(Debug)]
pub struct PipelineSpanContext {
    /// UUID v4 identifying this pipeline execution instance.
    pub execution_id: String,
    /// Monotonic start time (not serialised — use [`elapsed`] for durations).
    ///
    /// [`elapsed`]: PipelineSpanContext::elapsed
    start_time: Instant,
    /// Completed layer spans in chronological order of completion.
    pub layer_spans: Vec<LayerSpanRecord>,
    /// Top-level key-value attributes for the entire execution.
    pub attributes: HashMap<String, String>,
}

impl PipelineSpanContext {
    /// Begin a new pipeline execution context.
    ///
    /// Assigns a fresh UUID v4 `execution_id` and records `Instant::now()`
    /// as the start of the execution wall clock.
    #[must_use]
    pub fn new() -> Self {
        let execution_id = Uuid::new_v4().to_string();
        debug!(execution_id = %execution_id, "pipeline execution started");
        Self {
            execution_id,
            start_time: Instant::now(),
            layer_spans: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Record an attribute on the pipeline context.
    ///
    /// Attributes are included in the [`SpanReport`] and emitted at trace
    /// time via [`emit_traces`].
    ///
    /// [`emit_traces`]: PipelineSpanContext::emit_traces
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Begin tracking a layer execution.
    ///
    /// Returns a [`LayerSpan`] RAII guard. The guard **must** be explicitly
    /// committed (via [`LayerSpan::success`], [`LayerSpan::error`], or
    /// [`LayerSpan::skip`]) to record the correct status. If the guard is
    /// dropped without commitment (e.g., due to a panic), it records a
    /// `Error("dropped without completion")` status automatically.
    pub fn begin_layer(&mut self, layer_name: impl Into<String>) -> LayerSpan<'_> {
        let name = layer_name.into();
        debug!(layer = %name, "layer span started");
        LayerSpan {
            context: self,
            record: LayerSpanRecord::new_in_progress(name),
            start: Instant::now(),
            committed: false,
        }
    }

    /// Elapsed wall-clock time since this context was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Total elapsed milliseconds since this context was created.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Build a [`SpanReport`] summarising all completed layer spans.
    ///
    /// The report snapshot is taken at the moment of the call; any layer spans
    /// started after this point will not appear.
    #[must_use]
    pub fn report(&self) -> SpanReport {
        let total_ms = self.elapsed_ms();

        let success_count = self
            .layer_spans
            .iter()
            .filter(|s| s.status.is_success())
            .count();
        let error_count = self
            .layer_spans
            .iter()
            .filter(|s| s.status.is_error())
            .count();
        let skipped_count = self
            .layer_spans
            .iter()
            .filter(|s| s.status.is_skipped())
            .count();

        let slowest_layer = self
            .layer_spans
            .iter()
            .max_by_key(|s| s.duration_ms)
            .map(|s| s.layer_name.clone());

        SpanReport {
            execution_id: self.execution_id.clone(),
            total_ms,
            layer_spans: self.layer_spans.clone(),
            success_count,
            error_count,
            skipped_count,
            slowest_layer,
        }
    }

    /// Emit all span data as structured `tracing` events at DEBUG level.
    ///
    /// Each layer span is emitted as a separate event. The pipeline-level
    /// attributes are emitted as a single INFO event.
    pub fn emit_traces(&self) {
        info!(
            execution_id = %self.execution_id,
            total_ms = self.elapsed_ms(),
            layer_count = self.layer_spans.len(),
            "pipeline execution trace"
        );

        for span in &self.layer_spans {
            debug!(
                execution_id = %self.execution_id,
                layer = %span.layer_name,
                duration_ms = span.duration_ms,
                status = %span.status.label(),
                item_count = ?span.item_count,
                "layer span completed"
            );
        }

        for (k, v) in &self.attributes {
            debug!(
                execution_id = %self.execution_id,
                key = %k,
                value = %v,
                "pipeline attribute"
            );
        }
    }
}

impl Default for PipelineSpanContext {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LayerSpan  (RAII guard)
// ────────────────────────────────────────────────────────────────────────────

/// RAII guard returned by [`PipelineSpanContext::begin_layer`].
///
/// Records a [`LayerSpanRecord`] into the parent [`PipelineSpanContext`] when
/// committed or dropped. Obtain one via [`PipelineSpanContext::begin_layer`].
///
/// # Panicking behaviour
///
/// The `Drop` impl is panic-safe: it never panics even if the guard is dropped
/// during stack unwinding. If the guard is dropped without being committed, the
/// status is recorded as `Error("dropped without completion")`.
pub struct LayerSpan<'ctx> {
    context: &'ctx mut PipelineSpanContext,
    record: LayerSpanRecord,
    start: Instant,
    /// Tracks whether `success`, `error`, or `skip` has already been called so
    /// the `Drop` impl can decide whether to synthesise an error record.
    committed: bool,
}

impl LayerSpan<'_> {
    /// Record an arbitrary key-value attribute on this layer span.
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.record.attributes.insert(key.into(), value.into());
        self
    }

    /// Record the number of items processed during this layer (e.g., documents
    /// indexed or search results returned).
    ///
    /// Returns `&mut Self` for method chaining.
    pub fn set_item_count(&mut self, count: usize) -> &mut Self {
        self.record.item_count = Some(count);
        self
    }

    /// Commit this span as **successful** and record it in the parent context.
    ///
    /// Consumes `self`; the [`Drop`] impl will not add a duplicate record.
    pub fn success(mut self) {
        self.record.status = SpanStatus::Success;
        self.record.duration_ms =
            u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.committed = true;
        debug!(
            layer = %self.record.layer_name,
            duration_ms = self.record.duration_ms,
            "layer span succeeded"
        );
        self.context.layer_spans.push(self.record.clone());
    }

    /// Commit this span as **errored** and record it in the parent context.
    ///
    /// Consumes `self`.
    pub fn error(mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.record.status = SpanStatus::Error(msg);
        self.record.duration_ms =
            u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.committed = true;
        debug!(
            layer = %self.record.layer_name,
            duration_ms = self.record.duration_ms,
            status = "ERROR",
            "layer span errored"
        );
        self.context.layer_spans.push(self.record.clone());
    }

    /// Commit this span as **skipped** and record it in the parent context.
    ///
    /// Consumes `self`.
    pub fn skip(mut self) {
        self.record.status = SpanStatus::Skipped;
        self.record.duration_ms =
            u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.committed = true;
        debug!(
            layer = %self.record.layer_name,
            duration_ms = self.record.duration_ms,
            "layer span skipped"
        );
        self.context.layer_spans.push(self.record.clone());
    }
}

impl Drop for LayerSpan<'_> {
    /// If the guard is dropped without an explicit call to [`success`], [`error`],
    /// or [`skip`] — for example, because a panic is unwinding the stack — the
    /// span is committed with `Error("dropped without completion")` so the
    /// [`PipelineSpanContext`] always has a complete record.
    ///
    /// This `Drop` impl is intentionally panic-safe: it performs no operations
    /// that could themselves panic.
    ///
    /// [`success`]: LayerSpan::success
    /// [`error`]: LayerSpan::error
    /// [`skip`]: LayerSpan::skip
    fn drop(&mut self) {
        if !self.committed {
            self.record.status = SpanStatus::Error("dropped without completion".to_string());
            self.record.duration_ms =
                u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);
            // Push without panicking: Vec::push is infallible.
            self.context.layer_spans.push(self.record.clone());
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SpanReport
// ────────────────────────────────────────────────────────────────────────────

/// A serialisable snapshot of all layer spans from a pipeline execution.
///
/// Obtain one via [`PipelineSpanContext::report`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanReport {
    /// UUID v4 matching the originating [`PipelineSpanContext::execution_id`].
    pub execution_id: String,
    /// Total wall-clock milliseconds from execution start to the moment
    /// [`PipelineSpanContext::report`] was called.
    pub total_ms: u64,
    /// Ordered list of completed layer span records.
    pub layer_spans: Vec<LayerSpanRecord>,
    /// Count of layers that completed with [`SpanStatus::Success`].
    pub success_count: usize,
    /// Count of layers that completed with [`SpanStatus::Error`].
    pub error_count: usize,
    /// Count of layers that completed with [`SpanStatus::Skipped`].
    pub skipped_count: usize,
    /// Name of the layer with the highest `duration_ms`, or `None` if there
    /// are no layer spans.
    pub slowest_layer: Option<String>,
}

impl SpanReport {
    /// Format the report as a human-readable ASCII table.
    ///
    /// Columns: `Layer`, `Status`, `Duration (ms)`, `Items`.
    ///
    /// Example output:
    /// ```text
    /// ┌────────────────────────┬─────────┬──────────────┬───────┐
    /// │ Layer                  │ Status  │ Duration(ms) │ Items │
    /// ├────────────────────────┼─────────┼──────────────┼───────┤
    /// │ echo                   │ SUCCESS │           12 │     5 │
    /// │ speculator             │ SUCCESS │           34 │       │
    /// │ judge                  │ SKIPPED │            0 │       │
    /// └────────────────────────┴─────────┴──────────────┴───────┘
    /// Total: 3 layers | 2 success | 0 errors | 1 skipped | 46ms
    /// ```
    #[must_use]
    pub fn format_table(&self) -> String {
        // Column widths (minimum enforced).
        const W_LAYER: usize = 24;
        const W_STATUS: usize = 9;
        const W_DURATION: usize = 12;
        const W_ITEMS: usize = 7;

        let top = format!(
            "┌{:─<W_LAYER$}┬{:─<W_STATUS$}┬{:─<W_DURATION$}┬{:─<W_ITEMS$}┐",
            "", "", "", ""
        );
        let header = format!(
            "│ {:<width$}│ {:<ws$}│ {:<wd$}│ {:<wi$}│",
            "Layer",
            "Status",
            "Duration(ms)",
            "Items",
            width = W_LAYER - 1,
            ws = W_STATUS - 1,
            wd = W_DURATION - 1,
            wi = W_ITEMS - 1,
        );
        let sep = format!(
            "├{:─<W_LAYER$}┼{:─<W_STATUS$}┼{:─<W_DURATION$}┼{:─<W_ITEMS$}┤",
            "", "", "", ""
        );
        let bottom = format!(
            "└{:─<W_LAYER$}┴{:─<W_STATUS$}┴{:─<W_DURATION$}┴{:─<W_ITEMS$}┘",
            "", "", "", ""
        );

        let mut rows = Vec::with_capacity(self.layer_spans.len() + 6);
        rows.push(top);
        rows.push(header);
        rows.push(sep);

        for span in &self.layer_spans {
            let items_str = span.item_count.map_or_else(String::new, |c| c.to_string());

            // Truncate layer name to fit the column.
            let layer_display = if span.layer_name.len() > W_LAYER - 2 {
                format!("{}…", &span.layer_name[..W_LAYER - 3])
            } else {
                span.layer_name.clone()
            };

            let row = format!(
                "│ {:<width$}│ {:<ws$}│ {:>wd$} │ {:>wi$}│",
                layer_display,
                span.status.label(),
                span.duration_ms,
                items_str,
                width = W_LAYER - 1,
                ws = W_STATUS - 1,
                wd = W_DURATION - 2,
                wi = W_ITEMS - 1,
            );
            rows.push(row);
        }

        rows.push(bottom);
        rows.push(format!(
            "Total: {} layers | {} success | {} errors | {} skipped | {}ms",
            self.layer_spans.len(),
            self.success_count,
            self.error_count,
            self.skipped_count,
            self.total_ms,
        ));

        rows.join("\n")
    }

    /// Serialise the report to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if serialisation fails (in practice this
    /// should never occur for this type).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Free function
// ────────────────────────────────────────────────────────────────────────────

/// Record a named pipeline event using [`tracing`] at INFO level.
///
/// Emits a single structured log event with `event_name` as the primary field
/// and all `attributes` as additional key-value pairs.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use oxirag::observability::record_pipeline_event;
///
/// let mut attrs = HashMap::new();
/// attrs.insert("query_id".into(), "abc123".into());
/// record_pipeline_event("retrieval_complete", &attrs);
/// ```
pub fn record_pipeline_event<S: std::hash::BuildHasher>(
    event_name: &str,
    attributes: &HashMap<String, String, S>,
) {
    let attrs_json = serde_json::to_string(attributes).unwrap_or_else(|_| "{}".to_string());
    info!(
        event = %event_name,
        attributes = %attrs_json,
        "pipeline event"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // ── PipelineSpanContext ──────────────────────────────────────────────

    #[test]
    fn test_pipeline_context_new() {
        let ctx = PipelineSpanContext::new();
        // UUID v4 has 36 characters (8-4-4-4-12 with dashes).
        assert_eq!(
            ctx.execution_id.len(),
            36,
            "execution_id should be a UUID v4 string"
        );
        assert!(
            ctx.layer_spans.is_empty(),
            "no spans should exist at construction"
        );
        assert!(
            ctx.attributes.is_empty(),
            "no attributes should exist at construction"
        );
    }

    #[test]
    fn test_layer_span_success() {
        let mut ctx = PipelineSpanContext::new();
        {
            let span = ctx.begin_layer("echo");
            span.success();
        }
        assert_eq!(ctx.layer_spans.len(), 1);
        let record = &ctx.layer_spans[0];
        assert_eq!(record.layer_name, "echo");
        assert_eq!(record.status, SpanStatus::Success);
    }

    #[test]
    fn test_layer_span_error() {
        let mut ctx = PipelineSpanContext::new();
        {
            let span = ctx.begin_layer("judge");
            span.error("smt timeout");
        }
        assert_eq!(ctx.layer_spans.len(), 1);
        let record = &ctx.layer_spans[0];
        assert_eq!(record.layer_name, "judge");
        assert!(record.status.is_error());
        if let SpanStatus::Error(ref msg) = record.status {
            assert_eq!(msg, "smt timeout");
        }
    }

    #[test]
    fn test_layer_span_skip() {
        let mut ctx = PipelineSpanContext::new();
        {
            let span = ctx.begin_layer("speculator");
            span.skip();
        }
        assert_eq!(ctx.layer_spans.len(), 1);
        let record = &ctx.layer_spans[0];
        assert_eq!(record.status, SpanStatus::Skipped);
        assert!(record.status.is_skipped());
    }

    #[test]
    fn test_layer_span_attributes() {
        let mut ctx = PipelineSpanContext::new();
        {
            let mut span = ctx.begin_layer("echo");
            span.set_attribute("model", "bge-base");
            span.set_attribute("top_k", "10");
            span.success();
        }
        let record = &ctx.layer_spans[0];
        assert_eq!(
            record.attributes.get("model"),
            Some(&"bge-base".to_string())
        );
        assert_eq!(record.attributes.get("top_k"), Some(&"10".to_string()));
    }

    #[test]
    fn test_layer_span_item_count() {
        let mut ctx = PipelineSpanContext::new();
        {
            let mut span = ctx.begin_layer("echo");
            span.set_item_count(7);
            span.success();
        }
        let record = &ctx.layer_spans[0];
        assert_eq!(record.item_count, Some(7));
    }

    // ── SpanReport ───────────────────────────────────────────────────────

    #[test]
    fn test_span_report_counts() {
        let mut ctx = PipelineSpanContext::new();

        // success
        ctx.begin_layer("echo").success();
        // success
        ctx.begin_layer("speculator").success();
        // error
        ctx.begin_layer("judge").error("timeout");
        // skipped
        ctx.begin_layer("graph").skip();

        let report = ctx.report();
        assert_eq!(report.success_count, 2);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.skipped_count, 1);
        assert_eq!(report.layer_spans.len(), 4);
    }

    #[test]
    fn test_span_report_slowest_layer() {
        let mut ctx = PipelineSpanContext::new();

        // echo: fast (synthetic 0 ms — Instant resolution may be <1 ms in tests)
        ctx.begin_layer("echo").success();

        // judge: introduce a tiny sleep so duration_ms is guaranteed > 0.
        {
            let mut span = ctx.begin_layer("judge");
            thread::sleep(Duration::from_millis(5));
            span.set_item_count(1);
            span.success();
        }

        let report = ctx.report();
        // judge must have a higher duration_ms than echo.
        let judge_duration = report
            .layer_spans
            .iter()
            .find(|s| s.layer_name == "judge")
            .map(|s| s.duration_ms)
            .unwrap_or(0);
        let echo_duration = report
            .layer_spans
            .iter()
            .find(|s| s.layer_name == "echo")
            .map(|s| s.duration_ms)
            .unwrap_or(0);

        assert!(
            judge_duration >= echo_duration,
            "judge ({judge_duration}ms) should be at least as slow as echo ({echo_duration}ms)"
        );
        assert_eq!(
            report.slowest_layer.as_deref(),
            Some("judge"),
            "slowest_layer should be judge"
        );
    }

    #[test]
    fn test_span_report_format_table() {
        let mut ctx = PipelineSpanContext::new();
        ctx.begin_layer("echo").success();
        ctx.begin_layer("judge").error("fail");

        let report = ctx.report();
        let table = report.format_table();

        assert!(
            table.contains("echo"),
            "table must contain layer name 'echo'"
        );
        assert!(table.contains("SUCCESS"), "table must contain SUCCESS");
        assert!(table.contains("ERROR"), "table must contain ERROR");
        assert!(table.contains("Total:"), "table must contain summary line");
    }

    #[test]
    fn test_span_report_to_json() {
        let mut ctx = PipelineSpanContext::new();
        ctx.begin_layer("echo").success();

        let report = ctx.report();
        let json = report.to_json().expect("JSON serialisation must succeed");
        assert!(
            json.contains("execution_id"),
            "JSON must contain execution_id"
        );
        assert!(
            json.contains("layer_spans"),
            "JSON must contain layer_spans"
        );
    }

    // ── Elapsed / timing ─────────────────────────────────────────────────

    #[test]
    fn test_pipeline_elapsed_ms() {
        let ctx = PipelineSpanContext::new();
        thread::sleep(Duration::from_millis(5));
        let elapsed = ctx.elapsed_ms();
        // We slept at least 5ms; allow generous upper bound for slow CI.
        assert!(
            elapsed >= 4,
            "elapsed_ms should be at least ~5ms, got {elapsed}"
        );
        assert!(
            elapsed < 2_000,
            "elapsed_ms should not exceed 2s in a unit test, got {elapsed}"
        );
    }

    // ── Drop safety ───────────────────────────────────────────────────────

    #[test]
    fn test_layer_span_drop_without_commit_records_error() {
        let mut ctx = PipelineSpanContext::new();
        {
            let _span = ctx.begin_layer("uncommitted");
            // dropped here without success/error/skip
        }
        assert_eq!(ctx.layer_spans.len(), 1);
        let record = &ctx.layer_spans[0];
        assert!(
            record.status.is_error(),
            "uncommitted span must be recorded as Error"
        );
        if let SpanStatus::Error(ref msg) = record.status {
            assert!(
                msg.contains("dropped"),
                "error message should mention 'dropped', got '{msg}'"
            );
        }
    }

    // ── record_pipeline_event ─────────────────────────────────────────────

    #[test]
    fn test_record_pipeline_event_does_not_panic() {
        let mut attrs = HashMap::new();
        attrs.insert("query".to_string(), "test".to_string());
        // Just verify it does not panic; tracing subscriber is a no-op in tests.
        record_pipeline_event("test_event", &attrs);
    }

    // ── SpanStatus helpers ────────────────────────────────────────────────

    #[test]
    fn test_span_status_helpers() {
        assert!(SpanStatus::Success.is_success());
        assert!(!SpanStatus::Success.is_error());
        assert!(!SpanStatus::Success.is_skipped());

        assert!(SpanStatus::Error("x".into()).is_error());
        assert!(!SpanStatus::Error("x".into()).is_success());

        assert!(SpanStatus::Skipped.is_skipped());
        assert!(SpanStatus::InProgress.is_in_progress());

        assert_eq!(SpanStatus::Success.label(), "SUCCESS");
        assert_eq!(SpanStatus::Error("y".into()).label(), "ERROR");
        assert_eq!(SpanStatus::Skipped.label(), "SKIPPED");
        assert_eq!(SpanStatus::InProgress.label(), "IN_PROGRESS");
    }
}
