//! Pipeline visualization and debugging tools.
//!
//! This module provides comprehensive tracing, debugging, and visualization
//! capabilities for `OxiRAG` pipeline execution.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use uuid::Uuid;

/// Unique identifier for a trace session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(String);

impl TraceId {
    /// Create a new trace ID from a string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new random trace ID.
    #[must_use]
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the inner string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A single entry in a pipeline trace representing one layer's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Name of the layer that was executed.
    pub layer_name: String,
    /// Start time as milliseconds since trace began.
    pub start_time_ms: u64,
    /// End time as milliseconds since trace began.
    pub end_time_ms: u64,
    /// Duration of the layer execution in milliseconds.
    pub duration_ms: u64,
    /// Summary of the input to this layer.
    pub input_summary: Option<String>,
    /// Summary of the output from this layer.
    pub output_summary: Option<String>,
    /// Memory usage before layer execution (bytes, if available).
    pub memory_before: Option<u64>,
    /// Memory usage after layer execution (bytes, if available).
    pub memory_after: Option<u64>,
    /// Error message if the layer failed.
    pub error: Option<String>,
}

impl TraceEntry {
    /// Create a new trace entry for a layer.
    #[must_use]
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            start_time_ms: 0,
            end_time_ms: 0,
            duration_ms: 0,
            input_summary: None,
            output_summary: None,
            memory_before: None,
            memory_after: None,
            error: None,
        }
    }

    /// Set the timing information.
    #[must_use]
    pub fn with_timing(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.start_time_ms = start_ms;
        self.end_time_ms = end_ms;
        self.duration_ms = end_ms.saturating_sub(start_ms);
        self
    }

    /// Set the input summary.
    #[must_use]
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.input_summary = Some(input.into());
        self
    }

    /// Set the output summary.
    #[must_use]
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output_summary = Some(output.into());
        self
    }

    /// Set memory information.
    #[must_use]
    pub const fn with_memory(mut self, before: u64, after: u64) -> Self {
        self.memory_before = Some(before);
        self.memory_after = Some(after);
        self
    }

    /// Set an error message.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Check if this entry represents a failed execution.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the memory delta (positive means growth, negative means shrinkage).
    #[must_use]
    pub fn memory_delta(&self) -> Option<i64> {
        match (self.memory_before, self.memory_after) {
            (Some(before), Some(after)) =>
            {
                #[allow(clippy::cast_possible_wrap)]
                Some(after as i64 - before as i64)
            }
            _ => None,
        }
    }
}

/// A complete trace of a pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTrace {
    /// Unique identifier for this trace.
    pub trace_id: TraceId,
    /// The query ID that initiated this trace.
    pub query_id: String,
    /// Timestamp when the trace started (as string for serialization).
    pub started_at: String,
    /// Total duration of the pipeline execution in milliseconds.
    pub total_duration_ms: u64,
    /// Individual entries for each layer.
    pub entries: Vec<TraceEntry>,
    /// Additional metadata about the trace.
    pub metadata: HashMap<String, String>,
    /// Whether the pipeline completed successfully.
    pub success: bool,
    /// Final error if the pipeline failed.
    pub final_error: Option<String>,
}

impl PipelineTrace {
    /// Create a new pipeline trace.
    #[must_use]
    pub fn new(trace_id: TraceId, query_id: impl Into<String>) -> Self {
        Self {
            trace_id,
            query_id: query_id.into(),
            started_at: chrono::Utc::now().to_rfc3339(),
            total_duration_ms: 0,
            entries: Vec::new(),
            metadata: HashMap::new(),
            success: true,
            final_error: None,
        }
    }

    /// Add an entry to the trace.
    pub fn add_entry(&mut self, entry: TraceEntry) {
        if entry.is_error() {
            self.success = false;
        }
        self.entries.push(entry);
    }

    /// Add metadata to the trace.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Set the total duration.
    pub fn set_duration(&mut self, duration_ms: u64) {
        self.total_duration_ms = duration_ms;
    }

    /// Mark the trace as failed with an error.
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.success = false;
        self.final_error = Some(error.into());
    }

    /// Get all layer names in execution order.
    #[must_use]
    pub fn layer_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.layer_name.as_str()).collect()
    }

    /// Get the total number of layers executed.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the slowest layer entry.
    #[must_use]
    pub fn slowest_layer(&self) -> Option<&TraceEntry> {
        self.entries.iter().max_by_key(|e| e.duration_ms)
    }

    /// Get all failed entries.
    #[must_use]
    pub fn failed_entries(&self) -> Vec<&TraceEntry> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }
}

/// Configuration for the pipeline debugger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Whether to capture input data for each layer.
    pub capture_inputs: bool,
    /// Whether to capture output data for each layer.
    pub capture_outputs: bool,
    /// Maximum size of trace data to store (in bytes).
    pub max_trace_size: usize,
    /// How long to retain traces (in seconds).
    pub retention_time_secs: u64,
    /// Maximum number of traces to keep in memory.
    pub max_traces: usize,
    /// Whether to capture memory usage.
    pub capture_memory: bool,
    /// Maximum length for input/output summaries.
    pub max_summary_length: usize,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            capture_inputs: true,
            capture_outputs: true,
            max_trace_size: 1024 * 1024, // 1 MB
            retention_time_secs: 3600,   // 1 hour
            max_traces: 1000,
            capture_memory: false,
            max_summary_length: 500,
        }
    }
}

impl DebugConfig {
    /// Create a minimal configuration for production use.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            capture_inputs: false,
            capture_outputs: false,
            max_trace_size: 256 * 1024, // 256 KB
            retention_time_secs: 300,   // 5 minutes
            max_traces: 100,
            capture_memory: false,
            max_summary_length: 100,
        }
    }

    /// Create a verbose configuration for debugging.
    #[must_use]
    pub fn verbose() -> Self {
        Self {
            capture_inputs: true,
            capture_outputs: true,
            max_trace_size: 10 * 1024 * 1024, // 10 MB
            retention_time_secs: 86400,       // 24 hours
            max_traces: 10000,
            capture_memory: true,
            max_summary_length: 2000,
        }
    }
}

/// Internal state for an active trace.
struct ActiveTrace {
    trace: PipelineTrace,
    start_instant: Instant,
    current_entry: Option<(String, Instant, Option<String>)>,
}

/// Pipeline debugger for capturing and analyzing execution traces.
pub struct PipelineDebugger {
    config: DebugConfig,
    active_traces: RwLock<HashMap<TraceId, ActiveTrace>>,
    completed_traces: RwLock<Vec<PipelineTrace>>,
    trace_count: AtomicU64,
}

impl PipelineDebugger {
    /// Create a new pipeline debugger with the given configuration.
    #[must_use]
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            active_traces: RwLock::new(HashMap::new()),
            completed_traces: RwLock::new(Vec::new()),
            trace_count: AtomicU64::new(0),
        }
    }

    /// Create a new pipeline debugger with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DebugConfig::default())
    }

    /// Get the debugger configuration.
    #[must_use]
    pub fn config(&self) -> &DebugConfig {
        &self.config
    }

    /// Start a new trace for a query.
    #[must_use]
    pub fn start_trace(&self, query_id: &str) -> TraceId {
        let trace_id = TraceId::generate();
        let trace = PipelineTrace::new(trace_id.clone(), query_id);
        let active = ActiveTrace {
            trace,
            start_instant: Instant::now(),
            current_entry: None,
        };

        if let Ok(mut traces) = self.active_traces.write() {
            traces.insert(trace_id.clone(), active);
        }

        self.trace_count.fetch_add(1, Ordering::Relaxed);
        trace_id
    }

    /// Record entry into a layer.
    pub fn record_layer_entry(&self, trace_id: &TraceId, layer: &str, input: &dyn Debug) {
        let input_summary = if self.config.capture_inputs {
            let debug_str = format!("{input:?}");
            Some(truncate_string(&debug_str, self.config.max_summary_length))
        } else {
            None
        };

        if let Ok(mut traces) = self.active_traces.write()
            && let Some(active) = traces.get_mut(trace_id)
        {
            active.current_entry = Some((layer.to_string(), Instant::now(), input_summary));
        }
    }

    /// Record exit from a layer.
    pub fn record_layer_exit(&self, trace_id: &TraceId, layer: &str, output: &dyn Debug) {
        let output_summary = if self.config.capture_outputs {
            let debug_str = format!("{output:?}");
            Some(truncate_string(&debug_str, self.config.max_summary_length))
        } else {
            None
        };

        if let Ok(mut traces) = self.active_traces.write()
            && let Some(active) = traces.get_mut(trace_id)
            && let Some((entry_layer, entry_start, input_summary)) = active.current_entry.take()
            && entry_layer == layer
        {
            let start_ms =
                u64::try_from(entry_start.duration_since(active.start_instant).as_millis())
                    .unwrap_or(u64::MAX);

            let end_ms = u64::try_from(
                Instant::now()
                    .duration_since(active.start_instant)
                    .as_millis(),
            )
            .unwrap_or(u64::MAX);

            let mut entry = TraceEntry::new(layer).with_timing(start_ms, end_ms);

            if let Some(input) = input_summary {
                entry = entry.with_input(input);
            }
            if let Some(output) = output_summary {
                entry = entry.with_output(output);
            }

            active.trace.add_entry(entry);
        }
    }

    /// Record a layer error.
    pub fn record_layer_error(&self, trace_id: &TraceId, layer: &str, error: &str) {
        if let Ok(mut traces) = self.active_traces.write()
            && let Some(active) = traces.get_mut(trace_id)
        {
            let start_ms = if let Some((_, entry_start, _)) = &active.current_entry {
                u64::try_from(entry_start.duration_since(active.start_instant).as_millis())
                    .unwrap_or(0)
            } else {
                0
            };

            let end_ms = u64::try_from(
                Instant::now()
                    .duration_since(active.start_instant)
                    .as_millis(),
            )
            .unwrap_or(u64::MAX);

            let entry = TraceEntry::new(layer)
                .with_timing(start_ms, end_ms)
                .with_error(error);

            active.trace.add_entry(entry);
            active.current_entry = None;
        }
    }

    /// Complete a trace and move it to completed storage.
    pub fn complete_trace(&self, trace_id: &TraceId) {
        let completed = if let Ok(mut traces) = self.active_traces.write() {
            traces.remove(trace_id).map(|mut active| {
                active.trace.set_duration(
                    u64::try_from(active.start_instant.elapsed().as_millis()).unwrap_or(u64::MAX),
                );
                active.trace
            })
        } else {
            None
        };

        if let Some(trace) = completed
            && let Ok(mut completed_traces) = self.completed_traces.write()
        {
            // Enforce max traces limit
            while completed_traces.len() >= self.config.max_traces {
                completed_traces.remove(0);
            }
            completed_traces.push(trace);
        }
    }

    /// Get a trace by ID.
    #[must_use]
    pub fn get_trace(&self, trace_id: &TraceId) -> Option<PipelineTrace> {
        // First check active traces
        if let Ok(traces) = self.active_traces.read()
            && let Some(active) = traces.get(trace_id)
        {
            return Some(active.trace.clone());
        }

        // Then check completed traces
        if let Ok(traces) = self.completed_traces.read() {
            return traces.iter().find(|t| &t.trace_id == trace_id).cloned();
        }

        None
    }

    /// Get all completed traces.
    #[must_use]
    pub fn get_all_traces(&self) -> Vec<PipelineTrace> {
        if let Ok(traces) = self.completed_traces.read() {
            traces.clone()
        } else {
            Vec::new()
        }
    }

    /// Get the total number of traces created.
    #[must_use]
    pub fn trace_count(&self) -> u64 {
        self.trace_count.load(Ordering::Relaxed)
    }

    /// Clear all completed traces.
    pub fn clear_traces(&self) {
        if let Ok(mut traces) = self.completed_traces.write() {
            traces.clear();
        }
    }

    /// Format a trace using the specified formatter.
    #[must_use]
    pub fn format_trace<F: TraceFormatter>(trace: &PipelineTrace) -> String {
        F::format(trace)
    }
}

impl Default for PipelineDebugger {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Trait for formatting pipeline traces.
pub trait TraceFormatter {
    /// Format a pipeline trace into a string representation.
    fn format(trace: &PipelineTrace) -> String;
}

/// Plain text trace formatter.
pub struct TextTraceFormatter;

impl TraceFormatter for TextTraceFormatter {
    fn format(trace: &PipelineTrace) -> String {
        let mut output = String::new();

        let _ = writeln!(output, "=== Pipeline Trace: {} ===", trace.trace_id);
        let _ = writeln!(output, "Query ID: {}", trace.query_id);
        let _ = writeln!(output, "Started: {}", trace.started_at);
        let _ = writeln!(output, "Total Duration: {}ms", trace.total_duration_ms);
        let status = if trace.success { "Success" } else { "Failed" };
        let _ = writeln!(output, "Status: {status}");

        if let Some(ref error) = trace.final_error {
            let _ = writeln!(output, "Error: {error}");
        }

        output.push_str("\n--- Layer Execution ---\n");

        for (i, entry) in trace.entries.iter().enumerate() {
            let _ = writeln!(output, "\n[{}] {}", i + 1, entry.layer_name);
            let _ = writeln!(
                output,
                "    Timing: {}ms - {}ms (duration: {}ms)",
                entry.start_time_ms, entry.end_time_ms, entry.duration_ms
            );

            if let Some(ref input) = entry.input_summary {
                let _ = writeln!(output, "    Input: {input}");
            }
            if let Some(ref out) = entry.output_summary {
                let _ = writeln!(output, "    Output: {out}");
            }
            if let Some(delta) = entry.memory_delta() {
                let _ = writeln!(output, "    Memory Delta: {delta} bytes");
            }
            if let Some(ref error) = entry.error {
                let _ = writeln!(output, "    ERROR: {error}");
            }
        }

        if !trace.metadata.is_empty() {
            output.push_str("\n--- Metadata ---\n");
            for (key, value) in &trace.metadata {
                let _ = writeln!(output, "  {key}: {value}");
            }
        }

        output
    }
}

/// JSON trace formatter.
pub struct JsonTraceFormatter;

impl TraceFormatter for JsonTraceFormatter {
    fn format(trace: &PipelineTrace) -> String {
        serde_json::to_string_pretty(trace).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }
}

/// Mermaid diagram trace formatter.
pub struct MermaidTraceFormatter;

impl TraceFormatter for MermaidTraceFormatter {
    fn format(trace: &PipelineTrace) -> String {
        let mut output = String::new();

        output.push_str("```mermaid\n");
        output.push_str("sequenceDiagram\n");
        output.push_str("    participant Q as Query\n");
        output.push_str("    participant P as Pipeline\n");

        // Add participants for each unique layer
        let mut seen_layers = Vec::new();
        for entry in &trace.entries {
            if !seen_layers.contains(&entry.layer_name) {
                let safe_name = sanitize_mermaid_id(&entry.layer_name);
                let _ = writeln!(
                    output,
                    "    participant {safe_name} as {}",
                    entry.layer_name
                );
                seen_layers.push(entry.layer_name.clone());
            }
        }

        output.push('\n');

        // Add the query start
        let _ = writeln!(output, "    Q->>P: Start ({})", trace.query_id);

        // Add each layer execution
        for entry in &trace.entries {
            let safe_name = sanitize_mermaid_id(&entry.layer_name);

            if entry.is_error() {
                let _ = writeln!(
                    output,
                    "    P-x{safe_name}: Error ({}ms)",
                    entry.duration_ms
                );
                if let Some(ref error) = entry.error {
                    let short_error = truncate_string(error, 30);
                    let _ = writeln!(output, "    Note right of {safe_name}: {short_error}");
                }
            } else {
                let _ = writeln!(
                    output,
                    "    P->>{safe_name}: Process ({}ms)",
                    entry.duration_ms
                );
                let _ = writeln!(output, "    {safe_name}-->>P: Complete");
            }
        }

        // Add result
        if trace.success {
            let _ = writeln!(
                output,
                "    P-->>Q: Success ({}ms total)",
                trace.total_duration_ms
            );
        } else {
            let _ = writeln!(
                output,
                "    P--xQ: Failed ({}ms total)",
                trace.total_duration_ms
            );
        }

        output.push_str("```\n");
        output
    }
}

/// Gantt chart trace formatter using Mermaid.
pub struct GanttTraceFormatter;

impl TraceFormatter for GanttTraceFormatter {
    fn format(trace: &PipelineTrace) -> String {
        let mut output = String::new();

        output.push_str("```mermaid\n");
        output.push_str("gantt\n");
        let _ = writeln!(output, "    title Pipeline Trace: {}", trace.query_id);
        output.push_str("    dateFormat x\n");
        output.push_str("    axisFormat %L ms\n\n");

        for entry in &trace.entries {
            let status = if entry.is_error() { "crit," } else { "" };
            let _ = writeln!(
                output,
                "    {} :{status}{}ms, {}ms",
                entry.layer_name, entry.start_time_ms, entry.duration_ms
            );
        }

        output.push_str("```\n");
        output
    }
}

/// Thread-safe debugger wrapper for shared access.
pub type SharedPipelineDebugger = Arc<PipelineDebugger>;

/// Create a new shared pipeline debugger.
#[must_use]
pub fn create_shared_debugger(config: DebugConfig) -> SharedPipelineDebugger {
    Arc::new(PipelineDebugger::new(config))
}

/// Guard that automatically records layer entry and exit.
pub struct LayerTraceGuard<'a> {
    debugger: &'a PipelineDebugger,
    trace_id: TraceId,
    layer_name: String,
    output: Option<Box<dyn Debug + Send + Sync>>,
}

impl<'a> LayerTraceGuard<'a> {
    /// Create a new guard that records layer entry.
    #[must_use]
    pub fn new(
        debugger: &'a PipelineDebugger,
        trace_id: TraceId,
        layer_name: impl Into<String>,
        input: &dyn Debug,
    ) -> Self {
        let layer = layer_name.into();
        debugger.record_layer_entry(&trace_id, &layer, input);
        Self {
            debugger,
            trace_id,
            layer_name: layer,
            output: None,
        }
    }

    /// Set the output to be recorded on drop.
    pub fn set_output(&mut self, output: impl Debug + Send + Sync + 'static) {
        self.output = Some(Box::new(output));
    }

    /// Record an error and mark this layer as failed.
    pub fn record_error(&self, error: &str) {
        self.debugger
            .record_layer_error(&self.trace_id, &self.layer_name, error);
    }
}

impl Drop for LayerTraceGuard<'_> {
    fn drop(&mut self) {
        if let Some(ref output) = self.output {
            self.debugger
                .record_layer_exit(&self.trace_id, &self.layer_name, output.as_ref());
        } else {
            // Record with empty output if none was set
            self.debugger.record_layer_exit(
                &self.trace_id,
                &self.layer_name,
                &"<no output recorded>",
            );
        }
    }
}

/// Truncate a string to a maximum length, adding "..." if truncated.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Sanitize a string for use as a Mermaid participant ID.
fn sanitize_mermaid_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_trace_id_generation() {
        let id1 = TraceId::generate();
        let id2 = TraceId::generate();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_trace_id_from_string() {
        let id = TraceId::new("test-trace-123");
        assert_eq!(id.as_str(), "test-trace-123");
        assert_eq!(id.to_string(), "test-trace-123");
    }

    #[test]
    fn test_trace_entry_creation() {
        let entry = TraceEntry::new("Echo")
            .with_timing(0, 100)
            .with_input("query text")
            .with_output("search results")
            .with_memory(1000, 2000);

        assert_eq!(entry.layer_name, "Echo");
        assert_eq!(entry.start_time_ms, 0);
        assert_eq!(entry.end_time_ms, 100);
        assert_eq!(entry.duration_ms, 100);
        assert_eq!(entry.input_summary, Some("query text".to_string()));
        assert_eq!(entry.output_summary, Some("search results".to_string()));
        assert_eq!(entry.memory_before, Some(1000));
        assert_eq!(entry.memory_after, Some(2000));
        assert!(!entry.is_error());
    }

    #[test]
    fn test_trace_entry_with_error() {
        let entry = TraceEntry::new("Judge")
            .with_timing(50, 150)
            .with_error("Verification timeout");

        assert!(entry.is_error());
        assert_eq!(entry.error, Some("Verification timeout".to_string()));
    }

    #[test]
    fn test_trace_entry_memory_delta() {
        let entry = TraceEntry::new("Echo").with_memory(1000, 1500);
        assert_eq!(entry.memory_delta(), Some(500));

        let shrink_entry = TraceEntry::new("Echo").with_memory(2000, 1500);
        assert_eq!(shrink_entry.memory_delta(), Some(-500));

        let no_memory_entry = TraceEntry::new("Echo");
        assert_eq!(no_memory_entry.memory_delta(), None);
    }

    #[test]
    fn test_pipeline_trace_creation() {
        let trace_id = TraceId::new("test-trace");
        let trace = PipelineTrace::new(trace_id, "query-123");

        assert_eq!(trace.query_id, "query-123");
        assert!(trace.success);
        assert!(trace.entries.is_empty());
    }

    #[test]
    fn test_pipeline_trace_add_entries() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");

        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(TraceEntry::new("Speculator").with_timing(50, 100));
        trace.add_entry(TraceEntry::new("Judge").with_timing(100, 200));

        assert_eq!(trace.layer_count(), 3);
        assert_eq!(trace.layer_names(), vec!["Echo", "Speculator", "Judge"]);
    }

    #[test]
    fn test_pipeline_trace_slowest_layer() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");

        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(TraceEntry::new("Speculator").with_timing(50, 200));
        trace.add_entry(TraceEntry::new("Judge").with_timing(200, 250));

        let slowest = trace.slowest_layer().unwrap();
        assert_eq!(slowest.layer_name, "Speculator");
        assert_eq!(slowest.duration_ms, 150);
    }

    #[test]
    fn test_pipeline_trace_failed_entries() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");

        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(
            TraceEntry::new("Speculator")
                .with_timing(50, 100)
                .with_error("Failed"),
        );
        trace.add_entry(TraceEntry::new("Judge").with_timing(100, 150));

        let failed = trace.failed_entries();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].layer_name, "Speculator");
        assert!(!trace.success);
    }

    #[test]
    fn test_debug_config_default() {
        let config = DebugConfig::default();
        assert!(config.capture_inputs);
        assert!(config.capture_outputs);
        assert_eq!(config.max_traces, 1000);
    }

    #[test]
    fn test_debug_config_minimal() {
        let config = DebugConfig::minimal();
        assert!(!config.capture_inputs);
        assert!(!config.capture_outputs);
        assert_eq!(config.max_traces, 100);
    }

    #[test]
    fn test_debug_config_verbose() {
        let config = DebugConfig::verbose();
        assert!(config.capture_inputs);
        assert!(config.capture_outputs);
        assert!(config.capture_memory);
        assert_eq!(config.max_traces, 10000);
    }

    #[test]
    fn test_pipeline_debugger_start_trace() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("query-1");

        assert_eq!(debugger.trace_count(), 1);

        let trace = debugger.get_trace(&trace_id).unwrap();
        assert_eq!(trace.query_id, "query-1");
    }

    #[test]
    fn test_pipeline_debugger_record_layer() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("query-1");

        debugger.record_layer_entry(&trace_id, "Echo", &"input data");
        std::thread::sleep(Duration::from_millis(10));
        debugger.record_layer_exit(&trace_id, "Echo", &"output data");

        let trace = debugger.get_trace(&trace_id).unwrap();
        assert_eq!(trace.entries.len(), 1);
        assert_eq!(trace.entries[0].layer_name, "Echo");
        assert!(trace.entries[0].duration_ms >= 10);
    }

    #[test]
    fn test_pipeline_debugger_record_error() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("query-1");

        debugger.record_layer_entry(&trace_id, "Judge", &"input");
        debugger.record_layer_error(&trace_id, "Judge", "Timeout occurred");

        let trace = debugger.get_trace(&trace_id).unwrap();
        assert_eq!(trace.entries.len(), 1);
        assert!(trace.entries[0].is_error());
        assert_eq!(trace.entries[0].error, Some("Timeout occurred".to_string()));
    }

    #[test]
    fn test_pipeline_debugger_complete_trace() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("query-1");

        debugger.record_layer_entry(&trace_id, "Echo", &"input");
        debugger.record_layer_exit(&trace_id, "Echo", &"output");

        debugger.complete_trace(&trace_id);

        let traces = debugger.get_all_traces();
        assert_eq!(traces.len(), 1);
        // Duration is always defined (u64), just ensure we can access it
        let _ = traces[0].total_duration_ms;
    }

    #[test]
    fn test_pipeline_debugger_max_traces() {
        let config = DebugConfig {
            max_traces: 3,
            ..Default::default()
        };
        let debugger = PipelineDebugger::new(config);

        for i in 0..5 {
            let trace_id = debugger.start_trace(&format!("query-{i}"));
            debugger.complete_trace(&trace_id);
        }

        let traces = debugger.get_all_traces();
        assert_eq!(traces.len(), 3);
        // Should have kept the most recent traces
        assert_eq!(traces[0].query_id, "query-2");
        assert_eq!(traces[1].query_id, "query-3");
        assert_eq!(traces[2].query_id, "query-4");
    }

    #[test]
    fn test_pipeline_debugger_clear_traces() {
        let debugger = PipelineDebugger::with_defaults();

        let trace_id = debugger.start_trace("query-1");
        debugger.complete_trace(&trace_id);

        assert_eq!(debugger.get_all_traces().len(), 1);

        debugger.clear_traces();
        assert!(debugger.get_all_traces().is_empty());
    }

    #[test]
    fn test_text_trace_formatter() {
        let mut trace = PipelineTrace::new(TraceId::new("test-trace"), "query-123");
        trace.add_entry(
            TraceEntry::new("Echo")
                .with_timing(0, 50)
                .with_input("test input"),
        );
        trace.set_duration(50);

        let formatted = TextTraceFormatter::format(&trace);

        assert!(formatted.contains("Pipeline Trace: test-trace"));
        assert!(formatted.contains("Query ID: query-123"));
        assert!(formatted.contains("Echo"));
        assert!(formatted.contains("50ms"));
    }

    #[test]
    fn test_json_trace_formatter() {
        let trace = PipelineTrace::new(TraceId::new("test-trace"), "query-123");
        let formatted = JsonTraceFormatter::format(&trace);

        assert!(formatted.contains("\"trace_id\""));
        assert!(formatted.contains("\"query_id\": \"query-123\""));

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&formatted).unwrap();
        assert_eq!(parsed["query_id"], "query-123");
    }

    #[test]
    fn test_mermaid_trace_formatter() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(TraceEntry::new("Speculator").with_timing(50, 100));
        trace.set_duration(100);

        let formatted = MermaidTraceFormatter::format(&trace);

        assert!(formatted.contains("```mermaid"));
        assert!(formatted.contains("sequenceDiagram"));
        assert!(formatted.contains("participant Echo"));
        assert!(formatted.contains("participant Speculator"));
        assert!(formatted.contains("Success"));
    }

    #[test]
    fn test_mermaid_trace_formatter_with_error() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(
            TraceEntry::new("Judge")
                .with_timing(50, 100)
                .with_error("Failed"),
        );
        trace.set_duration(100);

        let formatted = MermaidTraceFormatter::format(&trace);

        assert!(formatted.contains("P-xJudge: Error"));
        assert!(formatted.contains("Failed"));
    }

    #[test]
    fn test_gantt_trace_formatter() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.add_entry(TraceEntry::new("Echo").with_timing(0, 50));
        trace.add_entry(TraceEntry::new("Speculator").with_timing(50, 150));
        trace.set_duration(150);

        let formatted = GanttTraceFormatter::format(&trace);

        assert!(formatted.contains("```mermaid"));
        assert!(formatted.contains("gantt"));
        assert!(formatted.contains("Pipeline Trace: q1"));
        assert!(formatted.contains("Echo"));
        assert!(formatted.contains("Speculator"));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
        assert_eq!(truncate_string("hi", 2), "hi"); // fits exactly
        assert_eq!(truncate_string("hello", 3), "..."); // too short for content + ellipsis
        assert_eq!(truncate_string("", 5), "");
    }

    #[test]
    fn test_sanitize_mermaid_id() {
        assert_eq!(sanitize_mermaid_id("Echo"), "Echo");
        assert_eq!(sanitize_mermaid_id("Layer-1"), "Layer1");
        assert_eq!(sanitize_mermaid_id("My Layer!"), "MyLayer");
        assert_eq!(sanitize_mermaid_id("test_layer"), "test_layer");
    }

    #[test]
    fn test_shared_debugger() {
        let debugger = create_shared_debugger(DebugConfig::default());

        let trace_id = debugger.start_trace("shared-query");
        debugger.record_layer_entry(&trace_id, "Echo", &"input");
        debugger.record_layer_exit(&trace_id, "Echo", &"output");
        debugger.complete_trace(&trace_id);

        let traces = debugger.get_all_traces();
        assert_eq!(traces.len(), 1);
    }

    #[test]
    fn test_layer_trace_guard() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("guard-test");

        {
            let mut guard =
                LayerTraceGuard::new(&debugger, trace_id.clone(), "Echo", &"input data");
            guard.set_output("output data");
            // Guard drops here and records exit
        }

        let trace = debugger.get_trace(&trace_id).unwrap();
        assert_eq!(trace.entries.len(), 1);
        assert_eq!(trace.entries[0].layer_name, "Echo");
    }

    #[test]
    fn test_layer_trace_guard_with_error() {
        let debugger = PipelineDebugger::with_defaults();
        let trace_id = debugger.start_trace("guard-error-test");

        {
            let guard = LayerTraceGuard::new(&debugger, trace_id.clone(), "Judge", &"input");
            guard.record_error("Something went wrong");
            // Don't set output - the error was recorded
        }

        let trace = debugger.get_trace(&trace_id).unwrap();
        // Should have 2 entries: one error, one normal exit
        assert!(!trace.failed_entries().is_empty());
    }

    #[test]
    fn test_pipeline_trace_metadata() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.add_metadata("user_id", "user-123");
        trace.add_metadata("environment", "production");

        assert_eq!(trace.metadata.get("user_id"), Some(&"user-123".to_string()));
        assert_eq!(
            trace.metadata.get("environment"),
            Some(&"production".to_string())
        );
    }

    #[test]
    fn test_pipeline_trace_set_error() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        assert!(trace.success);

        trace.set_error("Pipeline failed");

        assert!(!trace.success);
        assert_eq!(trace.final_error, Some("Pipeline failed".to_string()));
    }

    #[test]
    fn test_debugger_config_accessor() {
        let config = DebugConfig {
            capture_inputs: false,
            max_traces: 500,
            ..Default::default()
        };
        let debugger = PipelineDebugger::new(config);

        assert!(!debugger.config().capture_inputs);
        assert_eq!(debugger.config().max_traces, 500);
    }

    #[test]
    fn test_text_formatter_with_metadata() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.add_metadata("key1", "value1");
        trace.add_metadata("key2", "value2");
        trace.set_duration(100);

        let formatted = TextTraceFormatter::format(&trace);

        assert!(formatted.contains("--- Metadata ---"));
        assert!(formatted.contains("key1: value1"));
        assert!(formatted.contains("key2: value2"));
    }

    #[test]
    fn test_text_formatter_with_error() {
        let mut trace = PipelineTrace::new(TraceId::new("test"), "q1");
        trace.set_error("Final error message");
        trace.set_duration(50);

        let formatted = TextTraceFormatter::format(&trace);

        assert!(formatted.contains("Status: Failed"));
        assert!(formatted.contains("Error: Final error message"));
    }

    #[test]
    fn test_multiple_traces_concurrent() {
        use std::thread;

        let debugger = Arc::new(PipelineDebugger::with_defaults());
        let mut handles = vec![];

        for i in 0..10 {
            let d = Arc::clone(&debugger);
            handles.push(thread::spawn(move || {
                let trace_id = d.start_trace(&format!("query-{i}"));
                d.record_layer_entry(&trace_id, "Echo", &i);
                d.record_layer_exit(&trace_id, "Echo", &(i * 2));
                d.complete_trace(&trace_id);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(debugger.trace_count(), 10);
        assert_eq!(debugger.get_all_traces().len(), 10);
    }

    #[test]
    fn test_trace_entry_timing_overflow() {
        let entry = TraceEntry::new("Test").with_timing(100, 50);
        // Saturating subtraction should result in 0, not overflow
        assert_eq!(entry.duration_ms, 0);
    }

    #[test]
    fn test_format_trace_static_dispatch() {
        let trace = PipelineTrace::new(TraceId::new("test"), "q1");

        let text = PipelineDebugger::format_trace::<TextTraceFormatter>(&trace);
        let json = PipelineDebugger::format_trace::<JsonTraceFormatter>(&trace);
        let mermaid = PipelineDebugger::format_trace::<MermaidTraceFormatter>(&trace);

        assert!(text.contains("Pipeline Trace"));
        assert!(json.contains("trace_id"));
        assert!(mermaid.contains("sequenceDiagram"));
    }
}
