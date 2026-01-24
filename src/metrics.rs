//! Pipeline metrics and tracing support.

use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrics collected for pipeline operations.
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// Total number of queries processed.
    pub query_count: u64,
    /// Total number of successful queries.
    pub success_count: u64,
    /// Total number of failed queries.
    pub error_count: u64,
    /// Average query latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Minimum query latency in milliseconds.
    pub min_latency_ms: u64,
    /// Maximum query latency in milliseconds.
    pub max_latency_ms: u64,
    /// Per-layer timing statistics.
    pub layer_timings: HashMap<String, LayerTiming>,
    /// Fast-path hit count.
    pub fast_path_hits: u64,
    /// Documents indexed count.
    pub documents_indexed: u64,
}

/// Timing statistics for a single layer.
#[derive(Debug, Clone, Default)]
pub struct LayerTiming {
    /// Total invocations of this layer.
    pub invocations: u64,
    /// Total time spent in this layer (milliseconds).
    pub total_time_ms: u64,
    /// Average time per invocation (milliseconds).
    pub avg_time_ms: f64,
    /// Minimum time (milliseconds).
    pub min_time_ms: u64,
    /// Maximum time (milliseconds).
    pub max_time_ms: u64,
}

impl LayerTiming {
    /// Record a timing observation.
    pub fn record(&mut self, duration_ms: u64) {
        self.invocations += 1;
        self.total_time_ms += duration_ms;

        if self.invocations == 1 {
            self.min_time_ms = duration_ms;
            self.max_time_ms = duration_ms;
        } else {
            self.min_time_ms = self.min_time_ms.min(duration_ms);
            self.max_time_ms = self.max_time_ms.max(duration_ms);
        }

        #[allow(clippy::cast_precision_loss)]
        {
            self.avg_time_ms = self.total_time_ms as f64 / self.invocations as f64;
        }
    }
}

/// Thread-safe metrics collector for pipeline operations.
pub struct MetricsCollector {
    query_count: AtomicU64,
    success_count: AtomicU64,
    error_count: AtomicU64,
    fast_path_hits: AtomicU64,
    documents_indexed: AtomicU64,
    total_latency_ms: AtomicU64,
    min_latency_ms: AtomicU64,
    max_latency_ms: AtomicU64,
    layer_timings: RwLock<HashMap<String, LayerTiming>>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    /// Create a new metrics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            fast_path_hits: AtomicU64::new(0),
            documents_indexed: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            min_latency_ms: AtomicU64::new(u64::MAX),
            max_latency_ms: AtomicU64::new(0),
            layer_timings: RwLock::new(HashMap::new()),
        }
    }

    /// Record a successful query with its duration.
    pub fn record_query_success(&self, duration: Duration) {
        let duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);

        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(duration_ms, Ordering::Relaxed);

        // Update min (using compare-and-swap loop)
        let mut current_min = self.min_latency_ms.load(Ordering::Relaxed);
        while duration_ms < current_min {
            match self.min_latency_ms.compare_exchange_weak(
                current_min,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        // Update max
        let mut current_max = self.max_latency_ms.load(Ordering::Relaxed);
        while duration_ms > current_max {
            match self.max_latency_ms.compare_exchange_weak(
                current_max,
                duration_ms,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    /// Record a failed query.
    pub fn record_query_error(&self) {
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fast-path hit.
    pub fn record_fast_path_hit(&self) {
        self.fast_path_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a document indexed.
    pub fn record_document_indexed(&self) {
        self.documents_indexed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record timing for a specific layer.
    pub fn record_layer_timing(&self, layer: &str, duration: Duration) {
        let duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);

        if let Ok(mut timings) = self.layer_timings.write() {
            timings
                .entry(layer.to_string())
                .or_default()
                .record(duration_ms);
        }
    }

    /// Get a snapshot of current metrics.
    #[must_use]
    pub fn snapshot(&self) -> PipelineMetrics {
        let query_count = self.query_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);

        #[allow(clippy::cast_precision_loss)]
        let avg_latency_ms = if success_count > 0 {
            total_latency as f64 / success_count as f64
        } else {
            0.0
        };

        let min_latency = self.min_latency_ms.load(Ordering::Relaxed);
        let max_latency = self.max_latency_ms.load(Ordering::Relaxed);

        let layer_timings = self
            .layer_timings
            .read()
            .map(|t| t.clone())
            .unwrap_or_default();

        PipelineMetrics {
            query_count,
            success_count,
            error_count: self.error_count.load(Ordering::Relaxed),
            avg_latency_ms,
            min_latency_ms: if min_latency == u64::MAX {
                0
            } else {
                min_latency
            },
            max_latency_ms: if max_latency == 0 && query_count == 0 {
                0
            } else {
                max_latency
            },
            layer_timings,
            fast_path_hits: self.fast_path_hits.load(Ordering::Relaxed),
            documents_indexed: self.documents_indexed.load(Ordering::Relaxed),
        }
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.query_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.fast_path_hits.store(0, Ordering::Relaxed);
        self.documents_indexed.store(0, Ordering::Relaxed);
        self.total_latency_ms.store(0, Ordering::Relaxed);
        self.min_latency_ms.store(u64::MAX, Ordering::Relaxed);
        self.max_latency_ms.store(0, Ordering::Relaxed);

        if let Ok(mut timings) = self.layer_timings.write() {
            timings.clear();
        }
    }
}

/// A guard that records timing when dropped.
pub struct TimingGuard<'a> {
    collector: &'a MetricsCollector,
    layer: String,
    start: Instant,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard for a layer.
    #[must_use]
    pub fn new(collector: &'a MetricsCollector, layer: impl Into<String>) -> Self {
        Self {
            collector,
            layer: layer.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for TimingGuard<'_> {
    fn drop(&mut self) {
        self.collector
            .record_layer_timing(&self.layer, self.start.elapsed());
    }
}

/// Extension trait for easy timing of operations.
pub trait TimedOperation {
    /// Start timing an operation on a layer.
    fn time_layer(&self, layer: impl Into<String>) -> TimingGuard<'_>;
}

impl TimedOperation for MetricsCollector {
    fn time_layer(&self, layer: impl Into<String>) -> TimingGuard<'_> {
        TimingGuard::new(self, layer)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_metrics_collector_basic() {
        let collector = MetricsCollector::new();

        collector.record_query_success(Duration::from_millis(100));
        collector.record_query_success(Duration::from_millis(200));
        collector.record_query_error();

        let metrics = collector.snapshot();

        assert_eq!(metrics.query_count, 3);
        assert_eq!(metrics.success_count, 2);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.min_latency_ms, 100);
        assert_eq!(metrics.max_latency_ms, 200);
        assert!((metrics.avg_latency_ms - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_layer_timing() {
        let collector = MetricsCollector::new();

        collector.record_layer_timing("Echo", Duration::from_millis(50));
        collector.record_layer_timing("Echo", Duration::from_millis(100));
        collector.record_layer_timing("Speculator", Duration::from_millis(200));

        let metrics = collector.snapshot();

        let echo_timing = metrics.layer_timings.get("Echo").unwrap();
        assert_eq!(echo_timing.invocations, 2);
        assert_eq!(echo_timing.min_time_ms, 50);
        assert_eq!(echo_timing.max_time_ms, 100);

        let spec_timing = metrics.layer_timings.get("Speculator").unwrap();
        assert_eq!(spec_timing.invocations, 1);
    }

    #[test]
    fn test_fast_path_tracking() {
        let collector = MetricsCollector::new();

        collector.record_fast_path_hit();
        collector.record_fast_path_hit();

        let metrics = collector.snapshot();
        assert_eq!(metrics.fast_path_hits, 2);
    }

    #[test]
    fn test_documents_indexed() {
        let collector = MetricsCollector::new();

        collector.record_document_indexed();
        collector.record_document_indexed();
        collector.record_document_indexed();

        let metrics = collector.snapshot();
        assert_eq!(metrics.documents_indexed, 3);
    }

    #[test]
    fn test_reset() {
        let collector = MetricsCollector::new();

        collector.record_query_success(Duration::from_millis(100));
        collector.record_layer_timing("Echo", Duration::from_millis(50));

        collector.reset();

        let metrics = collector.snapshot();
        assert_eq!(metrics.query_count, 0);
        assert_eq!(metrics.success_count, 0);
        assert!(metrics.layer_timings.is_empty());
    }

    #[test]
    fn test_timing_guard() {
        let collector = MetricsCollector::new();

        {
            let _guard = collector.time_layer("TestLayer");
            thread::sleep(Duration::from_millis(10));
        }

        let metrics = collector.snapshot();
        let timing = metrics.layer_timings.get("TestLayer").unwrap();
        assert_eq!(timing.invocations, 1);
        assert!(timing.total_time_ms >= 10);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let collector = Arc::new(MetricsCollector::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let c = Arc::clone(&collector);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    c.record_query_success(Duration::from_millis(10));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = collector.snapshot();
        assert_eq!(metrics.query_count, 1000);
        assert_eq!(metrics.success_count, 1000);
    }

    #[test]
    fn test_empty_metrics() {
        let collector = MetricsCollector::new();
        let metrics = collector.snapshot();

        assert_eq!(metrics.query_count, 0);
        assert_eq!(metrics.success_count, 0);
        assert_eq!(metrics.avg_latency_ms, 0.0);
        assert_eq!(metrics.min_latency_ms, 0);
        assert_eq!(metrics.max_latency_ms, 0);
    }
}
