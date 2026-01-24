//! Load testing utilities for `OxiRAG` pipeline performance evaluation.
//!
//! This module provides tools for conducting load tests on the RAG pipeline,
//! measuring latencies, throughput, and identifying performance bottlenecks.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::load_testing::{LoadTest, LoadTestConfig, MockQueryGenerator};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = LoadTestConfig::new()
//!         .with_concurrent_users(10)
//!         .with_total_requests(1000)
//!         .with_duration(Duration::from_secs(60));
//!
//!     let generator = MockQueryGenerator::new();
//!     let load_test = LoadTest::new(config, Box::new(generator));
//!     let result = load_test.run().await;
//!
//!     println!("Total requests: {}", result.total_requests);
//!     println!("P99 latency: {:?}", result.p99_latency);
//!     println!("Requests/sec: {:.2}", result.requests_per_second);
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::types::Query;

/// Configuration for load testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Number of concurrent simulated users.
    pub concurrent_users: usize,
    /// Total number of requests to execute.
    pub total_requests: usize,
    /// Duration of the test.
    pub duration: Duration,
    /// Time to ramp up to full concurrency.
    pub ramp_up_time: Duration,
    /// Time to ramp down from full concurrency.
    pub ramp_down_time: Duration,
    /// Delay between requests per user (think time) in milliseconds.
    pub think_time_ms: u64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrent_users: 1,
            total_requests: 100,
            duration: Duration::from_secs(60),
            ramp_up_time: Duration::from_secs(5),
            ramp_down_time: Duration::from_secs(5),
            think_time_ms: 0,
        }
    }
}

impl LoadTestConfig {
    /// Create a new load test configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of concurrent users.
    #[must_use]
    pub fn with_concurrent_users(mut self, users: usize) -> Self {
        self.concurrent_users = users;
        self
    }

    /// Set the total number of requests.
    #[must_use]
    pub fn with_total_requests(mut self, requests: usize) -> Self {
        self.total_requests = requests;
        self
    }

    /// Set the test duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set the ramp-up time.
    #[must_use]
    pub fn with_ramp_up_time(mut self, ramp_up: Duration) -> Self {
        self.ramp_up_time = ramp_up;
        self
    }

    /// Set the ramp-down time.
    #[must_use]
    pub fn with_ramp_down_time(mut self, ramp_down: Duration) -> Self {
        self.ramp_down_time = ramp_down;
        self
    }

    /// Set the think time in milliseconds.
    #[must_use]
    pub fn with_think_time_ms(mut self, think_time_ms: u64) -> Self {
        self.think_time_ms = think_time_ms;
        self
    }
}

/// Result of a load test execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    /// Total number of requests attempted.
    pub total_requests: u64,
    /// Number of successful requests.
    pub successful_requests: u64,
    /// Number of failed requests.
    pub failed_requests: u64,
    /// Minimum latency observed.
    pub min_latency: Duration,
    /// Maximum latency observed.
    pub max_latency: Duration,
    /// Average latency.
    pub avg_latency: Duration,
    /// 50th percentile latency (median).
    pub p50_latency: Duration,
    /// 95th percentile latency.
    pub p95_latency: Duration,
    /// 99th percentile latency.
    pub p99_latency: Duration,
    /// Requests processed per second.
    pub requests_per_second: f64,
    /// Total test duration.
    pub test_duration: Duration,
    /// Map of error types to counts.
    pub errors: HashMap<String, u64>,
    /// Individual request latencies for detailed analysis.
    pub latencies: Vec<Duration>,
}

impl Default for LoadTestResult {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            avg_latency: Duration::ZERO,
            p50_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            requests_per_second: 0.0,
            test_duration: Duration::ZERO,
            errors: HashMap::new(),
            latencies: Vec::new(),
        }
    }
}

impl LoadTestResult {
    /// Create a new empty load test result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate percentile latencies from recorded latencies.
    #[must_use]
    pub fn calculate_percentiles(mut self) -> Self {
        if self.latencies.is_empty() {
            return self;
        }

        let mut sorted_latencies = self.latencies.clone();
        sorted_latencies.sort();

        let len = sorted_latencies.len();
        // Use saturating_sub to avoid underflow when calculating percentile indices
        self.p50_latency = sorted_latencies[(len * 50).saturating_sub(1).max(1) / 100];
        self.p95_latency = sorted_latencies[(len * 95).saturating_sub(1).max(1) / 100];
        self.p99_latency = sorted_latencies
            .get((len * 99).saturating_sub(1).max(1) / 100)
            .copied()
            .unwrap_or(self.max_latency);

        self
    }

    /// Merge another result into this one.
    pub fn merge(&mut self, other: &LoadTestResult) {
        self.total_requests += other.total_requests;
        self.successful_requests += other.successful_requests;
        self.failed_requests += other.failed_requests;
        self.min_latency = self.min_latency.min(other.min_latency);
        self.max_latency = self.max_latency.max(other.max_latency);
        self.latencies.extend(other.latencies.iter().copied());

        for (error, count) in &other.errors {
            *self.errors.entry(error.clone()).or_insert(0) += count;
        }
    }

    /// Finalize the result by calculating derived metrics.
    pub fn finalize(&mut self, test_duration: Duration) {
        self.test_duration = test_duration;

        if !self.latencies.is_empty() {
            let mut sorted = self.latencies.clone();
            sorted.sort();

            let len = sorted.len();
            // Use saturating_sub to avoid underflow when calculating percentile indices
            self.p50_latency = sorted[(len * 50).saturating_sub(1).max(1) / 100];
            self.p95_latency = sorted[(len * 95).saturating_sub(1).max(1) / 100];
            self.p99_latency = sorted
                .get((len * 99).saturating_sub(1).max(1) / 100)
                .copied()
                .unwrap_or(self.max_latency);

            let total_ns: u128 = self
                .latencies
                .iter()
                .map(std::time::Duration::as_nanos)
                .sum();
            #[allow(clippy::cast_possible_truncation)]
            let avg_ns = (total_ns / len as u128) as u64;
            self.avg_latency = Duration::from_nanos(avg_ns);
        }

        let secs = test_duration.as_secs_f64();
        if secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let rps = self.total_requests as f64 / secs;
            self.requests_per_second = rps;
        }
    }
}

/// A single request execution result.
#[derive(Debug, Clone)]
pub struct RequestResult {
    /// Whether the request succeeded.
    pub success: bool,
    /// Latency of the request.
    pub latency: Duration,
    /// Error message if failed.
    pub error: Option<String>,
}

impl RequestResult {
    /// Create a successful request result.
    #[must_use]
    pub fn success(latency: Duration) -> Self {
        Self {
            success: true,
            latency,
            error: None,
        }
    }

    /// Create a failed request result.
    #[must_use]
    pub fn failure(latency: Duration, error: String) -> Self {
        Self {
            success: false,
            latency,
            error: Some(error),
        }
    }
}

/// Trait for generating queries during load testing.
#[async_trait]
pub trait QueryGenerator: Send + Sync {
    /// Generate a random query for testing.
    fn generate(&self) -> Query;
}

/// Mock query generator for testing purposes.
#[derive(Debug, Clone)]
pub struct MockQueryGenerator {
    queries: Vec<String>,
    counter: Arc<AtomicU64>,
}

impl Default for MockQueryGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MockQueryGenerator {
    /// Create a new mock query generator with default queries.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queries: vec![
                "What is machine learning?".to_string(),
                "How does natural language processing work?".to_string(),
                "Explain the RAG architecture".to_string(),
                "What are vector embeddings?".to_string(),
                "How to implement semantic search?".to_string(),
                "What is the difference between AI and ML?".to_string(),
                "How does BERT work?".to_string(),
                "What is transformer architecture?".to_string(),
                "Explain attention mechanism".to_string(),
                "What is knowledge graph?".to_string(),
            ],
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a mock query generator with custom queries.
    #[must_use]
    pub fn with_queries(queries: Vec<String>) -> Self {
        Self {
            queries,
            counter: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl QueryGenerator for MockQueryGenerator {
    fn generate(&self) -> Query {
        #[allow(clippy::cast_possible_truncation)]
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) as usize % self.queries.len();
        Query::new(&self.queries[idx])
    }
}

/// Trait for executing queries (to be implemented by pipeline wrappers).
#[async_trait]
pub trait QueryExecutor: Send + Sync {
    /// Execute a query and return the result.
    async fn execute(&self, query: Query) -> RequestResult;
}

/// Mock query executor for testing.
#[derive(Debug, Clone)]
pub struct MockQueryExecutor {
    /// Simulated latency per request.
    pub latency: Duration,
    /// Failure rate (0.0 to 1.0).
    pub failure_rate: f64,
    request_count: Arc<AtomicU64>,
}

impl Default for MockQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl MockQueryExecutor {
    /// Create a new mock executor with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latency: Duration::from_millis(10),
            failure_rate: 0.0,
            request_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set the simulated latency.
    #[must_use]
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Set the failure rate.
    #[must_use]
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Get the total number of requests executed.
    #[must_use]
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl QueryExecutor for MockQueryExecutor {
    async fn execute(&self, _query: Query) -> RequestResult {
        let count = self.request_count.fetch_add(1, Ordering::Relaxed);

        // Simulate processing time
        #[cfg(feature = "native")]
        tokio::time::sleep(self.latency).await;

        // Simulate failures based on failure rate
        #[allow(clippy::cast_precision_loss)]
        let should_fail = (count as f64 / 100.0).fract() < self.failure_rate;

        if should_fail {
            RequestResult::failure(self.latency, "Simulated failure".to_string())
        } else {
            RequestResult::success(self.latency)
        }
    }
}

/// Main load test executor.
pub struct LoadTest<G: QueryGenerator, E: QueryExecutor> {
    config: LoadTestConfig,
    generator: G,
    executor: E,
}

impl<G: QueryGenerator, E: QueryExecutor> LoadTest<G, E> {
    /// Create a new load test with the given configuration.
    #[must_use]
    pub fn new(config: LoadTestConfig, generator: G, executor: E) -> Self {
        Self {
            config,
            generator,
            executor,
        }
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &LoadTestConfig {
        &self.config
    }
}

#[cfg(feature = "native")]
impl<G: QueryGenerator + 'static, E: QueryExecutor + 'static> LoadTest<G, E> {
    /// Execute the load test.
    pub async fn run(&self) -> LoadTestResult {
        let start = Instant::now();
        let mut result = LoadTestResult::new();

        // Calculate requests per user
        let requests_per_user = self.config.total_requests / self.config.concurrent_users.max(1);
        let remaining = self.config.total_requests % self.config.concurrent_users.max(1);

        // Execute requests sequentially for simplicity (concurrent version below)
        for i in 0..self.config.concurrent_users {
            let user_requests = if i < remaining {
                requests_per_user + 1
            } else {
                requests_per_user
            };

            for _ in 0..user_requests {
                let query = self.generator.generate();
                let req_result = self.executor.execute(query).await;

                result.total_requests += 1;
                result.latencies.push(req_result.latency);
                result.min_latency = result.min_latency.min(req_result.latency);
                result.max_latency = result.max_latency.max(req_result.latency);

                if req_result.success {
                    result.successful_requests += 1;
                } else {
                    result.failed_requests += 1;
                    if let Some(error) = req_result.error {
                        *result.errors.entry(error).or_insert(0) += 1;
                    }
                }

                // Apply think time
                if self.config.think_time_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(self.config.think_time_ms)).await;
                }
            }
        }

        result.finalize(start.elapsed());
        result
    }

    /// Execute concurrent queries.
    pub async fn run_concurrent(&self, queries: Vec<Query>) -> LoadTestResult {
        use tokio::sync::Semaphore;

        let start = Instant::now();
        let semaphore = Arc::new(Semaphore::new(self.config.concurrent_users));
        let results = Arc::new(tokio::sync::Mutex::new(Vec::new()));

        for query in queries {
            let permit = semaphore.clone().acquire_owned().await;
            let results_clone = Arc::clone(&results);

            // Execute the query with semaphore limiting concurrency
            let req_start = Instant::now();
            let req_result = self.executor.execute(query).await;
            let latency = req_start.elapsed();

            if let Ok(_permit) = permit {
                results_clone.lock().await.push(RequestResult {
                    success: req_result.success,
                    latency,
                    error: req_result.error,
                });
            }
        }

        // Compile results
        let mut result = LoadTestResult::new();
        let collected = results.lock().await;

        for req_result in collected.iter() {
            result.total_requests += 1;
            result.latencies.push(req_result.latency);
            result.min_latency = result.min_latency.min(req_result.latency);
            result.max_latency = result.max_latency.max(req_result.latency);

            if req_result.success {
                result.successful_requests += 1;
            } else {
                result.failed_requests += 1;
                if let Some(ref error) = req_result.error {
                    *result.errors.entry(error.clone()).or_insert(0) += 1;
                }
            }
        }

        result.finalize(start.elapsed());
        result
    }

    /// Execute sustained load at a target QPS for a given duration.
    pub async fn run_sustained(&self, duration: Duration, qps: f64) -> LoadTestResult {
        let start = Instant::now();
        let mut result = LoadTestResult::new();

        let interval = if qps > 0.0 {
            Duration::from_secs_f64(1.0 / qps)
        } else {
            Duration::from_secs(1)
        };

        let mut next_request = Instant::now();

        while start.elapsed() < duration {
            // Wait until next scheduled request
            if Instant::now() < next_request {
                let sleep_duration = next_request - Instant::now();
                tokio::time::sleep(sleep_duration).await;
            }

            // Execute request
            let query = self.generator.generate();
            let req_result = self.executor.execute(query).await;

            result.total_requests += 1;
            result.latencies.push(req_result.latency);
            result.min_latency = result.min_latency.min(req_result.latency);
            result.max_latency = result.max_latency.max(req_result.latency);

            if req_result.success {
                result.successful_requests += 1;
            } else {
                result.failed_requests += 1;
                if let Some(error) = req_result.error {
                    *result.errors.entry(error).or_insert(0) += 1;
                }
            }

            // Schedule next request
            next_request += interval;
        }

        result.finalize(start.elapsed());
        result
    }
}

/// Statistics collector for ongoing load tests.
#[derive(Debug)]
pub struct LoadTestStats {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    start_time: Instant,
}

impl Default for LoadTestStats {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadTestStats {
    /// Create a new stats collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current total requests.
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get current successful requests.
    #[must_use]
    pub fn successful_requests(&self) -> u64 {
        self.successful_requests.load(Ordering::Relaxed)
    }

    /// Get current failed requests.
    #[must_use]
    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }

    /// Get current requests per second.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn requests_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_requests() as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get elapsed time since start.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Builder for creating load tests with a fluent API.
pub struct LoadTestBuilder<G, E> {
    config: LoadTestConfig,
    generator: Option<G>,
    executor: Option<E>,
}

impl<G: QueryGenerator, E: QueryExecutor> Default for LoadTestBuilder<G, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: QueryGenerator, E: QueryExecutor> LoadTestBuilder<G, E> {
    /// Create a new load test builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: LoadTestConfig::default(),
            generator: None,
            executor: None,
        }
    }

    /// Set the load test configuration.
    #[must_use]
    pub fn with_config(mut self, config: LoadTestConfig) -> Self {
        self.config = config;
        self
    }

    /// Set concurrent users.
    #[must_use]
    pub fn with_concurrent_users(mut self, users: usize) -> Self {
        self.config.concurrent_users = users;
        self
    }

    /// Set total requests.
    #[must_use]
    pub fn with_total_requests(mut self, requests: usize) -> Self {
        self.config.total_requests = requests;
        self
    }

    /// Set test duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.config.duration = duration;
        self
    }

    /// Set the query generator.
    #[must_use]
    pub fn with_generator(mut self, generator: G) -> Self {
        self.generator = Some(generator);
        self
    }

    /// Set the query executor.
    #[must_use]
    pub fn with_executor(mut self, executor: E) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Build the load test.
    ///
    /// # Panics
    ///
    /// Panics if generator or executor is not set.
    #[must_use]
    pub fn build(self) -> LoadTest<G, E> {
        LoadTest::new(
            self.config,
            self.generator.expect("Generator must be set"),
            self.executor.expect("Executor must be set"),
        )
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_load_test_config_default() {
        let config = LoadTestConfig::default();
        assert_eq!(config.concurrent_users, 1);
        assert_eq!(config.total_requests, 100);
        assert_eq!(config.duration, Duration::from_secs(60));
    }

    #[test]
    fn test_load_test_config_builder() {
        let config = LoadTestConfig::new()
            .with_concurrent_users(10)
            .with_total_requests(1000)
            .with_duration(Duration::from_secs(120))
            .with_ramp_up_time(Duration::from_secs(10))
            .with_ramp_down_time(Duration::from_secs(5))
            .with_think_time_ms(100);

        assert_eq!(config.concurrent_users, 10);
        assert_eq!(config.total_requests, 1000);
        assert_eq!(config.duration, Duration::from_secs(120));
        assert_eq!(config.ramp_up_time, Duration::from_secs(10));
        assert_eq!(config.ramp_down_time, Duration::from_secs(5));
        assert_eq!(config.think_time_ms, 100);
    }

    #[test]
    fn test_load_test_result_default() {
        let result = LoadTestResult::default();
        assert_eq!(result.total_requests, 0);
        assert_eq!(result.successful_requests, 0);
        assert_eq!(result.failed_requests, 0);
        assert_eq!(result.min_latency, Duration::MAX);
        assert_eq!(result.max_latency, Duration::ZERO);
    }

    #[test]
    fn test_load_test_result_merge() {
        let mut result1 = LoadTestResult::new();
        result1.total_requests = 10;
        result1.successful_requests = 8;
        result1.failed_requests = 2;
        result1.min_latency = Duration::from_millis(5);
        result1.max_latency = Duration::from_millis(50);
        result1.latencies = vec![Duration::from_millis(10), Duration::from_millis(20)];
        result1.errors.insert("timeout".to_string(), 1);

        let mut result2 = LoadTestResult::new();
        result2.total_requests = 5;
        result2.successful_requests = 4;
        result2.failed_requests = 1;
        result2.min_latency = Duration::from_millis(3);
        result2.max_latency = Duration::from_millis(100);
        result2.latencies = vec![Duration::from_millis(15)];
        result2.errors.insert("timeout".to_string(), 2);

        result1.merge(&result2);

        assert_eq!(result1.total_requests, 15);
        assert_eq!(result1.successful_requests, 12);
        assert_eq!(result1.failed_requests, 3);
        assert_eq!(result1.min_latency, Duration::from_millis(3));
        assert_eq!(result1.max_latency, Duration::from_millis(100));
        assert_eq!(result1.latencies.len(), 3);
        assert_eq!(result1.errors.get("timeout"), Some(&3));
    }

    #[test]
    fn test_load_test_result_finalize() {
        let mut result = LoadTestResult::new();
        result.total_requests = 100;
        result.successful_requests = 95;
        result.failed_requests = 5;
        result.min_latency = Duration::from_millis(1);
        result.max_latency = Duration::from_millis(100);

        // Add 100 latencies
        for i in 1..=100 {
            result.latencies.push(Duration::from_millis(i));
        }

        result.finalize(Duration::from_secs(10));

        assert_eq!(result.test_duration, Duration::from_secs(10));
        assert_eq!(result.requests_per_second, 10.0);
        assert_eq!(result.p50_latency, Duration::from_millis(50));
        assert_eq!(result.p95_latency, Duration::from_millis(95));
        assert_eq!(result.p99_latency, Duration::from_millis(99));
    }

    #[test]
    fn test_request_result_success() {
        let result = RequestResult::success(Duration::from_millis(10));
        assert!(result.success);
        assert_eq!(result.latency, Duration::from_millis(10));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_request_result_failure() {
        let result =
            RequestResult::failure(Duration::from_millis(5), "Connection refused".to_string());
        assert!(!result.success);
        assert_eq!(result.latency, Duration::from_millis(5));
        assert_eq!(result.error, Some("Connection refused".to_string()));
    }

    #[test]
    fn test_mock_query_generator() {
        let generator = MockQueryGenerator::new();
        let query1 = generator.generate();
        let query2 = generator.generate();

        assert!(!query1.text.is_empty());
        assert!(!query2.text.is_empty());
        assert_ne!(query1.text, query2.text);
    }

    #[test]
    fn test_mock_query_generator_cycles() {
        let generator =
            MockQueryGenerator::with_queries(vec!["Query 1".to_string(), "Query 2".to_string()]);

        let q1 = generator.generate();
        let q2 = generator.generate();
        let q3 = generator.generate();

        assert_eq!(q1.text, "Query 1");
        assert_eq!(q2.text, "Query 2");
        assert_eq!(q3.text, "Query 1"); // Should cycle back
    }

    #[test]
    fn test_mock_query_executor_default() {
        let executor = MockQueryExecutor::new();
        assert_eq!(executor.latency, Duration::from_millis(10));
        assert_eq!(executor.failure_rate, 0.0);
        assert_eq!(executor.request_count(), 0);
    }

    #[test]
    fn test_mock_query_executor_builder() {
        let executor = MockQueryExecutor::new()
            .with_latency(Duration::from_millis(50))
            .with_failure_rate(0.1);

        assert_eq!(executor.latency, Duration::from_millis(50));
        assert_eq!(executor.failure_rate, 0.1);
    }

    #[test]
    fn test_mock_query_executor_clamps_failure_rate() {
        let executor = MockQueryExecutor::new().with_failure_rate(1.5);
        assert_eq!(executor.failure_rate, 1.0);

        let executor2 = MockQueryExecutor::new().with_failure_rate(-0.5);
        assert_eq!(executor2.failure_rate, 0.0);
    }

    #[test]
    fn test_load_test_stats() {
        let stats = LoadTestStats::new();

        assert_eq!(stats.total_requests(), 0);
        assert_eq!(stats.successful_requests(), 0);
        assert_eq!(stats.failed_requests(), 0);

        stats.record_success();
        stats.record_success();
        stats.record_failure();

        assert_eq!(stats.total_requests(), 3);
        assert_eq!(stats.successful_requests(), 2);
        assert_eq!(stats.failed_requests(), 1);
    }

    #[test]
    fn test_load_test_stats_rps() {
        let stats = LoadTestStats::new();

        // Record some requests
        for _ in 0..10 {
            stats.record_success();
        }

        // RPS should be positive
        let rps = stats.requests_per_second();
        assert!(rps > 0.0);
    }

    #[test]
    fn test_load_test_builder() {
        let generator = MockQueryGenerator::new();
        let executor = MockQueryExecutor::new();

        let load_test = LoadTestBuilder::new()
            .with_concurrent_users(5)
            .with_total_requests(50)
            .with_duration(Duration::from_secs(30))
            .with_generator(generator)
            .with_executor(executor)
            .build();

        assert_eq!(load_test.config().concurrent_users, 5);
        assert_eq!(load_test.config().total_requests, 50);
    }

    #[tokio::test]
    async fn test_load_test_run() {
        let config = LoadTestConfig::new()
            .with_concurrent_users(2)
            .with_total_requests(10);

        let generator = MockQueryGenerator::new();
        let executor = MockQueryExecutor::new().with_latency(Duration::from_millis(1));

        let load_test = LoadTest::new(config, generator, executor);
        let result = load_test.run().await;

        assert_eq!(result.total_requests, 10);
        assert_eq!(result.successful_requests, 10);
        assert_eq!(result.failed_requests, 0);
        assert!(result.requests_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_load_test_run_with_failures() {
        let config = LoadTestConfig::new()
            .with_concurrent_users(1)
            .with_total_requests(100);

        let generator = MockQueryGenerator::new();
        let executor = MockQueryExecutor::new()
            .with_latency(Duration::from_millis(1))
            .with_failure_rate(0.1);

        let load_test = LoadTest::new(config, generator, executor);
        let result = load_test.run().await;

        assert_eq!(result.total_requests, 100);
        assert!(result.failed_requests > 0);
        assert!(result.errors.contains_key("Simulated failure"));
    }

    #[tokio::test]
    async fn test_load_test_run_concurrent() {
        let config = LoadTestConfig::new().with_concurrent_users(5);

        let generator = MockQueryGenerator::new();
        let executor = MockQueryExecutor::new().with_latency(Duration::from_millis(1));

        let queries: Vec<Query> = (0..20).map(|i| Query::new(format!("Query {i}"))).collect();

        let load_test = LoadTest::new(config, generator, executor);
        let result = load_test.run_concurrent(queries).await;

        assert_eq!(result.total_requests, 20);
        assert!(result.requests_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_load_test_run_sustained() {
        let config = LoadTestConfig::new();

        let generator = MockQueryGenerator::new();
        let executor = MockQueryExecutor::new().with_latency(Duration::from_millis(1));

        let load_test = LoadTest::new(config, generator, executor);
        let result = load_test
            .run_sustained(Duration::from_millis(100), 50.0)
            .await;

        assert!(result.total_requests >= 3); // At least a few requests in 100ms at 50 QPS
        assert!(result.requests_per_second > 0.0);
    }

    #[test]
    fn test_load_test_result_percentiles_empty() {
        let result = LoadTestResult::new().calculate_percentiles();
        assert_eq!(result.p50_latency, Duration::ZERO);
        assert_eq!(result.p95_latency, Duration::ZERO);
        assert_eq!(result.p99_latency, Duration::ZERO);
    }

    #[test]
    fn test_load_test_result_percentiles() {
        let mut result = LoadTestResult::new();
        for i in 1..=100 {
            result.latencies.push(Duration::from_millis(i));
        }
        result.max_latency = Duration::from_millis(100);

        let result = result.calculate_percentiles();
        assert_eq!(result.p50_latency, Duration::from_millis(50));
        assert_eq!(result.p95_latency, Duration::from_millis(95));
        assert_eq!(result.p99_latency, Duration::from_millis(99));
    }

    #[test]
    fn test_config_serialization() {
        let config = LoadTestConfig::new()
            .with_concurrent_users(10)
            .with_total_requests(1000);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: LoadTestConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.concurrent_users, 10);
        assert_eq!(parsed.total_requests, 1000);
    }

    #[test]
    fn test_result_serialization() {
        let mut result = LoadTestResult::new();
        result.total_requests = 100;
        result.successful_requests = 95;
        result.errors.insert("timeout".to_string(), 5);

        let json = serde_json::to_string(&result).unwrap();
        let parsed: LoadTestResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_requests, 100);
        assert_eq!(parsed.successful_requests, 95);
        assert_eq!(parsed.errors.get("timeout"), Some(&5));
    }
}
