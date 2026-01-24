//! Retry logic with exponential backoff for `OxiRAG` operations.
//!
//! This module provides a flexible retry mechanism with exponential backoff
//! and optional jitter for handling transient failures in the pipeline.

use std::future::Future;
use std::time::Duration;

use crate::config::RetryConfig;
use crate::error::{EmbeddingError, JudgeError, OxiRagError, SpeculatorError};

/// A retry policy that implements exponential backoff with optional jitter.
#[derive(Debug, Clone, Default)]
pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    /// Create a new retry policy with the given configuration.
    #[must_use]
    pub const fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create a retry policy with no retries (fail immediately).
    #[must_use]
    pub fn no_retry() -> Self {
        Self {
            config: RetryConfig {
                max_retries: 0,
                ..Default::default()
            },
        }
    }

    /// Get the underlying configuration.
    #[must_use]
    pub const fn config(&self) -> &RetryConfig {
        &self.config
    }

    /// Calculate the delay for a given attempt number (0-indexed).
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_wrap
    )]
    #[allow(clippy::cast_possible_truncation)]
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        // Note: attempt as i32 is safe because powi uses i32 and we won't have billions of retries
        let base_delay = self.config.initial_delay_ms as f64
            * self.config.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.config.max_delay_ms as f64);

        let final_delay = if self.config.add_jitter {
            // Add jitter: random value between 0.5 and 1.5 of the delay
            let jitter_factor = 0.5 + simple_random() * 1.0;
            capped_delay * jitter_factor
        } else {
            capped_delay
        };

        // final_delay is always positive and capped, safe to truncate to u64
        Duration::from_millis(final_delay as u64)
    }

    /// Execute an async operation with retry logic.
    ///
    /// The operation will be retried up to `max_retries` times if it returns
    /// a recoverable error. The delay between retries follows exponential
    /// backoff with optional jitter.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The success type
    /// * `E` - The error type (must implement `Retryable`)
    /// * `F` - The operation factory function
    /// * `Fut` - The future type returned by the operation
    ///
    /// # Errors
    ///
    /// Returns the last error if all retry attempts fail.
    ///
    /// # Panics
    ///
    /// This function should not panic under normal conditions. The internal
    /// expect is a safeguard that should never be reached.
    pub async fn retry<T, E, F, Fut>(&self, mut operation: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: Retryable,
    {
        let mut last_error: Option<E> = None;

        for attempt in 0..=self.config.max_retries {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    if !e.is_retryable() || attempt == self.config.max_retries {
                        return Err(e);
                    }

                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = self.config.max_retries,
                        "Operation failed, retrying: {:?}",
                        e.error_message()
                    );

                    last_error = Some(e);

                    let delay = self.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }

        // This should not be reached, but return the last error if it somehow is
        Err(last_error.expect("retry loop should have returned"))
    }

    /// Execute an async operation with retry logic and a custom retryable check.
    ///
    /// # Errors
    ///
    /// Returns the last error if all retry attempts fail.
    ///
    /// # Panics
    ///
    /// This function should not panic under normal conditions. The internal
    /// expect is a safeguard that should never be reached.
    pub async fn retry_with_check<T, E, F, Fut, C>(
        &self,
        mut operation: F,
        is_retryable: C,
    ) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        C: Fn(&E) -> bool,
    {
        let mut last_error: Option<E> = None;

        for attempt in 0..=self.config.max_retries {
            match operation().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    if !is_retryable(&e) || attempt == self.config.max_retries {
                        return Err(e);
                    }

                    last_error = Some(e);

                    let delay = self.calculate_delay(attempt);
                    sleep(delay).await;
                }
            }
        }

        Err(last_error.expect("retry loop should have returned"))
    }
}

/// Trait for errors that can indicate whether they are retryable.
pub trait Retryable {
    /// Returns true if the error is transient and the operation should be retried.
    fn is_retryable(&self) -> bool;

    /// Returns a human-readable error message for logging.
    fn error_message(&self) -> String;
}

impl Retryable for OxiRagError {
    fn is_retryable(&self) -> bool {
        match self {
            OxiRagError::Embedding(e) => e.is_retryable(),
            OxiRagError::Speculator(e) => e.is_retryable(),
            OxiRagError::Judge(e) => e.is_retryable(),
            // Vector store errors are generally not retryable (data issues)
            OxiRagError::VectorStore(_) => false,
            // Pipeline errors might be retryable depending on context
            OxiRagError::Pipeline(_) => false,
            // Config errors are not retryable
            OxiRagError::Config(_) => false,
            // IO errors might be transient
            #[cfg(feature = "native")]
            OxiRagError::Io(e) => {
                matches!(
                    e.kind(),
                    std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::Interrupted
                )
            }
            #[cfg(not(feature = "native"))]
            OxiRagError::Io(_) => false,
            // Serialization errors are not retryable
            OxiRagError::Serialization(_) => false,
            #[cfg(feature = "graphrag")]
            OxiRagError::Graph(_) => false,
            #[cfg(feature = "distillation")]
            OxiRagError::Distillation(_) => false,
            #[cfg(feature = "prefix-cache")]
            OxiRagError::PrefixCache(_) => false,
            // Hidden state errors are not retryable
            #[cfg(feature = "hidden-states")]
            OxiRagError::HiddenState(_) => false,
            // Circuit breaker errors are not retryable (by design)
            OxiRagError::CircuitBreaker(_) => false,
            // Memory errors are generally not retryable
            OxiRagError::Memory(_) => false,
        }
    }

    fn error_message(&self) -> String {
        self.to_string()
    }
}

impl Retryable for EmbeddingError {
    fn is_retryable(&self) -> bool {
        match self {
            // These errors might be transient (file system, resource, or backend issues)
            EmbeddingError::ModelLoad(_)
            | EmbeddingError::Inference(_)
            | EmbeddingError::Backend(_) => true,
            // These are not retryable (data issues)
            EmbeddingError::Tokenization(_)
            | EmbeddingError::DimensionMismatch { .. }
            | EmbeddingError::EmptyInput => false,
        }
    }

    fn error_message(&self) -> String {
        self.to_string()
    }
}

impl Retryable for SpeculatorError {
    fn is_retryable(&self) -> bool {
        match self {
            // These errors might be transient (resource, model, or backend issues)
            SpeculatorError::ModelLoad(_)
            | SpeculatorError::Generation(_)
            | SpeculatorError::Backend(_) => true,
            // These are not retryable (data issues)
            SpeculatorError::Verification(_)
            | SpeculatorError::InvalidDraft(_)
            | SpeculatorError::ContextTooLong { .. } => false,
        }
    }

    fn error_message(&self) -> String {
        self.to_string()
    }
}

impl Retryable for JudgeError {
    fn is_retryable(&self) -> bool {
        match self {
            // These errors might be transient (timeout or solver issues)
            JudgeError::Timeout(_) | JudgeError::SolverError(_) => true,
            // These are not retryable (data issues)
            JudgeError::ExtractionFailed(_)
            | JudgeError::EncodingFailed(_)
            | JudgeError::InconsistentClaims(_)
            | JudgeError::UnsupportedClaim(_) => false,
        }
    }

    fn error_message(&self) -> String {
        self.to_string()
    }
}

/// Simple pseudo-random number generator for jitter.
/// Uses a basic LCG (Linear Congruential Generator) seeded from system time.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn simple_random() -> f64 {
    use std::cell::Cell;
    use std::time::SystemTime;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
        );
    }

    SEED.with(|seed| {
        let mut s = seed.get();
        // LCG parameters from Numerical Recipes
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        seed.set(s);
        // Convert to f64 in range [0, 1)
        (s >> 11) as f64 / (1u64 << 53) as f64
    })
}

/// Platform-agnostic sleep function.
#[cfg(feature = "native")]
async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await;
}

/// Platform-agnostic sleep function for WASM.
#[cfg(all(feature = "wasm", not(feature = "native")))]
async fn sleep(duration: Duration) {
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let window = web_sys::window().expect("no window");
        window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve,
                duration.as_millis() as i32,
            )
            .expect("setTimeout failed");
    });
    let _ = JsFuture::from(promise).await;
}

/// Fallback sleep for when neither native nor wasm features are enabled.
#[cfg(all(not(feature = "native"), not(feature = "wasm")))]
async fn sleep(_duration: Duration) {
    // No-op in non-async context - this should not happen in practice
    std::hint::spin_loop();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_delay_ms, 100);
        assert_eq!(config.max_delay_ms, 5000);
        assert!((config.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert!(config.add_jitter);
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new()
            .with_max_retries(5)
            .with_initial_delay_ms(200)
            .with_max_delay_ms(10000)
            .with_backoff_multiplier(3.0)
            .with_jitter(false);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay_ms, 200);
        assert_eq!(config.max_delay_ms, 10000);
        assert!((config.backoff_multiplier - 3.0).abs() < f64::EPSILON);
        assert!(!config.add_jitter);
    }

    #[test]
    fn test_retry_policy_no_retry() {
        let policy = RetryPolicy::no_retry();
        assert_eq!(policy.config().max_retries, 0);
    }

    #[test]
    fn test_calculate_delay_no_jitter() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_initial_delay_ms(100)
                .with_backoff_multiplier(2.0)
                .with_max_delay_ms(10000)
                .with_jitter(false),
        );

        let delay0 = policy.calculate_delay(0);
        let delay1 = policy.calculate_delay(1);
        let delay2 = policy.calculate_delay(2);

        assert_eq!(delay0.as_millis(), 100);
        assert_eq!(delay1.as_millis(), 200);
        assert_eq!(delay2.as_millis(), 400);
    }

    #[test]
    fn test_calculate_delay_capped() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_initial_delay_ms(1000)
                .with_backoff_multiplier(10.0)
                .with_max_delay_ms(5000)
                .with_jitter(false),
        );

        // Attempt 2: 1000 * 10^2 = 100000, should be capped to 5000
        let delay = policy.calculate_delay(2);
        assert_eq!(delay.as_millis(), 5000);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn test_calculate_delay_with_jitter() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_initial_delay_ms(100)
                .with_backoff_multiplier(2.0)
                .with_max_delay_ms(10000)
                .with_jitter(true),
        );

        // With jitter, delay should be between 50% and 150% of base
        let delay = policy.calculate_delay(0);
        // Safe: delay is bounded by max_delay_ms which fits in u64
        let ms = delay.as_millis() as u64;

        // Base is 100ms, with jitter factor 0.5-1.5, result is 50-150ms
        assert!(ms >= 50, "delay {ms} should be >= 50");
        assert!(ms <= 150, "delay {ms} should be <= 150");
    }

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let policy = RetryPolicy::new(RetryConfig::new().with_max_retries(3));
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let result: Result<i32, EmbeddingError> = policy
            .retry(|| {
                let count = Arc::clone(&count_clone);
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_max_retries(3)
                .with_initial_delay_ms(1)
                .with_jitter(false),
        );
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let result: Result<i32, EmbeddingError> = policy
            .retry(|| {
                let count = Arc::clone(&count_clone);
                async move {
                    let attempts = count.fetch_add(1, Ordering::SeqCst);
                    if attempts < 2 {
                        Err(EmbeddingError::Inference("transient error".to_string()))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_all_attempts_fail() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_max_retries(2)
                .with_initial_delay_ms(1)
                .with_jitter(false),
        );
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let result: Result<i32, EmbeddingError> = policy
            .retry(|| {
                let count = Arc::clone(&count_clone);
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(EmbeddingError::Inference("persistent error".to_string()))
                }
            })
            .await;

        assert!(result.is_err());
        // Initial attempt + 2 retries = 3 total attempts
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_non_retryable_error() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_max_retries(3)
                .with_initial_delay_ms(1)
                .with_jitter(false),
        );
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let result: Result<i32, EmbeddingError> = policy
            .retry(|| {
                let count = Arc::clone(&count_clone);
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    // Empty input is not retryable
                    Err(EmbeddingError::EmptyInput)
                }
            })
            .await;

        assert!(result.is_err());
        // Should fail immediately without retries
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_with_custom_check() {
        let policy = RetryPolicy::new(
            RetryConfig::new()
                .with_max_retries(3)
                .with_initial_delay_ms(1)
                .with_jitter(false),
        );
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        // Custom check: only retry if error contains "retry"
        let result: Result<i32, String> = policy
            .retry_with_check(
                || {
                    let count = Arc::clone(&count_clone);
                    async move {
                        let attempts = count.fetch_add(1, Ordering::SeqCst);
                        if attempts < 2 {
                            Err("retry this error".to_string())
                        } else {
                            Ok(42)
                        }
                    }
                },
                |e: &String| e.contains("retry"),
            )
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&attempt_count);

        let result: Result<i32, EmbeddingError> = policy
            .retry(|| {
                let count = Arc::clone(&count_clone);
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(EmbeddingError::Inference("error".to_string()))
                }
            })
            .await;

        assert!(result.is_err());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_embedding_error_retryable() {
        assert!(EmbeddingError::ModelLoad("test".to_string()).is_retryable());
        assert!(EmbeddingError::Inference("test".to_string()).is_retryable());
        assert!(EmbeddingError::Backend("test".to_string()).is_retryable());
        assert!(!EmbeddingError::EmptyInput.is_retryable());
        assert!(!EmbeddingError::Tokenization("test".to_string()).is_retryable());
        assert!(
            !EmbeddingError::DimensionMismatch {
                expected: 10,
                actual: 20
            }
            .is_retryable()
        );
    }

    #[test]
    fn test_speculator_error_retryable() {
        assert!(SpeculatorError::ModelLoad("test".to_string()).is_retryable());
        assert!(SpeculatorError::Generation("test".to_string()).is_retryable());
        assert!(SpeculatorError::Backend("test".to_string()).is_retryable());
        assert!(!SpeculatorError::Verification("test".to_string()).is_retryable());
        assert!(!SpeculatorError::InvalidDraft("test".to_string()).is_retryable());
        assert!(!SpeculatorError::ContextTooLong { length: 10, max: 5 }.is_retryable());
    }

    #[test]
    fn test_judge_error_retryable() {
        assert!(JudgeError::Timeout(1000).is_retryable());
        assert!(JudgeError::SolverError("test".to_string()).is_retryable());
        assert!(!JudgeError::ExtractionFailed("test".to_string()).is_retryable());
        assert!(!JudgeError::EncodingFailed("test".to_string()).is_retryable());
        assert!(!JudgeError::InconsistentClaims("test".to_string()).is_retryable());
        assert!(!JudgeError::UnsupportedClaim("test".to_string()).is_retryable());
    }

    #[test]
    fn test_oxirag_error_retryable() {
        assert!(
            OxiRagError::Embedding(EmbeddingError::Inference("test".to_string())).is_retryable()
        );
        assert!(
            OxiRagError::Speculator(SpeculatorError::Generation("test".to_string())).is_retryable()
        );
        assert!(OxiRagError::Judge(JudgeError::Timeout(1000)).is_retryable());
        assert!(!OxiRagError::Config("test".to_string()).is_retryable());
    }

    #[test]
    fn test_simple_random_range() {
        for _ in 0..100 {
            let r = simple_random();
            assert!((0.0..1.0).contains(&r), "random value {r} out of range");
        }
    }
}
