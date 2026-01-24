//! Circuit breaker pattern implementation for resilient external service handling.
//!
//! The circuit breaker pattern prevents cascading failures when external services
//! are unavailable or degraded. It monitors failures and temporarily blocks requests
//! when a threshold is reached, allowing the service time to recover.
//!
//! # States
//!
//! - **Closed**: Normal operation. Requests are allowed through.
//! - **Open**: Service is failing. Requests are rejected immediately.
//! - **Half-Open**: Testing if the service has recovered. Limited requests are allowed.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, with_circuit_breaker};
//!
//! let config = CircuitBreakerConfig::default();
//! let breaker = CircuitBreaker::new(config);
//!
//! // Wrap an async operation with circuit breaker protection
//! let result = with_circuit_breaker(&breaker, async {
//!     // Your fallible operation here
//!     Ok::<_, std::io::Error>("success")
//! }).await;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CircuitState {
    /// Normal operation - requests are allowed through.
    #[default]
    Closed,
    /// Service is failing - requests are rejected immediately.
    Open,
    /// Testing if service recovered - limited requests are allowed.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: usize,
    /// Number of successes in half-open state required to close the circuit.
    pub success_threshold: usize,
    /// How long the circuit stays open before transitioning to half-open.
    pub timeout_duration: Duration,
    /// Maximum concurrent requests allowed in half-open state.
    pub half_open_max_requests: usize,
    /// Open the circuit if failure rate exceeds this threshold (0.0 to 1.0).
    pub failure_rate_threshold: f32,
    /// Minimum number of requests before checking failure rate.
    pub min_requests_for_rate: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_duration: Duration::from_secs(30),
            half_open_max_requests: 3,
            failure_rate_threshold: 0.5,
            min_requests_for_rate: 10,
        }
    }
}

/// Statistics for monitoring circuit breaker behavior.
#[derive(Debug, Clone, Default)]
pub struct CircuitBreakerStats {
    /// Total number of requests made.
    pub total_requests: u64,
    /// Number of successful requests.
    pub successful_requests: u64,
    /// Number of failed requests.
    pub failed_requests: u64,
    /// Number of requests rejected due to open circuit.
    pub rejected_requests: u64,
    /// Number of state changes.
    pub state_changes: u64,
    /// Current state of the circuit.
    pub current_state: CircuitState,
    /// Time spent in the current state (milliseconds).
    pub time_in_current_state_ms: u64,
}

/// Error returned when the circuit breaker rejects a request.
#[derive(Debug, Clone)]
pub struct CircuitBreakerError {
    /// Current state of the circuit.
    pub state: CircuitState,
    /// Suggested time to wait before retrying.
    pub retry_after: Option<Duration>,
    /// Human-readable error message.
    pub message: String,
}

impl fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CircuitBreakerError {}

/// Circuit breaker implementation.
///
/// Thread-safe circuit breaker that monitors failures and automatically
/// transitions between closed, open, and half-open states.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<CircuitState>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    half_open_requests: AtomicUsize,
    last_failure_time: RwLock<Option<Instant>>,
    last_state_change: RwLock<Instant>,
    // Stats tracking
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    rejected_requests: AtomicU64,
    state_changes: AtomicU64,
}

impl fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("config", &self.config)
            .field("failure_count", &self.failure_count.load(Ordering::Relaxed))
            .field("success_count", &self.success_count.load(Ordering::Relaxed))
            .field(
                "half_open_requests",
                &self.half_open_requests.load(Ordering::Relaxed),
            )
            .field(
                "total_requests",
                &self.total_requests.load(Ordering::Relaxed),
            )
            .field(
                "successful_requests",
                &self.successful_requests.load(Ordering::Relaxed),
            )
            .field(
                "failed_requests",
                &self.failed_requests.load(Ordering::Relaxed),
            )
            .field(
                "rejected_requests",
                &self.rejected_requests.load(Ordering::Relaxed),
            )
            .field("state_changes", &self.state_changes.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            half_open_requests: AtomicUsize::new(0),
            last_failure_time: RwLock::new(None),
            last_state_change: RwLock::new(Instant::now()),
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            state_changes: AtomicU64::new(0),
        }
    }

    /// Check if a request should be allowed and return a permit if so.
    ///
    /// # Errors
    ///
    /// Returns `CircuitBreakerError` if the circuit is open and rejecting requests.
    pub async fn allow_request(&self) -> Result<CircuitPermit<'_>, CircuitBreakerError> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.check_state_transition().await;

        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => Ok(CircuitPermit {
                breaker: self,
                started: Instant::now(),
            }),
            CircuitState::Open => {
                self.rejected_requests.fetch_add(1, Ordering::Relaxed);
                let last_failure = self.last_failure_time.read().await;
                let retry_after = last_failure.map(|t| {
                    let elapsed = t.elapsed();
                    self.config
                        .timeout_duration
                        .checked_sub(elapsed)
                        .unwrap_or(Duration::ZERO)
                });
                Err(CircuitBreakerError {
                    state,
                    retry_after,
                    message: format!(
                        "Circuit breaker is open. Retry after {:?}",
                        retry_after.unwrap_or(Duration::ZERO)
                    ),
                })
            }
            CircuitState::HalfOpen => {
                let current = self.half_open_requests.fetch_add(1, Ordering::SeqCst);
                if current >= self.config.half_open_max_requests {
                    self.half_open_requests.fetch_sub(1, Ordering::SeqCst);
                    self.rejected_requests.fetch_add(1, Ordering::Relaxed);
                    Err(CircuitBreakerError {
                        state,
                        retry_after: Some(Duration::from_millis(100)),
                        message: "Circuit breaker is half-open, max concurrent requests reached"
                            .to_string(),
                    })
                } else {
                    Ok(CircuitPermit {
                        breaker: self,
                        started: Instant::now(),
                    })
                }
            }
        }
    }

    /// Record a successful operation.
    pub async fn record_success(&self) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);

        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => {
                // Reset failure count on success in closed state
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                self.half_open_requests.fetch_sub(1, Ordering::SeqCst);
                if successes >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed).await;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but ignore
            }
        }
    }

    /// Record a failed operation.
    pub async fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        *self.last_failure_time.write().await = Some(Instant::now());

        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

                // Check threshold-based opening
                if failures >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open).await;
                    return;
                }

                // Check rate-based opening
                let total = self.total_requests.load(Ordering::Relaxed);
                let failed = self.failed_requests.load(Ordering::Relaxed);
                if total >= self.config.min_requests_for_rate as u64 {
                    #[allow(clippy::cast_precision_loss)]
                    let rate = failed as f32 / total as f32;
                    if rate >= self.config.failure_rate_threshold {
                        self.transition_to(CircuitState::Open).await;
                    }
                }
            }
            CircuitState::HalfOpen => {
                self.half_open_requests.fetch_sub(1, Ordering::SeqCst);
                // Any failure in half-open state reopens the circuit
                self.transition_to(CircuitState::Open).await;
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }
    }

    /// Get the current state of the circuit breaker.
    pub async fn state(&self) -> CircuitState {
        self.check_state_transition().await;
        *self.state.read().await
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            state_changes: self.state_changes.load(Ordering::Relaxed),
            current_state: CircuitState::default(), // Will be updated by caller if needed
            time_in_current_state_ms: 0,            // Will be updated by caller if needed
        }
    }

    /// Get current statistics including state information.
    ///
    /// This is an async version that includes accurate state information.
    pub async fn stats_async(&self) -> CircuitBreakerStats {
        let current_state = *self.state.read().await;
        let last_change = *self.last_state_change.read().await;
        CircuitBreakerStats {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            state_changes: self.state_changes.load(Ordering::Relaxed),
            current_state,
            #[allow(clippy::cast_possible_truncation)]
            time_in_current_state_ms: last_change.elapsed().as_millis() as u64,
        }
    }

    /// Force the circuit to a specific state (primarily for testing).
    pub async fn force_state(&self, new_state: CircuitState) {
        self.transition_to(new_state).await;
    }

    /// Reset the circuit breaker to its initial state.
    pub async fn reset(&self) {
        *self.state.write().await = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.half_open_requests.store(0, Ordering::Release);
        *self.last_failure_time.write().await = None;
        *self.last_state_change.write().await = Instant::now();
        self.total_requests.store(0, Ordering::Release);
        self.successful_requests.store(0, Ordering::Release);
        self.failed_requests.store(0, Ordering::Release);
        self.rejected_requests.store(0, Ordering::Release);
        // Don't reset state_changes - keep historical count
    }

    /// Check and perform any necessary state transitions based on time.
    async fn check_state_transition(&self) {
        let state = *self.state.read().await;
        if state == CircuitState::Open
            && let Some(last_failure) = *self.last_failure_time.read().await
            && last_failure.elapsed() >= self.config.timeout_duration
        {
            self.transition_to(CircuitState::HalfOpen).await;
        }
    }

    /// Transition to a new state.
    async fn transition_to(&self, new_state: CircuitState) {
        let mut state = self.state.write().await;
        if *state != new_state {
            *state = new_state;
            *self.last_state_change.write().await = Instant::now();
            self.state_changes.fetch_add(1, Ordering::Relaxed);

            // Reset counters based on new state
            match new_state {
                CircuitState::Closed => {
                    self.failure_count.store(0, Ordering::Release);
                    self.success_count.store(0, Ordering::Release);
                    self.half_open_requests.store(0, Ordering::Release);
                }
                CircuitState::Open | CircuitState::HalfOpen => {
                    self.success_count.store(0, Ordering::Release);
                    self.half_open_requests.store(0, Ordering::Release);
                }
            }
        }
    }
}

/// Permit to execute a request (RAII guard).
///
/// When dropped without calling `success()` or `failure()`, the request
/// is not counted. Use the methods to record the outcome.
#[derive(Debug)]
pub struct CircuitPermit<'a> {
    breaker: &'a CircuitBreaker,
    started: Instant,
}

impl CircuitPermit<'_> {
    /// Mark the request as successful.
    pub async fn success(self) {
        self.breaker.record_success().await;
    }

    /// Mark the request as failed.
    pub async fn failure(self) {
        self.breaker.record_failure().await;
    }

    /// Get the duration since the permit was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Registry for managing multiple circuit breakers.
///
/// Useful when you have multiple external services that each need their own
/// circuit breaker with potentially different configurations.
pub struct CircuitBreakerRegistry {
    breakers: RwLock<HashMap<String, Arc<CircuitBreaker>>>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    /// Create a new registry with the given default configuration.
    #[must_use]
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: RwLock::new(HashMap::new()),
            default_config,
        }
    }

    /// Get or create a circuit breaker for a service.
    pub async fn get_or_create(&self, service_name: &str) -> Arc<CircuitBreaker> {
        // Try read lock first
        {
            let breakers = self.breakers.read().await;
            if let Some(breaker) = breakers.get(service_name) {
                return Arc::clone(breaker);
            }
        }

        // Need write lock to create
        let mut breakers = self.breakers.write().await;
        // Double-check in case another task created it
        if let Some(breaker) = breakers.get(service_name) {
            return Arc::clone(breaker);
        }

        let breaker = Arc::new(CircuitBreaker::new(self.default_config.clone()));
        breakers.insert(service_name.to_string(), Arc::clone(&breaker));
        breaker
    }

    /// Get a circuit breaker for a service if it exists.
    pub async fn get(&self, service_name: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().await;
        breakers.get(service_name).cloned()
    }

    /// Get all circuit breaker stats.
    pub async fn all_stats(&self) -> HashMap<String, CircuitBreakerStats> {
        let breakers = self.breakers.read().await;
        let mut stats = HashMap::new();
        for (name, breaker) in breakers.iter() {
            stats.insert(name.clone(), breaker.stats_async().await);
        }
        stats
    }

    /// Reset all circuit breakers.
    pub async fn reset_all(&self) {
        let breakers = self.breakers.read().await;
        for breaker in breakers.values() {
            breaker.reset().await;
        }
    }

    /// Remove a circuit breaker from the registry.
    pub async fn remove(&self, service_name: &str) -> Option<Arc<CircuitBreaker>> {
        let mut breakers = self.breakers.write().await;
        breakers.remove(service_name)
    }

    /// Get the number of registered circuit breakers.
    pub async fn len(&self) -> usize {
        let breakers = self.breakers.read().await;
        breakers.len()
    }

    /// Check if the registry is empty.
    pub async fn is_empty(&self) -> bool {
        let breakers = self.breakers.read().await;
        breakers.is_empty()
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

/// Error type for operations wrapped with circuit breaker.
#[derive(Debug)]
pub enum CircuitBreakerOrOperationError<E> {
    /// Circuit breaker rejected the request.
    CircuitBreaker(CircuitBreakerError),
    /// The operation itself failed.
    Operation(E),
}

impl<E: fmt::Display> fmt::Display for CircuitBreakerOrOperationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CircuitBreaker(e) => write!(f, "Circuit breaker error: {e}"),
            Self::Operation(e) => write!(f, "Operation error: {e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CircuitBreakerOrOperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CircuitBreaker(e) => Some(e),
            Self::Operation(e) => Some(e),
        }
    }
}

impl<E> From<CircuitBreakerError> for CircuitBreakerOrOperationError<E> {
    fn from(err: CircuitBreakerError) -> Self {
        Self::CircuitBreaker(err)
    }
}

/// Wrap an async operation with circuit breaker protection.
///
/// This is the primary way to use the circuit breaker. It automatically
/// tracks successes and failures.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, with_circuit_breaker};
///
/// let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
///
/// let result = with_circuit_breaker(&breaker, async {
///     // Perform some fallible operation
///     external_service_call().await
/// }).await;
///
/// match result {
///     Ok(value) => println!("Success: {:?}", value),
///     Err(CircuitBreakerOrOperationError::CircuitBreaker(e)) => {
///         println!("Circuit open: {}", e);
///     }
///     Err(CircuitBreakerOrOperationError::Operation(e)) => {
///         println!("Operation failed: {}", e);
///     }
/// }
/// ```
///
/// # Errors
///
/// Returns `CircuitBreakerOrOperationError::CircuitBreaker` if the circuit is open,
/// or `CircuitBreakerOrOperationError::Operation` if the wrapped operation fails.
pub async fn with_circuit_breaker<F, T, E>(
    breaker: &CircuitBreaker,
    operation: F,
) -> Result<T, CircuitBreakerOrOperationError<E>>
where
    F: Future<Output = Result<T, E>>,
{
    let permit = breaker.allow_request().await?;

    match operation.await {
        Ok(result) => {
            permit.success().await;
            Ok(result)
        }
        Err(e) => {
            permit.failure().await;
            Err(CircuitBreakerOrOperationError::Operation(e))
        }
    }
}

/// Convenience function to wrap an operation with a circuit breaker from a registry.
///
/// # Errors
///
/// Returns `CircuitBreakerOrOperationError::CircuitBreaker` if the circuit is open,
/// or `CircuitBreakerOrOperationError::Operation` if the wrapped operation fails.
pub async fn with_service_circuit_breaker<F, T, E>(
    registry: &CircuitBreakerRegistry,
    service_name: &str,
    operation: F,
) -> Result<T, CircuitBreakerOrOperationError<E>>
where
    F: Future<Output = Result<T, E>>,
{
    let breaker = registry.get_or_create(service_name).await;
    with_circuit_breaker(&breaker, operation).await
}

#[cfg(test)]
#[allow(clippy::io_other_error)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use tokio::time::sleep;

    // Test 1: Initial state is closed
    #[tokio::test]
    async fn test_initial_state_is_closed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // Test 2: Requests allowed in closed state
    #[tokio::test]
    async fn test_requests_allowed_when_closed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        let permit = breaker.allow_request().await;
        assert!(permit.is_ok());
        permit.unwrap().success().await;
    }

    // Test 3: Transitions to open after failure threshold
    #[tokio::test]
    async fn test_transition_to_open_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Record 3 failures
        for _ in 0..3 {
            let permit = breaker.allow_request().await.unwrap();
            permit.failure().await;
        }

        assert_eq!(breaker.state().await, CircuitState::Open);
    }

    // Test 4: Requests rejected when open
    #[tokio::test]
    async fn test_requests_rejected_when_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        let result = breaker.allow_request().await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().state, CircuitState::Open);
    }

    // Test 5: Transitions to half-open after timeout
    #[tokio::test]
    async fn test_transition_to_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(50),
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        sleep(Duration::from_millis(60)).await;

        assert_eq!(breaker.state().await, CircuitState::HalfOpen);
    }

    // Test 6: Half-open closes on success
    #[tokio::test]
    async fn test_half_open_closes_on_success() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_duration: Duration::from_millis(10),
            half_open_max_requests: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        // Wait for half-open
        sleep(Duration::from_millis(20)).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        // Record successes
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.success().await;
        }

        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // Test 7: Half-open reopens on failure
    #[tokio::test]
    async fn test_half_open_reopens_on_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(10),
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        // Wait for half-open
        sleep(Duration::from_millis(20)).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        // Fail in half-open
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        assert_eq!(breaker.state().await, CircuitState::Open);
    }

    // Test 8: Half-open limits concurrent requests
    #[tokio::test]
    async fn test_half_open_max_requests() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(10),
            half_open_max_requests: 2,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        // Wait for half-open
        sleep(Duration::from_millis(20)).await;

        // Get permits up to max
        let permit1 = breaker.allow_request().await;
        let permit2 = breaker.allow_request().await;
        let permit3 = breaker.allow_request().await;

        assert!(permit1.is_ok());
        assert!(permit2.is_ok());
        assert!(permit3.is_err());
    }

    // Test 9: Success resets failure count in closed state
    #[tokio::test]
    async fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Record 2 failures
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.failure().await;
        }

        // Record a success
        let permit = breaker.allow_request().await.unwrap();
        permit.success().await;

        // Record 2 more failures - should not open
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.failure().await;
        }

        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // Test 10: Force state change
    #[tokio::test]
    async fn test_force_state() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        breaker.force_state(CircuitState::Open).await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        breaker.force_state(CircuitState::HalfOpen).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        breaker.force_state(CircuitState::Closed).await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // Test 11: Reset clears state
    #[tokio::test]
    async fn test_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        // Reset
        breaker.reset().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);

        // Should work again
        let permit = breaker.allow_request().await;
        assert!(permit.is_ok());
    }

    // Test 12: Statistics tracking
    #[tokio::test]
    async fn test_statistics_tracking() {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // 3 successes
        for _ in 0..3 {
            let permit = breaker.allow_request().await.unwrap();
            permit.success().await;
        }

        // 2 failures
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.failure().await;
        }

        let stats = breaker.stats_async().await;
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.successful_requests, 3);
        assert_eq!(stats.failed_requests, 2);
        assert_eq!(stats.rejected_requests, 0);
    }

    // Test 13: Rejected request tracking
    #[tokio::test]
    async fn test_rejected_request_tracking() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        // Try 3 rejected requests
        for _ in 0..3 {
            let _ = breaker.allow_request().await;
        }

        let stats = breaker.stats_async().await;
        assert_eq!(stats.rejected_requests, 3);
    }

    // Test 14: State change counting
    #[tokio::test]
    async fn test_state_change_counting() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(10),
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Closed -> Open
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        // Wait for Open -> HalfOpen
        sleep(Duration::from_millis(20)).await;
        let _ = breaker.state().await;

        // HalfOpen -> Closed
        let permit = breaker.allow_request().await.unwrap();
        permit.success().await;

        let stats = breaker.stats_async().await;
        assert_eq!(stats.state_changes, 3);
    }

    // Test 15: Registry get_or_create
    #[tokio::test]
    async fn test_registry_get_or_create() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        let breaker1 = registry.get_or_create("service-a").await;
        let breaker2 = registry.get_or_create("service-a").await;
        let breaker3 = registry.get_or_create("service-b").await;

        // Same service returns same breaker
        assert!(Arc::ptr_eq(&breaker1, &breaker2));
        // Different service returns different breaker
        assert!(!Arc::ptr_eq(&breaker1, &breaker3));
    }

    // Test 16: Registry get
    #[tokio::test]
    async fn test_registry_get() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        assert!(registry.get("nonexistent").await.is_none());

        let _ = registry.get_or_create("exists").await;
        assert!(registry.get("exists").await.is_some());
    }

    // Test 17: Registry all_stats
    #[tokio::test]
    async fn test_registry_all_stats() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        let breaker_a = registry.get_or_create("service-a").await;
        let breaker_b = registry.get_or_create("service-b").await;

        let permit = breaker_a.allow_request().await.unwrap();
        permit.success().await;

        let permit = breaker_b.allow_request().await.unwrap();
        permit.failure().await;

        let stats = registry.all_stats().await;
        assert_eq!(stats.len(), 2);
        assert_eq!(stats.get("service-a").unwrap().successful_requests, 1);
        assert_eq!(stats.get("service-b").unwrap().failed_requests, 1);
    }

    // Test 18: Registry reset_all
    #[tokio::test]
    async fn test_registry_reset_all() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let registry = CircuitBreakerRegistry::new(config);

        let breaker = registry.get_or_create("service").await;
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        registry.reset_all().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
    }

    // Test 19: with_circuit_breaker success
    #[tokio::test]
    async fn test_with_circuit_breaker_success() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_circuit_breaker(&breaker, async { Ok(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let stats = breaker.stats();
        assert_eq!(stats.successful_requests, 1);
    }

    // Test 20: with_circuit_breaker failure
    #[tokio::test]
    async fn test_with_circuit_breaker_failure() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_circuit_breaker(&breaker, async {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "test error"))
            })
            .await;

        assert!(matches!(
            result,
            Err(CircuitBreakerOrOperationError::Operation(_))
        ));

        let stats = breaker.stats();
        assert_eq!(stats.failed_requests, 1);
    }

    // Test 21: with_circuit_breaker rejected
    #[tokio::test]
    async fn test_with_circuit_breaker_rejected() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Open the circuit
        let permit = breaker.allow_request().await.unwrap();
        permit.failure().await;

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_circuit_breaker(&breaker, async { Ok(42) }).await;

        assert!(matches!(
            result,
            Err(CircuitBreakerOrOperationError::CircuitBreaker(_))
        ));
    }

    // Test 22: Concurrent access safety
    #[tokio::test]
    async fn test_concurrent_access() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1000,     // High enough to not trigger during test
            failure_rate_threshold: 1.0, // Disable rate-based opening
            ..Default::default()
        };
        let breaker = Arc::new(CircuitBreaker::new(config));
        let mut handles = Vec::new();

        for i in 0..100 {
            let breaker = Arc::clone(&breaker);
            handles.push(tokio::spawn(async move {
                let permit = breaker.allow_request().await.unwrap();
                if i % 2 == 0 {
                    permit.success().await;
                } else {
                    permit.failure().await;
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let stats = breaker.stats();
        assert_eq!(stats.total_requests, 100);
        assert_eq!(stats.successful_requests, 50);
        assert_eq!(stats.failed_requests, 50);
    }

    // Test 23: Failure rate threshold
    #[tokio::test]
    async fn test_failure_rate_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 100, // High enough to not trigger
            failure_rate_threshold: 0.5,
            min_requests_for_rate: 4,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // 2 successes
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.success().await;
        }

        // Keep recording failures until the circuit opens
        // Circuit should open when failure rate >= 50% and min_requests >= 4
        // After 2 successes + 2 failures: 4 total, 50% failure rate -> opens
        let mut failures_recorded = 0;
        for _ in 0..5 {
            match breaker.allow_request().await {
                Ok(permit) => {
                    permit.failure().await;
                    failures_recorded += 1;
                }
                Err(_) => {
                    // Circuit opened, which is expected
                    break;
                }
            }
        }

        assert_eq!(breaker.state().await, CircuitState::Open);
        // Should have opened after 2 failures (50% rate with 4 requests)
        assert!(
            failures_recorded >= 2,
            "Expected at least 2 failures before circuit opened, got {failures_recorded}"
        );
    }

    // Test 24: Error display
    #[tokio::test]
    async fn test_error_display() {
        let err = CircuitBreakerError {
            state: CircuitState::Open,
            retry_after: Some(Duration::from_secs(5)),
            message: "Test error".to_string(),
        };

        assert_eq!(err.to_string(), "Test error");
    }

    // Test 25: State display
    #[test]
    fn test_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "HalfOpen");
    }

    // Test 26: with_service_circuit_breaker
    #[tokio::test]
    async fn test_with_service_circuit_breaker() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_service_circuit_breaker(&registry, "test-service", async { Ok(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let stats = registry.all_stats().await;
        assert_eq!(stats.get("test-service").unwrap().successful_requests, 1);
    }

    // Test 27: CircuitPermit elapsed time
    #[tokio::test]
    async fn test_permit_elapsed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        let permit = breaker.allow_request().await.unwrap();

        sleep(Duration::from_millis(10)).await;

        assert!(permit.elapsed() >= Duration::from_millis(10));
        permit.success().await;
    }

    // Test 28: Registry len and is_empty
    #[tokio::test]
    async fn test_registry_len_and_is_empty() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        assert!(registry.is_empty().await);
        assert_eq!(registry.len().await, 0);

        let _ = registry.get_or_create("service-a").await;
        assert!(!registry.is_empty().await);
        assert_eq!(registry.len().await, 1);

        let _ = registry.get_or_create("service-b").await;
        assert_eq!(registry.len().await, 2);
    }

    // Test 29: Registry remove
    #[tokio::test]
    async fn test_registry_remove() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        let _ = registry.get_or_create("service").await;
        assert_eq!(registry.len().await, 1);

        let removed = registry.remove("service").await;
        assert!(removed.is_some());
        assert_eq!(registry.len().await, 0);

        let removed = registry.remove("nonexistent").await;
        assert!(removed.is_none());
    }

    // Test 30: Full state cycle
    #[tokio::test]
    async fn test_full_state_cycle() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout_duration: Duration::from_millis(10),
            half_open_max_requests: 5,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        // Closed
        assert_eq!(breaker.state().await, CircuitState::Closed);

        // Open after failures
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.failure().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Open);

        // HalfOpen after timeout
        sleep(Duration::from_millis(20)).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        // Back to Closed after successes
        for _ in 0..2 {
            let permit = breaker.allow_request().await.unwrap();
            permit.success().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Closed);

        // Verify we completed the full cycle
        let stats = breaker.stats_async().await;
        assert_eq!(stats.state_changes, 3); // Closed->Open, Open->HalfOpen, HalfOpen->Closed
    }

    // Test 31: Concurrent registry access
    #[tokio::test]
    async fn test_concurrent_registry_access() {
        let registry = Arc::new(CircuitBreakerRegistry::new(CircuitBreakerConfig::default()));
        let created = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        for i in 0..10 {
            let registry = Arc::clone(&registry);
            let created = Arc::clone(&created);
            handles.push(tokio::spawn(async move {
                let breaker = registry.get_or_create(&format!("service-{}", i % 3)).await;
                if !created.swap(true, Ordering::SeqCst) {
                    // First thread to create
                }
                let permit = breaker.allow_request().await.unwrap();
                permit.success().await;
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(registry.len().await, 3);
    }

    // Test 32: CircuitBreakerOrOperationError display
    #[test]
    fn test_circuit_breaker_or_operation_error_display() {
        let cb_err: CircuitBreakerOrOperationError<std::io::Error> =
            CircuitBreakerOrOperationError::CircuitBreaker(CircuitBreakerError {
                state: CircuitState::Open,
                retry_after: None,
                message: "test".to_string(),
            });
        assert!(cb_err.to_string().contains("Circuit breaker error"));

        let op_err: CircuitBreakerOrOperationError<std::io::Error> =
            CircuitBreakerOrOperationError::Operation(std::io::Error::new(
                std::io::ErrorKind::Other,
                "test",
            ));
        assert!(op_err.to_string().contains("Operation error"));
    }
}
