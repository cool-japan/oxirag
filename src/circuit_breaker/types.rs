//! Core state, config, stats and error types for the circuit breaker.

use std::fmt;
use std::time::Duration;

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
