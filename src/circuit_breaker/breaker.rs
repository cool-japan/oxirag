//! `CircuitBreaker` and `CircuitPermit` RAII guard.

use crate::time::Instant;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::sync::RwLock;

use super::types::{CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerStats, CircuitState};

/// Circuit breaker implementation.
///
/// Thread-safe circuit breaker that monitors failures and automatically
/// transitions between closed, open, and half-open states.
pub struct CircuitBreaker {
    pub(super) config: CircuitBreakerConfig,
    pub(super) state: RwLock<CircuitState>,
    pub(super) failure_count: AtomicUsize,
    pub(super) success_count: AtomicUsize,
    pub(super) half_open_requests: AtomicUsize,
    pub(super) last_failure_time: RwLock<Option<Instant>>,
    pub(super) last_state_change: RwLock<Instant>,
    // Stats tracking
    pub(super) total_requests: AtomicU64,
    pub(super) successful_requests: AtomicU64,
    pub(super) failed_requests: AtomicU64,
    pub(super) rejected_requests: AtomicU64,
    pub(super) state_changes: AtomicU64,
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
    pub(super) async fn check_state_transition(&self) {
        let state = *self.state.read().await;
        if state == CircuitState::Open
            && let Some(last_failure) = *self.last_failure_time.read().await
            && last_failure.elapsed() >= self.config.timeout_duration
        {
            self.transition_to(CircuitState::HalfOpen).await;
        }
    }

    /// Transition to a new state.
    pub(super) async fn transition_to(&self, new_state: CircuitState) {
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
    pub(super) breaker: &'a CircuitBreaker,
    pub(super) started: Instant,
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

/// Shared-ownership wrapper for `CircuitBreaker` used by the registry.
#[allow(dead_code)]
pub(super) type SharedBreaker = Arc<CircuitBreaker>;
