//! Circuit breaker pattern for resilient external service handling.

pub mod breaker;
pub mod helpers;
pub mod registry;
pub mod types;

pub use breaker::{CircuitBreaker, CircuitPermit};
pub use helpers::{with_circuit_breaker, with_service_circuit_breaker};
pub use registry::CircuitBreakerRegistry;
pub use types::{
    CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerOrOperationError, CircuitBreakerStats,
    CircuitState,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
