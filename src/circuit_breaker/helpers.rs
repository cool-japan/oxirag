//! Free helper functions `with_circuit_breaker` and `with_service_circuit_breaker`.

use std::future::Future;

use super::breaker::CircuitBreaker;
use super::registry::CircuitBreakerRegistry;
use super::types::CircuitBreakerOrOperationError;

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
