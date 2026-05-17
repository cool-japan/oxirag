//! Helper and integration tests for circuit breaker (helpers, concurrent, full-cycle).

#[cfg(test)]
#[allow(clippy::io_other_error)]
pub(super) mod helpers {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use tokio::time::sleep;

    use crate::circuit_breaker::breaker::{CircuitBreaker, CircuitPermit};
    use crate::circuit_breaker::helpers::{with_circuit_breaker, with_service_circuit_breaker};
    use crate::circuit_breaker::registry::CircuitBreakerRegistry;
    use crate::circuit_breaker::types::{
        CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerOrOperationError, CircuitState,
    };

    // Test 19: with_circuit_breaker success
    #[tokio::test]
    async fn test_with_circuit_breaker_success() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_circuit_breaker(&breaker, async { Ok(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.expect("test operation should succeed"), 42);

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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
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
            failure_threshold: 1000,
            failure_rate_threshold: 1.0,
            ..Default::default()
        };
        let breaker = Arc::new(CircuitBreaker::new(config));
        let mut handles = Vec::new();

        for i in 0..100 {
            let breaker = Arc::clone(&breaker);
            handles.push(tokio::spawn(async move {
                let permit = breaker
                    .allow_request()
                    .await
                    .expect("test operation should succeed");
                if i % 2 == 0 {
                    permit.success().await;
                } else {
                    permit.failure().await;
                }
            }));
        }

        for handle in handles {
            handle.await.expect("test operation should succeed");
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
            failure_threshold: 100,
            failure_rate_threshold: 0.5,
            min_requests_for_rate: 4,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
            permit.success().await;
        }

        let mut failures_recorded = 0;
        for _ in 0..5 {
            match breaker.allow_request().await {
                Ok(permit) => {
                    permit.failure().await;
                    failures_recorded += 1;
                }
                Err(_) => {
                    break;
                }
            }
        }

        assert_eq!(breaker.state().await, CircuitState::Open);
        assert!(
            failures_recorded >= 2,
            "Expected at least 2 failures before circuit opened, got {failures_recorded}"
        );
    }

    // Test 26: with_service_circuit_breaker
    #[tokio::test]
    async fn test_with_service_circuit_breaker() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());

        let result: Result<i32, CircuitBreakerOrOperationError<std::io::Error>> =
            with_service_circuit_breaker(&registry, "test-service", async { Ok(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.expect("test operation should succeed"), 42);

        let stats = registry.all_stats().await;
        assert_eq!(
            stats
                .get("test-service")
                .expect("test operation should succeed")
                .successful_requests,
            1
        );
    }

    // Test 27: CircuitPermit elapsed time
    #[tokio::test]
    async fn test_permit_elapsed() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig::default());
        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");

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

        assert_eq!(breaker.state().await, CircuitState::Closed);

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
            permit.failure().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Open);

        sleep(Duration::from_millis(20)).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
            permit.success().await;
        }
        assert_eq!(breaker.state().await, CircuitState::Closed);

        let stats = breaker.stats_async().await;
        assert_eq!(stats.state_changes, 3);
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
                let permit = breaker
                    .allow_request()
                    .await
                    .expect("test operation should succeed");
                permit.success().await;
            }));
        }

        for handle in handles {
            handle.await.expect("test operation should succeed");
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

    // Suppress unused import warning for CircuitPermit
    fn _use_circuit_permit() {
        let _ = std::mem::size_of::<CircuitPermit<'_>>();
    }
}
