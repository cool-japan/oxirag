//! Core circuit breaker tests (state transitions, stats, reset).

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::io_other_error)]
pub(super) mod core {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::time::sleep;

    use crate::circuit_breaker::breaker::CircuitBreaker;
    use crate::circuit_breaker::registry::CircuitBreakerRegistry;
    use crate::circuit_breaker::types::{CircuitBreakerConfig, CircuitBreakerError, CircuitState};

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
        permit
            .expect("test operation should succeed")
            .success()
            .await;
    }

    // Test 3: Transitions to open after failure threshold
    #[tokio::test]
    async fn test_transition_to_open_after_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);

        for _ in 0..3 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

        sleep(Duration::from_millis(20)).await;
        assert_eq!(breaker.state().await, CircuitState::HalfOpen);

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

        sleep(Duration::from_millis(20)).await;

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

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
            permit.failure().await;
        }

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.success().await;

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        breaker.reset().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);

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

        for _ in 0..3 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
            permit.success().await;
        }

        for _ in 0..2 {
            let permit = breaker
                .allow_request()
                .await
                .expect("test operation should succeed");
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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

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

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

        sleep(Duration::from_millis(20)).await;
        let _ = breaker.state().await;

        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
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

        assert!(Arc::ptr_eq(&breaker1, &breaker2));
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

        let permit = breaker_a
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.success().await;

        let permit = breaker_b
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;

        let stats = registry.all_stats().await;
        assert_eq!(stats.len(), 2);
        assert_eq!(
            stats
                .get("service-a")
                .expect("test operation should succeed")
                .successful_requests,
            1
        );
        assert_eq!(
            stats
                .get("service-b")
                .expect("test operation should succeed")
                .failed_requests,
            1
        );
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
        let permit = breaker
            .allow_request()
            .await
            .expect("test operation should succeed");
        permit.failure().await;
        assert_eq!(breaker.state().await, CircuitState::Open);

        registry.reset_all().await;
        assert_eq!(breaker.state().await, CircuitState::Closed);
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
}
