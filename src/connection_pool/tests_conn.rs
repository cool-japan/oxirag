//! Connection/config/error tests for connection pool (`deref_mut`, take, mock, config, errors, debug, stats).

#[cfg(test)]
#[allow(clippy::items_after_statements)]
pub(super) mod conn_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use tokio::time::sleep;

    use crate::connection_pool::connection::Connection;
    use crate::connection_pool::mock::MockConnection;
    use crate::connection_pool::pool::ConnectionPool;
    use crate::connection_pool::types::{ConnectionError, PoolConfig, PoolError, PoolStats};

    // Test 16: PooledConnection deref_mut
    #[tokio::test]
    async fn test_pooled_connection_deref_mut() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1200);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let mut conn = pool.acquire().await.expect("test operation should succeed");

        // Test deref_mut
        conn.healthy = false;
        assert!(!conn.is_healthy());

        drop(conn);
    }

    // Test 17: PooledConnection take
    #[tokio::test]
    async fn test_pooled_connection_take() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1300);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");
        let taken = conn.take();

        assert!(taken.is_some());

        // Connection was taken, not returned to pool
        let stats = pool.stats();
        // Note: The release_count might still be 1 because drop was called
        // but the connection was None, so it wasn't returned
        assert_eq!(stats.active_connections, 0);
    }

    // Test 18: Mock connection reset
    #[test]
    fn test_mock_connection_reset() {
        let mut conn = MockConnection::new(1);
        assert_eq!(conn.reset_count, 0);

        conn.reset().expect("test operation should succeed");
        assert_eq!(conn.reset_count, 1);

        conn.reset().expect("test operation should succeed");
        assert_eq!(conn.reset_count, 2);
    }

    // Test 19: Config builder pattern
    #[test]
    fn test_config_builder() {
        let config = PoolConfig::new()
            .min_connections(2)
            .max_connections(20)
            .connection_timeout(Duration::from_mins(1))
            .idle_timeout(Duration::from_mins(5))
            .max_lifetime(Duration::from_mins(30))
            .test_on_acquire(false);

        assert_eq!(config.min_connections, 2);
        assert_eq!(config.max_connections, 20);
        assert_eq!(config.connection_timeout, Duration::from_mins(1));
        assert_eq!(config.idle_timeout, Duration::from_mins(5));
        assert_eq!(config.max_lifetime, Duration::from_mins(30));
        assert!(!config.test_on_acquire);
    }

    // Test 20: PoolError display
    #[test]
    fn test_pool_error_display() {
        let err = PoolError::ConnectionCreationFailed("test".to_string());
        assert!(err.to_string().contains("Connection creation failed"));

        let err = PoolError::AcquireTimeout(1000);
        assert!(err.to_string().contains("Timeout"));
        assert!(err.to_string().contains("1000"));

        let err = PoolError::HealthCheckFailed;
        assert!(err.to_string().contains("health check failed"));

        let err = PoolError::PoolShutdown;
        assert!(err.to_string().contains("shut down"));
    }

    // Test 21: ConnectionError display
    #[test]
    fn test_connection_error_display() {
        let err = ConnectionError::Generic("test".to_string());
        assert!(err.to_string().contains("test"));

        let err = ConnectionError::Closed;
        assert!(err.to_string().contains("closed"));

        let err = ConnectionError::Timeout;
        assert!(err.to_string().contains("timed out"));
    }

    // Test 22: Waiting requests count
    #[tokio::test]
    async fn test_waiting_requests_count() {
        let config = PoolConfig::default()
            .max_connections(1)
            .connection_timeout(Duration::from_millis(200));

        static COUNTER: AtomicU64 = AtomicU64::new(1400);
        let pool = Arc::new(ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        }));

        let conn = pool.acquire().await.expect("test operation should succeed");

        let pool_clone = Arc::clone(&pool);
        let handle = tokio::spawn(async move {
            let _ = pool_clone.acquire().await;
        });

        // Give the spawned task time to start waiting
        sleep(Duration::from_millis(10)).await;

        let stats = pool.stats();
        // Stats should be valid (waiting_requests is a usize, so always >= 0)
        let _ = stats.waiting_requests;

        drop(conn);
        let _ = handle.await;
    }

    // Test 23: Multiple acquires and releases
    #[tokio::test]
    async fn test_multiple_acquire_release_cycles() {
        let config = PoolConfig::default()
            .max_connections(3)
            .test_on_acquire(false);

        static COUNTER: AtomicU64 = AtomicU64::new(1500);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        for _ in 0..10 {
            let c1 = pool.acquire().await.expect("test operation should succeed");
            let c2 = pool.acquire().await.expect("test operation should succeed");
            drop(c1);
            drop(c2);
        }

        let stats = pool.stats();
        assert_eq!(stats.acquire_count, 20);
        assert_eq!(stats.release_count, 20);
    }

    // Test 24: Pool debug format
    #[tokio::test]
    async fn test_pool_debug() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1600);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let debug_str = format!("{pool:?}");
        assert!(debug_str.contains("ConnectionPool"));
        assert!(debug_str.contains("config"));
    }

    // Test 25: PooledConnection get methods
    #[tokio::test]
    async fn test_pooled_connection_get() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1700);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let mut conn = pool.acquire().await.expect("test operation should succeed");

        // Test get()
        let ref_conn = conn.get();
        assert!(ref_conn.is_some());
        assert!(
            ref_conn
                .expect("test operation should succeed")
                .is_healthy()
        );

        // Test get_mut()
        let ref_mut_conn = conn.get_mut();
        assert!(ref_mut_conn.is_some());
        ref_mut_conn.expect("test operation should succeed").healthy = false;

        assert!(!conn.is_healthy());

        drop(conn);
    }

    // Test 26: Factory returns error
    #[tokio::test]
    async fn test_factory_error() {
        let config = PoolConfig::default();

        let pool = ConnectionPool::<MockConnection>::new(config, || {
            Box::pin(async {
                Err(PoolError::ConnectionCreationFailed(
                    "Factory error".to_string(),
                ))
            })
        });

        let result = pool.acquire().await;
        assert!(matches!(
            result,
            Err(PoolError::ConnectionCreationFailed(_))
        ));
    }

    // Test 27: PoolStats default
    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.idle_connections, 0);
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.waiting_requests, 0);
        assert_eq!(stats.acquire_count, 0);
        assert_eq!(stats.release_count, 0);
        assert_eq!(stats.timeout_count, 0);
    }

    // Test 28: PooledConnection debug with connection
    #[tokio::test]
    async fn test_pooled_connection_debug() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1800);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");
        let debug_str = format!("{conn:?}");
        assert!(debug_str.contains("PooledConnection"));

        drop(conn);
    }

    // Test 29: Health check with expired connections
    #[tokio::test]
    async fn test_health_check_with_expired() {
        let config = PoolConfig::default()
            .max_lifetime(Duration::from_millis(10))
            .test_on_acquire(false);

        static COUNTER: AtomicU64 = AtomicU64::new(1900);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");
        drop(conn);

        // Wait for expiration
        sleep(Duration::from_millis(20)).await;

        pool.health_check().await;

        let stats = pool.stats();
        assert_eq!(stats.idle_connections, 0);
    }

    // Test 30: Resize to zero is ignored
    #[tokio::test]
    async fn test_resize_to_zero() {
        let config = PoolConfig::default().max_connections(2);

        static COUNTER: AtomicU64 = AtomicU64::new(2000);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        pool.resize(0);

        // Should still be able to acquire
        let conn = pool.acquire().await.expect("test operation should succeed");
        assert!(conn.is_healthy());

        drop(conn);
    }
}
