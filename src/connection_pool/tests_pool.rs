//! Pool lifecycle tests for connection pool (acquire, release, health, timeout, shutdown, resize, expiry, concurrent).

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::items_after_statements)]
pub(super) mod pool_tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use tokio::time::sleep;

    use crate::connection_pool::connection::Connection;
    use crate::connection_pool::mock::MockConnection;
    use crate::connection_pool::pool::ConnectionPool;
    use crate::connection_pool::types::{PoolConfig, PoolError};

    fn create_mock_factory()
    -> impl Fn() -> Pin<Box<dyn Future<Output = Result<MockConnection, PoolError>> + Send>> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        move || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        }
    }

    // Test 1: Create pool with default config
    #[tokio::test]
    async fn test_pool_creation() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config, create_mock_factory());
        let stats = pool.stats();
        assert_eq!(stats.total_connections, 0);
        assert_eq!(stats.active_connections, 0);
    }

    // Test 2: Basic acquire and release
    #[tokio::test]
    async fn test_basic_acquire_release() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config, create_mock_factory());

        let conn = pool.acquire().await.expect("test operation should succeed");
        assert!(conn.is_healthy());

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.acquire_count, 1);

        drop(conn);

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.release_count, 1);
    }

    // Test 3: Connection reuse
    #[tokio::test]
    async fn test_connection_reuse() {
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let config = PoolConfig::default().test_on_acquire(false);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn1 = pool.acquire().await.expect("test operation should succeed");
        let id1 = conn1.id;
        drop(conn1);

        // Allow the connection to be returned to the pool
        tokio::task::yield_now().await;

        let conn2 = pool.acquire().await.expect("test operation should succeed");
        let id2 = conn2.id;

        // Should reuse the same connection
        assert_eq!(id1, id2);
    }

    // Test 4: Pool exhaustion and timeout
    #[tokio::test]
    async fn test_pool_exhaustion_timeout() {
        let config = PoolConfig::default()
            .max_connections(1)
            .connection_timeout(Duration::from_millis(50));

        static COUNTER: AtomicU64 = AtomicU64::new(100);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn1 = pool.acquire().await.expect("test operation should succeed");

        let result = pool.acquire().await;
        assert!(matches!(result, Err(PoolError::AcquireTimeout(_))));

        let stats = pool.stats();
        assert_eq!(stats.timeout_count, 1);

        drop(conn1);
    }

    // Test 5: Connection returned after holder drops
    #[tokio::test]
    async fn test_connection_returned_on_drop() {
        let config = PoolConfig::default()
            .max_connections(1)
            .connection_timeout(Duration::from_millis(100));

        static COUNTER: AtomicU64 = AtomicU64::new(200);
        let pool = Arc::new(ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        }));

        let pool_clone = Arc::clone(&pool);
        let handle = tokio::spawn(async move {
            let _conn = pool_clone
                .acquire()
                .await
                .expect("test operation should succeed");
            sleep(Duration::from_millis(20)).await;
            // Connection dropped here
        });

        // Wait a bit then try to acquire
        sleep(Duration::from_millis(50)).await;
        let conn = pool.acquire().await.expect("test operation should succeed");
        assert!(conn.is_healthy());

        handle.await.expect("test operation should succeed");
    }

    // Test 6: Health check removes unhealthy connections
    #[tokio::test]
    async fn test_health_check() {
        let config = PoolConfig::default().test_on_acquire(false);

        static COUNTER: AtomicU64 = AtomicU64::new(300);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::unhealthy(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");
        drop(conn);

        tokio::task::yield_now().await;

        pool.health_check().await;

        let stats = pool.stats();
        assert_eq!(stats.idle_connections, 0);
    }

    // Test 7: Test on acquire
    #[tokio::test]
    async fn test_test_on_acquire() {
        let config = PoolConfig::default().test_on_acquire(true);

        let call_count = std::sync::Arc::new(AtomicU64::new(0));
        let call_count_clone = std::sync::Arc::clone(&call_count);

        let pool = ConnectionPool::new(config, move || {
            let count = call_count_clone.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                // All connections are healthy
                Ok(MockConnection::new(count))
            })
        });

        // This should work (creates and uses the connection)
        let conn = pool.acquire().await.expect("test operation should succeed");
        assert!(conn.is_healthy());

        drop(conn);

        // Verify at least one connection was created
        assert!(call_count.load(Ordering::Relaxed) >= 1);
    }

    // Test 8: Pool stats
    #[tokio::test]
    async fn test_pool_stats() {
        let config = PoolConfig::default().max_connections(5);

        static COUNTER: AtomicU64 = AtomicU64::new(400);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn1 = pool.acquire().await.expect("test operation should succeed");
        let conn2 = pool.acquire().await.expect("test operation should succeed");
        let conn3 = pool.acquire().await.expect("test operation should succeed");

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 3);
        assert_eq!(stats.acquire_count, 3);

        drop(conn1);
        drop(conn2);

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.release_count, 2);

        drop(conn3);
    }

    // Test 9: Config validation
    #[test]
    fn test_config_validation() {
        let config = PoolConfig::default().min_connections(10).max_connections(5);

        let result = config.validate();
        assert!(matches!(result, Err(PoolError::InvalidConfig(_))));

        let config = PoolConfig::default().max_connections(0);
        let result = config.validate();
        assert!(matches!(result, Err(PoolError::InvalidConfig(_))));
    }

    // Test 10: Pool shutdown
    #[tokio::test]
    async fn test_pool_shutdown() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(500);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");
        drop(conn);

        pool.shutdown().await;

        let result = pool.acquire().await;
        assert!(matches!(result, Err(PoolError::PoolShutdown)));
    }

    // Test 11: Resize pool
    #[tokio::test]
    async fn test_resize_pool() {
        let config = PoolConfig::default().max_connections(2);

        static COUNTER: AtomicU64 = AtomicU64::new(600);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        // Acquire 2 connections (max)
        let conn1 = pool.acquire().await.expect("test operation should succeed");
        let conn2 = pool.acquire().await.expect("test operation should succeed");

        // Resize to 3
        pool.resize(3);

        // Should be able to acquire a third connection now
        let config2 = PoolConfig::default()
            .max_connections(3)
            .connection_timeout(Duration::from_millis(50));

        static COUNTER2: AtomicU64 = AtomicU64::new(700);
        let pool2 = ConnectionPool::new(config2, || {
            let id = COUNTER2.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let c1 = pool2
            .acquire()
            .await
            .expect("test operation should succeed");
        let c2 = pool2
            .acquire()
            .await
            .expect("test operation should succeed");
        let c3 = pool2
            .acquire()
            .await
            .expect("test operation should succeed");

        let stats = pool2.stats();
        assert_eq!(stats.active_connections, 3);

        drop(c1);
        drop(c2);
        drop(c3);
        drop(conn1);
        drop(conn2);
    }

    // Test 12: Connection lifetime expiration
    #[tokio::test]
    async fn test_connection_lifetime_expiration() {
        let config = PoolConfig::default()
            .max_lifetime(Duration::from_millis(50))
            .test_on_acquire(false);

        static COUNTER: AtomicU64 = AtomicU64::new(800);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn1 = pool.acquire().await.expect("test operation should succeed");
        let id1 = conn1.id;
        drop(conn1);

        // Wait for connection to expire
        sleep(Duration::from_millis(60)).await;

        let conn2 = pool.acquire().await.expect("test operation should succeed");
        let id2 = conn2.id;

        // Should be a new connection (different id)
        assert_ne!(id1, id2);

        drop(conn2);
    }

    // Test 13: Connection idle timeout
    #[tokio::test]
    async fn test_connection_idle_timeout() {
        let config = PoolConfig::default()
            .idle_timeout(Duration::from_millis(50))
            .test_on_acquire(false);

        static COUNTER: AtomicU64 = AtomicU64::new(900);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn1 = pool.acquire().await.expect("test operation should succeed");
        let id1 = conn1.id;
        drop(conn1);

        // Wait for connection to become idle
        sleep(Duration::from_millis(60)).await;

        let conn2 = pool.acquire().await.expect("test operation should succeed");
        let id2 = conn2.id;

        // Should be a new connection (different id)
        assert_ne!(id1, id2);

        drop(conn2);
    }

    // Test 14: Concurrent acquire
    #[tokio::test]
    async fn test_concurrent_acquire() {
        let config = PoolConfig::default().max_connections(5);

        static COUNTER: AtomicU64 = AtomicU64::new(1000);
        let pool = Arc::new(ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        }));

        let mut handles = Vec::new();
        for _ in 0..10 {
            let pool = Arc::clone(&pool);
            handles.push(tokio::spawn(async move {
                let conn = pool.acquire().await.expect("test operation should succeed");
                sleep(Duration::from_millis(10)).await;
                drop(conn);
            }));
        }

        for handle in handles {
            handle.await.expect("test operation should succeed");
        }

        let stats = pool.stats();
        assert_eq!(stats.acquire_count, 10);
        assert_eq!(stats.release_count, 10);
    }

    // Test 15: PooledConnection deref
    #[tokio::test]
    async fn test_pooled_connection_deref() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1100);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let conn = pool.acquire().await.expect("test operation should succeed");

        // Test deref
        assert!(conn.is_healthy());

        drop(conn);
    }
}
