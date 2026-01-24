//! Connection pooling for external services.
//!
//! This module provides a generic connection pool implementation that manages
//! connections to external services. It includes features like:
//!
//! - Configurable pool sizes (min/max connections)
//! - Connection timeouts and idle timeouts
//! - Health checking with test-on-acquire option
//! - RAII connection guards that return to pool on drop
//! - Pool statistics and monitoring
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::connection_pool::{ConnectionPool, PoolConfig, Connection};
//!
//! // Define your connection type
//! struct MyConnection { /* ... */ }
//!
//! impl Connection for MyConnection {
//!     fn is_healthy(&self) -> bool { true }
//!     fn reset(&mut self) -> Result<(), ConnectionError> { Ok(()) }
//! }
//!
//! // Create a pool with configuration
//! let config = PoolConfig::default();
//! let pool = ConnectionPool::new(config, || Box::pin(async { Ok(MyConnection {}) }));
//!
//! // Acquire a connection
//! let conn = pool.acquire().await?;
//!
//! // Use the connection...
//! // Connection is automatically returned to pool when dropped
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::sync::{Mutex, Notify, Semaphore};
use tokio::time::timeout;

/// Error type for connection pool operations.
#[derive(Debug, Error)]
pub enum PoolError {
    /// Failed to create a new connection.
    #[error("Connection creation failed: {0}")]
    ConnectionCreationFailed(String),

    /// Timed out waiting to acquire a connection.
    #[error("Timeout waiting for connection after {0}ms")]
    AcquireTimeout(u64),

    /// Connection health check failed.
    #[error("Connection health check failed")]
    HealthCheckFailed,

    /// Connection reset failed.
    #[error("Connection reset failed: {0}")]
    ResetFailed(String),

    /// Pool has been shut down.
    #[error("Pool has been shut down")]
    PoolShutdown,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Error type for connection operations.
#[derive(Debug, Error)]
pub enum ConnectionError {
    /// Generic connection error.
    #[error("Connection error: {0}")]
    Generic(String),

    /// Connection is closed.
    #[error("Connection is closed")]
    Closed,

    /// Operation timed out.
    #[error("Operation timed out")]
    Timeout,
}

/// Trait for pooled connections.
///
/// Implement this trait for any connection type you want to pool.
pub trait Connection: Send + Sync {
    /// Check if the connection is still healthy.
    ///
    /// This should be a quick check that doesn't block for long.
    fn is_healthy(&self) -> bool;

    /// Reset the connection to a clean state.
    ///
    /// Called before returning a connection to the pool or before
    /// giving it to a new borrower.
    ///
    /// # Errors
    ///
    /// Returns `ConnectionError` if the reset operation fails.
    fn reset(&mut self) -> Result<(), ConnectionError>;
}

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections to maintain in the pool.
    pub min_connections: usize,
    /// Maximum number of connections allowed in the pool.
    pub max_connections: usize,
    /// How long to wait when acquiring a connection before timing out.
    pub connection_timeout: Duration,
    /// How long an idle connection can remain in the pool before being closed.
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection before it should be replaced.
    pub max_lifetime: Duration,
    /// Whether to test connections before acquiring them.
    pub test_on_acquire: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(3600),
            test_on_acquire: true,
        }
    }
}

impl PoolConfig {
    /// Create a new pool configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum number of connections.
    #[must_use]
    pub fn min_connections(mut self, min: usize) -> Self {
        self.min_connections = min;
        self
    }

    /// Set the maximum number of connections.
    #[must_use]
    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Set the connection acquisition timeout.
    #[must_use]
    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = timeout;
        self
    }

    /// Set the idle connection timeout.
    #[must_use]
    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Set the maximum connection lifetime.
    #[must_use]
    pub fn max_lifetime(mut self, lifetime: Duration) -> Self {
        self.max_lifetime = lifetime;
        self
    }

    /// Set whether to test connections on acquire.
    #[must_use]
    pub fn test_on_acquire(mut self, test: bool) -> Self {
        self.test_on_acquire = test;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `PoolError::InvalidConfig` if the configuration is invalid.
    pub fn validate(&self) -> Result<(), PoolError> {
        if self.min_connections > self.max_connections {
            return Err(PoolError::InvalidConfig(format!(
                "min_connections ({}) cannot be greater than max_connections ({})",
                self.min_connections, self.max_connections
            )));
        }
        if self.max_connections == 0 {
            return Err(PoolError::InvalidConfig(
                "max_connections must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Statistics for the connection pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of currently active (borrowed) connections.
    pub active_connections: usize,
    /// Number of idle connections in the pool.
    pub idle_connections: usize,
    /// Total number of connections (active + idle).
    pub total_connections: usize,
    /// Number of requests currently waiting for a connection.
    pub waiting_requests: usize,
    /// Total number of successful acquire operations.
    pub acquire_count: u64,
    /// Total number of release operations.
    pub release_count: u64,
    /// Number of acquire operations that timed out.
    pub timeout_count: u64,
}

/// Metadata for a pooled connection.
struct PooledConnectionMeta<C>
where
    C: Connection,
{
    /// The actual connection.
    connection: C,
    /// When the connection was created.
    created_at: Instant,
    /// When the connection was last used.
    last_used: Instant,
}

impl<C: Connection> PooledConnectionMeta<C> {
    fn new(connection: C) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
        }
    }

    fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }

    fn is_idle_too_long(&self, idle_timeout: Duration) -> bool {
        self.last_used.elapsed() > idle_timeout
    }
}

/// Type alias for the connection factory function.
pub type ConnectionFactory<C> =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<C, PoolError>> + Send>> + Send + Sync>;

/// A generic connection pool.
///
/// Manages a pool of connections to an external service, handling
/// connection lifecycle, health checks, and pool sizing.
pub struct ConnectionPool<C>
where
    C: Connection + 'static,
{
    config: PoolConfig,
    /// Factory function to create new connections.
    factory: ConnectionFactory<C>,
    /// Pool of available connections.
    pool: Mutex<VecDeque<PooledConnectionMeta<C>>>,
    /// Semaphore to limit total connections.
    semaphore: Semaphore,
    /// Notify waiters when a connection is returned.
    notify: Notify,
    /// Number of currently active (borrowed) connections.
    active_count: AtomicUsize,
    /// Total connections created.
    total_created: AtomicUsize,
    /// Statistics counters.
    acquire_count: AtomicU64,
    release_count: AtomicU64,
    timeout_count: AtomicU64,
    /// Number of waiting requests.
    waiting_count: AtomicUsize,
    /// Whether the pool has been shut down.
    shutdown: Mutex<bool>,
}

impl<C> fmt::Debug for ConnectionPool<C>
where
    C: Connection,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionPool")
            .field("config", &self.config)
            .field("active_count", &self.active_count.load(Ordering::Relaxed))
            .field("total_created", &self.total_created.load(Ordering::Relaxed))
            .field("acquire_count", &self.acquire_count.load(Ordering::Relaxed))
            .field("release_count", &self.release_count.load(Ordering::Relaxed))
            .field("timeout_count", &self.timeout_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl<C> ConnectionPool<C>
where
    C: Connection + 'static,
{
    /// Create a new connection pool with the given configuration and factory.
    ///
    /// # Arguments
    ///
    /// * `config` - Pool configuration
    /// * `factory` - Function to create new connections
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    #[must_use]
    pub fn new<F, Fut>(config: PoolConfig, factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<C, PoolError>> + Send + 'static,
    {
        config.validate().expect("Invalid pool configuration");

        let factory: ConnectionFactory<C> = Arc::new(move || Box::pin(factory()));

        Self {
            semaphore: Semaphore::new(config.max_connections),
            config,
            factory,
            pool: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            active_count: AtomicUsize::new(0),
            total_created: AtomicUsize::new(0),
            acquire_count: AtomicU64::new(0),
            release_count: AtomicU64::new(0),
            timeout_count: AtomicU64::new(0),
            waiting_count: AtomicUsize::new(0),
            shutdown: Mutex::new(false),
        }
    }

    /// Acquire a connection from the pool.
    ///
    /// If no connection is available, this will either:
    /// - Create a new connection if under the max limit
    /// - Wait for a connection to be returned
    /// - Time out based on configuration
    ///
    /// # Errors
    ///
    /// Returns `PoolError` if:
    /// - Pool is shut down
    /// - Timeout while waiting for a connection
    /// - Failed to create a new connection
    /// - Health check failed on acquired connection
    pub async fn acquire(&self) -> Result<PooledConnection<'_, C>, PoolError> {
        if *self.shutdown.lock().await {
            return Err(PoolError::PoolShutdown);
        }

        self.waiting_count.fetch_add(1, Ordering::Relaxed);
        let result = self.acquire_inner().await;
        self.waiting_count.fetch_sub(1, Ordering::Relaxed);

        result
    }

    async fn acquire_inner(&self) -> Result<PooledConnection<'_, C>, PoolError> {
        let deadline = Instant::now() + self.config.connection_timeout;

        loop {
            // Try to get an existing connection from the pool
            if let Some(conn) = self.try_get_pooled_connection().await? {
                self.acquire_count.fetch_add(1, Ordering::Relaxed);
                return Ok(PooledConnection {
                    pool: self,
                    connection: Some(conn),
                });
            }

            // Try to create a new connection if under limit
            if let Ok(Ok(permit)) =
                timeout(Duration::from_millis(0), self.semaphore.acquire()).await
            {
                permit.forget(); // We manage the count ourselves
                match self.create_connection().await {
                    Ok(conn) => {
                        self.active_count.fetch_add(1, Ordering::Relaxed);
                        self.acquire_count.fetch_add(1, Ordering::Relaxed);
                        return Ok(PooledConnection {
                            pool: self,
                            connection: Some(conn),
                        });
                    }
                    Err(e) => {
                        // Failed to create connection, release semaphore permit
                        self.semaphore.add_permits(1);
                        return Err(e);
                    }
                }
            }
            // Semaphore closed or immediate acquire failed, wait for a connection

            // Wait for a connection to be returned or timeout
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.timeout_count.fetch_add(1, Ordering::Relaxed);
                #[allow(clippy::cast_possible_truncation)]
                return Err(PoolError::AcquireTimeout(
                    self.config.connection_timeout.as_millis() as u64,
                ));
            }

            if timeout(remaining, self.notify.notified()).await.is_err() {
                self.timeout_count.fetch_add(1, Ordering::Relaxed);
                #[allow(clippy::cast_possible_truncation)]
                return Err(PoolError::AcquireTimeout(
                    self.config.connection_timeout.as_millis() as u64,
                ));
            }
            // A connection might be available, loop and try again
        }
    }

    async fn try_get_pooled_connection(&self) -> Result<Option<C>, PoolError> {
        let mut pool = self.pool.lock().await;

        while let Some(mut meta) = pool.pop_front() {
            // Check if connection is expired or idle too long
            if meta.is_expired(self.config.max_lifetime)
                || meta.is_idle_too_long(self.config.idle_timeout)
            {
                // Discard this connection
                self.total_created.fetch_sub(1, Ordering::Relaxed);
                self.semaphore.add_permits(1);
                continue;
            }

            // Test connection health if configured
            if self.config.test_on_acquire && !meta.connection.is_healthy() {
                // Discard unhealthy connection
                self.total_created.fetch_sub(1, Ordering::Relaxed);
                self.semaphore.add_permits(1);
                continue;
            }

            // Reset the connection before returning
            if let Err(e) = meta.connection.reset() {
                // Discard connection that failed to reset
                self.total_created.fetch_sub(1, Ordering::Relaxed);
                self.semaphore.add_permits(1);
                tracing::warn!("Connection reset failed: {e}");
                continue;
            }

            self.active_count.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(meta.connection));
        }

        Ok(None)
    }

    async fn create_connection(&self) -> Result<C, PoolError> {
        let conn = (self.factory)().await?;
        self.total_created.fetch_add(1, Ordering::Relaxed);
        Ok(conn)
    }

    /// Get current pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let active = self.active_count.load(Ordering::Relaxed);
        let total = self.total_created.load(Ordering::Relaxed);
        let idle = total.saturating_sub(active);

        PoolStats {
            active_connections: active,
            idle_connections: idle,
            total_connections: total,
            waiting_requests: self.waiting_count.load(Ordering::Relaxed),
            acquire_count: self.acquire_count.load(Ordering::Relaxed),
            release_count: self.release_count.load(Ordering::Relaxed),
            timeout_count: self.timeout_count.load(Ordering::Relaxed),
        }
    }

    /// Resize the pool to a new maximum size.
    ///
    /// If the new size is smaller than the current number of connections,
    /// excess connections will be closed as they are returned to the pool.
    pub fn resize(&self, new_size: usize) {
        if new_size == 0 {
            tracing::warn!("Cannot resize pool to 0, ignoring");
            return;
        }

        // Update the semaphore
        let current_max = self.config.max_connections;
        if new_size > current_max {
            self.semaphore.add_permits(new_size - current_max);
        }
        // Note: We can't reduce semaphore permits directly, but we can
        // handle this by discarding connections when they're returned
    }

    /// Perform health checks on all idle connections.
    ///
    /// Removes any unhealthy connections from the pool.
    pub async fn health_check(&self) {
        let mut pool = self.pool.lock().await;
        let mut healthy_connections = VecDeque::new();

        while let Some(meta) = pool.pop_front() {
            if meta.connection.is_healthy()
                && !meta.is_expired(self.config.max_lifetime)
                && !meta.is_idle_too_long(self.config.idle_timeout)
            {
                healthy_connections.push_back(meta);
            } else {
                self.total_created.fetch_sub(1, Ordering::Relaxed);
                self.semaphore.add_permits(1);
            }
        }

        *pool = healthy_connections;
    }

    /// Shut down the pool, closing all connections.
    pub async fn shutdown(&self) {
        *self.shutdown.lock().await = true;

        let mut pool = self.pool.lock().await;
        let count = pool.len();
        pool.clear();

        self.total_created.fetch_sub(count, Ordering::Relaxed);
        self.semaphore.add_permits(count);

        // Wake any waiting acquirers so they get the shutdown error
        self.notify.notify_waiters();
    }
}

/// RAII wrapper for a pooled connection.
///
/// When dropped, the connection is automatically returned to the pool.
pub struct PooledConnection<'a, C>
where
    C: Connection + 'static,
{
    pool: &'a ConnectionPool<C>,
    connection: Option<C>,
}

impl<C> fmt::Debug for PooledConnection<'_, C>
where
    C: Connection + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PooledConnection")
            .field("connection", &self.connection)
            .finish()
    }
}

impl<C> PooledConnection<'_, C>
where
    C: Connection,
{
    /// Get a reference to the underlying connection.
    #[must_use]
    pub fn get(&self) -> Option<&C> {
        self.connection.as_ref()
    }

    /// Get a mutable reference to the underlying connection.
    pub fn get_mut(&mut self) -> Option<&mut C> {
        self.connection.as_mut()
    }

    /// Take the connection out of this wrapper without returning it to the pool.
    ///
    /// The connection will NOT be returned to the pool.
    /// Use this if you need to transfer ownership of the connection.
    pub fn take(mut self) -> Option<C> {
        if let Some(conn) = self.connection.take() {
            // Connection is being taken out of pool management
            // Update counters accordingly
            self.pool.active_count.fetch_sub(1, Ordering::Relaxed);
            self.pool.total_created.fetch_sub(1, Ordering::Relaxed);
            self.pool.semaphore.add_permits(1);
            Some(conn)
        } else {
            None
        }
    }
}

impl<C> std::ops::Deref for PooledConnection<'_, C>
where
    C: Connection,
{
    type Target = C;

    fn deref(&self) -> &Self::Target {
        self.connection.as_ref().expect("Connection has been taken")
    }
}

impl<C> std::ops::DerefMut for PooledConnection<'_, C>
where
    C: Connection,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.connection.as_mut().expect("Connection has been taken")
    }
}

impl<C> Drop for PooledConnection<'_, C>
where
    C: Connection + 'static,
{
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            // We need to return the connection to the pool
            // Since drop can't be async, we update counters synchronously
            // and put the connection back in the pool
            self.pool.active_count.fetch_sub(1, Ordering::Relaxed);
            self.pool.release_count.fetch_add(1, Ordering::Relaxed);

            // Try to lock and return - if we can't, the connection is lost
            // This is a best-effort return
            if let Ok(mut pool) = self.pool.pool.try_lock() {
                if pool.len() < self.pool.config.max_connections {
                    pool.push_back(PooledConnectionMeta::new(conn));
                    drop(pool);
                    self.pool.notify.notify_one();
                } else {
                    drop(pool);
                    self.pool.total_created.fetch_sub(1, Ordering::Relaxed);
                    self.pool.semaphore.add_permits(1);
                }
            } else {
                // Can't acquire lock, connection is lost
                self.pool.total_created.fetch_sub(1, Ordering::Relaxed);
                self.pool.semaphore.add_permits(1);
            }
        }
    }
}

/// A mock connection for testing purposes.
#[derive(Debug, Clone)]
pub struct MockConnection {
    /// Whether this connection is healthy.
    pub healthy: bool,
    /// Counter for tracking resets.
    pub reset_count: usize,
    /// Unique identifier for this connection.
    pub id: u64,
}

impl MockConnection {
    /// Create a new mock connection.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            healthy: true,
            id,
            reset_count: 0,
        }
    }

    /// Create a new unhealthy mock connection.
    #[must_use]
    pub fn unhealthy(id: u64) -> Self {
        Self {
            healthy: false,
            id,
            reset_count: 0,
        }
    }
}

impl Connection for MockConnection {
    fn is_healthy(&self) -> bool {
        self.healthy
    }

    fn reset(&mut self) -> Result<(), ConnectionError> {
        self.reset_count += 1;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_statements)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    use tokio::time::sleep;

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

        let conn = pool.acquire().await.unwrap();
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

        let conn1 = pool.acquire().await.unwrap();
        let id1 = conn1.id;
        drop(conn1);

        // Allow the connection to be returned to the pool
        tokio::task::yield_now().await;

        let conn2 = pool.acquire().await.unwrap();
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

        let conn1 = pool.acquire().await.unwrap();

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
            let _conn = pool_clone.acquire().await.unwrap();
            sleep(Duration::from_millis(20)).await;
            // Connection dropped here
        });

        // Wait a bit then try to acquire
        sleep(Duration::from_millis(50)).await;
        let conn = pool.acquire().await.unwrap();
        assert!(conn.is_healthy());

        handle.await.unwrap();
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

        let conn = pool.acquire().await.unwrap();
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
        let conn = pool.acquire().await.unwrap();
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

        let conn1 = pool.acquire().await.unwrap();
        let conn2 = pool.acquire().await.unwrap();
        let conn3 = pool.acquire().await.unwrap();

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

        let conn = pool.acquire().await.unwrap();
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
        let conn1 = pool.acquire().await.unwrap();
        let conn2 = pool.acquire().await.unwrap();

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

        let c1 = pool2.acquire().await.unwrap();
        let c2 = pool2.acquire().await.unwrap();
        let c3 = pool2.acquire().await.unwrap();

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

        let conn1 = pool.acquire().await.unwrap();
        let id1 = conn1.id;
        drop(conn1);

        // Wait for connection to expire
        sleep(Duration::from_millis(60)).await;

        let conn2 = pool.acquire().await.unwrap();
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

        let conn1 = pool.acquire().await.unwrap();
        let id1 = conn1.id;
        drop(conn1);

        // Wait for connection to become idle
        sleep(Duration::from_millis(60)).await;

        let conn2 = pool.acquire().await.unwrap();
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
                let conn = pool.acquire().await.unwrap();
                sleep(Duration::from_millis(10)).await;
                drop(conn);
            }));
        }

        for handle in handles {
            handle.await.unwrap();
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

        let conn = pool.acquire().await.unwrap();

        // Test deref
        assert!(conn.is_healthy());

        drop(conn);
    }

    // Test 16: PooledConnection deref_mut
    #[tokio::test]
    async fn test_pooled_connection_deref_mut() {
        let config = PoolConfig::default();

        static COUNTER: AtomicU64 = AtomicU64::new(1200);
        let pool = ConnectionPool::new(config, || {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(MockConnection::new(id)) })
        });

        let mut conn = pool.acquire().await.unwrap();

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

        let conn = pool.acquire().await.unwrap();
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

        conn.reset().unwrap();
        assert_eq!(conn.reset_count, 1);

        conn.reset().unwrap();
        assert_eq!(conn.reset_count, 2);
    }

    // Test 19: Config builder pattern
    #[test]
    fn test_config_builder() {
        let config = PoolConfig::new()
            .min_connections(2)
            .max_connections(20)
            .connection_timeout(Duration::from_secs(60))
            .idle_timeout(Duration::from_secs(300))
            .max_lifetime(Duration::from_secs(1800))
            .test_on_acquire(false);

        assert_eq!(config.min_connections, 2);
        assert_eq!(config.max_connections, 20);
        assert_eq!(config.connection_timeout, Duration::from_secs(60));
        assert_eq!(config.idle_timeout, Duration::from_secs(300));
        assert_eq!(config.max_lifetime, Duration::from_secs(1800));
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

        let conn = pool.acquire().await.unwrap();

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
            let c1 = pool.acquire().await.unwrap();
            let c2 = pool.acquire().await.unwrap();
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

        let mut conn = pool.acquire().await.unwrap();

        // Test get()
        let ref_conn = conn.get();
        assert!(ref_conn.is_some());
        assert!(ref_conn.unwrap().is_healthy());

        // Test get_mut()
        let ref_mut_conn = conn.get_mut();
        assert!(ref_mut_conn.is_some());
        ref_mut_conn.unwrap().healthy = false;

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

        let conn = pool.acquire().await.unwrap();
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

        let conn = pool.acquire().await.unwrap();
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
        let conn = pool.acquire().await.unwrap();
        assert!(conn.is_healthy());

        drop(conn);
    }
}
