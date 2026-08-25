//! `ConnectionPool<C>` core implementation.

use crate::time::Instant;
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, Notify, Semaphore};
use tokio::time::timeout;

use super::connection::{Connection, PooledConnection, PooledConnectionMeta};
use super::types::{PoolConfig, PoolError, PoolStats};

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
    pub(super) config: PoolConfig,
    /// Factory function to create new connections.
    factory: ConnectionFactory<C>,
    /// Pool of available connections.
    pub(super) pool: Mutex<VecDeque<PooledConnectionMeta<C>>>,
    /// Semaphore to limit total connections.
    pub(super) semaphore: Semaphore,
    /// Notify waiters when a connection is returned.
    pub(super) notify: Notify,
    /// Number of currently active (borrowed) connections.
    pub(super) active_count: AtomicUsize,
    /// Total connections created.
    pub(super) total_created: AtomicUsize,
    /// Statistics counters.
    pub(super) acquire_count: AtomicU64,
    pub(super) release_count: AtomicU64,
    pub(super) timeout_count: AtomicU64,
    /// Number of waiting requests.
    pub(super) waiting_count: AtomicUsize,
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
