//! `Connection` trait and `PooledConnection` RAII guard.

use std::fmt;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::pool::ConnectionPool;
use super::types::ConnectionError;

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

/// Metadata for a pooled connection.
pub(super) struct PooledConnectionMeta<C>
where
    C: Connection,
{
    /// The actual connection.
    pub(super) connection: C,
    /// When the connection was created.
    pub(super) created_at: Instant,
    /// When the connection was last used.
    pub(super) last_used: Instant,
}

impl<C: Connection> PooledConnectionMeta<C> {
    pub(super) fn new(connection: C) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
        }
    }

    pub(super) fn is_expired(&self, max_lifetime: std::time::Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }

    pub(super) fn is_idle_too_long(&self, idle_timeout: std::time::Duration) -> bool {
        self.last_used.elapsed() > idle_timeout
    }
}

/// RAII wrapper for a pooled connection.
///
/// When dropped, the connection is automatically returned to the pool.
pub struct PooledConnection<'a, C>
where
    C: Connection + 'static,
{
    pub(super) pool: &'a ConnectionPool<C>,
    pub(super) connection: Option<C>,
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
