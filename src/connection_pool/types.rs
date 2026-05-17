//! Error types, config, and stats for the connection pool.

use std::time::Duration;

use thiserror::Error;

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
            idle_timeout: Duration::from_mins(10),
            max_lifetime: Duration::from_hours(1),
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
