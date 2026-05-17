//! `MockConnection` — a public test helper for connection pool users.

use super::connection::Connection;
use super::types::ConnectionError;

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
