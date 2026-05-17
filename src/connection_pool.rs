//! Connection pooling for external services.

pub mod connection;
pub mod mock;
pub mod pool;
pub mod types;

pub use connection::{Connection, PooledConnection};
pub use mock::MockConnection;
pub use pool::{ConnectionFactory, ConnectionPool};
pub use types::{ConnectionError, PoolConfig, PoolError, PoolStats};

#[cfg(test)]
mod tests;
