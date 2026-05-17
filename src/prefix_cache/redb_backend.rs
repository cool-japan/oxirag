//! Redb-based persistent storage backend for prefix cache.
//!
//! This module provides an ACID-compliant, embedded database backend for the
//! prefix cache system using the `redb` pure-Rust key-value store. Unlike the
//! file-based [`crate::prefix_cache::PersistentPrefixCache`], this backend
//! guarantees transactional consistency, automatic crash recovery, and concurrent
//! read access without manual compaction.
//!
//! # Feature Flag
//!
//! This module is only compiled when the `prefix-cache-redb` feature is enabled.

#![cfg(feature = "prefix-cache-redb")]

pub mod ops;
pub mod store;
pub mod types;
#[cfg(test)]
mod tests;

pub use store::RedbPrefixCache;
pub use types::{PersistedKVEntry, RedbPrefixCacheConfig};
