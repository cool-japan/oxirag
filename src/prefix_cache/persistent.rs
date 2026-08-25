//! Persistent storage backend for prefix cache.
//!
//! This module provides a file-based persistent cache that can serve as L2/L3 storage,
//! implementing disk-based persistence for KV cache entries with indexing, compaction,
//! and hybrid memory+disk caching strategies.

pub mod io;
pub mod store;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use store::{HybridPersistentCache, PersistentPrefixCache};
pub use types::{CacheIndex, CompactionStats, IndexEntry, PersistedEntry, PersistentCacheConfig};
