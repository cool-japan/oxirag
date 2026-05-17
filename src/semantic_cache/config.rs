//! Configuration for the semantic cache.

use serde::{Deserialize, Serialize};

/// Configuration for an [`InMemorySemanticCache`].
///
/// Controls the similarity threshold for cache hits, maximum capacity (LRU),
/// and optional time-to-live for entries.
///
/// [`InMemorySemanticCache`]: super::cache::InMemorySemanticCache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Cosine similarity threshold for a cache hit (0.0–1.0).
    ///
    /// Two queries whose embeddings have cosine similarity ≥ this value are
    /// considered equivalent for caching purposes. The default value of `0.92`
    /// is intentionally conservative to avoid false positives.
    pub similarity_threshold: f32,

    /// Maximum number of entries stored before the oldest entry is evicted (LRU).
    ///
    /// Defaults to `1000`.
    pub max_entries: usize,

    /// Optional time-to-live in seconds.
    ///
    /// When `Some(secs)`, entries older than `secs` seconds are considered
    /// expired and will not be returned as cache hits. Expired entries are
    /// evicted lazily on lookup. When `None` entries never expire.
    pub ttl_secs: Option<u64>,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            max_entries: 1000,
            ttl_secs: None,
        }
    }
}

impl SemanticCacheConfig {
    /// Set the cosine similarity threshold for cache hits.
    ///
    /// Values closer to `1.0` require near-identical queries; values closer to
    /// `0.0` allow very different queries to share a cached answer.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set the maximum number of entries (LRU capacity).
    #[must_use]
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set the time-to-live in seconds (`None` = never expire).
    #[must_use]
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = Some(ttl_secs);
        self
    }
}
