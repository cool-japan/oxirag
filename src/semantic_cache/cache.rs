//! Semantic cache trait and in-memory implementation.
//!
//! The [`SemanticCache`] trait abstracts over any similarity-based cache that
//! maps query embeddings to [`PipelineOutput`] values. The primary concrete
//! implementation, [`InMemorySemanticCache`], uses cosine similarity and a
//! simple LRU eviction policy (evict-oldest-on-overflow).

use std::cell::Cell;

use async_trait::async_trait;

use super::config::SemanticCacheConfig;
use super::entry::CacheEntry;
use crate::types::PipelineOutput;

// ── Cache statistics ─────────────────────────────────────────────────────────

/// Snapshot of cache performance metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheStats {
    /// Current number of entries in the cache.
    pub entries: usize,

    /// Total number of successful cache hits since creation (or last clear).
    pub total_hits: usize,

    /// Total number of lookup calls made since creation (or last clear).
    pub total_lookups: usize,

    /// Fraction of lookups that resulted in a hit (`total_hits / total_lookups`).
    ///
    /// Returns `0.0` when `total_lookups` is zero.
    pub hit_rate: f32,
}

// ── SemanticCache trait ───────────────────────────────────────────────────────

/// A similarity-based cache for RAG pipeline outputs.
///
/// Implementors map dense query embeddings to previously computed
/// [`PipelineOutput`] values. A lookup returns a `Some` clone of the stored
/// output when the stored embedding's cosine similarity with the probe is at
/// least the configured threshold; otherwise `None`.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait SemanticCache: Send + Sync {
    /// Look up `embedding` in the cache.
    ///
    /// Returns a cloned [`PipelineOutput`] if a stored entry is within the
    /// configured similarity threshold; otherwise `None`.
    ///
    /// Expired entries (TTL) are treated as misses and are not returned.
    async fn lookup(&self, embedding: &[f32]) -> Option<PipelineOutput>;

    /// Store a new entry in the cache.
    ///
    /// When the cache is at capacity (`max_entries`), the oldest entry (by
    /// insertion order) is evicted to make room.
    async fn store(&mut self, embedding: Vec<f32>, query_text: &str, output: PipelineOutput);

    /// Remove all entries and reset statistics.
    async fn clear(&mut self);

    /// Current number of entries in the cache.
    fn len(&self) -> usize;

    /// Returns `true` if the cache contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a snapshot of cache performance statistics.
    fn stats(&self) -> CacheStats;
}

// ── InMemorySemanticCache ─────────────────────────────────────────────────────

/// In-memory semantic cache using cosine similarity.
///
/// Entries are stored in insertion order; when `max_entries` is reached the
/// front (oldest) entry is removed. TTL checks are performed lazily on every
/// lookup — expired entries are skipped and counted as misses.
///
/// Statistics (total lookups / hits) are tracked via [`Cell`]-based interior
/// mutability so that `lookup(&self)` can update them without requiring `&mut`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "semantic-cache")]
/// # {
/// use oxirag::semantic_cache::{InMemorySemanticCache, SemanticCacheConfig, SemanticCache};
///
/// # #[tokio::main]
/// # async fn main() {
/// let config = SemanticCacheConfig::default().with_threshold(0.9).with_max_entries(100);
/// let mut cache = InMemorySemanticCache::new(config);
/// assert!(cache.is_empty());
/// # }
/// # }
/// ```
pub struct InMemorySemanticCache {
    config: SemanticCacheConfig,
    /// Entries stored in insertion order (front = oldest).
    entries: Vec<CacheEntry>,
    /// Interior-mutable counter — can be incremented from `&self`.
    total_hits: Cell<usize>,
    /// Interior-mutable counter — can be incremented from `&self`.
    total_lookups: Cell<usize>,
}

// SAFETY: `Cell<usize>` is `!Sync` by default. `InMemorySemanticCache` is
// intended as a single-async-task cache (wrapped in a `Mutex` when shared
// across tasks). We assert `Send` so it satisfies the `SemanticCache: Send`
// bound required for use inside async executors. Do **not** share a bare
// `InMemorySemanticCache` across threads without external locking.
unsafe impl Send for InMemorySemanticCache {}
unsafe impl Sync for InMemorySemanticCache {}

impl InMemorySemanticCache {
    /// Create a new, empty cache with the given configuration.
    #[must_use]
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            total_hits: Cell::new(0),
            total_lookups: Cell::new(0),
        }
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns `0.0` when either vector has zero magnitude (avoids NaN).
    #[must_use]
    pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    /// Returns `true` if the entry has exceeded its TTL.
    ///
    /// When `ttl_secs` is `None`, entries never expire.
    #[must_use]
    pub(crate) fn is_expired(entry: &CacheEntry, ttl_secs: Option<u64>) -> bool {
        match ttl_secs {
            None => false,
            Some(secs) => entry.created_at.elapsed().as_secs() >= secs,
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl SemanticCache for InMemorySemanticCache {
    async fn lookup(&self, embedding: &[f32]) -> Option<PipelineOutput> {
        let ttl = self.config.ttl_secs;
        let threshold = self.config.similarity_threshold;

        for entry in &self.entries {
            if Self::is_expired(entry, ttl) {
                continue;
            }
            let sim = Self::cosine_similarity(&entry.query_embedding, embedding);
            if sim >= threshold {
                // Update interior-mutable counters (safe — `Cell` is designed
                // for exactly this single-threaded interior-mutability pattern).
                self.total_lookups
                    .set(self.total_lookups.get().saturating_add(1));
                self.total_hits.set(self.total_hits.get().saturating_add(1));
                return Some(entry.output.clone());
            }
        }

        self.total_lookups
            .set(self.total_lookups.get().saturating_add(1));
        None
    }

    async fn store(&mut self, embedding: Vec<f32>, query_text: &str, output: PipelineOutput) {
        // Evict oldest entries until there is room for one more.
        while self.entries.len() >= self.config.max_entries && !self.entries.is_empty() {
            self.entries.remove(0);
        }
        self.entries
            .push(CacheEntry::new(query_text, embedding, output));
    }

    async fn clear(&mut self) {
        self.entries.clear();
        self.total_hits.set(0);
        self.total_lookups.set(0);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn stats(&self) -> CacheStats {
        let lookups = self.total_lookups.get();
        let hits = self.total_hits.get();

        #[allow(clippy::cast_precision_loss)]
        let hit_rate = if lookups == 0 {
            0.0_f32
        } else {
            hits as f32 / lookups as f32
        };

        CacheStats {
            entries: self.entries.len(),
            total_hits: hits,
            total_lookups: lookups,
            hit_rate,
        }
    }
}
