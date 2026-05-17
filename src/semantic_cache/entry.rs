//! A single entry in the semantic cache.

use std::time::Instant;

use crate::types::PipelineOutput;

/// A single entry stored in the [`InMemorySemanticCache`].
///
/// Holds the original query text, its embedding vector, the cached pipeline
/// output, the creation timestamp (for TTL checks), and hit-count statistics.
///
/// [`InMemorySemanticCache`]: super::cache::InMemorySemanticCache
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Original query text that produced this entry.
    pub query_text: String,

    /// Dense embedding vector for `query_text`, used for similarity matching.
    pub query_embedding: Vec<f32>,

    /// The full pipeline output stored for this query.
    pub output: PipelineOutput,

    /// Wall-clock time when the entry was created (used for TTL checks).
    pub created_at: Instant,

    /// Number of times this entry has been returned as a cache hit.
    pub hit_count: usize,
}

impl CacheEntry {
    /// Create a new cache entry.
    ///
    /// `hit_count` starts at `0` and `created_at` is set to [`Instant::now`].
    #[must_use]
    pub fn new(query_text: impl Into<String>, query_embedding: Vec<f32>, output: PipelineOutput) -> Self {
        Self {
            query_text: query_text.into(),
            query_embedding,
            output,
            created_at: Instant::now(),
            hit_count: 0,
        }
    }
}
