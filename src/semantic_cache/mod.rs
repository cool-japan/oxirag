//! Semantic cache: similarity-based memoization of RAG pipeline outputs.
//!
//! The semantic cache stores `(query_embedding → PipelineOutput)` pairs.
//! On lookup it returns a cached output when the incoming query embedding's
//! cosine similarity with a stored entry meets or exceeds a configurable
//! threshold, avoiding redundant pipeline executions for near-duplicate queries.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "semantic-cache")]
//! # {
//! use oxirag::semantic_cache::{InMemorySemanticCache, SemanticCache, SemanticCacheConfig};
//!
//! # #[tokio::main]
//! # async fn main() {
//! let config = SemanticCacheConfig::default()
//!     .with_threshold(0.92)   // require ≥ 92 % cosine similarity for a hit
//!     .with_max_entries(500); // keep at most 500 entries (LRU eviction)
//!
//! let mut cache = InMemorySemanticCache::new(config);
//! assert!(cache.is_empty());
//! # }
//! # }
//! ```

pub mod cache;
pub mod config;
pub mod entry;

#[cfg(test)]
mod tests;

pub use cache::{CacheStats, InMemorySemanticCache, SemanticCache};
pub use config::SemanticCacheConfig;
pub use entry::CacheEntry;
