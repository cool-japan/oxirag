//! Semantic deduplication of retrieved passages via locality-sensitive hashing.
//!
//! Detects and removes *near*-duplicate passages from a retrieved set — not just
//! byte-identical ones — using two complementary, fully deterministic LSH
//! schemes:
//!
//! | Method | Signal | Threshold | Strength |
//! |--------|--------|-----------|----------|
//! | [`DedupMethod::SimHash`] | `Hamming` distance of 64-bit fingerprints | `simhash_max_hamming` | Small token edits |
//! | [`DedupMethod::MinHash`] | `Jaccard` estimate over word shingles | `minhash_min_jaccard` | Set/phrase overlap |
//!
//! Detected duplicates are grouped with single-linkage (union-find) clustering,
//! and one representative is kept per cluster per [`KeepPolicy`]. Survivors keep
//! their relative input order and are re-ranked from `0`.
//!
//! This module is **distinct** from the cosine-based `RedundancyFilter` in
//! `context_compression`: deduplication here is driven entirely by `SimHash`
//! (`Hamming`) and `MinHash` (`Jaccard`-over-shingles), with no embeddings.
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "semantic-dedup")] {
//! use oxirag::prelude::*;
//!
//! let dedup = SemanticDeduplicator::new(SemanticDedupConfig::default());
//! let unique = dedup.deduplicate(&results);
//! # }
//! ```

pub mod deduplicator;
pub mod minhash;
pub mod simhash;
#[cfg(test)]
mod tests;
pub mod types;

pub use deduplicator::SemanticDeduplicator;
pub use types::{DedupMethod, KeepPolicy, SemanticDedupConfig, SemanticDedupError};
