//! Pairwise tournament reranking (monoT5 / duoT5 inspired).
//!
//! Documents are compared head-to-head via a [`PairwiseComparer`]; win counts
//! are accumulated over one or more full round-robin passes; the top-`k`
//! results are returned as [`PairwiseHit`]s sorted by wins descending.
//!
//! The comparison logic is pluggable: implement [`PairwiseComparer`] to plug in
//! a real cross-encoder, or use the built-in [`MockPairwiseComparer`] which
//! scores by token-overlap heuristic for testing and prototyping.
//!
//! # Quick Start
//!
//! ```
//! use oxirag::pairwise_rerank::{MockPairwiseComparer, PairwiseConfig, PairwiseReranker};
//!
//! let config = PairwiseConfig::new().with_tournament_rounds(1);
//! let comparer = Box::new(MockPairwiseComparer::new(false));
//! let reranker = PairwiseReranker::new(config, comparer);
//!
//! let docs = vec![
//!     ("doc1".to_string(), "rust programming language systems".to_string()),
//!     ("doc2".to_string(), "banana smoothie recipe".to_string()),
//!     ("doc3".to_string(), "rust ownership memory safety".to_string()),
//! ];
//!
//! let results = reranker.rerank("rust memory systems", &docs, 2).unwrap();
//! assert_eq!(results.len(), 2);
//! assert!(results[0].wins >= results[1].wins);
//! ```

pub mod reranker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use reranker::PairwiseReranker;
pub use types::{
    MockPairwiseComparer, PairwiseComparer, PairwiseConfig, PairwiseError, PairwiseHit,
    PairwiseScoredPair,
};
