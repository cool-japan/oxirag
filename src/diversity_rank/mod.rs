//! Determinantal-point-process-inspired diverse subset selection.
//!
//! This module ranks retrieval results for **global diversity** using greedy
//! MAP inference for a determinantal point process (DPP). Where a basic top-`k`
//! cut keeps the highest-scoring items — often several near-duplicates — and
//! [`MmrReranker`](crate::advanced_retrieval::MmrReranker) penalises only the
//! single most-similar already-selected item, this ranker rewards items that
//! enlarge the *volume* spanned by the selected set.
//!
//! # Selection rule
//!
//! Items are chosen greedily. The first pick is the highest-quality candidate.
//! Each subsequent pick maximises a volume-style marginal gain that multiplies
//! quality² by a *product* of dissimilarities over the whole selected set `S`:
//!
//! ```text
//! gain_i = (quality_weight · quality_i)^2 · Π_{s ∈ S} (1 − sim(i, s))
//! ```
//!
//! Because the penalty is a product, redundancy with *any* selected item shrinks
//! the gain, and redundancy with *several* compounds it — yielding genuine
//! global diversity rather than MMR's nearest-neighbour penalty.
//!
//! # Honest approximation
//!
//! This is the standard greedy DPP heuristic, not exact (NP-hard) MAP inference,
//! and the `(1 − sim)` product is a tractable surrogate for the determinant of
//! the selected sub-kernel — exact in the orthogonal / identical limits but an
//! approximation in between. Embeddings are deterministic FNV-1a lexical
//! pseudo-embeddings, so "similarity" means lexical overlap.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "diversity-rank")]
//! # {
//! use oxirag::diversity_rank::{DiversityRanker, DiversityConfig};
//! use oxirag::types::{Document, SearchResult};
//!
//! let ranker = DiversityRanker::new(DiversityConfig::default().with_k(2));
//! let results = vec![
//!     SearchResult::new(Document::new("rust async tokio runtime").with_id("a"), 0.95, 0),
//!     SearchResult::new(Document::new("rust async tokio runtime").with_id("b"), 0.90, 1),
//!     SearchResult::new(Document::new("python pandas dataframe").with_id("c"), 0.50, 2),
//! ];
//!
//! // Despite "c" scoring lowest, diversity makes it the second pick over the
//! // near-duplicate "b".
//! let ranked = ranker.select(&results);
//! assert_eq!(ranked.len(), 2);
//! assert_eq!(ranked[0].document.id.as_str(), "a");
//! assert_eq!(ranked[1].document.id.as_str(), "c");
//! # }
//! ```

pub mod ranker;
#[cfg(test)]
mod tests;
pub mod types;

pub use ranker::{DiversityRanker, embed};
pub use types::{DiversityConfig, DiversityRankError, DiversitySelection, SimilarityKind};
