//! Listwise reranking via the `RankGPT` sliding-window strategy (Sun et al. 2023).
//!
//! Where a pointwise or pairwise cross-encoder scores candidates
//! independently, a *listwise* judge observes a whole
//! window of candidates at once and emits a full permutation — an ordering from
//! most to least relevant. The [`ListwiseReranker`] applies the classic `RankGPT`
//! sliding window: a window slides from the back of the ranking toward the
//! front, each window is permuted, and the permuted order is written back so
//! the strongest candidates bubble to the top across passes.
//!
//! ```
//! use oxirag::listwise_rerank::{ListwiseReranker, WindowConfig};
//! use oxirag::types::{Document, SearchResult};
//!
//! let results = vec![
//!     SearchResult::new(Document::new("the sky is mostly blue"), 0.5, 0),
//!     SearchResult::new(Document::new("bananas are a yellow fruit"), 0.4, 1),
//!     SearchResult::new(Document::new("quantum entanglement links particles"), 0.3, 2),
//! ];
//! let reranker = ListwiseReranker::new(WindowConfig::default());
//! let ranked = reranker.rerank("quantum entanglement particles", &results).unwrap();
//! assert_eq!(ranked[0].new_rank, 0);
//! ```
pub mod judge;
pub mod reranker;
#[cfg(test)]
mod tests;
pub mod types;

pub use judge::LexicalListwiseJudge;
pub use reranker::ListwiseReranker;
pub use types::{ListwiseError, ListwiseJudge, ListwiseResult, WindowConfig};
