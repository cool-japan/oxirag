//! `Setwise` reranking (Zhuang et al. 2024, "A Setwise Approach for
//! Effective and Highly Efficient Zero-shot Ranking with Large Language
//! Models").
//!
//! `Setwise` reranking is built around a single primitive,
//! [`SetwiseComparison`]: given a query and a **set** of `k` candidates, rank
//! all `k` of them in *one* comparison call. That primitive is then dropped
//! into the sift-down/swap step of a classical comparison-based sort — a
//! [`SetwiseSortStrategy::Heapsort`] or a [`SetwiseSortStrategy::Bubblesort`]
//! — in place of the usual scalar "compare two items" step, so the number of
//! comparison *calls* needed to find the top-`m` candidates falls toward
//! `O(n log n)` (heapsort) or `O(top_m · n)` (bubblesort), each call doing
//! `k`-times the work of a pairwise call.
//!
//! # How this differs from its neighbours in this crate
//!
//! - **`pairwise_rerank`** runs a full round-robin *tournament*: every
//!   ordered pair `(i, j)` is compared exactly once per round, so the
//!   comparison count is a fixed `O(n²)` — `n · (n − 1) / 2` calls per round,
//!   *regardless* of how many results are actually wanted. There is no
//!   notion of a "set" of more than two candidates.
//! - **`listwise_rerank`** slides a *window* of candidates across the whole
//!   ranking and asks a judge for a full **permutation** of that window
//!   (the `RankGPT` strategy). The window is repositioned by a fixed `step`
//!   over a fixed number of sweeps; there is no sort-algorithm structure
//!   driving *where* the window goes next, and no accounting of how many
//!   judge calls were needed relative to a pairwise baseline.
//! - **`setwise_rerank`** (this module) uses the k-way set comparison as the
//!   literal comparison primitive *inside* a sort algorithm's own schedule:
//!   heapsort's sift-down decides which of a node's up-to-`(k − 1)` children
//!   to promote using **one** [`SetwiseComparison`] call instead of up to
//!   `k − 1` pairwise child comparisons; bubblesort's adjacent-swap step
//!   becomes an adjacent-*window* step that ranks `k` neighbours at once and
//!   bubbles the winner forward. [`SetwiseResult::comparison_count`] reports
//!   exactly how many such calls were made, so the efficiency gain over the
//!   `n · (n − 1) / 2` pairwise-tournament baseline is directly measurable —
//!   see [`SetwiseReranker::rerank`].
//!
//! Since there is no live LLM behind the scenes, [`SetwiseComparison`] calls
//! are answered deterministically by
//! [`SetwiseReranker::score`]: a pseudo-embedding cosine similarity (FNV-1a
//! lexical hashing, mirroring the scheme used by `diversity_rank` and
//! `retrieval_diversity`) blended with lexical Jaccard overlap, weighted by
//! [`SetwiseConfig::semantic_weight`] / [`SetwiseConfig::lexical_weight`].
//! The blend is fully deterministic: the same query, candidates, and config
//! always produce the same [`SetwiseResult`].
//!
//! # Example
//!
//! ```
//! use oxirag::setwise_rerank::{
//!     SetwiseCandidate, SetwiseConfig, SetwiseReranker, SetwiseSortStrategy,
//! };
//!
//! let candidates = vec![
//!     SetwiseCandidate::new("a", "rust systems programming language memory safety"),
//!     SetwiseCandidate::new("b", "banana smoothie recipe with yogurt"),
//!     SetwiseCandidate::new("c", "rust ownership borrow checker compiler"),
//!     SetwiseCandidate::new("d", "python data science pandas numpy"),
//!     SetwiseCandidate::new("e", "javascript frontend react framework"),
//! ];
//!
//! let heap_config = SetwiseConfig::new()
//!     .with_comparison_size(3)
//!     .with_strategy(SetwiseSortStrategy::Heapsort);
//! let heap_reranker = SetwiseReranker::new(heap_config);
//! let heap_result = heap_reranker
//!     .rerank("rust memory safety systems", &candidates, 2)
//!     .unwrap();
//!
//! let bubble_config = SetwiseConfig::new()
//!     .with_comparison_size(3)
//!     .with_strategy(SetwiseSortStrategy::Bubblesort);
//! let bubble_reranker = SetwiseReranker::new(bubble_config);
//! let bubble_result = bubble_reranker
//!     .rerank("rust memory safety systems", &candidates, 2)
//!     .unwrap();
//!
//! // Both schedules are built on the same primitive and agree on the top-m.
//! assert_eq!(heap_result.ranking.ids(), bubble_result.ranking.ids());
//!
//! // Both used far fewer comparison calls than a full pairwise tournament
//! // over the same 5 candidates would need (5 * 4 / 2 = 10 calls).
//! let pairwise_baseline = candidates.len() * (candidates.len() - 1) / 2;
//! assert!(heap_result.comparison_count < pairwise_baseline);
//! assert!(bubble_result.comparison_count < pairwise_baseline);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::SetwiseReranker;
pub use types::{
    SetwiseCandidate, SetwiseComparison, SetwiseConfig, SetwiseError, SetwiseRankEntry,
    SetwiseRanking, SetwiseResult, SetwiseSortStrategy,
};
