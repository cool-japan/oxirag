//! Uncertainty-sampling active learning for retrieval-pool selection.
//!
//! Given a **pool** of already-scored, not-yet-verified candidate items —
//! each carrying a handful of candidate/label scores produced by some
//! upstream scorer, e.g. the top-2 or top-N relevance scores from one
//! retrieval pass — this module ranks the pool by how *informative* each
//! item would be to verify/label next, and selects the most valuable batch.
//! It implements classical pool-based uncertainty sampling (Lewis & Gale
//! 1994; see Settles' 2009 active-learning survey) over two measures:
//!
//! - [`UncertaintyMeasure::MarginSampling`] — the score gap between an
//!   item's top-1 and top-2 candidates. A *small* margin means the scorer
//!   could barely distinguish the best candidate from the runner-up, i.e.
//!   the item is *ambiguous* and thus informative to verify.
//! - [`UncertaintyMeasure::EntropySampling`] — the Shannon entropy of the
//!   item's whole normalized candidate-score distribution. *High* entropy
//!   means probability mass is spread broadly across many candidates rather
//!   than concentrated on one — again signalling ambiguity, but, unlike
//!   margin sampling, driven by the *entire* tail of the distribution, not
//!   just the top two entries.
//!
//! [`ActiveLearningSelector::select_batch`] computes a [`UncertaintySample`]
//! for every pool item under the chosen measure, stable-sorts the pool
//! descending by uncertainty, and returns the top `batch_size` as a
//! [`SelectionBatch`] — together with the full ranked list, for
//! transparency. An optional greedy diversity filter
//! ([`ActiveLearningConfig::diversity_enabled`]) skips a high-uncertainty
//! candidate whenever it is a near-duplicate (by cosine similarity of
//! [`PoolItem::features`]) of an item already chosen for *this* batch, so a
//! cluster of near-identical ambiguous items cannot monopolize the entire
//! labeling budget.
//!
//! # How this differs from `skr` and `dragin`
//!
//! Two other modules in this crate also condition retrieval on some notion
//! of model uncertainty, but at different granularities and for different
//! purposes than this one:
//!
//! - [`skr`](crate::skr) is a **per-query binary retrieve-or-not gate**: for
//!   a single incoming question, it decides `Skip` vs. `Retrieve` by a
//!   similarity-weighted vote over a memory of past self-knowledge
//!   exemplars. It never ranks a pool of candidates against one another —
//!   its output is one binary decision for one query, not an ordering over
//!   many items.
//! - [`dragin`](crate::dragin) fires retrieval **mid-generation**, at
//!   **token** granularity: the instant the generator is about to emit an
//!   uncertain *content token*, a query is formed from recent
//!   attention-salient tokens and retrieval is triggered on the spot. It
//!   never sees a discrete pool of candidate items either — it reacts to a
//!   live token stream, one trigger decision at a time.
//! - `active_learning_retrieval` (this module), by contrast, operates on a
//!   **pool of already-scored, discrete candidate items** — not a single
//!   query and not a token stream — and produces a **ranking** of that
//!   whole pool by informativeness, from which a batch is selected for
//!   labeling/verification. It answers "given everything we could verify
//!   right now, which items are most worth verifying *next*, to build a
//!   labeled set?" — a pool-level selection-and-prioritization problem that
//!   is orthogonal to both `skr`'s single-query gate and `dragin`'s
//!   token-level trigger.
//!
//! # Example
//!
//! ```
//! use oxirag::active_learning_retrieval::{
//!     ActiveLearningConfig, ActiveLearningSelector, PoolItem, UncertaintyMeasure,
//! };
//!
//! // Three already-scored candidates from a hypothetical retrieval pass.
//! // "ambiguous" has a near-tied top-2 (small margin => high uncertainty);
//! // "confident" has a clearly dominant top candidate (large margin => low
//! // uncertainty); "midpack" sits in between.
//! let pool = vec![
//!     PoolItem::new("ambiguous", vec![5.0, 4.9]),
//!     PoolItem::new("confident", vec![9.0, 1.0]),
//!     PoolItem::new("midpack", vec![6.0, 4.0]),
//! ];
//!
//! let selector = ActiveLearningSelector::new(ActiveLearningConfig::default())
//!     .expect("default config is valid");
//! let batch = selector
//!     .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
//!     .expect("scores are well-formed");
//!
//! // The two most ambiguous items are selected, most-ambiguous first.
//! assert_eq!(batch.selected_ids(), vec!["ambiguous", "midpack"]);
//! assert_eq!(batch.ranked.len(), 3);
//! assert!(batch.selected[0].uncertainty_score > batch.selected[1].uncertainty_score);
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::ActiveLearningSelector;
pub use types::{
    ActiveLearningConfig, ActiveLearningError, ActiveLearningResult, PoolItem, SelectionBatch,
    UncertaintyMeasure, UncertaintySample,
};
