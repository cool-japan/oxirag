//! Pre-query shard/collection selection via classical distributed-IR
//! resource selection (CORI / `GlOSS`-style).
//!
//! Given a federation of shards (or collections, or corpora — any
//! partitioned set of independently-queryable sources), a real distributed
//! retrieval system cannot afford to query every shard for every request:
//! network round-trips, per-shard compute, and rate limits all make
//! "query everything, merge afterwards" impractical at scale. Classical
//! distributed information retrieval solves this with **resource
//! selection**: each shard publishes a small, document-free summary of its
//! contents (a *resource description* — term document frequencies, average
//! document length, and the like), and a broker scores those summaries
//! *before* querying anything, to decide which handful of shards are worth
//! the cost of an actual query.
//!
//! This module implements the best-known such algorithm, **CORI** (Callan,
//! Lu & Croft, 1995, "Searching Distributed Collections with Inference
//! Networks"), which treats resource selection as an inference-network
//! belief computation: a term's belief in a shard combines a
//! document-frequency-and-size term `T` component with a
//! discriminating-power `I` component (the resource-selection analogue of
//! IDF, computed *across shards* rather than across documents). It belongs
//! to the same family as **`GlOSS`** (Gravano, García-Molina & Tomasic,
//! 1994, "The Effectiveness of `GlOSS` for the Text Database Discovery
//! Problem"), which this module does not implement but which motivates the
//! same "lightweight per-shard summary, no document access" approach.
//!
//! # How this differs from every other retrieval-fusion module here
//!
//! Several modules in this crate combine results from multiple
//! sources — but every one of them operates strictly *after* every source
//! has already been queried:
//!
//! | Module | Runs... | ...to do what |
//! |---|---|---|
//! | [`ensemble_retriever`](crate::ensemble_retriever) | after every sub-retriever has been queried | per-retriever min-max score normalization + weighted/RRF fusion of the *returned candidates* |
//! | [`rank_fusion`](crate::rank_fusion) | after every ranked list has been produced | cross-list score normalization (min-max, Z-score, sum-to-1) + one of seven rank/score fusion methods |
//! | [`collections`](crate::collections) | after every collection has been searched | RRF-based federated merge of per-collection hits into one ranked list |
//! | `shard_selection` (this module) | **before any shard is queried at all** | CORI belief scoring of published resource-description *summaries*, to decide which shards are worth querying in the first place |
//!
//! `shard_selection` never sees a query result, a document, an embedding, or
//! a similarity score, and it performs **no score calibration and no result
//! merging** — those remain the job of the three modules above. Its only
//! inputs are [`ShardDescriptor`] summaries and a query's terms; its only
//! output is a ranked list of shard ids with CORI belief scores. Once you
//! know *which* shards to query, you query them (with whatever retriever you
//! like) and hand the *results* to `ensemble_retriever`, `rank_fusion`, or
//! `collections` — this module's job is finished before that ever starts.
//!
//! # The CORI formula
//!
//! For a query term with document frequency `df` in a shard (`0` if the
//! term is absent), whose collection word count is `cw` (≈
//! `doc_count × avg_doc_length`), scored against a candidate set of
//! `num_shards` shards whose average collection word count is `avgcw`, and
//! where the term appears in `shard_frequency` of those `num_shards` shards:
//!
//! ```text
//! T      = df / (df + 50 + 150 * cw / avgcw)
//! I      = ln((num_shards + 0.5) / shard_frequency) / ln(num_shards + 1.0)
//! belief = 0.4 + 0.6 * T * I
//! ```
//!
//! `T` rewards shards with a high document frequency for the term *relative
//! to their size* (so a small shard with the same raw `df` as a large shard
//! scores higher — `df` packed into fewer documents is stronger evidence of
//! topical focus). `I` rewards terms that are rare *across shards* (present
//! in few of the candidates) over terms present in nearly every shard, since
//! a term everyone has is useless for telling shards apart. A shard's
//! overall score is the mean `belief` across every query-term occurrence.
//! `50`, `150`, `0.4`, and `0.6` are the classic CORI constants, exposed as
//! configurable via [`ShardSelectionConfig`].
//!
//! Because `T = 0` whenever a term is absent from a shard, that shard's
//! belief for the term collapses to exactly the floor (`0.4` by default) —
//! **the same deterministic rule** applies whether only one query term is
//! absent or every one of them is, including the degenerate case of a shard
//! that contains none of the query's terms at all: its score is exactly the
//! floor, never zero and never negative.
//!
//! # Determinism
//!
//! All scoring is pure floating-point arithmetic over the supplied
//! descriptors — no randomness anywhere. Ties in the final ranking are
//! broken by ascending `shard_id`, so identical inputs always produce an
//! identical, stably-ordered ranking.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "shard-selection")]
//! # {
//! use oxirag::shard_selection::{ShardDescriptor, ShardSelectionConfig, ShardSelector};
//!
//! // Two same-size shards; "shard-a" has much richer coverage of "rust".
//! let shard_a = ShardDescriptor::new("shard-a", 1_000, 250.0)
//!     .with_term("rust", 80, 220)
//!     .with_term("python", 5, 6);
//! let shard_b = ShardDescriptor::new("shard-b", 1_000, 250.0)
//!     .with_term("rust", 4, 4)
//!     .with_term("python", 90, 300);
//!
//! let selector = ShardSelector::new(ShardSelectionConfig::default());
//! let query_terms = vec!["rust".to_string()];
//! let ranked = selector
//!     .select(&[shard_a, shard_b], &query_terms, 10)
//!     .expect("non-empty shards and query terms");
//!
//! // "shard-a" has both higher document frequency and equal size, so it
//! // wins the CORI belief score — even though nothing has been queried yet.
//! assert_eq!(ranked[0].shard_id, "shard-a");
//! # }
//! ```

pub mod engine;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::{ShardSelectionEngine, ShardSelector};
pub use types::{
    ShardDescriptor, ShardSelectionConfig, ShardSelectionError, ShardSelectionResult,
    ShardSelectionScore, ShardTermStats,
};
