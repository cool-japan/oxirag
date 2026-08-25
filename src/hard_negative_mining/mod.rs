//! `hard_negative_mining` — ANCE-style **asynchronous** hard-negative mining
//! (Xiong et al. 2020, "Approximate Nearest Neighbor Negative Contrastive
//! Learning for Dense Text Retrieval").
//!
//! Contrastive dense-retrieval training needs *hard* negatives: documents an
//! embedding model ranks highly for a query even though they are wrong. The
//! catch ANCE identifies is that hard negatives are defined *by the model being
//! trained*, which keeps changing — so negatives mined once quickly go stale.
//! ANCE's answer is to re-mine **asynchronously**, periodically re-ranking the
//! corpus under the latest model checkpoint. This module implements exactly
//! that loop: a [`HardNegativeMiner`] holds a fixed labelled set and corpus and
//! re-mines under successive [`EmbeddingVersion`]s, each
//! [`HardNegativeMiner::refresh`] reporting how far the mined negatives drifted
//! from the previous round.
//!
//! # How it differs from neighbouring modules
//!
//! This is deliberately distinct from two nearby modules that also touch
//! "negatives" or "training data":
//!
//! - **`synthetic_eval`** mines hard-negative *distractors* **one-shot** and
//!   **lexically** — a single pass over a corpus, matching documents that share
//!   salient terms with a question, in order to *build an offline evaluation
//!   set*. Its purpose is eval-set construction and its mechanism is a lexical
//!   one-shot. `hard_negative_mining`, by contrast, re-mines **asynchronously
//!   and periodically** under an **evolving embedding function**, as an
//!   index/training *quality* mechanism (a training/calibration signal), not to
//!   build a test set. The purpose (training/index quality vs eval-set
//!   construction) and the mechanism (dense re-ranking plus an explicit refresh
//!   cycle that measures drift vs a lexical single pass) are both different.
//! - **`distillation`** collects live-traffic question/answer pairs to distil a
//!   smaller model; it has **no concept of negatives at all**. This module is
//!   about the negatives an evolving retriever considers hard, and about
//!   measuring their staleness as the retriever drifts.
//!
//! # Algorithm
//!
//! Under a given [`EmbeddingVersion`] (a deterministic `FNV-1a`/`splitmix64`
//! pseudo-embedding parameterised so that different versions produce different
//! vectors for the same text — standing in for a re-trained model):
//!
//! 1. Embed the whole corpus and every labelled query.
//! 2. For each query, rank the full corpus by embedding similarity and record
//!    the labelled positive's rank (the retrieval-quality diagnostic).
//! 3. Select the top-ranked documents that are *not* the positive as
//!    [`HardNegativeSample`]s — hard because the current model ranks them
//!    highly despite being wrong — subject to a budget `n` and a top-K "hard"
//!    cutoff (documents ranked below the cutoff are *easy* negatives and are
//!    skipped).
//!
//! [`HardNegativeMiner::refresh`] re-runs this under a new version and computes
//! a [`HardNegativeStaleness`] report: the per-query Jaccard overlap between the
//! old and new mined negative sets. A version that barely perturbs the
//! embeddings keeps the sets nearly identical (high overlap, low staleness); a
//! version that scrambles them turns the sets over (low overlap, high
//! staleness).
//!
//! # Example
//!
//! ```
//! use oxirag::hard_negative_mining::{
//!     EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeMiner,
//!     HardNegativePositivePair,
//! };
//!
//! let corpus = vec![
//!     HardNegativeDocument::new("d_pos", "alpha beta gamma"),
//!     HardNegativeDocument::new("d_hard", "alpha beta gamma delta"),
//!     HardNegativeDocument::new("d_easy", "zulu yankee xray whiskey"),
//! ];
//! let positives = vec![HardNegativePositivePair::new("alpha beta gamma", "d_pos")];
//! let config = HardNegativeConfig::new()
//!     .with_negatives_per_query(1)
//!     .with_hard_rank_cutoff(2)
//!     .with_embedding_dim(256);
//!
//! let mut miner = HardNegativeMiner::new(config, positives, corpus).expect("valid inputs");
//!
//! // Round 0 under the canonical embedding model.
//! let round0 = miner.mine(EmbeddingVersion::canonical()).expect("mine");
//! let mined = round0.negative_ids("alpha beta gamma").expect("query present");
//! // The lexically-close "d_hard" is mined; the disjoint "d_easy" is not.
//! assert_eq!(mined.iter().map(|id| id.as_str()).collect::<Vec<_>>(), ["d_hard"]);
//!
//! // The embedding model is re-trained: refresh under a scrambling version and
//! // observe how stale the previous negatives became.
//! let refreshed = miner.refresh(EmbeddingVersion::new(9_999, 0.9)).expect("refresh");
//! assert_eq!(refreshed.round_index, 1);
//! let staleness = miner.latest_staleness().expect("refresh recorded staleness");
//! assert!((0.0..=1.0).contains(&staleness.mean_jaccard));
//! ```

pub mod engine;
pub mod miner;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::HardNegativeMiner;
pub use types::{
    EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeError,
    HardNegativePositivePair, HardNegativeQueryOverlap, HardNegativeQueryResult,
    HardNegativeResult, HardNegativeSample, HardNegativeStaleness, MiningRound,
};
