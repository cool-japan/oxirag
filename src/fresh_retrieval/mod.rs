//! Fresh retrieval: time-sensitivity detection, freshness scoring, and staleness flagging.
//!
//! `FreshLLM`-style freshness awareness (Vu et al., 2023, *"`FreshLLMs`: Refreshing
//! Large Language Models with Search Engine Augmentation"*). Many questions ask
//! about **fast-changing** facts — prices, scores, weather, the *latest* release —
//! whose correct answer drifts as the world moves on. Answering such a question
//! from a stale corpus yields confidently-wrong output. This module decides, from
//! the *query text alone*, how **time-sensitive** a question is, scores how
//! **fresh** a document is from its age, and **flags** answers that are likely
//! stale.
//!
//! # Distinct from the `temporal` module
//!
//! The `temporal` module performs recency-decay **re-ranking**: it always
//! down-weights older documents by a fixed decay curve, regardless of the query.
//! Fresh retrieval is **query-aware**. It first asks *"is this question even
//! time-sensitive?"* via [`FreshnessAnalyzer::classify`], and only then lets
//! freshness influence ranking — proportionally to the detected
//! [`FreshnessAssessment::freshness_demand`]. A static question
//! (*"Who wrote Hamlet?"*) is left essentially untouched, while a fast-changing
//! one (*"What is the latest iPhone price?"*) has fresher documents promoted.
//! Fresh retrieval additionally exposes explicit **staleness flagging** via
//! [`FreshnessAnalyzer::is_stale`], which the `temporal` module does not.
//!
//! # Scoring
//!
//! | Step | Method | Output |
//! |------|--------|--------|
//! | Classify | [`FreshnessAnalyzer::classify`] | [`TimeSensitivity`] + demand in `[0,1]` |
//! | Freshness | [`FreshnessAnalyzer::freshness_score`] | `0.5^(age_days / half_life_days)` |
//! | Staleness | [`FreshnessAnalyzer::is_stale`] | `bool` (fast-changing + old answer) |
//! | Re-rank | [`FreshnessAnalyzer::rerank`] | freshness-blended [`crate::types::SearchResult`]s |
//!
//! All logic is deterministic — no model, no randomness, no wall-clock reads.
//! Document ages are always supplied explicitly as `age_days`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "fresh-retrieval")] {
//! use oxirag::fresh_retrieval::{FreshConfig, FreshnessAnalyzer, TimeSensitivity};
//!
//! let analyzer = FreshnessAnalyzer::new(FreshConfig::new());
//! let assessment = analyzer.classify("What is the latest iPhone price?")?;
//! assert_eq!(assessment.sensitivity, TimeSensitivity::FastChanging);
//! # Ok::<(), oxirag::fresh_retrieval::FreshError>(())
//! # }
//! ```

pub mod analyzer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use analyzer::FreshnessAnalyzer;
pub use types::{FreshConfig, FreshError, FreshnessAssessment, TimeSensitivity};
