//! Noise / distractor filtering for robust RAG (RAAT-inspired).
//!
//! After retrieval, a result set frequently contains **distractors**: passages
//! that share surface keywords with the query yet carry little real relevance,
//! or topic-drift outliers that diverge from what the rest of the retrieved set
//! is "about". Feeding such passages to a generator degrades faithfulness. This
//! module detects and drops them *before* generation.
//!
//! Unlike the [`crate::context_compression`] `RedundancyFilter` — which removes
//! near-**duplicate** sentences — the [`NoiseFilter`] removes **irrelevant or
//! distracting** passages, keeping the genuinely on-topic ones.
//!
//! # Scoring
//!
//! Each passage is scored on two axes and the two are blended:
//!
//! | Axis | Meaning |
//! |------|---------|
//! | **relevance** | cosine(query, passage) blended with lexical overlap |
//! | **consensus** | cosine(passage, centroid of *all* retrieved passages) |
//!
//! `combined = (1 - consensus_weight) * relevance + consensus_weight * consensus`,
//! and any passage with `combined < relevance_threshold` is flagged as noise.
//! At least [`NoiseConfig::keep_min`] of the highest-scoring passages are always
//! retained, even when every passage is flagged.
//!
//! All scoring uses deterministic FNV-1a pseudo-embeddings — no model, no I/O,
//! no randomness.
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "noise-filter")] {
//! use oxirag::noise_filter::{NoiseConfig, NoiseFilter};
//!
//! let filter = NoiseFilter::new(NoiseConfig::new());
//! let kept = filter.filter("query text", &results);
//! # }
//! ```

pub mod filter;
#[cfg(test)]
mod tests;
pub mod types;

pub use filter::NoiseFilter;
pub use types::{NoiseConfig, NoiseFilterError, NoiseReport, PassageAssessment};
