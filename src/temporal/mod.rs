//! Recency-decay re-scoring for retrieved documents.
//!
//! Re-ranks search results by blending the original retrieval score with a
//! temporal decay factor derived from document timestamps stored in metadata.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`TemporalReranker`] | Applies recency decay to result scores |
//! | [`DecayFunction`] | Pluggable decay: Exponential / Linear / Gaussian / None |
//! | [`TemporalScore`] | Per-result breakdown: original, decayed, age |
//!
//! # Metadata keys
//!
//! Timestamps are read from `"updated_at"` and/or `"created_at"` metadata
//! fields (ISO 8601 date/datetime or Unix epoch seconds).
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "temporal-retrieval")] {
//! use oxirag::prelude::*;
//!
//! let reranker = TemporalReranker::new();
//! # }
//! ```

pub mod reranker;
#[cfg(test)]
mod tests;
pub mod types;

pub use reranker::TemporalReranker;
pub use types::{DecayFunction, TemporalConfig, TemporalError, TemporalScore};
