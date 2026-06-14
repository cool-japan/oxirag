//! Ensemble retrieval: orchestrate multiple weighted retriever strategies.
//!
//! An [`EnsembleRetriever`] combines several pluggable [`SubRetriever`]
//! strategies — for example a sparse lexical retriever, a dense vector
//! retriever, and a metadata-aware retriever — each registered with a non-negative
//! weight. For a query it runs every sub-retriever, then fuses their candidate
//! lists into a single ranking. Because a document surfaced by several retrievers
//! accumulates a contribution from each, cross-retriever *agreement* lifts a
//! document's rank.
//!
//! # Relationship to the `rank_fusion` module
//!
//! This module is deliberately distinct from `rank_fusion`. `rank_fusion` fuses
//! *already-produced* ranked lists; it knows nothing about where the lists came
//! from. An [`EnsembleRetriever`] instead *owns* the retrievers: it produces the
//! lists by calling each [`SubRetriever`], then fuses them. In short, `rank_fusion`
//! is the fusion math, while this module is the retriever orchestration that
//! drives it.
//!
//! # Fusion strategies
//!
//! | Strategy | Per-retriever contribution | Notes |
//! |---|---|---|
//! | [`EnsembleFusion::WeightedScore`] | `weight × min-max-normalized score` | score-scale agnostic via per-retriever normalization |
//! | [`EnsembleFusion::WeightedRrf`] | `weight × 1 / (rrf_k + rank + 1)` | rank-based; the robust default |
//!
//! # Determinism
//!
//! All operations are deterministic and use no randomness or floating-point
//! reductions whose order varies. Documents are keyed by their [`DocumentId`]
//! string, and ties in every ranking are broken by ascending [`DocumentId`]
//! string order, so identical inputs always yield identical outputs.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "ensemble-retriever")]
//! # {
//! use oxirag::ensemble_retriever::{
//!     EnsembleConfig, EnsembleFusion, EnsembleRetriever, LexicalSubRetriever,
//! };
//! use oxirag::types::Document;
//!
//! // Two corpora behind two lexical retrievers.
//! let primary = vec![
//!     Document::new("rust async runtime tokio").with_id("a"),
//!     Document::new("rust ownership and borrowing").with_id("b"),
//! ];
//! let secondary = vec![
//!     Document::new("async tokio scheduler internals").with_id("a"),
//!     Document::new("python asyncio event loop").with_id("c"),
//! ];
//!
//! let config = EnsembleConfig::new().with_fusion(EnsembleFusion::WeightedRrf);
//! let mut ensemble = EnsembleRetriever::new(config);
//! ensemble.add_retriever(Box::new(LexicalSubRetriever::new("primary", primary)), 1.0);
//! ensemble.add_retriever(Box::new(LexicalSubRetriever::new("secondary", secondary)), 1.0);
//!
//! let ranked = ensemble.retrieve("async tokio", 10).unwrap();
//! // "a" is surfaced by *both* retrievers, so agreement ranks it first.
//! assert_eq!(ranked[0].0.as_str(), "a");
//! # }
//! ```
//!
//! [`DocumentId`]: crate::types::DocumentId

mod retriever;
mod types;

#[cfg(test)]
mod tests;

pub use retriever::{EnsembleRetriever, LexicalSubRetriever, SubRetriever};
pub use types::{EnsembleConfig, EnsembleError, EnsembleFusion};
