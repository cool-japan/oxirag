//! Source credibility / authority scoring.
//!
//! Distinct from [`crate::trust_score`], which scores how trustworthy a
//! generated **answer** is (grounding, consistency, …).  This module scores how
//! authoritative a retrieved **source** is, blending three deterministic
//! signals into a per-document credibility score:
//!
//! 1. **Citation-graph `PageRank`** — sources that are cited by many (themselves
//!    authoritative) sources score higher. See [`SourceGraph`].
//! 2. **Recency** — exponential decay by age (half-life configurable). Age is
//!    always passed in explicitly as `age_days`; the wall clock is never read,
//!    keeping scoring fully deterministic.
//! 3. **Metadata authority** — membership in a trusted-source set plus an
//!    author-reputation lookup. See [`AuthoritySignals`].
//!
//! Results can optionally be re-ranked by blending relevance with credibility
//! via [`CredibilityScorer::rerank`].
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`SourceGraph`] | Directed citation graph + `PageRank` |
//! | [`AuthoritySignals`] | Trusted sources + author reputation |
//! | [`CredibilityConfig`] | Signal weights, damping, half-life |
//! | [`CredibilityScorer`] | Blend signals; score & re-rank |
//! | [`CredibilityScore`] | Per-document result (total + components) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "source-credibility")] {
//! use oxirag::source_credibility::{
//!     AuthoritySignals, CredibilityConfig, CredibilityScorer, SourceGraph,
//! };
//! use oxirag::types::DocumentId;
//!
//! let mut graph = SourceGraph::new();
//! graph.add_citation(DocumentId::from("a"), DocumentId::from("b"));
//! let ranks = graph.pagerank(0.85, 50);
//!
//! let signals = AuthoritySignals::new().with_trusted_source("nature.com");
//! let scorer = CredibilityScorer::with_signals(CredibilityConfig::new(), signals);
//! # let _ = (ranks, scorer);
//! # }
//! ```

pub mod graph;
pub mod scorer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use graph::SourceGraph;
pub use scorer::CredibilityScorer;
pub use types::{AuthoritySignals, CredibilityConfig, CredibilityScore, SourceCredibilityError};
