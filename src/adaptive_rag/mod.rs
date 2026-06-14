//! Adaptive-RAG: query-complexity routing for retrieval depth.
//!
//! This module implements the query-complexity routing idea of Jeong et al.
//! (2024), *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
//! Models through Question Complexity*. Each query is classified into one of
//! three complexity tiers, and that tier selects the **depth** of retrieval:
//!
//! - [`QueryComplexity::Straightforward`] ⇒ no retrieval (answer directly),
//! - [`QueryComplexity::SingleStep`] ⇒ one retrieval pass,
//! - [`QueryComplexity::MultiStep`] ⇒ iterative / multi-hop retrieval.
//!
//! # Distinct from `query_router`
//!
//! `query_router` classifies a query's semantic **intent** (factual,
//! comparative, navigational, …) and maps it to a retrieval *modality* (vector,
//! hybrid, graph). Adaptive-RAG instead classifies a query's **complexity** and
//! maps it to a retrieval *depth*. The two are complementary.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ComplexityClassifier`] | Classify a query into a [`ComplexityClassification`] |
//! | [`AdaptiveRagRouter`] | Map a complexity tier to a [`RoutingPlan`] |
//! | [`AdaptiveRagConfig`] | Thresholds + retrieval-depth parameters |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "adaptive-rag")] {
//! use oxirag::adaptive_rag::{AdaptiveRagConfig, AdaptiveRagRouter, RetrievalStrategy};
//!
//! let router = AdaptiveRagRouter::new(AdaptiveRagConfig::default());
//! let plan = router.route("Compare the GDP of France and Germany after 2010").unwrap();
//! assert!(matches!(plan.strategy, RetrievalStrategy::MultiStep { .. }));
//! # }
//! ```

pub mod classifier;
pub mod router;
#[cfg(test)]
mod tests;
pub mod types;

pub use classifier::ComplexityClassifier;
pub use router::AdaptiveRagRouter;
pub use types::{
    AdaptiveRagConfig, AdaptiveRagError, ComplexityClassification, ComplexitySignal,
    QueryComplexity, RetrievalStrategy, RoutingPlan,
};
