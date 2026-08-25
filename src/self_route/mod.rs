//! Self-Route: answerability-based routing between RAG and long context.
//!
//! This module implements the routing idea of Li et al. (2024), *Retrieval
//! Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid
//! Approach*. For each query, the retrieved context is inspected to estimate
//! whether it is **sufficient to answer the query** (in which case cheap
//! retrieval-augmented generation is used) or whether the query is effectively
//! unanswerable from retrieval and should fall back to a **long-context** model
//! fed the full document / corpus:
//!
//! - [`RouteDecision::Rag`] ⇒ answer from the retrieved documents,
//! - [`RouteDecision::LongContext`] ⇒ feed the full document / corpus.
//!
//! # Distinct from `adaptive_rag`
//!
//! `adaptive_rag` classifies a query's **complexity** and maps it to a
//! retrieval *depth* (no retrieval, single pass, multi-hop). Self-Route instead
//! inspects the **retrieved context** for the query and decides between cheap
//! RAG and an expensive long-context fallback based on *answerability*. The two
//! are complementary: Adaptive-RAG picks *how much* to retrieve; Self-Route
//! decides *whether retrieval was enough*.
//!
//! # Answerability heuristic
//!
//! Answerability is a deterministic, convex blend of two cheap signals:
//!
//! ```text
//! answerability = coverage_weight * query_coverage
//!               + (1 - coverage_weight) * top_relevance
//! ```
//!
//! where `query_coverage` is the fraction of distinct query content tokens
//! appearing anywhere in the retrieved documents and `top_relevance` is the best
//! query↔document token-overlap (Jaccard). When the blend meets
//! [`SelfRouteConfig::answerability_threshold`] the query routes to
//! [`RouteDecision::Rag`]; otherwise to [`RouteDecision::LongContext`].
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`SelfRouter`] | Assess answerability and produce a [`RouteAssessment`] / [`RouteDecision`] |
//! | [`SelfRouteConfig`] | Blend weight + answerability threshold |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "self-route")] {
//! use oxirag::self_route::{RouteDecision, SelfRouteConfig, SelfRouter};
//! use oxirag::types::Document;
//!
//! let router = SelfRouter::new(SelfRouteConfig::default());
//! let docs = vec![Document::new("The Eiffel Tower is located in Paris, France.")];
//! let decision = router.route("Where is the Eiffel Tower located?", &docs).unwrap();
//! assert_eq!(decision, RouteDecision::Rag);
//! # }
//! ```

pub mod router;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use router::SelfRouter;
pub use types::{RouteAssessment, RouteDecision, SelfRouteConfig, SelfRouteError};
