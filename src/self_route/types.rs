//! Core types for Self-Route answerability-based routing.
//!
//! This module is **distinct** from `adaptive_rag`: where `adaptive_rag`
//! classifies a query's *complexity* and maps it to a retrieval **depth**,
//! Self-Route inspects the *retrieved context* and decides whether that context
//! is sufficient to answer the query with cheap RAG, or whether the query is
//! effectively unanswerable from retrieval and should fall back to feeding the
//! full document / corpus as **long context**, following Li et al. (2024),
//! "Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study
//! and Hybrid Approach".

use thiserror::Error;

// ── RouteDecision ─────────────────────────────────────────────────────────────

/// The route chosen for a query given its retrieved context.
///
/// | Decision | Meaning |
/// |----------|---------|
/// | [`Rag`](RouteDecision::Rag) | The retrieved context is sufficient; answer with cheap RAG |
/// | [`LongContext`](RouteDecision::LongContext) | The context is insufficient; fall back to feeding the full document / corpus |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteDecision {
    /// The retrieved context answers the query; use retrieval-augmented
    /// generation over the retrieved documents only.
    Rag,
    /// The retrieved context is insufficient; fall back to a long-context model
    /// fed the full document or corpus.
    LongContext,
}

impl RouteDecision {
    /// Returns a static string identifier for this decision.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rag => "rag",
            Self::LongContext => "long_context",
        }
    }
}

impl std::fmt::Display for RouteDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── SelfRouteConfig ───────────────────────────────────────────────────────────

/// Configuration governing the answerability blend and routing threshold.
///
/// The answerability score `a` of a query against its retrieved context is a
/// convex blend of two signals, controlled by [`coverage_weight`](SelfRouteConfig::coverage_weight) `w`:
///
/// ```text
/// a = w * query_coverage + (1 - w) * top_relevance
/// ```
///
/// where `query_coverage` is the fraction of query content tokens appearing
/// anywhere in the retrieved documents and `top_relevance` is the best
/// query↔document token-overlap. The decision is then:
///
/// - `a >= answerability_threshold` ⇒ [`RouteDecision::Rag`]
/// - `a < answerability_threshold` ⇒ [`RouteDecision::LongContext`]
#[derive(Debug, Clone, PartialEq)]
pub struct SelfRouteConfig {
    /// Answerability score at or above which the query routes to
    /// [`RouteDecision::Rag`]; below it routes to [`RouteDecision::LongContext`].
    ///
    /// Default: `0.4`.
    pub answerability_threshold: f32,
    /// Weight `w` applied to query coverage in the answerability blend; the top
    /// relevance receives weight `1 - w`.
    ///
    /// Default: `0.5`.
    pub coverage_weight: f32,
}

impl Default for SelfRouteConfig {
    fn default() -> Self {
        Self {
            answerability_threshold: 0.4,
            coverage_weight: 0.5,
        }
    }
}

impl SelfRouteConfig {
    /// Creates a new [`SelfRouteConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the answerability threshold that gates RAG vs. long-context routing.
    #[must_use]
    pub fn with_answerability_threshold(mut self, threshold: f32) -> Self {
        self.answerability_threshold = threshold;
        self
    }

    /// Sets the weight applied to query coverage in the answerability blend.
    #[must_use]
    pub fn with_coverage_weight(mut self, weight: f32) -> Self {
        self.coverage_weight = weight;
        self
    }
}

// ── RouteAssessment ───────────────────────────────────────────────────────────

/// The full result of assessing a query against its retrieved context.
///
/// Carries the resolved [`RouteDecision`], the blended `answerability` score,
/// the two component signals (`query_coverage` and `top_relevance`), and a
/// human-readable `reason`.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteAssessment {
    /// The resolved route.
    pub decision: RouteDecision,
    /// The blended answerability score in `[0.0, 1.0]`.
    pub answerability: f32,
    /// Fraction of query content tokens found anywhere in the retrieved
    /// documents, in `[0.0, 1.0]`.
    pub query_coverage: f32,
    /// Best query↔document token-overlap across the retrieved documents, in
    /// `[0.0, 1.0]`.
    pub top_relevance: f32,
    /// A human-readable explanation of why the decision was made.
    pub reason: String,
}

// ── SelfRouteError ────────────────────────────────────────────────────────────

/// Errors that can arise during Self-Route assessment or routing.
#[derive(Debug, Error)]
pub enum SelfRouteError {
    /// The query string was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
}
