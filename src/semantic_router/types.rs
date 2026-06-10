//! Types for the `semantic_router` module.

use thiserror::Error;

// ── RoutingTarget ─────────────────────────────────────────────────────────────

/// The retrieval strategy to route a query to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RoutingTarget {
    /// Standard vector similarity search.
    #[default]
    VectorSearch,
    /// Graph-based traversal search.
    GraphSearch,
    /// Combined vector and graph search.
    HybridSearch,
    /// Multi-hop graph traversal with chained retrieval.
    MultiHop,
    /// Agentic tool-use search with reasoning loops.
    AgenticSearch,
    /// Answer generated directly without retrieval.
    DirectAnswer,
}

impl RoutingTarget {
    /// Return the canonical string label for this target.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::VectorSearch => "vector_search",
            Self::GraphSearch => "graph_search",
            Self::HybridSearch => "hybrid_search",
            Self::MultiHop => "multi_hop",
            Self::AgenticSearch => "agentic_search",
            Self::DirectAnswer => "direct_answer",
        }
    }

    /// Parse a [`RoutingTarget`] from a string (case-insensitive).
    ///
    /// Returns `None` if the string does not match any known target.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "vector_search" | "vector" => Some(Self::VectorSearch),
            "graph_search" | "graph" => Some(Self::GraphSearch),
            "hybrid_search" | "hybrid" => Some(Self::HybridSearch),
            "multi_hop" | "multihop" | "multi-hop" => Some(Self::MultiHop),
            "agentic_search" | "agentic" => Some(Self::AgenticSearch),
            "direct_answer" | "direct" => Some(Self::DirectAnswer),
            _ => None,
        }
    }
}

// ── RouterExample ─────────────────────────────────────────────────────────────

/// A labelled example query used to train the semantic router.
#[derive(Debug, Clone)]
pub struct RouterExample {
    /// The example query text.
    pub query: String,
    /// The intended routing target for this query.
    pub target: RoutingTarget,
}

impl RouterExample {
    /// Create a new [`RouterExample`].
    #[must_use]
    pub fn new(query: impl Into<String>, target: RoutingTarget) -> Self {
        Self {
            query: query.into(),
            target,
        }
    }
}

// ── RoutingDecision ───────────────────────────────────────────────────────────

/// The result of routing a query to a retrieval strategy.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The chosen routing target.
    pub target: RoutingTarget,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Human-readable reasoning for the decision.
    pub reasoning: String,
}

impl RoutingDecision {
    /// Return `true` when the confidence meets or exceeds `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

// ── SemanticRoutingConfig ─────────────────────────────────────────────────────

/// Configuration for `SemanticRouter`.
#[derive(Debug, Clone)]
pub struct SemanticRoutingConfig {
    /// Minimum similarity required to accept the best-match example.
    ///
    /// Defaults to `0.5`.
    pub threshold: f32,
    /// Target to use when no example meets the threshold.
    ///
    /// Defaults to [`RoutingTarget::VectorSearch`].
    pub fallback: RoutingTarget,
}

impl Default for SemanticRoutingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            fallback: RoutingTarget::VectorSearch,
        }
    }
}

impl SemanticRoutingConfig {
    /// Set the similarity threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the fallback routing target.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RoutingTarget) -> Self {
        self.fallback = fallback;
        self
    }
}

// ── RouterError ───────────────────────────────────────────────────────────────

/// Errors from the `semantic_router` module.
#[derive(Debug, Error)]
pub enum RouterError {
    /// No routing examples were provided to the router.
    #[error("No routing examples provided")]
    NoExamples,
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
}
