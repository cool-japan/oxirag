//! Core types for query-complexity classification and depth-aware routing.
//!
//! This module is **distinct** from `query_router`: where `query_router`
//! classifies a query's semantic *intent* (factual, comparative, navigational,
//! …), Adaptive-RAG classifies a query's *complexity* and maps it to a
//! retrieval **depth** — no retrieval, a single retrieval pass, or iterative
//! multi-hop retrieval — following Jeong et al. (2024), "Adaptive-RAG: Learning
//! to Adapt Retrieval-Augmented Large Language Models through Question
//! Complexity".

use thiserror::Error;

// ── QueryComplexity ───────────────────────────────────────────────────────────

/// The complexity tier of a user query.
///
/// Each tier maps to a distinct retrieval [`RetrievalStrategy`]:
///
/// | Tier | Meaning | Strategy |
/// |------|---------|----------|
/// | [`Straightforward`](QueryComplexity::Straightforward) | Answerable without retrieval | [`RetrievalStrategy::NoRetrieval`] |
/// | [`SingleStep`](QueryComplexity::SingleStep) | One retrieval pass suffices | [`RetrievalStrategy::SingleStep`] |
/// | [`MultiStep`](QueryComplexity::MultiStep) | Needs iterative / multi-hop retrieval | [`RetrievalStrategy::MultiStep`] |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryComplexity {
    /// The query is answerable without any retrieval (very short, trivial, or a
    /// closed yes/no question).
    Straightforward,
    /// A single retrieval pass is sufficient (a typical factoid asking about one
    /// entity: who / what / when / where).
    SingleStep,
    /// The query requires iterative or multi-hop retrieval (comparisons,
    /// multiple entities, temporal chains, or several sub-questions).
    MultiStep,
}

impl QueryComplexity {
    /// Returns a static string identifier for this complexity tier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Straightforward => "straightforward",
            Self::SingleStep => "single_step",
            Self::MultiStep => "multi_step",
        }
    }
}

impl std::fmt::Display for QueryComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── ComplexitySignal ──────────────────────────────────────────────────────────

/// A single named signal that fired during complexity classification.
///
/// The `weight` is the (possibly signed) contribution this signal made to the
/// raw complexity score before normalisation. Positive weights push a query
/// toward [`QueryComplexity::MultiStep`]; negative weights push it toward
/// [`QueryComplexity::Straightforward`].
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexitySignal {
    /// A short, stable identifier for the signal (e.g. `"multi_hop_cue"`).
    pub name: String,
    /// The contribution of this signal to the raw score (may be negative).
    pub weight: f32,
}

impl ComplexitySignal {
    /// Creates a new [`ComplexitySignal`].
    #[must_use]
    pub fn new(name: impl Into<String>, weight: f32) -> Self {
        Self {
            name: name.into(),
            weight,
        }
    }
}

// ── ComplexityClassification ──────────────────────────────────────────────────

/// The full result of classifying a query's complexity.
///
/// Carries the resolved [`QueryComplexity`] tier, the normalised `score` in
/// `[0.0, 1.0]`, and the ordered list of [`ComplexitySignal`]s that fired.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityClassification {
    /// The resolved complexity tier.
    pub complexity: QueryComplexity,
    /// Normalised complexity score in `[0.0, 1.0]`.
    ///
    /// Higher values indicate greater complexity. The score is mapped to a tier
    /// via [`AdaptiveRagConfig::single_step_threshold`] and
    /// [`AdaptiveRagConfig::multi_step_threshold`].
    pub score: f32,
    /// The signals that contributed to the score, in detection order.
    pub signals: Vec<ComplexitySignal>,
}

// ── RetrievalStrategy ─────────────────────────────────────────────────────────

/// The retrieval depth selected for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrievalStrategy {
    /// No retrieval is performed; the answer is generated directly.
    NoRetrieval,
    /// A single retrieval pass is performed.
    SingleStep,
    /// Iterative, multi-hop retrieval is performed up to `max_hops` passes.
    MultiStep {
        /// The maximum number of retrieval hops to perform.
        max_hops: usize,
    },
}

impl RetrievalStrategy {
    /// Returns a static string identifier for this strategy variant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoRetrieval => "no_retrieval",
            Self::SingleStep => "single_step",
            Self::MultiStep { .. } => "multi_step",
        }
    }
}

impl std::fmt::Display for RetrievalStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── RoutingPlan ───────────────────────────────────────────────────────────────

/// The plan produced by [`crate::adaptive_rag::AdaptiveRagRouter::route`].
///
/// Bundles the detected [`QueryComplexity`], the chosen [`RetrievalStrategy`],
/// and a recommended `top_k` for the retrieval step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingPlan {
    /// The detected complexity tier.
    pub complexity: QueryComplexity,
    /// The retrieval strategy selected for the tier.
    pub strategy: RetrievalStrategy,
    /// The recommended number of documents to retrieve.
    ///
    /// `0` for [`QueryComplexity::Straightforward`] (no retrieval).
    pub recommended_top_k: usize,
}

// ── AdaptiveRagConfig ─────────────────────────────────────────────────────────

/// Configuration governing complexity thresholds and retrieval depth.
///
/// A normalised complexity score `s` is mapped to a tier as follows:
///
/// - `s < single_step_threshold` ⇒ [`QueryComplexity::Straightforward`]
/// - `single_step_threshold <= s < multi_step_threshold` ⇒ [`QueryComplexity::SingleStep`]
/// - `s >= multi_step_threshold` ⇒ [`QueryComplexity::MultiStep`]
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveRagConfig {
    /// Score below which a query is [`QueryComplexity::Straightforward`].
    ///
    /// Default: `0.25`.
    pub single_step_threshold: f32,
    /// Score at or above which a query is [`QueryComplexity::MultiStep`].
    ///
    /// Default: `0.6`.
    pub multi_step_threshold: f32,
    /// Base number of documents to retrieve for a single-step query.
    ///
    /// Default: `5`. Multi-step queries retrieve `2 * base_top_k`.
    pub base_top_k: usize,
    /// Maximum number of retrieval hops for a multi-step query.
    ///
    /// Default: `3`.
    pub multi_step_max_hops: usize,
}

impl Default for AdaptiveRagConfig {
    fn default() -> Self {
        Self {
            single_step_threshold: 0.25,
            multi_step_threshold: 0.6,
            base_top_k: 5,
            multi_step_max_hops: 3,
        }
    }
}

impl AdaptiveRagConfig {
    /// Creates a new [`AdaptiveRagConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the single-step (straightforward → single-step) threshold.
    #[must_use]
    pub fn with_single_step_threshold(mut self, threshold: f32) -> Self {
        self.single_step_threshold = threshold;
        self
    }

    /// Sets the multi-step (single-step → multi-step) threshold.
    #[must_use]
    pub fn with_multi_step_threshold(mut self, threshold: f32) -> Self {
        self.multi_step_threshold = threshold;
        self
    }

    /// Sets the base top-k document count for single-step retrieval.
    #[must_use]
    pub fn with_base_top_k(mut self, base_top_k: usize) -> Self {
        self.base_top_k = base_top_k;
        self
    }

    /// Sets the maximum number of hops for multi-step retrieval.
    #[must_use]
    pub fn with_multi_step_max_hops(mut self, max_hops: usize) -> Self {
        self.multi_step_max_hops = max_hops;
        self
    }

    /// Maps a normalised score to a [`QueryComplexity`] tier using the
    /// configured thresholds.
    #[must_use]
    pub fn tier_for_score(&self, score: f32) -> QueryComplexity {
        if score < self.single_step_threshold {
            QueryComplexity::Straightforward
        } else if score < self.multi_step_threshold {
            QueryComplexity::SingleStep
        } else {
            QueryComplexity::MultiStep
        }
    }
}

// ── AdaptiveRagError ──────────────────────────────────────────────────────────

/// Errors that can arise during adaptive complexity classification or routing.
#[derive(Debug, Error)]
pub enum AdaptiveRagError {
    /// The query string was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
}
