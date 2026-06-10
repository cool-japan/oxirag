//! Core types for query intent classification and retrieval-strategy routing.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── QueryIntent ───────────────────────────────────────────────────────────────

/// The semantic intent category of a user query.
///
/// Variants map to distinct retrieval strategies in [`RouterConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryIntent {
    /// A query expecting a specific factual answer (who/what/where/why/how).
    Factual,
    /// A query asking for a definition or explanation of a concept.
    Definitional,
    /// A query comparing two or more entities or concepts.
    Comparative,
    /// A query navigating to a specific resource, page, or location.
    Navigational,
    /// A query requiring reasoning across multiple documents or facts.
    MultiHop,
    /// A conversational or follow-up query referring to prior context.
    Conversational,
    /// An open-ended query seeking broad coverage of a topic.
    Exploratory,
    /// A query about events at a specific time or asking about history/timelines.
    Temporal,
    /// A query requiring aggregation, counting, or list-generation over a corpus.
    Aggregation,
    /// Intent could not be determined with sufficient confidence.
    Unknown,
}

impl QueryIntent {
    /// Returns a static string identifier for this intent variant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Factual => "factual",
            Self::Definitional => "definitional",
            Self::Comparative => "comparative",
            Self::Navigational => "navigational",
            Self::MultiHop => "multi_hop",
            Self::Conversational => "conversational",
            Self::Exploratory => "exploratory",
            Self::Temporal => "temporal",
            Self::Aggregation => "aggregation",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── RoutingStrategy ───────────────────────────────────────────────────────────

/// The retrieval strategy that should be used to answer a query.
///
/// Strategies are produced by [`RouterConfig::strategy_for`] based on the
/// detected [`QueryIntent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Dense-vector nearest-neighbour search over the embedding store.
    VectorSearch,
    /// Combination of dense-vector and sparse (BM25-style) search.
    HybridSearch,
    /// Traversal and subgraph queries over the knowledge graph layer.
    GraphSearch,
    /// Iterative, multi-step retrieval for complex reasoning chains.
    MultiHop,
    /// Retrieval enriched with conversation history and session context.
    Conversational,
    /// No retrieval required; the answer can be generated directly.
    DirectAnswer,
}

impl RoutingStrategy {
    /// Returns a static string identifier for this strategy variant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VectorSearch => "vector_search",
            Self::HybridSearch => "hybrid_search",
            Self::GraphSearch => "graph_search",
            Self::MultiHop => "multi_hop",
            Self::Conversational => "conversational",
            Self::DirectAnswer => "direct_answer",
        }
    }
}

impl std::fmt::Display for RoutingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── IntentScores ──────────────────────────────────────────────────────────────

/// A scored, descending-sorted distribution over [`QueryIntent`] variants.
///
/// Scores are in the range `[0.0, 1.0]`. The list is always non-empty because
/// the classifier ensures a floor entry of `(Unknown, 0.05)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentScores {
    /// Scored intents sorted in descending order of confidence.
    pub scores: Vec<(QueryIntent, f32)>,
}

impl IntentScores {
    /// Creates a new [`IntentScores`] from a raw list, sorting descending by score.
    #[must_use]
    pub fn new(mut scores: Vec<(QueryIntent, f32)>) -> Self {
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Self { scores }
    }

    /// Returns the highest-confidence `(intent, score)` pair.
    ///
    /// If the list is empty (should not happen in practice), returns
    /// `(QueryIntent::Unknown, 0.0)` as a safe default.
    #[must_use]
    pub fn top(&self) -> (QueryIntent, f32) {
        self.scores
            .first()
            .copied()
            .unwrap_or((QueryIntent::Unknown, 0.0))
    }

    /// Returns the confidence score for a specific intent, or `0.0` if absent.
    #[must_use]
    pub fn confidence_of(&self, intent: QueryIntent) -> f32 {
        self.scores
            .iter()
            .find(|(i, _)| *i == intent)
            .map_or(0.0, |(_, s)| *s)
    }

    /// Returns `true` if no intent scores are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

// ── RoutingDecision ───────────────────────────────────────────────────────────

/// The output of [`crate::query_router::QueryRouter::route`].
///
/// Encodes the top intent, the chosen retrieval strategy, ordered fallbacks,
/// an optional top-k override, and a human-readable reasoning string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The primary detected intent.
    pub intent: QueryIntent,
    /// Confidence score for the primary intent, in `[0.0, 1.0]`.
    pub intent_confidence: f32,
    /// The retrieval strategy selected for the intent.
    pub strategy: RoutingStrategy,
    /// Ordered fallback strategies if the primary strategy is unavailable.
    pub fallback_strategies: Vec<RoutingStrategy>,
    /// Optional override for the number of documents to retrieve (`top_k`).
    pub top_k_override: Option<usize>,
    /// Human-readable explanation of the routing decision.
    pub reasoning: String,
}

// ── RouterConfig ──────────────────────────────────────────────────────────────

/// Configuration tables that govern how intents map to retrieval strategies.
///
/// The default instance carries complete, production-ready tables covering all
/// ten [`QueryIntent`] variants with sensible fallbacks and top-k overrides.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Primary mapping from [`QueryIntent`] to [`RoutingStrategy`].
    pub intent_strategy_table: HashMap<QueryIntent, RoutingStrategy>,
    /// Fallback strategies indexed by primary [`RoutingStrategy`].
    pub fallback_table: HashMap<RoutingStrategy, Vec<RoutingStrategy>>,
    /// Minimum confidence below which the `default_strategy` is used instead.
    ///
    /// Default: `0.3`.
    pub min_confidence: f32,
    /// Strategy used when confidence falls below `min_confidence`.
    pub default_strategy: RoutingStrategy,
    /// Per-intent overrides for the number of retrieved documents.
    pub top_k_overrides: HashMap<QueryIntent, usize>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        let mut intent_strategy_table = HashMap::new();
        intent_strategy_table.insert(QueryIntent::Factual, RoutingStrategy::VectorSearch);
        intent_strategy_table.insert(QueryIntent::Definitional, RoutingStrategy::VectorSearch);
        intent_strategy_table.insert(QueryIntent::Unknown, RoutingStrategy::VectorSearch);
        intent_strategy_table.insert(QueryIntent::Comparative, RoutingStrategy::HybridSearch);
        intent_strategy_table.insert(QueryIntent::Exploratory, RoutingStrategy::HybridSearch);
        intent_strategy_table.insert(QueryIntent::Temporal, RoutingStrategy::HybridSearch);
        intent_strategy_table.insert(QueryIntent::Navigational, RoutingStrategy::DirectAnswer);
        intent_strategy_table.insert(QueryIntent::MultiHop, RoutingStrategy::MultiHop);
        intent_strategy_table.insert(QueryIntent::Conversational, RoutingStrategy::Conversational);
        intent_strategy_table.insert(QueryIntent::Aggregation, RoutingStrategy::GraphSearch);

        let mut fallback_table: HashMap<RoutingStrategy, Vec<RoutingStrategy>> = HashMap::new();
        // Every strategy falls back to VectorSearch by default
        for strategy in [
            RoutingStrategy::VectorSearch,
            RoutingStrategy::HybridSearch,
            RoutingStrategy::DirectAnswer,
            RoutingStrategy::Conversational,
        ] {
            fallback_table.insert(strategy, vec![RoutingStrategy::VectorSearch]);
        }
        // MultiHop and GraphSearch have richer fallback chains
        fallback_table.insert(
            RoutingStrategy::MultiHop,
            vec![RoutingStrategy::HybridSearch, RoutingStrategy::VectorSearch],
        );
        fallback_table.insert(
            RoutingStrategy::GraphSearch,
            vec![RoutingStrategy::HybridSearch, RoutingStrategy::VectorSearch],
        );

        let mut top_k_overrides = HashMap::new();
        top_k_overrides.insert(QueryIntent::MultiHop, 20_usize);
        top_k_overrides.insert(QueryIntent::Aggregation, 50_usize);
        top_k_overrides.insert(QueryIntent::Exploratory, 15_usize);

        Self {
            intent_strategy_table,
            fallback_table,
            min_confidence: 0.3,
            default_strategy: RoutingStrategy::VectorSearch,
            top_k_overrides,
        }
    }
}

impl RouterConfig {
    /// Creates a new [`RouterConfig`] with the default tables.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the strategy for a single intent.
    #[must_use]
    pub fn with_route(mut self, intent: QueryIntent, strategy: RoutingStrategy) -> Self {
        self.intent_strategy_table.insert(intent, strategy);
        self
    }

    /// Overrides the fallback list for a single primary strategy.
    #[must_use]
    pub fn with_fallbacks(
        mut self,
        strategy: RoutingStrategy,
        fallbacks: Vec<RoutingStrategy>,
    ) -> Self {
        self.fallback_table.insert(strategy, fallbacks);
        self
    }

    /// Sets the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Sets the default strategy used when confidence is too low.
    #[must_use]
    pub fn with_default_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.default_strategy = strategy;
        self
    }

    /// Sets a top-k document-count override for a specific intent.
    #[must_use]
    pub fn with_top_k_override(mut self, intent: QueryIntent, top_k: usize) -> Self {
        self.top_k_overrides.insert(intent, top_k);
        self
    }

    /// Returns the routing strategy for the given intent.
    ///
    /// Falls back to `self.default_strategy` if the intent is not in the table.
    #[must_use]
    pub fn strategy_for(&self, intent: QueryIntent) -> RoutingStrategy {
        self.intent_strategy_table
            .get(&intent)
            .copied()
            .unwrap_or(self.default_strategy)
    }
}

// ── QueryRouterError ──────────────────────────────────────────────────────────

/// Errors that can arise during query routing.
#[derive(Debug, Error)]
pub enum QueryRouterError {
    /// The query string was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,

    /// The underlying intent classifier failed.
    #[error("classification failed: {0}")]
    ClassificationFailed(String),

    /// No routing strategy could be found for the detected intent.
    #[error("no strategy for intent: {0}")]
    NoStrategyForIntent(String),
}
