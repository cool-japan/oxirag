//! The [`QueryRouter`] — maps classified intent scores to a [`RoutingDecision`].

use super::classifier::IntentClassifier;
use super::types::{IntentScores, QueryRouterError, RouterConfig, RoutingDecision};

// ── QueryRouter ───────────────────────────────────────────────────────────────

/// Routes an incoming query to a retrieval strategy by classifying its intent.
///
/// # Type parameter
///
/// `C` is any [`IntentClassifier`] implementation. Use
/// [`crate::query_router::HeuristicIntentClassifier`] for production and
/// [`crate::query_router::MockIntentClassifier`] for tests.
///
/// # Example
///
/// ```rust,ignore
/// # #[cfg(feature = "query-routing")] {
/// use oxirag::query_router::{
///     HeuristicIntentClassifier, QueryRouter, RouterConfig, RoutingStrategy,
/// };
///
/// # #[tokio::main]
/// # async fn main() {
/// let router = QueryRouter::new(
///     HeuristicIntentClassifier::new(),
///     RouterConfig::default(),
/// );
/// let decision = router.route("compare Rust vs Go").await.unwrap();
/// assert_eq!(decision.strategy, RoutingStrategy::HybridSearch);
/// # }
/// # }
/// ```
pub struct QueryRouter<C: IntentClassifier> {
    /// The underlying intent classifier.
    classifier: C,
    /// Routing configuration tables.
    config: RouterConfig,
}

impl<C: IntentClassifier> QueryRouter<C> {
    /// Creates a new [`QueryRouter`] with the given classifier and configuration.
    #[must_use]
    pub fn new(classifier: C, config: RouterConfig) -> Self {
        Self { classifier, config }
    }

    /// Replaces the configuration, returning `self` for chaining.
    #[must_use]
    pub fn with_config(mut self, config: RouterConfig) -> Self {
        self.config = config;
        self
    }

    /// Classifies `query` and returns a [`RoutingDecision`].
    ///
    /// # Errors
    ///
    /// - [`QueryRouterError::EmptyQuery`] if `query` trims to empty.
    /// - [`QueryRouterError::ClassificationFailed`] if the classifier fails.
    pub async fn route(&self, query: &str) -> Result<RoutingDecision, QueryRouterError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(QueryRouterError::EmptyQuery);
        }

        let scores = self.classifier.classify(trimmed).await?;
        Ok(self.decide(trimmed, &scores))
    }

    /// Synchronous helper: converts pre-computed [`IntentScores`] into a
    /// [`RoutingDecision`] without touching the async classifier.
    ///
    /// This is useful in synchronous test contexts and for composing the
    /// routing logic with externally produced scores.
    #[must_use]
    pub fn decide(&self, query: &str, scores: &IntentScores) -> RoutingDecision {
        let (intent, conf) = scores.top();

        let (strategy, low_confidence_note) = if conf < self.config.min_confidence {
            (
                self.config.default_strategy,
                format!(
                    " (low confidence {conf:.2} < threshold {:.2}, using default)",
                    self.config.min_confidence
                ),
            )
        } else {
            (self.config.strategy_for(intent), String::new())
        };

        // Build fallback list: look up the table; ensure it is never empty.
        let mut fallback_strategies = self
            .config
            .fallback_table
            .get(&strategy)
            .cloned()
            .unwrap_or_default();
        if fallback_strategies.is_empty() {
            fallback_strategies.push(self.config.default_strategy);
        }

        let top_k_override = self.config.top_k_overrides.get(&intent).copied();

        let reasoning = format!(
            "query=\"{query}\" intent={intent} (conf={conf:.2}) -> strategy={strategy}{low_confidence_note}"
        );

        RoutingDecision {
            intent,
            intent_confidence: conf,
            strategy,
            fallback_strategies,
            top_k_override,
            reasoning,
        }
    }
}
