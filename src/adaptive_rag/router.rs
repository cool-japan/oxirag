//! Depth-aware retrieval router.
//!
//! The [`AdaptiveRagRouter`] wraps a [`ComplexityClassifier`] and translates the
//! detected [`QueryComplexity`] tier into a concrete [`RoutingPlan`]: a
//! [`RetrievalStrategy`] plus a recommended `top_k`.
//!
//! Routing policy:
//!
//! | Complexity | Strategy | `top_k` |
//! |------------|----------|---------|
//! | [`Straightforward`](QueryComplexity::Straightforward) | [`NoRetrieval`](RetrievalStrategy::NoRetrieval) | `0` |
//! | [`SingleStep`](QueryComplexity::SingleStep) | [`SingleStep`](RetrievalStrategy::SingleStep) | `base_top_k` |
//! | [`MultiStep`](QueryComplexity::MultiStep) | [`MultiStep`](RetrievalStrategy::MultiStep) | `2 * base_top_k` |

use super::classifier::ComplexityClassifier;
use super::types::{
    AdaptiveRagConfig, AdaptiveRagError, QueryComplexity, RetrievalStrategy, RoutingPlan,
};

// ── AdaptiveRagRouter ─────────────────────────────────────────────────────────

/// Routes a query to a retrieval depth based on its classified complexity.
#[derive(Debug, Clone)]
pub struct AdaptiveRagRouter {
    /// The underlying complexity classifier.
    pub classifier: ComplexityClassifier,
    /// The configuration governing thresholds and retrieval depth.
    pub config: AdaptiveRagConfig,
}

impl AdaptiveRagRouter {
    /// Creates a new router from a configuration.
    ///
    /// The router builds its own [`ComplexityClassifier`] from a clone of
    /// `config` so that classifier and router share identical thresholds.
    #[must_use]
    pub fn new(config: AdaptiveRagConfig) -> Self {
        let classifier = ComplexityClassifier::new(config.clone());
        Self { classifier, config }
    }

    /// Classifies `query` and produces a [`RoutingPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`AdaptiveRagError::EmptyQuery`] if `query` is empty or contains
    /// only whitespace.
    pub fn route(&self, query: &str) -> Result<RoutingPlan, AdaptiveRagError> {
        let classification = self.classifier.classify(query)?;
        let complexity = classification.complexity;
        let (strategy, recommended_top_k) = match complexity {
            QueryComplexity::Straightforward => (RetrievalStrategy::NoRetrieval, 0),
            QueryComplexity::SingleStep => (RetrievalStrategy::SingleStep, self.config.base_top_k),
            QueryComplexity::MultiStep => (
                RetrievalStrategy::MultiStep {
                    max_hops: self.config.multi_step_max_hops,
                },
                self.config.base_top_k.saturating_mul(2),
            ),
        };

        Ok(RoutingPlan {
            complexity,
            strategy,
            recommended_top_k,
        })
    }
}
