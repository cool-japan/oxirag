//! `PromptOptimizer` implementation.
use crate::prompt_optimization::types::{
    DevExample, OutputScorer, PromptOptimizationConfig, PromptOptimizationError, PromptVariant,
    VariantScore,
};

// ── PromptOptimizer ───────────────────────────────────────────────────────────

/// Evaluates prompt variants over a dev set and selects the best.
#[derive(Debug, Clone)]
pub struct PromptOptimizer {
    /// Configuration.
    pub config: PromptOptimizationConfig,
}

impl PromptOptimizer {
    /// Create a new optimizer with the given config.
    #[must_use]
    pub fn new(config: PromptOptimizationConfig) -> Self {
        Self { config }
    }

    /// Evaluate all `variants` over `dev_set` using `scorer`.
    ///
    /// # Errors
    ///
    /// Returns [`PromptOptimizationError::EmptyDevSet`] if `dev_set` is empty.
    pub fn evaluate_variants<S: OutputScorer>(
        &self,
        variants: &[PromptVariant],
        dev_set: &[DevExample],
        scorer: &S,
    ) -> Result<Vec<VariantScore>, PromptOptimizationError> {
        if dev_set.is_empty() {
            return Err(PromptOptimizationError::EmptyDevSet);
        }
        Ok(variants
            .iter()
            .map(|v| {
                let per_example: Vec<f32> = dev_set
                    .iter()
                    .map(|ex| {
                        let produced = v.render(&ex.input);
                        scorer.score(&produced, &ex.expected)
                    })
                    .collect();
                #[allow(clippy::cast_precision_loss)]
                let score = if per_example.is_empty() {
                    0.0
                } else {
                    per_example.iter().sum::<f32>() / per_example.len() as f32
                };
                VariantScore {
                    variant_name: v.name.clone(),
                    score,
                    per_example,
                }
            })
            .collect())
    }

    /// Return the best variant from a scored list.
    #[must_use]
    pub fn best<'a>(
        &self,
        _variants: &[PromptVariant],
        scores: &'a [VariantScore],
    ) -> Option<&'a VariantScore> {
        scores.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

impl Default for PromptOptimizer {
    fn default() -> Self {
        Self::new(PromptOptimizationConfig::default())
    }
}
