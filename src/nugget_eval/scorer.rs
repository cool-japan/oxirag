//! Nugget coverage scoring of system answers.
use crate::nugget_eval::extractor::{HeuristicNuggetExtractor, NuggetExtractor, tokenize};
use crate::nugget_eval::types::{
    Nugget, NuggetConfig, NuggetEvalError, NuggetImportance, NuggetScore,
};
use std::collections::HashSet;

// ── NuggetScorer ──────────────────────────────────────────────────────────────

/// Scores a system answer by the weighted fraction of reference nuggets it
/// covers.
///
/// Coverage is decided lexically: a nugget is covered when the fraction of its
/// tokens present in the answer reaches
/// [`NuggetConfig::coverage_threshold`]. This is a deterministic, LLM-free
/// approximation of lexical entailment, distinct from the embedding- or
/// model-based scorers elsewhere in `OxiRAG`.
#[derive(Debug, Clone, PartialEq)]
pub struct NuggetScorer<E: NuggetExtractor> {
    /// Configuration controlling extraction and coverage.
    pub config: NuggetConfig,
    /// The extractor used to decompose reference answers.
    pub extractor: E,
}

impl NuggetScorer<HeuristicNuggetExtractor> {
    /// Create a scorer with the given config and the default
    /// [`HeuristicNuggetExtractor`].
    ///
    /// The extractor's minimum-token setting is taken from
    /// [`NuggetConfig::min_nugget_tokens`] so that extraction and configuration
    /// stay consistent.
    #[must_use]
    pub fn new(config: NuggetConfig) -> Self {
        let extractor = HeuristicNuggetExtractor::with_min_tokens(config.min_nugget_tokens);
        Self { config, extractor }
    }
}

impl<E: NuggetExtractor> NuggetScorer<E> {
    /// Create a scorer with a custom [`NuggetExtractor`].
    #[must_use]
    pub fn with_extractor(config: NuggetConfig, extractor: E) -> Self {
        Self { config, extractor }
    }

    /// Returns `true` when the fraction of `nugget` tokens present in `answer`
    /// reaches [`NuggetConfig::coverage_threshold`].
    ///
    /// A nugget with no tokens (e.g. punctuation only) is never covered.
    #[must_use]
    pub fn is_covered(&self, nugget: &Nugget, answer: &str) -> bool {
        let nugget_tokens = tokenize(&nugget.text);
        if nugget_tokens.is_empty() {
            return false;
        }
        let answer_tokens: HashSet<String> = tokenize(answer).into_iter().collect();
        let present = nugget_tokens
            .iter()
            .filter(|tok| answer_tokens.contains(*tok))
            .count();
        #[allow(clippy::cast_precision_loss)]
        let fraction = present as f32 / nugget_tokens.len() as f32;
        fraction >= self.config.coverage_threshold
    }

    /// Score `answer` against a pre-extracted nugget slice.
    ///
    /// - `coverage` is `covered / total`.
    /// - `vital_coverage` is `covered_vital / total_vital` (`0.0` when there are
    ///   no vital nuggets).
    /// - `weighted_score` is `Σ weight(covered) / Σ weight(all)`.
    ///
    /// An empty `nuggets` slice yields an all-zero [`NuggetScore`].
    #[must_use]
    pub fn score_with_nuggets(&self, nuggets: &[Nugget], answer: &str) -> NuggetScore {
        let total = nuggets.len();
        if total == 0 {
            return NuggetScore {
                coverage: 0.0,
                vital_coverage: 0.0,
                weighted_score: 0.0,
                covered: Vec::new(),
                missed: Vec::new(),
            };
        }

        let mut covered = Vec::new();
        let mut missed = Vec::new();
        let mut total_vital = 0usize;
        let mut covered_vital = 0usize;
        let mut weight_all = 0.0f32;
        let mut weight_covered = 0.0f32;

        for (idx, nugget) in nuggets.iter().enumerate() {
            let is_vital = matches!(nugget.importance, NuggetImportance::Vital);
            if is_vital {
                total_vital += 1;
            }
            let weight = self.config.weight_for(nugget.importance);
            weight_all += weight;

            if self.is_covered(nugget, answer) {
                covered.push(idx);
                weight_covered += weight;
                if is_vital {
                    covered_vital += 1;
                }
            } else {
                missed.push(idx);
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let coverage = covered.len() as f32 / total as f32;
        #[allow(clippy::cast_precision_loss)]
        let vital_coverage = if total_vital == 0 {
            0.0
        } else {
            covered_vital as f32 / total_vital as f32
        };
        let weighted_score = if weight_all == 0.0 {
            0.0
        } else {
            weight_covered / weight_all
        };

        NuggetScore {
            coverage,
            vital_coverage,
            weighted_score,
            covered,
            missed,
        }
    }

    /// Extract nuggets from `reference` and score `answer` against them.
    ///
    /// # Errors
    ///
    /// Returns [`NuggetEvalError::EmptyReference`] when `reference` is empty or
    /// contains only whitespace.
    pub fn score(&self, reference: &str, answer: &str) -> Result<NuggetScore, NuggetEvalError> {
        if reference.trim().is_empty() {
            return Err(NuggetEvalError::EmptyReference);
        }
        let nuggets = self.extractor.extract(reference);
        Ok(self.score_with_nuggets(&nuggets, answer))
    }
}
