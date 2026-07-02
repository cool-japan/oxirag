//! [`LtrEngine`] — the trainable Learning-to-Rank engine.
//!
//! The engine is a thin, stateless façade over the training loop in
//! [`super::train`] and the scoring on [`super::types::LtrModel`]: it fits a
//! model from labelled preference data and then ranks candidate feature
//! vectors by the learned linear score. The learning lives entirely in the
//! fitted weights — contrast `cross_encoder`, whose combiner weights are fixed
//! defaults that are never trained.

use std::cmp::Ordering;

use super::train::train_model;
use super::types::{LtrConfig, LtrFeatureVector, LtrModel, LtrResult, LtrTrainingSet};

/// A stateless engine that trains [`LtrModel`]s and ranks candidates with them.
#[derive(Debug, Clone, Copy, Default)]
pub struct LtrEngine;

impl LtrEngine {
    /// Construct a new engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Fit a model to `training_set` under `config`.
    ///
    /// # Errors
    ///
    /// Propagates every validation and training error from the training loop:
    /// an invalid configuration, an empty training set, a feature-dimension
    /// mismatch, or an out-of-range label.
    pub fn train(&self, training_set: &LtrTrainingSet, config: &LtrConfig) -> LtrResult<LtrModel> {
        train_model(training_set, config)
    }

    /// Rank `candidates` by the model's score, highest first.
    ///
    /// Returns `(original_index, score)` pairs sorted in descending score
    /// order, with ties broken by ascending original index so the ordering is
    /// deterministic. Candidates are scored with [`LtrModel::score`], which
    /// takes the dot product over the common prefix and never panics; use
    /// [`LtrEngine::rank_checked`] to reject mis-dimensioned candidates.
    #[must_use]
    pub fn rank(&self, model: &LtrModel, candidates: &[LtrFeatureVector]) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(index, features)| (index, model.score(features)))
            .collect();
        scored.sort_by(|(index_a, score_a), (index_b, score_b)| {
            match score_b.partial_cmp(score_a) {
                Some(Ordering::Equal) | None => index_a.cmp(index_b),
                Some(order) => order,
            }
        });
        scored
    }

    /// Rank `candidates` after validating each against the model's feature
    /// dimension.
    ///
    /// # Errors
    ///
    /// Returns [`crate::learning_to_rank::LtrError::DimensionMismatch`] or
    /// [`crate::learning_to_rank::LtrError::NonFiniteFeature`] for the first
    /// candidate that fails [`LtrFeatureVector::validate`].
    pub fn rank_checked(
        &self,
        model: &LtrModel,
        candidates: &[LtrFeatureVector],
    ) -> LtrResult<Vec<(usize, f64)>> {
        for candidate in candidates {
            candidate.validate(model.feature_dim)?;
        }
        Ok(self.rank(model, candidates))
    }

    /// Rank `candidates` and return only the original indices, highest score
    /// first.
    #[must_use]
    pub fn rank_indices(&self, model: &LtrModel, candidates: &[LtrFeatureVector]) -> Vec<usize> {
        self.rank(model, candidates)
            .into_iter()
            .map(|(index, _)| index)
            .collect()
    }
}
