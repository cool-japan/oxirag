//! Mondrian (group-conditional) split conformal prediction.
//!
//! A single global threshold guarantees *marginal* coverage — averaged over the
//! whole population — but can systematically over-cover easy queries and
//! under-cover hard ones. Mondrian conformal prediction restores balance by
//! calibrating a **separate** threshold per caller-supplied group key (for
//! example a query-difficulty band), so each group gets its own
//! `1 - alpha` coverage. Within each group the exact same finite-sample quantile
//! is applied, so the per-group guarantee is the same distribution-free
//! guarantee, now conditional on the group.
//!
//! Groups that are too small (below [`ConformalConfig::min_group_size`], or that
//! never appeared during calibration) fall back to a pooled **global**
//! calibrator so predictions are always well defined.

use std::collections::BTreeMap;

use super::calibrator::ConformalCalibrator;
use super::types::{ConformalConfig, ConformalError, NonconformityScorer, PredictionSet};

/// A group-conditional (Mondrian) split-conformal predictor.
///
/// Holds one [`ConformalCalibrator`] per group key plus a pooled global
/// calibrator used as a fallback for unseen or undersized groups.
#[derive(Debug, Clone, PartialEq)]
pub struct MondrianConformalCalibrator {
    config: ConformalConfig,
    /// Per-group calibrators, keyed by group. A [`BTreeMap`] keeps
    /// [`groups`](Self::groups) iteration deterministic.
    per_group: BTreeMap<String, ConformalCalibrator>,
    /// Pooled global calibrator over *all* calibration scores, used for groups
    /// that never appeared during calibration or that fell below
    /// [`ConformalConfig::min_group_size`].
    global: ConformalCalibrator,
}

impl MondrianConformalCalibrator {
    /// Calibrates one threshold per group from labelled `examples`
    /// `(group_key, nonconformity)`.
    ///
    /// Each example's `nonconformity` must be the score of the *true* label for
    /// that calibration example (as in [`ConformalCalibrator::calibrate`]). A
    /// group whose count reaches [`ConformalConfig::min_group_size`] earns its
    /// own threshold; smaller groups are served by the pooled global threshold
    /// (still calibrated, just less group-specific).
    ///
    /// # Errors
    ///
    /// - Any error from [`ConformalCalibrator::calibrate`] (invalid `alpha`,
    ///   empty calibration, non-finite score).
    /// - [`ConformalError::EmptyGroup`] can never actually be returned here (a
    ///   group present in `examples` always has at least one score); it exists
    ///   for completeness of the group-conditional API surface.
    pub fn calibrate(
        config: ConformalConfig,
        examples: &[(String, f64)],
    ) -> Result<Self, ConformalError> {
        config.validate()?;

        // Pool all scores for the global fallback and bucket by group.
        let mut all: Vec<f64> = Vec::with_capacity(examples.len());
        let mut buckets: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (group, score) in examples {
            all.push(*score);
            buckets.entry(group.clone()).or_default().push(*score);
        }

        let global = ConformalCalibrator::calibrate(config.clone(), &all)?;

        let mut per_group: BTreeMap<String, ConformalCalibrator> = BTreeMap::new();
        for (group, scores) in buckets {
            if scores.len() >= config.min_group_size {
                let calibrator = ConformalCalibrator::calibrate(config.clone(), &scores)?;
                per_group.insert(group, calibrator);
            }
            // Undersized groups intentionally omitted: they resolve to `global`.
        }

        Ok(Self {
            config,
            per_group,
            global,
        })
    }

    /// The configuration used to calibrate.
    #[must_use]
    pub fn config(&self) -> &ConformalConfig {
        &self.config
    }

    /// The pooled global calibrator used as the fallback for unseen or
    /// undersized groups.
    #[must_use]
    pub fn global(&self) -> &ConformalCalibrator {
        &self.global
    }

    /// The calibrator responsible for `group`: the group's own calibrator when
    /// it earned one, otherwise the pooled [`global`](Self::global) fallback.
    #[must_use]
    pub fn calibrator_for(&self, group: &str) -> &ConformalCalibrator {
        self.per_group.get(group).unwrap_or(&self.global)
    }

    /// The calibrated threshold for `group` (its own, or the global fallback's).
    #[must_use]
    pub fn threshold_for(&self, group: &str) -> f64 {
        self.calibrator_for(group).threshold()
    }

    /// Returns the group's *own* calibrator, or `None` if the group never
    /// earned one (and would be served by the global fallback).
    #[must_use]
    pub fn group_calibrator(&self, group: &str) -> Option<&ConformalCalibrator> {
        self.per_group.get(group)
    }

    /// Iterates over the group keys that earned their own calibrator, in sorted
    /// order.
    pub fn groups(&self) -> impl Iterator<Item = &str> {
        self.per_group.keys().map(String::as_str)
    }

    /// The number of groups that earned their own calibrator.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.per_group.len()
    }

    /// Builds a [`PredictionSet`] for `query` in `group`, using the group's
    /// calibrated threshold (or the global fallback) and `scorer`.
    #[must_use]
    pub fn predict_set<C, S>(
        &self,
        group: &str,
        scorer: &S,
        query: &str,
        candidates: &[C],
    ) -> PredictionSet
    where
        C: AsRef<str>,
        S: NonconformityScorer + ?Sized,
    {
        self.calibrator_for(group)
            .predict_set(scorer, query, candidates)
    }

    /// Builds a [`PredictionSet`] for `group` from precomputed candidate
    /// nonconformity `scores`.
    #[must_use]
    pub fn predict_set_from_scores(&self, group: &str, scores: &[f64]) -> PredictionSet {
        self.calibrator_for(group).predict_set_from_scores(scores)
    }
}
