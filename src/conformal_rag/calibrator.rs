//! The split-conformal calibration and prediction-set engine.
//!
//! This module holds the mathematical heart of `conformal_rag`: the
//! finite-sample quantile that turns a calibration set of nonconformity scores
//! into a threshold with a proven marginal coverage guarantee, and the
//! prediction-set constructor that applies that threshold to fresh candidates.

use super::types::{
    ConformalConfig, ConformalError, ConformalMember, NonconformityScorer, PredictionKind,
    PredictionSet,
};

/// Computes the finite-sample conformal rank
/// `k = ceil((n + 1) * (1 - alpha))`.
///
/// This is the **1-indexed order statistic** of the calibration scores that
/// serves as the calibrated threshold: the `k`-th smallest calibration score.
/// The `(n + 1)` (rather than `n`) is the finite-sample correction that makes
/// the marginal coverage guarantee hold *exactly* — not merely asymptotically —
/// at any calibration size `n`. Using `n` here instead of `n + 1` (i.e. the
/// naive empirical `(1 - alpha)`-quantile) under-covers by roughly
/// `1 / (n + 1)`.
///
/// When `k > n` (which happens iff `alpha < 1 / (n + 1)`) there is no finite
/// order statistic large enough, and the calibrated threshold is conceptually
/// `+infinity` — the prediction set admits everything. Callers detect this via
/// the returned `k`: `k == n + 1` signals the include-everything regime.
///
/// # Float robustness
///
/// `(n + 1) * (1 - alpha)` is evaluated in `f64`, where a mathematically-integer
/// product can round to `integer + epsilon` and a naive `ceil` would then jump
/// to the *next* integer, silently over-covering by one order statistic. To
/// avoid that classic bug, a product within a small relative tolerance of an
/// integer is snapped to that integer before the `ceil` is applied. The
/// tolerance (`|q| * 1e-9`, floored at `1e-9`) sits far above the `~|q| * 2e-16`
/// floating-point rounding error yet far below the gap to the nearest genuine
/// non-integer target for any realistic `alpha`, so it never mis-snaps a real
/// value.
#[must_use]
pub(super) fn conformal_rank(n: usize, alpha: f64) -> usize {
    #[allow(clippy::cast_precision_loss)]
    let n1 = (n + 1) as f64;
    let q = n1 * (1.0 - alpha);
    let rounded = q.round();
    let eps = q.abs().max(1.0) * 1e-9;
    let corrected = if (q - rounded).abs() < eps {
        rounded
    } else {
        q
    };
    // `corrected.ceil()` is >= 1 for alpha in (0, 1) and <= n + 1, so the cast
    // is always in range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let k = corrected.ceil() as usize;
    k.max(1)
}

/// A calibrated split-conformal predictor.
///
/// Built by [`calibrate`](Self::calibrate) from a set of nonconformity scores
/// computed on held-out **known-correct** examples and a target miscoverage
/// rate `alpha`. It stores the calibrated threshold — the
/// `ceil((n + 1) * (1 - alpha))`-th smallest calibration score — and applies it
/// via [`predict_set`](Self::predict_set) to admit fresh candidates whose
/// nonconformity is at or below that threshold.
///
/// The resulting prediction sets carry the guarantee
/// `P(s(query, true_label) <= threshold) >= 1 - alpha`
/// whenever the calibration and test scores are exchangeable — no assumption on
/// the underlying scorer's distribution is required.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalCalibrator {
    config: ConformalConfig,
    /// The calibrated threshold; `f64::INFINITY` in the include-everything
    /// regime (`rank > calibration_size`).
    threshold: f64,
    /// The number of calibration scores the threshold was computed from.
    calibration_size: usize,
    /// The finite-sample rank `k = ceil((n + 1) * (1 - alpha))`.
    rank: usize,
}

impl ConformalCalibrator {
    /// Calibrates a threshold from a set of nonconformity `scores` computed on
    /// held-out **known-correct** examples.
    ///
    /// The scores must be the nonconformity of the *true* label for each
    /// calibration example (e.g. `s = 1 - relevance` of the correct passage).
    /// The calibrated threshold is the
    /// `ceil((n + 1) * (1 - config.alpha))`-th smallest of them (or
    /// `+infinity` when that rank exceeds `n`).
    ///
    /// # Errors
    ///
    /// - [`ConformalError::InvalidAlpha`] if `config.alpha` is not in `(0, 1)`.
    /// - [`ConformalError::EmptyCalibration`] if `scores` is empty.
    /// - [`ConformalError::NonFiniteScore`] if any score is `NaN` or infinite.
    pub fn calibrate(config: ConformalConfig, scores: &[f64]) -> Result<Self, ConformalError> {
        config.validate()?;
        if scores.is_empty() {
            return Err(ConformalError::EmptyCalibration);
        }
        for (index, &s) in scores.iter().enumerate() {
            if !s.is_finite() {
                return Err(ConformalError::NonFiniteScore { index });
            }
        }

        let n = scores.len();
        let rank = conformal_rank(n, config.alpha);

        let threshold = if rank > n {
            // alpha < 1 / (n + 1): no finite order statistic is large enough,
            // so the calibrated set admits everything.
            f64::INFINITY
        } else {
            let mut sorted = scores.to_vec();
            sorted.sort_by(f64::total_cmp);
            // `rank` is 1-indexed and in `1..=n` here; select the k-th smallest.
            sorted[rank - 1]
        };

        Ok(Self {
            config,
            threshold,
            calibration_size: n,
            rank,
        })
    }

    /// Calibrates from labelled calibration examples `(nonconformity, was_correct)`.
    ///
    /// Only the examples with `was_correct == true` participate in calibration:
    /// those are the ones whose nonconformity is the score of the *true* label,
    /// which is exactly the quantity the coverage guarantee is stated over.
    /// Examples with `was_correct == false` are ignored.
    ///
    /// # Errors
    ///
    /// Same as [`calibrate`](Self::calibrate); in particular
    /// [`ConformalError::EmptyCalibration`] if no example is `was_correct`.
    pub fn calibrate_labeled(
        config: ConformalConfig,
        examples: &[(f64, bool)],
    ) -> Result<Self, ConformalError> {
        let scores: Vec<f64> = examples
            .iter()
            .filter(|(_, was_correct)| *was_correct)
            .map(|(s, _)| *s)
            .collect();
        Self::calibrate(config, &scores)
    }

    /// The calibrated threshold. `f64::INFINITY` in the include-everything
    /// regime.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The configuration used to calibrate.
    #[must_use]
    pub fn config(&self) -> &ConformalConfig {
        &self.config
    }

    /// The number of calibration scores the threshold was computed from.
    #[must_use]
    pub fn calibration_size(&self) -> usize {
        self.calibration_size
    }

    /// The finite-sample rank `k = ceil((n + 1) * (1 - alpha))` used to pick the
    /// threshold order statistic. When `k == calibration_size + 1` the
    /// calibrator is in the include-everything ([`is_trivial`](Self::is_trivial))
    /// regime.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns `true` when the calibrated threshold admits *every* candidate
    /// (the `+infinity` regime), i.e. `alpha` is too small for this calibration
    /// size to ever exclude anything.
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.threshold.is_infinite()
    }

    /// Returns `true` if `nonconformity` is admitted by the calibrated
    /// threshold (i.e. `nonconformity <= threshold`).
    #[must_use]
    pub fn admits(&self, nonconformity: f64) -> bool {
        nonconformity <= self.threshold
    }

    /// Builds a [`PredictionSet`] from precomputed candidate nonconformity
    /// `scores`.
    ///
    /// Every candidate whose score is at or below the calibrated threshold is
    /// admitted; the members are returned sorted by ascending nonconformity.
    /// A non-finite candidate score is never admitted (`NaN <= x` is `false`).
    #[must_use]
    pub fn predict_set_from_scores(&self, scores: &[f64]) -> PredictionSet {
        let mut members: Vec<ConformalMember> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| self.admits(s))
            .map(|(index, &s)| ConformalMember {
                index,
                nonconformity: s,
                label: None,
            })
            .collect();
        members.sort_by(|a, b| a.nonconformity.total_cmp(&b.nonconformity));
        PredictionSet {
            kind: PredictionKind::from_len(members.len()),
            threshold: self.threshold,
            total_candidates: scores.len(),
            members,
        }
    }

    /// Builds a [`PredictionSet`] for `query` over `candidates`, computing each
    /// candidate's nonconformity with `scorer`.
    ///
    /// This is the primary entry point: pass any [`NonconformityScorer`] (a
    /// custom struct, a `Fn(&str, &str) -> f64` closure, or the default
    /// [`LexicalOverlapScorer`](super::types::LexicalOverlapScorer)) together
    /// with the candidate answers/passages. Admitted members carry their text in
    /// [`ConformalMember::label`].
    #[must_use]
    pub fn predict_set<C, S>(&self, scorer: &S, query: &str, candidates: &[C]) -> PredictionSet
    where
        C: AsRef<str>,
        S: NonconformityScorer + ?Sized,
    {
        let mut members: Vec<ConformalMember> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let text = candidate.as_ref();
                let s = scorer.nonconformity(query, text);
                if self.admits(s) {
                    Some(ConformalMember {
                        index,
                        nonconformity: s,
                        label: Some(text.to_string()),
                    })
                } else {
                    None
                }
            })
            .collect();
        members.sort_by(|a, b| a.nonconformity.total_cmp(&b.nonconformity));
        PredictionSet {
            kind: PredictionKind::from_len(members.len()),
            threshold: self.threshold,
            total_candidates: candidates.len(),
            members,
        }
    }
}
