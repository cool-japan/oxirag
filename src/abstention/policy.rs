//! The answer-or-abstain policy and risk–coverage analysis.
//!
//! [`AbstentionPolicy`] makes the per-query selective-prediction decision and
//! characterises a whole labelled dataset with a risk–coverage tradeoff.
//!
//! Per-query policy (see [`AbstentionPolicy::assess`]):
//!
//! ```text
//! combined = support_weight * support + (1 - support_weight) * confidence
//! decision = Answer  iff  combined >= confidence_threshold  AND  support >= min_support
//!            Abstain otherwise
//! ```
//!
//! Dataset analysis (see [`AbstentionPolicy::risk_coverage`]) treats each entry
//! as a `(combined_confidence, is_correct)` pair: a query is *answered* when its
//! confidence is at or above the operating threshold, *coverage* is the fraction
//! answered, and *risk* is the error rate among the answered.

use super::types::{
    AbstentionAssessment, AbstentionConfig, AbstentionDecision, AbstentionError, RiskCoverage,
};

// ── AbstentionPolicy ──────────────────────────────────────────────────────────

/// A selective-prediction policy: decide whether to answer or abstain, and
/// analyse the risk–coverage tradeoff of a dataset.
///
/// The policy is a pure, deterministic function of its [`AbstentionConfig`]; it
/// holds no mutable state and is cheap to clone.
#[derive(Debug, Clone)]
pub struct AbstentionPolicy {
    /// The configuration governing thresholds and signal blending.
    pub config: AbstentionConfig,
}

impl AbstentionPolicy {
    /// Creates a new policy from a configuration.
    #[must_use]
    pub fn new(config: AbstentionConfig) -> Self {
        Self { config }
    }

    /// Returns the [`support_weight`](AbstentionConfig::support_weight) clamped
    /// into `[0.0, 1.0]` for use in the blend.
    fn support_weight(&self) -> f32 {
        self.config.support_weight.clamp(0.0, 1.0)
    }

    /// Blends a (clamped) confidence and support into the `combined` score.
    ///
    /// `combined = support_weight * support + (1 - support_weight) * confidence`.
    fn combine(&self, confidence: f32, support: f32) -> f32 {
        let w = self.support_weight();
        (w * support + (1.0 - w) * confidence).clamp(0.0, 1.0)
    }

    /// Assesses a single query and returns a full [`AbstentionAssessment`].
    ///
    /// Both inputs are clamped to `[0.0, 1.0]` first. The blended `combined`
    /// score is
    /// `support_weight * support + (1 - support_weight) * confidence`, and the
    /// decision is [`AbstentionDecision::Answer`] **iff** `combined` is at or
    /// above [`confidence_threshold`](AbstentionConfig::confidence_threshold)
    /// **and** `support` is at or above
    /// [`min_support`](AbstentionConfig::min_support); otherwise it is
    /// [`AbstentionDecision::Abstain`]. The returned `reason` names the gate that
    /// drove the verdict.
    #[must_use]
    pub fn assess(&self, answer_confidence: f32, retrieval_support: f32) -> AbstentionAssessment {
        let confidence = answer_confidence.clamp(0.0, 1.0);
        let support = retrieval_support.clamp(0.0, 1.0);
        let combined = self.combine(confidence, support);

        let support_ok = support >= self.config.min_support;
        let confidence_ok = combined >= self.config.confidence_threshold;

        let (decision, reason) = if !support_ok {
            (
                AbstentionDecision::Abstain,
                format!(
                    "abstain: support {support:.3} below min_support {:.3}",
                    self.config.min_support
                ),
            )
        } else if !confidence_ok {
            (
                AbstentionDecision::Abstain,
                format!(
                    "abstain: combined {combined:.3} below confidence_threshold {:.3}",
                    self.config.confidence_threshold
                ),
            )
        } else {
            (
                AbstentionDecision::Answer,
                format!(
                    "answer: combined {combined:.3} >= {:.3} and support {support:.3} >= {:.3}",
                    self.config.confidence_threshold, self.config.min_support
                ),
            )
        };

        AbstentionAssessment {
            decision,
            confidence,
            support,
            combined,
            reason,
        }
    }

    /// Returns just the [`AbstentionDecision`] for a query.
    ///
    /// A convenience wrapper over [`assess`](Self::assess) when the scores and
    /// reason are not needed.
    #[must_use]
    pub fn decide(&self, confidence: f32, support: f32) -> AbstentionDecision {
        self.assess(confidence, support).decision
    }

    /// Computes the risk–coverage operating point at the configured
    /// [`confidence_threshold`](AbstentionConfig::confidence_threshold).
    ///
    /// `scored` is a slice of `(combined_confidence, is_correct)` pairs. A query
    /// is *answered* when its (clamped) confidence is at or above the threshold.
    /// *Coverage* is `answered / total`; *risk* is `wrong / answered` (the error
    /// rate among the answered), or `0.0` when nothing is answered.
    ///
    /// # Errors
    ///
    /// Returns [`AbstentionError::EmptyData`] if `scored` is empty.
    pub fn risk_coverage(&self, scored: &[(f32, bool)]) -> Result<RiskCoverage, AbstentionError> {
        self.risk_coverage_at(scored, self.config.confidence_threshold)
    }

    /// Computes the risk–coverage operating point at an explicit `threshold`.
    ///
    /// Identical to [`risk_coverage`](Self::risk_coverage) but answers a query
    /// when its (clamped) confidence is at or above `threshold` rather than the
    /// configured one.
    ///
    /// # Errors
    ///
    /// Returns [`AbstentionError::EmptyData`] if `scored` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn risk_coverage_at(
        &self,
        scored: &[(f32, bool)],
        threshold: f32,
    ) -> Result<RiskCoverage, AbstentionError> {
        if scored.is_empty() {
            return Err(AbstentionError::EmptyData);
        }
        let total = scored.len();
        let mut answered = 0usize;
        let mut wrong = 0usize;
        for &(conf, correct) in scored {
            if conf.clamp(0.0, 1.0) >= threshold {
                answered += 1;
                if !correct {
                    wrong += 1;
                }
            }
        }

        let coverage = answered as f32 / total as f32;
        let risk = if answered == 0 {
            0.0
        } else {
            wrong as f32 / answered as f32
        };

        Ok(RiskCoverage {
            coverage,
            risk,
            answered,
            total,
        })
    }

    /// Sweeps the answer threshold across `[0.0, 1.0]` and returns one
    /// [`RiskCoverage`] point per step.
    ///
    /// The threshold takes `steps` evenly spaced values starting at `0.0`. With
    /// `steps == 1` only the single point at threshold `0.0` is produced (full
    /// coverage). `steps` is clamped to at least one. Because raising the
    /// threshold can only drop queries from the answered set, the resulting
    /// [`coverage`](RiskCoverage::coverage) is monotonically non-increasing along
    /// the returned vector.
    ///
    /// Each point is computed against the full dataset, so an empty `scored`
    /// slice yields an empty vector rather than an error (every point would
    /// otherwise be [`AbstentionError::EmptyData`]).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn risk_coverage_curve(&self, scored: &[(f32, bool)], steps: usize) -> Vec<RiskCoverage> {
        if scored.is_empty() {
            return Vec::new();
        }
        let steps = steps.max(1);
        let mut curve = Vec::with_capacity(steps);
        for i in 0..steps {
            // Thresholds 0.0, 1/(steps-1), ..., 1.0 (or just 0.0 when steps == 1).
            let threshold = if steps == 1 {
                0.0
            } else {
                i as f32 / (steps - 1) as f32
            };
            // The dataset is non-empty here, so this never returns EmptyData.
            if let Ok(point) = self.risk_coverage_at(scored, threshold) {
                curve.push(point);
            }
        }
        curve
    }
}
