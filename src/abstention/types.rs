//! Types for the `abstention` module.
//!
//! These types describe *selective prediction* over a single query (answer or
//! refuse) and the *risk–coverage* characterisation of a whole labelled dataset.
//! They are **distinct** from the critique/reflection types of `self_rag` and the
//! depth-routing types of `adaptive_rag`: here the only decision is whether the
//! system should commit to an answer at all, given a confidence and a retrieval
//! support score, in exchange for a measurable risk–coverage tradeoff.

use thiserror::Error;

// ── AbstentionDecision ────────────────────────────────────────────────────────

/// The outcome of a selective-prediction decision: commit to an answer, or
/// refuse.
///
/// This is the binary "answer-or-refuse" verdict at the heart of selective
/// prediction. It carries no scores itself; the scores that produced it live on
/// the surrounding [`AbstentionAssessment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbstentionDecision {
    /// The system is confident enough (and sufficiently grounded) to answer.
    Answer,
    /// The system refuses to answer (abstains) rather than risk being wrong.
    Abstain,
}

impl AbstentionDecision {
    /// Returns a stable, lower-case string identifier for the decision.
    ///
    /// `"answer"` for [`Answer`](AbstentionDecision::Answer) and `"abstain"` for
    /// [`Abstain`](AbstentionDecision::Abstain).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Answer => "answer",
            Self::Abstain => "abstain",
        }
    }

    /// Returns `true` if this decision is to [`Answer`](AbstentionDecision::Answer).
    #[must_use]
    pub fn is_answer(&self) -> bool {
        matches!(self, Self::Answer)
    }

    /// Returns `true` if this decision is to [`Abstain`](AbstentionDecision::Abstain).
    #[must_use]
    pub fn is_abstain(&self) -> bool {
        matches!(self, Self::Abstain)
    }
}

impl std::fmt::Display for AbstentionDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── AbstentionConfig ──────────────────────────────────────────────────────────

/// Configuration governing the answer-or-abstain policy.
///
/// A query is answered only when its blended score clears
/// [`confidence_threshold`](Self::confidence_threshold) *and* its retrieval
/// support clears [`min_support`](Self::min_support); otherwise it is abstained
/// on. The blend mixes support and confidence using
/// [`support_weight`](Self::support_weight):
///
/// ```text
/// combined = support_weight * support + (1 - support_weight) * confidence
/// ```
///
/// All three fields are clamped to sensible ranges when the policy uses them, so
/// out-of-range builder inputs never panic.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstentionConfig {
    /// Minimum blended `combined` score required to answer. Default `0.5`.
    pub confidence_threshold: f32,
    /// Minimum retrieval support required to answer, regardless of confidence.
    ///
    /// Default `0.2`. This is a hard gate: a query whose support falls below it
    /// is abstained on even when its confidence is maximal.
    pub min_support: f32,
    /// Weight of the retrieval-support signal in the blended score, in
    /// `[0.0, 1.0]`. Default `0.4`. The confidence signal receives the
    /// complementary weight `1 - support_weight`.
    pub support_weight: f32,
}

impl Default for AbstentionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            min_support: 0.2,
            support_weight: 0.4,
        }
    }
}

impl AbstentionConfig {
    /// Creates a new [`AbstentionConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum blended `combined` score required to answer.
    #[must_use]
    pub fn with_confidence_threshold(mut self, confidence_threshold: f32) -> Self {
        self.confidence_threshold = confidence_threshold;
        self
    }

    /// Sets the minimum retrieval support required to answer.
    #[must_use]
    pub fn with_min_support(mut self, min_support: f32) -> Self {
        self.min_support = min_support;
        self
    }

    /// Sets the weight of the retrieval-support signal in the blended score.
    #[must_use]
    pub fn with_support_weight(mut self, support_weight: f32) -> Self {
        self.support_weight = support_weight;
        self
    }
}

// ── AbstentionAssessment ──────────────────────────────────────────────────────

/// The full result of a single answer-or-abstain assessment.
///
/// Produced by [`AbstentionPolicy::assess`]. It records the resolved
/// [`decision`](Self::decision), the (clamped) input signals, the blended
/// [`combined`](Self::combined) score the decision was made on, and a short,
/// human-readable [`reason`](Self::reason) explaining the verdict.
///
/// [`AbstentionPolicy::assess`]: super::policy::AbstentionPolicy::assess
#[derive(Debug, Clone, PartialEq)]
pub struct AbstentionAssessment {
    /// The resolved answer-or-abstain decision.
    pub decision: AbstentionDecision,
    /// The answer confidence used, clamped to `[0.0, 1.0]`.
    pub confidence: f32,
    /// The retrieval support used, clamped to `[0.0, 1.0]`.
    pub support: f32,
    /// The blended score the decision was made on, in `[0.0, 1.0]`.
    pub combined: f32,
    /// A short explanation of why the decision was reached.
    pub reason: String,
}

impl AbstentionAssessment {
    /// Returns `true` if this assessment decided to answer.
    #[must_use]
    pub fn is_answer(&self) -> bool {
        self.decision.is_answer()
    }

    /// Returns `true` if this assessment decided to abstain.
    #[must_use]
    pub fn is_abstain(&self) -> bool {
        self.decision.is_abstain()
    }
}

// ── RiskCoverage ──────────────────────────────────────────────────────────────

/// A single point on the risk–coverage curve.
///
/// *Coverage* is the fraction of queries the policy chose to answer; *risk* is
/// the error rate **among the answered queries** (the empirical selective risk).
/// A good selective predictor trades a small drop in coverage for a larger drop
/// in risk: abstaining on the queries it is least sure of.
///
/// When no query is answered, [`coverage`](Self::coverage) is `0.0` and
/// [`risk`](Self::risk) is defined to be `0.0` (there are no answered queries to
/// be wrong about).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskCoverage {
    /// Fraction of queries answered, in `[0.0, 1.0]` (`answered / total`).
    pub coverage: f32,
    /// Error rate among answered queries, in `[0.0, 1.0]` (`wrong / answered`).
    pub risk: f32,
    /// Number of queries answered at this operating point.
    pub answered: usize,
    /// Total number of queries in the dataset.
    pub total: usize,
}

impl RiskCoverage {
    /// Number of answered queries that were wrong, `risk * answered` rounded to
    /// the nearest integer (reconstructed exactly from the stored counts).
    ///
    /// Returns `0` when nothing was answered.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn wrong(&self) -> usize {
        if self.answered == 0 {
            0
        } else {
            (self.risk * self.answered as f32).round() as usize
        }
    }
}

// ── AbstentionError ───────────────────────────────────────────────────────────

/// Errors produced by the `abstention` module.
#[derive(Debug, Error)]
pub enum AbstentionError {
    /// The supplied scored dataset was empty.
    #[error("dataset is empty")]
    EmptyData,
}
