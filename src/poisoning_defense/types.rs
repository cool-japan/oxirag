//! Types for the `poisoning_defense` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── PoisonConfig ────────────────────────────────────────────────────────────────

/// Configuration for the corpus-poisoning / adversarial-passage detector.
///
/// Each retrieved passage is scored on three signals — keyword **stuffing**
/// (query-term density), lexical **diversity** (type/token ratio) and corpus
/// **anomaly** (disagreement with the retrieved-set consensus) — and the three
/// are blended into a single poison `risk`. A passage is flagged when its risk
/// crosses [`PoisonConfig::risk_threshold`], or when either individual hard
/// signal trips: stuffing at or above [`PoisonConfig::stuffing_threshold`], or
/// diversity at or below [`PoisonConfig::diversity_threshold`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoisonConfig {
    /// Query-term density at or above which a passage is treated as keyword
    /// stuffing, regardless of the blended risk.
    ///
    /// Defaults to `0.35`.
    pub stuffing_threshold: f32,
    /// Type/token ratio at or below which a passage is treated as abnormally
    /// repetitive (low lexical diversity), regardless of the blended risk.
    ///
    /// Defaults to `0.4`.
    pub diversity_threshold: f32,
    /// Blended poison-risk score at or above which a passage is flagged.
    ///
    /// Defaults to `0.5`.
    pub risk_threshold: f32,
}

impl Default for PoisonConfig {
    fn default() -> Self {
        Self {
            stuffing_threshold: 0.35,
            diversity_threshold: 0.4,
            risk_threshold: 0.5,
        }
    }
}

impl PoisonConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the query-term-density threshold above which a passage is stuffing.
    #[must_use]
    pub fn with_stuffing_threshold(mut self, v: f32) -> Self {
        self.stuffing_threshold = v;
        self
    }

    /// Set the type/token-ratio threshold below which a passage is low-diversity.
    #[must_use]
    pub fn with_diversity_threshold(mut self, v: f32) -> Self {
        self.diversity_threshold = v;
        self
    }

    /// Set the blended-risk threshold above which a passage is flagged poisoned.
    #[must_use]
    pub fn with_risk_threshold(mut self, v: f32) -> Self {
        self.risk_threshold = v;
        self
    }
}

// ── PoisonAssessment ────────────────────────────────────────────────────────────

/// Per-passage poison assessment produced by
/// [`crate::poisoning_defense::PoisoningDetector::assess`].
///
/// The three signal scores and the blended `risk` are all in `[0, 1]`, where a
/// higher value indicates a stronger poisoning signal — with the deliberate
/// exception of `diversity_score`, where a *lower* value is the suspicious one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoisonAssessment {
    /// Index of the passage within the input slice.
    pub index: usize,
    /// Query-term density in the passage (keyword-stuffing signal); higher is
    /// more suspicious.
    pub stuffing_score: f32,
    /// Type/token ratio of the passage (lexical-diversity signal); *lower* is
    /// more suspicious.
    pub diversity_score: f32,
    /// Outlier score versus the corpus consensus (anomaly signal); higher is
    /// more suspicious.
    pub anomaly_score: f32,
    /// Blended poison-risk score in `[0, 1]`; higher is more suspicious.
    pub risk: f32,
    /// `true` when the passage is flagged as adversarial / poisoned.
    pub is_poisoned: bool,
}

// ── PoisonError ─────────────────────────────────────────────────────────────────

/// Errors from the `poisoning_defense` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PoisonError {
    /// The query string was empty (or whitespace only).
    #[error("query must not be empty")]
    EmptyQuery,
}
