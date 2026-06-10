//! Types for the `hallucination_detector` module.

use thiserror::Error;

// ── ClaimSupport ──────────────────────────────────────────────────────────────

/// A single claim extracted from an answer, with its source-grounding score.
#[derive(Debug, Clone)]
pub struct ClaimSupport {
    /// The claim text extracted from the answer.
    pub claim: String,
    /// Score in `[0.0, 1.0]` measuring lexical overlap with the best source.
    pub support_score: f32,
    /// IDs/descriptions of source documents that contributed support.
    pub supporting_sources: Vec<String>,
    /// Whether this claim is considered a hallucination (support below threshold).
    pub is_hallucination: bool,
}

impl ClaimSupport {
    /// Construct a `ClaimSupport` with explicit hallucination flag.
    #[must_use]
    pub fn new(
        claim: impl Into<String>,
        support_score: f32,
        supporting_sources: Vec<String>,
        is_hallucination: bool,
    ) -> Self {
        Self {
            claim: claim.into(),
            support_score,
            supporting_sources,
            is_hallucination,
        }
    }

    /// Human-readable support level for this claim.
    ///
    /// Returns `"strong"` (≥0.6), `"weak"` (≥0.2), or `"unsupported"` (<0.2).
    #[must_use]
    pub fn support_level(&self) -> &str {
        if self.support_score >= 0.6 {
            "strong"
        } else if self.support_score >= 0.2 {
            "weak"
        } else {
            "unsupported"
        }
    }
}

// ── HallucinationReport ───────────────────────────────────────────────────────

/// Aggregated hallucination-detection result for a full answer.
#[derive(Debug, Clone)]
pub struct HallucinationReport {
    /// Per-claim support analysis.
    pub claims: Vec<ClaimSupport>,
    /// Fraction of claims that are hallucinations (`0.0` if no claims).
    pub hallucination_rate: f32,
    /// Number of claims that are *not* hallucinations.
    pub supported_count: usize,
}

impl HallucinationReport {
    /// Build a `HallucinationReport` by aggregating the supplied claims.
    ///
    /// `hallucination_rate` = hallucination count / total count (0.0 when empty).
    #[must_use]
    pub fn new(claims: Vec<ClaimSupport>) -> Self {
        let total = claims.len();
        let hallucination_count = claims.iter().filter(|c| c.is_hallucination).count();
        let supported_count = total - hallucination_count;
        #[allow(clippy::cast_precision_loss)]
        let hallucination_rate = if total == 0 {
            0.0_f32
        } else {
            hallucination_count as f32 / total as f32
        };
        Self {
            claims,
            hallucination_rate,
            supported_count,
        }
    }

    /// Returns `true` when the answer is effectively clean (rate < 5 %).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.hallucination_rate < 0.05
    }

    /// Count of claims flagged as hallucinations.
    #[must_use]
    pub fn hallucination_count(&self) -> usize {
        self.claims.iter().filter(|c| c.is_hallucination).count()
    }
}

// ── HallucinationConfig ───────────────────────────────────────────────────────

/// Configuration for the hallucination detector.
#[derive(Debug, Clone)]
pub struct HallucinationConfig {
    /// Minimum Jaccard overlap required to consider a claim supported.
    ///
    /// Default: `0.2`.
    pub min_support: f32,
    /// When `true`, the answer is split into individual sentences before scoring.
    ///
    /// Default: `true`.
    pub sentence_level: bool,
}

impl Default for HallucinationConfig {
    fn default() -> Self {
        Self {
            min_support: 0.2,
            sentence_level: true,
        }
    }
}

impl HallucinationConfig {
    /// Create a new `HallucinationConfig` with the given `min_support` threshold.
    #[must_use]
    pub fn new(min_support: f32, sentence_level: bool) -> Self {
        Self {
            min_support,
            sentence_level,
        }
    }

    /// Set the `min_support` threshold.
    #[must_use]
    pub fn with_min_support(mut self, min_support: f32) -> Self {
        self.min_support = min_support;
        self
    }

    /// Enable or disable sentence-level splitting.
    #[must_use]
    pub fn with_sentence_level(mut self, sentence_level: bool) -> Self {
        self.sentence_level = sentence_level;
        self
    }
}

// ── HallucinationDetector ─────────────────────────────────────────────────────

/// Re-exported from `detector` — see [`crate::hallucination_detector::detector`].
pub use crate::hallucination_detector::detector::HallucinationDetector;

// ── HallucinationError ────────────────────────────────────────────────────────

/// Errors produced by [`HallucinationDetector`].
#[derive(Debug, Error)]
pub enum HallucinationError {
    /// The answer string was empty.
    #[error("Answer must not be empty")]
    EmptyAnswer,
    /// No source documents were provided.
    #[error("At least one source document is required")]
    EmptySources,
}
