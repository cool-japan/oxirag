//! Types for the `citation_verification` module.
//!
//! Defines configuration, error, input, result, and aggregate report types
//! used by the citation-level grounding heuristic.

use thiserror::Error;

// ── CitationConfig ─────────────────────────────────────────────────────────────

/// Configuration knobs for the citation-verification heuristic.
///
/// The combined grounding score is:
///
/// ```text
/// score = token_overlap_weight × token_overlap_ratio
///       + substring_weight     × ngram_score
/// ```
///
/// where `token_overlap_ratio` is the fraction of the claim's distinct tokens
/// that also appear in the source, and `ngram_score` is `1.0` when at least one
/// 3-gram from the claim matches a contiguous substring of the source (after
/// normalisation), or `0.0` otherwise.
///
/// A citation is declared *grounded* when `score ≥ min_overlap_ratio`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CitationConfig {
    /// Minimum combined score for a citation to be considered grounded.
    ///
    /// Lies in `[0.0, 1.0]`. Defaults to `0.5`.
    pub min_overlap_ratio: f32,
    /// Weight applied to the n-gram substring-match signal.
    ///
    /// Defaults to `0.3`.
    pub substring_weight: f32,
    /// Weight applied to the token overlap ratio signal.
    ///
    /// Defaults to `0.7`.
    pub token_overlap_weight: f32,
}

impl Default for CitationConfig {
    fn default() -> Self {
        Self {
            min_overlap_ratio: 0.5,
            substring_weight: 0.3,
            token_overlap_weight: 0.7,
        }
    }
}

impl CitationConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum overlap ratio threshold (clamped to `[0.0, 1.0]`).
    #[must_use]
    pub fn with_min_overlap_ratio(mut self, ratio: f32) -> Self {
        self.min_overlap_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set the substring (n-gram) weight.
    #[must_use]
    pub fn with_substring_weight(mut self, weight: f32) -> Self {
        self.substring_weight = weight;
        self
    }

    /// Set the token overlap weight.
    #[must_use]
    pub fn with_token_overlap_weight(mut self, weight: f32) -> Self {
        self.token_overlap_weight = weight;
        self
    }
}

// ── CitationError ──────────────────────────────────────────────────────────────

/// Errors produced by the citation-verification module.
#[derive(Debug, Error)]
pub enum CitationError {
    /// The claim string was empty (or contained only whitespace).
    #[error("claim must not be empty")]
    EmptyClaim,
    /// The source-passage string was empty (or contained only whitespace).
    #[error("source passage must not be empty")]
    EmptySource,
    /// The verification step failed for an internal reason.
    #[error("verification failed: {0}")]
    VerificationFailed(String),
}

// ── CitationCheck ──────────────────────────────────────────────────────────────

/// Input for a single citation grounding check.
///
/// Pairs a *claim* (a sentence or quoted span from the generated answer) with
/// the *source passage* that is supposed to support it.
#[derive(Debug, Clone, PartialEq)]
pub struct CitationCheck {
    /// The claim or quoted span to verify.
    pub claim: String,
    /// The retrieved passage that should entail the claim.
    pub source_passage: String,
}

impl CitationCheck {
    /// Create a new [`CitationCheck`] from any string-like inputs.
    #[must_use]
    pub fn new(claim: impl Into<String>, source_passage: impl Into<String>) -> Self {
        Self {
            claim: claim.into(),
            source_passage: source_passage.into(),
        }
    }
}

// ── VerifiedCitation ───────────────────────────────────────────────────────────

/// Result of verifying a single citation against its source passage.
///
/// Carries the original [`CitationCheck`] together with the computed grounding
/// score, a boolean verdict, and the n-gram spans that were found in the source.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedCitation {
    /// The input check that was verified.
    pub check: CitationCheck,
    /// Combined grounding score in `[0.0, 1.0]`.
    pub grounding_score: f32,
    /// Whether the citation is considered grounded (score ≥ `min_overlap_ratio`).
    pub is_grounded: bool,
    /// The 3-gram token sequences from the claim that were matched in the source.
    pub matched_spans: Vec<String>,
}

// ── CitationReport ─────────────────────────────────────────────────────────────

/// Aggregate report produced by [`crate::citation_verification::CitationVerifier::verify_batch`].
///
/// Bundles all per-citation results together with summary statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct CitationReport {
    /// The individual verified citations (invalid inputs are omitted).
    pub verified: Vec<VerifiedCitation>,
    /// Mean grounding score across all verified citations (`0.0` when empty).
    pub overall_score: f32,
    /// Number of citations whose `is_grounded` flag is `true`.
    pub grounded_count: usize,
    /// Total number of successfully verified citations.
    pub total_count: usize,
}
