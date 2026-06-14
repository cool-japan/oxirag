//! Types for the `fact_check` module.

use crate::types::DocumentId;
use thiserror::Error;

// ── Verdict ─────────────────────────────────────────────────────────────────────

/// A FEVER-style three-way verdict for a claim against a corpus of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The evidence supports the claim.
    Supports,
    /// The evidence refutes (contradicts) the claim.
    Refutes,
    /// There is not enough relevant evidence to decide.
    NotEnoughInfo,
}

impl Verdict {
    /// Return a stable string representation of the verdict.
    ///
    /// Uses the canonical FEVER labels: `"SUPPORTS"`, `"REFUTES"`, and
    /// `"NOT_ENOUGH_INFO"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Supports => "SUPPORTS",
            Self::Refutes => "REFUTES",
            Self::NotEnoughInfo => "NOT_ENOUGH_INFO",
        }
    }
}

// ── Evidence ────────────────────────────────────────────────────────────────────

/// A single piece of retrieved evidence for a claim.
///
/// Evidence is a sentence drawn from a source [`crate::types::Document`], paired
/// with its lexical-overlap `score` against the claim and a `contradicts` flag
/// indicating whether a contradiction signal (negation mismatch or numeric
/// mismatch) was detected relative to the claim.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// The identifier of the document this sentence was drawn from.
    pub doc_id: DocumentId,
    /// The evidence sentence text.
    pub sentence: String,
    /// Token-overlap score against the claim, in `[0.0, 1.0]`.
    pub score: f32,
    /// Whether this evidence contradicts the claim (negation or numeric mismatch).
    pub contradicts: bool,
}

impl Evidence {
    /// Construct a new [`Evidence`] item.
    #[must_use]
    pub fn new(
        doc_id: DocumentId,
        sentence: impl Into<String>,
        score: f32,
        contradicts: bool,
    ) -> Self {
        Self {
            doc_id,
            sentence: sentence.into(),
            score,
            contradicts,
        }
    }
}

// ── FactCheckConfig ─────────────────────────────────────────────────────────────

/// Configuration for the [`crate::fact_check::FactChecker`].
#[derive(Debug, Clone)]
pub struct FactCheckConfig {
    /// Number of top-scoring evidence sentences to retain.
    ///
    /// Default: `3`.
    pub evidence_top_k: usize,
    /// Minimum top-evidence overlap required to return [`Verdict::Supports`].
    ///
    /// Default: `0.4`.
    pub support_threshold: f32,
    /// Overlap floor below which the verdict is [`Verdict::NotEnoughInfo`].
    ///
    /// Default: `0.15`.
    pub nei_threshold: f32,
}

impl Default for FactCheckConfig {
    fn default() -> Self {
        Self {
            evidence_top_k: 3,
            support_threshold: 0.4,
            nei_threshold: 0.15,
        }
    }
}

impl FactCheckConfig {
    /// Create a new [`FactCheckConfig`] with default values.
    ///
    /// Equivalent to [`FactCheckConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of top-scoring evidence sentences to retain.
    #[must_use]
    pub fn with_evidence_top_k(mut self, evidence_top_k: usize) -> Self {
        self.evidence_top_k = evidence_top_k;
        self
    }

    /// Set the minimum overlap required to support a claim.
    #[must_use]
    pub fn with_support_threshold(mut self, support_threshold: f32) -> Self {
        self.support_threshold = support_threshold;
        self
    }

    /// Set the overlap floor below which the verdict is not-enough-info.
    #[must_use]
    pub fn with_nei_threshold(mut self, nei_threshold: f32) -> Self {
        self.nei_threshold = nei_threshold;
        self
    }
}

// ── FactCheckResult ─────────────────────────────────────────────────────────────

/// The outcome of verifying a single claim against a corpus.
#[derive(Debug, Clone)]
pub struct FactCheckResult {
    /// The assigned three-way verdict.
    pub verdict: Verdict,
    /// Confidence in the verdict, in `[0.0, 1.0]`.
    pub confidence: f32,
    /// The supporting (or refuting) evidence, ordered by descending score.
    pub evidence: Vec<Evidence>,
    /// A human-readable rationale for the verdict.
    pub rationale: String,
}

impl FactCheckResult {
    /// Construct a new [`FactCheckResult`].
    #[must_use]
    pub fn new(
        verdict: Verdict,
        confidence: f32,
        evidence: Vec<Evidence>,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            verdict,
            confidence,
            evidence,
            rationale: rationale.into(),
        }
    }

    /// Whether the verdict is [`Verdict::Supports`].
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.verdict == Verdict::Supports
    }

    /// Whether the verdict is [`Verdict::Refutes`].
    #[must_use]
    pub fn is_refuted(&self) -> bool {
        self.verdict == Verdict::Refutes
    }
}

// ── FactChecker ─────────────────────────────────────────────────────────────────

/// Re-exported from `checker` — see [`crate::fact_check::checker`].
pub use crate::fact_check::checker::FactChecker;

// ── FactCheckError ──────────────────────────────────────────────────────────────

/// Errors produced by [`FactChecker`].
#[derive(Debug, Error)]
pub enum FactCheckError {
    /// The claim string was empty.
    #[error("claim must not be empty")]
    EmptyClaim,
    /// No evidence documents were provided.
    #[error("evidence corpus is empty")]
    EmptyCorpus,
}
