//! Types for the `ragchecker` module.
//!
//! Defines the configuration, the six diagnostic metrics, the per-check result,
//! and the error type used by the claim-level `RAGChecker` diagnostics.

use thiserror::Error;

// ── RagCheckerConfig ──────────────────────────────────────────────────────────

/// Configuration for the claim-level `RAGChecker` diagnostics.
///
/// The single knob is the lexical entailment threshold: the minimum fraction of
/// a claim's tokens that must appear in a piece of text for that text to be said
/// to *entail* the claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagCheckerConfig {
    /// Minimum fraction of a claim's tokens that must appear in a text for the
    /// text to entail the claim.
    ///
    /// Must lie in `[0.0, 1.0]`. Defaults to `0.5`.
    pub entailment_threshold: f32,
}

impl Default for RagCheckerConfig {
    fn default() -> Self {
        Self {
            entailment_threshold: 0.5,
        }
    }
}

impl RagCheckerConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the lexical entailment threshold.
    ///
    /// The supplied value is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_entailment_threshold(mut self, entailment_threshold: f32) -> Self {
        self.entailment_threshold = entailment_threshold.clamp(0.0, 1.0);
        self
    }
}

// ── RagCheckerMetrics ─────────────────────────────────────────────────────────

/// The six claim-level diagnostic metrics produced by `RAGChecker`.
///
/// The metrics are partitioned into *retriever* diagnostics (`claim_recall`,
/// `context_precision`) and *generator* diagnostics (`faithfulness`,
/// `hallucination_rate`, `correctness`, `noise_sensitivity`). Every field lies
/// in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagCheckerMetrics {
    /// Retriever: fraction of ground-truth claims entailed by the retrieved context.
    pub claim_recall: f32,
    /// Retriever: fraction of retrieved passages that entail at least one ground-truth claim.
    pub context_precision: f32,
    /// Generator: fraction of response claims entailed by the context.
    pub faithfulness: f32,
    /// Generator: fraction of response claims entailed by neither context nor ground truth.
    pub hallucination_rate: f32,
    /// Generator: fraction of response claims entailed by the ground truth.
    pub correctness: f32,
    /// Generator: fraction of response claims entailed by the context but not the ground truth.
    pub noise_sensitivity: f32,
}

// ── RagCheckResult ────────────────────────────────────────────────────────────

/// The outcome of a single `RAGChecker` run.
///
/// Bundles the computed [`RagCheckerMetrics`] together with the decomposed
/// response and ground-truth claims, so callers can inspect exactly which atomic
/// claims drove each score.
#[derive(Debug, Clone, PartialEq)]
pub struct RagCheckResult {
    /// The six diagnostic metrics.
    pub metrics: RagCheckerMetrics,
    /// The atomic claims decomposed from the response.
    pub response_claims: Vec<String>,
    /// The atomic claims decomposed from the ground-truth answer.
    pub gt_claims: Vec<String>,
}

// ── RagCheckerError ───────────────────────────────────────────────────────────

/// Errors from the `ragchecker` module.
#[derive(Debug, Error)]
pub enum RagCheckerError {
    /// The response string was empty.
    #[error("response must not be empty")]
    EmptyResponse,
    /// The ground-truth string was empty.
    #[error("ground truth must not be empty")]
    EmptyGroundTruth,
}
