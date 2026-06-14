//! Types for the `faithfulness_eval` module.

use thiserror::Error;

// ── FaithfulnessConfig ────────────────────────────────────────────────────────

/// Configuration for [`FaithfulnessEvaluator`](super::evaluator::FaithfulnessEvaluator).
///
/// Controls the entailment threshold and the characters used to split an answer
/// into individual atomic claims.
#[derive(Debug, Clone, PartialEq)]
pub struct FaithfulnessConfig {
    /// Minimum coverage score for a claim to be considered entailed by the context.
    ///
    /// A claim is entailed when the fraction of its unique vocabulary tokens that
    /// appear in the best-matching context passage is at least this value. Default
    /// `0.5`.
    pub min_entailment_score: f32,
    /// Strings (typically single characters) used as sentence-boundary markers
    /// when splitting an answer into individual claims.
    ///
    /// Each character of every entry in this list is treated as a split point.
    /// Default: `[".", "!", "?", ";"]`.
    pub claim_split_tokens: Vec<String>,
}

impl Default for FaithfulnessConfig {
    fn default() -> Self {
        Self {
            min_entailment_score: 0.5,
            claim_split_tokens: vec![
                ".".to_owned(),
                "!".to_owned(),
                "?".to_owned(),
                ";".to_owned(),
            ],
        }
    }
}

impl FaithfulnessConfig {
    /// Create a configuration with default values (`min_entailment_score = 0.5`,
    /// split tokens `[".", "!", "?", ";"]`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the minimum entailment score threshold.
    #[must_use]
    pub fn with_min_entailment_score(mut self, score: f32) -> Self {
        self.min_entailment_score = score;
        self
    }

    /// Override the set of claim-splitting tokens.
    #[must_use]
    pub fn with_claim_split_tokens(mut self, tokens: Vec<String>) -> Self {
        self.claim_split_tokens = tokens;
        self
    }
}

// ── ClaimEntailment ───────────────────────────────────────────────────────────

/// Entailment result for a single atomic claim.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimEntailment {
    /// The atomic claim extracted from the answer.
    pub claim: String,
    /// Computed entailment score in `[0.0, 1.0]`.
    ///
    /// Represents the fraction of the claim's unique vocabulary tokens that appear
    /// in the best-matching context passage.
    pub entailment_score: f32,
    /// Whether the claim is considered entailed by the context.
    ///
    /// `true` when `entailment_score >= min_entailment_score` from the evaluator
    /// configuration.
    pub is_entailed: bool,
    /// The context passage that best supports this claim, if any token overlap was
    /// found (`Some` when `entailment_score > 0.0`, `None` otherwise).
    pub supporting_passage: Option<String>,
}

// ── FaithfulnessScore ─────────────────────────────────────────────────────────

/// Aggregate faithfulness evaluation result for a complete answer.
#[derive(Debug, Clone, PartialEq)]
pub struct FaithfulnessScore {
    /// The original answer that was evaluated.
    pub answer: String,
    /// Per-claim entailment results for every atomic claim extracted from the answer.
    pub claims: Vec<ClaimEntailment>,
    /// Aggregate faithfulness score: `entailed_count / total_claims` in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when the answer produced no atomic claims.
    pub faithfulness: f32,
    /// Number of claims whose `entailment_score` met or exceeded the configured
    /// threshold.
    pub entailed_count: usize,
    /// Total number of atomic claims extracted from the answer.
    pub total_claims: usize,
}

// ── FaithfulnessError ─────────────────────────────────────────────────────────

/// Errors produced by the `faithfulness_eval` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FaithfulnessError {
    /// The answer string was empty (or contained only whitespace).
    #[error("answer is empty")]
    EmptyAnswer,
    /// The context slice was empty — at least one passage is required.
    #[error("context is empty")]
    EmptyContext,
    /// Evaluation could not be completed; the inner string describes the reason.
    #[error("evaluation failed: {0}")]
    EvaluationFailed(String),
}
