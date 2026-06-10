//! Types for the `answer_aggregator` module.

use thiserror::Error;

// ── AggregationStrategy ───────────────────────────────────────────────────────

/// Strategy used to combine multiple candidate answers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Select sentences that appear in the majority of candidates.
    #[default]
    MajorityVote,
    /// Weight each candidate's sentences by confidence score.
    WeightedFusion,
    /// Take the union of all sentences after deduplication.
    Extractive,
}

impl AggregationStrategy {
    /// Return a short human-readable label for the strategy.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::MajorityVote => "majority_vote",
            Self::WeightedFusion => "weighted_fusion",
            Self::Extractive => "extractive",
        }
    }
}

// ── CandidateAnswer ───────────────────────────────────────────────────────────

/// A single candidate answer with an associated confidence score.
#[derive(Debug, Clone)]
pub struct CandidateAnswer {
    /// The text of the candidate answer.
    pub text: String,
    /// Confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Optional provenance identifier (e.g., model name, document ID).
    pub source_id: Option<String>,
}

impl CandidateAnswer {
    /// Create a new [`CandidateAnswer`] with the given text and confidence.
    #[must_use]
    pub fn new(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            text: text.into(),
            confidence,
            source_id: None,
        }
    }

    /// Attach a source identifier to this candidate.
    #[must_use]
    pub fn with_source(mut self, id: impl Into<String>) -> Self {
        self.source_id = Some(id.into());
        self
    }

    /// Returns `true` when the candidate is well-formed:
    /// - Non-empty (after trimming).
    /// - Confidence is in `[0.0, 1.0]`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.text.trim().is_empty() && self.confidence >= 0.0 && self.confidence <= 1.0
    }
}

// ── AggregatedAnswer ──────────────────────────────────────────────────────────

/// The final aggregated answer produced by `AnswerAggregator`.
#[derive(Debug, Clone)]
pub struct AggregatedAnswer {
    /// The aggregated answer text.
    pub text: String,
    /// Mean confidence of the contributing candidates.
    pub consensus_score: f32,
    /// Texts of the candidates that contributed to this answer.
    pub contributing_candidates: Vec<String>,
}

impl AggregatedAnswer {
    /// Returns `true` when the answer text is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

// ── AggregationConfig ─────────────────────────────────────────────────────────

/// Configuration for `AnswerAggregator`.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Minimum number of valid candidates required for aggregation.
    ///
    /// Defaults to `2`.
    pub min_candidates: usize,
    /// Jaccard threshold above which two sentences are considered duplicates.
    ///
    /// Defaults to `0.6`.
    pub dedup_threshold: f32,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            min_candidates: 2,
            dedup_threshold: 0.6,
        }
    }
}

impl AggregationConfig {
    /// Set the minimum number of candidates required.
    #[must_use]
    pub fn with_min_candidates(mut self, min_candidates: usize) -> Self {
        self.min_candidates = min_candidates;
        self
    }

    /// Set the deduplication threshold.
    #[must_use]
    pub fn with_dedup_threshold(mut self, dedup_threshold: f32) -> Self {
        self.dedup_threshold = dedup_threshold;
        self
    }
}

// ── AggregationError ──────────────────────────────────────────────────────────

/// Errors that can occur during answer aggregation.
#[derive(Debug, Error)]
pub enum AggregationError {
    /// Fewer valid candidates than `min_candidates` were supplied.
    #[error("Insufficient candidates: need at least {0}")]
    InsufficientCandidates(usize),
    /// All supplied candidates have empty text.
    #[error("All candidate answers are empty")]
    AllCandidatesEmpty,
}
