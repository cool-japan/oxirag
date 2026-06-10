//! Types for the `consistency_checker` module.

use thiserror::Error;

// ── ConflictType ──────────────────────────────────────────────────────────────

/// Category of a detected inconsistency between two claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// Conflicting numerical values (e.g. "20 km" vs "50 km").
    Numerical,
    /// Conflicting temporal references (e.g. "founded in 1990" vs "founded in 2005").
    Temporal,
    /// One claim negates the other (e.g. "X is true" vs "X is not true").
    Negation,
    /// General factual discrepancy detected.
    Factual,
}

impl ConflictType {
    /// Return a lowercase string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Numerical => "numerical",
            Self::Temporal => "temporal",
            Self::Negation => "negation",
            Self::Factual => "factual",
        }
    }
}

// ── Inconsistency ─────────────────────────────────────────────────────────────

/// A single detected inconsistency between two sentences.
#[derive(Debug, Clone)]
pub struct Inconsistency {
    /// First claim involved.
    pub claim_a: String,
    /// Second claim involved.
    pub claim_b: String,
    /// The type of conflict.
    pub conflict_type: ConflictType,
    /// Confidence in the detected conflict, in `[0.0, 1.0]`.
    pub confidence: f32,
}

impl Inconsistency {
    /// Describe the inconsistency in human-readable form.
    #[must_use]
    pub fn description(&self) -> String {
        format!("{:?} conflict between claims", self.conflict_type)
    }
}

// ── ConsistencyReport ─────────────────────────────────────────────────────────

/// Aggregated result of a consistency check over a text passage.
#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    /// All detected inconsistencies.
    pub inconsistencies: Vec<Inconsistency>,
    /// `true` when no inconsistencies were found.
    pub is_consistent: bool,
    /// Consistency score in `[0.0, 1.0]` (1.0 = perfectly consistent).
    pub score: f32,
}

impl ConsistencyReport {
    /// Build a `ConsistencyReport` from detected inconsistencies.
    ///
    /// `is_consistent` = `inconsistencies.is_empty()`.
    /// `score` = `1.0` when consistent, otherwise `1.0 - (count / 10.0).min(1.0)`.
    #[must_use]
    pub fn new(inconsistencies: Vec<Inconsistency>) -> Self {
        let is_consistent = inconsistencies.is_empty();
        #[allow(clippy::cast_precision_loss)]
        let score = if is_consistent {
            1.0_f32
        } else {
            1.0_f32 - (inconsistencies.len() as f32 / 10.0_f32).min(1.0_f32)
        };
        Self {
            inconsistencies,
            is_consistent,
            score,
        }
    }

    /// Number of detected inconsistencies.
    #[must_use]
    pub fn inconsistency_count(&self) -> usize {
        self.inconsistencies.len()
    }

    /// Whether any numerical conflicts were detected.
    #[must_use]
    pub fn has_numerical_conflicts(&self) -> bool {
        self.inconsistencies
            .iter()
            .any(|i| i.conflict_type == ConflictType::Numerical)
    }

    /// Whether any temporal conflicts were detected.
    #[must_use]
    pub fn has_temporal_conflicts(&self) -> bool {
        self.inconsistencies
            .iter()
            .any(|i| i.conflict_type == ConflictType::Temporal)
    }
}

// ── ConsistencyConfig ─────────────────────────────────────────────────────────

/// Configuration for `ConsistencyChecker`.
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    /// Minimum confidence required to include a detected inconsistency.
    ///
    /// Default: `0.5`.
    pub min_confidence: f32,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
        }
    }
}

impl ConsistencyConfig {
    /// Create a new `ConsistencyConfig` with the given minimum confidence.
    #[must_use]
    pub fn new(min_confidence: f32) -> Self {
        Self { min_confidence }
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }
}

// ── ConsistencyChecker ────────────────────────────────────────────────────────

/// Re-exported from `checker` — see [`crate::consistency_checker::checker`].
pub use crate::consistency_checker::checker::ConsistencyChecker;

// ── ConsistencyError ──────────────────────────────────────────────────────────

/// Errors produced by [`ConsistencyChecker`].
#[derive(Debug, Error)]
pub enum ConsistencyError {
    /// The input text was empty.
    #[error("Input must not be empty")]
    EmptyInput,
    /// The text contained fewer than two sentences.
    #[error("Need at least 2 sentences to check consistency")]
    TooFewSentences,
}
