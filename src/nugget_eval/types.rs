//! Types for the `nugget_eval` module.
use thiserror::Error;

// ── NuggetImportance ──────────────────────────────────────────────────────────

/// Importance class assigned to an information nugget.
///
/// Following the TREC nugget-evaluation methodology, each atomic information
/// unit is graded as either essential (`Vital`) or merely desirable (`Okay`).
/// The two classes are weighted differently when computing coverage so that
/// missing a vital nugget penalises a system answer more than missing an okay
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NuggetImportance {
    /// An essential fact that a complete answer must contain.
    Vital,
    /// A useful but non-essential fact.
    Okay,
}

// ── Nugget ────────────────────────────────────────────────────────────────────

/// A single atomic information nugget extracted from a reference answer.
///
/// A nugget is a short factual unit (typically one clause) paired with an
/// [`NuggetImportance`] grade. A system answer "covers" a nugget when enough of
/// the nugget's tokens are present in the answer (see
/// [`NuggetScorer::is_covered`](crate::nugget_eval::NuggetScorer::is_covered)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nugget {
    /// The factual text of the nugget.
    pub text: String,
    /// How important this nugget is to a complete answer.
    pub importance: NuggetImportance,
}

impl Nugget {
    /// Create a new nugget with the given text and importance.
    #[must_use]
    pub fn new(text: impl Into<String>, importance: NuggetImportance) -> Self {
        Self {
            text: text.into(),
            importance,
        }
    }

    /// Create a [`NuggetImportance::Vital`] nugget.
    #[must_use]
    pub fn vital(text: impl Into<String>) -> Self {
        Self::new(text, NuggetImportance::Vital)
    }

    /// Create an [`NuggetImportance::Okay`] nugget.
    #[must_use]
    pub fn okay(text: impl Into<String>) -> Self {
        Self::new(text, NuggetImportance::Okay)
    }

    /// Returns `true` if this nugget is graded [`NuggetImportance::Vital`].
    #[must_use]
    pub fn is_vital(&self) -> bool {
        matches!(self.importance, NuggetImportance::Vital)
    }
}

// ── NuggetConfig ──────────────────────────────────────────────────────────────

/// Configuration for nugget extraction and coverage scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct NuggetConfig {
    /// Fraction of a nugget's tokens that must appear in a system answer for the
    /// nugget to count as covered. In `[0.0, 1.0]`; defaults to `0.5`.
    pub coverage_threshold: f32,
    /// Weight applied to a covered [`NuggetImportance::Vital`] nugget when
    /// computing the weighted score. Defaults to `1.0`.
    pub vital_weight: f32,
    /// Weight applied to a covered [`NuggetImportance::Okay`] nugget when
    /// computing the weighted score. Defaults to `0.5`.
    pub okay_weight: f32,
    /// Minimum number of tokens a clause must contain to become a nugget.
    /// Shorter clauses are discarded. Defaults to `2`.
    pub min_nugget_tokens: usize,
}

impl Default for NuggetConfig {
    fn default() -> Self {
        Self {
            coverage_threshold: 0.5,
            vital_weight: 1.0,
            okay_weight: 0.5,
            min_nugget_tokens: 2,
        }
    }
}

impl NuggetConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the [`NuggetConfig::coverage_threshold`].
    #[must_use]
    pub fn with_coverage_threshold(mut self, threshold: f32) -> Self {
        self.coverage_threshold = threshold;
        self
    }

    /// Set the [`NuggetConfig::vital_weight`].
    #[must_use]
    pub fn with_vital_weight(mut self, weight: f32) -> Self {
        self.vital_weight = weight;
        self
    }

    /// Set the [`NuggetConfig::okay_weight`].
    #[must_use]
    pub fn with_okay_weight(mut self, weight: f32) -> Self {
        self.okay_weight = weight;
        self
    }

    /// Set the [`NuggetConfig::min_nugget_tokens`].
    #[must_use]
    pub fn with_min_nugget_tokens(mut self, min_tokens: usize) -> Self {
        self.min_nugget_tokens = min_tokens;
        self
    }

    /// Return the scoring weight for a nugget of the given importance.
    #[must_use]
    pub fn weight_for(&self, importance: NuggetImportance) -> f32 {
        match importance {
            NuggetImportance::Vital => self.vital_weight,
            NuggetImportance::Okay => self.okay_weight,
        }
    }
}

// ── NuggetScore ───────────────────────────────────────────────────────────────

/// The result of scoring a system answer against a set of nuggets.
///
/// All score fields lie in `[0.0, 1.0]` where higher is better.
#[derive(Debug, Clone, PartialEq)]
pub struct NuggetScore {
    /// Fraction of all nuggets covered by the answer (`covered / total`).
    pub coverage: f32,
    /// Fraction of [`NuggetImportance::Vital`] nuggets covered. `0.0` when there
    /// are no vital nuggets.
    pub vital_coverage: f32,
    /// Importance-weighted coverage: `Σ weight(covered) / Σ weight(all)`.
    pub weighted_score: f32,
    /// Indices (into the scored nugget slice) of the covered nuggets.
    pub covered: Vec<usize>,
    /// Indices (into the scored nugget slice) of the missed nuggets.
    pub missed: Vec<usize>,
}

// ── NuggetEvalError ───────────────────────────────────────────────────────────

/// Errors from the `nugget_eval` module.
#[derive(Debug, Error)]
pub enum NuggetEvalError {
    /// The reference answer was empty (or whitespace only).
    #[error("reference must not be empty")]
    EmptyReference,
}
