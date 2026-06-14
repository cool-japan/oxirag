//! Types for the `rgb_eval` module.
use thiserror::Error;

// ── RgbAbility ────────────────────────────────────────────────────────────────

/// One of the four fundamental RAG abilities probed by the RGB benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RgbAbility {
    /// Answering correctly despite irrelevant / noise contexts mixed in.
    NoiseRobustness,
    /// Rejecting ("I cannot answer …") when no context contains the answer.
    NegativeRejection,
    /// Integrating facts spread across multiple contexts into one answer.
    InformationIntegration,
    /// Giving the true gold answer despite counterfactual info planted in a context.
    CounterfactualRobustness,
}

impl RgbAbility {
    /// Return a stable, human-readable label for this ability.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::NoiseRobustness => "noise_robustness",
            Self::NegativeRejection => "negative_rejection",
            Self::InformationIntegration => "information_integration",
            Self::CounterfactualRobustness => "counterfactual_robustness",
        }
    }
}

// ── RgbTestCase ───────────────────────────────────────────────────────────────

/// A single labeled RGB test case.
///
/// Each case pairs a query with its gold answer, the contexts supplied to the
/// system, and the [`RgbAbility`] it is designed to probe. For
/// [`RgbAbility::InformationIntegration`] cases, `sub_answers` lists every fact
/// the answer must cover; for [`RgbAbility::NegativeRejection`] cases,
/// `has_answer` is `false` because no context holds the answer.
#[derive(Debug, Clone)]
pub struct RgbTestCase {
    /// The user query.
    pub query: String,
    /// The reference (gold) answer.
    pub gold_answer: String,
    /// Required sub-answers for integration cases (each must be covered).
    pub sub_answers: Vec<String>,
    /// Contexts presented to the system (may include noise / counterfactuals).
    pub contexts: Vec<String>,
    /// Whether any supplied context actually contains the answer.
    pub has_answer: bool,
    /// The ability this case probes.
    pub ability: RgbAbility,
}

impl RgbTestCase {
    /// Create a test case with `has_answer = true` and no sub-answers.
    #[must_use]
    pub fn new(
        query: impl Into<String>,
        gold_answer: impl Into<String>,
        contexts: Vec<String>,
        ability: RgbAbility,
    ) -> Self {
        Self {
            query: query.into(),
            gold_answer: gold_answer.into(),
            sub_answers: Vec::new(),
            contexts,
            has_answer: true,
            ability,
        }
    }

    /// Set the required sub-answers (used by integration cases).
    #[must_use]
    pub fn with_sub_answers(mut self, sub_answers: Vec<String>) -> Self {
        self.sub_answers = sub_answers;
        self
    }

    /// Set whether any context contains the answer.
    #[must_use]
    pub fn with_has_answer(mut self, has_answer: bool) -> Self {
        self.has_answer = has_answer;
        self
    }
}

// ── RgbScores ─────────────────────────────────────────────────────────────────

/// Per-ability accuracies plus an overall mean, all in `[0.0, 1.0]`.
///
/// An ability score is `0.0` when no case of that ability was present; the
/// `overall` field is the mean over only the abilities that were present.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RgbScores {
    /// Accuracy over [`RgbAbility::NoiseRobustness`] cases.
    pub noise_robustness: f32,
    /// Accuracy over [`RgbAbility::NegativeRejection`] cases.
    pub negative_rejection: f32,
    /// Accuracy over [`RgbAbility::InformationIntegration`] cases.
    pub information_integration: f32,
    /// Accuracy over [`RgbAbility::CounterfactualRobustness`] cases.
    pub counterfactual_robustness: f32,
    /// Mean of the accuracies over only the abilities that were present.
    pub overall: f32,
}

// ── RgbConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`crate::rgb_eval::RgbEvaluator`].
#[derive(Debug, Clone)]
pub struct RgbConfig {
    /// Minimum gold-token-overlap fraction for an answer to count as correct.
    ///
    /// Defaults to `0.5`.
    pub match_threshold: f32,
    /// Phrases whose presence marks an answer as a rejection.
    ///
    /// Defaults to `["cannot answer", "not enough information", "no answer", "insufficient"]`.
    pub rejection_phrases: Vec<String>,
}

impl Default for RgbConfig {
    fn default() -> Self {
        Self {
            match_threshold: 0.5,
            rejection_phrases: vec![
                "cannot answer".to_string(),
                "not enough information".to_string(),
                "no answer".to_string(),
                "insufficient".to_string(),
            ],
        }
    }
}

impl RgbConfig {
    /// Create a config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the gold-overlap match threshold.
    #[must_use]
    pub fn with_match_threshold(mut self, threshold: f32) -> Self {
        self.match_threshold = threshold;
        self
    }

    /// Replace the set of rejection phrases.
    #[must_use]
    pub fn with_rejection_phrases(mut self, phrases: Vec<String>) -> Self {
        self.rejection_phrases = phrases;
        self
    }
}

// ── RgbError ──────────────────────────────────────────────────────────────────

/// Errors from the `rgb_eval` module.
#[derive(Debug, Error)]
pub enum RgbError {
    /// No test cases were supplied to [`crate::rgb_eval::RgbEvaluator::evaluate`].
    #[error("no test cases")]
    EmptyCases,
    /// The number of answers did not match the number of cases.
    #[error("answers length {answers} != cases length {cases}")]
    LengthMismatch {
        /// Number of supplied answers.
        answers: usize,
        /// Number of supplied test cases.
        cases: usize,
    },
}
