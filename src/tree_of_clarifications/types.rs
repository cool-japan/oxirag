//! Traits, configuration, and deterministic mocks for the `tree_of_clarifications`
//! module.
//!
//! Tree of Clarifications (`ToC`; Kim et al., EMNLP 2023, "Tree of
//! Clarifications: Answering Ambiguous Questions with Retrieval-Augmented
//! Large Language Models") handles an *ambiguous* question by branching it into
//! several disambiguated readings, retrieving and answering each reading, then
//! recursively pruning the branches that turned out to have weak retrieval
//! support before aggregating the survivors into one long-form answer.
//!
//! Three **pure-sync** traits model the caller-supplied components — a
//! [`Disambiguator`] that proposes alternate readings of a question, a
//! [`ClarificationRetriever`] that fetches supporting passages for a single
//! reading, and a [`ClarificationAnswerer`] that drafts an answer from those
//! passages. Deterministic [`MockDisambiguator`], [`MockClarificationRetriever`],
//! and [`MockClarificationAnswerer`] implementations are provided for testing;
//! there is no async and no I/O anywhere in this module.

use thiserror::Error;

// ── Disambiguator ─────────────────────────────────────────────────────────────

/// Proposes disambiguated readings of a (possibly ambiguous) question.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete disambiguator (e.g. wrapping an LLM prompted to enumerate readings);
/// [`MockDisambiguator`] is provided for tests.
pub trait Disambiguator {
    /// Propose candidate disambiguated readings of `question`.
    ///
    /// Returns an empty vector when `question` is judged unambiguous as-is —
    /// the caller should then treat `question` itself as the single resolved
    /// reading rather than branching further.
    fn disambiguate(&self, question: &str) -> Vec<String>;
}

// ── ClarificationRetriever ────────────────────────────────────────────────────

/// Retrieves supporting passages for a single disambiguated question.
///
/// Implementations are **pure sync**. The caller supplies a concrete retriever
/// (e.g. wrapping a vector index or `Echo` layer); [`MockClarificationRetriever`]
/// is provided for tests.
pub trait ClarificationRetriever {
    /// Retrieve up to `top_k` passages relevant to `query`.
    ///
    /// May return fewer than `top_k` passages (including zero) when no
    /// sufficiently relevant evidence is available.
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<String>;
}

// ── ClarificationAnswerer ─────────────────────────────────────────────────────

/// Drafts an answer to a single disambiguated question from retrieved passages.
///
/// Implementations are **pure sync**. The caller supplies a concrete answerer
/// (e.g. wrapping an LLM); [`MockClarificationAnswerer`] is provided for tests.
pub trait ClarificationAnswerer {
    /// Answer `question` using the supplied `passages` as evidence.
    ///
    /// `passages` may be empty, in which case the implementation should still
    /// return its best-effort answer (this fallback answer is what a node
    /// falls back to when every one of its children is later pruned).
    fn answer(&self, question: &str, passages: &[String]) -> String;
}

// ── MockDisambiguator ─────────────────────────────────────────────────────────

/// Deterministic [`Disambiguator`] for tests.
///
/// Holds an ordered list of `(question substring, candidate readings)`
/// mappings. [`Disambiguator::disambiguate`] lower-cases `question` and returns
/// the candidates of the first mapping whose (lower-cased) substring is
/// contained in it; an empty needle never matches (this guards against an
/// accidental catch-all entry silently causing infinite branching). When no
/// mapping matches, an empty vector is returned, signalling that the question
/// is not ambiguous.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockDisambiguator {
    /// Ordered `(question substring, candidate disambiguated readings)` pairs.
    pub mappings: Vec<(String, Vec<String>)>,
}

impl MockDisambiguator {
    /// Create a mock disambiguator from explicit substring mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, Vec<String>)>) -> Self {
        Self { mappings }
    }

    /// Create a mock disambiguator that never finds ambiguity (every question
    /// is treated as already unambiguous).
    #[must_use]
    pub fn unambiguous() -> Self {
        Self::default()
    }
}

impl Disambiguator for MockDisambiguator {
    fn disambiguate(&self, question: &str) -> Vec<String> {
        let lower = question.to_lowercase();
        for (needle, candidates) in &self.mappings {
            if needle.is_empty() {
                continue;
            }
            if lower.contains(needle.to_lowercase().as_str()) {
                return candidates.clone();
            }
        }
        Vec::new()
    }
}

// ── MockClarificationRetriever ────────────────────────────────────────────────

/// Deterministic [`ClarificationRetriever`] for tests.
///
/// Holds an ordered list of `(query substring, passages)` mappings.
/// [`ClarificationRetriever::retrieve`] lower-cases `query` and returns (up to
/// `top_k` of) the passages of the first mapping whose (lower-cased) substring
/// is contained in it; an empty needle never matches. When no mapping matches,
/// an empty vector is returned (no supporting evidence found).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockClarificationRetriever {
    /// Ordered `(query substring, passages)` pairs.
    pub corpus: Vec<(String, Vec<String>)>,
}

impl MockClarificationRetriever {
    /// Create a mock retriever from explicit substring mappings.
    #[must_use]
    pub fn new(corpus: Vec<(String, Vec<String>)>) -> Self {
        Self { corpus }
    }

    /// Create a mock retriever that never finds supporting passages for any
    /// query.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }
}

impl ClarificationRetriever for MockClarificationRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<String> {
        let lower = query.to_lowercase();
        for (needle, passages) in &self.corpus {
            if needle.is_empty() {
                continue;
            }
            if lower.contains(needle.to_lowercase().as_str()) {
                return passages.iter().take(top_k).cloned().collect();
            }
        }
        Vec::new()
    }
}

// ── MockClarificationAnswerer ─────────────────────────────────────────────────

/// Deterministic [`ClarificationAnswerer`] for tests.
///
/// Answers by templating the question together with the retrieved passages
/// (joined by a single space). When no passages are supplied, a fixed
/// "no supporting evidence" template is used instead, so the answerer always
/// produces a non-empty, reproducible string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MockClarificationAnswerer;

impl ClarificationAnswerer for MockClarificationAnswerer {
    fn answer(&self, question: &str, passages: &[String]) -> String {
        if passages.is_empty() {
            format!("No supporting evidence was retrieved for: {question}")
        } else {
            format!("{question} -- {}", passages.join(" "))
        }
    }
}

// ── ToCConfig ──────────────────────────────────────────────────────────────────

/// Configuration for [`ToCEngine`](crate::tree_of_clarifications::engine::ToCEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct ToCConfig {
    /// Maximum number of disambiguation rounds allowed while building the
    /// tree. Defaults to `2`.
    ///
    /// A top-level disambiguation of the original question already consumes
    /// one round; each further round may again disambiguate an already
    /// disambiguated (but still ambiguous) reading, up to this cap.
    pub max_depth: usize,
    /// Number of passages requested per [`ClarificationRetriever::retrieve`]
    /// call. Defaults to `3`.
    pub top_k: usize,
    /// Minimum `relevance_score` a node must reach to survive
    /// [`ClarificationTree::prune`](crate::tree_of_clarifications::tree::ClarificationTree::prune).
    /// Defaults to `0.2`.
    pub prune_threshold: f32,
}

impl Default for ToCConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            top_k: 3,
            prune_threshold: 0.2,
        }
    }
}

impl ToCConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum disambiguation depth.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the number of passages retrieved per node.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the pruning threshold.
    #[must_use]
    pub fn with_prune_threshold(mut self, prune_threshold: f32) -> Self {
        self.prune_threshold = prune_threshold;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ToCError::InvalidConfig`] when `max_depth` or `top_k` is `0`,
    /// or when `prune_threshold` is not a finite value within `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), ToCError> {
        if self.max_depth == 0 {
            return Err(ToCError::InvalidConfig(
                "max_depth must be at least 1".to_string(),
            ));
        }
        if self.top_k == 0 {
            return Err(ToCError::InvalidConfig(
                "top_k must be at least 1".to_string(),
            ));
        }
        if !self.prune_threshold.is_finite() || !(0.0..=1.0).contains(&self.prune_threshold) {
            return Err(ToCError::InvalidConfig(format!(
                "prune_threshold must be a finite value within [0.0, 1.0], got {}",
                self.prune_threshold
            )));
        }
        Ok(())
    }
}

// ── ToCError ───────────────────────────────────────────────────────────────────

/// Errors from the `tree_of_clarifications` module.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ToCError {
    /// The original question was empty (or whitespace-only) after trimming.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// The supplied [`ToCConfig`] failed validation.
    #[error("invalid ToC configuration: {0}")]
    InvalidConfig(String),
}
