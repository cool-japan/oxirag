//! Reflector trait and implementations for the `self_rag` module.
//!
//! A [`Reflector`] scores the relevance and support of a retrieved document
//! against an answer segment, and the overall utility of an answer.
//! All implementations are **pure sync** — no I/O, no async.

use std::collections::HashSet;

use crate::types::SearchResult;

use super::types::SelfRagError;

// ── tokenize helper ───────────────────────────────────────────────────────────

/// Lowercase tokeniser used by all heuristic scorers.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

/// Jaccard similarity between two token sets.
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

// ── Reflector trait ───────────────────────────────────────────────────────────

/// Synchronous relevance, support, and utility scorer for Self-RAG.
///
/// Implementations must be **pure compute** — no I/O, no async.
pub trait Reflector {
    /// Score how relevant `doc` is to the `query` string.
    ///
    /// Returns a value in `[0.0, 1.0]` (higher = more relevant).
    ///
    /// # Errors
    ///
    /// Returns [`SelfRagError::ReflectionFailed`] on internal scorer errors.
    fn relevance(&self, query: &str, doc: &SearchResult) -> Result<f32, SelfRagError>;

    /// Score how well `doc` supports the `segment` text.
    ///
    /// Returns a value in `[0.0, 1.0]` (higher = more supported).
    ///
    /// # Errors
    ///
    /// Returns [`SelfRagError::ReflectionFailed`] on internal scorer errors.
    fn support(&self, segment: &str, doc: &SearchResult) -> Result<f32, SelfRagError>;

    /// Score the overall utility of `answer` for the `query`.
    ///
    /// Returns a value in `[0.0, 1.0]` (higher = more useful).
    ///
    /// # Errors
    ///
    /// Returns [`SelfRagError::ReflectionFailed`] on internal scorer errors.
    fn utility(&self, query: &str, answer: &str) -> Result<f32, SelfRagError>;
}

// ── HeuristicReflector ────────────────────────────────────────────────────────

/// Lexical Jaccard heuristic reflector.
///
/// Scores are computed as token-set Jaccard similarities:
///
/// - `relevance`: Jaccard(query tokens, doc tokens)
/// - `support`: Jaccard(segment tokens, doc tokens)
/// - `utility`: Jaccard(query tokens, answer tokens) × coverage bonus
#[derive(Debug, Clone, Default)]
pub struct HeuristicReflector;

impl HeuristicReflector {
    /// Create a new [`HeuristicReflector`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Reflector for HeuristicReflector {
    fn relevance(&self, query: &str, doc: &SearchResult) -> Result<f32, SelfRagError> {
        let q = tokenize(query);
        let d = tokenize(&doc.document.content);
        Ok(jaccard(&q, &d))
    }

    fn support(&self, segment: &str, doc: &SearchResult) -> Result<f32, SelfRagError> {
        let s = tokenize(segment);
        let d = tokenize(&doc.document.content);
        Ok(jaccard(&s, &d))
    }

    fn utility(&self, query: &str, answer: &str) -> Result<f32, SelfRagError> {
        if answer.trim().is_empty() {
            return Ok(0.0);
        }
        let q = tokenize(query);
        let a = tokenize(answer);
        // Boost score if answer is longer than 3 words (coverage signal)
        let base = jaccard(&q, &a);
        let length_bonus = if a.len() > 3 { 0.15_f32 } else { 0.0_f32 };
        Ok((base + length_bonus).min(1.0))
    }
}

// ── MockReflector ─────────────────────────────────────────────────────────────

/// Scripted reflector for unit tests.
///
/// Returns fixed scores for all inputs.
#[derive(Debug, Clone)]
pub struct MockReflector {
    /// Fixed relevance score returned for every call.
    pub relevance_score: f32,
    /// Fixed support score returned for every call.
    pub support_score: f32,
    /// Fixed utility score returned for every call.
    pub utility_score: f32,
}

impl MockReflector {
    /// Create a mock that returns the same score for all three metrics.
    #[must_use]
    pub fn new(score: f32) -> Self {
        Self {
            relevance_score: score,
            support_score: score,
            utility_score: score,
        }
    }

    /// Create a mock with independent scores for each metric.
    #[must_use]
    pub fn with_scores(relevance: f32, support: f32, utility: f32) -> Self {
        Self {
            relevance_score: relevance,
            support_score: support,
            utility_score: utility,
        }
    }
}

impl Reflector for MockReflector {
    fn relevance(&self, _query: &str, _doc: &SearchResult) -> Result<f32, SelfRagError> {
        Ok(self.relevance_score)
    }

    fn support(&self, _segment: &str, _doc: &SearchResult) -> Result<f32, SelfRagError> {
        Ok(self.support_score)
    }

    fn utility(&self, _query: &str, _answer: &str) -> Result<f32, SelfRagError> {
        Ok(self.utility_score)
    }
}
