//! Retrieval graders for CRAG.
//!
//! A [`RetrievalGrader`] assigns a relevance score in `[0, 1]` to each
//! retrieved document with respect to the current query.  Two implementations
//! are provided:
//!
//! * [`HeuristicRetrievalGrader`] — lexical Jaccard overlap (no ML dependency).
//! * [`MockRetrievalGrader`] — scripted scores for unit tests.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::types::SearchResult;

use super::types::{CorrectiveRagError, CragConfig, GradedDocument, RetrievalGrade};

// ── tokenise helper (shared) ──────────────────────────────────────────────────

/// Lowercase-split tokeniser for relevance grading.
///
/// Splits on every non-alphanumeric character and filters tokens shorter than
/// 2 characters.
fn tokenize_for_grading(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|t| t.len() >= 2)
        .collect()
}

// ── RetrievalGrader trait ─────────────────────────────────────────────────────

/// A grader that assigns per-document relevance scores to retrieval results.
///
/// Implementations must be `Send + Sync` to work safely in the multi-threaded
/// CRAG engine.
#[async_trait]
pub trait RetrievalGrader: Send + Sync {
    /// Grade a single `result` against the `query`.
    ///
    /// Returns a relevance score in `[0.0, 1.0]` (higher = more relevant).
    ///
    /// # Errors
    ///
    /// Returns [`CorrectiveRagError::GradingFailed`] when the grading operation
    /// cannot be completed (e.g., an external model call fails).
    async fn grade(&self, query: &str, result: &SearchResult) -> Result<f32, CorrectiveRagError>;

    /// Grade a batch of `results`, returning [`GradedDocument`] values.
    ///
    /// The default implementation calls [`Self::grade`] in a sequential loop,
    /// but implementors may override for batched efficiency.
    ///
    /// # Errors
    ///
    /// Propagates the first error returned by [`Self::grade`].
    async fn grade_all(
        &self,
        query: &str,
        results: &[SearchResult],
        cfg: &CragConfig,
    ) -> Result<Vec<GradedDocument>, CorrectiveRagError> {
        let mut graded = Vec::with_capacity(results.len());
        for result in results {
            let relevance = self.grade(query, result).await?;
            let grade = RetrievalGrade::from_score(relevance, cfg);
            graded.push(GradedDocument {
                result: result.clone(),
                relevance,
                grade,
            });
        }
        Ok(graded)
    }
}

// ── HeuristicRetrievalGrader ──────────────────────────────────────────────────

/// Lexical-Jaccard heuristic grader.
///
/// Grades each document by computing the Jaccard similarity between the token
/// sets of the query and the document content:
///
/// ```text
/// score = |query_tokens ∩ doc_tokens| / |query_tokens ∪ doc_tokens|
/// ```
///
/// Tokens are lowercased and must be at least 2 characters long.  When either
/// set is empty the score is `0.0`.
///
/// # Configuration
///
/// `min_overlap` is informational and does not filter results; grading always
/// produces a `[0.0, 1.0]` float that is later converted to a
/// [`RetrievalGrade`] by [`RetrievalGrade::from_score`].
#[derive(Debug, Clone)]
pub struct HeuristicRetrievalGrader {
    /// Minimum Jaccard overlap considered meaningful for logging/debugging.
    pub min_overlap: f32,
}

impl HeuristicRetrievalGrader {
    /// Create a new grader with the default `min_overlap` of `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self { min_overlap: 0.0 }
    }

    /// Set a custom `min_overlap` threshold.
    #[must_use]
    pub fn with_min_overlap(mut self, min_overlap: f32) -> Self {
        self.min_overlap = min_overlap;
        self
    }

    /// Compute the Jaccard coefficient between the token sets of `a` and `b`.
    fn jaccard(a: &str, b: &str) -> f32 {
        let set_a = tokenize_for_grading(a);
        let set_b = tokenize_for_grading(b);

        if set_a.is_empty() && set_b.is_empty() {
            return 0.0;
        }

        let intersection: usize = set_a.intersection(&set_b).count();
        let union: usize = set_a.union(&set_b).count();

        if union == 0 {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let score = intersection as f32 / union as f32;
        score
    }
}

impl Default for HeuristicRetrievalGrader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RetrievalGrader for HeuristicRetrievalGrader {
    async fn grade(&self, query: &str, result: &SearchResult) -> Result<f32, CorrectiveRagError> {
        let score = Self::jaccard(query, &result.document.content);
        Ok(score)
    }
}

// ── MockRetrievalGrader ───────────────────────────────────────────────────────

/// Scripted grader for unit tests.
///
/// Returns a per-document-ID score when available, falling back to a uniform
/// fixed score.
#[derive(Debug, Clone)]
pub struct MockRetrievalGrader {
    /// Default score returned when no per-ID override exists.
    score: f32,
    /// Per-document-ID score overrides (keyed by [`DocumentId::as_str`]).
    scores: HashMap<String, f32>,
}

impl MockRetrievalGrader {
    /// Create a grader that always returns `score`.
    #[must_use]
    pub fn new(score: f32) -> Self {
        Self {
            score,
            scores: HashMap::new(),
        }
    }

    /// Create a grader driven entirely by the provided per-ID `map`.
    ///
    /// Documents whose IDs are absent from `map` receive the default score of
    /// `0.0`.
    #[must_use]
    pub fn with_scores(map: HashMap<String, f32>) -> Self {
        Self {
            score: 0.0,
            scores: map,
        }
    }
}

#[async_trait]
impl RetrievalGrader for MockRetrievalGrader {
    async fn grade(&self, _query: &str, result: &SearchResult) -> Result<f32, CorrectiveRagError> {
        let s = self
            .scores
            .get(result.document.id.as_str())
            .copied()
            .unwrap_or(self.score);
        Ok(s)
    }
}
