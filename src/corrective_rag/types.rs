//! Core types for Corrective RAG (CRAG).
//!
//! Defines configuration, grading enumerations, structured outputs, and the
//! error type for the CRAG pipeline.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── CragConfig ────────────────────────────────────────────────────────────────

/// Configuration for the CRAG engine.
///
/// Controls thresholds for grading, knowledge-strip refinement, MMR
/// recomposition, and loop limits.
#[derive(Debug, Clone)]
pub struct CragConfig {
    /// Score threshold above which a document is graded [`RetrievalGrade::Correct`].
    ///
    /// Defaults to `0.7`.
    pub upper_threshold: f32,

    /// Score threshold at or below which a document is graded
    /// [`RetrievalGrade::Incorrect`].
    ///
    /// Defaults to `0.3`.
    pub lower_threshold: f32,

    /// Maximum number of corrective re-retrieval rounds before the loop stops.
    ///
    /// Defaults to `2`.
    pub max_corrections: usize,

    /// Whether to apply knowledge-strip decomposition/recomposition.
    ///
    /// When disabled, results from the grading step are passed directly to MMR
    /// recomposition without sentence-level filtering.
    ///
    /// Defaults to `true`.
    pub enable_knowledge_strip: bool,

    /// Minimum per-strip relevance score required to keep a strip (Jaccard vs
    /// the current query).
    ///
    /// Defaults to `0.5`.
    pub strip_relevance_threshold: f32,

    /// The λ parameter passed to [`crate::advanced_retrieval::MmrReranker`].
    ///
    /// Controls the relevance–diversity trade-off (`0.0` = pure diversity,
    /// `1.0` = pure relevance).
    ///
    /// Defaults to `0.5`.
    pub mmr_lambda: f32,

    /// Dimensionality of the lexical pseudo-embeddings used by MMR.
    ///
    /// Defaults to `256`.
    pub recomposition_dim: usize,

    /// Maximum number of results returned in [`CragOutput::refined_results`].
    ///
    /// Defaults to `5`.
    pub top_k: usize,
}

impl Default for CragConfig {
    fn default() -> Self {
        Self {
            upper_threshold: 0.7,
            lower_threshold: 0.3,
            max_corrections: 2,
            enable_knowledge_strip: true,
            strip_relevance_threshold: 0.5,
            mmr_lambda: 0.5,
            recomposition_dim: 256,
            top_k: 5,
        }
    }
}

impl CragConfig {
    /// Set the upper correctness threshold.
    #[must_use]
    pub fn with_upper_threshold(mut self, v: f32) -> Self {
        self.upper_threshold = v;
        self
    }

    /// Set the lower incorrectness threshold.
    #[must_use]
    pub fn with_lower_threshold(mut self, v: f32) -> Self {
        self.lower_threshold = v;
        self
    }

    /// Set the maximum number of corrective loops.
    #[must_use]
    pub fn with_max_corrections(mut self, v: usize) -> Self {
        self.max_corrections = v;
        self
    }

    /// Enable or disable knowledge-strip refinement.
    #[must_use]
    pub fn with_knowledge_strip(mut self, v: bool) -> Self {
        self.enable_knowledge_strip = v;
        self
    }

    /// Set the strip relevance threshold.
    #[must_use]
    pub fn with_strip_relevance_threshold(mut self, v: f32) -> Self {
        self.strip_relevance_threshold = v;
        self
    }

    /// Set the MMR λ parameter.
    #[must_use]
    pub fn with_mmr_lambda(mut self, v: f32) -> Self {
        self.mmr_lambda = v;
        self
    }

    /// Set the lexical pseudo-embedding dimensionality.
    #[must_use]
    pub fn with_recomposition_dim(mut self, v: usize) -> Self {
        self.recomposition_dim = v;
        self
    }

    /// Set the maximum number of results returned.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }
}

// ── RetrievalGrade ────────────────────────────────────────────────────────────

/// The relevance grade assigned to a retrieved document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalGrade {
    /// The document is clearly relevant (`score >= upper_threshold`).
    Correct,
    /// The document's relevance is uncertain
    /// (`lower_threshold < score < upper_threshold`).
    Ambiguous,
    /// The document is clearly not relevant (`score <= lower_threshold`).
    Incorrect,
}

impl RetrievalGrade {
    /// Derive a grade from a raw relevance score and a config.
    ///
    /// Priority when `lower_threshold == upper_threshold`:
    /// `Correct` takes precedence (score exactly on the boundary is `Correct`).
    #[must_use]
    pub fn from_score(score: f32, cfg: &CragConfig) -> Self {
        if score >= cfg.upper_threshold {
            Self::Correct
        } else if score <= cfg.lower_threshold {
            Self::Incorrect
        } else {
            Self::Ambiguous
        }
    }

    /// Return a human-readable string label for this grade.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Ambiguous => "ambiguous",
            Self::Incorrect => "incorrect",
        }
    }
}

// ── CorrectiveAction ──────────────────────────────────────────────────────────

/// The corrective action decided by the engine for this retrieval round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectiveAction {
    /// All documents were graded `Correct`; use them as-is (possibly after
    /// knowledge-strip refinement).
    UseAsIs,
    /// Some docs are correct, some incorrect; keep correct ones and refine
    /// the ambiguous ones via knowledge strips.
    Augment,
    /// Most documents are ambiguous; apply knowledge-strip refinement
    /// without re-querying.
    Refine,
    /// All documents are incorrect; rewrite the query and re-retrieve.
    Rewrite,
    /// A document was filtered out entirely (no kept strips).
    Discard,
}

// ── GradedDocument ────────────────────────────────────────────────────────────

/// A search result annotated with its relevance score and grade.
#[derive(Debug, Clone)]
pub struct GradedDocument {
    /// The original search result.
    pub result: crate::types::SearchResult,
    /// Raw relevance score produced by the [`crate::corrective_rag::RetrievalGrader`].
    pub relevance: f32,
    /// Grade derived from `relevance` and the current [`CragConfig`] thresholds.
    pub grade: RetrievalGrade,
}

// ── KnowledgeStrip ────────────────────────────────────────────────────────────

/// A single sentence-level fragment extracted from a retrieved document during
/// knowledge-strip decomposition.
#[derive(Debug, Clone)]
pub struct KnowledgeStrip {
    /// The ID of the document this strip was extracted from.
    pub source_id: crate::types::DocumentId,
    /// The sentence text.
    pub text: String,
    /// Jaccard relevance of this strip vs the current query.
    pub relevance: f32,
    /// Whether this strip survived the relevance filter and will be recomposed.
    pub kept: bool,
}

// ── CragOutput ────────────────────────────────────────────────────────────────

/// The complete output of a CRAG run.
#[derive(Debug, Clone)]
pub struct CragOutput {
    /// Final reranked documents, ready for answer synthesis.
    pub refined_results: Vec<crate::types::SearchResult>,
    /// Per-document grades from the last grading round.
    pub grades: Vec<GradedDocument>,
    /// Corrective actions taken across all rounds (one per round).
    pub actions_taken: Vec<CorrectiveAction>,
    /// Number of corrective re-retrieval rounds performed.
    pub correction_rounds: usize,
    /// All knowledge strips produced (kept and discarded), for introspection.
    pub knowledge_strips: Vec<KnowledgeStrip>,
    /// The query string that was ultimately used for the last retrieval.
    ///
    /// Equals the original query when no rewrite happened.
    pub final_query: String,
}

// ── CorrectiveRagError ────────────────────────────────────────────────────────

/// Errors that can occur during a CRAG pipeline run.
#[derive(Debug, Error)]
pub enum CorrectiveRagError {
    /// The underlying retrieval (Echo search) failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),

    /// The document grader returned an error.
    #[error("Grading failed: {0}")]
    GradingFailed(String),

    /// The query string was empty after trimming.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// The configured thresholds are contradictory (lower > upper).
    #[error("Invalid thresholds: lower ({lower}) > upper ({upper})")]
    InvalidThresholds {
        /// The lower threshold value.
        lower: f32,
        /// The upper threshold value.
        upper: f32,
    },
}
