//! Types for the `self_rag` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::SearchResult;

// ── ReflectionToken ───────────────────────────────────────────────────────────

/// A Self-RAG reflection token assigned to a segment or document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectionToken {
    /// The engine decided to retrieve documents for this segment.
    Retrieve,
    /// The engine decided retrieval was unnecessary.
    NoRetrieve,
    /// The retrieved document is relevant to the query.
    Relevant,
    /// The retrieved document is not relevant to the query.
    Irrelevant,
    /// The answer segment is fully supported by the retrieved document.
    Supported,
    /// The answer segment is partially supported by the retrieved document.
    PartiallySupported,
    /// The answer segment is not supported by any retrieved document.
    Unsupported,
    /// The generated answer is useful for the query.
    Useful,
    /// The generated answer is not useful for the query.
    NotUseful,
}

// ── SelfRagConfig ─────────────────────────────────────────────────────────────

/// Configuration for the Self-RAG engine.
#[derive(Debug, Clone)]
pub struct SelfRagConfig {
    /// Relevance score threshold above which a document is marked [`ReflectionToken::Relevant`].
    ///
    /// Defaults to `0.5`.
    pub relevance_threshold: f32,
    /// Support score threshold above which a segment is marked [`ReflectionToken::Supported`].
    ///
    /// Defaults to `0.5`.
    pub support_threshold: f32,
    /// Utility score threshold above which the output is marked [`ReflectionToken::Useful`].
    ///
    /// Defaults to `0.4`.
    pub utility_threshold: f32,
    /// Number of documents to retrieve per segment.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// Maximum number of answer segments to process.
    ///
    /// Defaults to `8`.
    pub max_segments: usize,
}

impl Default for SelfRagConfig {
    fn default() -> Self {
        Self {
            relevance_threshold: 0.5,
            support_threshold: 0.5,
            utility_threshold: 0.4,
            top_k: 5,
            max_segments: 8,
        }
    }
}

impl SelfRagConfig {
    /// Set the relevance threshold.
    #[must_use]
    pub fn with_relevance_threshold(mut self, v: f32) -> Self {
        self.relevance_threshold = v;
        self
    }

    /// Set the support threshold.
    #[must_use]
    pub fn with_support_threshold(mut self, v: f32) -> Self {
        self.support_threshold = v;
        self
    }

    /// Set the utility threshold.
    #[must_use]
    pub fn with_utility_threshold(mut self, v: f32) -> Self {
        self.utility_threshold = v;
        self
    }

    /// Set the number of documents to retrieve per segment.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }

    /// Set the maximum number of segments.
    #[must_use]
    pub fn with_max_segments(mut self, v: usize) -> Self {
        self.max_segments = v;
        self
    }
}

// ── SegmentCritique ───────────────────────────────────────────────────────────

/// Critique of one answer segment after retrieval and grading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentCritique {
    /// The original segment text.
    pub segment: String,
    /// The relevance token assigned to the best matching document.
    pub relevance_token: ReflectionToken,
    /// The support token for this segment.
    pub support_token: ReflectionToken,
    /// Relevance score of the best matching document (`[0.0, 1.0]`).
    pub relevance_score: f32,
    /// Support score for this segment (`[0.0, 1.0]`).
    pub support_score: f32,
    /// Documents retrieved for this segment.
    pub retrieved_docs: Vec<SearchResult>,
}

// ── SelfRagOutput ─────────────────────────────────────────────────────────────

/// Output produced by the Self-RAG engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRagOutput {
    /// The original query.
    pub query: String,
    /// The final answer (composed from supported segments).
    pub answer: String,
    /// Whether the engine retrieved documents during this run.
    pub retrieved: bool,
    /// All reflection tokens emitted during the run.
    pub reflection_tokens: Vec<ReflectionToken>,
    /// Per-segment critiques.
    pub critiques: Vec<SegmentCritique>,
    /// Overall utility token for the final answer.
    pub utility_token: ReflectionToken,
}

impl SelfRagOutput {
    /// Return `true` if the overall answer is marked useful.
    #[must_use]
    pub fn is_useful(&self) -> bool {
        self.utility_token == ReflectionToken::Useful
    }

    /// Return the fraction of segments that are at least partially supported.
    ///
    /// Returns `0.0` when there are no critiques.
    #[must_use]
    pub fn support_rate(&self) -> f32 {
        if self.critiques.is_empty() {
            return 0.0;
        }
        let supported = self
            .critiques
            .iter()
            .filter(|c| {
                matches!(
                    c.support_token,
                    ReflectionToken::Supported | ReflectionToken::PartiallySupported
                )
            })
            .count();
        #[allow(clippy::cast_precision_loss)]
        let rate = supported as f32 / self.critiques.len() as f32;
        rate
    }
}

// ── SelfRagError ──────────────────────────────────────────────────────────────

/// Errors from the `self_rag` module.
#[derive(Debug, Error)]
pub enum SelfRagError {
    /// The query string was empty after trimming.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// The retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),

    /// The reflection/grading step failed.
    #[error("Reflection failed: {0}")]
    ReflectionFailed(String),
}
