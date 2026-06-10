//! Types for the `query_decomposition` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::SearchResult;

// ── DecompositionStrategy ─────────────────────────────────────────────────────

/// Strategy used to decompose a complex query into sub-questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Answer all sub-questions independently in parallel, then fuse results.
    #[default]
    Parallel,
    /// Answer simpler questions first; each answer informs the next (chain-of-thought).
    LeastToMost,
    /// Generate an abstracted "step-back" question before the specific sub-questions.
    StepBack,
}

impl DecompositionStrategy {
    /// Return the strategy name as a static string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::LeastToMost => "least-to-most",
            Self::StepBack => "step-back",
        }
    }
}

// ── SubQuestion ───────────────────────────────────────────────────────────────

/// A single sub-question derived from the original query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubQuestion {
    /// The sub-question text.
    pub text: String,
    /// Zero-based index within the decomposed set.
    pub index: usize,
}

// ── SubAnswer ─────────────────────────────────────────────────────────────────

/// The answer to a single sub-question, with supporting documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAnswer {
    /// The sub-question that was answered.
    pub question: SubQuestion,
    /// Heuristic answer derived from the top retrieved document.
    pub answer: String,
    /// Documents retrieved for this sub-question.
    pub results: Vec<SearchResult>,
}

// ── DecompositionConfig ───────────────────────────────────────────────────────

/// Configuration for the query decomposition engine.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Decomposition strategy.
    ///
    /// Defaults to [`DecompositionStrategy::Parallel`].
    pub strategy: DecompositionStrategy,
    /// Maximum number of sub-questions to generate.
    ///
    /// Defaults to `4`.
    pub max_sub_questions: usize,
    /// Top-k documents to retrieve per sub-question.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// Reciprocal rank fusion constant `k`.
    ///
    /// Defaults to `60.0`.
    pub rrf_k: f32,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            strategy: DecompositionStrategy::Parallel,
            max_sub_questions: 4,
            top_k: 5,
            rrf_k: 60.0,
        }
    }
}

impl DecompositionConfig {
    /// Set the decomposition strategy.
    #[must_use]
    pub fn with_strategy(mut self, v: DecompositionStrategy) -> Self {
        self.strategy = v;
        self
    }

    /// Set the maximum number of sub-questions.
    #[must_use]
    pub fn with_max_sub_questions(mut self, v: usize) -> Self {
        self.max_sub_questions = v;
        self
    }

    /// Set the retrieval top-k.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }

    /// Set the RRF constant.
    #[must_use]
    pub fn with_rrf_k(mut self, v: f32) -> Self {
        self.rrf_k = v;
        self
    }
}

// ── DecomposedQuery ───────────────────────────────────────────────────────────

/// The complete result of decomposing and answering a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposedQuery {
    /// The original query string.
    pub original_query: String,
    /// Generated sub-questions.
    pub sub_questions: Vec<SubQuestion>,
    /// Answers to each sub-question.
    pub sub_answers: Vec<SubAnswer>,
    /// Final synthesized answer (concatenation of sub-answers).
    pub final_answer: String,
    /// RRF-fused retrieval results from all sub-questions.
    pub fused_results: Vec<SearchResult>,
}

impl DecomposedQuery {
    /// Return `true` if no sub-questions were generated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sub_questions.is_empty()
    }
}

// ── QueryDecompositionError ───────────────────────────────────────────────────

/// Errors from the `query_decomposition` module.
#[derive(Debug, Error)]
pub enum QueryDecompositionError {
    /// The query string was empty after trimming.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// The retrieval step failed for a sub-question.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
}
