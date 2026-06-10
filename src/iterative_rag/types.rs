//! Types for the `iterative_rag` module.

use thiserror::Error;

// ── IterativeConfig ───────────────────────────────────────────────────────────

/// Configuration for `IterativeRagEngine`.
#[derive(Debug, Clone)]
pub struct IterativeConfig {
    /// Maximum number of retrieval iterations to perform.
    ///
    /// Defaults to `3`.
    pub max_iterations: usize,
    /// Number of documents to retrieve per iteration.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// Number of expansion terms to extract from each iteration's draft.
    ///
    /// Defaults to `3`.
    pub expansion_terms: usize,
}

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 3,
            top_k: 5,
            expansion_terms: 3,
        }
    }
}

impl IterativeConfig {
    /// Create a new [`IterativeConfig`] with the given maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the number of documents to retrieve per iteration.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the number of expansion terms to extract.
    #[must_use]
    pub fn with_expansion_terms(mut self, expansion_terms: usize) -> Self {
        self.expansion_terms = expansion_terms;
        self
    }
}

// ── IterationStep ─────────────────────────────────────────────────────────────

/// A single step in the iterative retrieval process.
#[derive(Debug, Clone)]
pub struct IterationStep {
    /// The iteration index (0-based).
    pub iteration: usize,
    /// The expanded query used for this iteration.
    pub expanded_query: String,
    /// Number of documents retrieved during this iteration.
    pub retrieved_count: usize,
    /// The draft answer assembled from this iteration's results.
    pub draft: String,
}

impl IterationStep {
    /// Returns `true` if the query was expanded beyond a single token.
    #[must_use]
    pub fn is_expanded(&self) -> bool {
        self.expanded_query.split_whitespace().count() > 1
    }
}

// ── IterativeOutput ───────────────────────────────────────────────────────────

/// The complete output of an `IterativeRagEngine` run.
#[derive(Debug, Clone)]
pub struct IterativeOutput {
    /// The sequence of iteration steps executed.
    pub steps: Vec<IterationStep>,
    /// The synthesised final answer after all iterations.
    pub final_answer: String,
    /// Total number of distinct documents retrieved across all iterations.
    pub total_docs_retrieved: usize,
}

impl IterativeOutput {
    /// Returns the number of iterations that were executed.
    #[must_use]
    pub fn total_iterations(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the final answer is an empty string.
    #[must_use]
    pub fn is_empty_answer(&self) -> bool {
        self.final_answer.is_empty()
    }
}

// ── IterativeRagError ─────────────────────────────────────────────────────────

/// Errors that can occur in `IterativeRagEngine`.
#[derive(Debug, Error)]
pub enum IterativeRagError {
    /// The query was empty or contained only whitespace.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// An underlying retrieval call failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
}
