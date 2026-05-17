//! Shared types for the advanced retrieval module.

use serde::{Deserialize, Serialize};

/// Error type for advanced retrieval operations.
#[derive(Debug, thiserror::Error)]
pub enum AdvancedRetrievalError {
    /// Wraps an embedding or embedding-provider error.
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Wraps a search / vector-store error.
    #[error("Search error: {0}")]
    Search(String),

    /// Internal / unexpected error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Strategy selector for choosing between advanced retrieval algorithms at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    /// Reciprocal Rank Fusion across multiple query variants.
    RagFusion,
    /// Hypothetical Document Embedding retrieval.
    Hyde,
    /// Maximal Marginal Relevance reranking.
    Mmr,
}

/// Common retrieval configuration shared across strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Optional minimum score threshold.
    pub min_score: Option<f32>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            min_score: None,
        }
    }
}

impl RetrievalConfig {
    /// Set the maximum number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum score threshold.
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}
