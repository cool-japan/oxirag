//! Types for the `graph_summarization` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::super::graph_community::types::CommunityId;

// ── CommunitySummary ──────────────────────────────────────────────────────────

/// An extractive summary of a single community.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunitySummary {
    /// The community this summary describes.
    pub community_id: CommunityId,
    /// Short descriptive title (most-connected entity names).
    pub title: String,
    /// Extractive summary (relationship-phrase sentences).
    pub summary: String,
    /// Names of the most central entities in this community.
    pub key_entities: Vec<String>,
}

impl CommunitySummary {
    /// Return `true` when the summary text is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summary.is_empty()
    }
}

// ── SummaryReport ─────────────────────────────────────────────────────────────

/// The result of a global or local graph search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryReport {
    /// Final synthesised answer.
    pub answer: String,
    /// Community ids that contributed to this answer.
    pub used_communities: Vec<CommunityId>,
}

impl SummaryReport {
    /// Return `true` when no communities contributed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.answer.is_empty()
    }
}

// ── GraphSummarizationConfig ──────────────────────────────────────────────────

/// Configuration for the graph summarization pipeline.
#[derive(Debug, Clone)]
pub struct GraphSummarizationConfig {
    /// Maximum sentences per community summary.
    ///
    /// Defaults to `3`.
    pub max_summary_sentences: usize,
    /// Number of top-scoring communities to include in a global search.
    ///
    /// Defaults to `5`.
    pub top_communities: usize,
}

impl Default for GraphSummarizationConfig {
    fn default() -> Self {
        Self {
            max_summary_sentences: 3,
            top_communities: 5,
        }
    }
}

impl GraphSummarizationConfig {
    /// Set max summary sentences.
    #[must_use]
    pub fn with_max_summary_sentences(mut self, v: usize) -> Self {
        self.max_summary_sentences = v;
        self
    }

    /// Set top communities for global search.
    #[must_use]
    pub fn with_top_communities(mut self, v: usize) -> Self {
        self.top_communities = v;
        self
    }
}

// ── GraphSummarizationError ───────────────────────────────────────────────────

/// Errors from the `graph_summarization` module.
#[derive(Debug, Error)]
pub enum GraphSummarizationError {
    /// No community summaries were available to search.
    #[error("No community summaries available")]
    NoCommunities,
}
