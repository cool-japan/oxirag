//! Types for the `listwise_rerank` module.
use crate::types::Document;
use thiserror::Error;

// ── WindowConfig ──────────────────────────────────────────────────────────────

/// Configuration for the `RankGPT` sliding-window listwise reranker.
///
/// The reranker slides a window of [`window_size`](WindowConfig::window_size)
/// candidates from the back of the ranking to the front in steps of
/// [`step`](WindowConfig::step), permuting each window with a
/// [`ListwiseJudge`].
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Number of candidates considered together in each window.
    ///
    /// Defaults to `4`.
    pub window_size: usize,
    /// How far the window advances toward the front of the list per pass.
    ///
    /// A value of `0` is treated as `1` to avoid an infinite loop.
    ///
    /// Defaults to `2`.
    pub step: usize,
    /// Maximum number of results to return. `0` means return all.
    ///
    /// Defaults to `0`.
    pub top_n: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_size: 4,
            step: 2,
            top_n: 0,
        }
    }
}

impl WindowConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the sliding-window size.
    #[must_use]
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the window step (advance per pass).
    #[must_use]
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Set how many results to return (`0` = all).
    #[must_use]
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Effective window size, clamped to at least `1`.
    #[must_use]
    pub fn effective_window(&self) -> usize {
        self.window_size.max(1)
    }

    /// Effective step, clamped to at least `1` (guards against `step == 0`).
    #[must_use]
    pub fn effective_step(&self) -> usize {
        self.step.max(1)
    }
}

// ── ListwiseResult ────────────────────────────────────────────────────────────

/// A single result produced by listwise reranking.
#[derive(Debug, Clone)]
pub struct ListwiseResult {
    /// The underlying document.
    pub document: Document,
    /// Score from the retrieval step that produced the candidate.
    pub original_score: f32,
    /// Rank before reranking (0-indexed).
    pub original_rank: usize,
    /// Rank after reranking (0-indexed).
    pub new_rank: usize,
}

// ── ListwiseError ─────────────────────────────────────────────────────────────

/// Errors from the `listwise_rerank` module.
#[derive(Debug, Error)]
pub enum ListwiseError {
    /// The query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
    /// No candidate results were supplied.
    #[error("candidates must not be empty")]
    EmptyCandidates,
}

// ── ListwiseJudge ─────────────────────────────────────────────────────────────

/// A listwise judge that orders a window of candidates.
///
/// Unlike a pointwise or pairwise scorer, a listwise judge observes all
/// documents in the window simultaneously and returns a *permutation*: the
/// window-local indices reordered from most relevant to least relevant.
pub trait ListwiseJudge {
    /// Return window-local indices in best→worst order.
    ///
    /// The returned vector is a permutation of `0..docs.len()`.
    fn permute(&self, query: &str, docs: &[Document]) -> Vec<usize>;
}
