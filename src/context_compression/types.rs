//! Types for the `context_compression` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::SearchResult;

// ── CompressedContext ─────────────────────────────────────────────────────────

/// A compressed context ready to pass to a generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedContext {
    /// The compressed text (sentences joined by space).
    pub text: String,
    /// The sentences that were kept after scoring and redundancy filtering.
    pub kept_sentences: Vec<String>,
    /// Approximate token count of the original context.
    pub original_tokens: usize,
    /// Approximate token count of the compressed context.
    pub compressed_tokens: usize,
    /// Compression ratio (`compressed / original`); `0.0` when original is 0.
    pub ratio: f32,
}

impl CompressedContext {
    /// Return `true` if the compressed context is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

// ── CompressionConfig ─────────────────────────────────────────────────────────

/// Configuration for the context compression engine.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Approximate token budget for the compressed output.
    ///
    /// Defaults to `512`.
    pub token_budget: usize,
    /// Minimum relevance score for a sentence to be kept.
    ///
    /// Defaults to `0.1`.
    pub relevance_threshold: f32,
    /// Jaccard threshold above which two sentences are considered near-duplicates.
    ///
    /// Defaults to `0.8`.
    pub redundancy_threshold: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            token_budget: 512,
            relevance_threshold: 0.1,
            redundancy_threshold: 0.8,
        }
    }
}

impl CompressionConfig {
    /// Set the token budget.
    #[must_use]
    pub fn with_token_budget(mut self, v: usize) -> Self {
        self.token_budget = v;
        self
    }

    /// Set the relevance threshold.
    #[must_use]
    pub fn with_relevance_threshold(mut self, v: f32) -> Self {
        self.relevance_threshold = v;
        self
    }

    /// Set the redundancy threshold.
    #[must_use]
    pub fn with_redundancy_threshold(mut self, v: f32) -> Self {
        self.redundancy_threshold = v;
        self
    }
}

// ── ContextCompressor trait ───────────────────────────────────────────────────

/// Trait for synchronous context compressors.
///
/// Implementations must be **pure compute** — no I/O, no async.
pub trait ContextCompressor {
    /// Compress the context in `sources` relevant to `query` within `budget` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`CompressionError::EmptyInput`] when `sources` is empty.
    fn compress(
        &self,
        query: &str,
        sources: &[SearchResult],
        config: &CompressionConfig,
    ) -> Result<CompressedContext, CompressionError>;
}

// ── CompressionError ──────────────────────────────────────────────────────────

/// Errors from the `context_compression` module.
#[derive(Debug, Error)]
pub enum CompressionError {
    /// The source list was empty.
    #[error("Input must not be empty")]
    EmptyInput,
}
