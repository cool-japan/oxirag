//! Types for the `auto_merging` module.

use thiserror::Error;

// ── AutoMergeConfig ───────────────────────────────────────────────────────────

/// Configuration for the auto-merging retriever.
///
/// Controls the hierarchy granularity (parent / child window sizes), the
/// embedding dimension, and the fraction of a parent's children that must be
/// retrieved before those children collapse into their parent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoMergeConfig {
    /// Fraction of a parent's children that must appear in the retrieved leaf
    /// set before they are merged into the parent.
    ///
    /// Ranges over `(0.0, 1.0]`; a value of `0.5` merges once half of a
    /// parent's children are hit. Defaults to `0.5`.
    pub merge_threshold: f32,
    /// Target size, in whitespace-delimited words, of each parent window.
    ///
    /// Defaults to `80`.
    pub parent_words: usize,
    /// Target size, in whitespace-delimited words, of each child (leaf) window.
    ///
    /// Defaults to `20`.
    pub child_words: usize,
    /// Dimension of the deterministic lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for AutoMergeConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.5,
            parent_words: 80,
            child_words: 20,
            dim: 128,
        }
    }
}

impl AutoMergeConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the merge threshold (fraction of children that triggers a merge).
    #[must_use]
    pub fn with_merge_threshold(mut self, v: f32) -> Self {
        self.merge_threshold = v;
        self
    }

    /// Set the target parent-window size in words.
    #[must_use]
    pub fn with_parent_words(mut self, v: usize) -> Self {
        self.parent_words = v;
        self
    }

    /// Set the target child-window size in words.
    #[must_use]
    pub fn with_child_words(mut self, v: usize) -> Self {
        self.child_words = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}

// ── MergedHit ─────────────────────────────────────────────────────────────────

/// A single retrieval result emitted by the auto-merging retriever.
///
/// A hit is either a leaf chunk (when too few of its siblings were retrieved to
/// trigger a merge) or a merged ancestor chunk (when at least
/// [`AutoMergeConfig::merge_threshold`] of a parent's children were retrieved).
#[derive(Debug, Clone, PartialEq)]
pub struct MergedHit {
    /// The text of the emitted node (leaf text, or the merged parent text).
    pub text: String,
    /// Identifier of the emitted node within the [`ChunkHierarchy`].
    ///
    /// [`ChunkHierarchy`]: crate::auto_merging::ChunkHierarchy
    pub node_id: usize,
    /// Hierarchy level of the emitted node (`0` for leaves, higher for parents).
    pub level: usize,
    /// Relevance score of the hit (cosine similarity, higher is more relevant).
    ///
    /// For a merged parent this is the best score among its collapsed children.
    pub score: f32,
    /// Number of leaf children that collapsed into this hit.
    ///
    /// Equals `1` when the hit is an un-merged leaf, and is greater than `1`
    /// when children were merged into a parent.
    pub merged_from: usize,
}

// ── AutoMergeError ────────────────────────────────────────────────────────────

/// Errors from the `auto_merging` module.
#[derive(Debug, Error)]
pub enum AutoMergeError {
    /// The document supplied to `build` had no indexable content.
    #[error("document must not be empty")]
    EmptyDocument,
    /// The query supplied to `search` was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
    /// `search` was called before `build` populated the hierarchy.
    #[error("retriever not built")]
    NotBuilt,
}
