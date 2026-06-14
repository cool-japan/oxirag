//! Types for the `context_pruning` module.

use thiserror::Error;

// ── PruneConfig ───────────────────────────────────────────────────────────────

/// Configuration for token-level context pruning.
///
/// `target_ratio` is the fraction of tokens to **keep** (not drop): a value of
/// `0.5` keeps roughly half of the words. The kept count is additionally floored
/// at `min_tokens` so that very short contexts retain enough signal to remain
/// useful.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneConfig {
    /// Fraction of tokens to keep, in `[0.0, 1.0]`.
    ///
    /// Defaults to `0.5` (keep half). Values are clamped into range when used.
    pub target_ratio: f32,
    /// Minimum number of tokens to keep regardless of `target_ratio`.
    ///
    /// Defaults to `5`.
    pub min_tokens: usize,
    /// Whether capitalized entities are always preserved.
    ///
    /// When `true`, words whose first alphabetic character is uppercase are kept
    /// even if their rarity score would otherwise drop them. Defaults to `true`.
    pub preserve_entities: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.5,
            min_tokens: 5,
            preserve_entities: true,
        }
    }
}

impl PruneConfig {
    /// Create a [`PruneConfig`] with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target keep ratio (fraction of tokens to keep).
    #[must_use]
    pub fn with_target_ratio(mut self, ratio: f32) -> Self {
        self.target_ratio = ratio;
        self
    }

    /// Set the minimum number of tokens to keep.
    #[must_use]
    pub fn with_min_tokens(mut self, min_tokens: usize) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    /// Set whether capitalized entities are always preserved.
    #[must_use]
    pub fn with_preserve_entities(mut self, preserve_entities: bool) -> Self {
        self.preserve_entities = preserve_entities;
        self
    }
}

// ── PrunedContext ─────────────────────────────────────────────────────────────

/// The result of pruning a context to a target token budget.
///
/// The kept words appear in [`PrunedContext::text`] in their original relative
/// order, joined by single spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunedContext {
    /// The pruned text: kept words in original order joined by spaces.
    pub text: String,
    /// Number of words kept.
    pub kept_tokens: usize,
    /// Number of words in the original context.
    pub original_tokens: usize,
    /// Compression ratio (`kept_tokens / original_tokens`); `0.0` when original is 0.
    pub compression_ratio: f32,
}

impl PrunedContext {
    /// Return `true` if no tokens were kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kept_tokens == 0
    }
}

// ── ContextPruningError ───────────────────────────────────────────────────────

/// Errors from the `context_pruning` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ContextPruningError {
    /// The context contained no words.
    #[error("context must not be empty")]
    EmptyContext,
}
