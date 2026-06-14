//! Core types for the quote-grounding module.
//!
//! Defines the configuration, the per-claim grounded-quote record, and the
//! error variants used by the grounding pipeline. The actual grounding logic
//! lives in the [`grounder`](super::grounder) submodule.

use thiserror::Error;

use crate::types::DocumentId;

// ── GroundedQuote ─────────────────────────────────────────────────────────────

/// A single claim paired with the minimal verbatim quote that supports it.
///
/// The `quote` field is always a contiguous, verbatim span taken from the
/// content of the source document identified by `source_id`. It is the source
/// sentence with maximal token overlap against the claim, optionally trimmed to
/// a window of at most `QuoteConfig::max_quote_tokens` tokens centred on the
/// overlapping span.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundedQuote {
    /// The claim sentence taken from the generated answer.
    pub claim: String,
    /// The minimal verbatim supporting quote drawn from the source document.
    pub quote: String,
    /// Identifier of the source document the quote was taken from.
    pub source_id: DocumentId,
    /// Overlap score between the claim and the full source sentence, in `[0, 1]`.
    pub score: f32,
}

impl GroundedQuote {
    /// Build a new [`GroundedQuote`].
    #[must_use]
    pub fn new(
        claim: impl Into<String>,
        quote: impl Into<String>,
        source_id: impl Into<DocumentId>,
        score: f32,
    ) -> Self {
        Self {
            claim: claim.into(),
            quote: quote.into(),
            source_id: source_id.into(),
            score,
        }
    }
}

// ── QuoteConfig ───────────────────────────────────────────────────────────────

/// Configuration knobs for the quote-grounding pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteConfig {
    /// Minimum overlap score a source sentence must reach to ground a claim.
    ///
    /// Scores below this threshold are treated as "no supporting quote".
    /// Defaults to `0.3`.
    pub min_support: f32,
    /// Maximum number of tokens retained when trimming a long source sentence.
    ///
    /// The retained window is centred on the tokens that overlap the claim.
    /// Defaults to `30`.
    pub max_quote_tokens: usize,
}

impl Default for QuoteConfig {
    fn default() -> Self {
        Self {
            min_support: 0.3,
            max_quote_tokens: 30,
        }
    }
}

impl QuoteConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the minimum support threshold.
    #[must_use]
    pub fn with_min_support(mut self, min_support: f32) -> Self {
        self.min_support = min_support;
        self
    }

    /// Override the maximum quote length in tokens.
    #[must_use]
    pub fn with_max_quote_tokens(mut self, max_quote_tokens: usize) -> Self {
        self.max_quote_tokens = max_quote_tokens;
        self
    }
}

// ── QuoteError ────────────────────────────────────────────────────────────────

/// Errors returned by the quote-grounding pipeline.
#[derive(Debug, Error)]
pub enum QuoteError {
    /// The supplied answer was empty after trimming.
    #[error("answer must not be empty")]
    EmptyAnswer,
}
