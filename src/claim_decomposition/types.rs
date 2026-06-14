//! Types for the `claim_decomposition` module.

use thiserror::Error;

// ── AtomicClaim ───────────────────────────────────────────────────────────────

/// A single atomic, self-contained, verifiable fact extracted from an answer.
///
/// Each `AtomicClaim` corresponds to one clause of the source answer that has
/// been (optionally) decontextualized so that it stands alone without relying on
/// surrounding sentences. This mirrors the `FActScore` decomposition step of
/// Min et al. (2023), where a generated answer is broken into independently
/// verifiable atomic facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicClaim {
    /// The atomic claim text.
    pub text: String,
    /// Index of the source sentence this claim was derived from (0-based).
    pub source_sentence: usize,
    /// Whether a leading pronoun was rewritten to the answer's main subject.
    pub decontextualized: bool,
}

impl AtomicClaim {
    /// Construct a new `AtomicClaim`.
    #[must_use]
    pub fn new(text: impl Into<String>, source_sentence: usize, decontextualized: bool) -> Self {
        Self {
            text: text.into(),
            source_sentence,
            decontextualized,
        }
    }
}

// ── ClaimDecompConfig ─────────────────────────────────────────────────────────

/// Configuration for atomic claim decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimDecompConfig {
    /// Minimum number of tokens a clause must contain to become a claim.
    ///
    /// Defaults to `3`.
    pub min_tokens: usize,
    /// When `true`, a leading pronoun is rewritten to the answer's main subject.
    ///
    /// Defaults to `true`.
    pub decontextualize: bool,
    /// Maximum number of atomic claims retained per answer.
    ///
    /// Defaults to `64`.
    pub max_claims: usize,
}

impl Default for ClaimDecompConfig {
    fn default() -> Self {
        Self {
            min_tokens: 3,
            decontextualize: true,
            max_claims: 64,
        }
    }
}

impl ClaimDecompConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum number of tokens a clause must contain.
    #[must_use]
    pub fn with_min_tokens(mut self, min_tokens: usize) -> Self {
        self.min_tokens = min_tokens;
        self
    }

    /// Set whether a leading pronoun is rewritten to the answer's main subject.
    #[must_use]
    pub fn with_decontextualize(mut self, decontextualize: bool) -> Self {
        self.decontextualize = decontextualize;
        self
    }

    /// Set the maximum number of atomic claims retained per answer.
    #[must_use]
    pub fn with_max_claims(mut self, max_claims: usize) -> Self {
        self.max_claims = max_claims;
        self
    }
}

// ── ClaimDecompError ──────────────────────────────────────────────────────────

/// Errors from the `claim_decomposition` module.
#[derive(Debug, Error)]
pub enum ClaimDecompError {
    /// The answer string was empty.
    #[error("answer must not be empty")]
    EmptyAnswer,
}
