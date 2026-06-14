//! Types for the `pairwise_rerank` module.
use std::collections::HashSet;
use std::fmt;

use thiserror::Error;

// ── helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── PairwiseConfig ─────────────────────────────────────────────────────────────

/// Configuration for the pairwise tournament reranker.
///
/// Controls the number of round-robin tournament passes and the tiebreaking
/// preference when two documents have equal token overlap with the query.
///
/// # Examples
///
/// ```
/// use oxirag::pairwise_rerank::PairwiseConfig;
///
/// let cfg = PairwiseConfig::new()
///     .with_tournament_rounds(3)
///     .with_prefer_longer(true);
/// assert_eq!(cfg.tournament_rounds, 3);
/// assert!(cfg.prefer_longer);
/// ```
#[derive(Debug, Clone)]
pub struct PairwiseConfig {
    /// Number of full round-robin tournament passes to perform.
    ///
    /// Higher values make the ranking more stable at the cost of O(n²·rounds)
    /// comparisons. Defaults to `2`.
    pub tournament_rounds: usize,
    /// Whether to prefer longer documents when token overlap is equal.
    ///
    /// When `true`, a tiebreak comparison gives the win to the longer document.
    /// When `false` the first document (lower index) wins on ties.
    ///
    /// Defaults to `false`.
    pub prefer_longer: bool,
}

impl Default for PairwiseConfig {
    fn default() -> Self {
        Self {
            tournament_rounds: 2,
            prefer_longer: false,
        }
    }
}

impl PairwiseConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of round-robin tournament passes.
    #[must_use]
    pub fn with_tournament_rounds(mut self, tournament_rounds: usize) -> Self {
        self.tournament_rounds = tournament_rounds;
        self
    }

    /// Set whether to prefer longer documents in tie situations.
    #[must_use]
    pub fn with_prefer_longer(mut self, prefer_longer: bool) -> Self {
        self.prefer_longer = prefer_longer;
        self
    }

    /// Effective number of rounds; at least 1 to guard against 0.
    #[must_use]
    pub fn effective_rounds(&self) -> usize {
        self.tournament_rounds.max(1)
    }
}

// ── PairwiseError ──────────────────────────────────────────────────────────────

/// Errors returned by the `pairwise_rerank` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PairwiseError {
    /// The document corpus was empty; nothing to rank.
    #[error("corpus must not be empty")]
    EmptyCorpus,
    /// `k` was zero; at least one result must be requested.
    #[error("k must be at least 1, got {0}")]
    InvalidK(usize),
}

// ── PairwiseHit ───────────────────────────────────────────────────────────────

/// A single result produced by pairwise tournament reranking.
///
/// Documents are sorted by `wins` descending; `score` normalises wins into
/// `[0.0, 1.0]` so callers can compare across different corpus sizes and round
/// counts.
#[derive(Debug, Clone)]
pub struct PairwiseHit {
    /// Identifier of the document (mirrors the `id` field of the input pair).
    pub id: String,
    /// Full text content of the document.
    pub content: String,
    /// Normalised relevance score in `[0.0, 1.0]`:
    /// `wins / (effective_rounds × (n − 1))`.
    pub score: f32,
    /// Raw win count accumulated over all tournament rounds.
    pub wins: usize,
}

// ── PairwiseScoredPair ────────────────────────────────────────────────────────

/// Intermediate result for a single pairwise comparison within a tournament.
///
/// `winner_idx` follows the same convention as [`PairwiseComparer::compare`]:
/// `0` means `doc_a` won; `1` means `doc_b` won.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairwiseScoredPair {
    /// Index of the first document in the corpus slice.
    pub doc_a_idx: usize,
    /// Index of the second document in the corpus slice.
    pub doc_b_idx: usize,
    /// `0` if `doc_a` won this comparison; `1` if `doc_b` won.
    pub winner_idx: usize,
}

// ── PairwiseComparer ──────────────────────────────────────────────────────────

/// An oracle that decides which of two documents is more relevant to a query.
///
/// Returns `0` if `doc_a` wins (is more relevant) and `1` if `doc_b` wins.
///
/// Implementations must also implement [`fmt::Debug`] so that
/// [`PairwiseReranker`](crate::pairwise_rerank::PairwiseReranker) can
/// derive a useful debug representation.
pub trait PairwiseComparer: fmt::Debug {
    /// Compare `doc_a` and `doc_b` with respect to `query`.
    ///
    /// # Returns
    ///
    /// - `0` — `doc_a` is more relevant.
    /// - `1` — `doc_b` is more relevant.
    fn compare(&self, query: &str, doc_a: &str, doc_b: &str) -> usize;
}

// ── MockPairwiseComparer ──────────────────────────────────────────────────────

/// A heuristic [`PairwiseComparer`] backed by token-overlap scoring.
///
/// Computes the intersection between the query's token set and each document's
/// token set. The document with the greater intersection wins. On a tie:
///
/// - If `prefer_longer` is `true`, the longer document (by byte length) wins.
/// - Otherwise `doc_a` wins (first-arg bias).
///
/// The tokeniser used is:
/// ```text
/// text.split(|c: char| !c.is_alphanumeric())
///     .filter(|t| t.len() >= 2)
///     .map(str::to_lowercase)
/// ```
#[derive(Debug, Clone, Default)]
pub struct MockPairwiseComparer {
    /// Prefer longer documents on tie.
    pub prefer_longer: bool,
}

impl MockPairwiseComparer {
    /// Create a new mock comparer.
    ///
    /// If `prefer_longer` is `true`, tiebreaks favour the document with more
    /// bytes; if `false` the first document (`doc_a`) wins on ties.
    #[must_use]
    pub fn new(prefer_longer: bool) -> Self {
        Self { prefer_longer }
    }

    fn overlap(query_tokens: &HashSet<String>, doc: &str) -> usize {
        let doc_tokens = tokenize(doc);
        query_tokens.intersection(&doc_tokens).count()
    }
}

impl PairwiseComparer for MockPairwiseComparer {
    fn compare(&self, query: &str, doc_a: &str, doc_b: &str) -> usize {
        let q_tokens = tokenize(query);
        let overlap_a = Self::overlap(&q_tokens, doc_a);
        let overlap_b = Self::overlap(&q_tokens, doc_b);

        if overlap_a > overlap_b {
            0
        } else if overlap_b > overlap_a {
            1
        } else if self.prefer_longer {
            // tiebreak: longer document wins; if equal length, doc_a wins
            usize::from(doc_b.len() > doc_a.len())
        } else {
            0
        }
    }
}
