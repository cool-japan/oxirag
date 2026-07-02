//! Core types and traits for the `query2doc` module.
//!
//! This file defines the configuration ([`Query2DocConfig`], [`Query2DocVariant`]),
//! the error type ([`Query2DocError`]), the pseudo-document generation contract
//! ([`PseudoDocGenerator`]) together with its deterministic reference
//! implementation ([`MockPseudoDocGenerator`]), and the expansion result type
//! ([`PseudoDocument`]).

use std::collections::HashMap;
use thiserror::Error;

// ── Query2DocVariant ──────────────────────────────────────────────────────────

/// Selects how the original query is combined with the generated pseudo-document.
///
/// `Query2Doc` (Wang et al., EMNLP 2023) prescribes two combination strategies
/// depending on the downstream retriever:
///
/// - [`Sparse`](Query2DocVariant::Sparse) — the query is **repeated**
///   [`query_repetitions`](Query2DocConfig::query_repetitions) times before the
///   pseudo-document. Sparse retrievers such as BM25 score documents by
///   term-frequency overlap, so repeating the (short) query keeps its terms
///   from being diluted by the (much longer) pseudo-document.
/// - [`Dense`](Query2DocVariant::Dense) — the query appears **once**, followed
///   by a literal `[SEP]` token and the pseudo-document. Dense (embedding)
///   retrievers are not term-frequency sensitive, so repetition is unnecessary
///   and would only waste context budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Query2DocVariant {
    /// Repeat the query `query_repetitions` times, then append the
    /// pseudo-document. Intended for sparse (BM25-style) retrieval.
    #[default]
    Sparse,
    /// `<query> [SEP] <pseudo_doc>`, with the query appearing exactly once.
    /// Intended for dense (embedding-based) retrieval.
    Dense,
}

impl Query2DocVariant {
    /// Returns a stable lower-case identifier for the variant.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Dense => "dense",
        }
    }
}

impl std::fmt::Display for Query2DocVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Query2DocConfig ───────────────────────────────────────────────────────────

/// Configuration for [`Query2DocExpander`](crate::query2doc::Query2DocExpander).
#[derive(Debug, Clone, PartialEq)]
pub struct Query2DocConfig {
    /// Number of times the original query is repeated ahead of the
    /// pseudo-document in the [`Sparse`](Query2DocVariant::Sparse) variant.
    ///
    /// Ignored by the [`Dense`](Query2DocVariant::Dense) variant, which always
    /// emits the query exactly once regardless of this value.
    ///
    /// Defaults to `5`, matching the original `Query2Doc` paper's BM25 setup.
    pub query_repetitions: usize,
    /// Maximum length, in `char`s, of the generated pseudo-document.
    ///
    /// Pseudo-documents longer than this are truncated (on a `char` boundary)
    /// before being concatenated with the query. Defaults to `400`.
    pub max_pseudo_len: usize,
    /// Which combination strategy to use when building the expanded query.
    ///
    /// Defaults to [`Query2DocVariant::Sparse`].
    pub variant: Query2DocVariant,
}

impl Default for Query2DocConfig {
    fn default() -> Self {
        Self {
            query_repetitions: 5,
            max_pseudo_len: 400,
            variant: Query2DocVariant::Sparse,
        }
    }
}

impl Query2DocConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of query repetitions used by the sparse variant.
    #[must_use]
    pub fn with_query_repetitions(mut self, n: usize) -> Self {
        self.query_repetitions = n;
        self
    }

    /// Sets the maximum pseudo-document length, in characters.
    #[must_use]
    pub fn with_max_pseudo_len(mut self, n: usize) -> Self {
        self.max_pseudo_len = n;
        self
    }

    /// Sets the combination variant (sparse vs. dense).
    #[must_use]
    pub fn with_variant(mut self, variant: Query2DocVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// - [`Query2DocError::InvalidConfig`] when `query_repetitions` is `0`
    ///   (a query that is never repeated provides no term-weighting benefit
    ///   and is almost certainly a misconfiguration).
    /// - [`Query2DocError::InvalidConfig`] when `max_pseudo_len` is `0` (no
    ///   pseudo-document could ever survive truncation).
    pub fn validate(&self) -> Result<(), Query2DocError> {
        if self.query_repetitions == 0 {
            return Err(Query2DocError::InvalidConfig(
                "query_repetitions must be greater than zero".to_string(),
            ));
        }
        if self.max_pseudo_len == 0 {
            return Err(Query2DocError::InvalidConfig(
                "max_pseudo_len must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

// ── Query2DocError ────────────────────────────────────────────────────────────

/// Errors produced by the `query2doc` module.
#[derive(Debug, Error, PartialEq, Clone)]
pub enum Query2DocError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// The [`Query2DocConfig`] failed validation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

// ── PseudoDocGenerator ────────────────────────────────────────────────────────

/// Generates a pseudo-document: a hypothetical passage that would plausibly
/// answer a given query.
///
/// Implementations are typically backed by an LLM in production; this crate
/// ships [`MockPseudoDocGenerator`], a fully deterministic reference
/// implementation suitable for tests and offline pipelines.
pub trait PseudoDocGenerator {
    /// Generates a pseudo-document for `query`.
    ///
    /// Implementations should return an empty string for an empty or
    /// whitespace-only query rather than fabricating unrelated content.
    fn generate(&self, query: &str) -> String;
}

// ── MockPseudoDocGenerator ────────────────────────────────────────────────────

/// Words excluded when extracting "salient" keywords from a query.
///
/// Deliberately small and specific to short question-style queries, mirroring
/// the stop-word filter used by the `advanced_retrieval::hyde` mock generator.
const MOCK_STOP_WORDS: &[&str] = &[
    "what", "is", "are", "was", "were", "be", "been", "being", "the", "a", "an", "how", "why",
    "when", "where", "who", "which", "does", "do", "did", "can", "could", "would", "should",
    "will", "shall", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "and", "or",
    "but", "this", "that", "these", "those",
];

/// Extracts deterministic, order-preserving "salient" keywords from `query`.
///
/// A token is kept when, after stripping non-alphanumeric characters, it is
/// longer than two characters and is not a member of [`MOCK_STOP_WORDS`].
fn extract_keywords(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .filter_map(|word| {
            let stripped: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
            let lower = stripped.to_lowercase();
            if stripped.len() > 2 && !MOCK_STOP_WORDS.contains(&lower.as_str()) {
                Some(stripped)
            } else {
                None
            }
        })
        .collect()
}

/// A deterministic, offline [`PseudoDocGenerator`] reference implementation.
///
/// No LLM, network access, or randomness is involved: the generator either
/// (a) looks up a canned fact for the first recognised keyword in an optional
/// `knowledge` map, or (b) falls back to a fixed template built from the
/// query's salient keywords. Given the same input it always produces the same
/// output, which makes it suitable for doctests and unit tests that assert on
/// exact strings.
#[derive(Debug, Clone, Default)]
pub struct MockPseudoDocGenerator {
    /// Canned facts keyed by lower-cased keyword.
    ///
    /// When a query contains a keyword present in this map (case-insensitive),
    /// the generator emits `"{keyword} {fact}"` instead of the generic
    /// fallback template.
    pub knowledge: HashMap<String, String>,
}

impl MockPseudoDocGenerator {
    /// Creates a generator with an empty knowledge map (always uses the
    /// generic fallback template).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a canned fact for `term` (case-insensitive lookup).
    #[must_use]
    pub fn with_knowledge(mut self, term: impl Into<String>, fact: impl Into<String>) -> Self {
        self.knowledge
            .insert(term.into().to_lowercase(), fact.into());
        self
    }
}

impl PseudoDocGenerator for MockPseudoDocGenerator {
    fn generate(&self, query: &str) -> String {
        let query = query.trim();
        if query.is_empty() {
            return String::new();
        }

        let keywords = extract_keywords(query);

        if let Some(term) = keywords
            .iter()
            .find(|k| self.knowledge.contains_key(&k.to_lowercase()))
        {
            // `contains_key` above guarantees this lookup succeeds.
            if let Some(fact) = self.knowledge.get(&term.to_lowercase()) {
                return format!("{term} {fact}");
            }
        }

        if keywords.is_empty() {
            return format!(
                "This passage provides background information relevant to the question \"{query}\"."
            );
        }

        format!(
            "{} is closely related to {}. This passage elaborates on {} and provides the context needed to answer \"{query}\".",
            keywords[0],
            keywords.join(", "),
            keywords[0],
        )
    }
}

// ── PseudoDocument ────────────────────────────────────────────────────────────

/// The result of expanding a query with a pseudo-document.
///
/// Produced by
/// [`Query2DocExpander::expand`](crate::query2doc::Query2DocExpander::expand).
#[derive(Debug, Clone, PartialEq)]
pub struct PseudoDocument {
    /// The original (trimmed) query text.
    pub query: String,
    /// The generated pseudo-document, after truncation to
    /// [`Query2DocConfig::max_pseudo_len`].
    pub pseudo_doc: String,
    /// The final string to feed to the retriever: the query (repeated for the
    /// sparse variant, or singular with a `[SEP]` marker for the dense
    /// variant) concatenated with `pseudo_doc`.
    pub expanded_query: String,
}
