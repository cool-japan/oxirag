//! Types and traits for the `multi_query` module.
//!
//! Multi-Query Retrieval expands a single query into `num_variants` paraphrase
//! variants, retrieves candidates for each variant independently, and fuses the
//! per-variant ranked lists using Reciprocal Rank Fusion (RRF).
//!
//! This module defines the data types, the [`QueryVariantGenerator`] trait, and
//! the deterministic [`MockQueryVariantGenerator`] used for testing.

use thiserror::Error;

// ── MultiQueryConfig ──────────────────────────────────────────────────────────

/// Configuration for [`MultiQueryGenerator`](crate::multi_query::generator::MultiQueryGenerator).
///
/// All fields have sensible defaults that follow the original Multi-Query
/// Retrieval literature:
///
/// | Field | Default | Meaning |
/// |-------|---------|---------|
/// | `num_variants` | 3 | Query variants generated per retrieval call |
/// | `top_k` | 5 | Documents retrieved per variant before fusion |
/// | `rrf_k` | 60 | RRF smoothing constant (`1/(k+rank)`) |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiQueryConfig {
    /// Number of query variants to generate from the original query.
    ///
    /// Defaults to `3`.
    pub num_variants: usize,
    /// Maximum number of documents to retrieve per variant query before fusion.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// RRF smoothing constant `k`.
    ///
    /// Larger values reduce the reward given to very high-ranked documents.
    /// The original RRF paper recommends `60`. Defaults to `60`.
    pub rrf_k: usize,
}

impl Default for MultiQueryConfig {
    fn default() -> Self {
        Self {
            num_variants: 3,
            top_k: 5,
            rrf_k: 60,
        }
    }
}

impl MultiQueryConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of query variants to generate.
    #[must_use]
    pub fn with_num_variants(mut self, num_variants: usize) -> Self {
        self.num_variants = num_variants;
        self
    }

    /// Set the per-variant retrieval depth.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the RRF smoothing constant.
    #[must_use]
    pub fn with_rrf_k(mut self, rrf_k: usize) -> Self {
        self.rrf_k = rrf_k;
        self
    }
}

// ── MultiQueryError ───────────────────────────────────────────────────────────

/// Errors produced by the `multi_query` module.
#[derive(Debug, Error)]
pub enum MultiQueryError {
    /// Query variant generation failed (e.g. the input query was empty).
    #[error("query variant generation failed: {0}")]
    GenerationFailed(String),
    /// Document retrieval failed for a variant query.
    #[error("retrieval failed: {0}")]
    RetrievalFailed(String),
    /// The document corpus is empty — cannot retrieve from an empty collection.
    #[error("document corpus is empty")]
    EmptyCorpus,
}

// ── GeneratedQuery ────────────────────────────────────────────────────────────

/// A single generated query variant produced by a [`QueryVariantGenerator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedQuery {
    /// The text of this query variant.
    pub text: String,
    /// Zero-based index of this variant among all generated variants.
    pub variant_idx: usize,
}

impl GeneratedQuery {
    /// Create a new generated query.
    #[must_use]
    pub fn new(text: impl Into<String>, variant_idx: usize) -> Self {
        Self {
            text: text.into(),
            variant_idx,
        }
    }
}

// ── MultiQueryHit ─────────────────────────────────────────────────────────────

/// A single fused retrieval result after Reciprocal Rank Fusion.
///
/// The [`rrf_score`](Self::rrf_score) accumulates `1/(rrf_k + rank)` from
/// every variant query that returned this document in its top-`k` list.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiQueryHit {
    /// Document identifier.
    pub id: String,
    /// Document content.
    pub content: String,
    /// Accumulated Reciprocal Rank Fusion score.
    pub rrf_score: f64,
}

impl MultiQueryHit {
    /// Create a new hit with the given id, content, and RRF score.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>, rrf_score: f64) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            rrf_score,
        }
    }
}

// ── MultiQueryResult ──────────────────────────────────────────────────────────

/// The complete result of a multi-query retrieval run.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiQueryResult {
    /// The original query that was expanded.
    pub original_query: String,
    /// The generated query variants used for retrieval.
    pub generated_queries: Vec<GeneratedQuery>,
    /// Fused retrieval results, sorted by descending RRF score.
    pub hits: Vec<MultiQueryHit>,
}

// ── QueryVariantGenerator trait ───────────────────────────────────────────────

/// A strategy for generating query variants from an original query.
///
/// Implementations drive the expansion phase of
/// [`MultiQueryGenerator`](crate::multi_query::generator::MultiQueryGenerator).
/// Given an original query string and a requested count `n`, the generator
/// must return a `Vec` of exactly `n` strings (or fewer if it is unable to
/// produce the requested count).
///
/// The variants should semantically cover the same intent as the original query
/// while varying in lexical form, allowing retrieval to cast a wider net over
/// the document corpus.
pub trait QueryVariantGenerator {
    /// Generate up to `n` query variants from `query`.
    ///
    /// # Errors
    ///
    /// Returns [`MultiQueryError::GenerationFailed`] when variant generation
    /// encounters an unrecoverable error (e.g. empty input query).
    fn generate_variants(&self, query: &str, n: usize) -> Result<Vec<String>, MultiQueryError>;
}

// ── MockQueryVariantGenerator ─────────────────────────────────────────────────

/// A deterministic [`QueryVariantGenerator`] for tests and examples.
///
/// Produces variants by prepending a cycling list of heuristic prefixes to the
/// original query:
///
/// 1. `"What is <query>"`
/// 2. `"Describe <query>"`
/// 3. `"Explain <query>"`
/// 4. `"Tell me about <query>"`
/// 5. `"Summarize <query>"`
/// 6. `"Elaborate on <query>"`
///
/// Once the six built-in prefixes are exhausted the generator falls back to a
/// numeric suffix pattern `"<query> [variant N]"` so that arbitrarily large `n`
/// values are satisfied without failure.
///
/// Returns [`MultiQueryError::GenerationFailed`] if `query` is empty after
/// trimming.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockQueryVariantGenerator;

impl MockQueryVariantGenerator {
    /// Create a new mock variant generator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The ordered heuristic prefixes used by this mock.
    #[must_use]
    pub fn prefixes() -> &'static [&'static str] {
        &[
            "What is",
            "Describe",
            "Explain",
            "Tell me about",
            "Summarize",
            "Elaborate on",
        ]
    }
}

impl QueryVariantGenerator for MockQueryVariantGenerator {
    fn generate_variants(&self, query: &str, n: usize) -> Result<Vec<String>, MultiQueryError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(MultiQueryError::GenerationFailed(
                "query must not be empty".to_string(),
            ));
        }

        let prefixes = Self::prefixes();
        let mut variants = Vec::with_capacity(n);

        for i in 0..n {
            let text = if i < prefixes.len() {
                format!("{} {query}", prefixes[i])
            } else {
                format!("{query} [variant {i}]")
            };
            variants.push(text);
        }

        Ok(variants)
    }
}
