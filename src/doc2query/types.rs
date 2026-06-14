//! Types and traits for the `doc2query` module.
use crate::types::Document;
use thiserror::Error;

// ── Doc2QueryConfig ───────────────────────────────────────────────────────────

/// Configuration for [`Doc2QueryExpander`](crate::doc2query::Doc2QueryExpander).
///
/// Controls how many hypothetical questions are generated per document and which
/// template families are enabled.
#[derive(Debug, Clone)]
pub struct Doc2QueryConfig {
    /// Maximum number of hypothetical queries to generate per document.
    ///
    /// Defaults to `5`.
    pub num_queries: usize,
    /// Whether to emit definitional questions (e.g. `"What is <term>?"`).
    ///
    /// Defaults to `true`.
    pub include_definitional: bool,
    /// Whether to emit relational questions linking two entities.
    ///
    /// Defaults to `true`.
    pub include_relational: bool,
    /// Separator inserted between the original content and each appended query.
    ///
    /// Defaults to `"\n"`.
    pub append_separator: String,
}

impl Default for Doc2QueryConfig {
    fn default() -> Self {
        Self {
            num_queries: 5,
            include_definitional: true,
            include_relational: true,
            append_separator: "\n".to_string(),
        }
    }
}

impl Doc2QueryConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of hypothetical queries per document.
    #[must_use]
    pub fn with_num_queries(mut self, v: usize) -> Self {
        self.num_queries = v;
        self
    }

    /// Enable or disable definitional questions.
    #[must_use]
    pub fn with_include_definitional(mut self, v: bool) -> Self {
        self.include_definitional = v;
        self
    }

    /// Enable or disable relational questions.
    #[must_use]
    pub fn with_include_relational(mut self, v: bool) -> Self {
        self.include_relational = v;
        self
    }

    /// Set the separator inserted between the content and appended queries.
    #[must_use]
    pub fn with_append_separator(mut self, v: impl Into<String>) -> Self {
        self.append_separator = v.into();
        self
    }
}

// ── QueryGenerator ────────────────────────────────────────────────────────────

/// Generates hypothetical queries that a document answers.
///
/// This is the index-time analogue of query expansion: instead of expanding the
/// user query, it expands the document by attaching questions the passage would
/// be a good answer to (Nogueira et al., "Document Expansion by Query
/// Prediction").
pub trait QueryGenerator {
    /// Generate up to `n` hypothetical queries the document answers.
    fn generate(&self, doc: &Document, n: usize) -> Vec<String>;
}

// ── ExpandedDocument ──────────────────────────────────────────────────────────

/// A document paired with the hypothetical queries generated for it.
///
/// The [`expanded_content`](ExpandedDocument::expanded_content) field is the
/// original content followed by every generated query, each separated by the
/// configured [`append_separator`](Doc2QueryConfig::append_separator).
#[derive(Debug, Clone)]
pub struct ExpandedDocument {
    /// The original, unmodified document.
    pub original: Document,
    /// The hypothetical queries generated for the document.
    pub generated_queries: Vec<String>,
    /// The original content with the generated queries appended.
    pub expanded_content: String,
}

// ── Doc2QueryError ────────────────────────────────────────────────────────────

/// Errors from the `doc2query` module.
#[derive(Debug, Error)]
pub enum Doc2QueryError {
    /// The supplied document had no usable content.
    #[error("document must not be empty")]
    EmptyDocument,
}
