//! The `Query2Doc` expansion driver: combines a query with a generated
//! pseudo-document into a single retriever-ready string.

use super::types::{
    PseudoDocGenerator, PseudoDocument, Query2DocConfig, Query2DocError, Query2DocVariant,
};

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Truncates `text` to at most `max_len` `char`s, respecting `char` boundaries
/// (never splitting inside a multi-byte UTF-8 scalar value).
fn truncate_to_char_limit(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }
    text.chars().take(max_len).collect()
}

/// Builds `"<query> <query> ... <query>"` with `repetitions` copies of `query`
/// separated by a single space. `repetitions` is assumed to be `>= 1`
/// (enforced by [`Query2DocConfig::validate`]).
fn repeat_query(query: &str, repetitions: usize) -> String {
    let mut repeated = String::with_capacity(query.len().saturating_mul(repetitions) + repetitions);
    for i in 0..repetitions {
        if i > 0 {
            repeated.push(' ');
        }
        repeated.push_str(query);
    }
    repeated
}

// ── Query2DocExpander ─────────────────────────────────────────────────────────

/// Expands a user query into a pseudo-document-augmented string suitable for
/// BM25 (sparse) or dense-embedding retrieval.
///
/// Wraps a [`Query2DocConfig`] and a [`PseudoDocGenerator`]. For each query it:
///
/// 1. Asks the generator for a pseudo-document (a hypothetical passage that
///    would plausibly answer the query).
/// 2. Truncates the pseudo-document to
///    [`max_pseudo_len`](Query2DocConfig::max_pseudo_len) characters.
/// 3. Concatenates the (repetition-weighted) query with the pseudo-document,
///    per [`Query2DocConfig::variant`].
#[derive(Debug, Clone)]
pub struct Query2DocExpander<G: PseudoDocGenerator> {
    /// Expansion configuration.
    pub config: Query2DocConfig,
    /// Pseudo-document generator used to draft the hypothetical passage.
    pub generator: G,
}

impl<G: PseudoDocGenerator> Query2DocExpander<G> {
    /// Creates a new expander from a configuration and a pseudo-document
    /// generator.
    #[must_use]
    pub fn new(config: Query2DocConfig, generator: G) -> Self {
        Self { config, generator }
    }

    /// Expands `query` into a [`PseudoDocument`].
    ///
    /// # Errors
    ///
    /// - [`Query2DocError::InvalidConfig`] if [`self.config`](Self::config)
    ///   fails validation (see [`Query2DocConfig::validate`]).
    /// - [`Query2DocError::EmptyQuery`] if `query` is empty or contains only
    ///   whitespace after trimming.
    pub fn expand(&self, query: &str) -> Result<PseudoDocument, Query2DocError> {
        self.config.validate()?;

        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(Query2DocError::EmptyQuery);
        }

        let raw_pseudo_doc = self.generator.generate(trimmed);
        let pseudo_doc = truncate_to_char_limit(&raw_pseudo_doc, self.config.max_pseudo_len);

        let expanded_query = match self.config.variant {
            Query2DocVariant::Sparse => {
                let repeated_query = repeat_query(trimmed, self.config.query_repetitions);
                if pseudo_doc.is_empty() {
                    repeated_query
                } else {
                    format!("{repeated_query} {pseudo_doc}")
                }
            }
            Query2DocVariant::Dense => {
                if pseudo_doc.is_empty() {
                    trimmed.to_string()
                } else {
                    format!("{trimmed} [SEP] {pseudo_doc}")
                }
            }
        };

        Ok(PseudoDocument {
            query: trimmed.to_string(),
            pseudo_doc,
            expanded_query,
        })
    }
}
