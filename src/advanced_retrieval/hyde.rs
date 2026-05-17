//! `HyDE`: Hypothetical Document Embedding retrieval.
//!
//! `HyDE` ("Hypothetical Document Embeddings", Gao et al. 2022) improves
//! retrieval quality by generating a *hypothetical* document that would
//! plausibly answer the query and then using *that document's* embedding as
//! the search vector rather than the raw query embedding.
//!
//! This implementation generates the hypothetical document using a lightweight
//! heuristic expansion — no LLM inference is required — making it suitable for
//! pure-Rust, offline usage.  The expansion produces a grammatically plausible
//! but content-agnostic answer stub; the value comes from the dense embedding
//! of the longer, answer-shaped text.

use crate::layer1_echo::traits::Echo;
use crate::types::SearchResult;

use super::types::AdvancedRetrievalError;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the [`HydeRetrieval`] strategy.
#[derive(Debug, Clone)]
pub struct HydeConfig {
    /// Number of results to return.
    ///
    /// Defaults to `10`.
    pub top_k: usize,

    /// Prefix prepended to the query when constructing the hypothetical
    /// document.
    ///
    /// Defaults to `"Answer to: "`.
    pub hypothetical_prefix: String,
}

impl Default for HydeConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            hypothetical_prefix: "Answer to: ".to_string(),
        }
    }
}

impl HydeConfig {
    /// Set the number of results to return.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the hypothetical prefix string.
    #[must_use]
    pub fn with_hypothetical_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.hypothetical_prefix = prefix.into();
        self
    }
}

// ── HydeRetrieval ─────────────────────────────────────────────────────────────

/// Retriever that implements the Hypothetical Document Embedding strategy.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "advanced-retrieval")]
/// # {
/// use oxirag::advanced_retrieval::{HydeRetrieval, HydeConfig};
///
/// let hyde = HydeRetrieval::new(HydeConfig::default());
/// let hypothetical = hyde.generate_hypothetical_doc("What is Rust?");
/// assert!(!hypothetical.is_empty());
/// # }
/// ```
pub struct HydeRetrieval {
    config: HydeConfig,
}

impl HydeRetrieval {
    /// Create a new [`HydeRetrieval`] with the given configuration.
    #[must_use]
    pub fn new(config: HydeConfig) -> Self {
        Self { config }
    }

    /// Generate a hypothetical document by expanding `query` into a
    /// plausible answer stub.
    ///
    /// The expansion uses a template of the form:
    ///
    /// ```text
    /// {prefix}{query}. This is because [background context].
    /// In particular, {query} can be understood as follows.
    /// The key aspects are: {keyword1}, {keyword2}, ...
    /// ```
    ///
    /// The resulting text is longer and more answer-shaped than the bare
    /// query, which shifts its embedding toward the "answer space" in the
    /// vector store.
    #[must_use]
    pub fn generate_hypothetical_doc(&self, query: &str) -> String {
        let query = query.trim();
        let prefix = self.config.hypothetical_prefix.as_str();

        // Extract content words (skip stop-words) for keyword padding.
        let stop_words = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must",
            "can", "could", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "about", "what", "how", "why", "when", "where", "who", "which",
            "that", "this", "these", "those", "and", "or", "but", "nor", "so", "yet",
        ];

        let keywords: Vec<&str> = query
            .split_whitespace()
            .filter(|w| {
                let lw = w.to_lowercase();
                let stripped = lw.trim_matches(|c: char| !c.is_alphanumeric());
                !stop_words.contains(&stripped) && stripped.len() > 2
            })
            .collect();

        // Build a multi-sentence answer stub.
        let mut doc = String::with_capacity(query.len() * 6);

        // Opening sentence using the configured prefix.
        doc.push_str(prefix);
        doc.push_str(query);
        doc.push_str(". This is because [background context].");

        // Second sentence referencing the query.
        doc.push_str(" In particular, ");
        doc.push_str(query);
        doc.push_str(" can be understood as follows.");

        // Keyword enumeration to anchor the embedding.
        if !keywords.is_empty() {
            doc.push_str(" The key aspects are: ");
            doc.push_str(&keywords.join(", "));
            doc.push('.');
        }

        // Closing elaboration to add length / context.
        doc.push_str(" This topic involves multiple interconnected concepts");
        if keywords.len() >= 2 {
            doc.push_str(", including ");
            // repeat the first two keywords for emphasis
            doc.push_str(keywords[0]);
            doc.push_str(" and ");
            doc.push_str(keywords[keywords.len() - 1]);
        }
        doc.push_str(", which are relevant to answering the original question.");

        doc
    }

    /// Retrieve documents using the hypothetical document's text as the
    /// search query (the [`Echo`] layer will embed the expanded text before
    /// searching).
    ///
    /// # Errors
    ///
    /// Returns [`AdvancedRetrievalError::Search`] if the underlying search
    /// fails.
    pub async fn retrieve<E>(
        &self,
        query: &str,
        echo: &E,
    ) -> Result<Vec<SearchResult>, AdvancedRetrievalError>
    where
        E: Echo + ?Sized,
    {
        let hypothetical = self.generate_hypothetical_doc(query);

        echo.search(&hypothetical, self.config.top_k, None)
            .await
            .map_err(|e| AdvancedRetrievalError::Search(e.to_string()))
    }
}
