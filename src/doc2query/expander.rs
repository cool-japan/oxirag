//! Document expansion driver that appends generated queries to content.
use crate::doc2query::generator::HeuristicQueryGenerator;
use crate::doc2query::types::{Doc2QueryConfig, Doc2QueryError, ExpandedDocument, QueryGenerator};
use crate::types::Document;

// ── Doc2QueryExpander ─────────────────────────────────────────────────────────

/// Expands documents with hypothetical queries before indexing.
///
/// Wraps a [`QueryGenerator`] and a [`Doc2QueryConfig`]. For each document it
/// generates up to [`num_queries`](Doc2QueryConfig::num_queries) hypothetical
/// questions and appends them to the content, which boosts both lexical and
/// semantic recall at retrieval time.
#[derive(Debug, Clone)]
pub struct Doc2QueryExpander<G: QueryGenerator> {
    /// Expansion configuration.
    pub config: Doc2QueryConfig,
    /// Query generator used to produce hypothetical questions.
    pub generator: G,
}

impl Doc2QueryExpander<HeuristicQueryGenerator> {
    /// Create an expander backed by the default [`HeuristicQueryGenerator`].
    #[must_use]
    pub fn new(config: Doc2QueryConfig) -> Self {
        let generator = HeuristicQueryGenerator::new(config.clone());
        Self { config, generator }
    }
}

impl<G: QueryGenerator> Doc2QueryExpander<G> {
    /// Create an expander with a custom [`QueryGenerator`].
    #[must_use]
    pub fn with_generator(config: Doc2QueryConfig, generator: G) -> Self {
        Self { config, generator }
    }

    /// Expand a single document into an [`ExpandedDocument`].
    ///
    /// # Errors
    ///
    /// Returns [`Doc2QueryError::EmptyDocument`] if the document content is
    /// empty or whitespace-only.
    pub fn expand(&self, doc: &Document) -> Result<ExpandedDocument, Doc2QueryError> {
        if doc.content.trim().is_empty() {
            return Err(Doc2QueryError::EmptyDocument);
        }
        let generated_queries = self.generator.generate(doc, self.config.num_queries);
        let separator = &self.config.append_separator;
        let mut expanded_content = doc.content.clone();
        for query in &generated_queries {
            expanded_content.push_str(separator);
            expanded_content.push_str(query);
        }
        Ok(ExpandedDocument {
            original: doc.clone(),
            generated_queries,
            expanded_content,
        })
    }

    /// Expand every document in `docs`, preserving order.
    ///
    /// # Errors
    ///
    /// Returns [`Doc2QueryError::EmptyDocument`] if any document is empty.
    pub fn expand_corpus(
        &self,
        docs: &[Document],
    ) -> Result<Vec<ExpandedDocument>, Doc2QueryError> {
        docs.iter().map(|doc| self.expand(doc)).collect()
    }

    /// Produce a new [`Document`] whose content is the expanded content.
    ///
    /// The returned document keeps the original `id`, `title`, `source`,
    /// `metadata` and timestamps; only `content` is replaced.
    ///
    /// # Errors
    ///
    /// Returns [`Doc2QueryError::EmptyDocument`] if the document is empty.
    pub fn to_indexable_document(&self, doc: &Document) -> Result<Document, Doc2QueryError> {
        let expanded = self.expand(doc)?;
        let mut indexable = doc.clone();
        indexable.content = expanded.expanded_content;
        Ok(indexable)
    }
}
