//! In-memory SPLADE-style sparse retrieval index.

use crate::sparse_retrieval::encoder::SparseEncoder;
use crate::sparse_retrieval::types::{SparseConfig, SparseHit, SparseRetrievalError, SparseVector};
use crate::types::{Document, DocumentId};

// ── SparseIndex ───────────────────────────────────────────────────────────────

/// An in-memory index of learned sparse document vectors.
///
/// Building the index fits the underlying [`SparseEncoder`] on the corpus and
/// encodes every document. Queries are encoded the same way and ranked by sparse
/// dot product against the stored document vectors.
#[derive(Debug, Clone)]
pub struct SparseIndex {
    /// The learned sparse encoder.
    encoder: SparseEncoder,
    /// Stored document identifiers paired with their sparse encodings.
    entries: Vec<(DocumentId, SparseVector)>,
}

impl SparseIndex {
    /// Create a new, empty index with the given configuration.
    #[must_use]
    pub fn new(config: SparseConfig) -> Self {
        Self {
            encoder: SparseEncoder::new(config),
            entries: Vec::new(),
        }
    }

    /// Fit the encoder on `corpus` and encode every document into the index.
    ///
    /// # Errors
    ///
    /// Returns [`SparseRetrievalError::EmptyCorpus`] when `corpus` is empty.
    pub fn build(&mut self, corpus: &[Document]) -> Result<(), SparseRetrievalError> {
        if corpus.is_empty() {
            return Err(SparseRetrievalError::EmptyCorpus);
        }
        self.encoder.fit(corpus);
        self.entries = corpus
            .iter()
            .map(|doc| (doc.id.clone(), self.encoder.encode(&doc.content)))
            .collect();
        Ok(())
    }

    /// Number of documents currently held by the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Access the underlying fitted encoder.
    #[must_use]
    pub fn encoder(&self) -> &SparseEncoder {
        &self.encoder
    }

    /// Search the index for the `top_k` documents most relevant to `query`.
    ///
    /// The query is encoded with the fitted encoder and scored against every
    /// stored document by sparse dot product. Hits are sorted by descending
    /// score, ties broken by ascending document id for determinism, then
    /// truncated to `top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseRetrievalError::EmptyQuery`] when `query` is blank,
    /// and [`SparseRetrievalError::NotBuilt`] when the index has not been built.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SparseHit>, SparseRetrievalError> {
        if query.trim().is_empty() {
            return Err(SparseRetrievalError::EmptyQuery);
        }
        if self.entries.is_empty() || !self.encoder.is_fitted() {
            return Err(SparseRetrievalError::NotBuilt);
        }
        let q_vec = self.encoder.encode(query);
        let mut hits: Vec<SparseHit> = self
            .entries
            .iter()
            .map(|(id, vec)| SparseHit {
                id: id.clone(),
                score: q_vec.dot(vec),
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
