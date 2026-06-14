//! Coarse-to-fine retriever built on nested Matryoshka embeddings.
//!
//! Two-stage search: stage 1 shortlists candidates fast using the cheap
//! [`MatryoshkaConfig::shortlist_dim`]-truncated prefix, then stage 2 reranks
//! that shortlist with the full-dimension cosine similarity.

use super::encoder::{MatryoshkaEmbedding, MatryoshkaEncoder};
use super::types::{MatryoshkaConfig, MatryoshkaError, MatryoshkaHit};
use crate::types::{Document, DocumentId};

// ── MatryoshkaRetriever ───────────────────────────────────────────────────────

/// Coarse-to-fine document retriever using nested embeddings.
pub struct MatryoshkaRetriever {
    /// Retriever configuration (validated at construction).
    pub config: MatryoshkaConfig,
    /// Encoder used for documents and queries.
    pub encoder: MatryoshkaEncoder,
    /// Indexed `(id, embedding)` pairs.
    pub entries: Vec<(DocumentId, MatryoshkaEmbedding)>,
}

impl MatryoshkaRetriever {
    /// Create a new retriever, validating `config`.
    ///
    /// # Errors
    ///
    /// Returns [`MatryoshkaError::InvalidDim`] when the configuration fails
    /// [`MatryoshkaConfig::validate`].
    pub fn try_new(config: MatryoshkaConfig) -> Result<Self, MatryoshkaError> {
        config.validate()?;
        let encoder = MatryoshkaEncoder::new(config.clone());
        Ok(Self {
            config,
            encoder,
            entries: Vec::new(),
        })
    }

    /// Encode and index a [`Document`] by its content.
    pub fn add_document(&mut self, doc: &Document) {
        let embedding = self.encoder.encode(&doc.content);
        self.entries.push((doc.id.clone(), embedding));
    }

    /// Encode and index raw `text` under the supplied `id`.
    pub fn add_text(&mut self, id: DocumentId, text: &str) {
        let embedding = self.encoder.encode(text);
        self.entries.push((id, embedding));
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Two-stage coarse-to-fine search.
    ///
    /// Stage 1 scores every document by cosine of the `shortlist_dim`-truncated
    /// prefix and keeps the top `top_k * shortlist_multiplier`. Stage 2 reranks
    /// that shortlist by full-dimension cosine and returns the top `top_k`. Each
    /// returned [`MatryoshkaHit`] carries both its stage-1 `shortlist_score` and
    /// its final full-dimension `score`.
    ///
    /// # Errors
    ///
    /// Returns [`MatryoshkaError::EmptyQuery`] when `query` is blank or
    /// [`MatryoshkaError::EmptyCorpus`] when no documents are indexed.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<MatryoshkaHit>, MatryoshkaError> {
        if query.trim().is_empty() {
            return Err(MatryoshkaError::EmptyQuery);
        }
        if self.entries.is_empty() {
            return Err(MatryoshkaError::EmptyCorpus);
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let query_emb = self.encoder.encode(query);
        let query_short = query_emb.truncate(self.config.shortlist_dim);

        // Stage 1: cheap shortlist over the truncated prefix.
        let mut stage1: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, emb))| {
                let doc_short = emb.truncate(self.config.shortlist_dim);
                (i, MatryoshkaEncoder::cosine(&query_short, &doc_short))
            })
            .collect();
        stage1.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let shortlist_size = top_k
            .saturating_mul(self.config.shortlist_multiplier.max(1))
            .min(stage1.len());

        // Stage 2: rerank the shortlist with the full dimension.
        let mut stage2: Vec<MatryoshkaHit> = stage1[..shortlist_size]
            .iter()
            .map(|&(i, shortlist_score)| {
                let (id, emb) = &self.entries[i];
                MatryoshkaHit {
                    id: id.clone(),
                    score: MatryoshkaEncoder::cosine(&query_emb.full, &emb.full),
                    shortlist_score,
                }
            })
            .collect();
        stage2.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stage2.truncate(top_k);
        Ok(stage2)
    }

    /// Single-stage full-dimension search, for comparison with [`Self::search`].
    ///
    /// Scores every document by full-dimension cosine and returns the top
    /// `top_k`. The `shortlist_score` of each hit mirrors its full `score`.
    ///
    /// # Errors
    ///
    /// Returns [`MatryoshkaError::EmptyQuery`] when `query` is blank or
    /// [`MatryoshkaError::EmptyCorpus`] when no documents are indexed.
    pub fn search_single_stage(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<MatryoshkaHit>, MatryoshkaError> {
        if query.trim().is_empty() {
            return Err(MatryoshkaError::EmptyQuery);
        }
        if self.entries.is_empty() {
            return Err(MatryoshkaError::EmptyCorpus);
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let query_emb = self.encoder.encode(query);
        let mut hits: Vec<MatryoshkaHit> = self
            .entries
            .iter()
            .map(|(id, emb)| {
                let score = MatryoshkaEncoder::cosine(&query_emb.full, &emb.full);
                MatryoshkaHit {
                    id: id.clone(),
                    score,
                    shortlist_score: score,
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
