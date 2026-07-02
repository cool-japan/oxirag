//! [`MuveraIndex`]: a single-vector store over Fixed Dimensional Encodings.
//!
//! Each added document is reduced to one [`FixedDimEncoding`] and stored
//! alongside its original token vectors. A search encodes the query set once,
//! then ranks documents by a single dot product per document — a maximum
//! inner-product search (MIPS) — instead of the quadratic `MaxSim` scan that
//! plain late interaction performs at query time. When
//! [`MuveraConfig::rerank`] is enabled, the top FDE candidates are re-scored
//! with the exact Chamfer similarity for a final, faithful ordering.

use super::fde::{FixedDimEncoding, MuveraEncoder, chamfer_similarity};
use super::types::{MuveraConfig, MuveraDocument, MuveraError, MuveraResult, MuveraSimilarity};

// ── StoredDocument ───────────────────────────────────────────────────────────

/// A document held by the index: its user-facing record plus the precomputed
/// FDE used for approximate ranking.
#[derive(Debug, Clone)]
struct StoredDocument {
    document: MuveraDocument,
    encoding: FixedDimEncoding,
}

// ── MuveraIndex ──────────────────────────────────────────────────────────────

/// An in-memory MUVERA index.
///
/// [`add`](Self::add) encodes each document's multi-vector set into a single
/// [`FixedDimEncoding`] and stores it. [`search`](Self::search) encodes the
/// query set, ranks every stored document by the dot product of the two
/// encodings (the single-vector proxy for Chamfer), and — when
/// [`MuveraConfig::rerank`] is set — re-ranks the leading candidates by the
/// exact Chamfer similarity recomputed from the retained token vectors.
#[derive(Debug, Clone)]
pub struct MuveraIndex {
    encoder: MuveraEncoder,
    documents: Vec<StoredDocument>,
}

impl MuveraIndex {
    /// Create an empty index from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::InvalidConfig`] when
    /// [`MuveraConfig::validate`] rejects the configuration.
    pub fn new(config: MuveraConfig) -> MuveraResult<Self> {
        let encoder = MuveraEncoder::new(config)?;
        Ok(Self {
            encoder,
            documents: Vec::new(),
        })
    }

    /// Build an index from `config` and an iterator of `(id, token_vectors)`
    /// documents.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::InvalidConfig`] for a bad configuration,
    /// [`MuveraError::DuplicateId`] when two documents share an id, and any
    /// error from [`add`](Self::add) (empty set or dimension mismatch).
    pub fn build(
        config: MuveraConfig,
        documents: impl IntoIterator<Item = (String, Vec<Vec<f32>>)>,
    ) -> MuveraResult<Self> {
        let mut index = Self::new(config)?;
        for (id, token_vectors) in documents {
            index.add(id, token_vectors)?;
        }
        Ok(index)
    }

    /// Borrow the index configuration.
    #[must_use]
    pub fn config(&self) -> &MuveraConfig {
        self.encoder.config()
    }

    /// Borrow the underlying encoder.
    #[must_use]
    pub fn encoder(&self) -> &MuveraEncoder {
        &self.encoder
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Return `true` when the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Return `true` when a document with `id` is already indexed.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.documents.iter().any(|s| s.document.id == id)
    }

    /// Borrow the stored document with `id`, or `None` when absent.
    #[must_use]
    pub fn document(&self, id: &str) -> Option<&MuveraDocument> {
        self.documents
            .iter()
            .find(|s| s.document.id == id)
            .map(|s| &s.document)
    }

    /// Borrow the [`FixedDimEncoding`] stored for the document with `id`, or
    /// `None` when absent.
    #[must_use]
    pub fn encoding(&self, id: &str) -> Option<&FixedDimEncoding> {
        self.documents
            .iter()
            .find(|s| s.document.id == id)
            .map(|s| &s.encoding)
    }

    /// Encode `token_vectors` and add the document under `id`.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::DuplicateId`] when `id` is already present,
    /// [`MuveraError::EmptyMultiVector`] when `token_vectors` is empty, and
    /// [`MuveraError::DimensionMismatch`] when any token's length differs from
    /// [`MuveraConfig::dim`].
    pub fn add(&mut self, id: impl Into<String>, token_vectors: Vec<Vec<f32>>) -> MuveraResult<()> {
        let id = id.into();
        if self.contains(&id) {
            return Err(MuveraError::DuplicateId(id));
        }
        let encoding = self.encoder.encode_document(&token_vectors)?;
        self.documents.push(StoredDocument {
            document: MuveraDocument::new(id, token_vectors),
            encoding,
        });
        Ok(())
    }

    /// Search for the `k` documents most similar to the query multi-vector set.
    ///
    /// The query is encoded once; documents are ranked by the dot product of
    /// their FDEs (descending, ties broken by ascending id). When
    /// [`MuveraConfig::rerank`] is enabled, the leading
    /// [`MuveraConfig::rerank_depth`] candidates (the whole corpus when that is
    /// `0`, never fewer than `k`) are re-scored with the exact Chamfer
    /// similarity and re-sorted before truncation.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::EmptyIndex`] when the index is empty,
    /// [`MuveraError::EmptyMultiVector`] when `query` is empty, and
    /// [`MuveraError::DimensionMismatch`] when any query token's length differs
    /// from [`MuveraConfig::dim`].
    pub fn search(&self, query: &[Vec<f32>], k: usize) -> MuveraResult<Vec<MuveraSimilarity>> {
        if self.documents.is_empty() {
            return Err(MuveraError::EmptyIndex);
        }
        let query_encoding = self.encoder.encode_query(query)?;

        // Approximate ranking by FDE dot product.
        let mut ranked: Vec<(usize, f32)> = self
            .documents
            .iter()
            .enumerate()
            .map(|(idx, stored)| (idx, query_encoding.dot(&stored.encoding)))
            .collect();
        Self::sort_by_score(&mut ranked, &self.documents);

        if self.config().rerank {
            let depth = if self.config().rerank_depth == 0 {
                ranked.len()
            } else {
                self.config().rerank_depth.max(k)
            }
            .min(ranked.len());

            let mut reranked: Vec<(usize, f32)> = ranked[..depth]
                .iter()
                .map(|&(idx, _)| {
                    let score =
                        chamfer_similarity(query, &self.documents[idx].document.token_vectors);
                    (idx, score)
                })
                .collect();
            Self::sort_by_score(&mut reranked, &self.documents);
            return Ok(self.take_hits(&reranked, k));
        }

        Ok(self.take_hits(&ranked, k))
    }

    /// Search using [`MuveraConfig::top_k`] as the result count.
    ///
    /// # Errors
    ///
    /// Propagates every error of [`search`](Self::search).
    pub fn search_default(&self, query: &[Vec<f32>]) -> MuveraResult<Vec<MuveraSimilarity>> {
        self.search(query, self.config().top_k)
    }

    /// The exact Chamfer / `MaxSim` similarity between `query` and the stored
    /// document with `id`.
    ///
    /// # Errors
    ///
    /// Returns [`MuveraError::EmptyIndex`] when no document has `id` (reusing
    /// the empty-store error to signal "not found"), and any error from
    /// [`MuveraEncoder::chamfer`] (empty set or dimension mismatch).
    pub fn exact_similarity(&self, query: &[Vec<f32>], id: &str) -> MuveraResult<f32> {
        let stored = self
            .documents
            .iter()
            .find(|s| s.document.id == id)
            .ok_or(MuveraError::EmptyIndex)?;
        self.encoder.chamfer(query, &stored.document.token_vectors)
    }

    /// Sort `(index, score)` pairs by descending score, ties broken by
    /// ascending document id (stable, deterministic ordering).
    fn sort_by_score(ranked: &mut [(usize, f32)], documents: &[StoredDocument]) {
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| documents[a.0].document.id.cmp(&documents[b.0].document.id))
        });
    }

    /// Materialise the first `k` ranked pairs into [`MuveraSimilarity`] hits.
    fn take_hits(&self, ranked: &[(usize, f32)], k: usize) -> Vec<MuveraSimilarity> {
        ranked
            .iter()
            .take(k)
            .map(|&(idx, score)| {
                MuveraSimilarity::new(self.documents[idx].document.id.clone(), score)
            })
            .collect()
    }
}
