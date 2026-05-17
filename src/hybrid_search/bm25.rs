//! BM25 sparse retrieval: encoder and in-memory store.
//!
//! This module provides:
//! - `BM25Encoder`: converts text documents and queries into BM25-weighted sparse vectors
//! - `InMemorySparseStore`: an in-memory implementation of `SparseVectorStore` backed by
//!   an inverted index for efficient candidate retrieval

#![allow(clippy::cast_precision_loss)] // Intentional: usize to f32 for scoring

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;

use crate::error::VectorStoreError;
use crate::types::DocumentId;

use super::types::{BM25Params, SparseVector, SparseVectorStore};

/// BM25 encoder for converting text to sparse vectors.
///
/// Implements the BM25 weighting scheme with IDF (Inverse Document Frequency)
/// for effective lexical matching.
pub struct BM25Encoder {
    /// Vocabulary mapping from terms to indices.
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each term.
    idf: HashMap<String, f32>,
    /// Document frequency for each term.
    document_frequencies: HashMap<String, usize>,
    /// Total number of documents indexed.
    total_documents: usize,
    /// Average document length (in terms).
    average_doc_length: f32,
    /// Sum of all document lengths for computing average.
    total_terms: usize,
    /// BM25 parameters.
    params: BM25Params,
    /// Vocabulary size (dimension of sparse vectors).
    vocab_size: usize,
}

impl BM25Encoder {
    /// Create a new BM25 encoder with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_doc_length: 0.0,
            total_terms: 0,
            params: BM25Params::default(),
            vocab_size: 0,
        }
    }

    /// Create a BM25 encoder with custom parameters.
    #[must_use]
    pub fn with_params(params: BM25Params) -> Self {
        Self {
            vocabulary: HashMap::new(),
            idf: HashMap::new(),
            document_frequencies: HashMap::new(),
            total_documents: 0,
            average_doc_length: 0.0,
            total_terms: 0,
            params,
            vocab_size: 0,
        }
    }

    /// Get the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the total number of documents indexed.
    #[must_use]
    pub fn total_documents(&self) -> usize {
        self.total_documents
    }

    /// Get the BM25 parameters.
    #[must_use]
    pub fn params(&self) -> &BM25Params {
        &self.params
    }

    /// Tokenize text into terms.
    pub(crate) fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .map(String::from)
            .collect()
    }

    /// Fit the encoder on a corpus of documents.
    ///
    /// This builds the vocabulary and computes IDF values.
    pub fn fit(&mut self, documents: &[&str]) {
        self.vocabulary.clear();
        self.document_frequencies.clear();
        self.idf.clear();
        self.total_documents = documents.len();
        self.total_terms = 0;

        // First pass: build vocabulary and count document frequencies
        for doc in documents {
            let terms = Self::tokenize(doc);
            self.total_terms += terms.len();

            let unique_terms: HashSet<_> = terms.into_iter().collect();
            for term in unique_terms {
                // Add to vocabulary if new
                if !self.vocabulary.contains_key(&term) {
                    let idx = self.vocabulary.len();
                    self.vocabulary.insert(term.clone(), idx);
                }

                // Increment document frequency
                *self.document_frequencies.entry(term).or_insert(0) += 1;
            }
        }

        self.vocab_size = self.vocabulary.len();
        self.average_doc_length = if self.total_documents > 0 {
            self.total_terms as f32 / self.total_documents as f32
        } else {
            0.0
        };

        // Compute IDF for all terms using BM25 IDF variant with +1 smoothing
        let n = self.total_documents as f32;
        for (term, df) in &self.document_frequencies {
            let df_f = *df as f32;
            // BM25 IDF formula: log((N + 1) / (df + delta))
            // This variant ensures IDF is always positive
            let idf = ((n + 1.0) / (df_f + self.params.delta)).ln();
            self.idf.insert(term.clone(), idf.max(f32::EPSILON));
        }
    }

    /// Fit the encoder on owned strings.
    pub fn fit_owned(&mut self, documents: &[String]) {
        let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
        self.fit(&refs);
    }

    /// Encode a single text into a sparse vector.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// A sparse vector with BM25-weighted term frequencies.
    #[must_use]
    pub fn encode(&self, text: &str) -> SparseVector {
        if self.vocab_size == 0 {
            return SparseVector::empty(0);
        }

        let terms = Self::tokenize(text);
        let doc_length = terms.len() as f32;

        // Count term frequencies
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for term in terms {
            *term_freqs.entry(term).or_insert(0) += 1;
        }

        // Compute BM25 scores
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (term, tf) in term_freqs {
            if let Some(&idx) = self.vocabulary.get(&term) {
                let idf = self.idf.get(&term).copied().unwrap_or(0.0);
                let tf_f = tf as f32;

                // BM25 term frequency component
                let length_norm = 1.0 - self.params.b
                    + self.params.b * doc_length / self.average_doc_length.max(1.0);
                let tf_score =
                    (tf_f * (self.params.k1 + 1.0)) / (tf_f + self.params.k1 * length_norm);

                let score = idf * tf_score;
                if score > 0.0 {
                    indices.push(idx);
                    values.push(score);
                }
            }
        }

        // Sort by indices for efficient sparse operations
        let mut pairs: Vec<_> = indices.into_iter().zip(values).collect();
        pairs.sort_by_key(|(idx, _)| *idx);

        let (indices, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

        SparseVector::new(indices, values, self.vocab_size)
    }

    /// Encode multiple texts into sparse vectors.
    ///
    /// # Arguments
    /// * `texts` - The texts to encode
    ///
    /// # Returns
    /// A vector of sparse vectors.
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<SparseVector> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Encode multiple owned strings into sparse vectors.
    #[must_use]
    pub fn encode_batch_owned(&self, texts: &[String]) -> Vec<SparseVector> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
}

impl Default for BM25Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory sparse vector store for development and testing.
pub struct InMemorySparseStore {
    /// Stored sparse vectors by document ID.
    vectors: RwLock<HashMap<DocumentId, SparseVector>>,
    /// Inverted index: term index -> set of document IDs.
    inverted_index: RwLock<HashMap<usize, HashSet<DocumentId>>>,
}

impl InMemorySparseStore {
    /// Create a new in-memory sparse store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            inverted_index: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySparseStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SparseVectorStore for InMemorySparseStore {
    async fn insert(
        &mut self,
        id: DocumentId,
        vector: SparseVector,
    ) -> Result<(), VectorStoreError> {
        // Update inverted index
        {
            let mut index = self.inverted_index.write().await;
            for &idx in &vector.indices {
                index.entry(idx).or_default().insert(id.clone());
            }
        }

        // Store the vector
        {
            let mut vectors = self.vectors.write().await;
            vectors.insert(id, vector);
        }

        Ok(())
    }

    async fn search(
        &self,
        query: &SparseVector,
        top_k: usize,
    ) -> Result<Vec<(DocumentId, f32)>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        let inverted_index = self.inverted_index.read().await;

        // Find candidate documents using inverted index
        let mut candidates: HashSet<DocumentId> = HashSet::new();
        for &idx in &query.indices {
            if let Some(doc_ids) = inverted_index.get(&idx) {
                candidates.extend(doc_ids.iter().cloned());
            }
        }

        // Score candidates
        let mut scored: Vec<(DocumentId, f32)> = candidates
            .into_iter()
            .filter_map(|id| {
                vectors.get(&id).map(|vec| {
                    let score = query.dot(vec);
                    (id, score)
                })
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.truncate(top_k);
        Ok(scored)
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<SparseVector>, VectorStoreError> {
        let vectors = self.vectors.read().await;
        Ok(vectors.get(id).cloned())
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        // Remove from vectors and get the vector for index cleanup
        let vector = {
            let mut vectors = self.vectors.write().await;
            vectors.remove(id)
        };

        if let Some(vec) = vector {
            // Clean up inverted index
            let mut index = self.inverted_index.write().await;
            for &idx in &vec.indices {
                if let Some(doc_ids) = index.get_mut(&idx) {
                    doc_ids.remove(id);
                    if doc_ids.is_empty() {
                        index.remove(&idx);
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn count(&self) -> usize {
        self.vectors.read().await.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.vectors.write().await.clear();
        self.inverted_index.write().await.clear();
        Ok(())
    }
}
