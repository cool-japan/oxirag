//! COIL encoding and scoring: contextualised token vectors, document-level
//! CLS vectors, and the [`CoilRetriever`] that ties encoding, the inverted
//! index, and COIL-tok / COIL-full scoring together.
//!
//! # Encoding
//!
//! A piece of text is tokenised into surface tokens. Each token is first given
//! a deterministic *base* embedding (an FNV-1a hash activates a handful of
//! dimensions, then the vector is L2-normalised). The token's **contextualised**
//! vector then blends that base embedding with the averaged base embeddings of
//! its neighbours inside a window, so the *same* surface token acquires a
//! *different* vector in different contexts (COIL's answer to polysemy). A
//! single document-level CLS vector is the L2-normalised mean of the base
//! embeddings.
//!
//! Everything is deterministic: identical text always encodes to identical
//! vectors, with no randomness and no machine-learning dependencies.
//!
//! # Scoring
//!
//! For a query token the contribution to a document is the **maximum** dot
//! product between the query token's contextualised vector and the
//! contextualised vector of any occurrence of the *same surface token* in that
//! document — exact-lexical gating. COIL-tok sums these contributions;
//! COIL-full adds `lambda * dot(query_cls, doc_cls)`.

use std::collections::HashMap;

use crate::coil_retrieval::index::CoilInvertedIndex;
use crate::coil_retrieval::types::{
    CoilConfig, CoilDocument, CoilError, CoilResult, CoilTokenVector,
};
use crate::types::{Document, DocumentId};

// ── Tokenisation ──────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens of length `>= 2`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Base embedding ────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of `bytes` seeded with `seed`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// Deterministic per-token *base* embedding.
///
/// The token's FNV-1a hash activates a small fixed number of dimensions with
/// signed magnitudes; the result is L2-normalised. Identical tokens map to
/// identical base vectors (cosine `= 1`) while unrelated tokens are
/// near-orthogonal.
fn base_embed(token: &str, dim: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dim];
    if dim == 0 {
        return vector;
    }
    let spread = 8usize.min(dim);
    let bytes = token.as_bytes();
    for k in 0..spread {
        let hash = fnv1a(bytes, (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        #[allow(clippy::cast_possible_truncation)]
        let index = (hash as usize) % dim;
        let sign = if (hash >> 33) & 1 == 0 { 1.0 } else { -1.0 };
        #[allow(clippy::cast_precision_loss)]
        let magnitude = 1.0 + ((hash >> 7) % 7) as f32;
        vector[index] += sign * magnitude;
    }
    l2_normalize(&mut vector);
    vector
}

/// L2-normalise `vector` in place (a no-op for a zero vector).
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// Dot product of two equal-length vectors.
///
/// Because both operands are L2-normalised on construction, this equals their
/// cosine similarity. The shorter length governs, so it never panics.
#[must_use]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Contextualisation ─────────────────────────────────────────────────────────

/// Blend each token's base embedding with the averaged base embeddings of its
/// neighbours within `window`, weighted by `weight`, then re-normalise.
///
/// A `window` or `weight` of zero (or a single-token sequence) yields the base
/// embeddings unchanged. Otherwise the same surface token in a different local
/// context produces a genuinely different vector.
fn contextualize(base: &[Vec<f32>], window: usize, weight: f32, dim: usize) -> Vec<Vec<f32>> {
    let count_tokens = base.len();
    let mut out = Vec::with_capacity(count_tokens);
    for (position, base_vector) in base.iter().enumerate() {
        let mut vector = base_vector.clone();
        if window > 0 && weight != 0.0 && count_tokens > 1 {
            let lo = position.saturating_sub(window);
            let hi = (position + window).min(count_tokens - 1);
            let mut accumulator = vec![0.0f32; dim];
            let mut neighbours = 0usize;
            for (neighbour_index, neighbour) in base.iter().enumerate().take(hi + 1).skip(lo) {
                if neighbour_index == position {
                    continue;
                }
                for d in 0..dim {
                    accumulator[d] += neighbour[d];
                }
                neighbours += 1;
            }
            if neighbours > 0 {
                #[allow(clippy::cast_precision_loss)]
                let scale = weight / neighbours as f32;
                for d in 0..dim {
                    vector[d] += accumulator[d] * scale;
                }
            }
        }
        l2_normalize(&mut vector);
        out.push(vector);
    }
    out
}

/// The document-level CLS vector: the L2-normalised mean of the base
/// embeddings. A zero vector is returned for an empty token sequence.
fn cls_vector(base: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut accumulator = vec![0.0f32; dim];
    if base.is_empty() {
        return accumulator;
    }
    for embedding in base {
        for d in 0..dim {
            accumulator[d] += embedding[d];
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / base.len() as f32;
    for value in &mut accumulator {
        *value *= scale;
    }
    l2_normalize(&mut accumulator);
    accumulator
}

/// Encode raw `text` into contextualised token vectors and a CLS vector.
fn encode_text(text: &str, config: &CoilConfig) -> (Vec<CoilTokenVector>, Vec<f32>) {
    let tokens = tokenize(text);
    let base: Vec<Vec<f32>> = tokens
        .iter()
        .map(|token| base_embed(token, config.dim))
        .collect();
    let contextualised = contextualize(
        &base,
        config.context_window,
        config.context_weight,
        config.dim,
    );
    let token_vectors = tokens
        .into_iter()
        .zip(contextualised)
        .enumerate()
        .map(|(position, (surface, vector))| CoilTokenVector::new(surface, position, vector))
        .collect();
    let cls = cls_vector(&base, config.dim);
    (token_vectors, cls)
}

// ── CoilRetriever ─────────────────────────────────────────────────────────────

/// A COIL retriever: encodes a corpus into a [`CoilInvertedIndex`] and answers
/// queries with COIL-tok or COIL-full scoring.
///
/// COIL (Gao, Dai, Callan 2021) matches a query token against a document token
/// *only when their surface strings are identical*, scoring that match with the
/// dot product of their contextualised vectors. This exact-lexical gating —
/// realised by the inverted list keyed on surface tokens — is the defining
/// difference from soft all-to-all late interaction (see
/// [`plaid_retrieval`](crate::plaid_retrieval)) and from term-expansion sparse
/// retrieval (see [`sparse_retrieval`](crate::sparse_retrieval)).
#[derive(Debug, Clone)]
pub struct CoilRetriever {
    /// Encoding and scoring configuration.
    config: CoilConfig,
    /// The contextualised inverted index built over the corpus.
    inverted: CoilInvertedIndex,
    /// Whether [`index`](Self::index) has populated the retriever.
    indexed: bool,
}

impl CoilRetriever {
    /// Create a new, empty retriever with the given configuration.
    #[must_use]
    pub fn new(config: CoilConfig) -> Self {
        Self {
            config,
            inverted: CoilInvertedIndex::new(),
            indexed: false,
        }
    }

    /// Borrow the retriever's configuration.
    #[must_use]
    pub fn config(&self) -> &CoilConfig {
        &self.config
    }

    /// Borrow the underlying contextualised inverted index.
    #[must_use]
    pub fn inverted_index(&self) -> &CoilInvertedIndex {
        &self.inverted
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inverted.num_documents()
    }

    /// Return `true` when the retriever holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inverted.is_empty()
    }

    /// Number of distinct surface tokens in the index.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.inverted.num_terms()
    }

    /// Return `true` once [`index`](Self::index) has populated the retriever.
    #[must_use]
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    /// Encode a single document into contextualised token vectors and a CLS
    /// vector.
    #[must_use]
    pub fn encode_document(&self, document: &Document) -> CoilDocument {
        let (token_vectors, cls_vector) = encode_text(&document.content, &self.config);
        CoilDocument::new(document.id.clone(), token_vectors, cls_vector)
    }

    /// Encode a query into contextualised token vectors and a CLS vector.
    ///
    /// # Errors
    ///
    /// Returns [`CoilError::EmptyQuery`] when the query yields no usable
    /// tokens.
    pub fn encode_query(&self, query: &str) -> CoilResult<(Vec<CoilTokenVector>, Vec<f32>)> {
        let (token_vectors, cls_vector) = encode_text(query, &self.config);
        if token_vectors.is_empty() {
            return Err(CoilError::EmptyQuery);
        }
        Ok((token_vectors, cls_vector))
    }

    /// Encode `documents` and (re)build the contextualised inverted index.
    ///
    /// Any previously indexed content is discarded first, so repeated calls are
    /// idempotent for a fixed corpus.
    ///
    /// # Errors
    ///
    /// Returns [`CoilError::EmptyCorpus`] when `documents` is empty.
    pub fn index(&mut self, documents: &[Document]) -> CoilResult<()> {
        if documents.is_empty() {
            return Err(CoilError::EmptyCorpus);
        }
        self.inverted.clear();
        for document in documents {
            let encoded = self.encode_document(document);
            self.inverted.insert_document(&encoded);
        }
        self.indexed = true;
        Ok(())
    }

    /// Search for the `k` best-scoring documents for `query`.
    ///
    /// Each query token contributes, to every document that contains the *same*
    /// surface token, the maximum dot product over that document's matching
    /// postings (COIL's exact-lexical gating). Contributions are summed across
    /// query tokens. In [`CoilScoreMode::Full`](crate::coil_retrieval::CoilScoreMode::Full)
    /// a `lambda`-weighted document-level CLS term is then added to every
    /// document — so semantically related documents can rank even without a
    /// lexical overlap — whereas in
    /// [`CoilScoreMode::Tok`](crate::coil_retrieval::CoilScoreMode::Tok) only
    /// documents sharing at least one surface token are scored at all.
    ///
    /// Results are sorted by descending score, ties broken by ascending
    /// document id for determinism, then truncated to `k`.
    ///
    /// # Errors
    ///
    /// - [`CoilError::NotIndexed`] when [`index`](Self::index) has not run.
    /// - [`CoilError::EmptyCorpus`] when the index holds no documents.
    /// - [`CoilError::EmptyQuery`] when `query` yields no usable tokens.
    pub fn search(&self, query: &str, k: usize) -> CoilResult<Vec<(DocumentId, f32)>> {
        if !self.indexed {
            return Err(CoilError::NotIndexed);
        }
        if self.inverted.is_empty() {
            return Err(CoilError::EmptyCorpus);
        }
        let (query_tokens, query_cls) = self.encode_query(query)?;

        let mut scores: HashMap<DocumentId, f32> = HashMap::new();

        // COIL-tok: exact-surface-token contextualised max-similarity.
        for query_token in &query_tokens {
            let postings = self.inverted.postings_for(&query_token.surface);
            if postings.is_empty() {
                continue;
            }
            let mut per_doc_max: HashMap<&DocumentId, f32> = HashMap::new();
            for posting in postings {
                let similarity = query_token.dot(&posting.token_vector);
                per_doc_max
                    .entry(&posting.doc_id)
                    .and_modify(|best| {
                        if similarity > *best {
                            *best = similarity;
                        }
                    })
                    .or_insert(similarity);
            }
            for (doc_id, best) in per_doc_max {
                *scores.entry(doc_id.clone()).or_insert(0.0) += best;
            }
        }

        // COIL-full: add the lambda-weighted document-level CLS term to every
        // document (the CLS "token" matches every document).
        if self.config.score_mode.includes_cls() {
            for doc_id in self.inverted.doc_ids() {
                if let Some(cls) = self.inverted.cls_for(doc_id) {
                    let term = self.config.lambda * dot(&query_cls, cls);
                    *scores.entry(doc_id.clone()).or_insert(0.0) += term;
                }
            }
        }

        let mut hits: Vec<(DocumentId, f32)> = scores.into_iter().collect();
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_str().cmp(b.0.as_str()))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Search using the configured [`CoilConfig::top_k`] result count.
    ///
    /// # Errors
    ///
    /// See [`search`](Self::search).
    pub fn search_default(&self, query: &str) -> CoilResult<Vec<(DocumentId, f32)>> {
        self.search(query, self.config.top_k)
    }
}
