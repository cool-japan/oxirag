//! Deterministic feature extraction for `learning_to_rank`.
//!
//! [`LtrFeatureExtractor`] turns a `(query, document)` pair into a fixed-order
//! [`LtrFeatureVector`] of numeric retrieval-metadata signals. Everything here
//! is self-contained and deterministic — the module deliberately owns its own
//! small lexical toolkit (a `BM25` scorer, an `FNV-1a` lexical pseudo-embedding
//! and a cosine similarity) rather than importing helpers from sibling
//! modules, mirroring the convention used across `OxiRAG`.
//!
//! The default layout is
//! `[bm25, recency, embedding_similarity, popularity, length]`; each dimension
//! is mapped through a documented, monotonic transform into `[0.0, 1.0]` so
//! the heterogeneous signals are comparably scaled for gradient descent while
//! still preserving their ordering.

use std::collections::HashMap;

use super::types::{LTR_DEFAULT_FEATURE_DIM, LtrDocument, LtrFeatureVector};

// ── lexical helpers ──────────────────────────────────────────────────────────

/// Tokenise `text`: split on non-alphanumeric boundaries, lowercase, and keep
/// fragments of at least two characters.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// `FNV-1a` 64-bit hash (offset basis `14695981039346656037`, prime
/// `1099511628211`) of a token's lowercased bytes.
fn fnv1a(token: &str) -> u64 {
    let mut hash: u64 = 14_695_981_039_346_656_037;
    for byte in token.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Tokens are hashed to buckets with `FNV-1a`, per-bucket counts accumulated,
/// and the vector `L2`-normalised. Because bucket counts are non-negative, the
/// cosine of two such embeddings lies in `[0.0, 1.0]`.
fn embed(text: &str, dim: usize) -> Vec<f64> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f64; dim];
    for token in tokenize(text) {
        #[allow(clippy::cast_possible_truncation)]
        let idx = (fnv1a(&token) as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f64 = buckets.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for value in &mut buckets {
            *value /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two equal-length vectors, clamped to `[-1.0, 1.0]`.
/// Returns `0.0` for mismatched, empty, or zero-magnitude inputs.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ── LtrFeatureExtractor ──────────────────────────────────────────────────────

/// Extracts a [`LtrFeatureVector`] from a `(query, document)` pair.
///
/// Construct one with [`LtrFeatureExtractor::new`] for a corpus-free extractor
/// (uniform inverse-document-frequency), or with
/// [`LtrFeatureExtractor::from_corpus`] to precompute inverse document
/// frequencies and the average document length from a set of documents, which
/// makes the `BM25` signal corpus-aware.
#[derive(Debug, Clone)]
pub struct LtrFeatureExtractor {
    embedding_dim: usize,
    reference_time: i64,
    recency_scale_days: f64,
    popularity_scale: f64,
    length_scale: f64,
    bm25_k1: f64,
    bm25_b: f64,
    idf: HashMap<String, f64>,
    avg_doc_len: f64,
}

impl LtrFeatureExtractor {
    /// Default pseudo-embedding dimension.
    pub const DEFAULT_EMBEDDING_DIM: usize = 64;

    /// Construct a corpus-free extractor. Every term is assigned a uniform
    /// inverse document frequency of `1.0` and the `BM25` length norm uses
    /// each document's own length as the average.
    #[must_use]
    pub fn new(reference_time: i64) -> Self {
        Self {
            embedding_dim: Self::DEFAULT_EMBEDDING_DIM,
            reference_time,
            recency_scale_days: 30.0,
            popularity_scale: 10.0,
            length_scale: 256.0,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            idf: HashMap::new(),
            avg_doc_len: 0.0,
        }
    }

    /// Construct a corpus-aware extractor: inverse document frequencies and the
    /// average document length are precomputed from `corpus`.
    #[must_use]
    pub fn from_corpus(corpus: &[LtrDocument], reference_time: i64) -> Self {
        let mut extractor = Self::new(reference_time);
        let n_docs = corpus.len();
        if n_docs == 0 {
            return extractor;
        }
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_len: usize = 0;
        for doc in corpus {
            let tokens = tokenize(&doc.content);
            total_len += tokens.len();
            let unique: std::collections::HashSet<String> = tokens.into_iter().collect();
            for term in unique {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let n = n_docs as f64;
        extractor.idf = doc_freq
            .into_iter()
            .map(|(term, df)| {
                #[allow(clippy::cast_precision_loss)]
                let df_f = df as f64;
                // Standard BM25 idf with the +0.5 smoothing, floored at a small
                // positive value so every observed term contributes.
                let idf = (((n - df_f + 0.5) / (df_f + 0.5)) + 1.0).ln();
                (term, idf.max(1e-6))
            })
            .collect();
        #[allow(clippy::cast_precision_loss)]
        {
            extractor.avg_doc_len = total_len as f64 / n;
        }
        if extractor.avg_doc_len > 0.0 {
            extractor.length_scale = extractor.avg_doc_len;
        }
        extractor
    }

    /// Set the pseudo-embedding dimension.
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the recency half-scale, in days: a document exactly this old scores
    /// `0.5` on the recency signal.
    #[must_use]
    pub fn with_recency_scale_days(mut self, recency_scale_days: f64) -> Self {
        self.recency_scale_days = recency_scale_days.max(1e-6);
        self
    }

    /// Set the popularity saturation scale: a document with this many
    /// popularity points scores `0.5` on the popularity signal.
    #[must_use]
    pub fn with_popularity_scale(mut self, popularity_scale: f64) -> Self {
        self.popularity_scale = popularity_scale.max(1e-6);
        self
    }

    /// The inverse document frequency of `term`, defaulting to `1.0` for terms
    /// unseen in the corpus (or for a corpus-free extractor).
    fn term_idf(&self, term: &str) -> f64 {
        self.idf.get(term).copied().unwrap_or(1.0)
    }

    /// Compute the raw `BM25` score of `doc_tokens` against `query_tokens`.
    fn bm25(&self, query_tokens: &[String], doc_tokens: &[String]) -> f64 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }
        let mut term_freq: HashMap<&str, usize> = HashMap::new();
        for token in doc_tokens {
            *term_freq.entry(token.as_str()).or_insert(0) += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let doc_len = doc_tokens.len() as f64;
        let avg_len = if self.avg_doc_len > 0.0 {
            self.avg_doc_len
        } else {
            doc_len
        };
        let mut score = 0.0;
        for term in query_tokens {
            let Some(&freq) = term_freq.get(term.as_str()) else {
                continue;
            };
            #[allow(clippy::cast_precision_loss)]
            let tf = freq as f64;
            let denom = tf + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * doc_len / avg_len);
            if denom > 0.0 {
                score += self.term_idf(term) * (tf * (self.bm25_k1 + 1.0)) / denom;
            }
        }
        score
    }

    /// Extract the default 5-signal feature vector for a `(query, document)`
    /// pair.
    ///
    /// The returned vector always has [`LTR_DEFAULT_FEATURE_DIM`] dimensions
    /// with every value in `[0.0, 1.0]`.
    #[must_use]
    pub fn extract(&self, query: &str, document: &LtrDocument) -> LtrFeatureVector {
        let query_tokens = tokenize(query);
        let doc_tokens = tokenize(&document.content);

        // BM25 relevance, saturated into [0, 1) via x / (x + (k1 + 1)).
        let bm25_raw = self.bm25(&query_tokens, &doc_tokens);
        let saturation = self.bm25_k1 + 1.0;
        let bm25 = if bm25_raw > 0.0 {
            bm25_raw / (bm25_raw + saturation)
        } else {
            0.0
        };

        // Recency: newer -> closer to 1.0. Documents in the future (timestamp
        // beyond the reference) are treated as maximally recent.
        #[allow(clippy::cast_precision_loss)]
        let age_seconds = (self.reference_time - document.timestamp).max(0) as f64;
        let age_days = age_seconds / 86_400.0;
        let recency = 1.0 / (1.0 + age_days / self.recency_scale_days);

        // Semantic similarity via FNV-1a pseudo-embeddings (already in [0, 1]).
        let query_embedding = embed(query, self.embedding_dim);
        let doc_embedding = embed(&document.content, self.embedding_dim);
        let embedding_similarity = cosine(&query_embedding, &doc_embedding).clamp(0.0, 1.0);

        // Popularity, saturated into [0, 1).
        #[allow(clippy::cast_precision_loss)]
        let popularity_raw = document.popularity as f64;
        let popularity = popularity_raw / (popularity_raw + self.popularity_scale);

        // Document length, saturated into [0, 1).
        #[allow(clippy::cast_precision_loss)]
        let length_raw = doc_tokens.len() as f64;
        let length = length_raw / (length_raw + self.length_scale);

        debug_assert_eq!(LTR_DEFAULT_FEATURE_DIM, 5);
        LtrFeatureVector::from_signals(bm25, recency, embedding_similarity, popularity, length)
    }
}
