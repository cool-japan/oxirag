//! Types and lexical primitives for the `generative_retrieval` module.

use thiserror::Error;

use crate::types::Document;

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise (non-alphanumeric split, length `>= 2`, lowercased) →
/// hash each token with FNV-1a into a bucket index → accumulate per-bucket
/// counts → L2-normalise to the requested `dim`.
///
/// The result is fully deterministic: identical input always yields an
/// identical vector, with no reliance on randomness or floating-point ordering.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in tokenize(text) {
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

/// Tokenise `text`: split on non-alphanumeric boundaries, keep tokens of length
/// `>= 2`, and lowercase each retained token.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Cosine similarity between two equal-length, L2-normalised vectors.
///
/// Returns `0.0` when the lengths differ or either vector is empty.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── SemanticDocId ─────────────────────────────────────────────────────────────

/// A hierarchical semantic document identifier.
///
/// The id is the path of cluster indices walked from the root of the centroid
/// tree down to the leaf that holds the document. For example a path of
/// `[2, 5, 1]` renders as the string `"2-5-1"`: the document lives in the third
/// top-level cluster, the sixth sub-cluster of that, and the second leaf below.
///
/// Documents that descend into the same leaf share an identical id; the id is
/// *semantic*, grouping related documents rather than uniquely tagging each one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticDocId {
    /// Cluster indices from root to leaf (root excluded).
    pub path: Vec<usize>,
}

impl SemanticDocId {
    /// Create a new semantic id from a cluster-index path.
    #[must_use]
    pub fn new(path: Vec<usize>) -> Self {
        Self { path }
    }

    /// Render the id as a hyphen-joined string such as `"2-5-1"`.
    ///
    /// An empty path renders as the empty string.
    #[must_use]
    pub fn as_string(&self) -> String {
        self.path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Return the number of hierarchy levels in the id (the path length).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Return `true` when the id carries no cluster indices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

impl std::fmt::Display for SemanticDocId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

// ── GenRetrievalConfig ────────────────────────────────────────────────────────

/// Configuration for hierarchical generative retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenRetrievalConfig {
    /// Maximum number of clusters (children) created per internal tree node.
    ///
    /// Defaults to `4`.
    pub branching: usize,
    /// Maximum depth of the semantic id tree (number of cluster indices).
    ///
    /// Defaults to `3`.
    pub max_depth: usize,
    /// Beam width retained at each level during constrained traversal.
    ///
    /// Defaults to `2`.
    pub beam: usize,
    /// Dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for GenRetrievalConfig {
    fn default() -> Self {
        Self {
            branching: 4,
            max_depth: 3,
            beam: 2,
            dim: 128,
        }
    }
}

impl GenRetrievalConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of clusters created per internal node.
    #[must_use]
    pub fn with_branching(mut self, branching: usize) -> Self {
        self.branching = branching;
        self
    }

    /// Set the maximum depth of the semantic id tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the beam width retained during traversal.
    #[must_use]
    pub fn with_beam(mut self, beam: usize) -> Self {
        self.beam = beam;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

// ── GenHit ────────────────────────────────────────────────────────────────────

/// A scored document returned by constrained generative retrieval.
#[derive(Debug, Clone)]
pub struct GenHit {
    /// The matched document.
    pub document: Document,
    /// The semantic id of the matched document.
    pub doc_id: SemanticDocId,
    /// Cosine similarity of the query to the document (higher is more relevant).
    pub score: f32,
}

// ── GenRetrievalError ─────────────────────────────────────────────────────────

/// Errors from the `generative_retrieval` module.
#[derive(Debug, Error)]
pub enum GenRetrievalError {
    /// `build` was called with no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// `search` was called with a blank query string.
    #[error("query must not be empty")]
    EmptyQuery,
    /// `search` was called before `build`.
    #[error("index not built")]
    NotBuilt,
}
