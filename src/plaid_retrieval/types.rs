//! Types for the `plaid_retrieval` module.

use thiserror::Error;

use crate::types::Document;

// ── PlaidConfig ───────────────────────────────────────────────────────────────

/// Configuration for the PLAID late-interaction retriever.
///
/// PLAID (Santhanam et al., 2022) accelerates `ColBERTv2` late interaction by
/// clustering every per-token embedding into a small set of centroids. Query
/// tokens probe their nearest centroids to cheaply gather a candidate set, which
/// is then re-scored with full `MaxSim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaidConfig {
    /// Number of centroids trained over all per-token embeddings.
    ///
    /// Defaults to `16`.
    pub num_centroids: usize,
    /// Number of nearest centroids each query token probes during candidate
    /// generation.
    ///
    /// Defaults to `4`.
    pub nprobe: usize,
    /// Dimensionality of every per-token embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Maximum number of k-means refinement iterations when training centroids.
    ///
    /// Defaults to `10`.
    pub kmeans_iters: usize,
}

impl Default for PlaidConfig {
    fn default() -> Self {
        Self {
            num_centroids: 16,
            nprobe: 4,
            dim: 128,
            kmeans_iters: 10,
        }
    }
}

impl PlaidConfig {
    /// Create a configuration with the default PLAID parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of centroids trained over all per-token embeddings.
    #[must_use]
    pub fn with_num_centroids(mut self, num_centroids: usize) -> Self {
        self.num_centroids = num_centroids;
        self
    }

    /// Set the number of nearest centroids each query token probes.
    #[must_use]
    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the per-token embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the maximum number of k-means refinement iterations.
    #[must_use]
    pub fn with_kmeans_iters(mut self, kmeans_iters: usize) -> Self {
        self.kmeans_iters = kmeans_iters;
        self
    }
}

// ── PlaidHit ──────────────────────────────────────────────────────────────────

/// A single scored result from a [`PlaidRetriever`](crate::plaid_retrieval::PlaidRetriever)
/// search.
#[derive(Debug, Clone)]
pub struct PlaidHit {
    /// The retrieved document.
    pub document: Document,
    /// The `MaxSim` score assigned to the document for the query.
    pub score: f32,
}

// ── PlaidError ────────────────────────────────────────────────────────────────

/// Errors produced by the `plaid_retrieval` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlaidError {
    /// The corpus passed to `build` contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The query passed to `search` was empty or whitespace-only.
    #[error("query must not be empty")]
    EmptyQuery,
    /// `search` was invoked before `build` populated the retriever.
    #[error("retriever not built")]
    NotBuilt,
}
