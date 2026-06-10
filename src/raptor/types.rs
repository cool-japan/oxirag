//! Types for the `raptor` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── ClusterStrategy ───────────────────────────────────────────────────────────

/// Strategy used to cluster document nodes within RAPTOR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClusterStrategy {
    /// Bottom-up agglomerative clustering (single-linkage by cosine similarity).
    #[default]
    Agglomerative,
    /// Lightweight k-means-lite clustering (Hamming-bucketed centroids).
    KMeansLite,
}

impl ClusterStrategy {
    /// Human-readable label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agglomerative => "agglomerative",
            Self::KMeansLite => "kmeans_lite",
        }
    }
}

// ── RaptorNode ────────────────────────────────────────────────────────────────

/// A single node in the RAPTOR tree (leaf = original chunk; internal = summary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorNode {
    /// Node identifier (sequential).
    pub id: usize,
    /// Text content (original chunk or extractive summary).
    pub text: String,
    /// Level in the tree (0 = leaf).
    pub level: usize,
    /// Indices of child nodes (empty for leaves).
    pub children: Vec<usize>,
    /// Lexical pseudo-embedding: hash-bucketed L2-normalised vector.
    pub embedding: Vec<f32>,
}

impl RaptorNode {
    /// Return `true` when this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ── RaptorTree ────────────────────────────────────────────────────────────────

/// The full RAPTOR summarization tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaptorTree {
    /// All nodes indexed by their `id` field.
    pub nodes: Vec<RaptorNode>,
    /// Sorted unique levels present in the tree.
    pub levels: Vec<usize>,
    /// Ids of the root nodes (highest level).
    pub root_ids: Vec<usize>,
}

impl RaptorTree {
    /// Return all nodes at a given level.
    #[must_use]
    pub fn nodes_at_level(&self, level: usize) -> Vec<&RaptorNode> {
        self.nodes.iter().filter(|n| n.level == level).collect()
    }

    /// Return total node count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` if the tree has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Collapsed retrieval: score all nodes by cosine similarity to `query_embedding`,
    /// return the top-k nodes.
    #[must_use]
    pub fn collapsed_retrieval(&self, query_embedding: &[f32], top_k: usize) -> Vec<&RaptorNode> {
        let mut scored: Vec<(f32, &RaptorNode)> = self
            .nodes
            .iter()
            .map(|n| (cosine(query_embedding, &n.embedding), n))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(_, n)| n).collect()
    }
}

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ── RaptorConfig ──────────────────────────────────────────────────────────────

/// Configuration for RAPTOR tree construction.
#[derive(Debug, Clone)]
pub struct RaptorConfig {
    /// Target cluster size (nodes per internal summary).
    ///
    /// Defaults to `4`.
    pub cluster_size: usize,
    /// Maximum number of tree levels (including leaf level).
    ///
    /// Defaults to `4`.
    pub max_levels: usize,
    /// Embedding dimension.
    ///
    /// Defaults to `256`.
    pub dim: usize,
    /// Number of sentences to include in each extractive summary node.
    ///
    /// Defaults to `3`.
    pub summary_sentences: usize,
    /// Clustering strategy.
    ///
    /// Defaults to [`ClusterStrategy::Agglomerative`].
    pub cluster_strategy: ClusterStrategy,
}

impl Default for RaptorConfig {
    fn default() -> Self {
        Self {
            cluster_size: 4,
            max_levels: 4,
            dim: 256,
            summary_sentences: 3,
            cluster_strategy: ClusterStrategy::Agglomerative,
        }
    }
}

impl RaptorConfig {
    /// Set the cluster size.
    #[must_use]
    pub fn with_cluster_size(mut self, v: usize) -> Self {
        self.cluster_size = v;
        self
    }

    /// Set the maximum number of levels.
    #[must_use]
    pub fn with_max_levels(mut self, v: usize) -> Self {
        self.max_levels = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }

    /// Set the number of summary sentences.
    #[must_use]
    pub fn with_summary_sentences(mut self, v: usize) -> Self {
        self.summary_sentences = v;
        self
    }

    /// Set the clustering strategy.
    #[must_use]
    pub fn with_cluster_strategy(mut self, v: ClusterStrategy) -> Self {
        self.cluster_strategy = v;
        self
    }
}

// ── RaptorError ───────────────────────────────────────────────────────────────

/// Errors from the `raptor` module.
#[derive(Debug, Error)]
pub enum RaptorError {
    /// No input texts were provided.
    #[error("Input must contain at least one text")]
    EmptyInput,
}
