//! Types for the `hippo_rag` module.

use thiserror::Error;

use crate::types::Document;

// ── HippoConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`HippoRagIndex`] construction and search.
///
/// Governs the Personalized `PageRank` (PPR) random walk used to spread query
/// relevance across the entity graph, plus the dimensionality reserved for any
/// auxiliary lexical hashing.
///
/// [`HippoRagIndex`]: crate::hippo_rag::HippoRagIndex
#[derive(Debug, Clone, PartialEq)]
pub struct HippoConfig {
    /// Random-walk restart probability complement (the *damping* factor `d`).
    ///
    /// At each PPR iteration, a walker continues along graph edges with
    /// probability `damping` and teleports back to the seed distribution with
    /// probability `1 - damping`. Defaults to `0.85`.
    pub damping: f32,
    /// Number of power-iteration steps performed when computing PPR.
    ///
    /// More iterations converge the rank vector more tightly at the cost of
    /// extra work. Defaults to `50`.
    pub ppr_iterations: usize,
    /// Dimensionality reserved for deterministic lexical hashing.
    ///
    /// Retained for parity with the wider `OxiRAG` retrieval modules and for
    /// future hybrid scoring; the graph walk itself is dimension-agnostic.
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for HippoConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            ppr_iterations: 50,
            dim: 128,
        }
    }
}

impl HippoConfig {
    /// Create a new [`HippoConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the random-walk damping factor (continuation probability).
    #[must_use]
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Set the number of PPR power-iteration steps.
    #[must_use]
    pub fn with_ppr_iterations(mut self, ppr_iterations: usize) -> Self {
        self.ppr_iterations = ppr_iterations;
        self
    }

    /// Set the dimensionality reserved for lexical hashing.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

// ── HippoHit ──────────────────────────────────────────────────────────────────

/// A passage scored by accumulated Personalized `PageRank` mass.
///
/// Returned by [`HippoRagIndex::search`]; the `score` is the sum of the PPR
/// mass of every entity the passage contains, so passages reachable from the
/// query entities through *multi-hop* co-occurrence links surface even when
/// they share no entity with the query directly.
///
/// [`HippoRagIndex::search`]: crate::hippo_rag::HippoRagIndex::search
#[derive(Debug, Clone)]
pub struct HippoHit {
    /// The retrieved passage.
    pub document: Document,
    /// Accumulated PPR mass of the passage's entities (higher is more relevant).
    pub score: f32,
}

// ── HippoRagError ─────────────────────────────────────────────────────────────

/// Errors from the `hippo_rag` module.
#[derive(Debug, Error)]
pub enum HippoRagError {
    /// [`HippoRagIndex::build`] was called with an empty corpus.
    ///
    /// [`HippoRagIndex::build`]: crate::hippo_rag::HippoRagIndex::build
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
    /// No entities could be extracted from the query text.
    #[error("no entities found in query")]
    NoEntities,
    /// A search was attempted before the index was built.
    #[error("index not built")]
    NotBuilt,
}
