//! Types for the `matryoshka` module.
//!
//! Implements Matryoshka Representation Learning (Kusupati et al. 2022): nested
//! "Russian doll" embeddings whose front prefixes are themselves usable
//! lower-dimensional embeddings. This module is about *dimension truncation*,
//! distinct from `quantization` which is about numeric *precision*.

use thiserror::Error;

// ── MatryoshkaError ───────────────────────────────────────────────────────────

/// Errors from the `matryoshka` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MatryoshkaError {
    /// The retriever corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A requested or configured dimension was invalid (e.g. zero or oversized).
    #[error("invalid dimension: {0}")]
    InvalidDim(usize),
}

// ── MatryoshkaConfig ──────────────────────────────────────────────────────────

/// Configuration for the Matryoshka encoder and retriever.
///
/// The full embedding has [`MatryoshkaConfig::full_dim`] components. Valid
/// nested prefix dimensions are listed in [`MatryoshkaConfig::nesting_dims`]
/// (ascending, all `<= full_dim`). Two-stage search shortlists candidates with
/// the cheap [`MatryoshkaConfig::shortlist_dim`]-truncated prefix before
/// reranking with the full dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct MatryoshkaConfig {
    /// Dimensionality of the full (longest) embedding.
    pub full_dim: usize,
    /// Valid nested prefix dimensions, ascending, each `<= full_dim`.
    pub nesting_dims: Vec<usize>,
    /// Prefix dimension used for the cheap stage-1 shortlist.
    pub shortlist_dim: usize,
    /// Shortlist size multiplier: stage 1 keeps `top_k * shortlist_multiplier`.
    pub shortlist_multiplier: usize,
    /// Per-dimension importance decay base applied before normalisation.
    ///
    /// Dimension `i` is scaled by `decay.powi(i)`, so a value in `(0, 1)` makes
    /// earlier dimensions dominate and prefixes meaningful.
    pub decay: f32,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            full_dim: 256,
            nesting_dims: vec![32, 64, 128, 256],
            shortlist_dim: 64,
            shortlist_multiplier: 4,
            decay: 0.98,
        }
    }
}

impl MatryoshkaConfig {
    /// Create a new configuration with the [`Default`] values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the full embedding dimension.
    #[must_use]
    pub fn with_full_dim(mut self, full_dim: usize) -> Self {
        self.full_dim = full_dim;
        self
    }

    /// Set the nested prefix dimensions (should be ascending, all `<= full_dim`).
    #[must_use]
    pub fn with_nesting_dims(mut self, nesting_dims: Vec<usize>) -> Self {
        self.nesting_dims = nesting_dims;
        self
    }

    /// Set the stage-1 shortlist prefix dimension.
    #[must_use]
    pub fn with_shortlist_dim(mut self, shortlist_dim: usize) -> Self {
        self.shortlist_dim = shortlist_dim;
        self
    }

    /// Set the shortlist size multiplier.
    #[must_use]
    pub fn with_shortlist_multiplier(mut self, shortlist_multiplier: usize) -> Self {
        self.shortlist_multiplier = shortlist_multiplier;
        self
    }

    /// Set the per-dimension importance decay base.
    #[must_use]
    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay = decay;
        self
    }

    /// Validate the configuration's invariants.
    ///
    /// # Errors
    ///
    /// Returns [`MatryoshkaError::InvalidDim`] when `full_dim` is zero, when
    /// `shortlist_dim` is zero or exceeds `full_dim`, or when any
    /// `nesting_dims` entry is zero, exceeds `full_dim`, or the sequence is not
    /// strictly ascending.
    pub fn validate(&self) -> Result<(), MatryoshkaError> {
        if self.full_dim == 0 {
            return Err(MatryoshkaError::InvalidDim(self.full_dim));
        }
        if self.shortlist_dim == 0 || self.shortlist_dim > self.full_dim {
            return Err(MatryoshkaError::InvalidDim(self.shortlist_dim));
        }
        let mut prev = 0usize;
        for &d in &self.nesting_dims {
            if d == 0 || d > self.full_dim || d <= prev {
                return Err(MatryoshkaError::InvalidDim(d));
            }
            prev = d;
        }
        Ok(())
    }
}

// ── MatryoshkaHit ─────────────────────────────────────────────────────────────

/// A single ranked search result from a [`crate::matryoshka::MatryoshkaRetriever`].
#[derive(Debug, Clone)]
pub struct MatryoshkaHit {
    /// Identifier of the retrieved document.
    pub id: crate::types::DocumentId,
    /// Final relevance score from the full-dimension rerank stage.
    pub score: f32,
    /// Cheap stage-1 score from the truncated shortlist prefix.
    pub shortlist_score: f32,
}
