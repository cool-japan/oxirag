//! Core data structures for SPLADE-style learned sparse retrieval.

use std::collections::HashMap;

use thiserror::Error;

use crate::types::DocumentId;

// ── SparseRetrievalError ──────────────────────────────────────────────────────

/// Errors raised while building or querying a [`SparseIndex`](crate::sparse_retrieval::SparseIndex).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SparseRetrievalError {
    /// The supplied corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// The supplied query was blank.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A search was attempted before the index had been built.
    #[error("index not built")]
    NotBuilt,
}

// ── SparseVector ──────────────────────────────────────────────────────────────

/// A sparse term → weight representation of a piece of text.
///
/// Only terms with a non-zero learned weight are stored. Two sparse vectors are
/// compared with a [`dot`](SparseVector::dot) product over their shared terms,
/// mirroring the inner-product retrieval used by SPLADE.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseVector {
    /// Map from term to its (learned, expansion-augmented) weight.
    pub terms: HashMap<String, f32>,
}

impl SparseVector {
    /// Create a sparse vector from an existing term → weight map.
    #[must_use]
    pub fn new(terms: HashMap<String, f32>) -> Self {
        Self { terms }
    }

    /// Sparse dot product over the terms shared by `self` and `other`.
    ///
    /// Terms present in only one vector contribute nothing, exactly as a dense
    /// dot product would treat their counterpart as zero.
    #[must_use]
    pub fn dot(&self, other: &SparseVector) -> f32 {
        // Iterate over the smaller map for efficiency; the result is identical.
        let (small, large) = if self.terms.len() <= other.terms.len() {
            (&self.terms, &other.terms)
        } else {
            (&other.terms, &self.terms)
        };
        small
            .iter()
            .filter_map(|(term, weight)| large.get(term).map(|w| weight * w))
            .sum()
    }

    /// Return the `n` highest-weighted terms, sorted by descending weight.
    ///
    /// Ties on weight are broken alphabetically by term so the ordering is
    /// deterministic.
    #[must_use]
    pub fn top_terms(&self, n: usize) -> Vec<(String, f32)> {
        let mut pairs: Vec<(String, f32)> = self
            .terms
            .iter()
            .map(|(term, weight)| (term.clone(), *weight))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        pairs.truncate(n);
        pairs
    }

    /// Number of non-zero terms held by the vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Return `true` when the vector holds no terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Euclidean (L2) norm of the weight vector.
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.terms.values().map(|w| w * w).sum::<f32>().sqrt()
    }
}

// ── SparseConfig ──────────────────────────────────────────────────────────────

/// Configuration controlling sparse encoding and term expansion.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseConfig {
    /// Number of co-occurring terms appended per present term during expansion.
    pub expansion_terms: usize,
    /// Multiplier applied inside the log-saturation of term weights.
    pub saturation: f32,
    /// Weights strictly below this threshold are pruned from the encoding.
    pub min_weight: f32,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            expansion_terms: 3,
            saturation: 1.0,
            min_weight: 0.01,
        }
    }
}

impl SparseConfig {
    /// Create a configuration with the default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of expansion terms added per present term.
    #[must_use]
    pub fn with_expansion_terms(mut self, expansion_terms: usize) -> Self {
        self.expansion_terms = expansion_terms;
        self
    }

    /// Set the saturation multiplier used in the log term-weight transform.
    #[must_use]
    pub fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation;
        self
    }

    /// Set the minimum weight below which terms are pruned.
    #[must_use]
    pub fn with_min_weight(mut self, min_weight: f32) -> Self {
        self.min_weight = min_weight;
        self
    }
}

// ── SparseHit ─────────────────────────────────────────────────────────────────

/// A single scored search result from a [`SparseIndex`](crate::sparse_retrieval::SparseIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct SparseHit {
    /// Identifier of the matched document.
    pub id: DocumentId,
    /// Sparse dot-product score against the query (higher is better).
    pub score: f32,
}
