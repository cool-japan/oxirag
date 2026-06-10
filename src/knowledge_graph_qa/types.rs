//! Types for the `knowledge_graph_qa` module.

use thiserror::Error;

use crate::fact_triple::Triple;

// ── KgqaConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`KgqaEngine`].
///
/// Controls BFS hop depth, retrieval breadth, and the minimum confidence
/// required for triples to appear in answers.
///
/// [`KgqaEngine`]: super::engine::KgqaEngine
#[derive(Debug, Clone)]
pub struct KgqaConfig {
    /// Maximum BFS hops when expanding the subgraph. Defaults to `2`.
    pub max_hops: usize,
    /// Maximum number of entities to include in the answer subgraph. Defaults
    /// to `5`.
    pub top_k: usize,
    /// Minimum triple confidence for evidence inclusion. Defaults to `0.3`.
    pub min_confidence: f32,
}

impl Default for KgqaConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            top_k: 5,
            min_confidence: 0.3,
        }
    }
}

impl KgqaConfig {
    /// Create a new [`KgqaConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of BFS hops.
    #[must_use]
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    /// Set the maximum number of entities in the answer subgraph.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum triple confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }
}

// ── KgqaAnswer ────────────────────────────────────────────────────────────────

/// The result of a knowledge-graph QA query.
#[derive(Debug, Clone)]
pub struct KgqaAnswer {
    /// Human-readable answer synthesised from the subgraph.
    pub answer: String,
    /// Names of graph entities used as evidence.
    pub evidence_entities: Vec<String>,
    /// Fact triples extracted from the subgraph.
    pub evidence_triples: Vec<Triple>,
    /// Overall answer confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

impl KgqaAnswer {
    /// Create a new [`KgqaAnswer`].
    #[must_use]
    pub fn new(
        answer: String,
        evidence_entities: Vec<String>,
        evidence_triples: Vec<Triple>,
        confidence: f32,
    ) -> Self {
        Self {
            answer,
            evidence_entities,
            evidence_triples,
            confidence,
        }
    }

    /// Returns `true` when the answer's confidence meets or exceeds `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Returns `true` when at least one evidence triple is present.
    #[must_use]
    pub fn has_triples(&self) -> bool {
        !self.evidence_triples.is_empty()
    }
}

// ── KgqaError ─────────────────────────────────────────────────────────────────

/// Errors that can occur during knowledge-graph QA.
#[derive(Debug, Error)]
pub enum KgqaError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// The graph contained no entities at all.
    #[error("No entities in graph")]
    NoEntities,

    /// The engine could not construct any answer from the available data.
    #[error("No answer found")]
    NoAnswer,
}
