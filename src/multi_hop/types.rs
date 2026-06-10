//! Types for the `multi_hop` module.

use thiserror::Error;

// ── HopConfig ─────────────────────────────────────────────────────────────────

/// Configuration for multi-hop entity-chain traversal.
///
/// Controls how many hops are followed, how many entities are expanded per
/// hop, and the number of documents retrieved per search step.
#[derive(Debug, Clone)]
pub struct HopConfig {
    /// Maximum number of graph hops to follow. Defaults to `3`.
    pub max_hops: usize,
    /// How many neighbour entities to expand at each hop. Defaults to `5`.
    pub entities_per_hop: usize,
    /// Top-k documents to retrieve per search step. Defaults to `5`.
    pub top_k: usize,
}

impl Default for HopConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            entities_per_hop: 5,
            top_k: 5,
        }
    }
}

impl HopConfig {
    /// Create a new [`HopConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of hops.
    #[must_use]
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    /// Set the number of entities expanded per hop.
    #[must_use]
    pub fn with_entities_per_hop(mut self, entities_per_hop: usize) -> Self {
        self.entities_per_hop = entities_per_hop;
        self
    }

    /// Set the top-k retrieval count per hop.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

// ── EntityMention ─────────────────────────────────────────────────────────────

/// A span in the query text that was matched to a graph entity.
#[derive(Debug, Clone)]
pub struct EntityMention {
    /// The raw text span that matched an entity name.
    pub text: String,
    /// The ID of the matched `GraphEntity`, if one was found.
    pub matched_id: Option<String>,
}

impl EntityMention {
    /// Create a new [`EntityMention`] with optional entity ID.
    #[must_use]
    pub fn new(text: impl Into<String>, matched_id: Option<String>) -> Self {
        Self {
            text: text.into(),
            matched_id,
        }
    }
}

// ── HopState ──────────────────────────────────────────────────────────────────

/// Snapshot of traversal state at a single hop.
#[derive(Debug, Clone, Default)]
pub struct HopState {
    /// The hop index (0-based).
    pub hop: usize,
    /// Entity IDs active at this hop.
    pub entity_ids: Vec<String>,
    /// Document IDs retrieved at this hop.
    pub doc_ids: Vec<String>,
}

impl HopState {
    /// Create a new [`HopState`].
    #[must_use]
    pub fn new(hop: usize, entity_ids: Vec<String>, doc_ids: Vec<String>) -> Self {
        Self {
            hop,
            entity_ids,
            doc_ids,
        }
    }
}

// ── MultiHopResult ────────────────────────────────────────────────────────────

/// The outcome of a multi-hop retrieval run.
#[derive(Debug, Clone)]
pub struct MultiHopResult {
    /// Synthesised answer built from the collected documents.
    pub answer: String,
    /// Per-hop traversal trace for inspection and debugging.
    pub trace: Vec<HopState>,
    /// How many hops were actually executed.
    pub hops_used: usize,
}

impl MultiHopResult {
    /// Create a new [`MultiHopResult`].
    #[must_use]
    pub fn new(answer: String, trace: Vec<HopState>, hops_used: usize) -> Self {
        Self {
            answer,
            trace,
            hops_used,
        }
    }

    /// Total number of unique documents collected across all hops.
    #[must_use]
    pub fn total_docs(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for state in &self.trace {
            for id in &state.doc_ids {
                seen.insert(id.as_str());
            }
        }
        seen.len()
    }

    /// Returns `true` if the answer string is non-empty.
    #[must_use]
    pub fn has_answer(&self) -> bool {
        !self.answer.trim().is_empty()
    }
}

// ── MultiHopError ─────────────────────────────────────────────────────────────

/// Errors that can occur during multi-hop traversal.
#[derive(Debug, Error)]
pub enum MultiHopError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// No entities in the graph matched the query text.
    #[error("No entities matched the query")]
    NoEntitiesFound,

    /// The underlying retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
}
