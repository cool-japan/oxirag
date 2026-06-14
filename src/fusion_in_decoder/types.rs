//! Types for the `fusion_in_decoder` module.

use crate::types::DocumentId;
use thiserror::Error;

// ── PassageEvidence ───────────────────────────────────────────────────────────

/// Evidence extracted independently from a single retrieved passage.
///
/// Each passage contributes exactly one piece of evidence: the sentence within
/// the passage that is most relevant to the query, paired with a relevance
/// weight (the query↔sentence token-overlap score).
#[derive(Debug, Clone, PartialEq)]
pub struct PassageEvidence {
    /// Identifier of the passage (document) the evidence was drawn from.
    pub passage_id: DocumentId,
    /// The most query-relevant sentence extracted from the passage.
    pub evidence: String,
    /// Relevance weight of the evidence relative to the query (`0.0..=1.0`).
    pub relevance: f32,
}

impl PassageEvidence {
    /// Create a new piece of passage evidence.
    #[must_use]
    pub fn new(passage_id: DocumentId, evidence: impl Into<String>, relevance: f32) -> Self {
        Self {
            passage_id,
            evidence: evidence.into(),
            relevance,
        }
    }

    /// Returns `true` if the evidence's relevance meets or exceeds `threshold`.
    #[must_use]
    pub fn is_relevant(&self, threshold: f32) -> bool {
        self.relevance >= threshold
    }
}

// ── FidConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`FusionInDecoder`](crate::fusion_in_decoder::FusionInDecoder).
#[derive(Debug, Clone, PartialEq)]
pub struct FidConfig {
    /// Maximum number of passages to retain after sorting by relevance.
    ///
    /// Defaults to `5`.
    pub top_passages: usize,
    /// Nominal evidence-encoding dimensionality (reserved for future scoring).
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Token-Jaccard similarity above which two evidence strings are considered
    /// near-identical and deduplicated.
    ///
    /// Defaults to `0.8`.
    pub dedup_threshold: f32,
}

impl Default for FidConfig {
    fn default() -> Self {
        Self {
            top_passages: 5,
            dim: 128,
            dedup_threshold: 0.8,
        }
    }
}

impl FidConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of passages to retain.
    #[must_use]
    pub fn with_top_passages(mut self, top_passages: usize) -> Self {
        self.top_passages = top_passages;
        self
    }

    /// Set the nominal evidence-encoding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the deduplication similarity threshold.
    #[must_use]
    pub fn with_dedup_threshold(mut self, dedup_threshold: f32) -> Self {
        self.dedup_threshold = dedup_threshold;
        self
    }
}

// ── FusedAnswer ───────────────────────────────────────────────────────────────

/// The result of fusing per-passage evidence into a single answer.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedAnswer {
    /// The fused answer text, ordered by descending passage relevance.
    pub answer: String,
    /// The surviving per-passage evidence that contributed to the answer.
    pub evidence: Vec<PassageEvidence>,
    /// Identifiers of the passages that contributed to the answer (no duplicates).
    pub attributions: Vec<DocumentId>,
}

impl FusedAnswer {
    /// Returns `true` when no evidence contributed to the answer.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.evidence.is_empty()
    }

    /// Returns the number of passages that contributed to the answer.
    #[must_use]
    pub fn passage_count(&self) -> usize {
        self.attributions.len()
    }
}

// ── FidError ──────────────────────────────────────────────────────────────────

/// Errors that can occur during fusion-in-decoder processing.
#[derive(Debug, Error)]
pub enum FidError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// No passages were provided for evidence extraction.
    #[error("corpus is empty")]
    EmptyCorpus,
}
