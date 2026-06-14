//! Types for the `source_credibility` module.
//!
//! These types describe the **authority signals**, the **blend configuration**,
//! and the resulting **per-document credibility score**.  They are completely
//! deterministic: no clock access, no randomness, no ML.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ── AuthoritySignals ──────────────────────────────────────────────────────────

/// Out-of-band authority knowledge supplied by the caller.
///
/// `trusted_sources` is a set of source strings (matched against
/// [`crate::types::Document::source`]) that are considered authoritative.
/// `author_reputation` maps an author name (matched against a document's
/// `"author"` metadata key) to a reputation score in `[0.0, 1.0]`.
#[derive(Debug, Clone, Default)]
pub struct AuthoritySignals {
    /// Set of trusted source identifiers (e.g. domains, file paths, URLs).
    pub trusted_sources: HashSet<String>,
    /// Per-author reputation scores; values are expected in `[0.0, 1.0]`.
    pub author_reputation: HashMap<String, f32>,
}

impl AuthoritySignals {
    /// Create an empty set of authority signals.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trusted_sources: HashSet::new(),
            author_reputation: HashMap::new(),
        }
    }

    /// Builder: register a trusted source.
    #[must_use]
    pub fn with_trusted_source(mut self, source: impl Into<String>) -> Self {
        self.trusted_sources.insert(source.into());
        self
    }

    /// Builder: register an author with the given reputation score.
    ///
    /// The reputation is stored as-provided; [`crate::source_credibility::CredibilityScorer`]
    /// clamps it to `[0.0, 1.0]` when scoring.
    #[must_use]
    pub fn with_author(mut self, name: impl Into<String>, reputation: f32) -> Self {
        self.author_reputation.insert(name.into(), reputation);
        self
    }

    /// Return `true` when `source` is in the trusted-source set.
    #[must_use]
    pub fn is_trusted(&self, source: &str) -> bool {
        self.trusted_sources.contains(source)
    }

    /// Look up an author's reputation, returning `None` when unknown.
    #[must_use]
    pub fn reputation_of(&self, author: &str) -> Option<f32> {
        self.author_reputation.get(author).copied()
    }
}

// ── CredibilityConfig ─────────────────────────────────────────────────────────

/// Blend configuration for [`crate::source_credibility::CredibilityScorer`].
///
/// The three signal weights (`pagerank_weight`, `recency_weight`,
/// `authority_weight`) are **normalised** before use, so only their *relative*
/// magnitudes matter.
#[derive(Debug, Clone)]
pub struct CredibilityConfig {
    /// Relative weight of the citation-graph `PageRank` signal. Default `0.4`.
    pub pagerank_weight: f32,
    /// Relative weight of the recency (age-decay) signal. Default `0.3`.
    pub recency_weight: f32,
    /// Relative weight of the metadata-authority signal. Default `0.3`.
    pub authority_weight: f32,
    /// `PageRank` damping factor in `(0.0, 1.0)`. Default `0.85`.
    pub damping: f32,
    /// Number of `PageRank` power-iterations. Default `50`.
    pub iterations: usize,
    /// Recency half-life in days (score halves every `half_life_days`). Default `365.0`.
    pub half_life_days: f32,
}

impl Default for CredibilityConfig {
    fn default() -> Self {
        Self {
            pagerank_weight: 0.4,
            recency_weight: 0.3,
            authority_weight: 0.3,
            damping: 0.85,
            iterations: 50,
            half_life_days: 365.0,
        }
    }
}

impl CredibilityConfig {
    /// Create a configuration with the default blend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the `PageRank` weight.
    #[must_use]
    pub fn with_pagerank_weight(mut self, w: f32) -> Self {
        self.pagerank_weight = w;
        self
    }

    /// Builder: set the recency weight.
    #[must_use]
    pub fn with_recency_weight(mut self, w: f32) -> Self {
        self.recency_weight = w;
        self
    }

    /// Builder: set the authority weight.
    #[must_use]
    pub fn with_authority_weight(mut self, w: f32) -> Self {
        self.authority_weight = w;
        self
    }

    /// Builder: set the `PageRank` damping factor.
    #[must_use]
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Builder: set the number of `PageRank` iterations.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Builder: set the recency half-life in days.
    #[must_use]
    pub fn with_half_life_days(mut self, half_life_days: f32) -> Self {
        self.half_life_days = half_life_days;
        self
    }

    /// Sum of the three signal weights.
    #[must_use]
    pub fn weight_sum(&self) -> f32 {
        self.pagerank_weight + self.recency_weight + self.authority_weight
    }

    /// Return a copy with the three signal weights normalised to sum to `1.0`.
    ///
    /// Falls back to a uniform `1/3` split when the original sum is zero.
    #[must_use]
    pub fn normalized_weights(&self) -> (f32, f32, f32) {
        let s = self.weight_sum();
        if s.abs() < f32::EPSILON {
            let third = 1.0 / 3.0;
            return (third, third, third);
        }
        (
            self.pagerank_weight / s,
            self.recency_weight / s,
            self.authority_weight / s,
        )
    }
}

// ── CredibilityScore ──────────────────────────────────────────────────────────

/// Per-document credibility score with its three contributing components.
///
/// `total` is the normalised weighted blend of `pagerank`, `recency`, and
/// `authority`, each of which lies in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CredibilityScore {
    /// Blended credibility in `[0.0, 1.0]`.
    pub total: f32,
    /// Normalised `PageRank` component in `[0.0, 1.0]`.
    pub pagerank: f32,
    /// Recency (age-decay) component in `[0.0, 1.0]`.
    pub recency: f32,
    /// Metadata-authority component in `[0.0, 1.0]`.
    pub authority: f32,
}

impl CredibilityScore {
    /// Construct a score from its parts.
    #[must_use]
    pub fn new(total: f32, pagerank: f32, recency: f32, authority: f32) -> Self {
        Self {
            total,
            pagerank,
            recency,
            authority,
        }
    }

    /// Human-readable credibility label.
    ///
    /// Returns `"high"` (≥0.7), `"medium"` (≥0.4), or `"low"` (<0.4).
    #[must_use]
    pub fn label(&self) -> &str {
        if self.total >= 0.7 {
            "high"
        } else if self.total >= 0.4 {
            "medium"
        } else {
            "low"
        }
    }
}

// ── SourceCredibilityError ────────────────────────────────────────────────────

/// Errors produced by the `source_credibility` module.
#[derive(Debug, Error)]
pub enum SourceCredibilityError {
    /// `PageRank` was requested on a graph with no nodes.
    #[error("empty graph")]
    EmptyGraph,
}
