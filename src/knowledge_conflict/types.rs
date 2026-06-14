//! Types for the `knowledge_conflict` module.

use thiserror::Error;

// ── ConflictPolicy ──────────────────────────────────────────────────────────────

/// Policy used to decide which side of a [`PassageConflict`] wins during
/// resolution.
///
/// Each variant maps to a distinct, deterministic tie-break strategy applied by
/// the [`crate::knowledge_conflict::ConflictResolver`]:
///
/// * [`ConflictPolicy::Recency`] — the passage backed by the *newer* source wins
///   (lower age in days).
/// * [`ConflictPolicy::Authority`] — the passage backed by the *higher-authority*
///   source wins.
/// * [`ConflictPolicy::Majority`] — the claim whose content terms are echoed by
///   *more* passages in the corpus wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictPolicy {
    /// The newer source wins (lower age in days).
    Recency,
    /// The higher-authority source wins.
    Authority,
    /// The claim supported by more passages wins.
    #[default]
    Majority,
}

impl ConflictPolicy {
    /// Return a stable lowercase string representation of the policy.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Recency => "recency",
            Self::Authority => "authority",
            Self::Majority => "majority",
        }
    }
}

// ── ConflictKind ────────────────────────────────────────────────────────────────

/// The category of contradiction detected between two passages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictKind {
    /// Exactly one of the two claims carries a negation marker about the same
    /// subject (e.g. "X is approved" vs "X is not approved").
    Negation,
    /// The two claims cite different numbers about the same subject (e.g.
    /// "population is 5 million" vs "population is 8 million").
    Numeric,
    /// The two claims cite different years about the same subject (e.g.
    /// "founded in 1990" vs "founded in 2005").
    Temporal,
}

impl ConflictKind {
    /// Return a stable lowercase string representation of the kind.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Negation => "negation",
            Self::Numeric => "numeric",
            Self::Temporal => "temporal",
        }
    }
}

// ── PassageConflict ─────────────────────────────────────────────────────────────

/// A single detected contradiction *between* two retrieved passages.
///
/// The `passage_a` and `passage_b` fields are indices into the document slice
/// that was passed to [`crate::knowledge_conflict::ConflictDetector::detect`].
/// `claim_a` and `claim_b` are the specific conflicting sentences drawn from
/// those passages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassageConflict {
    /// Index of the first passage involved in the conflict.
    pub passage_a: usize,
    /// Index of the second passage involved in the conflict.
    pub passage_b: usize,
    /// The conflicting sentence taken from `passage_a`.
    pub claim_a: String,
    /// The conflicting sentence taken from `passage_b`.
    pub claim_b: String,
    /// The category of the detected contradiction.
    pub kind: ConflictKind,
}

impl PassageConflict {
    /// Construct a new [`PassageConflict`].
    #[must_use]
    pub fn new(
        passage_a: usize,
        passage_b: usize,
        claim_a: impl Into<String>,
        claim_b: impl Into<String>,
        kind: ConflictKind,
    ) -> Self {
        Self {
            passage_a,
            passage_b,
            claim_a: claim_a.into(),
            claim_b: claim_b.into(),
            kind,
        }
    }
}

// ── ConflictResolution ──────────────────────────────────────────────────────────

/// The outcome of resolving a single [`PassageConflict`] under a
/// [`ConflictPolicy`].
///
/// `winner` is the index of the passage whose claim is selected; `losers` holds
/// the index (or indices) of the superseded passage(s). The `rationale` explains,
/// in human-readable form, why the winner was chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictResolution {
    /// Index of the passage whose claim was selected as authoritative.
    pub winner: usize,
    /// Indices of the passages whose claims were superseded.
    pub losers: Vec<usize>,
    /// The policy applied to produce this resolution.
    pub policy: ConflictPolicy,
    /// A human-readable explanation of the decision.
    pub rationale: String,
}

impl ConflictResolution {
    /// Construct a new [`ConflictResolution`].
    #[must_use]
    pub fn new(
        winner: usize,
        losers: Vec<usize>,
        policy: ConflictPolicy,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            winner,
            losers,
            policy,
            rationale: rationale.into(),
        }
    }
}

// ── KnowledgeConflictConfig ─────────────────────────────────────────────────────

/// Configuration shared by the [`crate::knowledge_conflict::ConflictDetector`]
/// and [`crate::knowledge_conflict::ConflictResolver`].
#[derive(Debug, Clone)]
pub struct KnowledgeConflictConfig {
    /// The resolution policy applied when a conflict is resolved.
    ///
    /// Default: [`ConflictPolicy::Majority`].
    pub policy: ConflictPolicy,
    /// Minimum number of shared content tokens required for two sentences to be
    /// considered as describing the *same subject*.
    ///
    /// Default: `2`.
    pub min_shared_terms: usize,
}

impl Default for KnowledgeConflictConfig {
    fn default() -> Self {
        Self {
            policy: ConflictPolicy::Majority,
            min_shared_terms: 2,
        }
    }
}

impl KnowledgeConflictConfig {
    /// Create a new [`KnowledgeConflictConfig`] with default values.
    ///
    /// Equivalent to [`KnowledgeConflictConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the resolution policy.
    #[must_use]
    pub fn with_policy(mut self, policy: ConflictPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the minimum number of shared content tokens required to treat two
    /// sentences as describing the same subject.
    #[must_use]
    pub fn with_min_shared_terms(mut self, min_shared_terms: usize) -> Self {
        self.min_shared_terms = min_shared_terms;
        self
    }
}

// ── KnowledgeConflictError ──────────────────────────────────────────────────────

/// Errors produced by the knowledge-conflict pipeline.
#[derive(Debug, Error)]
pub enum KnowledgeConflictError {
    /// No documents were provided to a checked detection call.
    #[error("corpus is empty")]
    EmptyCorpus,
}
