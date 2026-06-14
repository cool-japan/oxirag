//! Types for the `drift_search` module.

use thiserror::Error;

use crate::types::DocumentId;

// ── CommunityReport ─────────────────────────────────────────────────────────────

/// A pre-computed report describing a single graph community.
///
/// DRIFT search is self-contained: the caller supplies these reports together
/// with the passages they reference. A report bundles a natural-language
/// `summary`, the salient `entities` mentioned in the community, and the
/// identifiers of the passages assigned to it.
#[derive(Debug, Clone)]
pub struct CommunityReport {
    /// Stable identifier of the community (caller-assigned).
    pub id: usize,
    /// Natural-language summary of the community.
    pub summary: String,
    /// Salient entity names that characterise the community.
    pub entities: Vec<String>,
    /// Identifiers of the passages belonging to this community.
    pub passage_ids: Vec<DocumentId>,
}

impl CommunityReport {
    /// Create a new community report.
    #[must_use]
    pub fn new(
        id: usize,
        summary: impl Into<String>,
        entities: Vec<String>,
        passage_ids: Vec<DocumentId>,
    ) -> Self {
        Self {
            id,
            summary: summary.into(),
            entities,
            passage_ids,
        }
    }

    /// Return `true` when the report has no summary text.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summary.trim().is_empty()
    }
}

// ── DriftStepKind ───────────────────────────────────────────────────────────────

/// The phase of a DRIFT search that produced a particular step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftStepKind {
    /// A community-level (global) retrieval step.
    Global,
    /// An entity-level (local) retrieval step driven by a follow-up query.
    Local,
}

impl DriftStepKind {
    /// Return `true` when this is a [`DriftStepKind::Global`] step.
    #[must_use]
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    /// Return `true` when this is a [`DriftStepKind::Local`] step.
    #[must_use]
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local)
    }

    /// A short, lowercase label for the step kind.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Local => "local",
        }
    }
}

// ── DriftStep ───────────────────────────────────────────────────────────────────

/// A single retrieval step recorded during a DRIFT search.
///
/// Each step captures the `query` that was issued, whether it was a global or
/// local step (`kind`), and the scored `results` it produced. For global steps
/// the results are `(community-as-document, score)` synthetic identifiers; for
/// local steps they are real passage identifiers.
#[derive(Debug, Clone)]
pub struct DriftStep {
    /// The query text issued for this step.
    pub query: String,
    /// Whether this was a global (community) or local (passage) step.
    pub kind: DriftStepKind,
    /// Scored results, ordered best-first.
    pub results: Vec<(DocumentId, f32)>,
}

impl DriftStep {
    /// Create a new step.
    #[must_use]
    pub fn new(
        query: impl Into<String>,
        kind: DriftStepKind,
        results: Vec<(DocumentId, f32)>,
    ) -> Self {
        Self {
            query: query.into(),
            kind,
            results,
        }
    }

    /// Number of results captured by this step.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Return `true` when this step produced no results.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

// ── DriftAnswer ─────────────────────────────────────────────────────────────────

/// The complete result of a DRIFT search.
///
/// Records every [`DriftStep`] that was taken, the community ids that were
/// selected by the global phase, the aggregated `final_context` passages, and a
/// synthesized `summary` of the most relevant community reports.
#[derive(Debug, Clone)]
pub struct DriftAnswer {
    /// All steps taken, in execution order (global first, then locals).
    pub steps: Vec<DriftStep>,
    /// Identifiers of the communities selected by the global phase.
    pub communities: Vec<usize>,
    /// Aggregated, de-duplicated passage identifiers ordered by relevance.
    pub final_context: Vec<DocumentId>,
    /// Synthesized natural-language summary of the selected communities.
    pub summary: String,
}

impl DriftAnswer {
    /// Return `true` when no passages were aggregated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.final_context.is_empty()
    }

    /// Iterator over the [`DriftStepKind::Global`] steps.
    pub fn global_steps(&self) -> impl Iterator<Item = &DriftStep> {
        self.steps.iter().filter(|s| s.kind.is_global())
    }

    /// Iterator over the [`DriftStepKind::Local`] steps.
    pub fn local_steps(&self) -> impl Iterator<Item = &DriftStep> {
        self.steps.iter().filter(|s| s.kind.is_local())
    }
}

// ── DriftConfig ─────────────────────────────────────────────────────────────────

/// Configuration for a [`crate::drift_search::DriftSearchEngine`].
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// Number of top communities selected by the global phase.
    ///
    /// Defaults to `3`.
    pub top_communities: usize,
    /// Maximum number of follow-up sub-queries generated per search.
    ///
    /// Defaults to `2`.
    pub follow_ups: usize,
    /// Number of passages retained per local step.
    ///
    /// Defaults to `5`.
    pub top_local: usize,
    /// Embedding dimension of the lexical pseudo-embedding.
    ///
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            top_communities: 3,
            follow_ups: 2,
            top_local: 5,
            dim: 128,
        }
    }
}

impl DriftConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of communities selected by the global phase.
    #[must_use]
    pub fn with_top_communities(mut self, v: usize) -> Self {
        self.top_communities = v;
        self
    }

    /// Set the maximum number of follow-up sub-queries.
    #[must_use]
    pub fn with_follow_ups(mut self, v: usize) -> Self {
        self.follow_ups = v;
        self
    }

    /// Set the number of passages retained per local step.
    #[must_use]
    pub fn with_top_local(mut self, v: usize) -> Self {
        self.top_local = v;
        self
    }

    /// Set the embedding dimension.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
}

// ── DriftError ──────────────────────────────────────────────────────────────────

/// Errors from the `drift_search` module.
#[derive(Debug, Error)]
pub enum DriftError {
    /// No community reports were supplied to [`crate::drift_search::DriftSearchEngine::build`].
    #[error("no communities")]
    EmptyCommunities,
    /// The supplied query string was empty.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A search was attempted before the engine was built.
    #[error("engine not built")]
    NotBuilt,
}
