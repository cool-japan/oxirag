//! Core types for the `astute_rag` module.
//!
//! Defines the knowledge-source enumeration, knowledge statements, conflicts,
//! the consolidated-knowledge result, the [`InternalKnowledge`] recall trait
//! (plus a mock implementation), configuration, and the error type used by the
//! Astute RAG consolidator.

use thiserror::Error;

use crate::types::DocumentId;

// ── KnowledgeSource ───────────────────────────────────────────────────────────

/// Where a piece of knowledge originates.
///
/// Astute RAG reconciles two kinds of knowledge: the model's parametric
/// ([`KnowledgeSource::Internal`]) knowledge recalled from its weights, and
/// external knowledge extracted from a retrieved [`crate::types::Document`]
/// ([`KnowledgeSource::External`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeSource {
    /// Parametric knowledge recalled by the model itself.
    Internal,
    /// Knowledge extracted from the retrieved document with the given ID.
    External(DocumentId),
}

impl KnowledgeSource {
    /// Return a short lowercase label for this source kind.
    ///
    /// `Internal` maps to `"internal"`; any `External(..)` maps to `"external"`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Internal => "internal",
            Self::External(_) => "external",
        }
    }

    /// Whether this source is [`KnowledgeSource::Internal`].
    #[must_use]
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Internal)
    }

    /// Whether this source is [`KnowledgeSource::External`].
    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(self, Self::External(_))
    }

    /// Return the originating document ID when this is an external source.
    #[must_use]
    pub fn document_id(&self) -> Option<&DocumentId> {
        match self {
            Self::Internal => None,
            Self::External(id) => Some(id),
        }
    }
}

// ── KnowledgeStatement ────────────────────────────────────────────────────────

/// A single atomic knowledge claim with its provenance and reliability.
#[derive(Debug, Clone)]
pub struct KnowledgeStatement {
    /// The natural-language claim text.
    pub content: String,
    /// Where the claim came from.
    pub source: KnowledgeSource,
    /// Estimated reliability of the claim, in `[0.0, 1.0]`.
    ///
    /// For external statements this is derived from corroboration — how many
    /// retrieved documents share the claim's salient terms.
    pub reliability: f32,
}

impl KnowledgeStatement {
    /// Create a new knowledge statement.
    #[must_use]
    pub fn new(content: impl Into<String>, source: KnowledgeSource, reliability: f32) -> Self {
        Self {
            content: content.into(),
            source,
            reliability,
        }
    }

    /// Convenience constructor for an internal (parametric) statement.
    #[must_use]
    pub fn internal(content: impl Into<String>, reliability: f32) -> Self {
        Self::new(content, KnowledgeSource::Internal, reliability)
    }

    /// Convenience constructor for an external statement from `source_id`.
    #[must_use]
    pub fn external(content: impl Into<String>, source_id: DocumentId, reliability: f32) -> Self {
        Self::new(content, KnowledgeSource::External(source_id), reliability)
    }

    /// Whether this statement comes from parametric knowledge.
    #[must_use]
    pub fn is_internal(&self) -> bool {
        self.source.is_internal()
    }
}

// ── KnowledgeConflict ─────────────────────────────────────────────────────────

/// A detected disagreement between an internal claim and an external claim.
///
/// A conflict arises when the two statements share a subject but assert
/// contradictory facts (e.g. a negation mismatch or differing numbers). The
/// `resolution` field records which source was preferred.
#[derive(Debug, Clone)]
pub struct KnowledgeConflict {
    /// The internal (parametric) claim text.
    pub internal: String,
    /// The external (retrieved) claim text.
    pub external: String,
    /// Which source won the reconciliation.
    pub resolution: KnowledgeSource,
}

impl KnowledgeConflict {
    /// Create a new conflict record.
    #[must_use]
    pub fn new(
        internal: impl Into<String>,
        external: impl Into<String>,
        resolution: KnowledgeSource,
    ) -> Self {
        Self {
            internal: internal.into(),
            external: external.into(),
            resolution,
        }
    }

    /// Whether the conflict was resolved in favour of the external source.
    #[must_use]
    pub fn resolved_external(&self) -> bool {
        self.resolution.is_external()
    }
}

// ── ConsolidatedKnowledge ─────────────────────────────────────────────────────

/// The final reconciled output of an Astute RAG consolidation.
#[derive(Debug, Clone)]
pub struct ConsolidatedKnowledge {
    /// The surviving (winning) statements after reconciliation.
    pub statements: Vec<KnowledgeStatement>,
    /// All conflicts detected between internal and external knowledge.
    pub conflicts: Vec<KnowledgeConflict>,
    /// The synthesized consolidated answer with source attribution.
    pub answer: String,
}

impl ConsolidatedKnowledge {
    /// Create a new consolidated-knowledge result.
    #[must_use]
    pub fn new(
        statements: Vec<KnowledgeStatement>,
        conflicts: Vec<KnowledgeConflict>,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            statements,
            conflicts,
            answer: answer.into(),
        }
    }

    /// Number of surviving statements.
    #[must_use]
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// Number of detected conflicts.
    #[must_use]
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Whether any internal/external conflicts were detected.
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

// ── InternalKnowledge ─────────────────────────────────────────────────────────

/// A source of the model's parametric (internal) knowledge.
///
/// Implementors recall the claims the model already "knows" about a query,
/// independent of any retrieved documents. These statements are reconciled
/// against external knowledge by the consolidator.
pub trait InternalKnowledge {
    /// Recall parametric-knowledge statements for the query.
    fn recall(&self, query: &str) -> Vec<String>;
}

/// A scripted [`InternalKnowledge`] source for tests and examples.
///
/// Returns its fixed list of statements verbatim, ignoring the query.
#[derive(Debug, Clone, Default)]
pub struct MockInternalKnowledge {
    /// The statements returned by every call to [`MockInternalKnowledge::recall`].
    pub statements: Vec<String>,
}

impl MockInternalKnowledge {
    /// Create a mock source returning the given `statements`.
    #[must_use]
    pub fn new(statements: Vec<String>) -> Self {
        Self { statements }
    }

    /// Append a statement, returning `self` for chaining.
    #[must_use]
    pub fn with_statement(mut self, statement: impl Into<String>) -> Self {
        self.statements.push(statement.into());
        self
    }
}

impl InternalKnowledge for MockInternalKnowledge {
    fn recall(&self, _query: &str) -> Vec<String> {
        self.statements.clone()
    }
}

// ── AstuteConfig ──────────────────────────────────────────────────────────────

/// Configuration for the Astute RAG consolidator.
#[derive(Debug, Clone)]
pub struct AstuteConfig {
    /// Reliability threshold separating high- from low-reliability claims.
    ///
    /// External claims with reliability below this value are considered
    /// uncorroborated; in a conflict against an internal claim they may lose to
    /// internal knowledge even when `prefer_external` is set. Default: `0.5`.
    pub reliability_threshold: f32,

    /// Whether external knowledge is preferred when it is corroborated.
    ///
    /// When `true` (the default), a conflict is resolved toward the external
    /// claim provided its reliability meets `reliability_threshold`. When
    /// `false`, low-reliability external claims yield to internal knowledge.
    pub prefer_external: bool,
}

impl Default for AstuteConfig {
    fn default() -> Self {
        Self {
            reliability_threshold: 0.5,
            prefer_external: true,
        }
    }
}

impl AstuteConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to [`AstuteConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reliability threshold.
    #[must_use]
    pub fn with_reliability_threshold(mut self, reliability_threshold: f32) -> Self {
        self.reliability_threshold = reliability_threshold;
        self
    }

    /// Set whether corroborated external knowledge is preferred.
    #[must_use]
    pub fn with_prefer_external(mut self, prefer_external: bool) -> Self {
        self.prefer_external = prefer_external;
        self
    }
}

// ── AstuteError ───────────────────────────────────────────────────────────────

/// Errors produced by the Astute RAG consolidator.
#[derive(Debug, Error)]
pub enum AstuteError {
    /// The query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
    /// Neither internal nor external knowledge was supplied.
    #[error("no knowledge to consolidate")]
    NoKnowledge,
}
