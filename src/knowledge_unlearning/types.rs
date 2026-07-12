//! Types for the `knowledge_unlearning` module: the request/target
//! vocabulary, scope-resolution records, configuration, audit verdicts, and
//! the auditable [`UnlearningCertificate`].

use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::types::DocumentId;

use super::dedup::fnv1a;

// ── UnlearningTarget ─────────────────────────────────────────────────────────

/// What an [`UnlearningRequest`] asks to have forgotten.
///
/// Every variant is a *seed* for scope resolution, not the full set of
/// content that must be removed — see
/// [`crate::knowledge_unlearning::UnlearningScopeResolver`] for how a seed
/// expands into a full [`UnlearningScope`] covering near-duplicates and
/// derived artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnlearningTarget {
    /// Forget one specific document, named by its id.
    DocumentId(DocumentId),
    /// Forget every document whose [`crate::types::Document::source`]
    /// equals this string (a coarser, source-level request — e.g. "every
    /// chunk ingested from this URL").
    SourceId(String),
    /// Forget every document whose content contains this pattern as a
    /// case-insensitive substring.
    ///
    /// This is a literal substring predicate, not a regular expression —
    /// deliberately, to keep matching dependency-free and its behavior
    /// obvious to a compliance reviewer.
    ContentPattern(String),
}

impl UnlearningTarget {
    /// A stable string key identifying this target across repeated
    /// [`UnlearningEngine::unlearn`](super::engine::UnlearningEngine::unlearn)
    /// calls, used to recognize a request as a repeat of a target this
    /// engine has handled before (see [`UnlearningStatus::AlreadyUnlearned`]).
    #[must_use]
    pub fn history_key(&self) -> String {
        match self {
            Self::DocumentId(id) => format!("id:{}", id.as_str()),
            Self::SourceId(source) => format!("source:{source}"),
            Self::ContentPattern(pattern) => format!("pattern:{pattern}"),
        }
    }
}

// ── UnlearningRequest ────────────────────────────────────────────────────────

/// A single right-to-be-forgotten request.
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningRequest {
    /// What to forget.
    pub target: UnlearningTarget,
    /// An optional free-text justification (e.g. a ticket or legal-basis
    /// reference) carried through into the issued
    /// [`UnlearningCertificate`] for audit trails.
    pub reason: Option<String>,
    /// When this request was created.
    pub requested_at: DateTime<Utc>,
}

impl UnlearningRequest {
    /// Create a new request for `target`, timestamped now.
    #[must_use]
    pub fn new(target: UnlearningTarget) -> Self {
        Self {
            target,
            reason: None,
            requested_at: Utc::now(),
        }
    }

    /// Attach a free-text reason and return `self` for chaining.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

// ── UnlearningArtifactKind ───────────────────────────────────────────────────

/// The kind of derived artifact a document represents, as recorded in an
/// [`crate::knowledge_unlearning::UnlearningArtifactRegistry`] entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnlearningArtifactKind {
    /// A summary generated from (part of) the source document.
    Summary,
    /// A cached answer whose text was derived from the source document.
    CachedAnswer,
    /// An entity, fact, or triple extracted from the source document.
    ExtractedEntity,
    /// Any other derived-artifact relationship, named by the caller.
    Other(String),
}

// ── UnlearningScopeReason ────────────────────────────────────────────────────

/// Why a document appears in an [`UnlearningScope`].
#[derive(Debug, Clone, PartialEq)]
pub enum UnlearningScopeReason {
    /// This document *is* the [`UnlearningTarget::DocumentId`] named in the
    /// request.
    ExactMatch,
    /// This document's `source` matched an [`UnlearningTarget::SourceId`].
    SourceMatch,
    /// This document's content matched an
    /// [`UnlearningTarget::ContentPattern`].
    ContentMatch,
    /// This document was found to be a near-duplicate of a document already
    /// in scope, via shingle/`MinHash` similarity.
    NearDuplicate {
        /// The id of the already-in-scope document this one duplicates.
        of: DocumentId,
        /// The estimated `Jaccard` similarity that triggered the match, in
        /// `[0.0, 1.0]`.
        similarity: f64,
    },
    /// This document is a derived artifact (summary, cached answer,
    /// extracted entity, ...) of a document already in scope, per the
    /// [`crate::knowledge_unlearning::UnlearningArtifactRegistry`].
    DerivedArtifact {
        /// The id of the already-in-scope source document this artifact was
        /// derived from.
        of: DocumentId,
        /// What kind of derived artifact this is.
        kind: UnlearningArtifactKind,
    },
    /// This document was not part of the initial scope resolution but was
    /// added during a later audit round because a post-deletion leakage
    /// probe found it still carrying the forgotten content.
    Escalated {
        /// The audit round (1-based) that flagged this document.
        round: usize,
        /// The leakage score that triggered escalation.
        score: f64,
    },
}

// ── UnlearningScopeItem / UnlearningScope ────────────────────────────────────

/// A single document identified as needing removal, with the reason it was
/// included.
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningScopeItem {
    /// The id of the document that must be removed.
    pub document_id: DocumentId,
    /// Why this document is in scope.
    pub reason: UnlearningScopeReason,
}

/// The full set of documents an [`UnlearningRequest`] resolves to: not just
/// the named target, but every near-duplicate and derived artifact found to
/// carry the same content.
///
/// This is the module's core contribution over a bare `delete()` call: a
/// [`UnlearningScope`] is *evidence-backed and enumerable* — a compliance
/// reviewer can see exactly which documents were removed and why, rather
/// than trusting that one `delete()` call was sufficient.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnlearningScope {
    /// Every document identified for removal, each with its reason.
    pub items: Vec<UnlearningScopeItem>,
}

impl UnlearningScope {
    /// An empty scope (nothing found to remove).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when no documents were identified.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The number of documents identified.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` when `id` already appears somewhere in this scope.
    #[must_use]
    pub fn contains(&self, id: &DocumentId) -> bool {
        self.items.iter().any(|item| &item.document_id == id)
    }

    /// The ids of every document in this scope, in resolution order.
    #[must_use]
    pub fn document_ids(&self) -> Vec<DocumentId> {
        self.items
            .iter()
            .map(|item| item.document_id.clone())
            .collect()
    }
}

// ── UnlearningConfig ──────────────────────────────────────────────────────────

/// Tunable parameters for scope resolution, the post-deletion audit, and the
/// escalation loop.
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningConfig {
    /// Number of words per shingle for near-duplicate detection and
    /// audit-similarity scoring. Smaller shingles are more tolerant of
    /// paraphrasing (word reordering breaks fewer shingles) at the cost of
    /// coarser discrimination. Defaults to `2`.
    pub shingle_size: usize,
    /// Number of `MinHash` permutations (signature length). Larger values
    /// reduce estimation variance at the cost of more work. Defaults to
    /// `64`.
    pub num_perm: usize,
    /// Minimum `MinHash`-estimated `Jaccard` similarity, in `[0.0, 1.0]`,
    /// for a document to be treated as a near-duplicate carrier during
    /// scope resolution — and the same bar an audit-flagged surviving
    /// document's similarity must clear before the escalation loop will
    /// auto-delete it (see
    /// [`crate::knowledge_unlearning::UnlearningEngine::unlearn`]). Defaults
    /// to `0.3`.
    pub similarity_threshold: f64,
    /// Maximum number of delete → audit rounds the escalation loop will
    /// run before giving up and reporting
    /// [`UnlearningStatus::NotConverged`]. Must be at least `1`. Defaults to
    /// `5`.
    pub max_audit_rounds: usize,
    /// Minimum combined leakage score, in `[0.0, 1.0]`, for a surviving
    /// document to be reported as [`UnlearningLeakageVerdict::ResidualLeakage`]
    /// evidence at all. Defaults to `0.1`.
    pub leakage_threshold: f64,
    /// Word n-gram size used when extracting probe terms from deleted
    /// content for the post-deletion audit. Defaults to `3`.
    pub probe_ngram_size: usize,
    /// Maximum number of distinct probe terms extracted per audit round.
    /// Defaults to `6`.
    pub probe_top_n: usize,
    /// Weight applied to the verbatim probe-term hit ratio in the combined
    /// per-document leakage score. Defaults to `0.6`.
    pub verbatim_weight: f64,
    /// Weight applied to the shingle/`MinHash` similarity in the combined
    /// per-document leakage score. Defaults to `0.4`.
    pub similarity_weight: f64,
}

impl Default for UnlearningConfig {
    fn default() -> Self {
        Self {
            shingle_size: 2,
            num_perm: 64,
            similarity_threshold: 0.3,
            max_audit_rounds: 5,
            leakage_threshold: 0.1,
            probe_ngram_size: 3,
            probe_top_n: 6,
            verbatim_weight: 0.6,
            similarity_weight: 0.4,
        }
    }
}

impl UnlearningConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the shingle size (clamped to a minimum of `1`).
    #[must_use]
    pub fn with_shingle_size(mut self, shingle_size: usize) -> Self {
        self.shingle_size = shingle_size.max(1);
        self
    }

    /// Set the `MinHash` signature length (clamped to a minimum of `1`).
    #[must_use]
    pub fn with_num_perm(mut self, num_perm: usize) -> Self {
        self.num_perm = num_perm.max(1);
        self
    }

    /// Set the near-duplicate / escalation-delete similarity threshold.
    #[must_use]
    pub fn with_similarity_threshold(mut self, similarity_threshold: f64) -> Self {
        self.similarity_threshold = similarity_threshold;
        self
    }

    /// Set the maximum number of escalation-loop audit rounds.
    #[must_use]
    pub fn with_max_audit_rounds(mut self, max_audit_rounds: usize) -> Self {
        self.max_audit_rounds = max_audit_rounds;
        self
    }

    /// Set the minimum leakage score reported as residual leakage.
    #[must_use]
    pub fn with_leakage_threshold(mut self, leakage_threshold: f64) -> Self {
        self.leakage_threshold = leakage_threshold;
        self
    }

    /// Set the audit probe n-gram size.
    #[must_use]
    pub fn with_probe_ngram_size(mut self, probe_ngram_size: usize) -> Self {
        self.probe_ngram_size = probe_ngram_size;
        self
    }

    /// Set the maximum number of probe terms extracted per audit round.
    #[must_use]
    pub fn with_probe_top_n(mut self, probe_top_n: usize) -> Self {
        self.probe_top_n = probe_top_n;
        self
    }

    /// Set the verbatim-hit weight in the combined leakage score.
    #[must_use]
    pub fn with_verbatim_weight(mut self, verbatim_weight: f64) -> Self {
        self.verbatim_weight = verbatim_weight;
        self
    }

    /// Set the similarity weight in the combined leakage score.
    #[must_use]
    pub fn with_similarity_weight(mut self, similarity_weight: f64) -> Self {
        self.similarity_weight = similarity_weight;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`UnlearningError::InvalidConfig`] when `max_audit_rounds`
    /// is `0`, when `similarity_threshold` or `leakage_threshold` fall
    /// outside `[0.0, 1.0]`, or when any weight is negative or non-finite.
    pub fn validate(&self) -> Result<(), UnlearningError> {
        if self.max_audit_rounds == 0 {
            return Err(UnlearningError::InvalidConfig(
                "max_audit_rounds must be at least 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(UnlearningError::InvalidConfig(format!(
                "similarity_threshold must be in [0.0, 1.0], got {}",
                self.similarity_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.leakage_threshold) {
            return Err(UnlearningError::InvalidConfig(format!(
                "leakage_threshold must be in [0.0, 1.0], got {}",
                self.leakage_threshold
            )));
        }
        if !self.verbatim_weight.is_finite() || self.verbatim_weight < 0.0 {
            return Err(UnlearningError::InvalidConfig(format!(
                "verbatim_weight must be a non-negative finite value, got {}",
                self.verbatim_weight
            )));
        }
        if !self.similarity_weight.is_finite() || self.similarity_weight < 0.0 {
            return Err(UnlearningError::InvalidConfig(format!(
                "similarity_weight must be a non-negative finite value, got {}",
                self.similarity_weight
            )));
        }
        Ok(())
    }
}

// ── UnlearningError ───────────────────────────────────────────────────────────

/// Errors produced by the `knowledge_unlearning` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum UnlearningError {
    /// The requested target does not (and, as far as this engine's history
    /// records, never did) resolve to anything in the store.
    #[error("unlearning target not found: {0}")]
    TargetNotFound(String),
    /// The supplied configuration is invalid, with a human-readable
    /// explanation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Convenience alias for this module's fallible return type.
pub type UnlearningResult<T> = Result<T, UnlearningError>;

// ── UnlearningStatus ──────────────────────────────────────────────────────────

/// The final outcome of an [`UnlearningEngine::unlearn`](super::engine::UnlearningEngine::unlearn)
/// run, as recorded on its [`UnlearningCertificate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnlearningStatus {
    /// Deletion succeeded and the post-deletion audit found no residual
    /// leakage: the escalation loop converged to
    /// [`UnlearningLeakageVerdict::Clean`].
    Clean,
    /// This exact target had already been fully, cleanly unlearned by a
    /// previous request to the same engine, and nothing new was found to
    /// delete — a verified no-op, not a fresh success.
    AlreadyUnlearned,
    /// The escalation loop exhausted [`UnlearningConfig::max_audit_rounds`]
    /// (or could not make further progress, e.g. because remaining
    /// surviving carriers fell below the auto-delete similarity bar)
    /// while residual leakage was still detected. **This is a failure
    /// state and must never be reported as success.**
    NotConverged,
}

impl UnlearningStatus {
    /// `true` for [`Self::Clean`] or [`Self::AlreadyUnlearned`] — i.e. the
    /// content is verified unreachable. `false` for
    /// [`Self::NotConverged`].
    #[must_use]
    pub fn is_clean(self) -> bool {
        matches!(self, Self::Clean | Self::AlreadyUnlearned)
    }
}

// ── UnlearningAuditEvidence / UnlearningLeakageVerdict ───────────────────────

/// A single surviving document's leakage evidence from one audit round.
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningAuditEvidence {
    /// The surviving document's id.
    pub document_id: DocumentId,
    /// Probe terms (drawn from the deleted content) found verbatim
    /// (case-insensitively) inside this document.
    pub matched_terms: Vec<String>,
    /// The `MinHash`-estimated `Jaccard` similarity between this document
    /// and the deleted content, in `[0.0, 1.0]`.
    pub similarity: f64,
    /// The combined leakage score for this document:
    /// `verbatim_weight * (matched_terms.len() / probe_count) + similarity_weight * similarity`.
    pub leakage_score: f64,
}

/// The result of probing a store for residual leakage of deleted content.
#[derive(Debug, Clone, PartialEq)]
pub enum UnlearningLeakageVerdict {
    /// No surviving document scored at or above
    /// [`UnlearningConfig::leakage_threshold`]. The deleted content is
    /// verified unreachable, at least as far as this audit's probes can
    /// tell.
    Clean,
    /// At least one surviving document still carries the deleted content,
    /// verbatim or near-verbatim.
    ResidualLeakage {
        /// Every surviving document that scored at or above the leakage
        /// threshold, sorted by descending `leakage_score` (ties broken by
        /// ascending document id).
        evidence: Vec<UnlearningAuditEvidence>,
        /// The ids of the documents in `evidence`, in the same order —
        /// provided as a convenience for callers that just need the
        /// escalation target list.
        surviving_doc_ids: Vec<DocumentId>,
        /// The highest `leakage_score` among `evidence` — the audit's
        /// headline severity figure.
        score: f64,
    },
}

impl UnlearningLeakageVerdict {
    /// `true` for [`Self::Clean`].
    #[must_use]
    pub fn is_clean(&self) -> bool {
        matches!(self, Self::Clean)
    }
}

// ── UnlearningAuditRound ──────────────────────────────────────────────────────

/// One round of the post-deletion escalation loop: the verdict produced,
/// and (if any) which surviving documents were escalated for deletion as a
/// result.
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningAuditRound {
    /// The 1-based round number.
    pub round: usize,
    /// The leakage verdict this round produced.
    pub verdict: UnlearningLeakageVerdict,
    /// Document ids deleted as a direct result of this round's verdict
    /// (empty when the verdict was [`UnlearningLeakageVerdict::Clean`], or
    /// when every flagged document fell below the auto-delete similarity
    /// bar).
    pub escalated_document_ids: Vec<DocumentId>,
}

// ── UnlearningCertificate ─────────────────────────────────────────────────────

/// The auditable record of one [`UnlearningEngine::unlearn`](super::engine::UnlearningEngine::unlearn)
/// run: the original request, the resolved scope, everything actually
/// deleted, every audit round with its verdict, the final status, and a
/// deterministic content hash of the whole record.
///
/// A certificate is only as trustworthy as its honesty about `status`: a
/// run that did not converge is recorded as
/// [`UnlearningStatus::NotConverged`], never silently upgraded to
/// [`UnlearningStatus::Clean`].
#[derive(Debug, Clone, PartialEq)]
pub struct UnlearningCertificate {
    /// The request this certificate was issued for.
    pub request: UnlearningRequest,
    /// The scope resolved for this request (may be empty for an idempotent
    /// re-request — see [`UnlearningStatus::AlreadyUnlearned`]).
    pub scope: UnlearningScope,
    /// Every document id actually deleted from the store over the course of
    /// this run, in deletion order (initial scope first, then any
    /// escalated carriers).
    pub deleted_document_ids: Vec<DocumentId>,
    /// Every audit round performed, in order.
    pub audit_rounds: Vec<UnlearningAuditRound>,
    /// The final, honest outcome.
    pub status: UnlearningStatus,
    /// A deterministic `FNV-1a` hash of this certificate's content (every
    /// field above `content_hash` itself, and excluding the wall-clock
    /// `issued_at` timestamp so the hash is reproducible for identical
    /// inputs regardless of when the run happened).
    pub content_hash: u64,
    /// When this certificate was issued.
    pub issued_at: DateTime<Utc>,
}

impl UnlearningCertificate {
    /// Construct a certificate, computing `content_hash` from the other
    /// fields.
    #[must_use]
    pub(crate) fn new(
        request: UnlearningRequest,
        scope: UnlearningScope,
        deleted_document_ids: Vec<DocumentId>,
        audit_rounds: Vec<UnlearningAuditRound>,
        status: UnlearningStatus,
    ) -> Self {
        let issued_at = Utc::now();
        let content_hash = Self::compute_content_hash(
            &request,
            &scope,
            &deleted_document_ids,
            &audit_rounds,
            status,
        );
        Self {
            request,
            scope,
            deleted_document_ids,
            audit_rounds,
            status,
            content_hash,
            issued_at,
        }
    }

    /// Deterministically hash the record's content fields with `FNV-1a`.
    ///
    /// This exists so any two independently-run certificates for identical
    /// inputs are byte-for-byte comparable via a single `u64`, without
    /// requiring a full structural `PartialEq` walk.
    fn compute_content_hash(
        request: &UnlearningRequest,
        scope: &UnlearningScope,
        deleted_document_ids: &[DocumentId],
        audit_rounds: &[UnlearningAuditRound],
        status: UnlearningStatus,
    ) -> u64 {
        use std::fmt::Write as _;

        let mut buf = String::new();
        buf.push_str("target=");
        buf.push_str(&request.target.history_key());
        buf.push_str(";reason=");
        buf.push_str(request.reason.as_deref().unwrap_or(""));

        buf.push_str(";scope=");
        for item in &scope.items {
            buf.push('[');
            buf.push_str(item.document_id.as_str());
            buf.push(':');
            let _ = write!(buf, "{:?}", item.reason);
            buf.push(']');
        }

        buf.push_str(";deleted=");
        for id in deleted_document_ids {
            buf.push('[');
            buf.push_str(id.as_str());
            buf.push(']');
        }

        buf.push_str(";rounds=");
        for round in audit_rounds {
            buf.push('[');
            let _ = write!(buf, "{}", round.round);
            buf.push(':');
            match &round.verdict {
                UnlearningLeakageVerdict::Clean => buf.push_str("clean"),
                UnlearningLeakageVerdict::ResidualLeakage {
                    surviving_doc_ids,
                    score,
                    ..
                } => {
                    buf.push_str("leak(");
                    let _ = write!(buf, "{score:.6}");
                    buf.push(')');
                    for id in surviving_doc_ids {
                        buf.push(',');
                        buf.push_str(id.as_str());
                    }
                }
            }
            buf.push_str(";esc=");
            for id in &round.escalated_document_ids {
                buf.push(',');
                buf.push_str(id.as_str());
            }
            buf.push(']');
        }

        buf.push_str(";status=");
        let _ = write!(buf, "{status:?}");

        fnv1a(buf.as_bytes())
    }

    /// `true` when [`Self::status`] is verified clean (`Clean` or
    /// `AlreadyUnlearned`).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.status.is_clean()
    }
}
