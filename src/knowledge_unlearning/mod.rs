//! GDPR / right-to-be-forgotten unlearning for a RAG knowledge base:
//! **delete → re-audit for residual leakage → escalate if still reachable →
//! produce an auditable record.**
//!
//! # Why `delete()` is not enough
//!
//! Removing a document from an index does not guarantee its content is
//! gone. The same facts routinely survive in:
//!
//! - near-duplicate or overlapping chunks of the same source (a different
//!   document id that paraphrases or re-chunks the deleted text),
//! - derived artifacts built *from* the deleted document (summaries, cached
//!   answers, extracted entities/triples),
//! - other, unrelated documents that happen to quote or closely paraphrase
//!   the deleted content.
//!
//! A single `delete(id)` call only ever removes the one document it was
//! given. This module composes deletion with **verification**: after
//! deleting, it probes the surviving corpus with terms drawn from the
//! deleted content and reports honestly whether that content is still
//! retrievable — escalating to delete newly-found carriers and re-probing,
//! up to a configured bound, rather than assuming success.
//!
//! # How this differs from the primitives it's built from
//!
//! The pieces already exist elsewhere in this crate, but nothing composes
//! them into a verified deletion workflow:
//!
//! | Module | What it does | What it does *not* do |
//! |---|---|---|
//! | [`crate::membership_inference`] | Injects canaries, probes a RAG system, and computes a genuine rank-based (Mann-Whitney U) AUC leakage score; `MembershipDefense` mitigates by verbatim-span redaction. | Never deletes a source document from any index, and never re-verifies a deletion. It is an audit-and-redact tool, not a deletion workflow. |
//! | [`crate::layer1_echo`] | Exposes per-document `delete(&DocumentId)` across its vector-store backends. | Plain CRUD — no near-duplicate awareness, no derived-artifact cascade, no leakage verification. |
//! | [`crate::collections`] | `CollectionStore::delete(id)` removes an entire named collection. | Coarse, multi-tenant-level deletion — not single-document scope resolution, and no post-deletion audit either. |
//! | `knowledge_unlearning` (this module) | Resolves a request into a full deletion **scope** (exact match + near-duplicates + derived artifacts), deletes it, **re-audits the surviving corpus** with an [`UnlearningAuditor`], and escalates until clean or honestly reports it could not converge. | Does not implement its own vector index or retrieval pipeline — [`UnlearnableStore`] is a narrow trait a caller adapts any real store to. |
//!
//! # Mechanism
//!
//! 1. **Scope resolution** ([`UnlearningScopeResolver`]). An
//!    [`UnlearningRequest`] names an [`UnlearningTarget`] — a document id, a
//!    source id, or a content pattern. Resolution expands that seed into a
//!    full [`UnlearningScope`]: the named document(s), every other document
//!    in the store whose shingle/`MinHash` similarity clears
//!    [`UnlearningConfig::similarity_threshold`] (a real near-duplicate
//!    detector — see [`UnlearningNearDuplicateDetector`], not a hand-wave),
//!    and every derived artifact cascaded through an
//!    [`UnlearningArtifactRegistry`] the caller maintains.
//! 2. **Deletion**. Every document in the resolved scope is deleted from
//!    the store, behind the caller's own [`UnlearnableStore`] implementation
//!    ([`UnlearningMemoryStore`] is provided for tests and simple
//!    deployments).
//! 3. **Post-deletion audit** ([`UnlearningAuditor`]). Distinctive n-grams
//!    are extracted from the deleted content and used to probe every
//!    surviving document for verbatim or near-verbatim traces, producing an
//!    [`UnlearningLeakageVerdict`]: [`UnlearningLeakageVerdict::Clean`] or
//!    [`UnlearningLeakageVerdict::ResidualLeakage`] with the surviving
//!    carriers named.
//! 4. **Escalation**. On residual leakage, carriers whose similarity to the
//!    deleted content clears the *same* bar scope resolution used are
//!    deleted and the audit repeats, up to
//!    [`UnlearningConfig::max_audit_rounds`]. A carrier below that bar is
//!    reported but never auto-deleted (auto-deleting content the configured
//!    threshold does not consider a duplicate would itself be a data-loss
//!    risk) — this run instead honestly stops as
//!    [`UnlearningStatus::NotConverged`].
//! 5. **Certification** ([`UnlearningCertificate`]). The request, resolved
//!    scope, every document actually deleted, every audit round with its
//!    verdict, the final [`UnlearningStatus`], and a deterministic `FNV-1a`
//!    content hash are bundled into the record a compliance officer would
//!    keep. **A certificate for a run that did not converge says so** —
//!    [`UnlearningEngine`] never upgrades [`UnlearningStatus::NotConverged`]
//!    to [`UnlearningStatus::Clean`].
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "knowledge-unlearning")]
//! # {
//! use oxirag::knowledge_unlearning::{
//!     UnlearnableStore, UnlearningConfig, UnlearningEngine, UnlearningMemoryStore,
//!     UnlearningRequest, UnlearningTarget,
//! };
//! use oxirag::types::Document;
//!
//! let mut store = UnlearningMemoryStore::new();
//! let doc = Document::new("Jane Doe's account balance is $4,821.19 as of March.");
//! let doc_id = doc.id.clone();
//! store.insert(doc);
//!
//! let mut engine =
//!     UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
//! let certificate = engine
//!     .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(doc_id.clone())))
//!     .expect("target exists");
//!
//! assert!(certificate.is_clean());
//! assert!(engine.store().get(&doc_id).is_none());
//! # }
//! ```

pub mod audit;
pub mod dedup;
pub mod engine;
pub mod scope;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use audit::UnlearningAuditor;
pub use dedup::UnlearningNearDuplicateDetector;
pub use engine::{UnlearnableStore, UnlearningEngine, UnlearningMemoryStore};
pub use scope::{UnlearningArtifactLink, UnlearningArtifactRegistry, UnlearningScopeResolver};
pub use types::{
    UnlearningArtifactKind, UnlearningAuditEvidence, UnlearningAuditRound, UnlearningCertificate,
    UnlearningConfig, UnlearningError, UnlearningLeakageVerdict, UnlearningRequest,
    UnlearningResult, UnlearningScope, UnlearningScopeItem, UnlearningScopeReason,
    UnlearningStatus, UnlearningTarget,
};
