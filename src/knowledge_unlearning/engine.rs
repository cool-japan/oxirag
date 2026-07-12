//! [`UnlearnableStore`] (the storage abstraction this module targets),
//! [`UnlearningMemoryStore`] (an in-memory reference implementation), and
//! [`UnlearningEngine`] (the orchestrator tying scope resolution, deletion,
//! and the post-deletion escalation loop together into an auditable run).

use std::collections::HashMap;

use crate::types::{Document, DocumentId};

use super::audit::UnlearningAuditor;
use super::scope::{UnlearningArtifactRegistry, UnlearningScopeResolver};
use super::types::{UnlearningAuditRound, UnlearningTarget};
use super::types::{
    UnlearningCertificate, UnlearningConfig, UnlearningError, UnlearningLeakageVerdict,
    UnlearningRequest, UnlearningResult, UnlearningScope, UnlearningScopeReason, UnlearningStatus,
};

// ── UnlearnableStore ──────────────────────────────────────────────────────────

/// The minimal storage abstraction [`UnlearningEngine`] targets.
///
/// Deliberately independent of [`crate::layer1_echo::VectorStore`] and any
/// other feature-gated store in this crate: `knowledge_unlearning` must be
/// usable (and testable) with only its own feature flag enabled. Implement
/// this trait as a thin adapter over whatever real store a deployment
/// actually uses.
pub trait UnlearnableStore {
    /// Look up a document by id, if present.
    fn get(&self, id: &DocumentId) -> Option<Document>;

    /// A snapshot of every document currently in the store, in unspecified
    /// order.
    fn iter_documents(&self) -> Vec<Document>;

    /// Remove the document with `id`, if present. Returns `true` when a
    /// document was actually removed, `false` when `id` was not present.
    fn delete(&mut self, id: &DocumentId) -> bool;
}

// ── UnlearningMemoryStore ─────────────────────────────────────────────────────

/// A plain in-memory [`UnlearnableStore`], for tests and for callers who do
/// not need a persistent backend.
#[derive(Debug, Clone, Default)]
pub struct UnlearningMemoryStore {
    documents: HashMap<DocumentId, Document>,
}

impl UnlearningMemoryStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) `document`, keyed by its id. Returns the
    /// previous document at that id, if any.
    pub fn insert(&mut self, document: Document) -> Option<Document> {
        self.documents.insert(document.id.clone(), document)
    }

    /// The number of documents currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// `true` when the store holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// `true` when `id` is currently present.
    #[must_use]
    pub fn contains(&self, id: &DocumentId) -> bool {
        self.documents.contains_key(id)
    }
}

impl UnlearnableStore for UnlearningMemoryStore {
    fn get(&self, id: &DocumentId) -> Option<Document> {
        self.documents.get(id).cloned()
    }

    fn iter_documents(&self) -> Vec<Document> {
        self.documents.values().cloned().collect()
    }

    fn delete(&mut self, id: &DocumentId) -> bool {
        self.documents.remove(id).is_some()
    }
}

// ── UnlearningEngine ──────────────────────────────────────────────────────────

/// Orchestrates a full, verified unlearning run: resolve scope, delete,
/// audit for residual leakage, escalate and repeat if necessary, and issue
/// an [`UnlearningCertificate`] that honestly reflects what actually
/// happened.
///
/// # The escalation loop
///
/// After the initially-resolved [`UnlearningScope`] is deleted,
/// [`Self::unlearn`] repeatedly probes the store with an
/// [`UnlearningAuditor`] built from the same [`UnlearningConfig`]. If a
/// round's [`UnlearningLeakageVerdict`] reports surviving carriers, each
/// carrier whose similarity to the deleted content meets
/// [`UnlearningConfig::similarity_threshold`] — the *same* bar scope
/// resolution used to decide what counts as a duplicate in the first place
/// — is deleted and the loop repeats. A carrier whose similarity falls
/// *below* that bar is reported but never auto-deleted: auto-deleting
/// content the configured similarity bar does not consider a duplicate
/// would risk destroying unrelated data, so the run instead honestly stops
/// and reports [`UnlearningStatus::NotConverged`].
///
/// The loop also stops, with the same honest [`UnlearningStatus::NotConverged`],
/// after [`UnlearningConfig::max_audit_rounds`] rounds if leakage is still
/// present. **This engine never reports [`UnlearningStatus::Clean`] unless
/// an audit round actually returned [`UnlearningLeakageVerdict::Clean`].**
#[derive(Debug, Clone)]
pub struct UnlearningEngine<S: UnlearnableStore> {
    /// The configuration this engine resolves scope, deletes, and audits
    /// with.
    pub config: UnlearningConfig,
    store: S,
    registry: UnlearningArtifactRegistry,
    /// Final status of every target this engine has previously completed a
    /// run for, keyed by [`UnlearningTarget::history_key`] — used to
    /// recognize a repeat request for an already-handled target.
    history: HashMap<String, UnlearningStatus>,
}

impl<S: UnlearnableStore> UnlearningEngine<S> {
    /// Create a new engine wrapping `store`.
    ///
    /// # Errors
    ///
    /// Returns [`UnlearningError::InvalidConfig`] when `config` fails
    /// [`UnlearningConfig::validate`].
    pub fn new(config: UnlearningConfig, store: S) -> UnlearningResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            store,
            registry: UnlearningArtifactRegistry::new(),
            history: HashMap::new(),
        })
    }

    /// Replace this engine's derived-artifact registry and return `self`
    /// for chaining.
    #[must_use]
    pub fn with_registry(mut self, registry: UnlearningArtifactRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// Shared access to the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Mutable access to the underlying store, for direct manipulation in
    /// tests (e.g. seeding a corpus) or callers who need it.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Shared access to the derived-artifact registry.
    #[must_use]
    pub fn registry(&self) -> &UnlearningArtifactRegistry {
        &self.registry
    }

    /// Mutable access to the derived-artifact registry.
    pub fn registry_mut(&mut self) -> &mut UnlearningArtifactRegistry {
        &mut self.registry
    }

    /// The final status this engine previously recorded for `target`, if it
    /// has ever completed a run for it.
    #[must_use]
    pub fn history_status(&self, target: &UnlearningTarget) -> Option<UnlearningStatus> {
        self.history.get(&target.history_key()).copied()
    }

    /// Resolve `target` into a full [`UnlearningScope`] against the current
    /// store contents, without deleting or auditing anything.
    #[must_use]
    pub fn resolve_scope(&self, target: &UnlearningTarget) -> UnlearningScope {
        UnlearningScopeResolver::new(self.config.clone()).resolve(
            &self.store,
            &self.registry,
            target,
        )
    }

    /// Run a full, verified unlearning request: resolve scope, delete,
    /// audit, escalate as needed, and issue a certificate.
    ///
    /// # Errors
    ///
    /// Returns [`UnlearningError::TargetNotFound`] when `request.target`
    /// resolves to nothing in the current store *and* this engine has no
    /// record of ever having handled that exact target before. A repeat
    /// request for a target this engine previously finished handling is
    /// **not** an error — see [`UnlearningStatus::AlreadyUnlearned`].
    pub fn unlearn(
        &mut self,
        request: UnlearningRequest,
    ) -> UnlearningResult<UnlearningCertificate> {
        let target_key = request.target.history_key();
        let scope = self.resolve_scope(&request.target);

        if scope.is_empty() {
            return match self.history.get(&target_key).copied() {
                Some(previous_status) => {
                    let status = if previous_status.is_clean() {
                        UnlearningStatus::AlreadyUnlearned
                    } else {
                        // Honestly propagate: we never resolved the earlier
                        // leakage, and there is nothing left in the current
                        // store to re-probe with, so we cannot claim
                        // progress was made.
                        UnlearningStatus::NotConverged
                    };
                    self.history.insert(target_key, status);
                    Ok(UnlearningCertificate::new(
                        request,
                        scope,
                        Vec::new(),
                        Vec::new(),
                        status,
                    ))
                }
                None => Err(UnlearningError::TargetNotFound(target_key)),
            };
        }

        // Capture original content and delete every resolved scope item
        // before auditing, so the audit measures the *post-deletion* store.
        let mut deleted_document_ids: Vec<DocumentId> = Vec::new();
        let mut probe_source = String::new();
        for item in &scope.items {
            if let Some(doc) = self.store.get(&item.document_id) {
                if !probe_source.is_empty() {
                    probe_source.push(' ');
                }
                probe_source.push_str(&doc.content);
            }
            self.store.delete(&item.document_id);
            deleted_document_ids.push(item.document_id.clone());
        }

        let auditor = UnlearningAuditor::new(self.config.clone());
        let mut audit_rounds: Vec<UnlearningAuditRound> = Vec::new();
        let mut status = UnlearningStatus::NotConverged;

        for round in 1..=self.config.max_audit_rounds {
            let verdict = auditor.audit(&self.store, &probe_source);
            let verdict_for_round = verdict.clone();

            match verdict {
                UnlearningLeakageVerdict::Clean => {
                    audit_rounds.push(UnlearningAuditRound {
                        round,
                        verdict: verdict_for_round,
                        escalated_document_ids: Vec::new(),
                    });
                    status = UnlearningStatus::Clean;
                    break;
                }
                UnlearningLeakageVerdict::ResidualLeakage { evidence, .. } => {
                    let mut escalated: Vec<DocumentId> = Vec::new();
                    for item in &evidence {
                        if item.similarity >= self.config.similarity_threshold {
                            self.store.delete(&item.document_id);
                            deleted_document_ids.push(item.document_id.clone());
                            escalated.push(item.document_id.clone());
                        }
                    }
                    let made_progress = !escalated.is_empty();
                    audit_rounds.push(UnlearningAuditRound {
                        round,
                        verdict: verdict_for_round,
                        escalated_document_ids: escalated,
                    });
                    status = UnlearningStatus::NotConverged;
                    if !made_progress {
                        // Every remaining carrier fell below the
                        // auto-delete similarity bar: another round would
                        // find exactly the same, unresolved leakage. Stop
                        // honestly instead of spinning.
                        break;
                    }
                }
            }
        }

        // Enrich the scope report with anything escalated beyond the
        // initial resolution, so the certificate's scope reflects every
        // document actually removed, not just the first pass.
        let mut full_scope = scope;
        for round in &audit_rounds {
            for id in &round.escalated_document_ids {
                if !full_scope.contains(id) {
                    full_scope.items.push(super::types::UnlearningScopeItem {
                        document_id: id.clone(),
                        reason: UnlearningScopeReason::Escalated {
                            round: round.round,
                            score: match &round.verdict {
                                UnlearningLeakageVerdict::ResidualLeakage { score, .. } => *score,
                                UnlearningLeakageVerdict::Clean => 0.0,
                            },
                        },
                    });
                }
            }
        }

        self.history.insert(target_key, status);

        Ok(UnlearningCertificate::new(
            request,
            full_scope,
            deleted_document_ids,
            audit_rounds,
            status,
        ))
    }
}
