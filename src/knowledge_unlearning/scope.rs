//! Scope resolution: expanding an [`UnlearningTarget`] into the full
//! [`UnlearningScope`] of documents that must be removed for the request to
//! actually be effective — the named document, its near-duplicates, and
//! everything derived from either.

use std::collections::HashSet;

use crate::types::{Document, DocumentId};

use super::dedup::UnlearningNearDuplicateDetector;
use super::engine::UnlearnableStore;
use super::types::{
    UnlearningArtifactKind, UnlearningConfig, UnlearningScope, UnlearningScopeItem,
    UnlearningScopeReason, UnlearningTarget,
};

// ── UnlearningArtifactLink / UnlearningArtifactRegistry ─────────────────────

/// One `source → derived artifact` relationship.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnlearningArtifactLink {
    /// The id of the source document the artifact was derived from.
    pub source_id: DocumentId,
    /// The id of the derived artifact document itself (e.g. the document
    /// holding a cached summary's text).
    pub artifact_id: DocumentId,
    /// What kind of derived artifact this is.
    pub kind: UnlearningArtifactKind,
}

/// A registry of `source → derived artifact` relationships (summaries,
/// cached answers, extracted entities, ...), used by
/// [`UnlearningScopeResolver`] to cascade deletion beyond a source document
/// to everything built from it.
///
/// Nothing in [`crate::knowledge_unlearning`] populates this registry
/// automatically — a caller wires it up as documents (and their derived
/// artifacts) are ingested, mirroring how a real deployment's
/// summarization/caching/extraction pipelines would need to record their
/// own provenance for this module to be able to cascade deletions through
/// them.
#[derive(Debug, Clone, Default)]
pub struct UnlearningArtifactRegistry {
    links: Vec<UnlearningArtifactLink>,
}

impl UnlearningArtifactRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `artifact_id` (of kind `kind`) was derived from
    /// `source_id`.
    pub fn register(
        &mut self,
        source_id: DocumentId,
        artifact_id: DocumentId,
        kind: UnlearningArtifactKind,
    ) {
        self.links.push(UnlearningArtifactLink {
            source_id,
            artifact_id,
            kind,
        });
    }

    /// Every artifact link recorded with `source_id` as its source.
    #[must_use]
    pub fn derived_from(&self, source_id: &DocumentId) -> Vec<&UnlearningArtifactLink> {
        self.links
            .iter()
            .filter(|link| &link.source_id == source_id)
            .collect()
    }

    /// The number of recorded links.
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// `true` when no links have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
}

/// Add `document_id` to `items`/`in_scope` unless it is already present.
fn add_item(
    items: &mut Vec<UnlearningScopeItem>,
    in_scope: &mut HashSet<DocumentId>,
    document_id: DocumentId,
    reason: UnlearningScopeReason,
) {
    if in_scope.insert(document_id.clone()) {
        items.push(UnlearningScopeItem {
            document_id,
            reason,
        });
    }
}

// ── UnlearningScopeResolver ──────────────────────────────────────────────────

/// Expands an [`UnlearningTarget`] into a full [`UnlearningScope`] by
/// composing three passes over the current store contents:
///
/// 1. **Primary match** — documents the target names directly (by id,
///    source, or content pattern).
/// 2. **Near-duplicate scan** — every other document in the store whose
///    shingle/`MinHash` similarity to a primary match meets
///    [`UnlearningConfig::similarity_threshold`].
/// 3. **Derived-artifact cascade** — a fixed-point expansion over the
///    supplied [`UnlearningArtifactRegistry`]: every artifact derived from
///    a document already in scope is added, and *its* artifacts are
///    considered in turn, until nothing new is found.
///
/// Resolution never mutates the store — it only reads
/// [`UnlearnableStore::iter_documents`] and reports what it finds. Deletion
/// is a separate, explicit step performed by the caller (typically
/// [`crate::knowledge_unlearning::UnlearningEngine`]).
#[derive(Debug, Clone)]
pub struct UnlearningScopeResolver {
    /// The configuration this resolver scans with.
    pub config: UnlearningConfig,
}

impl UnlearningScopeResolver {
    /// Create a new resolver with the given configuration.
    #[must_use]
    pub fn new(config: UnlearningConfig) -> Self {
        Self { config }
    }

    /// Resolve `target` against `store` and `registry` into a full
    /// [`UnlearningScope`].
    ///
    /// Returns an empty scope when nothing in the current store matches
    /// `target` at all — this is not an error at the resolver level; it is
    /// the caller's job (see
    /// [`UnlearningEngine::unlearn`](super::engine::UnlearningEngine::unlearn))
    /// to decide whether an empty scope means "already unlearned" or
    /// "target never existed".
    #[must_use]
    pub fn resolve<S: UnlearnableStore>(
        &self,
        store: &S,
        registry: &UnlearningArtifactRegistry,
        target: &UnlearningTarget,
    ) -> UnlearningScope {
        let documents = store.iter_documents();

        let mut items: Vec<UnlearningScopeItem> = Vec::new();
        let mut in_scope: HashSet<DocumentId> = HashSet::new();

        let primary = Self::resolve_primary_matches(&documents, target, &mut items, &mut in_scope);
        self.scan_near_duplicates(&documents, &primary, &mut items, &mut in_scope);
        Self::cascade_derived_artifacts(registry, &mut items, &mut in_scope);

        UnlearningScope { items }
    }

    /// Pass 1: find every document `target` names directly, adding each to
    /// `items`/`in_scope` and returning `(id, content)` pairs for use as
    /// near-duplicate comparison seeds.
    fn resolve_primary_matches(
        documents: &[Document],
        target: &UnlearningTarget,
        items: &mut Vec<UnlearningScopeItem>,
        in_scope: &mut HashSet<DocumentId>,
    ) -> Vec<(DocumentId, String)> {
        let mut primary: Vec<(DocumentId, String)> = Vec::new();

        match target {
            UnlearningTarget::DocumentId(id) => {
                if let Some(doc) = documents.iter().find(|d| &d.id == id) {
                    add_item(
                        items,
                        in_scope,
                        doc.id.clone(),
                        UnlearningScopeReason::ExactMatch,
                    );
                    primary.push((doc.id.clone(), doc.content.clone()));
                }
            }
            UnlearningTarget::SourceId(source) => {
                for doc in documents {
                    if doc.source.as_deref() == Some(source.as_str()) {
                        add_item(
                            items,
                            in_scope,
                            doc.id.clone(),
                            UnlearningScopeReason::SourceMatch,
                        );
                        primary.push((doc.id.clone(), doc.content.clone()));
                    }
                }
            }
            UnlearningTarget::ContentPattern(pattern) => {
                let needle = pattern.to_lowercase();
                if !needle.is_empty() {
                    for doc in documents {
                        if doc.content.to_lowercase().contains(&needle) {
                            add_item(
                                items,
                                in_scope,
                                doc.id.clone(),
                                UnlearningScopeReason::ContentMatch,
                            );
                            primary.push((doc.id.clone(), doc.content.clone()));
                        }
                    }
                }
            }
        }

        primary
    }

    /// Pass 2: scan every document not already in scope against every
    /// `primary` seed, adding it as a [`UnlearningScopeReason::NearDuplicate`]
    /// when its similarity clears [`UnlearningConfig::similarity_threshold`].
    fn scan_near_duplicates(
        &self,
        documents: &[Document],
        primary: &[(DocumentId, String)],
        items: &mut Vec<UnlearningScopeItem>,
        in_scope: &mut HashSet<DocumentId>,
    ) {
        if primary.is_empty() {
            return;
        }
        let detector =
            UnlearningNearDuplicateDetector::new(self.config.shingle_size, self.config.num_perm);
        for (source_id, source_content) in primary {
            for doc in documents {
                if in_scope.contains(&doc.id) {
                    continue;
                }
                let similarity = detector.similarity(source_content, &doc.content);
                if similarity >= self.config.similarity_threshold {
                    add_item(
                        items,
                        in_scope,
                        doc.id.clone(),
                        UnlearningScopeReason::NearDuplicate {
                            of: source_id.clone(),
                            similarity,
                        },
                    );
                }
            }
        }
    }

    /// Pass 3: fixed-point expansion over `registry` — every artifact
    /// derived from a document already in scope is added, and its own
    /// artifacts are considered in turn.
    fn cascade_derived_artifacts(
        registry: &UnlearningArtifactRegistry,
        items: &mut Vec<UnlearningScopeItem>,
        in_scope: &mut HashSet<DocumentId>,
    ) {
        let mut frontier: Vec<DocumentId> = items.iter().map(|i| i.document_id.clone()).collect();
        while let Some(source_id) = frontier.pop() {
            for link in registry.derived_from(&source_id) {
                if in_scope.contains(&link.artifact_id) {
                    continue;
                }
                add_item(
                    items,
                    in_scope,
                    link.artifact_id.clone(),
                    UnlearningScopeReason::DerivedArtifact {
                        of: source_id.clone(),
                        kind: link.kind.clone(),
                    },
                );
                frontier.push(link.artifact_id.clone());
            }
        }
    }
}
