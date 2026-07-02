//! The COIL contextualised inverted index.
//!
//! [`CoilInvertedIndex`] is the data structure at the heart of COIL (Gao, Dai,
//! Callan 2021): a mapping from each **surface token string** to the list of
//! [`CoilPosting`]s — one per occurrence — that carry that token's
//! contextualised vector in some document. Document-level CLS vectors are held
//! in a separate map so COIL-full scoring can add a semantic term.
//!
//! This inverted layout is what gives COIL its *exact-lexical gating*: a query
//! token only ever probes the postings list of its own identical surface
//! string, so it can contribute only to documents that literally contain that
//! token.

use std::collections::HashMap;

use crate::coil_retrieval::types::{CoilDocument, CoilPosting};
use crate::types::DocumentId;

/// A COIL contextualised inverted index.
///
/// Maps each surface token to a postings list keyed by that exact string, and
/// separately records every document's CLS vector. Insertion order of the
/// documents is preserved so that scoring and iteration are deterministic.
#[derive(Debug, Clone, Default)]
pub struct CoilInvertedIndex {
    /// Surface token → postings (one posting per occurrence, across all
    /// documents).
    postings: HashMap<String, Vec<CoilPosting>>,
    /// Document id → its L2-normalised CLS vector.
    cls_vectors: HashMap<DocumentId, Vec<f32>>,
    /// Document ids in first-insertion order (deduplicated).
    doc_order: Vec<DocumentId>,
}

impl CoilInvertedIndex {
    /// Create a new, empty inverted index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a fully encoded document into the index.
    ///
    /// Every token vector becomes a [`CoilPosting`] appended to its surface
    /// token's postings list, and the document's CLS vector is recorded. The
    /// document id is appended to the insertion order the first time it is
    /// seen. Re-inserting the same id overwrites its CLS vector but *appends*
    /// fresh postings, so callers that rebuild an index should
    /// [`clear`](Self::clear) it first.
    pub fn insert_document(&mut self, document: &CoilDocument) {
        if !self.cls_vectors.contains_key(&document.doc_id) {
            self.doc_order.push(document.doc_id.clone());
        }
        self.cls_vectors
            .insert(document.doc_id.clone(), document.cls_vector.clone());
        for token_vector in &document.token_vectors {
            self.postings
                .entry(token_vector.surface.clone())
                .or_default()
                .push(CoilPosting::new(
                    document.doc_id.clone(),
                    token_vector.clone(),
                ));
        }
    }

    /// Return the postings for an exact surface token, or an empty slice when
    /// the token never occurred in the corpus.
    #[must_use]
    pub fn postings_for(&self, surface: &str) -> &[CoilPosting] {
        self.postings.get(surface).map_or(&[], Vec::as_slice)
    }

    /// Return the CLS vector recorded for a document, if present.
    #[must_use]
    pub fn cls_for(&self, doc_id: &DocumentId) -> Option<&[f32]> {
        self.cls_vectors.get(doc_id).map(Vec::as_slice)
    }

    /// Return the document ids in first-insertion order.
    #[must_use]
    pub fn doc_ids(&self) -> &[DocumentId] {
        &self.doc_order
    }

    /// Return `true` when the corpus contains at least one occurrence of the
    /// exact surface token.
    #[must_use]
    pub fn contains_token(&self, surface: &str) -> bool {
        self.postings.contains_key(surface)
    }

    /// Number of documents held by the index.
    #[must_use]
    pub fn num_documents(&self) -> usize {
        self.doc_order.len()
    }

    /// Number of distinct surface tokens (postings lists) in the index.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    /// Total number of postings (token occurrences) across every list.
    #[must_use]
    pub fn num_postings(&self) -> usize {
        self.postings.values().map(Vec::len).sum()
    }

    /// Return `true` when the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_order.is_empty()
    }

    /// Remove every posting, CLS vector, and document id from the index.
    pub fn clear(&mut self) {
        self.postings.clear();
        self.cls_vectors.clear();
        self.doc_order.clear();
    }
}
