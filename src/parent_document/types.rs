//! Types for the `parent_document` module.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{Document, DocumentId};

// ── ParentChildIndex ──────────────────────────────────────────────────────────

/// Maps child chunk ids to their parent document id, and stores parent documents.
#[derive(Debug, Clone, Default)]
pub struct ParentChildIndex {
    /// `child_id` → `parent_id`
    pub child_to_parent: HashMap<String, DocumentId>,
    /// `parent_id` → parent [`Document`]
    pub parents: HashMap<String, Document>,
}

impl ParentChildIndex {
    /// Create a new empty [`ParentChildIndex`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a parent document and all of its child chunks.
    pub fn register(&mut self, parent: Document, children: &[Document]) {
        let parent_id = parent.id.as_str().to_string();
        for child in children {
            self.child_to_parent
                .insert(child.id.as_str().to_string(), parent.id.clone());
        }
        self.parents.insert(parent_id, parent);
    }

    /// Look up the parent document for a given child id.
    #[must_use]
    pub fn parent_of(&self, child_id: &str) -> Option<&Document> {
        let parent_id = self.child_to_parent.get(child_id)?;
        self.parents.get(parent_id.as_str())
    }

    /// Return the number of registered parent documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Return `true` when no parents are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }
}

// ── ChunkHierarchy ────────────────────────────────────────────────────────────

/// Helper for building parent-child document pairs.
pub struct ChunkHierarchy;

impl ChunkHierarchy {
    /// Chunk `parent` using the given `chunker` and inject `parent_id` metadata
    /// into every child chunk so the index can reverse-lookup.
    ///
    /// Returns the list of child chunks (each carries `"parent_id"` in metadata).
    #[must_use]
    #[cfg(feature = "chunking")]
    pub fn build(parent: &Document, chunker: &crate::chunking::DocumentChunker) -> Vec<Document> {
        chunker
            .chunk_document(parent)
            .into_iter()
            .map(|mut child| {
                child
                    .metadata
                    .insert("parent_id".to_string(), parent.id.as_str().to_string());
                child
            })
            .collect()
    }
}

// ── ExpandedResult ────────────────────────────────────────────────────────────

/// A retrieval result expanded from a child match to its parent context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedResult {
    /// The parent document (full context).
    pub parent: Document,
    /// The child chunk ids that matched the query.
    pub matched_children: Vec<DocumentId>,
    /// The highest child match score.
    pub score: f32,
}

impl ExpandedResult {
    /// Return the number of matching child chunks.
    #[must_use]
    pub fn child_hit_count(&self) -> usize {
        self.matched_children.len()
    }
}

// ── ParentDocumentConfig ──────────────────────────────────────────────────────

/// Configuration for the parent-document retriever.
#[derive(Debug, Clone)]
pub struct ParentDocumentConfig {
    /// Number of child chunks to retrieve per search.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
    /// Whether to expand child hits to the full parent document.
    ///
    /// Defaults to `true`.
    pub return_parent: bool,
}

impl Default for ParentDocumentConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            return_parent: true,
        }
    }
}

impl ParentDocumentConfig {
    /// Set the retrieval top-k.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }

    /// Set whether to return parent documents.
    #[must_use]
    pub fn with_return_parent(mut self, v: bool) -> Self {
        self.return_parent = v;
        self
    }
}

// ── WindowConfig ──────────────────────────────────────────────────────────────

/// Configuration for sentence-window expansion (currently unused in the
/// heuristic implementation; reserved for future sentence-window retrieval).
#[derive(Debug, Clone, Default)]
pub struct WindowConfig {
    /// Number of neighbour chunks to include around each matched chunk.
    ///
    /// Defaults to `1`.
    pub window_size: usize,
}

impl WindowConfig {
    /// Set the window size.
    #[must_use]
    pub fn with_window_size(mut self, v: usize) -> Self {
        self.window_size = v;
        self
    }
}

// ── ParentDocumentError ───────────────────────────────────────────────────────

/// Errors from the `parent_document` module.
#[derive(Debug, Error)]
pub enum ParentDocumentError {
    /// The query was empty after trimming.
    #[error("Query must not be empty")]
    EmptyQuery,

    /// A child chunk referred to a parent that is not in the index.
    #[error("Parent not found for child '{child_id}'")]
    ParentNotFound {
        /// The child chunk id whose parent was missing.
        child_id: String,
    },

    /// The retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
}
