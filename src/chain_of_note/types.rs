//! Types for the `chain_of_note` module.

use thiserror::Error;

// ── NoteConfig ────────────────────────────────────────────────────────────────

/// Configuration for `ChainOfNoteEngine`.
#[derive(Debug, Clone)]
pub struct NoteConfig {
    /// Maximum number of relevant sentences to extract per document.
    ///
    /// Defaults to `3`.
    pub max_notes_per_doc: usize,
    /// Maximum character length of each extracted note.
    ///
    /// Defaults to `150`.
    pub max_note_length: usize,
    /// Minimum Jaccard relevance score for a note to be included in the chain.
    ///
    /// Defaults to `0.1`.
    pub relevance_threshold: f32,
    /// Maximum number of documents to process.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
}

impl Default for NoteConfig {
    fn default() -> Self {
        Self {
            max_notes_per_doc: 3,
            max_note_length: 150,
            relevance_threshold: 0.1,
            top_k: 5,
        }
    }
}

impl NoteConfig {
    /// Set the maximum number of sentences per document.
    #[must_use]
    pub fn with_max_notes_per_doc(mut self, max_notes_per_doc: usize) -> Self {
        self.max_notes_per_doc = max_notes_per_doc;
        self
    }

    /// Set the maximum note length in characters.
    #[must_use]
    pub fn with_max_note_length(mut self, max_note_length: usize) -> Self {
        self.max_note_length = max_note_length;
        self
    }

    /// Set the minimum relevance threshold for note inclusion.
    #[must_use]
    pub fn with_relevance_threshold(mut self, relevance_threshold: f32) -> Self {
        self.relevance_threshold = relevance_threshold;
        self
    }

    /// Set the maximum number of documents to process.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

// ── DocumentNote ──────────────────────────────────────────────────────────────

/// An extractive note derived from a single document.
#[derive(Debug, Clone)]
pub struct DocumentNote {
    /// The document identifier.
    pub doc_id: String,
    /// The extracted note text.
    pub note: String,
    /// Jaccard relevance score of this note relative to the query.
    pub relevance_score: f32,
}

impl DocumentNote {
    /// Returns `true` if the note's relevance score meets or exceeds `threshold`.
    #[must_use]
    pub fn is_relevant(&self, threshold: f32) -> bool {
        self.relevance_score >= threshold
    }
}

// ── NoteChain ─────────────────────────────────────────────────────────────────

/// A chain of per-document notes and their synthesised answer.
#[derive(Debug, Clone)]
pub struct NoteChain {
    /// Per-document notes, sorted by descending relevance.
    pub notes: Vec<DocumentNote>,
    /// The synthesised final answer combining all relevant notes.
    pub synthesized: String,
    /// The original query these notes were generated for.
    pub query: String,
}

impl NoteChain {
    /// Returns `true` when no notes are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }

    /// Returns the number of notes whose relevance score meets `threshold`.
    #[must_use]
    pub fn relevant_count(&self, threshold: f32) -> usize {
        self.notes
            .iter()
            .filter(|n| n.is_relevant(threshold))
            .count()
    }
}

// ── ChainOfNoteError ──────────────────────────────────────────────────────────

/// Errors that can occur in `ChainOfNoteEngine`.
#[derive(Debug, Error)]
pub enum ChainOfNoteError {
    /// The query was empty or contained only whitespace.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// No documents were provided for note extraction.
    #[error("No documents provided")]
    EmptyDocuments,
}
