//! Chain-of-Note: per-document extractive notes synthesized into a final answer.
//!
//! This module implements the Chain-of-Note pattern for RAG: each retrieved
//! document is summarised into an extractive note via sentence-level Jaccard
//! scoring against the original query.  Notes are then chained together into
//! a synthesised answer, preserving the document provenance at each step.

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::ChainOfNoteEngine;
pub use types::{ChainOfNoteError, DocumentNote, NoteChain, NoteConfig};
