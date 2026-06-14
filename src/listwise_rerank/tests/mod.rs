#![allow(clippy::float_cmp)]
//! Tests for the `listwise_rerank` module.

mod config_judge;
mod rerank;

use crate::types::{Document, DocumentId, SearchResult};

/// Build a [`SearchResult`] with the given id, content, score, and rank.
fn make_result(id: &str, content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank,
    }
}

/// Build a bare [`Document`] from its content.
fn doc(content: &str) -> Document {
    Document::new(content)
}
