//! Knowledge-graph question answering via subgraph extraction and fact synthesis.
//!
//! The [`KgqaEngine`] takes a natural-language query plus a slice of
//! [`GraphEntity`] and [`GraphRelationship`] records, expands a subgraph via
//! BFS, and synthesises a [`KgqaAnswer`] that includes evidence entities and
//! [`Triple`]s extracted with the `fact-triples` machinery.
//!
//! This module is gated behind the `knowledge-graph-qa` Cargo feature, which
//! implies both `graphrag` and `fact-triples`.
//!
//! [`GraphEntity`]: crate::layer4_graph::types::GraphEntity
//! [`GraphRelationship`]: crate::layer4_graph::types::GraphRelationship
//! [`Triple`]: crate::fact_triple::Triple

pub mod engine;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::KgqaEngine;
pub use types::{KgqaAnswer, KgqaConfig, KgqaError};
