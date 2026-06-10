//! Multi-hop entity-chain traversal — finds entities matching the query,
//! then follows [`GraphRelationship`] edges to discover connected context.
//!
//! The module is gated behind the `multi-hop` Cargo feature which implies
//! `graphrag`.  The entry point is [`MultiHopRetriever::run`], which executes
//! the hop loop and returns a [`MultiHopResult`] containing the synthesised
//! answer and a per-hop [`HopState`] trace.
//!
//! [`GraphRelationship`]: crate::layer4_graph::types::GraphRelationship

pub mod traversal;
pub mod types;

#[cfg(test)]
mod tests;

pub use traversal::{MultiHopRetriever, detect_entity_mentions, expand_hop};
pub use types::{EntityMention, HopConfig, HopState, MultiHopError, MultiHopResult};
