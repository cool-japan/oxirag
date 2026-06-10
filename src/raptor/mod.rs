//! Recursive abstractive processing for tree-organised retrieval (RAPTOR).
//!
//! Builds a tree of summaries from a flat list of text chunks using lexical
//! pseudo-embeddings and greedy agglomerative or k-means-lite clustering.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`RaptorBuilder`] | Builds the tree from text chunks |
//! | [`RaptorTree`] | Full tree + collapsed retrieval |
//! | [`RaptorNode`] | Individual tree node (leaf or summary) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "raptor")] {
//! use oxirag::prelude::*;
//!
//! let builder = RaptorBuilder::new();
//! let tree = builder.build(&texts, &RaptorConfig::default()).unwrap();
//! # }
//! ```

pub mod cluster;
#[cfg(test)]
mod tests;
pub mod tree;
pub mod types;

pub use tree::RaptorBuilder;
pub use types::{ClusterStrategy, RaptorConfig, RaptorError, RaptorNode, RaptorTree};
