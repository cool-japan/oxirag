//! Auto-merging retrieval (LlamaIndex-style).
//!
//! Indexes fine-grained **leaf** chunks that each reference a coarser parent.
//! At query time the most relevant leaves are retrieved; when at least
//! [`AutoMergeConfig::merge_threshold`] of a parent's children appear in the
//! retrieved set, those children are **collapsed** into the single parent chunk
//! — recursively up the hierarchy. The merged parent replaces its children in
//! the result set, yielding a more coherent context window than scattered
//! sibling fragments.
//!
//! This is *distinct* from parent-document (small-to-big) retrieval, which
//! expands one matched leaf to its parent regardless of how many siblings were
//! retrieved. Auto-merging only promotes a parent once a threshold fraction of
//! its children are independently relevant.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ChunkNode`] / [`ChunkHierarchy`] | Parent ↔ child chunk tree |
//! | [`AutoMergeConfig`] | Window sizes, embedding dim, merge threshold |
//! | [`AutoMergingRetriever`] | Build hierarchy, retrieve + merge leaves |
//! | [`MergedHit`] | A leaf or merged-parent result |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "auto-merging")] {
//! use oxirag::prelude::*;
//!
//! let mut retriever = AutoMergingRetriever::new(AutoMergeConfig::default());
//! retriever.build(&Document::new("... long document text ...")).unwrap();
//! let hits = retriever.search("a query", 5).unwrap();
//! # }
//! ```

pub mod hierarchy;
pub mod retriever;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use hierarchy::{ChunkHierarchy, ChunkNode};
pub use retriever::AutoMergingRetriever;
pub use types::{AutoMergeConfig, AutoMergeError, MergedHit};
