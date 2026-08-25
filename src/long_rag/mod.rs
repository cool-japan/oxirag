//! `LongRAG`: retrieval over long units (Jiang et al., 2024).
//!
//! Conventional RAG fragments a corpus into many short chunks, which scatters a
//! single answer across multiple passages and hurts recall. `LongRAG` instead
//! groups related chunks and documents into a small number of **long retrieval
//! units** — each up to a token budget — so the retriever ranks fewer, longer,
//! more self-contained units.
//!
//! This is deliberately **distinct** from the `parent_document` module:
//! parent-document retrieval matches a *small* child chunk and then expands
//! outward to its parent, whereas `LongRAG` builds the long units up-front and
//! retrieves over them directly.
//!
//! # Grouping strategies
//!
//! | Strategy | Behaviour |
//! |----------|-----------|
//! | [`GroupingStrategy::ByDocument`] | One unit per document, capped at the budget |
//! | [`GroupingStrategy::FixedTokenWindow`] | Concatenate, then slice into fixed windows |
//! | [`GroupingStrategy::BySemanticAdjacency`] | Merge similar consecutive documents |
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`LongUnitGrouper`] | Bundles documents into long units |
//! | [`LongRagRetriever`] | Builds units, then ranks them against a query |
//! | [`LongUnit`] | One long unit + its sources + embedding |
//! | [`LongHit`] | A scored long unit |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "long-rag")] {
//! use oxirag::long_rag::{LongRagConfig, LongRagRetriever};
//! use oxirag::types::Document;
//!
//! let mut retriever = LongRagRetriever::new(LongRagConfig::default());
//! retriever
//!     .build(&[
//!         Document::new("Rust is a systems programming language."),
//!         Document::new("It guarantees memory safety without a garbage collector."),
//!     ])
//!     .unwrap();
//! let hits = retriever.search("memory safety", 5).unwrap();
//! # }
//! ```

pub mod grouper;
pub mod retriever;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use grouper::{LongUnitGrouper, embed, token_count, tokenize};
pub use retriever::LongRagRetriever;
pub use types::{GroupingStrategy, LongHit, LongRagConfig, LongRagError, LongUnit};
