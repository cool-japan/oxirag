//! Document chunking pipeline for `OxiRAG`.
//!
//! This module provides multiple strategies for splitting large documents into
//! overlapping fragments ("chunks") before they are indexed in the Echo layer.
//! Smaller, more focused chunks improve retrieval precision and keep embedding
//! context windows well within model limits.
//!
//! # Quick Start
//!
//! ```rust
//! # #[cfg(feature = "chunking")]
//! # {
//! use oxirag::chunking::{ChunkConfig, DocumentChunker};
//! use oxirag::types::Document;
//!
//! let config = ChunkConfig::default()
//!     .with_chunk_size(512)
//!     .with_chunk_overlap(64);
//!
//! let chunker = DocumentChunker::with_recursive(config);
//!
//! let doc = Document::new(
//!     "Large document text that needs splitting into overlapping windows…",
//! );
//!
//! // Returns Vec<Document> ready for echo-layer indexing.
//! let chunks = chunker.chunk_document(&doc);
//! println!("Produced {} chunks", chunks.len());
//! # }
//! ```
//!
//! # Strategies
//!
//! | Type | Use when |
//! |---|---|
//! | [`FixedSizeChunker`] | Speed is paramount and linguistic boundaries don't matter |
//! | [`SentenceChunker`] | Text has clear sentence structure (prose, articles) |
//! | [`RecursiveChunker`] | General-purpose; handles mixed content well |
//! | [`MarkdownChunker`] | Source is Markdown documentation or notes |
//!
//! # Features
//!
//! This module is gated behind the `chunking` Cargo feature and has no
//! external dependencies beyond the Rust standard library.

pub mod chunk;
pub mod chunker;
pub mod config;
pub mod strategies;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use chunk::Chunk;
pub use chunker::DocumentChunker;
pub use config::ChunkConfig;
pub use strategies::{
    ChunkStrategy, FixedSizeChunker, MarkdownChunker, RecursiveChunker, SentenceChunker,
};
