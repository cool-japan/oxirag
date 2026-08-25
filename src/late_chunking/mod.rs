//! Late chunking for document-context-aware chunk embeddings (Jina AI 2024).
//!
//! Naive chunking embeds each chunk in isolation, so a chunk cannot "see" the
//! rest of the document it came from. Late chunking flips the order: it embeds
//! the *whole* document into token-level contextual vectors first, then pools
//! the token vectors belonging to each chunk's span. Every chunk embedding thus
//! carries whole-document context — the "late" pooling step happens after
//! contextualisation.
//!
//! This is a pure-Rust simulation built on deterministic FNV-1a pseudo-embeddings
//! (no external models, no randomness): identical input always produces identical
//! output.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`LateChunker`] | Encodes documents/queries and tiles token spans |
//! | [`LateChunkConfig`] | Dimensionality, window/overlap, context weight, pooling |
//! | [`LateChunk`] | One chunk: text, token span, late-pooled embedding |
//! | [`LatePooling`] | Mean or Max span pooling |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "late-chunking")] {
//! use oxirag::prelude::*;
//!
//! let chunker = LateChunker::new(LateChunkConfig::default());
//! let chunks = chunker.encode_document("the whole document text goes here").unwrap();
//! let query = chunker.encode_query("a question");
//! let score = LateChunker::cosine(&chunks[0].embedding, &query);
//! # }
//! ```

pub mod chunker;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use chunker::LateChunker;
pub use types::{LateChunk, LateChunkConfig, LateChunkError, LatePooling};
