//! Integrated document processing pipeline.
//!
//! This module wires `OxiRAG`'s chunking, semantic caching, and advanced
//! retrieval components into two cooperating pipeline objects:
//!
//! - [`IndexingPipeline`] — takes raw `Document`s, chunks them with a
//!   configurable strategy, optionally deduplicates repeated content, indexes
//!   every surviving chunk into an [`Echo`] store, and records per-chunk
//!   provenance.
//!
//! - [`RetrievalPipeline`] — searches the same [`Echo`] store, optionally
//!   reranks results with [`MmrReranker`], looks up provenance to annotate each
//!   result with its source document UUID and chunk position, and caches query
//!   results in a [`InMemorySemanticCache`].
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "document-pipeline")]
//! # {
//! use oxirag::document_pipeline::{DocumentPipelineBuilder, IndexingConfig};
//! use oxirag::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
//! use oxirag::types::Document;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let (indexing, mut retrieval) = DocumentPipelineBuilder::new()
//!     .with_indexing_config(IndexingConfig::default())
//!     .with_mmr(0.5)
//!     .with_semantic_cache(0.9, 100)
//!     .build();
//!
//! let mut echo = EchoLayer::new(
//!     MockEmbeddingProvider::new(64),
//!     InMemoryVectorStore::new(64),
//! );
//!
//! let doc = Document::new("Long document text to be chunked and indexed.");
//! indexing.index_document(&mut echo, doc).await.unwrap();
//!
//! let results = retrieval.search(&mut echo, "document text", 5, None).await.unwrap();
//! println!("{} results", results.len());
//! # }
//! # }
//! ```
//!
//! [`Echo`]: crate::layer1_echo::Echo
//! [`MmrReranker`]: crate::advanced_retrieval::MmrReranker
//! [`InMemorySemanticCache`]: crate::semantic_cache::InMemorySemanticCache

pub mod indexing;
pub mod retrieval;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use indexing::IndexingPipeline;
pub use retrieval::{DocumentPipelineBuilder, RetrievalPipeline};
pub use types::{
    ChunkProvenance, ChunkStrategyKind, DocumentAwareResult, DocumentPipelineError, IndexingConfig,
    IndexingResult, PipelineStats,
};
