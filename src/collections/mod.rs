//! Multi-tenant, namespaced vector-store collections.
//!
//! This module provides a Chroma/Qdrant-style collection API for `OxiRAG`:
//! isolated vector stores, each with its own embedding dimension, similarity
//! metric, and capacity limit, all addressable by a normalised string name.
//!
//! # Architecture
//!
//! ```text
//! CollectionIndex<S>
//!     │
//!     ├─ CollectionStore (S) ─── collection registry (metadata, config, stats)
//!     │
//!     └─ per-collection InMemoryVectorStore ─── actual embeddings + documents
//! ```
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "collections")]
//! # {
//! use oxirag::collections::{
//!     CollectionConfig, CollectionId, CollectionIndex, CollectionMetadata,
//!     InMemoryCollectionStore,
//! };
//! use oxirag::types::Document;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), oxirag::collections::CollectionError> {
//! let store = InMemoryCollectionStore::new();
//! let index = CollectionIndex::new(store);
//!
//! let id = CollectionId::new("my collection")?;   // normalised to "my_collection"
//! let config = CollectionConfig::new(64);
//!
//! index.create_collection(id.clone(), config, CollectionMetadata::default()).await?;
//!
//! let doc = Document::new("Hello, world!");
//! let embedding = vec![0.0_f32; 64];
//! index.index_document(&id, doc, embedding).await?;
//!
//! let results = index.search(&id, &vec![0.0_f32; 64], 5).await?;
//! println!("{} results", results.len());
//! # Ok(())
//! # }
//! # }
//! ```

pub mod index;
pub mod store;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use index::CollectionIndex;
pub use store::{CollectionStore, InMemoryCollectionStore};
pub use types::{
    Collection, CollectionConfig, CollectionError, CollectionId, CollectionMetadata,
    CollectionStats, FederatedResult, SimilarityMetric as CollectionSimilarityMetric,
};
