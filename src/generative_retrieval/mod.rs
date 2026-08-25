//! Generative retrieval (DSI-style, Tay et al., 2022).
//!
//! Treats retrieval as id *generation*. Every document is assigned a
//! hierarchical semantic id by recursively clustering the corpus with a
//! deterministic KMeans-lite over lexical pseudo-embeddings; the path of cluster
//! indices from the root of the centroid tree down to a document's leaf becomes
//! its [`SemanticDocId`] (rendered like `"2-5-1"`).
//!
//! At query time the [`GenerativeRetriever`] "generates" ids by **constrained
//! traversal**: starting at the root it beam-searches downward, keeping at each
//! level the `beam` clusters whose centroid best matches the query, until it
//! reaches leaf documents. The reached leaves are then ranked by full lexical
//! cosine similarity.
//!
//! Everything is pure Rust and fully deterministic — identical corpora produce
//! identical ids across runs, with no randomness or ML dependencies.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`GenerativeRetriever`] | Builds semantic ids and generates them on search |
//! | [`SemanticDocId`] | Hierarchical cluster-index path identifying a document |
//! | [`GenRetrievalConfig`] | Branching, depth, beam width, and embedding dimension |
//! | [`GenHit`] | A scored document with its semantic id |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "generative-retrieval")] {
//! use oxirag::prelude::*;
//!
//! let mut retriever = GenerativeRetriever::new(GenRetrievalConfig::default());
//! retriever.build(&documents).unwrap();
//! let hits = retriever.search("vector search", 5).unwrap();
//! # }
//! ```

pub mod retriever;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod tree;
pub mod types;

pub use retriever::GenerativeRetriever;
pub use types::{
    GenHit, GenRetrievalConfig, GenRetrievalError, SemanticDocId, cosine, embed, tokenize,
};
