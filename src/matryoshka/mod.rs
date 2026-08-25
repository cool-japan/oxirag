//! Matryoshka Representation Learning: nested "Russian doll" embeddings.
//!
//! Implements the core idea of Kusupati et al. (2022): an embedding whose front
//! *prefixes* are themselves valid lower-dimensional embeddings. Earlier
//! dimensions are made to carry more information (via a per-dimension importance
//! decay), so truncating a prefix still yields a usable, coarser embedding.
//!
//! This enables **coarse-to-fine retrieval**: shortlist candidates cheaply with
//! a short truncated prefix, then rerank that shortlist with the full
//! dimension. Unlike `quantization` (which reduces numeric *precision*), this
//! module reduces *dimensionality*.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`MatryoshkaConfig`] | Dimensions, shortlist sizing, decay |
//! | [`MatryoshkaEncoder`] | Text → nested [`MatryoshkaEmbedding`] |
//! | [`MatryoshkaEmbedding`] | Full vector + prefix [`MatryoshkaEmbedding::truncate`] |
//! | [`MatryoshkaRetriever`] | Two-stage coarse-to-fine [`MatryoshkaRetriever::search`] |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "matryoshka")] {
//! use oxirag::prelude::*;
//!
//! let mut retriever = MatryoshkaRetriever::try_new(MatryoshkaConfig::default()).unwrap();
//! retriever.add_text(DocumentId::from_string("a"), "rust systems programming");
//! retriever.add_text(DocumentId::from_string("b"), "python data science");
//! let hits = retriever.search("rust programming", 1).unwrap();
//! # }
//! ```

pub mod encoder;
pub mod retriever;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use encoder::{MatryoshkaEmbedding, MatryoshkaEncoder};
pub use retriever::MatryoshkaRetriever;
pub use types::{MatryoshkaConfig, MatryoshkaError, MatryoshkaHit};
