//! SPLADE-style learned sparse retrieval.
//!
//! Represents text as a sparse **term → weight** map rather than a dense vector.
//! Term weights use a log-saturation transform `log(1 + max(0, tf * idf))`, so
//! repeated terms keep adding signal but with diminishing returns. Crucially the
//! [`SparseEncoder`] *expands* the term set: for every present term it appends a
//! handful of co-occurring related terms (drawn from a co-occurrence table fitted
//! on the corpus) at a discounted weight, deterministically simulating the neural
//! query/document expansion that gives [SPLADE](https://arxiv.org/abs/2107.05720)
//! its recall advantage. Scoring is a sparse dot product over shared terms.
//!
//! This is *distinct* from the BM25 lexical matching in `hybrid_search`: weights
//! are learned (IDF-saturated) and the term set is expanded beyond the literal
//! surface tokens.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`SparseVector`] | Term → weight map with dot product and top-term inspection |
//! | [`SparseConfig`] | Expansion count, saturation, and pruning threshold |
//! | [`SparseEncoder`] | Learns IDF + co-occurrence, then encodes & expands text |
//! | [`SparseIndex`] | Builds over a corpus and ranks documents by sparse dot product |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "sparse-retrieval")] {
//! use oxirag::prelude::*;
//!
//! let corpus = vec![
//!     Document::new("Rust delivers memory safety without a garbage collector."),
//!     Document::new("Python is a dynamic scripting language."),
//! ];
//! let mut index = SparseIndex::new(SparseConfig::default());
//! index.build(&corpus).unwrap();
//! let hits = index.search("memory safety", 5).unwrap();
//! # }
//! ```

pub mod encoder;
pub mod index;
#[cfg(test)]
mod tests;
pub mod types;

pub use encoder::SparseEncoder;
pub use index::SparseIndex;
pub use types::{SparseConfig, SparseHit, SparseRetrievalError, SparseVector};
