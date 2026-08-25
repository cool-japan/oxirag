//! `HippoRAG` — neurobiologically-inspired single-step multi-hop retrieval
//! (Gutiérrez et al., 2024).
//!
//! `HippoRAG` draws on the hippocampal indexing theory of human long-term memory:
//! the neocortex stores rich representations, while a sparse hippocampal *index*
//! of associations lets the brain retrieve memories connected only indirectly.
//! Here the analogue is an **entity graph**. Each passage contributes the
//! entities mentioned within it; entities that co-occur in the same passage are
//! linked, with the edge weight rising as they share more passages. The result
//! is an associative memory over the corpus.
//!
//! Retrieval is a **single step**. The query's entities seed a Personalized
//! `PageRank` (PPR) random walk over the entity graph, spreading relevance across
//! co-occurrence edges; each passage is then scored by the total PPR mass of the
//! entities it contains. A passage connected to the query only through an
//! intermediate shared entity — a genuine multi-hop path — receives non-zero
//! mass and is retrieved, all without an explicit hop loop. This is what
//! distinguishes `HippoRAG` from the breadth-first `multi_hop` traversal, which
//! follows entity-chain edges one discrete hop at a time.
//!
//! The implementation is pure Rust and fully deterministic: entity extraction
//! is rule-based (capitalised multi-character tokens) and the PPR walk is the
//! standard restart-biased power method.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`extract_entities`] | Rule-based entity surface-form extraction |
//! | [`personalized_pagerank`] | Restart-biased PPR power iteration |
//! | [`HippoRagIndex`] | Entity graph construction and single-step search |
//! | [`HippoConfig`] | Damping, iteration count, hashing dimension |
//! | [`HippoHit`] | A passage scored by accumulated PPR mass |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "hipporag")] {
//! use oxirag::hippo_rag::{HippoConfig, HippoRagIndex};
//! use oxirag::types::Document;
//!
//! let mut index = HippoRagIndex::new(HippoConfig::default());
//! index
//!     .build(&[
//!         Document::new("Alice collaborates with Bob at Acme."),
//!         Document::new("Bob mentored Carol at Cambridge."),
//!     ])
//!     .unwrap();
//! let hits = index.search("Alice", 5).unwrap();
//! # }
//! ```

pub mod graph;
pub mod index;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use graph::{extract_entities, personalized_pagerank};
pub use index::HippoRagIndex;
pub use types::{HippoConfig, HippoHit, HippoRagError};
