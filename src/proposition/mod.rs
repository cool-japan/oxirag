//! Proposition / Dense-X retrieval (Chen et al., 2023).
//!
//! Decomposes passages into atomic, self-contained **propositions** — each a
//! standalone factual statement — indexes every proposition against its parent
//! document, and retrieves parent documents via their best-matching proposition.
//! This yields fine-grained retrieval that is more precise than passage-level
//! matching.
//!
//! Propositions are natural-language clauses; they are *distinct* from
//! subject-verb-object graph triples.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`HeuristicPropositionExtractor`] | Rule-based decomposition into propositions |
//! | [`PropositionIndex`] | Embeds, stores, and retrieves propositions |
//! | [`Proposition`] | A single atomic statement + its parent + embedding |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "proposition-retrieval")] {
//! use oxirag::prelude::*;
//!
//! let mut index = PropositionIndex::new(PropositionConfig::default());
//! index.add_document(&Document::new("Rust is fast and it improves safety."));
//! let docs = index.search_documents("safety", 5).unwrap();
//! # }
//! ```

pub mod extractor;
pub mod index;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use extractor::{HeuristicPropositionExtractor, PropositionExtractor};
pub use index::PropositionIndex;
pub use types::{Proposition, PropositionConfig, PropositionError, PropositionHit};
