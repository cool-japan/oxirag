//! Sub-question decomposition → sub-answers → RRF recombination.
//!
//! Decomposes a complex query into independent sub-questions, retrieves
//! supporting documents for each, and recombines results via Reciprocal
//! Rank Fusion.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`QueryDecomposer`] | Heuristic clause/conjunction splitter |
//! | [`QueryDecompositionEngine`] | Orchestrates decompose → retrieve → fuse |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "query-decomposition")] {
//! use oxirag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let engine = QueryDecompositionEngine::new(DecompositionConfig::default());
//! # }
//! # }
//! ```

pub mod decomposer;
pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use decomposer::QueryDecomposer;
pub use engine::QueryDecompositionEngine;
pub use types::{
    DecomposedQuery, DecompositionConfig, DecompositionStrategy, QueryDecompositionError,
    SubAnswer, SubQuestion,
};
