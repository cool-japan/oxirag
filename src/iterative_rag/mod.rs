//! ITER-RETGEN iterative retrieval-augmented generation.
//!
//! This module implements the iterative retrieval loop from the ITER-RETGEN paper:
//! each iteration expands the query with terms extracted from the previous draft,
//! retrieves new documents, and builds a refined draft until convergence or a
//! maximum-iteration limit is reached.

pub mod engine;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::IterativeRagEngine;
pub use types::{IterationStep, IterativeConfig, IterativeOutput, IterativeRagError};
