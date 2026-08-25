//! Embedding-based retrieval strategy routing.
//!
//! Routes queries to the most appropriate retrieval strategy using FNV-1a
//! pseudo-embeddings and cosine similarity against labelled examples.

pub mod router;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use router::SemanticRouter;
pub use types::{
    RouterError, RouterExample, RoutingDecision, RoutingTarget, SemanticRoutingConfig,
};
