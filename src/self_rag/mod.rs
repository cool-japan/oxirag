//! Self-Reflective RAG with reflection tokens.
//!
//! Stub — implementation pending.

pub mod engine;
pub mod reflect;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::SelfRagEngine;
pub use reflect::{HeuristicReflector, MockReflector, Reflector};
pub use types::{ReflectionToken, SelfRagConfig, SelfRagError, SelfRagOutput};
