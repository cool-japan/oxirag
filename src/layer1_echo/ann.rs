//! Approximate Nearest Neighbor (ANN) index using HNSW algorithm.
//!
//! This module provides a Hierarchical Navigable Small World (HNSW) graph
//! implementation for fast approximate nearest neighbor search.

pub mod config;
pub mod hnsw;
pub mod stats;
pub mod store;

pub use config::AnnConfig;
pub use hnsw::{HnswIndex, HnswNode};
pub use stats::AnnStats;
pub use store::AnnVectorStore;

#[cfg(test)]
mod tests;
