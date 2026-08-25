//! PLAID / `ColBERTv2` late-interaction retrieval (Santhanam et al., 2022).
//!
//! Each document is represented as a *set* of per-token embeddings rather than a
//! single pooled vector. Relevance is the `MaxSim` late-interaction score
//!
//! ```text
//! score(q, d) = Σ_{t ∈ q} max_{s ∈ d} cosine(t, s)
//! ```
//!
//! Naive `MaxSim` compares every query token against every token of every
//! document. PLAID accelerates this by clustering all corpus token embeddings
//! into a small set of **centroids**. Each query token probes its `nprobe`
//! nearest centroids; documents owning a token assigned to any probed centroid
//! become candidates, and only those candidates are re-scored with full
//! `MaxSim`. This is *distinct* from the plain `ColBERT` `MaxSim` in
//! [`layer1_echo::multi_vector`](crate::layer1_echo::multi_vector), which adds no
//! centroid-based candidate generation or pruning.
//!
//! Embeddings are deterministic: every token is hashed (FNV-1a) into a dense
//! L2-normalised vector, and centroids are trained with a spread-initialised
//! k-means. No randomness, no machine-learning dependencies.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`PlaidConfig`] | Centroid count, probe width, dimension, k-means iterations |
//! | [`PlaidRetriever`] | Builds centroids, generates candidates, scores `MaxSim` |
//! | [`PlaidHit`] | A scored document result |
//! | [`PlaidError`] | Error conditions |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "plaid")] {
//! use oxirag::prelude::*;
//!
//! let mut retriever = PlaidRetriever::new(PlaidConfig::default());
//! retriever.build(&[Document::new("Rust delivers memory safety without a garbage collector.")]).unwrap();
//! let hits = retriever.search("memory safety", 5).unwrap();
//! # }
//! ```

pub mod retriever;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use retriever::PlaidRetriever;
pub use types::{PlaidConfig, PlaidError, PlaidHit};
