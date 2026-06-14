//! Multi-Query Retrieval with Reciprocal Rank Fusion (Langchain-style).
//!
//! This module implements **Multi-Query Retrieval**: given a single user query,
//! a [`QueryVariantGenerator`] produces `N` paraphrase / reformulation variants
//! of that query, retrieval is performed independently for each variant using a
//! BM25-lite (token-overlap) heuristic, and the resulting per-variant ranked
//! lists are fused with **Reciprocal Rank Fusion** (RRF):
//!
//! ```text
//! rrf_score(doc) = Σ_i  1 / (rrf_k + rank_i(doc))
//! ```
//!
//! where the sum ranges over every variant list that contains `doc` in its
//! top-`k` results.  The final list is sorted by descending RRF score.
//!
//! The approach follows the Multi-Query Retriever described in:
//! > Ma et al. 2023 — *Query2Doc* and the Langchain Multi-Query Retriever.
//!
//! # Design
//!
//! - **Zero new dependencies** — implemented with `std` + `thiserror` only.
//! - **No async** — all retrieval is synchronous and corpus-local.
//! - **Pluggable variant generation** via the [`QueryVariantGenerator`] trait.
//!   The bundled [`MockQueryVariantGenerator`] uses deterministic heuristic
//!   prefixes and is suitable for testing and offline evaluation.
//!
//! # Example
//!
//! ```
//! use oxirag::multi_query::{
//!     MockQueryVariantGenerator, MultiQueryConfig, MultiQueryGenerator,
//! };
//!
//! let docs = vec![
//!     ("doc1".to_string(), "Rust systems programming language memory safety".to_string()),
//!     ("doc2".to_string(), "Python scripting dynamic typing applications".to_string()),
//!     ("doc3".to_string(), "Rust ownership borrowing borrow checker".to_string()),
//! ];
//!
//! let config = MultiQueryConfig::default();
//! let generator = MultiQueryGenerator::new(config, MockQueryVariantGenerator::new());
//! let result = generator.retrieve("Rust memory safety", &docs).unwrap();
//!
//! assert_eq!(result.original_query, "Rust memory safety");
//! assert_eq!(result.generated_queries.len(), 3);
//! assert!(!result.hits.is_empty());
//! // The Rust docs should rank above the Python doc.
//! assert_ne!(result.hits[0].id, "doc2");
//! ```

pub mod generator;
pub mod types;

#[cfg(test)]
mod tests;

pub use generator::MultiQueryGenerator;
pub use types::{
    GeneratedQuery, MockQueryVariantGenerator, MultiQueryConfig, MultiQueryError, MultiQueryHit,
    MultiQueryResult, QueryVariantGenerator,
};
