//! Advanced retrieval strategies for RAG pipelines.
//!
//! This module provides three complementary retrieval algorithms that extend
//! the basic vector-similarity search provided by the [`Echo`] layer:
//!
//! | Strategy | Idea |
//! |---|---|
//! | [`RagFusion`] | Multi-query expansion + Reciprocal Rank Fusion |
//! | [`HydeRetrieval`] | Hypothetical Document Embedding retrieval |
//! | [`MmrReranker`] | Maximal Marginal Relevance reranking |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "advanced-retrieval")]
//! # {
//! use oxirag::advanced_retrieval::{RagFusion, RagFusionConfig, HydeRetrieval, HydeConfig};
//!
//! // Generate multiple query variants for RAG-Fusion.
//! let fusion = RagFusion::new(RagFusionConfig::default());
//! let variants = fusion.generate_variants("What is Rust?");
//! assert_eq!(variants[0], "What is Rust?");
//!
//! // Build a hypothetical document for HyDE.
//! let hyde = HydeRetrieval::new(HydeConfig::default());
//! let doc = hyde.generate_hypothetical_doc("How does async work?");
//! assert!(!doc.is_empty());
//! # }
//! ```
//!
//! [`Echo`]: crate::layer1_echo::traits::Echo

pub mod hyde;
pub mod mmr;
pub mod rag_fusion;
pub mod types;

#[cfg(test)]
mod tests;

pub use hyde::{HydeConfig, HydeRetrieval};
pub use mmr::{MmrConfig, MmrReranker};
pub use rag_fusion::{RagFusion, RagFusionConfig};
pub use types::{AdvancedRetrievalError, RetrievalConfig, RetrievalStrategy};
