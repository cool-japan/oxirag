//! Index-time document expansion via hypothetical query prediction (`Doc2Query` / `HyPE`).
//!
//! Where [`query_expansion`](crate::query_expansion) expands the *user query*,
//! this module expands the *document*: it predicts hypothetical questions a
//! passage answers and appends them to the passage before indexing. Following
//! Nogueira et al. ("Document Expansion by Query Prediction"), this enriches the
//! indexed text with terms a searcher is likely to use, boosting both lexical
//! and semantic recall.
//!
//! # Example
//!
//! ```
//! use oxirag::doc2query::{Doc2QueryConfig, Doc2QueryExpander};
//! use oxirag::types::Document;
//!
//! let doc = Document::new(
//!     "Photosynthesis converts sunlight into chemical energy in plants.",
//! );
//! let expander = Doc2QueryExpander::new(Doc2QueryConfig::default());
//! let expanded = expander.expand(&doc).expect("non-empty document");
//! assert!(expanded.expanded_content.contains(&doc.content));
//! assert!(!expanded.generated_queries.is_empty());
//! ```
pub mod expander;
pub mod generator;
#[cfg(test)]
mod tests;
pub mod types;

pub use expander::Doc2QueryExpander;
pub use generator::HeuristicQueryGenerator;
pub use types::{Doc2QueryConfig, Doc2QueryError, ExpandedDocument, QueryGenerator};
