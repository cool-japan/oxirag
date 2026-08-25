//! `Query2Doc`: pseudo-document query expansion (Wang et al., EMNLP 2023).
//!
//! `Query2Doc` expands a short user query into a longer, answer-shaped string by
//! (1) asking a [`PseudoDocGenerator`] to draft a *pseudo-document* — a
//! hypothetical passage that would plausibly answer the query — and then
//! (2) **concatenating** the original query, repeated
//! [`query_repetitions`](Query2DocConfig::query_repetitions) times, with that
//! pseudo-document. The repetition keeps a sparse retriever's term weighting
//! anchored on the original query terms so they are not "swamped" by the
//! (much longer) pseudo-document.
//!
//! ```text
//! sparse:  "<query> <query> <query> <query> <query> <pseudo_doc>"   (n = query_repetitions, default 5)
//! dense:   "<query> [SEP] <pseudo_doc>"                              (n = 1, fixed)
//! ```
//!
//! ## Distinct from `advanced_retrieval` (`HyDE`) and `doc2query`
//!
//! | Module | Direction | What is fed to the retriever |
//! |--------|-----------|-------------------------------|
//! | [`advanced_retrieval::hyde`](crate::advanced_retrieval::hyde) | Query-time | The hypothetical document **alone**; the original query text is not concatenated. |
//! | [`doc2query`](crate::doc2query) | **Index-time**: document → hypothetical questions appended to the passage before indexing. | The corpus is expanded, not the query. |
//! | **`query2doc`** | Query-time | The **repeated query concatenated with** the pseudo-document ([`PseudoDocument::expanded_query`]). |
//!
//! ## Example
//!
//! ```
//! use oxirag::query2doc::{
//!     MockPseudoDocGenerator, Query2DocConfig, Query2DocExpander, Query2DocVariant,
//! };
//!
//! let generator = MockPseudoDocGenerator::new().with_knowledge(
//!     "rust",
//!     "is a systems programming language focused on safety, speed, and concurrency.",
//! );
//! let expander = Query2DocExpander::new(Query2DocConfig::default(), generator);
//!
//! let expanded = expander.expand("What is Rust?").unwrap();
//! assert_eq!(expanded.query, "What is Rust?");
//! assert_eq!(
//!     expanded.pseudo_doc,
//!     "Rust is a systems programming language focused on safety, speed, and concurrency."
//! );
//! assert_eq!(
//!     expanded.expanded_query,
//!     "What is Rust? What is Rust? What is Rust? What is Rust? What is Rust? Rust is a systems programming language focused on safety, speed, and concurrency."
//! );
//!
//! // The dense variant concatenates the query only once, separated by [SEP].
//! let dense_expander = Query2DocExpander::new(
//!     Query2DocConfig::default().with_variant(Query2DocVariant::Dense),
//!     MockPseudoDocGenerator::new(),
//! );
//! let dense = dense_expander.expand("What is Rust?").unwrap();
//! assert!(dense.expanded_query.starts_with("What is Rust? [SEP] "));
//! ```

pub mod expander;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use expander::Query2DocExpander;
pub use types::{
    MockPseudoDocGenerator, PseudoDocGenerator, PseudoDocument, Query2DocConfig, Query2DocError,
    Query2DocVariant,
};
