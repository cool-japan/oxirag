//! COIL — Contextualized Inverted List (Gao, Dai, Callan 2021, "COIL:
//! Revisit Exact Lexical Match in Information Retrieval with Contextualized
//! Inverted List").
//!
//! COIL bridges classic lexical retrieval and neural late interaction. A
//! document is encoded into one **contextualised token vector** per position:
//! a token's own deterministic hash-embedding blended with a small window of
//! its neighbours, so the *same* surface token receives a *different* vector in
//! a different context. Those vectors are stored in a
//! [`CoilInvertedIndex`] keyed by the **surface token string**, and each
//! document also contributes a single document-level CLS vector.
//!
//! At query time each query token is contextualised the same way, then — and
//! this is the crux — it only ever looks up the postings list of its *own
//! identical surface string*. A query token's contribution to a document is the
//! **maximum** dot product over that document's occurrences of the *same*
//! surface token. Contributions are summed over the query tokens (COIL-tok);
//! COIL-full additionally adds a `lambda`-weighted document-level CLS
//! similarity.
//!
//! # How this differs from its neighbours
//!
//! - [`plaid_retrieval`](crate::plaid_retrieval) (PLAID / `ColBERTv2`) performs
//!   *soft, all-to-all* late interaction: it computes `MaxSim` between **every**
//!   query token and **every** document token regardless of surface form. COIL
//!   instead *gates on exact surface-token equality* through the inverted list,
//!   so a query token can only ever match an identical token — cheaper, and
//!   faithful to lexical precision.
//! - [`sparse_retrieval`](crate::sparse_retrieval) (SPLADE) learns a scalar
//!   weight per (expanded) term and scores a bag-of-terms dot product; it has
//!   **no per-occurrence contextual vector**. COIL keeps a full contextualised
//!   vector for each token occurrence, so it distinguishes the *same* term used
//!   in different senses (polysemy) rather than collapsing it to one weight.
//!
//! Every embedding is deterministic (FNV-1a hashing, no randomness, no
//! machine-learning dependencies), so identical inputs always produce identical
//! scores.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`CoilConfig`] | Dimension, context window, blend weight, CLS `lambda`, mode |
//! | [`CoilScoreMode`] | COIL-tok versus COIL-full scoring |
//! | [`CoilTokenVector`] | One contextualised token embedding |
//! | [`CoilPosting`] | One token occurrence inside a document |
//! | [`CoilDocument`] | A fully encoded document (token vectors + CLS) |
//! | [`CoilInvertedIndex`] | Surface token → postings, plus per-document CLS |
//! | [`CoilRetriever`] | Encodes a corpus and scores queries |
//! | [`CoilError`] / [`CoilResult`] | Error type and result alias |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "coil-retrieval")] {
//! use oxirag::coil_retrieval::{CoilConfig, CoilRetriever, CoilScoreMode};
//! use oxirag::types::Document;
//!
//! let corpus = vec![
//!     Document::new("Rust delivers memory safety without a garbage collector.").with_id("rust"),
//!     Document::new("Python is a dynamic scripting language.").with_id("python"),
//! ];
//!
//! // COIL-tok: only documents sharing an exact surface token are scored.
//! let mut retriever = CoilRetriever::new(CoilConfig::new().with_score_mode(CoilScoreMode::Tok));
//! retriever.index(&corpus).unwrap();
//! let hits = retriever.search("memory safety", 5).unwrap();
//! assert_eq!(hits[0].0.as_str(), "rust");
//! # }
//! ```

pub mod engine;
pub mod index;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::CoilRetriever;
pub use index::CoilInvertedIndex;
pub use types::{
    CoilConfig, CoilDocument, CoilError, CoilPosting, CoilResult, CoilScoreMode, CoilTokenVector,
};
