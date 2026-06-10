//! Heuristic SVO triple extraction from plain text.
//!
//! This module provides a pure-Rust, zero-dependency fact extractor that
//! tokenises sentences, detects verb pivots, and produces
//! subject–predicate–object [`Triple`]s with confidence scores.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::fact_triple::{TripleExtractor, TripleConfig};
//!
//! let extractor = TripleExtractor::new(TripleConfig::default());
//! let triples = extractor.extract("Rust uses LLVM as its backend.", None);
//! for t in &triples {
//!     println!("{}", t); // "Rust USES LLVM"
//! }
//! ```

pub mod extractor;
pub mod types;

#[cfg(test)]
mod tests;

pub use extractor::{COMMON_VERBS, detect_verb, extract_from_sentence, split_sentences};
pub use types::{Triple, TripleConfig, TripleError, TripleExtractor, TripleStore};
