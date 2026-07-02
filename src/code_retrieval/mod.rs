//! `code_retrieval` — AST/structure-aware code search over a code corpus.
//!
//! This module retrieves source code by combining a genuine **structural**
//! signal with a **textual** one. A lightweight, deterministic structural
//! tokenizer (see [`parser`]) scans each corpus item — a chunk of
//! general-purpose, function-like source text — into [`CodeRetrievalUnit`]s:
//! one per detected function/method definition (name, top-level parameter
//! count, and a rough body span found by brace-depth matching for brace
//! languages or indentation-block tracking for Python-like ones), plus a
//! whole-file fallback unit when no function is found. For each unit it records
//! the identifier set ([`CodeRetrievalSymbol`]s with occurrence counts), the
//! imported module/paths, and the heuristic call-sites (an identifier
//! immediately followed by `(`). Search then blends a **structural similarity**
//! (Jaccard overlap of identifier, import, and call-site sets) with a
//! **textual similarity** (cosine of a deterministic FNV-1a pseudo-embedding of
//! the raw text) under a configurable weight.
//!
//! # Difference from [`program_of_thought`](crate::program_of_thought)
//!
//! Both modules involve "parsing", but toward opposite ends.
//! [`program_of_thought`](crate::program_of_thought) builds an abstract syntax
//! tree for a **tiny arithmetic DSL** and then *executes* that tree in an
//! interpreter to *compute a numeric answer* — the AST exists to be evaluated,
//! and evaluation is the whole point.
//!
//! `code_retrieval`, by contrast, **never executes anything**. It parses
//! **general-purpose source code** (function-like text in a variety of
//! C-/Rust-/Python-like syntaxes) into *structural units* purely so those units
//! can be **searched and ranked** over a code corpus. There is no evaluator, no
//! environment, and no notion of a program's runtime value: the parse output is
//! a set of structural *features* (identifiers, imports, call-sites, body
//! spans) used only for retrieval scoring.
//!
//! # Example
//!
//! ```
//! use oxirag::code_retrieval::{CodeRetrievalConfig, CodeRetrievalEngine};
//!
//! let corpus = vec![
//!     (
//!         "tokenizer.rs",
//!         "fn scan(buffer) { let token = read_token(buffer); advance_cursor(buffer); }",
//!     ),
//!     ("math.rs", "fn add(a, b) { return a + b; }"),
//! ];
//!
//! let engine = CodeRetrievalEngine::new(CodeRetrievalConfig::default());
//! let index = engine.index(corpus).expect("index builds");
//!
//! // A query sharing identifiers and calls with the tokenizer function.
//! let result = engine
//!     .search(&index, "read_token(buffer) then advance_cursor(buffer)")
//!     .expect("search succeeds");
//!
//! let top = result.top().expect("at least one hit");
//! assert_eq!(top.unit.name, "scan");
//! assert!(top.structural_score > 0.0);
//! ```

pub mod engine;
pub mod parser;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::{CodeRetrievalEngine, CodeRetrievalIndex};
pub use parser::parse_source;
pub use types::{
    CodeRetrievalConfig, CodeRetrievalError, CodeRetrievalHit, CodeRetrievalLanguageHint,
    CodeRetrievalResult, CodeRetrievalSymbol, CodeRetrievalUnit, CodeRetrievalUnitKind,
};
