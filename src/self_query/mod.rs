//! Self-querying retriever (à la `LangChain` `SelfQueryRetriever`).
//!
//! Parses a natural-language query into a *structured* metadata filter plus a
//! *residual* semantic query string, driven by a configurable schema of known
//! metadata fields. For example:
//!
//! ```text
//! "papers about transformers after 2020 by Vaswani rated above 4"
//!   ⇒ filter   = [year > 2020, author = "Vaswani", rating > 4]
//!     semantic = "papers about transformers"
//! ```
//!
//! Everything is pure Rust, `std`-only, and fully deterministic — no regex,
//! no `rand`, no ML.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`FieldType`] / [`FieldSpec`] / [`FilterSchema`] | Configurable field schema |
//! | [`FilterOp`] / [`FilterCondition`] / [`ParsedFilter`] | Self-contained filter type |
//! | [`StructuredQuery`] | Semantic query + extracted filter |
//! | [`SelfQueryConfig`] | Parser behaviour (phrase stripping) |
//! | [`SelfQueryParser`] | Natural-language → [`StructuredQuery`] |
//! | [`SelfQueryRetriever`] | Parse + filter a document set |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "self-query")] {
//! use oxirag::prelude::*;
//!
//! let schema = FilterSchema::new(vec![
//!     FieldSpec::new("year", FieldType::Date),
//!     FieldSpec::new("author", FieldType::Text),
//!     FieldSpec::new("rating", FieldType::Number),
//! ]);
//! let parser = SelfQueryParser::new(schema);
//! let parsed = parser
//!     .parse("papers about transformers after 2020 by Vaswani rated above 4")
//!     .unwrap();
//! assert_eq!(parsed.semantic_query, "papers about transformers");
//! assert_eq!(parsed.filter.conditions.len(), 3);
//! # }
//! ```

pub mod parser;
#[cfg(test)]
mod tests;
pub mod types;

pub use parser::{SelfQueryParser, SelfQueryRetriever};
pub use types::{
    FieldSpec, FieldType, FilterCondition, FilterOp, FilterSchema, ParsedFilter, SelfQueryConfig,
    SelfQueryError, StructuredQuery,
};
