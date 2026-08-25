//! JSON-schema-constrained typed extraction from unstructured text.
//!
//! Extracts structured records from free-form text using keyword-proximity
//! heuristics.  No regex library is used.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ExtractionSchema`] | Field type + keyword descriptors |
//! | [`SchemaExtractor`] | Keyword-proximity value extractor |
//! | [`ExtractedRecord`] | Typed key-value result |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "structured-extraction")] {
//! use oxirag::prelude::*;
//!
//! let schema = ExtractionSchema::new()
//!     .with_field(FieldSchema::new("price", FieldType::Number).with_keyword("price"));
//! let extractor = SchemaExtractor::new();
//! # }
//! ```

pub mod extractor;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use extractor::SchemaExtractor;
pub use types::{
    ExtractedRecord, ExtractedValue, ExtractionConfig, ExtractionSchema, FieldSchema, FieldType,
    StructuredExtractionError,
};
