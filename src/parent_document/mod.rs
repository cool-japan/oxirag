//! Small-to-big retrieval via parent-document indexing.
//!
//! Indexes child chunks alongside their parent documents, retrieves the most
//! relevant chunks, then expands each hit to its full parent context.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ParentChildIndex`] | child → parent mapping |
//! | [`ChunkHierarchy`] | Splits a parent into child chunks |
//! | [`ParentDocumentRetriever`] | Retrieves children, expands to parents |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(all(feature = "parent-document", feature = "chunking"))] {
//! use oxirag::prelude::*;
//!
//! let mut index = ParentChildIndex::new();
//! # }
//! ```

pub mod retriever;
#[cfg(test)]
mod tests;
pub mod types;

pub use retriever::ParentDocumentRetriever;
pub use types::{
    ChunkHierarchy, ExpandedResult, ParentChildIndex, ParentDocumentConfig, ParentDocumentError,
    WindowConfig,
};
