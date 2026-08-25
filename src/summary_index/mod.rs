//! Document Summary Index (`LlamaIndex` `DocumentSummaryIndex`).
//!
//! Builds one extractive summary per document, indexes the **summary**
//! embeddings, and at query time matches the query against those summaries while
//! returning the **full parent document**. This couples the recall precision of
//! summary-level matching with the completeness of whole-document context.
//!
//! Salient sentences are selected by term-frequency centrality: each sentence is
//! scored by the summed document-local frequency of its tokens, the top-N are
//! kept, and they are re-emitted in their original order.
//!
//! This is a *flat*, one-summary-per-document index. It is deliberately
//! **distinct** from [`raptor`](crate::raptor), which builds a recursive
//! multi-level *tree* of summaries.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ExtractiveSummarizer`] | TF-centrality extractive summarization |
//! | [`SummaryIndex`] | Embeds, stores, and retrieves document summaries |
//! | [`DocumentSummary`] | A document's summary + its embedding |
//! | [`SummaryHit`] | A scored hit carrying the full parent document |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "summary-index")] {
//! use oxirag::prelude::*;
//!
//! let mut index = SummaryIndex::new(SummaryConfig::default());
//! index.add_document(&Document::new(
//!     "Rust is fast. Rust improves safety. The borrow checker prevents bugs.",
//! ));
//! let hits = index.search("safety", 5).unwrap();
//! // `hits[0].document` is the FULL document, not the summary.
//! # }
//! ```

pub mod index;
pub mod summarizer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use index::SummaryIndex;
pub use summarizer::{ExtractiveSummarizer, Summarizer};
pub use types::{DocumentSummary, SummaryConfig, SummaryHit, SummaryIndexError};
