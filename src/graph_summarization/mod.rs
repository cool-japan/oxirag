//! Community summaries and Microsoft-GraphRAG-style global/local search.
//!
//! Builds extractive community summaries from a `CommunityGraph`, then
//! answers queries via map-reduce (global) or entity-neighbourhood (local)
//! strategies.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`CommunitySummarizer`] | Builds extractive summaries per community |
//! | [`GlobalSearchEngine`] | Map-reduce over all summaries |
//! | [`LocalSearchEngine`] | Entity-neighbourhood expansion + gather |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(all(feature = "graph-summarization", feature = "graphrag"))] {
//! use oxirag::prelude::*;
//!
//! let summarizer = CommunitySummarizer::new();
//! # }
//! ```

pub mod search;
pub mod summarizer;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use search::GlobalSearchEngine;
pub use types::{
    CommunitySummary, GraphSummarizationConfig, GraphSummarizationError, SummaryReport,
};

#[cfg(feature = "graphrag")]
pub use search::LocalSearchEngine;
#[cfg(feature = "graphrag")]
pub use summarizer::CommunitySummarizer;
