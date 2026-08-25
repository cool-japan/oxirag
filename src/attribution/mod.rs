//! Inline citation and source attribution for generated answers.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`AlignmentScorer`] | Score sentence-to-source relevance |
//! | [`LexicalAligner`] | Token-Jaccard alignment scorer |
//! | [`SentenceAligner`] | Split answer + attach citations above threshold |
//! | [`Attributor`] | Orchestrate align → deduplicate → annotate → faithfulness |
//! | [`CitationFormatter`] | Format `[1]` / `[^1]` / author-style markers |
//! | [`FaithfulnessChecker`] | Compute grounding fraction |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "attribution")] {
//! use oxirag::prelude::*;
//!
//! let answer = "Rust is safe. It prevents data races.";
//! let sources = vec![/* SearchResult values */];
//! let result = Attributor::new(AttributionConfig::default())
//!     .attribute(answer, &sources)
//!     .unwrap();
//! println!("Faithfulness: {:.2}", result.overall_faithfulness);
//! # }
//! ```

pub mod aligner;
pub mod citation;
pub mod faithfulness;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use aligner::{AlignmentScorer, Attributor, LexicalAligner, SentenceAligner};
pub use citation::{CitationFormatter, CitationStyle};
pub use faithfulness::FaithfulnessChecker;
pub use types::{
    AttributedAnswer, AttributionConfig, AttributionError, Citation, CitationId, CitedSpan,
};
