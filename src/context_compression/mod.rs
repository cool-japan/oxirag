//! Extractive context distillation before generation.
//!
//! Compresses a set of retrieved documents into a token-budget-bounded
//! context string by selecting and deduplicating the most relevant sentences.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`ContextCompressor`] | Trait for sync compressors |
//! | [`ExtractiveCompressor`] | Score → deduplicate → pack |
//! | [`RedundancyFilter`] | Jaccard-based near-duplicate removal |
//! | [`MockCompressor`] | Scripted test double |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "context-compression")] {
//! use oxirag::prelude::*;
//!
//! let compressor = ExtractiveCompressor::new();
//! # }
//! ```

pub mod compressor;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use compressor::{ExtractiveCompressor, MockCompressor, RedundancyFilter};
pub use types::{CompressedContext, CompressionConfig, CompressionError, ContextCompressor};
