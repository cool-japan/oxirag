//! Fusion-in-Decoder (`FiD`, Izacard & Grave 2021): a heuristic, pure-Rust
//! reading-comprehension fusion strategy for RAG.
//!
//! Each retrieved passage is processed **independently** to extract its single
//! most query-relevant sentence together with a relevance weight (the
//! query↔sentence token overlap).  The per-passage evidence is then **fused**
//! across passages — deduplicated and concatenated in descending relevance
//! order — into one answer that carries multi-passage attribution.
//!
//! This is deliberately distinct from `chain_of_note`: that module builds
//! per-document extractive *notes* and then *synthesises* them, whereas here
//! extraction is strictly per-passage and independent, and fusion is
//! relevance-weighted with explicit deduplication of near-identical evidence.
//!
//! # Example
//!
//! ```
//! use oxirag::fusion_in_decoder::{FidConfig, FusionInDecoder};
//! use oxirag::types::Document;
//!
//! let docs = vec![
//!     Document::new("The Eiffel Tower is located in Paris, France.").with_id("a"),
//!     Document::new("Paris is the capital of France and a major city.").with_id("b"),
//! ];
//! let fid = FusionInDecoder::new(FidConfig::default());
//! let fused = fid.fuse("Where is the Eiffel Tower?", &docs).unwrap();
//! assert!(!fused.attributions.is_empty());
//! ```

pub mod fusion;
pub mod types;

#[cfg(test)]
mod tests;

pub use fusion::FusionInDecoder;
pub use types::{FidConfig, FidError, FusedAnswer, PassageEvidence};
