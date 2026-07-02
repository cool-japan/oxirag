//! RECOMP: Improving Retrieval-Augmented LMs with Compression (Xu et al.,
//! ICLR 2024).
//!
//! RECOMP compresses retrieved passages **before** they reach the generator,
//! using one of two compressor families, gated by a *selective-augmentation*
//! decision that can skip compression (and, implicitly, retrieval
//! augmentation) entirely when the retrieved set does not look relevant
//! enough to be worth the cost.
//!
//! # Pipeline
//!
//! 1. **Selective-augmentation gate** — [`pipeline::RecompPipeline::compress`]
//!    first scores the overall relevance of the passage set against the
//!    query (mean Jaccard token overlap per passage). If that score is below
//!    [`RecompConfig::augmentation_threshold`], the call returns
//!    [`RecompDecision::Skip`] without compressing anything.
//! 2. **Compression** — otherwise, the configured [`CompressorStrategy`] runs:
//!    - [`CompressorStrategy::Extractive`] via [`ExtractiveSummaryCompressor`]
//!      selects and repacks whole sentences, preserving their original order
//!      across passages, within [`RecompConfig::token_budget`].
//!    - [`CompressorStrategy::AbstractiveLite`] via
//!      [`AbstractiveSummaryCompressor`] selects the same top-scored
//!      sentences but stitches them into a single template-fused paragraph.
//!
//!    Both compressors deduplicate near-identical sentences using
//!    [`RecompConfig::dedup_similarity_threshold`].
//!
//! # Distinct from `context_compression` and `context_pruning`
//!
//! | Module | Granularity | Compressor modes | Selective gate |
//! |--------|-------------|-------------------|-----------------|
//! | `context_compression` | Sentence | Single extractive strategy + redundancy filter, over `SearchResult` sources | No |
//! | `context_pruning` | Token (`LLMLingua`-style importance pruning) | Single strategy | No |
//! | **`recomp`** | Sentence | **Dual**: extractive *and* abstractive-lite, over raw passage strings | **Yes** — [`RecompDecision::Skip`] when retrieval is judged low-value |
//!
//! `recomp`'s extractive compressor additionally restores **original
//! document order** in its output (`context_compression`'s compressor emits
//! sentences in score-ranked order instead), and its selective-augmentation
//! gate has no equivalent in either of the other two modules.
//!
//! # Honest scope note
//!
//! Both compressors are deterministic, pure-Rust, model-free heuristics.
//! "Abstractive-lite" is an extractive-fusion **template**: it selects and
//! lightly restitches the same top-scoring sentences with connective phrases.
//! It does not perform neural abstractive summarization or paraphrase — see
//! [`abstractive`] for the full honesty note.
//!
//! # Example
//!
//! ```
//! use oxirag::recomp::{RecompConfig, RecompDecision, RecompPipeline};
//!
//! let passages = vec![
//!     "The Rust borrow checker enforces memory safety at compile time.".to_string(),
//!     "Ownership rules in Rust prevent data races without a garbage collector.".to_string(),
//! ];
//!
//! let pipeline = RecompPipeline::new(RecompConfig::default());
//! let outcome = pipeline
//!     .compress("What is the Rust borrow checker?", &passages)
//!     .unwrap();
//!
//! match &outcome.decision {
//!     RecompDecision::Augment(text) => {
//!         assert!(!text.is_empty());
//!         assert!(outcome.compressed_token_count <= RecompConfig::default().token_budget);
//!     }
//!     RecompDecision::Skip { reason } => panic!("expected augmentation, got skip: {reason}"),
//! }
//! ```

pub mod abstractive;
pub mod extractive;
pub mod pipeline;
#[cfg(test)]
mod tests;
pub mod types;

pub use abstractive::AbstractiveSummaryCompressor;
pub use extractive::ExtractiveSummaryCompressor;
pub use pipeline::RecompPipeline;
pub use types::{CompressorStrategy, RecompConfig, RecompDecision, RecompError, RecompOutcome};
