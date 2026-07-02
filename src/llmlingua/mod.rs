//! `LLMLingua`-style prompt compression via a self-fit perplexity model
//! (Jiang et al., 2023, "`LLMLingua`: Compressing Prompts for Accelerated
//! Inference of Large Language Models").
//!
//! The real `LLMLingua` scores each token's **surprisal** — its perplexity
//! contribution `-log2 P(token | history)` — with a small pretrained causal LM
//! and prunes the low-surprisal tokens, because predictable text carries little
//! unique information for the downstream LLM to lose. This crate is pure Rust
//! with no pretrained-model access (mirroring how `selfcheckgpt` and
//! `eigenscore` stand in for real LLM sampling with deterministic statistics),
//! so instead of faking the signal with a stopword or word-length heuristic it
//! builds a genuine **smoothed n-gram surrogate language model fit on the input
//! itself** and drives real coarse-to-fine compression from its surprisal
//! estimates.
//!
//! # Pipeline
//!
//! 1. **Surrogate LM** — [`PerplexityModel`] is a Jelinek-Mercer interpolated
//!    n-gram model (default trigram, backing off to bigram/unigram) with an
//!    add-k smoothed base, so probabilities sum to one over the vocabulary and
//!    no token ever has zero probability (hence never infinite surprisal). See
//!    the [`ngram_model`] docs for the smoothing derivation.
//! 2. **Stage A — coarse pruning** — the input is split into segments; each
//!    segment's **density** is its mean token surprisal. To reach the budget,
//!    the lowest-density segments are dropped first, never below
//!    [`LlmLinguaConfig::min_segments_retained`].
//! 3. **Stage B — fine pruning** — the [`BudgetController`] allocates the token
//!    budget across survivors *proportionally to density* (a water-filling
//!    optimisation, not a flat cut), then within each survivor the
//!    lowest-surprisal, unprotected tokens are dropped up to
//!    [`LlmLinguaConfig::max_local_drop_ratio`]. Numbers, capitalized tokens,
//!    and segment boundaries are protected so structurally important content is
//!    never stripped.
//!
//! The [`CompressionResult`] reports the compressed text, the requested and
//! achieved ratios (which can differ because of the floors and protection), and
//! per-segment [`SegmentStats`].
//!
//! # Distinct from `recomp` and `context_compression`
//!
//! Both of those compress by scoring sentences for **relevance** and selecting
//! whole sentences. This module is the only one whose keep/drop signal is
//! **token-level surprisal from a fitted language model**, and the only one
//! that prunes *within* a sentence and allocates a budget proportionally to
//! per-segment information density.
//!
//! # Example
//!
//! ```
//! use oxirag::llmlingua::{LlmLinguaConfig, PerplexityCompressor};
//!
//! // A boilerplate sentence is repeated (predictable, low information), then a
//! // novel sentence with a protected number carries the real content.
//! let text = "The quarterly report is attached. The quarterly report is attached. \
//!             Net revenue reached 4200000 dollars in the Tokyo division.";
//!
//! let compressor = PerplexityCompressor::new(LlmLinguaConfig::new().with_ratio(0.6));
//! let result = compressor.compress(text).expect("non-empty input");
//!
//! // Compression never grows the prompt, and the achieved ratio is reported.
//! assert!(result.compressed_token_count <= result.original_token_count);
//! assert!(result.achieved_ratio <= 1.0);
//! // Numeric tokens are protected, so the revenue figure survives.
//! assert!(result.compressed_text.contains("4200000"));
//! ```

pub mod budget;
pub mod compressor;
pub mod ngram_model;
#[cfg(test)]
mod tests;
pub mod types;

pub use budget::BudgetController;
pub use compressor::PerplexityCompressor;
pub use ngram_model::PerplexityModel;
pub use types::{
    CompressionResult, CompressionTarget, LlmLinguaConfig, LlmLinguaError, SegmentStats,
};
