//! FILCO: Learning to Filter Context for Retrieval-Augmented Generation
//! (Wang et al., 2023).
//!
//! FILCO filters retrieved context at **sentence granularity**, keeping only
//! the sentences useful for generating an answer and dropping the rest,
//! rather than compressing to a fixed budget or discarding whole passages.
//! [`FilcoFilter::filter`] splits every passage into sentences, scores each
//! sentence with one of three named measures, keeps the sentences that clear
//! the measure's threshold, and reassembles the survivors per passage in
//! their original relative order.
//!
//! # The three measures
//!
//! | Measure | Signal | Threshold |
//! |---------|--------|-----------|
//! | [`FilterMeasure::StrInc`] (default) | String Inclusion — does the sentence contain a plausible answer span? | implicit binary match |
//! | [`FilterMeasure::LexicalOverlap`] | Jaccard token overlap between the sentence and the query | [`FilcoConfig::lexical_threshold`] (default `0.15`) |
//! | [`FilterMeasure::CxmiLite`] | Heuristic proxy for Conditional Cross-Mutual Information | [`FilcoConfig::cxmi_threshold`] (default `0.05`) |
//!
//! STRINC has two variants, both implemented in [`measures`]:
//!
//! - **Heuristic** (used by [`FilcoFilter::filter`] when no answer is known —
//!   the general inference-time case): a sentence is kept when it shares a
//!   contiguous token n-gram of length `>= `[`FilcoConfig::strinc_min_ngram`]
//!   with the query, or contains a capitalized/numeric token that the query
//!   also implies. This is a deterministic, model-free stand-in for "does
//!   this sentence look like it carries the answer" — see
//!   `measures::strinc_heuristic_score` for the full rationale.
//! - **Strict / paper-faithful** (used by [`FilcoFilter::filter_with_answer`]
//!   when the answer *is* known, e.g. building training data): a sentence is
//!   kept exactly when it contains the expected answer as a case-insensitive
//!   substring, matching the paper's original STRINC definition.
//!
//! CXMI-lite is an explicit **heuristic proxy** for the paper's real CXMI
//! (which requires a language model to estimate `P(answer | query,
//! sentence)`): it substitutes a deterministic TF-IDF-lite relevance
//! estimate and reports the gain a sentence provides over an empty context.
//! This mirrors the honesty convention of this codebase's other `-lite`
//! heuristic modules, e.g. `context_pruning`'s LLMLingua-perplexity-lite
//! token pruning and `semantic_entropy`'s entropy-lite uncertainty proxy —
//! see [`measures`] for the full derivation.
//!
//! # Distinct from `noise_filter`, `context_pruning`, and `recomp`
//!
//! | Module | Granularity | Mechanism |
//! |--------|-------------|-----------|
//! | [`crate::noise_filter`] | Passage | RAAT-inspired relevance + retrieved-set-consensus blend; drops whole distractor/topic-drift **passages** |
//! | [`crate::context_pruning`] | Token | LLMLingua-style importance ranking; drops low-importance **tokens** down to a target compression ratio |
//! | [`crate::recomp`] | Sentence | Selects/repacks or lightly restitches sentences into a fixed **token budget**, gated by a selective-augmentation decision |
//! | **`filco`** (this module) | Sentence | Applies one of **three named measures** (STRINC / lexical overlap / CXMI-lite) as a per-sentence **binary keep/drop filter** — no budget, no passage-level consensus, no summarization |
//!
//! # Example
//!
//! ```
//! use oxirag::filco::{FilcoConfig, FilcoFilter, FilterMeasure};
//!
//! let config = FilcoConfig::new().with_measure(FilterMeasure::LexicalOverlap);
//! let filter = FilcoFilter::new(config);
//!
//! let passages = vec![
//!     "Paris is the capital of France.".to_string(),
//!     "Bananas are yellow and rich in potassium.".to_string(),
//! ];
//!
//! let report = filter
//!     .filter("What is the capital of France?", &passages)
//!     .unwrap();
//!
//! // The on-topic sentence survives; the irrelevant one is filtered out.
//! assert_eq!(report.filtered_passages.len(), 2);
//! assert!(report.filtered_passages[0].contains("Paris"));
//! assert!(report.filtered_passages[1].is_empty());
//! assert!(report.compression_ratio > 0.0 && report.compression_ratio <= 1.0);
//! ```

pub mod filter;
pub mod measures;
#[cfg(test)]
mod tests;
pub mod types;

pub use filter::FilcoFilter;
pub use types::{FilcoConfig, FilcoError, FilcoReport, FilterMeasure, ScoredSentence};
