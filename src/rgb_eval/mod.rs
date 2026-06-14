//! RGB benchmark — evaluating four fundamental abilities of RAG systems.
//!
//! Implements the *Retrieval-Augmented Generation Benchmark* (RGB) of
//! Chen et al. (2023), *Benchmarking Large Language Models in
//! Retrieval-Augmented Generation*.  Unlike the `retrieval_eval` module (which
//! scores IR ranking metrics) and the `evaluation` module (which scores RAGAS
//! answer quality), RGB is a **capability harness**: it labels each test case
//! with one of four abilities and decides, per ability, whether the system's
//! answer is correct.
//!
//! # The four abilities
//!
//! | Ability | What it probes | Correct when … |
//! |---------|----------------|----------------|
//! | [`RgbAbility::NoiseRobustness`] | answering despite irrelevant/noise contexts | the answer matches the gold answer |
//! | [`RgbAbility::NegativeRejection`] | declining when no context holds the answer | the answer is a rejection ("I cannot answer …") |
//! | [`RgbAbility::InformationIntegration`] | fusing facts from multiple contexts | the answer covers **all** required sub-answers |
//! | [`RgbAbility::CounterfactualRobustness`] | resisting wrong info planted in context | the answer gives the **true** gold answer (not the counterfactual) |
//!
//! Answer correctness is judged heuristically by lexical token overlap with the
//! gold answer, thresholded by [`RgbConfig::match_threshold`]; rejections are
//! detected by matching [`RgbConfig::rejection_phrases`]. No LLM, randomness, or
//! ML is involved — the evaluation is fully deterministic.
//!
//! # Quick start
//!
//! ```rust
//! use oxirag::rgb_eval::{RgbAbility, RgbConfig, RgbEvaluator, RgbTestCase};
//!
//! let evaluator = RgbEvaluator::new(RgbConfig::default());
//!
//! let case = RgbTestCase::new(
//!     "Who wrote the Rust book?",
//!     "Steve Klabnik and Carol Nichols",
//!     vec![
//!         "Unrelated noise about cooking.".to_string(),
//!         "The Rust book was written by Steve Klabnik and Carol Nichols.".to_string(),
//!     ],
//!     RgbAbility::NoiseRobustness,
//! );
//!
//! let answer = "It was written by Steve Klabnik and Carol Nichols.";
//! assert!(evaluator.evaluate_case(&case, answer));
//! ```
pub mod evaluator;
#[cfg(test)]
mod tests;
pub mod types;

pub use evaluator::RgbEvaluator;
pub use types::{RgbAbility, RgbConfig, RgbError, RgbScores, RgbTestCase};
