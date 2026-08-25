//! Chain-of-Density summarization (Adams et al. 2023).
//!
//! This module implements the Chain-of-Density (`CoD`) procedure for producing
//! increasingly *entity-dense* summaries of a source text, using only
//! deterministic, pure-Rust heuristics (no models, no randomness).
//!
//! The procedure begins with a sparse initial summary built from the source's
//! lead sentence(s). Each subsequent iteration:
//!
//! 1. identifies salient entities present in the source but **missing** from the
//!    current summary (entities are capitalized multi-character tokens or
//!    numbers, extracted in first-appearance order);
//! 2. rewrites the summary to incorporate
//!    [`entities_per_step`](CodConfig::entities_per_step) of them through a
//!    compact clause; and
//! 3. compresses the result by dropping low-information filler so the summary
//!    stays close to the [`target_words`](CodConfig::target_words) budget.
//!
//! The net effect is a chain of summaries whose entity density rises while their
//! length stays roughly fixed. Every iteration is captured as a [`DensityStep`],
//! and the densest summary is exposed as
//! [`ChainOfDensityOutput::final_summary`].
//!
//! # Examples
//!
//! ```
//! use oxirag::chain_of_density::{ChainOfDensityEngine, CodConfig};
//!
//! let source = "Ada Lovelace worked with Charles Babbage in London. \
//!               The Analytical Engine was designed in 1837. \
//!               She wrote the first algorithm for the machine.";
//! let engine = ChainOfDensityEngine::new(CodConfig::default());
//! let output = engine.summarize(source).unwrap();
//!
//! assert_eq!(output.steps.len(), 3);
//! assert_eq!(output.final_summary, output.steps.last().unwrap().summary);
//! ```

pub mod engine;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use engine::ChainOfDensityEngine;
pub use types::{ChainOfDensityOutput, CodConfig, CodError, DensityStep};
