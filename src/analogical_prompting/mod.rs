//! Analogical Prompting (Yasunaga et al. 2023) — self-generated exemplars and
//! high-level knowledge for in-context reasoning.
//!
//! Before solving a target problem, the model **self-generates** a handful of
//! relevant exemplars (similar problems it has already solved) and, optionally, a
//! high-level knowledge tutorial. It then solves the problem conditioned on the
//! exemplars and knowledge it produced for itself. Unlike `prompt_optimization`,
//! which *selects* demonstrations from a fixed pool, Analogical Prompting requires
//! no pool — the model invents its own demonstrations on demand.
//!
//! The model is supplied by the caller as the [`AnalogicalModel`] trait; a
//! deterministic [`MockAnalogicalModel`] is provided for testing.
//!
//! # Example
//!
//! ```
//! use oxirag::analogical_prompting::{
//!     AnalogicalConfig, AnalogicalEngine, Exemplar, MockAnalogicalModel,
//! };
//!
//! let model = MockAnalogicalModel::new(
//!     vec![
//!         Exemplar::new("What is 2 + 3?", "2 + 3 = 5"),
//!         Exemplar::new("What is 10 + 4?", "10 + 4 = 14"),
//!     ],
//!     "Addition combines two numbers into their sum.",
//!     "7 + 8 = 15",
//! );
//!
//! let engine = AnalogicalEngine::new(AnalogicalConfig::default());
//! let output = engine.run("What is 7 + 8?", &model).unwrap();
//!
//! assert_eq!(output.exemplars.len(), 2);
//! assert!(!output.knowledge.is_empty());
//! assert_eq!(output.answer, "7 + 8 = 15");
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::AnalogicalEngine;
pub use types::{
    AnalogicalConfig, AnalogicalError, AnalogicalModel, AnalogicalOutput, Exemplar,
    MockAnalogicalModel,
};
