//! Self-Refine (Madaan et al. 2023) — iterative output improvement via the same
//! model's own feedback.
//!
//! A single model generates an initial output, then repeatedly critiques and
//! refines it: each iteration the model produces [`Feedback`] on the current
//! output and, unless that feedback says to stop, refines the output using it.
//! The loop ends when the feedback's `stop` flag is set, its `score` reaches the
//! configured threshold, or the iteration cap is hit.
//!
//! This uses **no retrieval and no episodic memory** — the same model fills the
//! generate, critique, and refine roles on its own previous output. That is the
//! key contrast with `reflexion`, which retrieves documents, scores attempts
//! with a separate evaluator, and accumulates memory across retries.
//!
//! # Example
//!
//! ```
//! use oxirag::self_refine::{
//!     Feedback, MockRefiner, SelfRefineConfig, SelfRefineEngine,
//! };
//!
//! // Initial draft is poor (0.4), the first refinement is good enough (0.95).
//! let refiner = MockRefiner::new(
//!     "rough draft",
//!     vec![
//!         (Feedback::new("add detail", 0.4, false), "polished draft".to_string()),
//!         (Feedback::new("looks great", 0.95, true), String::new()),
//!     ],
//! );
//!
//! let engine = SelfRefineEngine::new(SelfRefineConfig::default());
//! let out = engine.run("write a summary", &refiner).unwrap();
//!
//! assert_eq!(out.iterations, 2);
//! assert_eq!(out.steps[0].output, "rough draft");
//! assert_eq!(out.final_output, "polished draft");
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::SelfRefineEngine;
pub use types::{
    Feedback, MockRefiner, RefineStep, Refiner, SelfRefineConfig, SelfRefineError, SelfRefineOutput,
};
