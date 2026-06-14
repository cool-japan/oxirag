//! Least-to-Most Prompting (Zhou et al. 2022) for RAG pipelines.
//!
//! Given a complex question, the engine first uses a [`LtmSolver`] to
//! decompose it into an ordered list of simpler sub-problems (from *least* to
//! *most* complex), then solves each sub-problem sequentially. Every step
//! receives the answers to all prior steps plus a set of context documents
//! selected by token overlap with the current sub-question. The final answer
//! is the result of the last step.
//!
//! The solver logic is pluggable via the [`LtmSolver`] trait; a deterministic
//! [`MockLtmSolver`] is provided for testing and prototyping.
//!
//! # Example
//!
//! ```
//! use oxirag::least_to_most::{LtmConfig, LtmEngine, MockLtmSolver};
//!
//! let engine = LtmEngine::new(LtmConfig::default(), Box::new(MockLtmSolver));
//!
//! let docs = vec![
//!     "France is a country in Western Europe.".to_string(),
//!     "Paris is the capital and largest city of France.".to_string(),
//! ];
//!
//! let result = engine
//!     .run("What is France and what is its capital?", &docs)
//!     .unwrap();
//!
//! assert_eq!(
//!     result.original_query,
//!     "What is France and what is its capital?"
//! );
//! assert_eq!(result.subproblems.len(), 2);
//! assert_eq!(result.subproblems[0].step_index, 0);
//! assert_eq!(result.subproblems[1].step_index, 1);
//! assert_eq!(result.final_answer, result.subproblems[1].answer);
//! assert!(!result.final_answer.is_empty());
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::LtmEngine;
pub use types::{LtmConfig, LtmError, LtmResult, LtmSolver, MockLtmSolver, SubProblem};
