//! Step-Back Prompting (Zheng et al. 2023) for RAG pipelines.
//!
//! Given a specific question, the engine first abstracts it into a
//! higher-level "step-back" question — for example, `"What is the speed of
//! light?"` becomes `"What are the fundamental constants of physics?"` — then
//! retrieves documents using the abstracted question and synthesizes the final
//! answer from the broader context.
//!
//! The abstraction and synthesis logic is supplied by the caller through the
//! [`StepBackModel`] trait.  A deterministic [`MockStepBackModel`] is provided
//! for testing and prototyping.
//!
//! # Example
//!
//! ```
//! use oxirag::step_back::{MockStepBackModel, StepBackConfig, StepBackEngine};
//!
//! let model = MockStepBackModel;
//! let engine = StepBackEngine::new(StepBackConfig::default(), Box::new(model));
//!
//! let docs = vec![
//!     "Physics constants include the speed of light".to_string(),
//!     "The gravitational constant governs attraction".to_string(),
//! ];
//!
//! let result = engine.run("What is the speed of light?", &docs).unwrap();
//!
//! assert_eq!(result.original_query, "What is the speed of light?");
//! assert!(result.abstract_query.starts_with("In general terms, "));
//! assert!(!result.retrieved_docs.is_empty());
//! assert!(!result.final_answer.is_empty());
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::StepBackEngine;
pub use types::{MockStepBackModel, StepBackConfig, StepBackError, StepBackModel, StepBackResult};
