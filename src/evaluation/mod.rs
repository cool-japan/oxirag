//! RAG Evaluation Framework — RAGAS-style lexical metrics.
//!
//! This module provides evaluation tools for measuring retrieval quality and
//! answer faithfulness without requiring external LLM calls.  All scorers are
//! heuristic / lexical and operate on token sets derived from the input text.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::evaluation::{EvaluationSample, RagEvaluator};
//!
//! #[tokio::main]
//! async fn main() {
//!     let evaluator = RagEvaluator::default_metrics();
//!
//!     let sample = EvaluationSample::new(
//!         "sample-1",
//!         "What is Rust?",
//!         vec!["Rust is a systems programming language focused on safety.".to_string()],
//!         "Rust is a safe systems language.",
//!     );
//!
//!     let result = evaluator.evaluate(&sample).await.expect("evaluation failed");
//!     println!("Overall score: {:.3}", result.overall);
//! }
//! ```
//!
//! # Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | [`AnswerRelevanceScorer`] | Jaccard overlap between answer and query tokens |
//! | [`FaithfulnessScorer`] | Fraction of answer sentences supported by context |
//! | [`ContextPrecisionScorer`] | Fraction of context chunks relevant to the query |
//! | [`ContextRecallScorer`] | Fraction of ground-truth tokens present in context |
//! | [`OverallScorer`] | Weighted combination of the four metrics above |

pub mod dataset;
pub mod evaluator;
pub mod metrics;
pub mod types;

#[cfg(test)]
mod tests;

pub use dataset::{DatasetStats, EvaluationDataset};
pub use evaluator::{AggregateStats, RagEvaluator};
pub use metrics::{
    AnswerRelevanceScorer, ContextPrecisionScorer, ContextRecallScorer, EvaluationMetric,
    FaithfulnessScorer, OverallScorer,
};
pub use types::{EvalError, EvaluationResult, EvaluationSample};
