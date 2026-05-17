//! Distillation metrics and evaluation module.
//!
//! This module provides types and functionality for evaluating distilled models,
//! comparing teacher and student model performance, and tracking training metrics.

pub mod eval;
pub mod evaluator;
pub mod plot;
pub mod tracker;
pub mod transfer;

pub use eval::{
    ComparisonResult, ComparisonSummary, EvaluationResult, TestExample, TestExampleMetadata,
};
pub use evaluator::{DistillationEvaluator, EvalStudentModel, EvalTeacherModel, EvaluatorConfig};
pub use plot::{ExtraEpochMetrics, PlotData, TrainingEpochMetrics};
pub use tracker::{MetricsTracker, TrackerSummary};
pub use transfer::{KnowledgeTransferMetrics, LayerSimilarity};

#[cfg(test)]
mod tests;
