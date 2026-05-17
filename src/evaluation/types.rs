//! Core types for the RAG evaluation framework.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::PipelineOutput;

/// A single evaluation sample containing query, retrieved context, and generated answer.
///
/// Optionally includes a ground truth answer for reference-based scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSample {
    /// Unique identifier for this sample.
    pub id: String,
    /// The input query text.
    pub query: String,
    /// Retrieved context passages (document contents from search results).
    pub context: Vec<String>,
    /// The generated answer to evaluate.
    pub answer: String,
    /// Optional ground truth answer for reference-based scoring (e.g., context recall).
    pub ground_truth: Option<String>,
    /// Source document IDs that were retrieved.
    pub source_doc_ids: Vec<String>,
}

impl EvaluationSample {
    /// Create a new evaluation sample with required fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        query: impl Into<String>,
        context: Vec<String>,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            query: query.into(),
            context,
            answer: answer.into(),
            ground_truth: None,
            source_doc_ids: Vec::new(),
        }
    }

    /// Attach a ground truth answer to this sample.
    #[must_use]
    pub fn with_ground_truth(mut self, gt: impl Into<String>) -> Self {
        self.ground_truth = Some(gt.into());
        self
    }

    /// Attach source document IDs to this sample.
    #[must_use]
    pub fn with_source_doc_ids(mut self, ids: Vec<String>) -> Self {
        self.source_doc_ids = ids;
        self
    }

    /// Build an [`EvaluationSample`] from a [`PipelineOutput`].
    ///
    /// The sample ID is derived from a freshly generated UUID. Context passages
    /// are taken from the search-result document contents in ranked order.
    #[must_use]
    pub fn from_pipeline_output(output: &PipelineOutput) -> Self {
        let id = Uuid::new_v4().to_string();
        let context: Vec<String> = output
            .search_results
            .iter()
            .map(|sr| sr.document.content.clone())
            .collect();
        let source_doc_ids: Vec<String> = output
            .search_results
            .iter()
            .map(|sr| sr.document.id.to_string())
            .collect();
        Self {
            id,
            query: output.query.text.clone(),
            context,
            answer: output.final_answer.clone(),
            ground_truth: None,
            source_doc_ids,
        }
    }
}

/// Evaluation scores for a single [`EvaluationSample`].
///
/// All scores are in the range `[0.0, 1.0]` where higher is better.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// ID of the sample that was evaluated.
    pub sample_id: String,
    /// Lexical overlap between the answer and the query (Jaccard-based).
    pub answer_relevance: f32,
    /// Fraction of answer sentences supported by the retrieved context.
    pub faithfulness: f32,
    /// Fraction of retrieved context chunks relevant to the query.
    pub context_precision: f32,
    /// Fraction of ground-truth tokens appearing in the context (requires `ground_truth`).
    pub context_recall: f32,
    /// Weighted average across all four metrics.
    pub overall: f32,
}

impl EvaluationResult {
    /// Compute a weighted average across the four core metrics.
    ///
    /// When `cr` is `None` (no ground truth available), the context-recall weight
    /// is redistributed proportionally across the other three metrics so that the
    /// weights still sum to 1.0.
    #[must_use]
    pub fn weighted_average(ar: f32, faith: f32, cp: f32, cr: Option<f32>) -> f32 {
        const AR_W: f32 = 0.25;
        const FAITH_W: f32 = 0.35;
        const CP_W: f32 = 0.25;
        const CR_W: f32 = 0.15;

        if let Some(cr_val) = cr {
            ar * AR_W + faith * FAITH_W + cp * CP_W + cr_val * CR_W
        } else {
            // Redistribute CR weight proportionally among the other three.
            let total = AR_W + FAITH_W + CP_W;
            let ar_adj = AR_W / total;
            let faith_adj = FAITH_W / total;
            let cp_adj = CP_W / total;
            ar * ar_adj + faith * faith_adj + cp * cp_adj
        }
    }
}

/// Errors that can occur during evaluation.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// Returned when a context-recall score is requested but no ground truth was provided.
    #[error("Missing ground truth for recall scoring")]
    MissingGroundTruth,
    /// Returned when the context list is empty and a metric requires context.
    #[error("Empty context")]
    EmptyContext,
    /// A general evaluation failure with a descriptive message.
    #[error("Evaluation failed: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_average_with_recall() {
        let score = EvaluationResult::weighted_average(1.0, 1.0, 1.0, Some(1.0));
        let diff = (score - 1.0_f32).abs();
        assert!(diff < 1e-6_f32, "All-1s should give 1.0, got {score}");
    }

    #[test]
    fn test_weighted_average_without_recall() {
        let score = EvaluationResult::weighted_average(1.0, 1.0, 1.0, None);
        let diff = (score - 1.0_f32).abs();
        assert!(diff < 1e-6_f32, "All-1s without recall should give 1.0, got {score}");
    }

    #[test]
    fn test_weighted_average_zeros() {
        let score = EvaluationResult::weighted_average(0.0, 0.0, 0.0, Some(0.0));
        assert!((score).abs() < 1e-6_f32);
    }
}
