//! The [`RagEvaluator`] orchestrates metric scoring across samples and datasets.

use serde::{Deserialize, Serialize};

use super::{
    dataset::EvaluationDataset,
    metrics::{
        AnswerRelevanceScorer, ContextPrecisionScorer, ContextRecallScorer, EvaluationMetric,
        FaithfulnessScorer,
    },
    types::{EvalError, EvaluationResult, EvaluationSample},
};

// ---------------------------------------------------------------------------
// Aggregate statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics computed across all samples in an [`EvaluationDataset`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    /// Mean answer-relevance score.
    pub mean_answer_relevance: f32,
    /// Mean faithfulness score.
    pub mean_faithfulness: f32,
    /// Mean context-precision score.
    pub mean_context_precision: f32,
    /// Mean context-recall score (only present when at least one sample had ground truth).
    pub mean_context_recall: Option<f32>,
    /// Mean overall (weighted-average) score.
    pub mean_overall: f32,
    /// Number of samples that were successfully evaluated.
    pub sample_count: usize,
}

// ---------------------------------------------------------------------------
// RagEvaluator
// ---------------------------------------------------------------------------

/// Orchestrates the evaluation of [`EvaluationSample`]s using a collection of
/// [`EvaluationMetric`] implementations.
pub struct RagEvaluator {
    metrics: Vec<Box<dyn EvaluationMetric>>,
}

impl RagEvaluator {
    /// Create a [`RagEvaluator`] using all four default RAGAS-style metrics.
    #[must_use]
    pub fn default_metrics() -> Self {
        Self {
            metrics: vec![
                Box::new(AnswerRelevanceScorer::default()),
                Box::new(FaithfulnessScorer::default()),
                Box::new(ContextPrecisionScorer::default()),
                Box::new(ContextRecallScorer),
            ],
        }
    }

    /// Create a [`RagEvaluator`] with a custom set of metrics.
    #[must_use]
    pub fn new(metrics: Vec<Box<dyn EvaluationMetric>>) -> Self {
        Self { metrics }
    }

    /// Evaluate a single [`EvaluationSample`] using all registered metrics.
    ///
    /// Metric results are matched by name: `answer_relevance`, `faithfulness`,
    /// `context_precision`, and `context_recall`.  Any unknown metric names are
    /// silently ignored for the per-metric fields but do affect the overall score
    /// via the weighted-average helper.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] when a registered metric fails for a reason other
    /// than a missing ground truth (which is handled gracefully).
    pub async fn evaluate(&self, sample: &EvaluationSample) -> Result<EvaluationResult, EvalError> {
        let mut answer_relevance_score: f32 = 0.0;
        let mut faithfulness_score: f32 = 0.0;
        let mut context_precision_score: f32 = 0.0;
        let mut context_recall_score: Option<f32> = None;

        for metric in &self.metrics {
            let score_result = metric.score(sample).await;
            match metric.name() {
                "answer_relevance" => {
                    answer_relevance_score = score_result?;
                }
                "faithfulness" => {
                    faithfulness_score = score_result?;
                }
                "context_precision" => {
                    context_precision_score = score_result?;
                }
                "context_recall" => {
                    match score_result {
                        Ok(v) => context_recall_score = Some(v),
                        // Missing ground truth is expected — just leave cr as None.
                        Err(EvalError::MissingGroundTruth) => {}
                        Err(e) => return Err(e),
                    }
                }
                _ => {
                    // Unknown metric — run it to surface errors, but ignore the value
                    // for the structured fields.
                    let _ = score_result?;
                }
            }
        }

        let overall = EvaluationResult::weighted_average(
            answer_relevance_score,
            faithfulness_score,
            context_precision_score,
            context_recall_score,
        );
        Ok(EvaluationResult {
            sample_id: sample.id.clone(),
            answer_relevance: answer_relevance_score,
            faithfulness: faithfulness_score,
            context_precision: context_precision_score,
            context_recall: context_recall_score.unwrap_or(0.0),
            overall,
        })
    }

    /// Evaluate every sample in `dataset`, returning one result (or error) per sample.
    pub async fn evaluate_dataset(
        &self,
        dataset: &EvaluationDataset,
    ) -> Vec<Result<EvaluationResult, EvalError>> {
        let mut results = Vec::with_capacity(dataset.len());
        for sample in dataset.iter() {
            results.push(self.evaluate(sample).await);
        }
        results
    }

    /// Compute [`AggregateStats`] across all successfully evaluated samples.
    ///
    /// Samples that produce an error are silently skipped; only successful
    /// evaluations contribute to the aggregated means.
    #[allow(clippy::cast_precision_loss)]
    pub async fn aggregate_stats(&self, dataset: &EvaluationDataset) -> AggregateStats {
        let raw_results = self.evaluate_dataset(dataset).await;

        let results: Vec<EvaluationResult> = raw_results
            .into_iter()
            .filter_map(Result::ok)
            .collect();

        if results.is_empty() {
            return AggregateStats {
                mean_answer_relevance: 0.0,
                mean_faithfulness: 0.0,
                mean_context_precision: 0.0,
                mean_context_recall: None,
                mean_overall: 0.0,
                sample_count: 0,
            };
        }

        let count = results.len() as f32;
        let mean_ar = results.iter().map(|r| r.answer_relevance).sum::<f32>() / count;
        let mean_faith = results.iter().map(|r| r.faithfulness).sum::<f32>() / count;
        let mean_cp = results.iter().map(|r| r.context_precision).sum::<f32>() / count;
        let mean_ov = results.iter().map(|r| r.overall).sum::<f32>() / count;

        // context_recall is only meaningful when at least one sample had ground truth.
        // We use the dataset to determine whether that is the case.
        let gt_count = dataset
            .iter()
            .filter(|s| s.ground_truth.is_some())
            .count();
        let mean_recall = if gt_count > 0 {
            Some(results.iter().map(|r| r.context_recall).sum::<f32>() / count)
        } else {
            None
        };

        AggregateStats {
            mean_answer_relevance: mean_ar,
            mean_faithfulness: mean_faith,
            mean_context_precision: mean_cp,
            mean_context_recall: mean_recall,
            mean_overall: mean_ov,
            sample_count: results.len(),
        }
    }
}
