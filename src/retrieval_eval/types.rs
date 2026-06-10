//! Types for the `retrieval_eval` module.
use std::collections::HashMap;
use thiserror::Error;
// ── RelevanceJudgment ─────────────────────────────────────────────────────────
/// A graded relevance judgment for one document.
#[derive(Debug, Clone)]
pub struct RelevanceJudgment {
    /// Document identifier.
    pub doc_id: String,
    /// Graded relevance gain (e.g. 0 = not relevant, 1 = relevant, 2 = highly relevant).
    pub gain: f32,
}
impl RelevanceJudgment {
    /// Create a judgment with binary relevance (`gain = 1.0`).
    #[must_use]
    pub fn relevant(doc_id: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            gain: 1.0,
        }
    }
    /// Create a judgment with a specific gain.
    #[must_use]
    pub fn graded(doc_id: impl Into<String>, gain: f32) -> Self {
        Self {
            doc_id: doc_id.into(),
            gain,
        }
    }
}
// ── Qrels ─────────────────────────────────────────────────────────────────────
/// A complete set of relevance judgments for a single query.
#[derive(Debug, Clone, Default)]
pub struct Qrels {
    /// Map from document ID to graded gain.
    pub judgments: HashMap<String, f32>,
}
impl Qrels {
    /// Create from a list of judgments.
    #[must_use]
    pub fn from_judgments(judgments: Vec<RelevanceJudgment>) -> Self {
        Self {
            judgments: judgments.into_iter().map(|j| (j.doc_id, j.gain)).collect(),
        }
    }
    /// Return true if `doc_id` is relevant at or above `threshold`.
    #[must_use]
    pub fn is_relevant(&self, doc_id: &str, threshold: f32) -> bool {
        self.judgments.get(doc_id).copied().unwrap_or(0.0) >= threshold
    }
    /// Return the graded gain for `doc_id` (0.0 if absent).
    #[must_use]
    pub fn gain(&self, doc_id: &str) -> f32 {
        self.judgments.get(doc_id).copied().unwrap_or(0.0)
    }
}
// ── RetrievalScores ───────────────────────────────────────────────────────────
/// Per-query retrieval metric scores.
#[derive(Debug, Clone, Default)]
pub struct RetrievalScores {
    /// Precision@k.
    pub precision_at_k: f32,
    /// Recall@k.
    pub recall_at_k: f32,
    /// F1@k.
    pub f1_at_k: f32,
    /// Hit rate@k (1 if any relevant doc in top-k, else 0).
    pub hit_rate_at_k: f32,
    /// Reciprocal rank of the first relevant result.
    pub reciprocal_rank: f32,
    /// Average precision (area under precision-recall curve).
    pub average_precision: f32,
    /// Discounted cumulative gain@k.
    pub dcg_at_k: f32,
    /// Normalised discounted cumulative gain@k.
    pub ndcg_at_k: f32,
    /// k used for all @k metrics.
    pub k: usize,
}
// ── AggregateScores ───────────────────────────────────────────────────────────
/// Macro-averaged retrieval scores over multiple queries.
#[derive(Debug, Clone, Default)]
pub struct AggregateScores {
    /// Mean average precision over all queries.
    pub map: f32,
    /// Mean reciprocal rank over all queries.
    pub mrr: f32,
    /// Mean nDCG@k over all queries.
    pub mean_ndcg_at_k: f32,
    /// Mean recall@k over all queries.
    pub mean_recall_at_k: f32,
    /// Mean precision@k over all queries.
    pub mean_precision_at_k: f32,
    /// Number of queries evaluated.
    pub query_count: usize,
}
// ── RetrievalEvalConfig ───────────────────────────────────────────────────────
/// Configuration for [`RetrievalEvaluator`].
#[derive(Debug, Clone)]
pub struct RetrievalEvalConfig {
    /// Cutoff k for @k metrics. Defaults to `10`.
    pub k: usize,
    /// Minimum gain to consider a document relevant. Defaults to `1.0`.
    pub relevance_threshold: f32,
    /// Logarithm base for DCG computation. Defaults to `2.0`.
    pub log_base: f32,
}
impl Default for RetrievalEvalConfig {
    fn default() -> Self {
        Self {
            k: 10,
            relevance_threshold: 1.0,
            log_base: 2.0,
        }
    }
}
impl RetrievalEvalConfig {
    /// Set the cutoff k.
    #[must_use]
    pub fn with_k(mut self, v: usize) -> Self {
        self.k = v;
        self
    }
    /// Set the relevance threshold.
    #[must_use]
    pub fn with_relevance_threshold(mut self, v: f32) -> Self {
        self.relevance_threshold = v;
        self
    }
    /// Set the log base for DCG.
    #[must_use]
    pub fn with_log_base(mut self, v: f32) -> Self {
        self.log_base = v;
        self
    }
}
// ── RetrievalEvalError ────────────────────────────────────────────────────────
/// Errors from the `retrieval_eval` module.
#[derive(Debug, Error)]
pub enum RetrievalEvalError {
    /// The result list was empty.
    #[error("Result list must not be empty")]
    EmptyResults,
    /// The Qrels set was empty.
    #[error("Qrels set must not be empty")]
    EmptyQrels,
    /// The value of k was 0.
    #[error("k must be at least 1")]
    InvalidK,
}
// ── RetrievalEvaluator ────────────────────────────────────────────────────────
/// Evaluator that computes retrieval metrics for a given result list and qrels.
#[derive(Debug, Clone, Default)]
pub struct RetrievalEvaluator {
    /// Configuration for this evaluator.
    pub config: RetrievalEvalConfig,
}
impl RetrievalEvaluator {
    /// Create a new evaluator with the given config.
    #[must_use]
    pub fn new(config: RetrievalEvalConfig) -> Self {
        Self { config }
    }
    /// Evaluate a single query's results against `qrels`.
    ///
    /// # Errors
    ///
    /// Returns [`RetrievalEvalError`] if results or qrels are empty or k is 0.
    pub fn evaluate(
        &self,
        results: &[crate::types::SearchResult],
        qrels: &Qrels,
        k: usize,
    ) -> Result<RetrievalScores, RetrievalEvalError> {
        use crate::retrieval_eval::metrics::{
            average_precision, dcg_at_k, f1_at_k, hit_rate_at_k, ndcg_at_k, precision_at_k,
            recall_at_k, reciprocal_rank,
        };
        if results.is_empty() {
            return Err(RetrievalEvalError::EmptyResults);
        }
        if qrels.judgments.is_empty() {
            return Err(RetrievalEvalError::EmptyQrels);
        }
        if k == 0 {
            return Err(RetrievalEvalError::InvalidK);
        }
        let thr = self.config.relevance_threshold;
        let lb = self.config.log_base;
        Ok(RetrievalScores {
            precision_at_k: precision_at_k(results, qrels, k, thr),
            recall_at_k: recall_at_k(results, qrels, k, thr),
            f1_at_k: f1_at_k(results, qrels, k, thr),
            hit_rate_at_k: hit_rate_at_k(results, qrels, k, thr),
            reciprocal_rank: reciprocal_rank(results, qrels, thr),
            average_precision: average_precision(results, qrels, thr),
            dcg_at_k: dcg_at_k(results, qrels, k, lb),
            ndcg_at_k: ndcg_at_k(results, qrels, k, lb),
            k,
        })
    }
    /// Evaluate multiple queries and return macro-averaged scores.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered.
    pub fn evaluate_batch(
        &self,
        queries: &[(&[crate::types::SearchResult], &Qrels)],
    ) -> Result<AggregateScores, RetrievalEvalError> {
        if queries.is_empty() {
            return Ok(AggregateScores::default());
        }
        let k = self.config.k;
        let mut map_sum = 0.0_f32;
        let mut mrr_sum = 0.0_f32;
        let mut ndcg_sum = 0.0_f32;
        let mut recall_sum = 0.0_f32;
        let mut prec_sum = 0.0_f32;
        let n = queries.len();
        for (results, qrels) in queries {
            let s = self.evaluate(results, qrels, k)?;
            map_sum += s.average_precision;
            mrr_sum += s.reciprocal_rank;
            ndcg_sum += s.ndcg_at_k;
            recall_sum += s.recall_at_k;
            prec_sum += s.precision_at_k;
        }
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f32;
        Ok(AggregateScores {
            map: map_sum / nf,
            mrr: mrr_sum / nf,
            mean_ndcg_at_k: ndcg_sum / nf,
            mean_recall_at_k: recall_sum / nf,
            mean_precision_at_k: prec_sum / nf,
            query_count: n,
        })
    }
}
