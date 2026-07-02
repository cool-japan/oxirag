//! The eRAG evaluator: per-document downstream-task utility and cross-case
//! rank correlation.

use crate::erag::correlation::{kendall_tau, spearman_rho};
use crate::erag::types::{
    AggregationMethod, DownstreamTask, ERagBatchReport, ERagCase, ERagConfig, ERagError,
    ERagReport, PerDocScore, UtilityMetric,
};

/// Minimum number of cases required to compute a rank correlation.
const MIN_CASES_FOR_CORRELATION: usize = 2;

// ── ERagEvaluator ─────────────────────────────────────────────────────────────

/// Evaluates a retriever via eRAG: per-document downstream-task utility
/// (Salemi & Zamani, 2024).
///
/// Rather than scoring an end-to-end RAG pipeline as a whole, eRAG isolates
/// the *retriever's* contribution by running the downstream task separately
/// on each individually retrieved document, scoring each per-document output
/// against a gold answer, and aggregating those scores into a single
/// retriever-quality signal. The end-to-end reference run (the task on the
/// *full* retrieved set) is computed alongside it so the two can be compared;
/// [`evaluate_batch`](Self::evaluate_batch) correlates them across a batch of
/// cases, which is where the paper's sample-efficiency claim applies (a
/// single case has no rank to correlate).
///
/// `ERagEvaluator` holds no mutable state beyond its [`ERagConfig`]; it is
/// safe to use from multiple threads simultaneously.
#[derive(Debug, Clone, Copy, Default)]
pub struct ERagEvaluator {
    /// The evaluator's configuration (aggregation method and nDCG log base).
    pub config: ERagConfig,
}

impl ERagEvaluator {
    /// Create a new evaluator with the given configuration.
    #[must_use]
    pub fn new(config: ERagConfig) -> Self {
        Self { config }
    }

    /// Evaluate a single query's retrieved documents with eRAG.
    ///
    /// For every document `i` in `retrieved_docs`, runs `task.run(query,
    /// &[doc_i])` and scores the output against `gold_answer` with `utility`,
    /// producing `per_doc_scores[i]`. Those scores are aggregated via
    /// [`ERagConfig::aggregation`] into `aggregated_score`. Separately, runs
    /// `task.run(query, retrieved_docs)` (the *full* set at once) and scores
    /// that output against `gold_answer` to produce `end_to_end_score` — the
    /// reference the per-document aggregate is meant to approximate.
    ///
    /// # Errors
    ///
    /// - [`ERagError::InvalidConfig`] — `self.config` fails
    ///   [`ERagConfig::validate`].
    /// - [`ERagError::EmptyQuery`] — `query` is empty or whitespace-only.
    /// - [`ERagError::EmptyDocs`] — `retrieved_docs` is empty.
    /// - [`ERagError::EmptyGold`] — `gold_answer` is empty or whitespace-only.
    pub fn evaluate(
        &self,
        query: &str,
        gold_answer: &str,
        retrieved_docs: &[String],
        task: &dyn DownstreamTask,
        utility: &dyn UtilityMetric,
    ) -> Result<ERagReport, ERagError> {
        self.config.validate()?;
        if query.trim().is_empty() {
            return Err(ERagError::EmptyQuery);
        }
        if retrieved_docs.is_empty() {
            return Err(ERagError::EmptyDocs);
        }
        if gold_answer.trim().is_empty() {
            return Err(ERagError::EmptyGold);
        }

        let mut per_doc_scores = Vec::with_capacity(retrieved_docs.len());
        for (doc_index, doc) in retrieved_docs.iter().enumerate() {
            let single_doc_context = std::slice::from_ref(doc);
            let output = task.run(query, single_doc_context);
            let utility_score = utility.score(&output, gold_answer);
            per_doc_scores.push(PerDocScore {
                doc_index,
                utility: utility_score,
            });
        }

        let aggregated_score = aggregate(&per_doc_scores, self.config);

        let end_to_end_output = task.run(query, retrieved_docs);
        let end_to_end_score = utility.score(&end_to_end_output, gold_answer);

        Ok(ERagReport {
            per_doc_scores,
            aggregated_score,
            end_to_end_score,
        })
    }

    /// Evaluate a batch of [`ERagCase`]s and compute the rank correlation
    /// between each case's `aggregated_score` and `end_to_end_score`.
    ///
    /// Runs [`evaluate`](Self::evaluate) once per case (in order), then
    /// computes Kendall's tau-b and Spearman's rho between the two resulting
    /// per-case score sequences. This is where eRAG's central claim —
    /// cheap per-document aggregates track expensive end-to-end quality — is
    /// actually testable: a single case produces one aggregate and one
    /// end-to-end score, which has no rank to correlate, so at least two
    /// cases are required.
    ///
    /// # Errors
    ///
    /// - [`ERagError::InsufficientCases`] — `cases.len() < 2`.
    /// - Any error [`evaluate`](Self::evaluate) can return, for the first
    ///   case that triggers it.
    pub fn evaluate_batch(
        &self,
        cases: &[ERagCase],
        task: &dyn DownstreamTask,
        utility: &dyn UtilityMetric,
    ) -> Result<ERagBatchReport, ERagError> {
        if cases.len() < MIN_CASES_FOR_CORRELATION {
            return Err(ERagError::InsufficientCases {
                got: cases.len(),
                need: MIN_CASES_FOR_CORRELATION,
            });
        }

        let mut per_case_reports = Vec::with_capacity(cases.len());
        for case in cases {
            let report = self.evaluate(
                &case.query,
                &case.gold_answer,
                &case.retrieved_docs,
                task,
                utility,
            )?;
            per_case_reports.push(report);
        }

        let aggregated_scores: Vec<f32> = per_case_reports
            .iter()
            .map(|r| r.aggregated_score)
            .collect();
        let end_to_end_scores: Vec<f32> = per_case_reports
            .iter()
            .map(|r| r.end_to_end_score)
            .collect();

        let tau = kendall_tau(&aggregated_scores, &end_to_end_scores);
        let rho = spearman_rho(&aggregated_scores, &end_to_end_scores);

        Ok(ERagBatchReport {
            per_case_reports,
            kendall_tau: tau,
            spearman_rho: rho,
        })
    }
}

// ── Aggregation ───────────────────────────────────────────────────────────────

/// Aggregate `per_doc_scores` via `config.aggregation`.
///
/// Callers guarantee `per_doc_scores` is non-empty (checked in
/// [`ERagEvaluator::evaluate`] via the `EmptyDocs` check, since one score is
/// produced per retrieved document).
#[allow(clippy::cast_precision_loss)]
fn aggregate(per_doc_scores: &[PerDocScore], config: ERagConfig) -> f32 {
    match config.aggregation {
        AggregationMethod::Mean => {
            let sum: f32 = per_doc_scores.iter().map(|s| s.utility).sum();
            sum / per_doc_scores.len() as f32
        }
        AggregationMethod::Max => per_doc_scores
            .iter()
            .map(|s| s.utility)
            .fold(f32::NEG_INFINITY, f32::max),
        AggregationMethod::Sum => per_doc_scores.iter().map(|s| s.utility).sum(),
        AggregationMethod::NdcgWeighted => ndcg_weighted(per_doc_scores, config.ndcg_log_base),
    }
}

/// nDCG-style rank-weighted average of `per_doc_scores`.
///
/// Documents are ranked by their own utility score in descending order (ties
/// broken by ascending original `doc_index`, for determinism); the rank-`r`
/// (0-based) document is weighted by `1 / log_base(r + 2)`. The aggregate is
/// the weighted average `sum(score_r * weight_r) / sum(weight_r)`, mirroring
/// the paper's framing of eRAG as an nDCG-like retrieval metric: the
/// highest-utility documents dominate the aggregate, with a logarithmic
/// discount applied to the rest.
///
/// Callers guarantee `per_doc_scores` is non-empty and `log_base` is finite
/// and `> 1.0` (checked by [`ERagConfig::validate`]).
#[allow(clippy::cast_precision_loss)]
fn ndcg_weighted(per_doc_scores: &[PerDocScore], log_base: f32) -> f32 {
    let mut sorted: Vec<&PerDocScore> = per_doc_scores.iter().collect();
    sorted.sort_by(|a, b| {
        b.utility
            .partial_cmp(&a.utility)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.doc_index.cmp(&b.doc_index))
    });

    let mut weighted_sum = 0.0_f32;
    let mut weight_total = 0.0_f32;
    for (rank, doc) in sorted.iter().enumerate() {
        let weight = 1.0_f32 / (rank as f32 + 2.0_f32).log(log_base);
        weighted_sum += doc.utility * weight;
        weight_total += weight;
    }

    if weight_total <= 0.0 {
        0.0
    } else {
        weighted_sum / weight_total
    }
}
