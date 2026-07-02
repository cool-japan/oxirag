//! Core types for the `erag` module: retriever evaluation via per-document
//! downstream utility (Salemi & Zamani, 2024).
//!
//! eRAG evaluates a *retriever* — not the end-to-end RAG pipeline — by running
//! a downstream task on each individually retrieved document, scoring every
//! per-document output against a gold reference answer, and aggregating those
//! scores into a single retriever-quality signal. This file defines the two
//! extension traits ([`DownstreamTask`], [`UtilityMetric`]), the aggregation
//! strategy ([`AggregationMethod`]), the evaluator configuration
//! ([`ERagConfig`]), the result types ([`PerDocScore`], [`ERagReport`],
//! [`ERagCase`], [`ERagBatchReport`]), and the error type ([`ERagError`]).

use thiserror::Error;

// ── DownstreamTask ────────────────────────────────────────────────────────────

/// A downstream task that consumes a query and a context (a set of retrieved
/// documents) and produces a task output (e.g. a generated answer).
///
/// [`ERagEvaluator`](super::evaluator::ERagEvaluator) calls
/// [`run`](Self::run) twice per query: once per individually retrieved
/// document (`context.len() == 1`, to compute each [`PerDocScore`]) and once
/// with the *full* retrieved set (to compute the end-to-end reference score).
/// Implementations must not assume a fixed `context` length.
pub trait DownstreamTask {
    /// Run the task on `query` with the given `context` documents, returning
    /// the task's textual output.
    fn run(&self, query: &str, context: &[String]) -> String;
}

// ── UtilityMetric ─────────────────────────────────────────────────────────────

/// A label-free utility metric that scores a downstream-task output against a
/// gold reference answer.
///
/// Implementations should return a value in `[0.0, 1.0]`, with `1.0` denoting
/// a perfect lexical match against `gold`. See
/// [`RougeLiteUtility`](super::metrics::RougeLiteUtility) for the default
/// token-F1 implementation.
pub trait UtilityMetric {
    /// Score `output` (the downstream task's answer) against `gold` (the gold
    /// reference answer). Higher is better.
    fn score(&self, output: &str, gold: &str) -> f32;
}

// ── AggregationMethod ─────────────────────────────────────────────────────────

/// How the per-document utility scores of [`ERagReport::per_doc_scores`] are
/// aggregated into a single [`ERagReport::aggregated_score`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationMethod {
    /// Arithmetic mean of all per-document utility scores. Default.
    #[default]
    Mean,
    /// Maximum per-document utility score (the single most useful retrieved
    /// document).
    Max,
    /// Sum of all per-document utility scores (rewards larger sets of
    /// uniformly useful documents; unlike the other methods this is not
    /// bounded to `[0.0, 1.0]`).
    Sum,
    /// nDCG-style rank-weighted average. Documents are ranked by their own
    /// utility score in descending order, the rank-`r` (0-based) document is
    /// weighted by `1 / log_base(r + 2)` (see
    /// [`ERagConfig::ndcg_log_base`]), and the aggregate is the weighted
    /// average `sum(score_r * weight_r) / sum(weight_r)`. This mirrors the
    /// paper's framing of eRAG as an nDCG-like retrieval metric: the
    /// highest-utility documents dominate the aggregate, with a logarithmic
    /// discount applied to the rest.
    NdcgWeighted,
}

// ── ERagConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`ERagEvaluator`](super::evaluator::ERagEvaluator).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ERagConfig {
    /// The aggregation method applied to per-document utility scores.
    /// Default: [`AggregationMethod::Mean`].
    pub aggregation: AggregationMethod,
    /// The logarithm base used by [`AggregationMethod::NdcgWeighted`]'s
    /// rank-discount weighting (`1 / log_base(rank + 2)`). Must be finite and
    /// strictly greater than `1.0`, mirroring
    /// [`RetrievalEvalConfig::log_base`](crate::retrieval_eval::RetrievalEvalConfig::log_base).
    /// Default: `2.0`.
    pub ndcg_log_base: f32,
}

impl Default for ERagConfig {
    fn default() -> Self {
        Self {
            aggregation: AggregationMethod::Mean,
            ndcg_log_base: 2.0,
        }
    }
}

impl ERagConfig {
    /// Create a new configuration with the default aggregation
    /// ([`AggregationMethod::Mean`]) and default log base (`2.0`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the aggregation method.
    #[must_use]
    pub fn with_aggregation(mut self, aggregation: AggregationMethod) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Set the nDCG rank-discount log base.
    #[must_use]
    pub fn with_ndcg_log_base(mut self, ndcg_log_base: f32) -> Self {
        self.ndcg_log_base = ndcg_log_base;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ERagError::InvalidConfig`] when `ndcg_log_base` is not
    /// finite or is not strictly greater than `1.0` (a base of `1.0` or below
    /// makes the rank-discount denominator zero, negative, or sign-flipping,
    /// which would make [`AggregationMethod::NdcgWeighted`] ill-defined).
    pub fn validate(&self) -> Result<(), ERagError> {
        if !self.ndcg_log_base.is_finite() || self.ndcg_log_base <= 1.0 {
            return Err(ERagError::InvalidConfig(format!(
                "ndcg_log_base must be finite and > 1.0, got {}",
                self.ndcg_log_base
            )));
        }
        Ok(())
    }
}

// ── PerDocScore ───────────────────────────────────────────────────────────────

/// The per-document downstream utility for a single retrieved document.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerDocScore {
    /// Index of the document within the original `retrieved_docs` slice.
    pub doc_index: usize,
    /// The downstream utility score for running the task on this document
    /// alone, in `[0.0, 1.0]`.
    pub utility: f32,
}

// ── ERagReport ────────────────────────────────────────────────────────────────

/// The result of evaluating a single query's retrieved documents with eRAG.
#[derive(Debug, Clone, PartialEq)]
pub struct ERagReport {
    /// Per-document downstream utility scores, in the original retrieval
    /// order.
    pub per_doc_scores: Vec<PerDocScore>,
    /// The per-document scores aggregated via the evaluator's
    /// [`AggregationMethod`].
    pub aggregated_score: f32,
    /// The downstream task's utility when run once on the *full* retrieved
    /// set (the end-to-end reference score).
    pub end_to_end_score: f32,
}

impl ERagReport {
    /// The [`PerDocScore`] with the highest utility, or `None` if
    /// `per_doc_scores` is empty.
    ///
    /// Ties are broken by the smallest `doc_index` (the earliest-ranked
    /// document among those tied for the highest utility).
    #[must_use]
    pub fn best_doc(&self) -> Option<&PerDocScore> {
        self.per_doc_scores
            .iter()
            .fold(None, |best, candidate| match best {
                None => Some(candidate),
                Some(current_best) if candidate.utility > current_best.utility => Some(candidate),
                _ => best,
            })
    }
}

// ── ERagCase ──────────────────────────────────────────────────────────────────

/// A single query/gold/retrieved-docs case for batch eRAG evaluation and
/// cross-case correlation analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ERagCase {
    /// The natural-language query.
    pub query: String,
    /// The gold reference answer for this query.
    pub gold_answer: String,
    /// The documents retrieved for this query.
    pub retrieved_docs: Vec<String>,
}

impl ERagCase {
    /// Create a new [`ERagCase`].
    #[must_use]
    pub fn new(
        query: impl Into<String>,
        gold_answer: impl Into<String>,
        retrieved_docs: Vec<String>,
    ) -> Self {
        Self {
            query: query.into(),
            gold_answer: gold_answer.into(),
            retrieved_docs,
        }
    }
}

// ── ERagBatchReport ───────────────────────────────────────────────────────────

/// The result of evaluating a batch of [`ERagCase`]s and correlating their
/// aggregated eRAG scores with their end-to-end scores.
///
/// The correlation is the empirical evidence for eRAG's central claim: that
/// cheap, per-document utility aggregation tracks expensive end-to-end
/// quality, but computing it requires *multiple* query observations (a single
/// case has no rank to correlate).
#[derive(Debug, Clone, PartialEq)]
pub struct ERagBatchReport {
    /// The per-case [`ERagReport`], in the same order as the input `cases`.
    pub per_case_reports: Vec<ERagReport>,
    /// Kendall's tau-b rank correlation between `aggregated_score` and
    /// `end_to_end_score` across all cases, in `[-1.0, 1.0]`.
    pub kendall_tau: f32,
    /// Spearman's rho rank correlation between `aggregated_score` and
    /// `end_to_end_score` across all cases, in `[-1.0, 1.0]`.
    pub spearman_rho: f32,
}

// ── ERagError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `erag` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ERagError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query is empty")]
    EmptyQuery,
    /// The retrieved-document set was empty.
    #[error("retrieved document set is empty")]
    EmptyDocs,
    /// The gold reference answer was empty or contained only whitespace.
    #[error("gold answer is empty")]
    EmptyGold,
    /// The supplied [`ERagConfig`] failed validation.
    #[error("invalid eRAG configuration: {0}")]
    InvalidConfig(String),
    /// Cross-case correlation ([`ERagEvaluator::evaluate_batch`](super::evaluator::ERagEvaluator::evaluate_batch))
    /// requires at least two cases; fewer were supplied.
    #[error("insufficient cases for correlation: got {got}, need at least {need}")]
    InsufficientCases {
        /// Number of cases actually supplied.
        got: usize,
        /// Minimum number of cases required (always `2`).
        need: usize,
    },
}
