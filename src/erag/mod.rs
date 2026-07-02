//! eRAG: retriever evaluation via per-document downstream utility
//! (Salemi & Zamani, 2024).
//!
//! Most RAG evaluations score the *end-to-end* pipeline — the final answer
//! produced after generation — which conflates retrieval quality with
//! generation quality and requires expensive full-pipeline runs to get any
//! signal at all. eRAG instead evaluates the **retriever alone**: it runs the
//! downstream task *separately* on each individually retrieved document (not
//! the whole retrieved set together), scores every per-document output
//! against a gold answer, and aggregates the resulting per-document utility
//! scores into a single retriever-quality signal. The paper shows this
//! per-document aggregate correlates strongly with end-to-end quality while
//! being far more sample-efficient (`N` documents give `N` utility
//! observations from one query, instead of one observation from one
//! full-pipeline run).
//!
//! ## Algorithm
//!
//! 1. For each retrieved document `i`, run the [`DownstreamTask`] with a
//!    context of exactly that one document, then score the output against the
//!    gold answer with a [`UtilityMetric`] to get `per_doc_scores[i]`
//!    ([`PerDocScore`]).
//! 2. Aggregate `per_doc_scores` via a configurable [`AggregationMethod`]
//!    (`Mean`, `Max`, `Sum`, or `NdcgWeighted`) into `aggregated_score`.
//! 3. Separately, run the [`DownstreamTask`] once with the *full* retrieved
//!    set as context, and score that output the same way, to get
//!    `end_to_end_score` — the reference the aggregate is meant to
//!    approximate.
//! 4. Across a **batch** of query cases (a single case has no rank to
//!    correlate), compute Kendall's tau-b and Spearman's rho between each
//!    case's `aggregated_score` and `end_to_end_score`
//!    ([`ERagEvaluator::evaluate_batch`]) — this is where the paper's
//!    correlation claim is actually testable.
//!
//! ## Distinct from `retrieval_eval`, `ares_eval`, and `ragchecker`
//!
//! | Module | Ground truth needed | Unit of evaluation | Central technique |
//! |--------|---------------------|---------------------|--------------------|
//! | [`retrieval_eval`](crate::retrieval_eval) | Graded relevance labels (qrels) per document | Whole ranked result list | Classic IR metrics (nDCG, MAP, MRR, P@k, R@k) computed directly from labels — no downstream task is ever run |
//! | [`ares_eval`](crate::ares_eval) | A *small* labeled subset alongside a large judge-labeled set | A judge's rate prediction (e.g. context relevance) | Prediction-powered inference (PPI): debias the judge's mean on unlabeled data using the rectifier measured on the labeled subset |
//! | [`ragchecker`](crate::ragchecker) | A gold answer, decomposed into claims | Individual atomic claims of the response/ground truth | Lexical entailment of each claim against context and ground truth, split into retriever- vs generator-side metrics |
//! | **`erag`** | A gold reference answer (whole-answer, not claim-level or relevance-graded) | Each individually retrieved document, run standalone through the downstream task | Per-document downstream-task utility, aggregated and rank-correlated against the end-to-end run |
//!
//! `erag` never consults a relevance-label set (unlike `retrieval_eval`),
//! never mixes labeled and unlabeled judge predictions (unlike `ares_eval`),
//! and never decomposes text into claims (unlike `ragchecker`); its
//! distinguishing move is running the downstream task once *per retrieved
//! document* and comparing that per-document signal to a full-context
//! reference run.
//!
//! ## Example
//!
//! ```
//! use oxirag::erag::{ERagConfig, ERagEvaluator, MockDownstreamTask, RougeLiteUtility};
//!
//! let evaluator = ERagEvaluator::new(ERagConfig::default());
//! let task = MockDownstreamTask::new();
//! let utility = RougeLiteUtility::new();
//!
//! let docs = vec![
//!     "The Eiffel Tower is located in Paris, France.".to_string(),
//!     "Bananas are a good source of potassium.".to_string(),
//! ];
//!
//! let report = evaluator
//!     .evaluate(
//!         "Where is the Eiffel Tower?",
//!         "The Eiffel Tower is in Paris.",
//!         &docs,
//!         &task,
//!         &utility,
//!     )
//!     .expect("non-empty query, docs, and gold answer");
//!
//! // The relevant document (index 0) scores higher than the irrelevant one.
//! assert!(report.per_doc_scores[0].utility > report.per_doc_scores[1].utility);
//! assert!(report.aggregated_score >= 0.0 && report.aggregated_score <= 1.0);
//! assert!(report.end_to_end_score >= 0.0 && report.end_to_end_score <= 1.0);
//! ```

pub mod correlation;
pub mod evaluator;
pub mod metrics;
#[cfg(test)]
mod tests;
pub mod types;

pub use correlation::{kendall_tau, spearman_rho};
pub use evaluator::ERagEvaluator;
pub use metrics::{MockDownstreamTask, RougeLiteUtility};
pub use types::{
    AggregationMethod, DownstreamTask, ERagBatchReport, ERagCase, ERagConfig, ERagError,
    ERagReport, PerDocScore, UtilityMetric,
};
