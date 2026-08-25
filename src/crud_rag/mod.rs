//! CRUD-RAG — evaluating RAG systems across the four CRUD operation types.
//!
//! Implements the taxonomy of Lyu et al. 2024, *CRUD-RAG: A Comprehensive
//! Chinese Benchmark for Retrieval-Augmented Generation of Large Language
//! Models*, adapted here as a general-purpose, language-agnostic lexical
//! harness. Unlike [`crate::rgb_eval`], which partitions RAG evaluation into
//! the RGB benchmark's four **abilities** (noise robustness, negative
//! rejection, information integration, counterfactual robustness — all
//! variations on "did the answer come out right given a tricky context"),
//! `crud_rag` partitions evaluation into the four **CRUD operation types** a
//! RAG system is asked to perform, each scored by metric(s) suited to that
//! operation's own notion of "correct":
//!
//! | Operation | RAG task it models | Metrics |
//! |-----------|---------------------|---------|
//! | [`CrudOperation::Create`] | continuation / creative generation | [`CrudMetric::RougeL`] (LCS F-beta) + [`CrudMetric::Bleu`] (clipped n-gram precision, brevity-penalized) |
//! | [`CrudOperation::Read`] | single-document question answering | [`CrudMetric::ExactMatch`] + [`CrudMetric::TokenF1`] |
//! | [`CrudOperation::Update`] | hallucination / error correction | [`CrudMetric::CorrectionSimilarity`] + [`CrudMetric::ErrorRemoval`] + [`CrudMetric::CorrectSpanIntroduced`] |
//! | [`CrudOperation::Delete`] | multi-document summarization / redundancy removal | [`CrudMetric::Coverage`] minus weighted [`CrudMetric::Redundancy`] |
//!
//! Every operation additionally reports a [`CrudMetric::Combined`] score —
//! the operation-specific primary number that [`CrudRagReport`] averages
//! into its per-operation means and overall score. Scoring is entirely
//! lexical (tokenization, longest-common-subsequence, n-gram counting, set
//! overlap) with configurable normalization
//! ([`CrudRagConfig::lowercase`], [`CrudRagConfig::strip_punctuation`]) — no
//! LLM, randomness, or ML model is involved, so a
//! [`CrudRagHarness::evaluate`] run is fully deterministic and reproducible.
//!
//! # Quick start
//!
//! ```rust
//! use oxirag::crud_rag::{CrudCase, CrudOperation, CrudRagConfig, CrudRagHarness};
//!
//! let harness = CrudRagHarness::new(CrudRagConfig::default());
//!
//! let case = CrudCase::new("read-1", CrudOperation::Read, "Paris")
//!     .with_reference("Paris");
//!
//! let report = harness.evaluate(&[case]).expect("evaluation should succeed");
//! assert_eq!(report.read_mean, 1.0);
//! assert_eq!(report.overall, 1.0);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::CrudRagHarness;
pub use types::{
    CrudCase, CrudMetric, CrudOperation, CrudRagConfig, CrudRagError, CrudRagReport, CrudRagResult,
    CrudScore,
};
