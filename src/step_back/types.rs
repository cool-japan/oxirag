//! Types and traits for the `step_back` module.
//!
//! Step-Back Prompting (Zheng et al. 2023) decomposes the retrieval process
//! into two phases: first the model *abstracts* a specific question into a
//! higher-level form, then retrieval is performed on the abstracted form and
//! the answer is synthesized from the broader context.
//!
//! The model is supplied by the caller as the [`StepBackModel`] trait; a
//! deterministic [`MockStepBackModel`] is provided for testing.

use std::fmt;

use thiserror::Error;

// ── StepBackConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`StepBackEngine`](crate::step_back::engine::StepBackEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepBackConfig {
    /// Maximum number of abstraction iterations.
    ///
    /// Defaults to `2`. The engine calls [`StepBackModel::abstract_query`] at
    /// most this many times, stopping early when the query is no longer
    /// considered more abstract than the previous iteration.
    pub max_abstraction_depth: usize,

    /// Maximum number of documents to retrieve for the abstract query.
    ///
    /// Defaults to `5`.
    pub top_k: usize,
}

impl Default for StepBackConfig {
    fn default() -> Self {
        Self {
            max_abstraction_depth: 2,
            top_k: 5,
        }
    }
}

impl StepBackConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of abstraction iterations.
    #[must_use]
    pub fn with_max_abstraction_depth(mut self, depth: usize) -> Self {
        self.max_abstraction_depth = depth;
        self
    }

    /// Set the maximum number of documents to retrieve.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

// ── StepBackError ──────────────────────────────────────────────────────────────

/// Errors from the `step_back` module.
#[derive(Debug, Error)]
pub enum StepBackError {
    /// The abstraction step failed (e.g., empty query or model error).
    #[error("abstraction failed: {0}")]
    AbstractionFailed(String),

    /// The document retrieval step failed.
    #[error("retrieval failed: {0}")]
    RetrievalFailed(String),

    /// The synthesis step failed.
    #[error("synthesis failed: {0}")]
    SynthesisFailed(String),
}

// ── StepBackResult ─────────────────────────────────────────────────────────────

/// The full record of a Step-Back Prompting run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepBackResult {
    /// The original (specific) query supplied by the caller.
    pub original_query: String,

    /// The abstracted query produced by the heuristic pipeline.
    pub abstract_query: String,

    /// Documents retrieved using the abstract query, ordered by descending
    /// token-overlap relevance.
    pub retrieved_docs: Vec<String>,

    /// The final synthesized answer.
    pub final_answer: String,

    /// The number of successful abstraction iterations performed.
    pub abstraction_depth: usize,
}

// ── StepBackModel ──────────────────────────────────────────────────────────────

/// A model that can abstract questions and synthesize answers for
/// Step-Back Prompting.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies
/// a concrete implementation (e.g. wrapping an LLM); [`MockStepBackModel`] is
/// provided for testing.
///
/// # Object Safety
///
/// This trait is object-safe and may be stored as `Box<dyn StepBackModel>`.
pub trait StepBackModel: fmt::Debug {
    /// Abstract a specific question into a higher-level, more general form.
    ///
    /// For example, `"What is the speed of light?"` might become
    /// `"What are the fundamental constants of physics?"`.
    ///
    /// # Errors
    ///
    /// Returns [`StepBackError::AbstractionFailed`] if the query cannot be
    /// abstracted.
    fn abstract_query(&self, query: &str) -> Result<String, StepBackError>;

    /// Synthesize a final answer from the original query, the abstract query,
    /// and the retrieved documents.
    ///
    /// # Errors
    ///
    /// Returns [`StepBackError::SynthesisFailed`] if synthesis fails.
    fn synthesize(
        &self,
        original: &str,
        abstract_query: &str,
        docs: &[String],
    ) -> Result<String, StepBackError>;
}

// ── MockStepBackModel ──────────────────────────────────────────────────────────

/// Deterministic [`StepBackModel`] for tests.
///
/// [`StepBackModel::abstract_query`] prepends `"In general terms, "` to the
/// query. [`StepBackModel::synthesize`] joins all retrieved documents with a
/// single space (returning the empty string when no documents are provided).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MockStepBackModel;

impl StepBackModel for MockStepBackModel {
    fn abstract_query(&self, query: &str) -> Result<String, StepBackError> {
        Ok(format!("In general terms, {query}"))
    }

    fn synthesize(
        &self,
        _original: &str,
        _abstract_query: &str,
        docs: &[String],
    ) -> Result<String, StepBackError> {
        Ok(docs.join(" "))
    }
}
