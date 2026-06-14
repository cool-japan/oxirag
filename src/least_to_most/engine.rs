//! [`LtmEngine`] — the Least-to-Most Prompting sequential solver.

use std::cmp::Reverse;
use std::collections::HashSet;

use crate::least_to_most::types::{
    LtmConfig, LtmError, LtmResult, LtmSolver, SubProblem, tokenize,
};

// ── LtmEngine ─────────────────────────────────────────────────────────────────

/// Drives the Least-to-Most Prompting pipeline (Zhou et al. 2022).
///
/// The engine:
/// 1. Calls [`LtmSolver::decompose`] to obtain an ordered list of
///    sub-questions.
/// 2. Returns [`LtmError::SubproblemLimitExceeded`] if the count exceeds
///    [`LtmConfig::max_subproblems`].
/// 3. Iterates from the simplest to the most complex sub-question, forwarding
///    all previously solved [`SubProblem`]s and the top context documents
///    (selected by token overlap with the current sub-question) to
///    [`LtmSolver::solve_step`].
/// 4. Returns an [`LtmResult`] with the original query, the full chain of
///    solved sub-problems, and the final answer (the last solve result).
#[derive(Debug)]
pub struct LtmEngine {
    /// Configuration for this engine.
    pub config: LtmConfig,

    /// The solver that performs decomposition and step solving.
    pub solver: Box<dyn LtmSolver>,
}

impl LtmEngine {
    /// Create a new engine with the given configuration and solver.
    #[must_use]
    pub fn new(config: LtmConfig, solver: Box<dyn LtmSolver>) -> Self {
        Self { config, solver }
    }

    /// Run Least-to-Most Prompting on `query`, using `docs` as the context
    /// pool.
    ///
    /// 1. Decomposes `query` into an ordered list of sub-questions.
    /// 2. Returns [`LtmError::SubproblemLimitExceeded`] (carrying the actual
    ///    sub-problem count) when the count exceeds
    ///    [`LtmConfig::max_subproblems`].
    /// 3. Solves each sub-question in order, forwarding the accumulated
    ///    [`SubProblem`] list and the top `max_context_docs` documents ranked
    ///    by token overlap with the current sub-question.
    /// 4. Returns an [`LtmResult`] with the full chain and the final answer
    ///    (the last solve result).
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`LtmError::DecompositionFailed`] if decomposition fails.
    /// - [`LtmError::SubproblemLimitExceeded`] if the sub-problem count exceeds
    ///   the configured maximum.
    /// - [`LtmError::SolveFailed`] if any solve step fails.
    pub fn run(&self, query: &str, docs: &[String]) -> Result<LtmResult, LtmError> {
        let sub_questions = self.solver.decompose(query)?;

        if sub_questions.len() > self.config.max_subproblems {
            return Err(LtmError::SubproblemLimitExceeded(sub_questions.len()));
        }

        let mut subproblems: Vec<SubProblem> = Vec::with_capacity(sub_questions.len());

        for (step_index, sub_question) in sub_questions.iter().enumerate() {
            let context_docs = self.select_context_docs(sub_question, docs);
            let answer = self
                .solver
                .solve_step(sub_question, &subproblems, &context_docs)?;

            subproblems.push(SubProblem {
                question: sub_question.clone(),
                answer,
                step_index,
            });
        }

        let final_answer = subproblems
            .last()
            .map(|sp| sp.answer.clone())
            .unwrap_or_default();

        Ok(LtmResult {
            original_query: query.to_string(),
            subproblems,
            final_answer,
        })
    }

    /// Select the top `max_context_docs` documents by descending token overlap
    /// with `question`.
    ///
    /// Ties are resolved by the document's original position (stable sort
    /// preserves insertion order for equal scores).
    fn select_context_docs(&self, question: &str, docs: &[String]) -> Vec<String> {
        if docs.is_empty() || self.config.max_context_docs == 0 {
            return Vec::new();
        }

        let q_tokens: HashSet<String> = tokenize(question).into_iter().collect();

        let mut scored: Vec<(usize, usize)> = docs
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let doc_tokens: HashSet<String> = tokenize(doc).into_iter().collect();
                let overlap = q_tokens.intersection(&doc_tokens).count();
                (idx, overlap)
            })
            .collect();

        // Stable descending sort: higher overlap first; ties keep original order.
        scored.sort_by_key(|&(_, overlap)| Reverse(overlap));
        scored.truncate(self.config.max_context_docs);
        // Restore original document order within the selected subset.
        scored.sort_by_key(|&(idx, _)| idx);

        scored
            .into_iter()
            .map(|(idx, _)| docs[idx].clone())
            .collect()
    }
}
