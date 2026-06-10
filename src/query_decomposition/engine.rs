//! Query decomposition engine.

use crate::advanced_retrieval::rag_fusion::reciprocal_rank_fusion;
use crate::types::SearchResult;

use super::decomposer::QueryDecomposer;
use super::types::{DecomposedQuery, DecompositionConfig, QueryDecompositionError, SubAnswer};

// ── answer helper ─────────────────────────────────────────────────────────────

/// Heuristic: derive an answer from retrieved documents.
fn answer_from_results(sub_q: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No information found for: {sub_q}");
    }
    // Take the top document's first 200 chars as the answer
    let snippet: String = results[0].document.content.chars().take(200).collect();
    snippet
}

// ── QueryDecompositionEngine ──────────────────────────────────────────────────

/// Decomposes a complex query into sub-questions, retrieves answers for each,
/// and recombines via Reciprocal Rank Fusion.
///
/// The [`Echo`] layer is provided *per call* through the generic `run<E>` method
/// so that the struct remains `Send + Sync` without holding a reference to an
/// unsized trait object.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
pub struct QueryDecompositionEngine {
    decomposer: QueryDecomposer,
    config: DecompositionConfig,
}

impl QueryDecompositionEngine {
    /// Create a new engine.
    #[must_use]
    pub fn new(config: DecompositionConfig) -> Self {
        let decomposer = QueryDecomposer::new(config.max_sub_questions);
        Self { decomposer, config }
    }

    /// Decompose `query`, retrieve sub-answers, and fuse results.
    ///
    /// # Errors
    ///
    /// Returns [`QueryDecompositionError::EmptyQuery`] when `query` is blank.
    /// Returns [`QueryDecompositionError::RetrievalFailed`] when Echo search fails.
    pub async fn run<E>(
        &self,
        query: &str,
        echo: &E,
    ) -> Result<DecomposedQuery, QueryDecompositionError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        let query_trimmed = query.trim();
        if query_trimmed.is_empty() {
            return Err(QueryDecompositionError::EmptyQuery);
        }

        let sub_questions = self
            .decomposer
            .decompose(query_trimmed, self.config.strategy);

        let mut result_lists: Vec<Vec<SearchResult>> = Vec::new();
        let mut sub_answers: Vec<SubAnswer> = Vec::new();

        for sub_q in &sub_questions {
            let results = echo
                .search(&sub_q.text, self.config.top_k, None)
                .await
                .map_err(|e| QueryDecompositionError::RetrievalFailed(e.to_string()))?;

            let answer = answer_from_results(&sub_q.text, &results);
            result_lists.push(results.clone());
            sub_answers.push(SubAnswer {
                question: sub_q.clone(),
                answer,
                results,
            });
        }

        // Fuse via RRF
        let fused_results = reciprocal_rank_fusion(&result_lists, self.config.rrf_k);

        // Compose final answer from sub-answers
        let final_answer = sub_answers
            .iter()
            .map(|sa| sa.answer.clone())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(DecomposedQuery {
            original_query: query_trimmed.to_string(),
            sub_questions,
            sub_answers,
            final_answer,
            fused_results,
        })
    }
}
