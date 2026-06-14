//! [`StepBackEngine`] — the Step-Back Prompting heuristic pipeline.

use std::collections::HashSet;

use crate::step_back::types::{StepBackConfig, StepBackError, StepBackModel, StepBackResult};

// ── helpers ────────────────────────────────────────────────────────────────────

/// Tokenize `text` using the canonical Step-Back tokenizer.
///
/// Splits on every non-alphanumeric character, discards tokens shorter than
/// two characters, and lowercases the remainder.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── StepBackEngine ─────────────────────────────────────────────────────────────

/// Drives Step-Back Prompting: the engine iteratively abstracts the query,
/// retrieves documents using the abstracted form, and synthesizes the final
/// answer from the broader context.
///
/// The model is stored inside the engine as a `Box<dyn StepBackModel>`, so the
/// engine owns it for its entire lifetime.
#[derive(Debug)]
pub struct StepBackEngine {
    /// Configuration for this engine.
    pub config: StepBackConfig,
    /// The model used for abstraction and synthesis.
    pub model: Box<dyn StepBackModel>,
}

impl StepBackEngine {
    /// Create a new engine with the given configuration and model.
    #[must_use]
    pub fn new(config: StepBackConfig, model: Box<dyn StepBackModel>) -> Self {
        Self { config, model }
    }

    /// Retrieve the top-[`StepBackConfig::top_k`] documents from `docs` that
    /// best match `abstract_query` by token-overlap.
    ///
    /// Documents are scored by the number of shared tokens with
    /// `abstract_query`, then sorted descending by score and ascending by
    /// original position to break ties stably.
    #[allow(clippy::unnecessary_wraps)]
    fn retrieve_docs(
        &self,
        abstract_query: &str,
        docs: &[String],
    ) -> Result<Vec<String>, StepBackError> {
        if docs.is_empty() || self.config.top_k == 0 {
            return Ok(Vec::new());
        }

        let query_tokens: HashSet<String> = tokenize(abstract_query).into_iter().collect();

        let mut scored: Vec<(usize, usize)> = docs
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let doc_tokens: HashSet<String> = tokenize(doc).into_iter().collect();
                let overlap = query_tokens.intersection(&doc_tokens).count();
                (idx, overlap)
            })
            .collect();

        // Stable descending sort: highest overlap first; ties broken by
        // original insertion order (ascending index).
        scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let result = scored
            .iter()
            .take(self.config.top_k)
            .map(|(idx, _)| docs[*idx].clone())
            .collect();

        Ok(result)
    }

    /// Run the Step-Back Prompting pipeline for `query` against `docs`.
    ///
    /// # Pipeline
    ///
    /// 1. **Abstract** the query up to `max_abstraction_depth` times, stopping
    ///    early when the abstracted query is no longer considered more abstract
    ///    than the current one.  A query is *more abstract* when the new form
    ///    has fewer unique tokens **or** is longer in character count.
    /// 2. **Retrieve** the top-[`StepBackConfig::top_k`] documents by token
    ///    overlap with the final abstract query.
    /// 3. **Synthesize** a final answer from the original query, the abstract
    ///    query, and the retrieved documents.
    ///
    /// # Errors
    ///
    /// - [`StepBackError::AbstractionFailed`] if `query` is empty or the model
    ///   fails to abstract.
    /// - [`StepBackError::RetrievalFailed`] if document retrieval fails.
    /// - [`StepBackError::SynthesisFailed`] if the model fails to synthesize.
    pub fn run(&self, query: &str, docs: &[String]) -> Result<StepBackResult, StepBackError> {
        if query.trim().is_empty() {
            return Err(StepBackError::AbstractionFailed(
                "query must not be empty".to_string(),
            ));
        }

        let mut current_query = query.to_string();
        let mut abstraction_depth: usize = 0;

        for _ in 0..self.config.max_abstraction_depth {
            let new_abstract = self.model.abstract_query(&current_query)?;

            if new_abstract.trim().is_empty() {
                break;
            }

            let current_tokens = tokenize(&current_query);
            let new_tokens = tokenize(&new_abstract);

            // "More abstract" heuristic: fewer unique tokens OR longer query.
            let is_more_abstract =
                new_tokens.len() < current_tokens.len() || new_abstract.len() > current_query.len();

            if is_more_abstract {
                current_query = new_abstract;
                abstraction_depth += 1;
            } else {
                break;
            }
        }

        let retrieved_docs = self.retrieve_docs(&current_query, docs)?;
        let final_answer = self
            .model
            .synthesize(query, &current_query, &retrieved_docs)?;

        Ok(StepBackResult {
            original_query: query.to_string(),
            abstract_query: current_query,
            retrieved_docs,
            final_answer,
            abstraction_depth,
        })
    }
}
