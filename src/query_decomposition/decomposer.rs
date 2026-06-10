//! Heuristic query decomposer for the `query_decomposition` module.

use super::types::{DecompositionStrategy, SubQuestion};

// ── QueryDecomposer ───────────────────────────────────────────────────────────

/// Heuristic query decomposer.
///
/// Splits a query into sub-questions using clause/conjunction boundary detection.
/// Always produces at least one sub-question (falls back to the whole query when
/// no split points are found).
#[derive(Debug, Clone, Default)]
pub struct QueryDecomposer {
    /// Maximum number of sub-questions to emit.
    max_sub_questions: usize,
}

impl QueryDecomposer {
    /// Create a new decomposer with the given `max_sub_questions` limit.
    #[must_use]
    pub fn new(max_sub_questions: usize) -> Self {
        Self { max_sub_questions }
    }

    /// Decompose `query` using the given `strategy`.
    ///
    /// Always returns at least one [`SubQuestion`].
    #[must_use]
    pub fn decompose(&self, query: &str, strategy: DecompositionStrategy) -> Vec<SubQuestion> {
        match strategy {
            DecompositionStrategy::Parallel => self.parallel_split(query),
            DecompositionStrategy::LeastToMost => self.least_to_most_split(query),
            DecompositionStrategy::StepBack => self.step_back_split(query),
        }
    }

    // Parallel: split on conjunctions and comparisons
    fn parallel_split(&self, query: &str) -> Vec<SubQuestion> {
        let lower = query.to_lowercase();

        // Try splitting on common multi-sub-question signals
        let separators = [
            " and ",
            " or ",
            ", and ",
            ", or ",
            " vs ",
            " versus ",
            " compare ",
            " between ",
            " also ",
            " as well as ",
        ];

        let mut parts: Vec<String> = Vec::new();
        let remaining = query.to_string();

        for sep in &separators {
            if lower.contains(sep) {
                // Split once on this separator
                if let Some(pos) = remaining.to_lowercase().find(sep) {
                    let left = remaining[..pos].trim().to_string();
                    let right = remaining[pos + sep.len()..].trim().to_string();
                    if !left.is_empty() {
                        parts.push(left);
                    }
                    if !right.is_empty() {
                        parts.push(right);
                    }
                    break; // stop after first successful split
                }
            }
        }

        if parts.is_empty() {
            // No split found: emit the full query as one sub-question
            parts.push(remaining);
        }

        // Respect max_sub_questions
        parts.truncate(self.max_sub_questions.max(1));
        parts
            .into_iter()
            .enumerate()
            .map(|(i, text)| SubQuestion { text, index: i })
            .collect()
    }

    // Least-to-most: emit the whole query plus a simplified version first
    fn least_to_most_split(&self, query: &str) -> Vec<SubQuestion> {
        // Heuristic: if query has a "how" or "why" clause, extract the "what" prerequisite
        let lower = query.to_lowercase();
        let mut parts: Vec<String> = Vec::new();

        if lower.contains("how") || lower.contains("why") {
            // Simpler prereq: replace "how/why ... is/are" with "what is"
            let simple = format!(
                "What is {}",
                query
                    .split_whitespace()
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            parts.push(simple);
        }
        parts.push(query.to_string());
        parts.truncate(self.max_sub_questions.max(1));
        parts
            .into_iter()
            .enumerate()
            .map(|(i, text)| SubQuestion { text, index: i })
            .collect()
    }

    // Step-back: generate an abstract version first, then the specific query
    fn step_back_split(&self, query: &str) -> Vec<SubQuestion> {
        // Abstract question: strip specifics (named entities) to get broader concept
        let abstract_q = format!(
            "What are the general principles related to: {}",
            query
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ")
        );
        let mut parts = vec![abstract_q, query.to_string()];
        parts.truncate(self.max_sub_questions.max(1));
        parts
            .into_iter()
            .enumerate()
            .map(|(i, text)| SubQuestion { text, index: i })
            .collect()
    }
}
