//! Query reformulator for query refinement and decomposition.

use std::collections::HashSet;

use crate::types::Query;

use super::prf::PseudoRelevanceFeedback;

/// Query reformulator for query refinement and decomposition.
#[derive(Debug, Clone)]
pub struct QueryReformulator {
    /// Stop words for simplification.
    stop_words: HashSet<String>,
}

impl Default for QueryReformulator {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryReformulator {
    /// Create a new query reformulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stop_words: PseudoRelevanceFeedback::default_stop_words(),
        }
    }

    /// Simplify a query by removing noise words and focusing on key terms.
    #[must_use]
    pub fn simplify(&self, query: &Query) -> Query {
        let key_words: Vec<&str> = query
            .text
            .split_whitespace()
            .filter(|w| {
                let word = w.to_lowercase();
                !self.stop_words.contains(&word) && word.len() > 1
            })
            .collect();

        let mut new_query = query.clone();
        new_query.text = if key_words.is_empty() {
            query.text.clone()
        } else {
            key_words.join(" ")
        };
        new_query
    }

    /// Clarify a query by adding context.
    #[must_use]
    pub fn clarify(&self, query: &Query, context: &str) -> Query {
        let mut new_query = query.clone();
        if !context.is_empty() {
            new_query.text = format!("{} {}", query.text, context);
        }
        new_query
    }

    /// Decompose a complex query into simpler sub-queries.
    #[must_use]
    pub fn decompose(&self, query: &Query) -> Vec<Query> {
        let text = &query.text;
        let mut sub_queries = Vec::new();

        // Split on common conjunctions and question markers
        let split_patterns = [
            " and ",
            " or ",
            " but ",
            " versus ",
            " vs ",
            " compared to ",
            "? ",
            ". ",
            "; ",
            " - ",
        ];

        let mut segments = vec![text.clone()];
        for pattern in &split_patterns {
            let mut new_segments = Vec::new();
            for segment in segments {
                new_segments.extend(
                    segment
                        .split(pattern)
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
            }
            segments = new_segments;
        }

        // Create sub-queries for each segment
        for segment in segments {
            if segment.len() > 2 {
                let mut sub_query = query.clone();
                sub_query.text = segment;
                sub_queries.push(sub_query);
            }
        }

        // If no decomposition happened, return the original
        if sub_queries.is_empty() {
            sub_queries.push(query.clone());
        }

        sub_queries
    }

    /// Rephrase a query as a different question type.
    #[must_use]
    pub fn rephrase_as_question(&self, query: &Query) -> Query {
        let text = query.text.trim();
        let mut new_query = query.clone();

        // If already a question, return as-is
        if text.ends_with('?') {
            return new_query;
        }

        // Try to rephrase as a question
        let lower = text.to_lowercase();

        // If already starts with a question pattern, just add '?'
        // Otherwise, wrap with "What is ...?"
        let rephrased = if lower.starts_with("how to ")
            || lower.starts_with("what is ")
            || lower.starts_with("what are ")
        {
            text.to_string() + "?"
        } else {
            format!("What is {text}?")
        };

        new_query.text = rephrased;
        new_query
    }
}
