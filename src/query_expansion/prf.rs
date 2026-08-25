//! Pseudo-relevance feedback query expander.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::types::{Query, SearchResult};

use super::types::{ExpansionConfig, QueryExpander};

/// Pseudo-relevance feedback expander.
///
/// Uses the top results from an initial search to expand the query
/// with terms that appear frequently in relevant documents.
#[derive(Debug, Clone)]
pub struct PseudoRelevanceFeedback {
    /// Configuration for expansion.
    pub(super) config: ExpansionConfig,
    /// Stop words to exclude from expansion.
    pub(super) stop_words: HashSet<String>,
}

impl Default for PseudoRelevanceFeedback {
    fn default() -> Self {
        Self::new(ExpansionConfig::default())
    }
}

impl PseudoRelevanceFeedback {
    /// Create a new PRF expander with the given configuration.
    #[must_use]
    pub fn new(config: ExpansionConfig) -> Self {
        Self {
            config,
            stop_words: Self::default_stop_words(),
        }
    }

    /// Get default stop words.
    pub(super) fn default_stop_words() -> HashSet<String> {
        [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "it", "its", "they", "them", "their", "we", "our", "you",
            "your", "he", "she", "him", "her", "his", "i", "me", "my", "not", "no", "yes", "if",
            "then", "else", "when", "where", "what", "which", "who", "whom", "how", "why", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "here", "there",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    /// Check if a word is a stop word.
    #[must_use]
    pub fn is_stop_word(&self, word: &str) -> bool {
        self.stop_words.contains(&word.to_lowercase())
    }

    /// Extract top terms from search results.
    pub fn extract_top_terms(&self, results: &[SearchResult], query: &Query) -> Vec<(String, f32)> {
        let mut term_scores: HashMap<String, f32> = HashMap::new();
        let query_words: HashSet<String> = query
            .text
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        let num_docs = results.len().min(self.config.prf_documents);

        for (i, result) in results.iter().take(num_docs).enumerate() {
            // Weight terms by document rank (higher rank = higher weight)
            #[allow(clippy::cast_precision_loss)]
            let doc_weight = 1.0 / (i + 1) as f32;

            for word in result.document.content.to_lowercase().split_whitespace() {
                let word = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string();

                if word.len() > 2 && !self.is_stop_word(&word) && !query_words.contains(&word) {
                    *term_scores.entry(word).or_insert(0.0) += doc_weight * result.score;
                }
            }
        }

        let mut sorted_terms: Vec<_> = term_scores.into_iter().collect();
        sorted_terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_terms
            .into_iter()
            .take(self.config.max_expansions)
            .collect()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl QueryExpander for PseudoRelevanceFeedback {
    async fn expand(&self, query: &Query) -> Vec<Query> {
        // Without results, we can't do PRF expansion
        vec![query.clone()]
    }

    async fn reformulate(&self, query: &Query, results: &[SearchResult]) -> Query {
        if results.is_empty() {
            return query.clone();
        }

        let top_terms = self.extract_top_terms(results, query);
        if top_terms.is_empty() {
            return query.clone();
        }

        let expansion_terms: Vec<String> = top_terms.into_iter().map(|(term, _)| term).collect();

        let mut new_query = query.clone();
        new_query.text = format!("{} {}", query.text, expansion_terms.join(" "));
        new_query
    }
}
