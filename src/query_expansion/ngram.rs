//! N-gram-based query expander.

use std::collections::HashSet;

use async_trait::async_trait;

use crate::types::{Query, SearchResult};

use super::types::{ExpandedQuery, ExpansionConfig, ExpansionMethod, QueryExpander};

/// Expander that generates n-grams from query terms.
#[derive(Debug, Clone)]
pub struct NGramExpander {
    /// Configuration for expansion.
    config: ExpansionConfig,
}

impl Default for NGramExpander {
    fn default() -> Self {
        Self::new(ExpansionConfig::default())
    }
}

impl NGramExpander {
    /// Create a new n-gram expander with the given configuration.
    #[must_use]
    pub fn new(config: ExpansionConfig) -> Self {
        Self { config }
    }

    /// Generate character n-grams for a word.
    #[must_use]
    pub fn char_ngrams(&self, word: &str, n: usize) -> Vec<String> {
        if word.len() < n {
            return vec![word.to_string()];
        }

        word.chars()
            .collect::<Vec<_>>()
            .windows(n)
            .map(|w| w.iter().collect::<String>())
            .collect()
    }

    /// Generate word n-grams for a phrase.
    #[must_use]
    pub fn word_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < n {
            return vec![text.to_string()];
        }

        words.windows(n).map(|w| w.join(" ")).collect()
    }
}

#[async_trait]
impl QueryExpander for NGramExpander {
    async fn expand(&self, query: &Query) -> Vec<Query> {
        let (min_n, max_n) = self.config.ngram_range;
        let mut ngrams: Vec<String> = Vec::new();

        // Generate word n-grams
        for n in min_n..=max_n {
            let word_ngrams = self.word_ngrams(&query.text, n);
            ngrams.extend(word_ngrams);
        }

        // Generate character n-grams for each word (for fuzzy matching)
        let words: Vec<&str> = query.text.split_whitespace().collect();
        for word in &words {
            if word.len() >= 4 {
                let char_ngrams = self.char_ngrams(word, 3);
                ngrams.extend(char_ngrams.into_iter().take(3));
            }
        }

        // Remove duplicates and limit
        let mut seen: HashSet<String> = HashSet::new();
        let unique_ngrams: Vec<String> = ngrams
            .into_iter()
            .filter(|ng| {
                let key = ng.to_lowercase();
                if seen.contains(&key) || key == query.text.to_lowercase() {
                    false
                } else {
                    seen.insert(key);
                    true
                }
            })
            .take(self.config.max_expansions)
            .collect();

        if unique_ngrams.is_empty() {
            return vec![query.clone()];
        }

        let expanded = ExpandedQuery::new(query.clone(), ExpansionMethod::NGrams)
            .with_terms(unique_ngrams)
            .with_weight(0.7);
        vec![expanded.to_query()]
    }

    async fn reformulate(&self, query: &Query, _results: &[SearchResult]) -> Query {
        // For n-grams, we don't really reformulate
        query.clone()
    }
}
