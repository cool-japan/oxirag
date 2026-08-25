//! Synonym-based query expander.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::types::{Query, SearchResult};

use super::types::{ExpandedQuery, ExpansionConfig, ExpansionMethod, QueryExpander};

/// Expander that adds synonyms from a built-in dictionary.
#[derive(Debug, Clone)]
pub struct SynonymExpander {
    /// Configuration for expansion.
    pub(super) config: ExpansionConfig,
    /// Synonym dictionary mapping words to their synonyms.
    dictionary: HashMap<String, Vec<String>>,
}

impl Default for SynonymExpander {
    fn default() -> Self {
        Self::new(ExpansionConfig::default())
    }
}

impl SynonymExpander {
    /// Create a new synonym expander with the given configuration.
    #[must_use]
    pub fn new(config: ExpansionConfig) -> Self {
        Self {
            config,
            dictionary: Self::build_default_dictionary(),
        }
    }

    /// Create a synonym expander with a custom dictionary.
    #[must_use]
    pub fn with_dictionary(
        config: ExpansionConfig,
        dictionary: HashMap<String, Vec<String>>,
    ) -> Self {
        Self { config, dictionary }
    }

    /// Add a synonym mapping.
    pub fn add_synonym(&mut self, word: impl Into<String>, synonyms: Vec<String>) {
        self.dictionary.insert(word.into().to_lowercase(), synonyms);
    }

    /// Get synonyms for a word.
    #[must_use]
    pub fn get_synonyms(&self, word: &str) -> Vec<String> {
        self.dictionary
            .get(&word.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// Build the default synonym dictionary.
    #[allow(clippy::too_many_lines)]
    fn build_default_dictionary() -> HashMap<String, Vec<String>> {
        let mut dict = HashMap::new();

        // Common technical synonyms
        dict.insert(
            "search".to_string(),
            vec![
                "find".to_string(),
                "query".to_string(),
                "lookup".to_string(),
            ],
        );
        dict.insert(
            "machine".to_string(),
            vec![
                "computer".to_string(),
                "device".to_string(),
                "system".to_string(),
            ],
        );
        dict.insert(
            "learning".to_string(),
            vec!["training".to_string(), "education".to_string()],
        );
        dict.insert(
            "algorithm".to_string(),
            vec![
                "method".to_string(),
                "procedure".to_string(),
                "technique".to_string(),
            ],
        );
        dict.insert(
            "data".to_string(),
            vec!["information".to_string(), "records".to_string()],
        );
        dict.insert(
            "fast".to_string(),
            vec![
                "quick".to_string(),
                "rapid".to_string(),
                "speedy".to_string(),
            ],
        );
        dict.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "massive".to_string(),
            ],
        );
        dict.insert(
            "small".to_string(),
            vec![
                "tiny".to_string(),
                "little".to_string(),
                "compact".to_string(),
            ],
        );
        dict.insert(
            "error".to_string(),
            vec![
                "mistake".to_string(),
                "fault".to_string(),
                "bug".to_string(),
            ],
        );
        dict.insert(
            "function".to_string(),
            vec![
                "method".to_string(),
                "procedure".to_string(),
                "routine".to_string(),
            ],
        );
        dict.insert(
            "create".to_string(),
            vec![
                "make".to_string(),
                "build".to_string(),
                "generate".to_string(),
            ],
        );
        dict.insert(
            "delete".to_string(),
            vec![
                "remove".to_string(),
                "erase".to_string(),
                "drop".to_string(),
            ],
        );
        dict.insert(
            "update".to_string(),
            vec![
                "modify".to_string(),
                "change".to_string(),
                "edit".to_string(),
            ],
        );
        dict.insert(
            "retrieve".to_string(),
            vec!["fetch".to_string(), "get".to_string(), "obtain".to_string()],
        );
        dict.insert(
            "store".to_string(),
            vec![
                "save".to_string(),
                "keep".to_string(),
                "persist".to_string(),
            ],
        );
        dict.insert(
            "process".to_string(),
            vec![
                "handle".to_string(),
                "manage".to_string(),
                "execute".to_string(),
            ],
        );

        dict
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl QueryExpander for SynonymExpander {
    async fn expand(&self, query: &Query) -> Vec<Query> {
        let words: Vec<&str> = query.text.split_whitespace().collect();
        let mut expanded_queries = Vec::new();

        // Create the main expanded query with all synonyms
        let mut all_synonyms: Vec<String> = Vec::new();
        for word in &words {
            let synonyms = self.get_synonyms(word);
            let limited = synonyms
                .into_iter()
                .take(self.config.max_expansions)
                .collect::<Vec<_>>();
            all_synonyms.extend(limited);
        }

        if !all_synonyms.is_empty() {
            let expanded = ExpandedQuery::new(query.clone(), ExpansionMethod::Synonyms)
                .with_terms(all_synonyms)
                .with_weight(self.config.synonym_weight);
            expanded_queries.push(expanded.to_query());
        }

        // Always include the original query
        if expanded_queries.is_empty() {
            expanded_queries.push(query.clone());
        }

        expanded_queries
    }

    async fn reformulate(&self, query: &Query, results: &[SearchResult]) -> Query {
        // Extract key terms from top results and add as synonyms
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        let query_words: HashSet<String> = query
            .text
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        for result in results.iter().take(self.config.prf_documents) {
            for word in result.document.content.to_lowercase().split_whitespace() {
                let word = word.trim_matches(|c: char| !c.is_alphanumeric());
                if !word.is_empty() && word.len() > 2 && !query_words.contains(word) {
                    *term_counts.entry(word.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Get top terms by frequency
        let mut top_terms: Vec<_> = term_counts
            .into_iter()
            .filter(|(_, count)| *count >= self.config.prf_min_frequency)
            .collect();
        top_terms.sort_by_key(|b| std::cmp::Reverse(b.1));

        let expanded_terms: Vec<String> = top_terms
            .into_iter()
            .take(self.config.max_expansions)
            .map(|(term, _)| term)
            .collect();

        let mut new_query = query.clone();
        if !expanded_terms.is_empty() {
            new_query.text = format!("{} {}", query.text, expanded_terms.join(" "));
        }
        new_query
    }
}
