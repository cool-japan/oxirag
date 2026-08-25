//! Stemming-based query expander.

use async_trait::async_trait;

use crate::types::{Query, SearchResult};

use super::types::{ExpandedQuery, ExpansionConfig, ExpansionMethod, QueryExpander};

/// Expander that adds stemmed/root word forms.
#[derive(Debug, Clone)]
pub struct StemExpander {
    /// Configuration for expansion.
    config: ExpansionConfig,
}

impl Default for StemExpander {
    fn default() -> Self {
        Self::new(ExpansionConfig::default())
    }
}

impl StemExpander {
    /// Create a new stem expander with the given configuration.
    #[must_use]
    pub fn new(config: ExpansionConfig) -> Self {
        Self { config }
    }

    /// Apply simple porter-style stemming to a word.
    #[must_use]
    pub fn stem(&self, word: &str) -> String {
        let word = word.to_lowercase();

        // Simple suffix removal (porter-style)
        let suffixes = [
            "ational", "tional", "enci", "anci", "izer", "isation", "ization", "ation", "ator",
            "alism", "iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "logi", "alli",
            "entli", "eli", "ousli", "ing", "edly", "ly", "ness", "ment", "ence", "ance", "able",
            "ible", "ant", "ent", "ism", "iti", "ous", "ive", "ize", "ise", "al", "er", "ed", "es",
            "s",
        ];

        let mut stem = word.clone();
        for suffix in &suffixes {
            if stem.len() > suffix.len() + 2 && stem.ends_with(suffix) {
                stem = stem[..stem.len() - suffix.len()].to_string();
                break;
            }
        }

        stem
    }

    /// Get common word forms from a stem.
    #[must_use]
    pub fn get_word_forms(&self, stem: &str) -> Vec<String> {
        let mut forms = Vec::new();
        let endings = [
            "", "s", "es", "ed", "ing", "er", "est", "ly", "ness", "ment", "tion", "ation",
        ];

        for ending in &endings {
            let form = format!("{stem}{ending}");
            if form != stem && !forms.contains(&form) {
                forms.push(form);
            }
        }

        forms
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl QueryExpander for StemExpander {
    async fn expand(&self, query: &Query) -> Vec<Query> {
        if !self.config.enable_stemming {
            return vec![query.clone()];
        }

        let words: Vec<&str> = query.text.split_whitespace().collect();
        let mut expanded_terms: Vec<String> = Vec::new();

        for word in &words {
            let stem = self.stem(word);
            if stem != word.to_lowercase() {
                expanded_terms.push(stem.clone());
            }
            // Add some word forms
            let forms = self.get_word_forms(&stem);
            for form in forms.into_iter().take(2) {
                if form.to_lowercase() != word.to_lowercase() {
                    expanded_terms.push(form);
                }
            }
        }

        let limited: Vec<String> = expanded_terms
            .into_iter()
            .take(self.config.max_expansions * words.len())
            .collect();

        if limited.is_empty() {
            return vec![query.clone()];
        }

        let expanded = ExpandedQuery::new(query.clone(), ExpansionMethod::Stemming)
            .with_terms(limited)
            .with_weight(0.9);
        vec![expanded.to_query()]
    }

    async fn reformulate(&self, query: &Query, _results: &[SearchResult]) -> Query {
        // For stemming, we just stem the existing terms
        let words: Vec<&str> = query.text.split_whitespace().collect();
        let stemmed: Vec<String> = words.iter().map(|w| self.stem(w)).collect();
        let mut new_query = query.clone();
        new_query.text = stemmed.join(" ");
        new_query
    }
}
