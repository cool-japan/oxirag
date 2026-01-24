//! Query expansion and reformulation for improved retrieval.
//!
//! This module provides tools for expanding and reformulating search queries
//! to improve retrieval effectiveness. Query expansion adds related terms
//! (synonyms, stems, n-grams) to broaden search coverage, while reformulation
//! restructures queries based on context or search results.
//!
//! # Expansion Methods
//!
//! - **Synonyms**: Add synonymous terms from a built-in dictionary
//! - **Stemming**: Add root word forms (porter-style stemming)
//! - **N-grams**: Generate character/word n-grams for fuzzy matching
//! - **Acronyms**: Expand common abbreviations
//! - **Spelling**: Add common spelling variations
//! - **Conceptual**: Add semantically related concepts
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirag::query_expansion::{QueryExpander, SynonymExpander, ExpansionConfig};
//! use oxirag::types::Query;
//!
//! let config = ExpansionConfig::default();
//! let expander = SynonymExpander::new(config);
//! let query = Query::new("machine learning algorithms");
//! let expanded = expander.expand(&query);
//! ```

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::types::{Query, SearchResult};

/// Method used to expand a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpansionMethod {
    /// Add synonyms from a built-in dictionary.
    Synonyms,
    /// Add stemmed/root word forms.
    Stemming,
    /// Generate n-grams (character or word level).
    NGrams,
    /// Expand acronyms and abbreviations.
    Acronyms,
    /// Add spelling variations.
    Spelling,
    /// Add semantically related concepts.
    Conceptual,
}

impl std::fmt::Display for ExpansionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Synonyms => write!(f, "synonyms"),
            Self::Stemming => write!(f, "stemming"),
            Self::NGrams => write!(f, "ngrams"),
            Self::Acronyms => write!(f, "acronyms"),
            Self::Spelling => write!(f, "spelling"),
            Self::Conceptual => write!(f, "conceptual"),
        }
    }
}

/// An expanded query with additional terms and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedQuery {
    /// The original query that was expanded.
    pub original_query: Query,
    /// Additional terms added through expansion.
    pub expanded_terms: Vec<String>,
    /// The method used for expansion.
    pub expansion_method: ExpansionMethod,
    /// Weight assigned to expanded terms (0.0 to 1.0).
    pub weight: f32,
}

impl ExpandedQuery {
    /// Create a new expanded query.
    #[must_use]
    pub fn new(original_query: Query, expansion_method: ExpansionMethod) -> Self {
        Self {
            original_query,
            expanded_terms: Vec::new(),
            expansion_method,
            weight: 1.0,
        }
    }

    /// Add an expanded term.
    #[must_use]
    pub fn with_term(mut self, term: impl Into<String>) -> Self {
        self.expanded_terms.push(term.into());
        self
    }

    /// Add multiple expanded terms.
    #[must_use]
    pub fn with_terms(mut self, terms: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.expanded_terms
            .extend(terms.into_iter().map(Into::into));
        self
    }

    /// Set the weight for expanded terms.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Convert the expanded query to a combined query string.
    #[must_use]
    pub fn to_combined_query(&self) -> String {
        let mut terms = vec![self.original_query.text.clone()];
        terms.extend(self.expanded_terms.clone());
        terms.join(" ")
    }

    /// Convert to a new Query with expanded text.
    #[must_use]
    pub fn to_query(&self) -> Query {
        let mut query = self.original_query.clone();
        query.text = self.to_combined_query();
        query
    }
}

/// Configuration for query expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    /// Maximum number of expanded terms to add per query term.
    pub max_expansions: usize,
    /// Weight for synonym expansions (0.0 to 1.0).
    pub synonym_weight: f32,
    /// Range for n-gram generation (min, max).
    pub ngram_range: (usize, usize),
    /// Number of top documents to use for pseudo-relevance feedback.
    pub prf_documents: usize,
    /// Minimum term frequency for PRF expansion.
    pub prf_min_frequency: usize,
    /// Whether to include stemmed terms.
    pub enable_stemming: bool,
    /// Whether to expand acronyms.
    pub enable_acronyms: bool,
    /// Whether to add spelling variations.
    pub enable_spelling: bool,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_expansions: 5,
            synonym_weight: 0.8,
            ngram_range: (2, 3),
            prf_documents: 3,
            prf_min_frequency: 2,
            enable_stemming: true,
            enable_acronyms: true,
            enable_spelling: true,
        }
    }
}

impl ExpansionConfig {
    /// Create a new expansion configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of expansions.
    #[must_use]
    pub fn with_max_expansions(mut self, max: usize) -> Self {
        self.max_expansions = max;
        self
    }

    /// Set the synonym weight.
    #[must_use]
    pub fn with_synonym_weight(mut self, weight: f32) -> Self {
        self.synonym_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the n-gram range.
    #[must_use]
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min.max(1), max.max(min.max(1)));
        self
    }

    /// Set the number of PRF documents.
    #[must_use]
    pub fn with_prf_documents(mut self, count: usize) -> Self {
        self.prf_documents = count;
        self
    }
}

/// Trait for query expansion strategies.
#[async_trait]
pub trait QueryExpander: Send + Sync + std::fmt::Debug {
    /// Generate expanded queries from the original query.
    ///
    /// Returns a vector of queries with additional terms added based on
    /// the expansion strategy.
    async fn expand(&self, query: &Query) -> Vec<Query>;

    /// Reformulate a query based on search results.
    ///
    /// Uses the results from an initial search to improve the query
    /// through pseudo-relevance feedback or similar techniques.
    async fn reformulate(&self, query: &Query, results: &[SearchResult]) -> Query;
}

/// Expander that adds synonyms from a built-in dictionary.
#[derive(Debug, Clone)]
pub struct SynonymExpander {
    /// Configuration for expansion.
    config: ExpansionConfig,
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

#[async_trait]
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
        top_terms.sort_by(|a, b| b.1.cmp(&a.1));

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

#[async_trait]
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

/// Pseudo-relevance feedback expander.
///
/// Uses the top results from an initial search to expand the query
/// with terms that appear frequently in relevant documents.
#[derive(Debug, Clone)]
pub struct PseudoRelevanceFeedback {
    /// Configuration for expansion.
    config: ExpansionConfig,
    /// Stop words to exclude from expansion.
    stop_words: HashSet<String>,
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
    fn default_stop_words() -> HashSet<String> {
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

#[async_trait]
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

/// A composite expander that combines multiple expansion strategies.
pub struct CompositeExpander {
    /// The expanders to use.
    expanders: Vec<Box<dyn QueryExpander>>,
    /// Whether to deduplicate expanded terms.
    deduplicate: bool,
}

impl std::fmt::Debug for CompositeExpander {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeExpander")
            .field("expanders_count", &self.expanders.len())
            .field("deduplicate", &self.deduplicate)
            .finish()
    }
}

impl Default for CompositeExpander {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeExpander {
    /// Create a new composite expander.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expanders: Vec::new(),
            deduplicate: true,
        }
    }

    /// Add an expander to the composite.
    pub fn add_expander(&mut self, expander: impl QueryExpander + 'static) {
        self.expanders.push(Box::new(expander));
    }

    /// Builder pattern for adding expanders.
    #[must_use]
    pub fn with_expander(mut self, expander: impl QueryExpander + 'static) -> Self {
        self.add_expander(expander);
        self
    }

    /// Set deduplication mode.
    #[must_use]
    pub fn with_deduplication(mut self, deduplicate: bool) -> Self {
        self.deduplicate = deduplicate;
        self
    }

    /// Create a default composite with common expanders.
    #[must_use]
    pub fn default_composite(config: ExpansionConfig) -> Self {
        Self::new()
            .with_expander(SynonymExpander::new(config.clone()))
            .with_expander(StemExpander::new(config.clone()))
            .with_expander(NGramExpander::new(config))
    }
}

#[async_trait]
impl QueryExpander for CompositeExpander {
    async fn expand(&self, query: &Query) -> Vec<Query> {
        let mut all_expanded = Vec::new();

        for expander in &self.expanders {
            let expanded = expander.expand(query).await;
            all_expanded.extend(expanded);
        }

        if self.deduplicate {
            let mut seen: HashSet<String> = HashSet::new();
            all_expanded.retain(|q| {
                let key = q.text.to_lowercase();
                if seen.contains(&key) {
                    false
                } else {
                    seen.insert(key);
                    true
                }
            });
        }

        if all_expanded.is_empty() {
            all_expanded.push(query.clone());
        }

        all_expanded
    }

    async fn reformulate(&self, query: &Query, results: &[SearchResult]) -> Query {
        // Apply all reformulations in sequence
        let mut current = query.clone();
        for expander in &self.expanders {
            current = expander.reformulate(&current, results).await;
        }
        current
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;
    use crate::types::Document;

    fn create_test_query(text: &str) -> Query {
        Query::new(text)
    }

    fn create_test_results() -> Vec<SearchResult> {
        vec![
            SearchResult::new(
                Document::new("Machine learning algorithms process large datasets efficiently"),
                0.9,
                0,
            ),
            SearchResult::new(
                Document::new("Deep learning neural networks achieve remarkable results"),
                0.85,
                1,
            ),
            SearchResult::new(
                Document::new("Training machine learning models requires significant computation"),
                0.8,
                2,
            ),
        ]
    }

    #[test]
    fn test_expansion_method_display() {
        assert_eq!(format!("{}", ExpansionMethod::Synonyms), "synonyms");
        assert_eq!(format!("{}", ExpansionMethod::Stemming), "stemming");
        assert_eq!(format!("{}", ExpansionMethod::NGrams), "ngrams");
    }

    #[test]
    fn test_expanded_query_creation() {
        let query = create_test_query("machine learning");
        let expanded = ExpandedQuery::new(query.clone(), ExpansionMethod::Synonyms)
            .with_term("computer")
            .with_term("training")
            .with_weight(0.8);

        assert_eq!(expanded.original_query.text, "machine learning");
        assert_eq!(expanded.expanded_terms.len(), 2);
        assert!((expanded.weight - 0.8).abs() < f32::EPSILON);
        assert_eq!(expanded.expansion_method, ExpansionMethod::Synonyms);
    }

    #[test]
    fn test_expanded_query_to_combined() {
        let query = create_test_query("search");
        let expanded =
            ExpandedQuery::new(query, ExpansionMethod::Synonyms).with_terms(vec!["find", "lookup"]);

        let combined = expanded.to_combined_query();
        assert!(combined.contains("search"));
        assert!(combined.contains("find"));
        assert!(combined.contains("lookup"));
    }

    #[test]
    fn test_expansion_config_default() {
        let config = ExpansionConfig::default();
        assert_eq!(config.max_expansions, 5);
        assert!((config.synonym_weight - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.ngram_range, (2, 3));
        assert_eq!(config.prf_documents, 3);
    }

    #[test]
    fn test_expansion_config_builder() {
        let config = ExpansionConfig::new()
            .with_max_expansions(10)
            .with_synonym_weight(0.9)
            .with_ngram_range(2, 4)
            .with_prf_documents(5);

        assert_eq!(config.max_expansions, 10);
        assert!((config.synonym_weight - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.ngram_range, (2, 4));
        assert_eq!(config.prf_documents, 5);
    }

    #[test]
    fn test_synonym_weight_clamping() {
        let config = ExpansionConfig::new().with_synonym_weight(1.5);
        assert!((config.synonym_weight - 1.0).abs() < f32::EPSILON);

        let config = ExpansionConfig::new().with_synonym_weight(-0.5);
        assert!(config.synonym_weight.abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_synonym_expander() {
        let expander = SynonymExpander::default();
        let query = create_test_query("search data");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());

        // Check that synonyms are added
        let combined_text = expanded
            .iter()
            .map(|q| q.text.clone())
            .collect::<Vec<_>>()
            .join(" ");

        assert!(combined_text.contains("search"));
        // Should have synonyms for "search" and/or "data"
        let has_synonym = combined_text.contains("find")
            || combined_text.contains("query")
            || combined_text.contains("information");
        assert!(has_synonym);
    }

    #[tokio::test]
    async fn test_synonym_expander_get_synonyms() {
        let expander = SynonymExpander::default();

        let synonyms = expander.get_synonyms("search");
        assert!(synonyms.contains(&"find".to_string()));

        let synonyms = expander.get_synonyms("SEARCH");
        assert!(synonyms.contains(&"find".to_string()));

        let synonyms = expander.get_synonyms("nonexistent");
        assert!(synonyms.is_empty());
    }

    #[tokio::test]
    async fn test_synonym_expander_reformulate() {
        // Use a custom config with lower min_frequency to ensure expansion
        let config = ExpansionConfig {
            prf_min_frequency: 1,
            ..Default::default()
        };
        let expander = SynonymExpander::new(config);
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = expander.reformulate(&query, &results).await;
        assert!(reformulated.text.contains("machine learning"));
        // Should have added terms from results (now that min_frequency is 1)
        assert!(reformulated.text.len() > query.text.len());
    }

    #[test]
    fn test_stem_expander_stem() {
        let expander = StemExpander::default();

        assert_eq!(expander.stem("running"), "runn");
        assert_eq!(expander.stem("algorithms"), "algorithm");
        assert_eq!(expander.stem("processing"), "process");
    }

    #[test]
    fn test_stem_expander_word_forms() {
        let expander = StemExpander::default();
        let forms = expander.get_word_forms("learn");

        assert!(forms.contains(&"learns".to_string()));
        assert!(forms.contains(&"learned".to_string()));
        assert!(forms.contains(&"learning".to_string()));
    }

    #[tokio::test]
    async fn test_stem_expander_expand() {
        let expander = StemExpander::default();
        let query = create_test_query("processing algorithms");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());

        let combined_text = expanded[0].text.clone();
        assert!(combined_text.contains("processing"));
    }

    #[test]
    fn test_ngram_expander_char_ngrams() {
        let expander = NGramExpander::default();
        let ngrams = expander.char_ngrams("hello", 3);

        assert!(ngrams.contains(&"hel".to_string()));
        assert!(ngrams.contains(&"ell".to_string()));
        assert!(ngrams.contains(&"llo".to_string()));
    }

    #[test]
    fn test_ngram_expander_word_ngrams() {
        let expander = NGramExpander::default();
        let ngrams = expander.word_ngrams("machine learning algorithms", 2);

        assert!(ngrams.contains(&"machine learning".to_string()));
        assert!(ngrams.contains(&"learning algorithms".to_string()));
    }

    #[tokio::test]
    async fn test_ngram_expander_expand() {
        let expander = NGramExpander::default();
        let query = create_test_query("machine learning");

        let expanded = expander.expand(&query).await;
        assert!(!expanded.is_empty());
    }

    #[test]
    fn test_prf_stop_words() {
        let prf = PseudoRelevanceFeedback::default();

        assert!(prf.is_stop_word("the"));
        assert!(prf.is_stop_word("and"));
        assert!(!prf.is_stop_word("algorithm"));
        assert!(!prf.is_stop_word("machine"));
    }

    #[test]
    fn test_prf_extract_top_terms() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let top_terms = prf.extract_top_terms(&results, &query);
        assert!(!top_terms.is_empty());

        // Should not include query words or stop words
        for (term, _) in &top_terms {
            assert_ne!(term, "machine");
            assert_ne!(term, "learning");
            assert!(!prf.is_stop_word(term));
        }
    }

    #[tokio::test]
    async fn test_prf_reformulate() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = prf.reformulate(&query, &results).await;
        assert!(reformulated.text.contains("machine learning"));
        assert!(reformulated.text.len() >= query.text.len());
    }

    #[tokio::test]
    async fn test_prf_reformulate_empty_results() {
        let prf = PseudoRelevanceFeedback::default();
        let query = create_test_query("machine learning");
        let results: Vec<SearchResult> = vec![];

        let reformulated = prf.reformulate(&query, &results).await;
        assert_eq!(reformulated.text, query.text);
    }

    #[test]
    fn test_query_reformulator_simplify() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("what is the machine learning algorithm");

        let simplified = reformulator.simplify(&query);
        assert!(!simplified.text.contains("the"));
        assert!(simplified.text.contains("machine"));
        assert!(simplified.text.contains("learning"));
    }

    #[test]
    fn test_query_reformulator_clarify() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning");

        let clarified = reformulator.clarify(&query, "neural networks");
        assert!(clarified.text.contains("machine learning"));
        assert!(clarified.text.contains("neural networks"));
    }

    #[test]
    fn test_query_reformulator_decompose() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning and deep learning");

        let decomposed = reformulator.decompose(&query);
        assert!(decomposed.len() >= 2);
    }

    #[test]
    fn test_query_reformulator_decompose_simple() {
        let reformulator = QueryReformulator::new();
        let query = create_test_query("machine learning");

        let decomposed = reformulator.decompose(&query);
        assert_eq!(decomposed.len(), 1);
        assert_eq!(decomposed[0].text, "machine learning");
    }

    #[test]
    fn test_query_reformulator_rephrase_as_question() {
        let reformulator = QueryReformulator::new();

        let query = create_test_query("machine learning");
        let rephrased = reformulator.rephrase_as_question(&query);
        assert!(rephrased.text.ends_with('?'));

        // Already a question
        let query = create_test_query("What is machine learning?");
        let rephrased = reformulator.rephrase_as_question(&query);
        assert_eq!(rephrased.text, "What is machine learning?");
    }

    #[tokio::test]
    async fn test_composite_expander() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config);

        let query = create_test_query("search data");
        let expanded = composite.expand(&query).await;

        assert!(!expanded.is_empty());
    }

    #[tokio::test]
    async fn test_composite_expander_reformulate() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config);

        let query = create_test_query("machine learning");
        let results = create_test_results();

        let reformulated = composite.reformulate(&query, &results).await;
        assert!(!reformulated.text.is_empty());
    }

    #[tokio::test]
    async fn test_composite_expander_deduplication() {
        let config = ExpansionConfig::default();
        let composite = CompositeExpander::default_composite(config).with_deduplication(true);

        let query = create_test_query("search");
        let expanded = composite.expand(&query).await;

        // Check for duplicates
        let mut seen: HashSet<String> = HashSet::new();
        for q in &expanded {
            let key = q.text.to_lowercase();
            assert!(!seen.contains(&key), "Duplicate found: {key}");
            seen.insert(key);
        }
    }

    #[tokio::test]
    async fn test_synonym_expander_custom_dictionary() {
        let mut dictionary = HashMap::new();
        dictionary.insert(
            "rust".to_string(),
            vec!["oxidation".to_string(), "corrosion".to_string()],
        );

        let config = ExpansionConfig::default();
        let expander = SynonymExpander::with_dictionary(config, dictionary);

        let synonyms = expander.get_synonyms("rust");
        assert!(synonyms.contains(&"oxidation".to_string()));
    }

    #[test]
    fn test_expanded_query_weight_clamping() {
        let query = create_test_query("test");
        let expanded =
            ExpandedQuery::new(query.clone(), ExpansionMethod::Synonyms).with_weight(1.5);
        assert!((expanded.weight - 1.0).abs() < f32::EPSILON);

        let expanded = ExpandedQuery::new(query, ExpansionMethod::Synonyms).with_weight(-0.5);
        assert!(expanded.weight.abs() < f32::EPSILON);
    }

    #[test]
    fn test_ngram_range_validation() {
        let config = ExpansionConfig::new().with_ngram_range(0, 2);
        assert_eq!(config.ngram_range.0, 1);

        let config = ExpansionConfig::new().with_ngram_range(3, 2);
        assert_eq!(config.ngram_range.0, 3);
        assert_eq!(config.ngram_range.1, 3);
    }

    #[tokio::test]
    async fn test_stem_expander_disabled() {
        let config = ExpansionConfig {
            enable_stemming: false,
            ..Default::default()
        };
        let expander = StemExpander::new(config);
        let query = create_test_query("processing");

        let expanded = expander.expand(&query).await;
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].text, "processing");
    }

    #[test]
    fn test_char_ngrams_short_word() {
        let expander = NGramExpander::default();
        let ngrams = expander.char_ngrams("hi", 3);
        assert_eq!(ngrams, vec!["hi"]);
    }

    #[test]
    fn test_word_ngrams_short_phrase() {
        let expander = NGramExpander::default();
        let ngrams = expander.word_ngrams("hello", 2);
        assert_eq!(ngrams, vec!["hello"]);
    }
}
