//! Core types and traits for query expansion.

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
