//! Composite query expander combining multiple strategies.

use std::collections::HashSet;

use async_trait::async_trait;

use crate::types::{Query, SearchResult};

use super::ngram::NGramExpander;
use super::stem::StemExpander;
use super::synonym::SynonymExpander;
use super::types::{ExpansionConfig, QueryExpander};

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
