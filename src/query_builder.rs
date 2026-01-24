//! Fluent query builder API for constructing complex queries.
//!
//! This module provides a `QueryBuilder` struct with a fluent interface
//! for building complex queries with support for metadata filters, scoring,
//! layer hints, timeouts, caching, and graph traversal.
//!
//! # Examples
//!
//! ```
//! use oxirag::query_builder::QueryBuilder;
//! use oxirag::layer1_echo::MetadataFilter;
//! use std::time::Duration;
//!
//! let query = QueryBuilder::new()
//!     .text("What is Rust?")
//!     .with_top_k(5)
//!     .with_min_score(0.7)
//!     .with_metadata(MetadataFilter::eq("category", "programming"))
//!     .with_timeout(Duration::from_secs(30))
//!     .with_cache_key("rust-query-v1")
//!     .build()
//!     .expect("Failed to build query");
//!
//! assert_eq!(query.text, "What is Rust?");
//! assert_eq!(query.top_k, 5);
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{OxiRagError, PipelineError};
use crate::layer1_echo::filter::MetadataFilter;
use crate::types::Query;

/// Hints for which layers to use during query processing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct LayerHints {
    /// Use the Echo layer (vector search).
    pub use_echo: bool,
    /// Use the Speculator layer (draft verification).
    pub use_speculator: bool,
    /// Use the Judge layer (SMT verification).
    pub use_judge: bool,
    /// Use the Graph layer (knowledge graph traversal).
    #[cfg(feature = "graphrag")]
    pub use_graph: bool,
}

impl LayerHints {
    /// Create layer hints with all layers enabled.
    #[must_use]
    pub fn all() -> Self {
        Self {
            use_echo: true,
            use_speculator: true,
            use_judge: true,
            #[cfg(feature = "graphrag")]
            use_graph: true,
        }
    }

    /// Create layer hints with only the Echo layer enabled.
    #[must_use]
    pub fn echo_only() -> Self {
        Self {
            use_echo: true,
            use_speculator: false,
            use_judge: false,
            #[cfg(feature = "graphrag")]
            use_graph: false,
        }
    }

    /// Create layer hints with Echo and Speculator layers enabled.
    #[must_use]
    pub fn echo_and_speculator() -> Self {
        Self {
            use_echo: true,
            use_speculator: true,
            use_judge: false,
            #[cfg(feature = "graphrag")]
            use_graph: false,
        }
    }

    /// Enable the Echo layer.
    #[must_use]
    pub fn with_echo(mut self) -> Self {
        self.use_echo = true;
        self
    }

    /// Enable the Speculator layer.
    #[must_use]
    pub fn with_speculator(mut self) -> Self {
        self.use_speculator = true;
        self
    }

    /// Enable the Judge layer.
    #[must_use]
    pub fn with_judge(mut self) -> Self {
        self.use_judge = true;
        self
    }

    /// Enable the Graph layer.
    #[cfg(feature = "graphrag")]
    #[must_use]
    pub fn with_graph(mut self) -> Self {
        self.use_graph = true;
        self
    }

    /// Disable the Echo layer.
    #[must_use]
    pub fn without_echo(mut self) -> Self {
        self.use_echo = false;
        self
    }

    /// Disable the Speculator layer.
    #[must_use]
    pub fn without_speculator(mut self) -> Self {
        self.use_speculator = false;
        self
    }

    /// Disable the Judge layer.
    #[must_use]
    pub fn without_judge(mut self) -> Self {
        self.use_judge = false;
        self
    }

    /// Disable the Graph layer.
    #[cfg(feature = "graphrag")]
    #[must_use]
    pub fn without_graph(mut self) -> Self {
        self.use_graph = false;
        self
    }
}

/// Graph traversal context hints for hybrid queries.
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphContext {
    /// Starting entity IDs for graph traversal.
    pub start_entities: Vec<String>,
    /// Maximum number of hops to traverse.
    pub max_hops: Option<usize>,
    /// Minimum confidence for graph paths.
    pub min_confidence: Option<f32>,
    /// Filter for specific relationship types.
    pub relationship_types: Option<Vec<crate::layer4_graph::types::RelationshipType>>,
    /// Filter for specific entity types.
    pub entity_types: Option<Vec<crate::layer4_graph::types::EntityType>>,
    /// Traversal direction.
    pub direction: Option<crate::layer4_graph::types::Direction>,
}

#[cfg(feature = "graphrag")]
impl GraphContext {
    /// Create a new empty graph context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add starting entities for graph traversal.
    #[must_use]
    pub fn with_start_entities(mut self, entities: Vec<String>) -> Self {
        self.start_entities = entities;
        self
    }

    /// Add a single starting entity.
    #[must_use]
    pub fn with_start_entity(mut self, entity: impl Into<String>) -> Self {
        self.start_entities.push(entity.into());
        self
    }

    /// Set the maximum number of hops.
    #[must_use]
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = Some(max_hops);
        self
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = Some(min_confidence);
        self
    }

    /// Set the relationship type filter.
    #[must_use]
    pub fn with_relationship_types(
        mut self,
        types: Vec<crate::layer4_graph::types::RelationshipType>,
    ) -> Self {
        self.relationship_types = Some(types);
        self
    }

    /// Set the entity type filter.
    #[must_use]
    pub fn with_entity_types(mut self, types: Vec<crate::layer4_graph::types::EntityType>) -> Self {
        self.entity_types = Some(types);
        self
    }

    /// Set the traversal direction.
    #[must_use]
    pub fn with_direction(mut self, direction: crate::layer4_graph::types::Direction) -> Self {
        self.direction = Some(direction);
        self
    }
}

/// Extended query with additional builder options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedQuery {
    /// The base query.
    pub query: Query,
    /// Query timeout.
    pub timeout: Option<Duration>,
    /// Custom cache key for prefix caching.
    pub cache_key: Option<String>,
    /// Layer hints for selective processing.
    pub layer_hints: Option<LayerHints>,
    /// Graph traversal context.
    #[cfg(feature = "graphrag")]
    pub graph_context: Option<GraphContext>,
}

impl ExtendedQuery {
    /// Create a new extended query from a base query.
    #[must_use]
    pub fn new(query: Query) -> Self {
        Self {
            query,
            timeout: None,
            cache_key: None,
            layer_hints: None,
            #[cfg(feature = "graphrag")]
            graph_context: None,
        }
    }

    /// Get the query text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.query.text
    }

    /// Get the top-k value.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.query.top_k
    }

    /// Get the minimum score threshold.
    #[must_use]
    pub fn min_score(&self) -> Option<f32> {
        self.query.min_score
    }

    /// Get the metadata filter.
    #[must_use]
    pub fn metadata_filter(&self) -> Option<&MetadataFilter> {
        self.query.metadata_filter.as_ref()
    }

    /// Get the timeout.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Get the cache key.
    #[must_use]
    pub fn cache_key(&self) -> Option<&str> {
        self.cache_key.as_deref()
    }

    /// Get the layer hints.
    #[must_use]
    pub fn layer_hints(&self) -> Option<&LayerHints> {
        self.layer_hints.as_ref()
    }

    /// Get the graph context.
    #[cfg(feature = "graphrag")]
    #[must_use]
    pub fn graph_context(&self) -> Option<&GraphContext> {
        self.graph_context.as_ref()
    }

    /// Convert to the base query, consuming the extended query.
    #[must_use]
    pub fn into_query(self) -> Query {
        self.query
    }
}

impl From<ExtendedQuery> for Query {
    fn from(extended: ExtendedQuery) -> Self {
        extended.query
    }
}

impl From<Query> for ExtendedQuery {
    fn from(query: Query) -> Self {
        Self::new(query)
    }
}

/// A fluent builder for constructing complex queries.
///
/// The `QueryBuilder` provides a fluent interface for building queries
/// with support for all query options including metadata filters,
/// layer hints, timeouts, and graph traversal context.
///
/// # Examples
///
/// ```
/// use oxirag::query_builder::QueryBuilder;
/// use oxirag::layer1_echo::MetadataFilter;
///
/// // Simple query
/// let query = QueryBuilder::new()
///     .text("What is Rust?")
///     .build()
///     .unwrap();
///
/// // Complex query with all options
/// let complex = QueryBuilder::new()
///     .text("Find documents about Rust")
///     .with_top_k(10)
///     .with_min_score(0.5)
///     .with_metadata(MetadataFilter::eq("lang", "en"))
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct QueryBuilder {
    text: Option<String>,
    top_k: Option<usize>,
    min_score: Option<f32>,
    filters: HashMap<String, String>,
    metadata_filter: Option<MetadataFilter>,
    timeout: Option<Duration>,
    cache_key: Option<String>,
    layer_hints: Option<LayerHints>,
    #[cfg(feature = "graphrag")]
    graph_context: Option<GraphContext>,
}

impl QueryBuilder {
    /// Create a new query builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let builder = QueryBuilder::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the query text.
    ///
    /// This is required for building a valid query.
    ///
    /// # Arguments
    ///
    /// * `text` - The query text to search for.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let query = QueryBuilder::new()
    ///     .text("What is machine learning?")
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(query.text, "What is machine learning?");
    /// ```
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set the maximum number of results to return.
    ///
    /// # Arguments
    ///
    /// * `top_k` - Maximum number of results (default is 10).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let query = QueryBuilder::new()
    ///     .text("query")
    ///     .with_top_k(5)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(query.top_k, 5);
    /// ```
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the minimum similarity score threshold.
    ///
    /// Results with scores below this threshold will be filtered out.
    ///
    /// # Arguments
    ///
    /// * `min_score` - Minimum score threshold (0.0 to 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let query = QueryBuilder::new()
    ///     .text("query")
    ///     .with_min_score(0.7)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(query.min_score, Some(0.7));
    /// ```
    #[must_use]
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    /// Add a simple key-value metadata filter (equality).
    ///
    /// Multiple calls to this method will add multiple filters
    /// that must all match (implicit AND).
    ///
    /// # Arguments
    ///
    /// * `key` - The metadata field name.
    /// * `value` - The value to match.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let query = QueryBuilder::new()
    ///     .text("query")
    ///     .with_filter("category", "science")
    ///     .with_filter("status", "published")
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(query.filters.get("category"), Some(&"science".to_string()));
    /// ```
    #[must_use]
    pub fn with_filter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set an advanced metadata filter.
    ///
    /// This replaces any previously set metadata filter.
    /// For complex filtering, use `MetadataFilter::and()` and `MetadataFilter::or()`.
    ///
    /// # Arguments
    ///
    /// * `filter` - The metadata filter to apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    /// use oxirag::layer1_echo::MetadataFilter;
    ///
    /// let query = QueryBuilder::new()
    ///     .text("query")
    ///     .with_metadata(MetadataFilter::and(vec![
    ///         MetadataFilter::eq("status", "published"),
    ///         MetadataFilter::or(vec![
    ///             MetadataFilter::eq("category", "science"),
    ///             MetadataFilter::eq("category", "tech"),
    ///         ]),
    ///     ]))
    ///     .build()
    ///     .unwrap();
    ///
    /// assert!(query.metadata_filter.is_some());
    /// ```
    #[must_use]
    pub fn with_metadata(mut self, filter: MetadataFilter) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Set the query timeout.
    ///
    /// If the query takes longer than this timeout, it will be cancelled.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for results.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    /// use std::time::Duration;
    ///
    /// let extended = QueryBuilder::new()
    ///     .text("query")
    ///     .with_timeout(Duration::from_secs(30))
    ///     .build_extended()
    ///     .unwrap();
    ///
    /// assert_eq!(extended.timeout, Some(Duration::from_secs(30)));
    /// ```
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a custom cache key for prefix caching.
    ///
    /// This allows you to control how the query results are cached.
    ///
    /// # Arguments
    ///
    /// * `key` - The custom cache key.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// let extended = QueryBuilder::new()
    ///     .text("query")
    ///     .with_cache_key("my-custom-key-v1")
    ///     .build_extended()
    ///     .unwrap();
    ///
    /// assert_eq!(extended.cache_key, Some("my-custom-key-v1".to_string()));
    /// ```
    #[must_use]
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }

    /// Set layer hints for selective processing.
    ///
    /// This allows you to control which layers are used during query processing.
    ///
    /// # Arguments
    ///
    /// * `hints` - The layer hints configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::{QueryBuilder, LayerHints};
    ///
    /// let extended = QueryBuilder::new()
    ///     .text("query")
    ///     .with_layer_hints(LayerHints::echo_only())
    ///     .build_extended()
    ///     .unwrap();
    ///
    /// let hints = extended.layer_hints.unwrap();
    /// assert!(hints.use_echo);
    /// assert!(!hints.use_speculator);
    /// ```
    #[must_use]
    pub fn with_layer_hints(mut self, hints: LayerHints) -> Self {
        self.layer_hints = Some(hints);
        self
    }

    /// Set graph traversal context for hybrid queries.
    ///
    /// This is only available when the `graphrag` feature is enabled.
    ///
    /// # Arguments
    ///
    /// * `context` - The graph traversal context.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use oxirag::query_builder::{QueryBuilder, GraphContext};
    ///
    /// let extended = QueryBuilder::new()
    ///     .text("query")
    ///     .with_graph_context(
    ///         GraphContext::new()
    ///             .with_start_entity("rust")
    ///             .with_max_hops(3)
    ///     )
    ///     .build_extended()
    ///     .unwrap();
    /// ```
    #[cfg(feature = "graphrag")]
    #[must_use]
    pub fn with_graph_context(mut self, context: GraphContext) -> Self {
        self.graph_context = Some(context);
        self
    }

    /// Build the query.
    ///
    /// This validates the builder state and constructs the final `Query`.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Query)` if the builder state is valid, or an error if
    /// required fields are missing (e.g., no query text).
    ///
    /// # Errors
    ///
    /// Returns `OxiRagError::Pipeline` if:
    /// - No query text was provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::QueryBuilder;
    ///
    /// // Valid query
    /// let query = QueryBuilder::new()
    ///     .text("What is Rust?")
    ///     .build()
    ///     .unwrap();
    ///
    /// // Missing text - will error
    /// let result = QueryBuilder::new().build();
    /// assert!(result.is_err());
    /// ```
    pub fn build(self) -> Result<Query, OxiRagError> {
        let text = self.text.ok_or_else(|| {
            OxiRagError::Pipeline(PipelineError::BuildError(
                "Query text is required".to_string(),
            ))
        })?;

        let mut query = Query::new(text);

        if let Some(top_k) = self.top_k {
            query = query.with_top_k(top_k);
        }

        if let Some(min_score) = self.min_score {
            query = query.with_min_score(min_score);
        }

        for (key, value) in self.filters {
            query = query.with_filter(key, value);
        }

        if let Some(filter) = self.metadata_filter {
            query = query.with_metadata_filter(filter);
        }

        Ok(query)
    }

    /// Build an extended query with additional options.
    ///
    /// This includes timeout, cache key, layer hints, and graph context
    /// in addition to the base query options.
    ///
    /// # Returns
    ///
    /// Returns `Ok(ExtendedQuery)` if the builder state is valid.
    ///
    /// # Errors
    ///
    /// Returns the same errors as `build()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxirag::query_builder::{QueryBuilder, LayerHints};
    /// use std::time::Duration;
    ///
    /// let extended = QueryBuilder::new()
    ///     .text("What is Rust?")
    ///     .with_top_k(5)
    ///     .with_timeout(Duration::from_secs(30))
    ///     .with_layer_hints(LayerHints::all())
    ///     .build_extended()
    ///     .unwrap();
    ///
    /// assert_eq!(extended.query.text, "What is Rust?");
    /// assert_eq!(extended.timeout, Some(Duration::from_secs(30)));
    /// ```
    pub fn build_extended(self) -> Result<ExtendedQuery, OxiRagError> {
        let timeout = self.timeout;
        let cache_key = self.cache_key.clone();
        let layer_hints = self.layer_hints.clone();
        #[cfg(feature = "graphrag")]
        let graph_context = self.graph_context.clone();

        let query = self.build()?;

        Ok(ExtendedQuery {
            query,
            timeout,
            cache_key,
            layer_hints,
            #[cfg(feature = "graphrag")]
            graph_context,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_simple_query() {
        let query = QueryBuilder::new()
            .text("What is Rust?")
            .build()
            .expect("Failed to build query");

        assert_eq!(query.text, "What is Rust?");
        assert_eq!(query.top_k, 10); // default
        assert!(query.min_score.is_none());
    }

    #[test]
    fn test_builder_with_top_k() {
        let query = QueryBuilder::new()
            .text("query")
            .with_top_k(5)
            .build()
            .expect("Failed to build query");

        assert_eq!(query.top_k, 5);
    }

    #[test]
    fn test_builder_with_min_score() {
        let query = QueryBuilder::new()
            .text("query")
            .with_min_score(0.7)
            .build()
            .expect("Failed to build query");

        assert!((query.min_score.unwrap() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_with_simple_filters() {
        let query = QueryBuilder::new()
            .text("query")
            .with_filter("category", "science")
            .with_filter("status", "published")
            .build()
            .expect("Failed to build query");

        assert_eq!(query.filters.get("category"), Some(&"science".to_string()));
        assert_eq!(query.filters.get("status"), Some(&"published".to_string()));
    }

    #[test]
    fn test_builder_with_metadata_filter() {
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("status", "published"),
            MetadataFilter::or(vec![
                MetadataFilter::eq("category", "science"),
                MetadataFilter::eq("category", "tech"),
            ]),
        ]);

        let query = QueryBuilder::new()
            .text("query")
            .with_metadata(filter.clone())
            .build()
            .expect("Failed to build query");

        assert_eq!(query.metadata_filter, Some(filter));
    }

    #[test]
    fn test_builder_missing_text_error() {
        let result = QueryBuilder::new().build();
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("Query text is required"));
    }

    #[test]
    fn test_builder_extended_query() {
        let extended = QueryBuilder::new()
            .text("query")
            .with_top_k(5)
            .with_timeout(Duration::from_secs(30))
            .with_cache_key("test-key")
            .with_layer_hints(LayerHints::echo_only())
            .build_extended()
            .expect("Failed to build extended query");

        assert_eq!(extended.query.text, "query");
        assert_eq!(extended.query.top_k, 5);
        assert_eq!(extended.timeout, Some(Duration::from_secs(30)));
        assert_eq!(extended.cache_key, Some("test-key".to_string()));

        let hints = extended.layer_hints.unwrap();
        assert!(hints.use_echo);
        assert!(!hints.use_speculator);
        assert!(!hints.use_judge);
    }

    #[test]
    fn test_extended_query_accessors() {
        let extended = QueryBuilder::new()
            .text("test query")
            .with_top_k(20)
            .with_min_score(0.5)
            .with_metadata(MetadataFilter::eq("key", "value"))
            .with_timeout(Duration::from_secs(60))
            .with_cache_key("cache-key")
            .with_layer_hints(LayerHints::all())
            .build_extended()
            .expect("Failed to build");

        assert_eq!(extended.text(), "test query");
        assert_eq!(extended.top_k(), 20);
        assert!((extended.min_score().unwrap() - 0.5).abs() < f32::EPSILON);
        assert!(extended.metadata_filter().is_some());
        assert_eq!(extended.timeout(), Some(Duration::from_secs(60)));
        assert_eq!(extended.cache_key(), Some("cache-key"));
        assert!(extended.layer_hints().is_some());
    }

    #[test]
    fn test_extended_query_into_query() {
        let extended = QueryBuilder::new()
            .text("test")
            .with_top_k(15)
            .build_extended()
            .expect("Failed to build");

        let query = extended.into_query();
        assert_eq!(query.text, "test");
        assert_eq!(query.top_k, 15);
    }

    #[test]
    fn test_extended_query_from_query() {
        let query = Query::new("original").with_top_k(8);
        let extended: ExtendedQuery = query.into();

        assert_eq!(extended.query.text, "original");
        assert_eq!(extended.query.top_k, 8);
        assert!(extended.timeout.is_none());
        assert!(extended.cache_key.is_none());
    }

    #[test]
    fn test_layer_hints_all() {
        let hints = LayerHints::all();
        assert!(hints.use_echo);
        assert!(hints.use_speculator);
        assert!(hints.use_judge);
    }

    #[test]
    fn test_layer_hints_echo_only() {
        let hints = LayerHints::echo_only();
        assert!(hints.use_echo);
        assert!(!hints.use_speculator);
        assert!(!hints.use_judge);
    }

    #[test]
    fn test_layer_hints_echo_and_speculator() {
        let hints = LayerHints::echo_and_speculator();
        assert!(hints.use_echo);
        assert!(hints.use_speculator);
        assert!(!hints.use_judge);
    }

    #[test]
    fn test_layer_hints_builder_methods() {
        let hints = LayerHints::default()
            .with_echo()
            .with_speculator()
            .with_judge();

        assert!(hints.use_echo);
        assert!(hints.use_speculator);
        assert!(hints.use_judge);

        let hints2 = hints.without_speculator().without_judge();
        assert!(hints2.use_echo);
        assert!(!hints2.use_speculator);
        assert!(!hints2.use_judge);
    }

    #[test]
    fn test_query_builder_clone() {
        let builder = QueryBuilder::new()
            .text("query")
            .with_top_k(5)
            .with_min_score(0.6);

        let cloned = builder.clone();
        let query = cloned.build().expect("Failed to build");

        assert_eq!(query.text, "query");
        assert_eq!(query.top_k, 5);
    }

    #[test]
    fn test_query_builder_default() {
        let builder = QueryBuilder::default();
        let result = builder.text("test").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_hints_equality() {
        let hints1 = LayerHints::all();
        let hints2 = LayerHints::all();
        let hints3 = LayerHints::echo_only();

        assert_eq!(hints1, hints2);
        assert_ne!(hints1, hints3);
    }

    #[cfg(feature = "graphrag")]
    mod graphrag_tests {
        use super::*;
        use crate::layer4_graph::types::{Direction, EntityType, RelationshipType};

        #[test]
        fn test_graph_context_new() {
            let ctx = GraphContext::new();
            assert!(ctx.start_entities.is_empty());
            assert!(ctx.max_hops.is_none());
            assert!(ctx.min_confidence.is_none());
        }

        #[test]
        fn test_graph_context_builder() {
            let ctx = GraphContext::new()
                .with_start_entity("rust")
                .with_start_entity("programming")
                .with_max_hops(3)
                .with_min_confidence(0.5)
                .with_direction(Direction::Outgoing)
                .with_relationship_types(vec![RelationshipType::RelatedTo])
                .with_entity_types(vec![EntityType::Technology]);

            assert_eq!(ctx.start_entities.len(), 2);
            assert_eq!(ctx.max_hops, Some(3));
            assert!((ctx.min_confidence.unwrap() - 0.5).abs() < f32::EPSILON);
            assert_eq!(ctx.direction, Some(Direction::Outgoing));
            assert!(ctx.relationship_types.is_some());
            assert!(ctx.entity_types.is_some());
        }

        #[test]
        fn test_graph_context_with_start_entities() {
            let ctx =
                GraphContext::new().with_start_entities(vec!["a".to_string(), "b".to_string()]);

            assert_eq!(ctx.start_entities, vec!["a", "b"]);
        }

        #[test]
        fn test_builder_with_graph_context() {
            let ctx = GraphContext::new()
                .with_start_entity("test")
                .with_max_hops(2);

            let extended = QueryBuilder::new()
                .text("query")
                .with_graph_context(ctx)
                .build_extended()
                .expect("Failed to build");

            let graph_ctx = extended.graph_context().unwrap();
            assert_eq!(graph_ctx.start_entities, vec!["test"]);
            assert_eq!(graph_ctx.max_hops, Some(2));
        }

        #[test]
        fn test_layer_hints_with_graph() {
            let hints = LayerHints::all();
            assert!(hints.use_graph);

            let hints2 = hints.without_graph();
            assert!(!hints2.use_graph);

            let hints3 = LayerHints::default().with_graph();
            assert!(hints3.use_graph);
        }
    }

    #[test]
    fn test_query_builder_full_chain() {
        let query = QueryBuilder::new()
            .text("Find documents about machine learning")
            .with_top_k(20)
            .with_min_score(0.6)
            .with_filter("language", "en")
            .with_filter("year", "2024")
            .with_metadata(MetadataFilter::exists("author"))
            .build()
            .expect("Failed to build");

        assert_eq!(query.text, "Find documents about machine learning");
        assert_eq!(query.top_k, 20);
        assert!((query.min_score.unwrap() - 0.6).abs() < f32::EPSILON);
        assert_eq!(query.filters.get("language"), Some(&"en".to_string()));
        assert_eq!(query.filters.get("year"), Some(&"2024".to_string()));
        assert!(query.metadata_filter.is_some());
    }

    #[test]
    fn test_extended_query_serialization() {
        let extended = QueryBuilder::new()
            .text("test")
            .with_top_k(5)
            .with_cache_key("key")
            .with_layer_hints(LayerHints::echo_only())
            .build_extended()
            .expect("Failed to build");

        let json = serde_json::to_string(&extended).expect("Failed to serialize");
        let parsed: ExtendedQuery = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(parsed.query.text, "test");
        assert_eq!(parsed.query.top_k, 5);
        assert_eq!(parsed.cache_key, Some("key".to_string()));
    }
}
