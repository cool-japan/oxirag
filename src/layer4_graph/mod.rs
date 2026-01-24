//! Layer 4: Graph - Knowledge graph-based retrieval.
//!
//! The Graph layer provides knowledge graph capabilities including:
//! - Entity extraction from text
//! - Relationship extraction between entities
//! - Graph storage and traversal
//! - Multi-hop reasoning over knowledge graphs

pub mod extractor;
pub mod memory;
pub mod traits;
pub mod traversal;
pub mod types;

pub use extractor::{
    MockEntityExtractor, MockRelationshipExtractor, PatternEntityExtractor,
    PatternRelationshipExtractor,
};
pub use memory::InMemoryGraphStore;
pub use traits::{EntityExtractor, Graph, GraphStore, RelationshipExtractor};
pub use traversal::{bfs_traverse, find_entities_within_hops, find_shortest_path};
pub use types::{
    Direction, EntityId, EntityType, GraphEntity, GraphPath, GraphQuery, GraphRelationship,
    HybridSearchResult, RelationshipType,
};

use async_trait::async_trait;

use crate::error::GraphError;
use crate::types::Document;

/// The default Graph implementation combining entity/relationship extractors and graph storage.
pub struct GraphLayer<E: EntityExtractor, R: RelationshipExtractor, S: GraphStore> {
    entity_extractor: E,
    relationship_extractor: R,
    store: S,
}

impl<E: EntityExtractor, R: RelationshipExtractor, S: GraphStore> GraphLayer<E, R, S> {
    /// Create a new Graph layer with the given extractors and store.
    #[must_use]
    pub fn new(entity_extractor: E, relationship_extractor: R, store: S) -> Self {
        Self {
            entity_extractor,
            relationship_extractor,
            store,
        }
    }

    /// Get a reference to the entity extractor.
    #[must_use]
    pub fn entity_extractor(&self) -> &E {
        &self.entity_extractor
    }

    /// Get a reference to the relationship extractor.
    #[must_use]
    pub fn relationship_extractor(&self) -> &R {
        &self.relationship_extractor
    }

    /// Get a reference to the graph store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get a mutable reference to the graph store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}

#[async_trait]
impl<E: EntityExtractor, R: RelationshipExtractor, S: GraphStore> Graph for GraphLayer<E, R, S> {
    async fn index_document(&mut self, document: &Document) -> Result<(), GraphError> {
        // Extract entities from document content
        let mut entities = self
            .entity_extractor
            .extract_entities(&document.content)
            .await?;

        // Set source document ID for all entities
        for entity in &mut entities {
            entity.source_doc_id = Some(document.id.clone());
        }

        // Add entities to store (need to add before extracting relationships)
        let entity_ids = self.store.add_entities(entities.clone()).await?;

        // Re-fetch entities with their assigned IDs for relationship extraction
        let stored_entities: Vec<GraphEntity> = {
            let mut result = Vec::with_capacity(entity_ids.len());
            for id in &entity_ids {
                if let Some(entity) = self.store.get_entity(id).await? {
                    result.push(entity);
                }
            }
            result
        };

        // Extract relationships between entities
        let relationships = self
            .relationship_extractor
            .extract_relationships(&document.content, &stored_entities)
            .await?;

        // Add relationships to store
        self.store.add_relationships(relationships).await?;

        Ok(())
    }

    async fn index_documents(&mut self, documents: &[Document]) -> Result<(), GraphError> {
        for document in documents {
            self.index_document(document).await?;
        }
        Ok(())
    }

    async fn query(&self, query: &GraphQuery) -> Result<Vec<GraphPath>, GraphError> {
        self.store.traverse(query).await
    }

    async fn find_related(
        &self,
        entity_name: &str,
        max_hops: usize,
    ) -> Result<Vec<GraphPath>, GraphError> {
        // Find entities by name
        let entities = self.store.find_entities_by_name(entity_name).await?;

        if entities.is_empty() {
            return Ok(Vec::new());
        }

        // Build query from found entities
        let start_ids: Vec<EntityId> = entities.into_iter().map(|e| e.id).collect();

        let query = GraphQuery::new(start_ids).with_max_hops(max_hops);

        self.store.traverse(&query).await
    }

    async fn entity_count(&self) -> usize {
        self.store.entity_count().await
    }

    async fn relationship_count(&self) -> usize {
        self.store.relationship_count().await
    }

    async fn clear(&mut self) -> Result<(), GraphError> {
        self.store.clear().await
    }
}

/// Builder for constructing a `GraphLayer`.
#[derive(Default)]
pub struct GraphLayerBuilder<E, R, S> {
    entity_extractor: Option<E>,
    relationship_extractor: Option<R>,
    store: Option<S>,
}

impl<E: EntityExtractor, R: RelationshipExtractor, S: GraphStore> GraphLayerBuilder<E, R, S> {
    /// Create a new graph layer builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entity_extractor: None,
            relationship_extractor: None,
            store: None,
        }
    }

    /// Set the entity extractor.
    #[must_use]
    pub fn with_entity_extractor(mut self, extractor: E) -> Self {
        self.entity_extractor = Some(extractor);
        self
    }

    /// Set the relationship extractor.
    #[must_use]
    pub fn with_relationship_extractor(mut self, extractor: R) -> Self {
        self.relationship_extractor = Some(extractor);
        self
    }

    /// Set the graph store.
    #[must_use]
    pub fn with_store(mut self, store: S) -> Self {
        self.store = Some(store);
        self
    }

    /// Build the graph layer.
    ///
    /// # Errors
    ///
    /// Returns an error if any required component is not configured.
    pub fn build(self) -> Result<GraphLayer<E, R, S>, GraphError> {
        let entity_extractor = self.entity_extractor.ok_or_else(|| {
            GraphError::ConfigurationError("Entity extractor not configured".to_string())
        })?;
        let relationship_extractor = self.relationship_extractor.ok_or_else(|| {
            GraphError::ConfigurationError("Relationship extractor not configured".to_string())
        })?;
        let store = self.store.ok_or_else(|| {
            GraphError::ConfigurationError("Graph store not configured".to_string())
        })?;

        Ok(GraphLayer::new(
            entity_extractor,
            relationship_extractor,
            store,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph_layer()
    -> GraphLayer<MockEntityExtractor, MockRelationshipExtractor, InMemoryGraphStore> {
        GraphLayer::new(
            MockEntityExtractor::new(),
            MockRelationshipExtractor::new(),
            InMemoryGraphStore::new(),
        )
    }

    #[tokio::test]
    async fn test_graph_layer_creation() {
        let layer = create_test_graph_layer();
        assert_eq!(layer.entity_count().await, 0);
        assert_eq!(layer.relationship_count().await, 0);
    }

    #[tokio::test]
    async fn test_graph_layer_with_mock_extractors() {
        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm"),
        ];
        let relationships = vec![GraphRelationship::new(
            "rust",
            "llvm",
            RelationshipType::Uses,
        )];

        let mut layer = GraphLayer::new(
            MockEntityExtractor::with_entities(entities),
            MockRelationshipExtractor::with_relationships(relationships),
            InMemoryGraphStore::new(),
        );

        let doc = Document::new("Rust uses LLVM for code generation.");
        layer.index_document(&doc).await.unwrap();

        assert_eq!(layer.entity_count().await, 2);
        assert_eq!(layer.relationship_count().await, 1);
    }

    #[tokio::test]
    async fn test_graph_layer_query() {
        let entities = vec![
            GraphEntity::new("A", EntityType::Concept).with_id("a"),
            GraphEntity::new("B", EntityType::Concept).with_id("b"),
        ];
        let relationships = vec![GraphRelationship::new(
            "a",
            "b",
            RelationshipType::RelatedTo,
        )];

        let mut layer = GraphLayer::new(
            MockEntityExtractor::with_entities(entities),
            MockRelationshipExtractor::with_relationships(relationships),
            InMemoryGraphStore::new(),
        );

        let doc = Document::new("A relates to B.");
        layer.index_document(&doc).await.unwrap();

        let query = GraphQuery::new(vec!["a".to_string()]).with_max_hops(1);
        let paths = layer.query(&query).await.unwrap();

        assert!(!paths.is_empty());
    }

    #[tokio::test]
    async fn test_graph_layer_find_related() {
        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("Cargo", EntityType::Technology).with_id("cargo"),
        ];
        let relationships = vec![GraphRelationship::new(
            "rust",
            "cargo",
            RelationshipType::Uses,
        )];

        let mut layer = GraphLayer::new(
            MockEntityExtractor::with_entities(entities),
            MockRelationshipExtractor::with_relationships(relationships),
            InMemoryGraphStore::new(),
        );

        let doc = Document::new("Rust uses Cargo.");
        layer.index_document(&doc).await.unwrap();

        let paths = layer.find_related("Rust", 2).await.unwrap();
        assert!(!paths.is_empty());
    }

    #[tokio::test]
    async fn test_graph_layer_clear() {
        let mut layer = create_test_graph_layer();

        // Add some data manually through the store
        layer
            .store_mut()
            .add_entity(GraphEntity::new("Test", EntityType::Concept))
            .await
            .unwrap();

        assert_eq!(layer.entity_count().await, 1);

        layer.clear().await.unwrap();
        assert_eq!(layer.entity_count().await, 0);
    }

    #[tokio::test]
    async fn test_graph_layer_builder() {
        let layer = GraphLayerBuilder::new()
            .with_entity_extractor(MockEntityExtractor::new())
            .with_relationship_extractor(MockRelationshipExtractor::new())
            .with_store(InMemoryGraphStore::new())
            .build()
            .unwrap();

        assert_eq!(layer.entity_count().await, 0);
    }

    #[tokio::test]
    async fn test_graph_layer_builder_missing_component() {
        let result = GraphLayerBuilder::<
            MockEntityExtractor,
            MockRelationshipExtractor,
            InMemoryGraphStore,
        >::new()
        .with_entity_extractor(MockEntityExtractor::new())
        .build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_index_multiple_documents() {
        // Create extractors with unique entities for each document
        // Note: MockEntityExtractor returns clones with same IDs, so entities are deduplicated
        let mut extractor = MockEntityExtractor::new();
        // Add two entities with different IDs so each document contributes
        extractor.add_entity(GraphEntity::new("Entity1", EntityType::Concept).with_id("e1"));
        extractor.add_entity(GraphEntity::new("Entity2", EntityType::Concept).with_id("e2"));

        let mut layer = GraphLayer::new(
            extractor,
            MockRelationshipExtractor::new(),
            InMemoryGraphStore::new(),
        );

        let docs = vec![
            Document::new("First document"),
            Document::new("Second document"),
        ];

        layer.index_documents(&docs).await.unwrap();

        // Each document returns 2 entities, but same IDs get deduplicated to 2 total
        // (same entities returned for both docs, overwrite happens on second doc)
        assert_eq!(layer.entity_count().await, 2);
    }

    #[tokio::test]
    async fn test_pattern_based_extraction() {
        let mut layer = GraphLayer::new(
            PatternEntityExtractor::new(),
            PatternRelationshipExtractor::new(),
            InMemoryGraphStore::new(),
        );

        let doc = Document::new("Rust is a programming language. Rust uses LLVM for compilation.");
        layer.index_document(&doc).await.unwrap();

        // Should extract at least "Rust" as a technology entity
        assert!(layer.entity_count().await > 0);
    }
}
