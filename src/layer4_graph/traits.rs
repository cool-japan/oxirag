//! Traits for the Graph (knowledge graph) layer.

use async_trait::async_trait;

use crate::error::GraphError;
use crate::types::Document;

use super::types::{
    Direction, EntityId, EntityType, GraphEntity, GraphPath, GraphQuery, GraphRelationship,
    RelationshipType,
};

/// Extracts entities from text.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract entities from the given text.
    async fn extract_entities(&self, text: &str) -> Result<Vec<GraphEntity>, GraphError>;

    /// Get the entity types supported by this extractor.
    fn supported_entity_types(&self) -> Vec<EntityType>;
}

/// Extracts relationships between entities from text.
#[async_trait]
pub trait RelationshipExtractor: Send + Sync {
    /// Extract relationships between the given entities from text.
    async fn extract_relationships(
        &self,
        text: &str,
        entities: &[GraphEntity],
    ) -> Result<Vec<GraphRelationship>, GraphError>;

    /// Get the relationship types supported by this extractor.
    fn supported_relationship_types(&self) -> Vec<RelationshipType>;
}

/// Storage for knowledge graph data with traversal capabilities.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Add an entity to the graph.
    async fn add_entity(&mut self, entity: GraphEntity) -> Result<EntityId, GraphError>;

    /// Add multiple entities to the graph.
    async fn add_entities(
        &mut self,
        entities: Vec<GraphEntity>,
    ) -> Result<Vec<EntityId>, GraphError>;

    /// Add a relationship to the graph.
    async fn add_relationship(
        &mut self,
        relationship: GraphRelationship,
    ) -> Result<String, GraphError>;

    /// Add multiple relationships to the graph.
    async fn add_relationships(
        &mut self,
        relationships: Vec<GraphRelationship>,
    ) -> Result<Vec<String>, GraphError>;

    /// Get an entity by its ID.
    async fn get_entity(&self, id: &EntityId) -> Result<Option<GraphEntity>, GraphError>;

    /// Get neighboring entities connected by relationships.
    async fn get_neighbors(
        &self,
        id: &EntityId,
        direction: Direction,
    ) -> Result<Vec<(GraphRelationship, GraphEntity)>, GraphError>;

    /// Find entities by their type.
    async fn find_entities_by_type(
        &self,
        entity_type: &EntityType,
    ) -> Result<Vec<GraphEntity>, GraphError>;

    /// Find entities by name (exact or partial match).
    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<GraphEntity>, GraphError>;

    /// Traverse the graph from starting entities.
    async fn traverse(&self, query: &GraphQuery) -> Result<Vec<GraphPath>, GraphError>;

    /// Get the total number of entities in the graph.
    async fn entity_count(&self) -> usize;

    /// Get the total number of relationships in the graph.
    async fn relationship_count(&self) -> usize;

    /// Clear all entities and relationships from the graph.
    async fn clear(&mut self) -> Result<(), GraphError>;
}

/// The Graph layer: combines entity extraction, relationship extraction, and graph storage.
#[async_trait]
pub trait Graph: Send + Sync {
    /// Index a document by extracting entities and relationships.
    async fn index_document(&mut self, document: &Document) -> Result<(), GraphError>;

    /// Index multiple documents.
    async fn index_documents(&mut self, documents: &[Document]) -> Result<(), GraphError>;

    /// Query the graph for related entities and paths.
    async fn query(&self, query: &GraphQuery) -> Result<Vec<GraphPath>, GraphError>;

    /// Find entities related to a given entity name.
    async fn find_related(
        &self,
        entity_name: &str,
        max_hops: usize,
    ) -> Result<Vec<GraphPath>, GraphError>;

    /// Get entity count.
    async fn entity_count(&self) -> usize;

    /// Get relationship count.
    async fn relationship_count(&self) -> usize;

    /// Clear the graph.
    async fn clear(&mut self) -> Result<(), GraphError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that traits compile correctly
    #[test]
    fn test_trait_object_safety() {
        fn _assert_entity_extractor_object_safe(_: &dyn EntityExtractor) {}
        fn _assert_relationship_extractor_object_safe(_: &dyn RelationshipExtractor) {}
        fn _assert_graph_store_object_safe(_: &dyn GraphStore) {}
        fn _assert_graph_object_safe(_: &dyn Graph) {}
    }
}
