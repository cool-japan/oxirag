//! In-memory graph storage implementation.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};

use crate::error::GraphError;
use crate::layer4_graph::traits::GraphStore;
use crate::layer4_graph::traversal::bfs_traverse;
use crate::layer4_graph::types::{
    Direction, EntityId, EntityType, GraphEntity, GraphPath, GraphQuery, GraphRelationship,
};

/// In-memory implementation of `GraphStore`.
#[derive(Debug, Default)]
pub struct InMemoryGraphStore {
    /// Entities indexed by ID.
    entities: HashMap<EntityId, GraphEntity>,
    /// Relationships indexed by ID.
    relationships: HashMap<String, GraphRelationship>,
    /// Adjacency list: entity ID -> outgoing relationship IDs.
    outgoing: HashMap<EntityId, HashSet<String>>,
    /// Reverse adjacency list: entity ID -> incoming relationship IDs.
    incoming: HashMap<EntityId, HashSet<String>>,
    /// Index: entity name (lowercase) -> entity IDs.
    name_index: HashMap<String, HashSet<EntityId>>,
    /// Index: entity type -> entity IDs.
    type_index: HashMap<EntityType, HashSet<EntityId>>,
}

impl InMemoryGraphStore {
    /// Create a new empty in-memory graph store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the internal entities map (for testing).
    #[cfg(test)]
    #[must_use]
    pub fn entities(&self) -> &HashMap<EntityId, GraphEntity> {
        &self.entities
    }

    /// Get the internal relationships map (for testing).
    #[cfg(test)]
    #[must_use]
    pub fn relationships_map(&self) -> &HashMap<String, GraphRelationship> {
        &self.relationships
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn add_entity(&mut self, entity: GraphEntity) -> Result<EntityId, GraphError> {
        let id = entity.id.clone();

        // Update name index
        let name_lower = entity.name.to_lowercase();
        self.name_index
            .entry(name_lower)
            .or_default()
            .insert(id.clone());

        // Update type index
        self.type_index
            .entry(entity.entity_type.clone())
            .or_default()
            .insert(id.clone());

        // Initialize adjacency lists
        self.outgoing.entry(id.clone()).or_default();
        self.incoming.entry(id.clone()).or_default();

        // Store entity
        self.entities.insert(id.clone(), entity);

        Ok(id)
    }

    async fn add_entities(
        &mut self,
        entities: Vec<GraphEntity>,
    ) -> Result<Vec<EntityId>, GraphError> {
        let mut ids = Vec::with_capacity(entities.len());
        for entity in entities {
            let id = self.add_entity(entity).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn add_relationship(
        &mut self,
        relationship: GraphRelationship,
    ) -> Result<String, GraphError> {
        let rel_id = relationship.id.clone();
        let source_id = relationship.source_id.clone();
        let target_id = relationship.target_id.clone();

        // Verify source and target exist
        if !self.entities.contains_key(&source_id) {
            return Err(GraphError::EntityNotFound(source_id));
        }
        if !self.entities.contains_key(&target_id) {
            return Err(GraphError::EntityNotFound(target_id));
        }

        // Update adjacency lists
        self.outgoing
            .entry(source_id)
            .or_default()
            .insert(rel_id.clone());
        self.incoming
            .entry(target_id)
            .or_default()
            .insert(rel_id.clone());

        // Store relationship
        self.relationships.insert(rel_id.clone(), relationship);

        Ok(rel_id)
    }

    async fn add_relationships(
        &mut self,
        relationships: Vec<GraphRelationship>,
    ) -> Result<Vec<String>, GraphError> {
        let mut ids = Vec::with_capacity(relationships.len());
        for rel in relationships {
            let id = self.add_relationship(rel).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn get_entity(&self, id: &EntityId) -> Result<Option<GraphEntity>, GraphError> {
        Ok(self.entities.get(id).cloned())
    }

    async fn get_neighbors(
        &self,
        id: &EntityId,
        direction: Direction,
    ) -> Result<Vec<(GraphRelationship, GraphEntity)>, GraphError> {
        let mut neighbors = Vec::new();

        match direction {
            Direction::Outgoing | Direction::Both => {
                if let Some(rel_ids) = self.outgoing.get(id) {
                    for rel_id in rel_ids {
                        if let Some(rel) = self.relationships.get(rel_id)
                            && let Some(entity) = self.entities.get(&rel.target_id)
                        {
                            neighbors.push((rel.clone(), entity.clone()));
                        }
                    }
                }
            }
            Direction::Incoming => {}
        }

        match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(rel_ids) = self.incoming.get(id) {
                    for rel_id in rel_ids {
                        if let Some(rel) = self.relationships.get(rel_id)
                            && let Some(entity) = self.entities.get(&rel.source_id)
                        {
                            neighbors.push((rel.clone(), entity.clone()));
                        }
                    }
                }
            }
            Direction::Outgoing => {}
        }

        Ok(neighbors)
    }

    async fn find_entities_by_type(
        &self,
        entity_type: &EntityType,
    ) -> Result<Vec<GraphEntity>, GraphError> {
        let entities = self
            .type_index
            .get(entity_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.entities.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default();

        Ok(entities)
    }

    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<GraphEntity>, GraphError> {
        let name_lower = name.to_lowercase();
        let mut results = Vec::new();

        // Exact match
        if let Some(ids) = self.name_index.get(&name_lower) {
            for id in ids {
                if let Some(entity) = self.entities.get(id) {
                    results.push(entity.clone());
                }
            }
        }

        // Partial match if no exact matches
        if results.is_empty() {
            for (indexed_name, ids) in &self.name_index {
                if indexed_name.contains(&name_lower) || name_lower.contains(indexed_name) {
                    for id in ids {
                        if let Some(entity) = self.entities.get(id) {
                            results.push(entity.clone());
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    async fn traverse(&self, query: &GraphQuery) -> Result<Vec<GraphPath>, GraphError> {
        bfs_traverse(self, query).await
    }

    async fn entity_count(&self) -> usize {
        self.entities.len()
    }

    async fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    async fn clear(&mut self) -> Result<(), GraphError> {
        self.entities.clear();
        self.relationships.clear();
        self.outgoing.clear();
        self.incoming.clear();
        self.name_index.clear();
        self.type_index.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_get_entity() {
        let mut store = InMemoryGraphStore::new();

        let entity = GraphEntity::new("Rust", EntityType::Technology).with_id("rust");

        let id = store.add_entity(entity).await.unwrap();
        assert_eq!(id, "rust");

        let retrieved = store.get_entity(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Rust");
    }

    #[tokio::test]
    async fn test_add_relationship() {
        let mut store = InMemoryGraphStore::new();

        let rust = GraphEntity::new("Rust", EntityType::Technology).with_id("rust");
        let llvm = GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm");

        store.add_entity(rust).await.unwrap();
        store.add_entity(llvm).await.unwrap();

        let rel = GraphRelationship::new(
            "rust",
            "llvm",
            crate::layer4_graph::types::RelationshipType::Uses,
        );

        let rel_id = store.add_relationship(rel).await.unwrap();
        assert!(!rel_id.is_empty());
    }

    #[tokio::test]
    async fn test_add_relationship_missing_entity() {
        let mut store = InMemoryGraphStore::new();

        let rust = GraphEntity::new("Rust", EntityType::Technology).with_id("rust");
        store.add_entity(rust).await.unwrap();

        let rel = GraphRelationship::new(
            "rust",
            "nonexistent",
            crate::layer4_graph::types::RelationshipType::Uses,
        );

        let result = store.add_relationship(rel).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_neighbors() {
        let mut store = InMemoryGraphStore::new();

        let rust = GraphEntity::new("Rust", EntityType::Technology).with_id("rust");
        let llvm = GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm");
        let cargo = GraphEntity::new("Cargo", EntityType::Technology).with_id("cargo");

        store.add_entity(rust).await.unwrap();
        store.add_entity(llvm).await.unwrap();
        store.add_entity(cargo).await.unwrap();

        store
            .add_relationship(GraphRelationship::new(
                "rust",
                "llvm",
                crate::layer4_graph::types::RelationshipType::Uses,
            ))
            .await
            .unwrap();
        store
            .add_relationship(GraphRelationship::new(
                "rust",
                "cargo",
                crate::layer4_graph::types::RelationshipType::Uses,
            ))
            .await
            .unwrap();

        let neighbors = store
            .get_neighbors(&"rust".to_string(), Direction::Outgoing)
            .await
            .unwrap();
        assert_eq!(neighbors.len(), 2);

        let incoming = store
            .get_neighbors(&"rust".to_string(), Direction::Incoming)
            .await
            .unwrap();
        assert!(incoming.is_empty());

        let llvm_incoming = store
            .get_neighbors(&"llvm".to_string(), Direction::Incoming)
            .await
            .unwrap();
        assert_eq!(llvm_incoming.len(), 1);
    }

    #[tokio::test]
    async fn test_find_by_type() {
        let mut store = InMemoryGraphStore::new();

        store
            .add_entity(GraphEntity::new("Rust", EntityType::Technology))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("Python", EntityType::Technology))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("Mozilla", EntityType::Organization))
            .await
            .unwrap();

        let tech = store
            .find_entities_by_type(&EntityType::Technology)
            .await
            .unwrap();
        assert_eq!(tech.len(), 2);

        let orgs = store
            .find_entities_by_type(&EntityType::Organization)
            .await
            .unwrap();
        assert_eq!(orgs.len(), 1);
    }

    #[tokio::test]
    async fn test_find_by_name() {
        let mut store = InMemoryGraphStore::new();

        store
            .add_entity(GraphEntity::new("Rust Language", EntityType::Technology))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("Rusty", EntityType::Person))
            .await
            .unwrap();

        // Exact match (case-insensitive)
        let exact = store.find_entities_by_name("rust language").await.unwrap();
        assert_eq!(exact.len(), 1);

        // Partial match
        let partial = store.find_entities_by_name("rust").await.unwrap();
        assert_eq!(partial.len(), 2);
    }

    #[tokio::test]
    async fn test_counts() {
        let mut store = InMemoryGraphStore::new();

        assert_eq!(store.entity_count().await, 0);
        assert_eq!(store.relationship_count().await, 0);

        store
            .add_entity(GraphEntity::new("A", EntityType::Concept).with_id("a"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("B", EntityType::Concept).with_id("b"))
            .await
            .unwrap();
        store
            .add_relationship(GraphRelationship::new(
                "a",
                "b",
                crate::layer4_graph::types::RelationshipType::RelatedTo,
            ))
            .await
            .unwrap();

        assert_eq!(store.entity_count().await, 2);
        assert_eq!(store.relationship_count().await, 1);
    }

    #[tokio::test]
    async fn test_clear() {
        let mut store = InMemoryGraphStore::new();

        store
            .add_entity(GraphEntity::new("A", EntityType::Concept).with_id("a"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("B", EntityType::Concept).with_id("b"))
            .await
            .unwrap();

        store.clear().await.unwrap();

        assert_eq!(store.entity_count().await, 0);
        assert_eq!(store.relationship_count().await, 0);
    }
}
