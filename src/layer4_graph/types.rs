//! Types for the Graph (knowledge graph) layer.

use crate::types::DocumentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for graph entities.
pub type EntityId = String;

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEntity {
    /// Unique identifier for this entity.
    pub id: EntityId,
    /// Human-readable name of the entity.
    pub name: String,
    /// The type/category of this entity.
    pub entity_type: EntityType,
    /// Additional properties as key-value pairs.
    pub properties: HashMap<String, String>,
    /// The document this entity was extracted from.
    pub source_doc_id: Option<DocumentId>,
    /// Confidence score of the extraction (0.0 to 1.0).
    pub confidence: f32,
}

impl GraphEntity {
    /// Create a new graph entity.
    #[must_use]
    pub fn new(name: impl Into<String>, entity_type: EntityType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            entity_type,
            properties: HashMap::new(),
            source_doc_id: None,
            confidence: 1.0,
        }
    }

    /// Set the entity ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add a property.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the source document ID.
    #[must_use]
    pub fn with_source(mut self, doc_id: DocumentId) -> Self {
        self.source_doc_id = Some(doc_id);
        self
    }

    /// Set the confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Entity type classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// A person or individual.
    Person,
    /// An organization, company, or institution.
    Organization,
    /// A geographical location.
    Location,
    /// A date or time reference.
    Date,
    /// An event or occurrence.
    Event,
    /// An abstract concept or idea.
    Concept,
    /// A technical term or technology.
    Technology,
    /// A custom entity type.
    Custom(String),
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Person => write!(f, "Person"),
            Self::Organization => write!(f, "Organization"),
            Self::Location => write!(f, "Location"),
            Self::Date => write!(f, "Date"),
            Self::Event => write!(f, "Event"),
            Self::Concept => write!(f, "Concept"),
            Self::Technology => write!(f, "Technology"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelationship {
    /// Unique identifier for this relationship.
    pub id: String,
    /// The source entity ID.
    pub source_id: EntityId,
    /// The target entity ID.
    pub target_id: EntityId,
    /// The type of relationship.
    pub relationship_type: RelationshipType,
    /// Additional properties as key-value pairs.
    pub properties: HashMap<String, String>,
    /// Confidence score of the extraction (0.0 to 1.0).
    pub confidence: f32,
}

impl GraphRelationship {
    /// Create a new graph relationship.
    #[must_use]
    pub fn new(
        source_id: impl Into<EntityId>,
        target_id: impl Into<EntityId>,
        relationship_type: RelationshipType,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            relationship_type,
            properties: HashMap::new(),
            confidence: 1.0,
        }
    }

    /// Set the relationship ID.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add a property.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Set the confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Relationship type classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Inheritance or subtype relationship.
    IsA,
    /// Part-whole relationship.
    PartOf,
    /// Membership relationship.
    MemberOf,
    /// Location containment.
    LocatedIn,
    /// Temporal occurrence.
    OccurredAt,
    /// Works for/employment.
    WorksFor,
    /// Founded by relationship.
    FoundedBy,
    /// Created by relationship.
    CreatedBy,
    /// Uses or utilizes.
    Uses,
    /// General relation.
    RelatedTo,
    /// Causal relationship.
    Causes,
    /// Dependency relationship.
    DependsOn,
    /// A custom relationship type.
    Custom(String),
}

impl std::fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IsA => write!(f, "IS_A"),
            Self::PartOf => write!(f, "PART_OF"),
            Self::MemberOf => write!(f, "MEMBER_OF"),
            Self::LocatedIn => write!(f, "LOCATED_IN"),
            Self::OccurredAt => write!(f, "OCCURRED_AT"),
            Self::WorksFor => write!(f, "WORKS_FOR"),
            Self::FoundedBy => write!(f, "FOUNDED_BY"),
            Self::CreatedBy => write!(f, "CREATED_BY"),
            Self::Uses => write!(f, "USES"),
            Self::RelatedTo => write!(f, "RELATED_TO"),
            Self::Causes => write!(f, "CAUSES"),
            Self::DependsOn => write!(f, "DEPENDS_ON"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A path through the knowledge graph.
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// The entities in the path, in order.
    pub entities: Vec<GraphEntity>,
    /// The relationships connecting the entities.
    pub relationships: Vec<GraphRelationship>,
    /// Combined confidence score for the path.
    pub total_confidence: f32,
}

impl GraphPath {
    /// Create a new empty path.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relationships: Vec::new(),
            total_confidence: 1.0,
        }
    }

    /// Create a path with a single entity.
    #[must_use]
    pub fn from_entity(entity: GraphEntity) -> Self {
        let confidence = entity.confidence;
        Self {
            entities: vec![entity],
            relationships: Vec::new(),
            total_confidence: confidence,
        }
    }

    /// Add an entity and relationship to the path.
    pub fn extend(&mut self, relationship: GraphRelationship, entity: GraphEntity) {
        self.total_confidence *= relationship.confidence * entity.confidence;
        self.relationships.push(relationship);
        self.entities.push(entity);
    }

    /// Get the length of the path (number of hops).
    #[must_use]
    pub fn len(&self) -> usize {
        self.relationships.len()
    }

    /// Check if the path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the starting entity.
    #[must_use]
    pub fn start(&self) -> Option<&GraphEntity> {
        self.entities.first()
    }

    /// Get the ending entity.
    #[must_use]
    pub fn end(&self) -> Option<&GraphEntity> {
        self.entities.last()
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Direction for graph traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Direction {
    /// Follow outgoing edges only.
    Outgoing,
    /// Follow incoming edges only.
    Incoming,
    /// Follow edges in both directions.
    #[default]
    Both,
}

/// Query for graph traversal.
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// Starting entity IDs for traversal.
    pub start_entities: Vec<EntityId>,
    /// Maximum number of hops to traverse.
    pub max_hops: usize,
    /// Optional filter for relationship types.
    pub relationship_filter: Option<Vec<RelationshipType>>,
    /// Optional filter for entity types.
    pub entity_filter: Option<Vec<EntityType>>,
    /// Minimum confidence threshold for paths.
    pub min_confidence: f32,
    /// Traversal direction.
    pub direction: Direction,
}

impl GraphQuery {
    /// Create a new graph query.
    #[must_use]
    pub fn new(start_entities: Vec<EntityId>) -> Self {
        Self {
            start_entities,
            max_hops: 2,
            relationship_filter: None,
            entity_filter: None,
            min_confidence: 0.0,
            direction: Direction::Both,
        }
    }

    /// Set the maximum number of hops.
    #[must_use]
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    /// Set the relationship filter.
    #[must_use]
    pub fn with_relationship_filter(mut self, types: Vec<RelationshipType>) -> Self {
        self.relationship_filter = Some(types);
        self
    }

    /// Set the entity filter.
    #[must_use]
    pub fn with_entity_filter(mut self, types: Vec<EntityType>) -> Self {
        self.entity_filter = Some(types);
        self
    }

    /// Set the minimum confidence.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set the traversal direction.
    #[must_use]
    pub fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }
}

/// Combined semantic + graph search result for hybrid queries.
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Semantic search results from the Echo layer.
    pub semantic_results: Vec<crate::types::SearchResult>,
    /// Graph traversal paths.
    pub graph_paths: Vec<GraphPath>,
    /// Merged score combining both sources.
    pub merged_score: f32,
}

impl HybridSearchResult {
    /// Create a new hybrid search result.
    #[must_use]
    pub fn new(
        semantic_results: Vec<crate::types::SearchResult>,
        graph_paths: Vec<GraphPath>,
    ) -> Self {
        let semantic_score = semantic_results.first().map_or(0.0, |r| r.score);
        let graph_score = graph_paths.first().map_or(0.0, |p| p.total_confidence);
        let merged_score = f32::midpoint(semantic_score, graph_score);

        Self {
            semantic_results,
            graph_paths,
            merged_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let entity = GraphEntity::new("Rust", EntityType::Technology)
            .with_property("paradigm", "systems")
            .with_confidence(0.95);

        assert_eq!(entity.name, "Rust");
        assert_eq!(entity.entity_type, EntityType::Technology);
        assert_eq!(
            entity.properties.get("paradigm"),
            Some(&"systems".to_string())
        );
        assert!((entity.confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_relationship_creation() {
        let rel = GraphRelationship::new("rust", "llvm", RelationshipType::Uses)
            .with_property("since", "1.0")
            .with_confidence(0.9);

        assert_eq!(rel.source_id, "rust");
        assert_eq!(rel.target_id, "llvm");
        assert_eq!(rel.relationship_type, RelationshipType::Uses);
        assert_eq!(rel.properties.get("since"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_graph_path() {
        let mut path = GraphPath::new();
        assert!(path.is_empty());

        let entity1 = GraphEntity::new("A", EntityType::Concept).with_confidence(0.9);
        let entity2 = GraphEntity::new("B", EntityType::Concept).with_confidence(0.8);
        let rel =
            GraphRelationship::new("a", "b", RelationshipType::RelatedTo).with_confidence(0.85);

        path.entities.push(entity1);
        path.extend(rel, entity2);

        assert_eq!(path.len(), 1);
        assert!(!path.is_empty());
        assert_eq!(path.start().unwrap().name, "A");
        assert_eq!(path.end().unwrap().name, "B");
    }

    #[test]
    fn test_graph_query_builder() {
        let query = GraphQuery::new(vec!["start".to_string()])
            .with_max_hops(3)
            .with_min_confidence(0.5)
            .with_direction(Direction::Outgoing);

        assert_eq!(query.max_hops, 3);
        assert!((query.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(query.direction, Direction::Outgoing);
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::Person.to_string(), "Person");
        assert_eq!(
            EntityType::Custom("Custom".to_string()).to_string(),
            "Custom"
        );
    }

    #[test]
    fn test_relationship_type_display() {
        assert_eq!(RelationshipType::IsA.to_string(), "IS_A");
        assert_eq!(RelationshipType::PartOf.to_string(), "PART_OF");
    }
}
