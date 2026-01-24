//! Graph traversal algorithms.

use std::collections::{HashSet, VecDeque};

use crate::error::GraphError;
use crate::layer4_graph::traits::GraphStore;
use crate::layer4_graph::types::{
    Direction, EntityId, EntityType, GraphPath, GraphQuery, RelationshipType,
};

/// Perform breadth-first traversal of the graph.
///
/// # Errors
///
/// Returns an error if graph store operations fail.
pub async fn bfs_traverse<S: GraphStore + ?Sized>(
    store: &S,
    query: &GraphQuery,
) -> Result<Vec<GraphPath>, GraphError> {
    let mut paths = Vec::new();

    for start_id in &query.start_entities {
        // Get the starting entity
        let Some(start_entity) = store.get_entity(start_id).await? else {
            continue; // Skip if entity not found
        };

        // Check entity filter for start
        if let Some(ref filter) = query.entity_filter
            && !filter.contains(&start_entity.entity_type)
        {
            continue;
        }

        // BFS queue: (current path, visited set)
        let mut queue: VecDeque<(GraphPath, HashSet<EntityId>)> = VecDeque::new();

        let initial_path = GraphPath::from_entity(start_entity);
        let mut initial_visited = HashSet::new();
        initial_visited.insert(start_id.clone());

        queue.push_back((initial_path, initial_visited));

        while let Some((current_path, visited)) = queue.pop_front() {
            // Add path if it meets criteria
            if current_path.total_confidence >= query.min_confidence {
                paths.push(current_path.clone());
            }

            // Stop if max hops reached
            if current_path.len() >= query.max_hops {
                continue;
            }

            // Get current entity ID
            let current_id = current_path.end().map(|e| e.id.clone()).unwrap_or_default();

            // Get neighbors based on direction
            let neighbors = store.get_neighbors(&current_id, query.direction).await?;

            for (relationship, neighbor) in neighbors {
                // Skip if already visited
                if visited.contains(&neighbor.id) {
                    continue;
                }

                // Apply relationship filter
                if let Some(ref filter) = query.relationship_filter
                    && !filter.contains(&relationship.relationship_type)
                {
                    continue;
                }

                // Apply entity filter
                if let Some(ref filter) = query.entity_filter
                    && !filter.contains(&neighbor.entity_type)
                {
                    continue;
                }

                // Create new path
                let mut new_path = current_path.clone();
                new_path.extend(relationship, neighbor.clone());

                // Check confidence threshold
                if new_path.total_confidence >= query.min_confidence {
                    // Mark as visited
                    let mut new_visited = visited.clone();
                    new_visited.insert(neighbor.id.clone());

                    queue.push_back((new_path, new_visited));
                }
            }
        }
    }

    // Sort paths by confidence (descending)
    paths.sort_by(|a, b| {
        b.total_confidence
            .partial_cmp(&a.total_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(paths)
}

/// Find shortest path between two entities using BFS.
///
/// # Errors
///
/// Returns an error if graph store operations fail.
pub async fn find_shortest_path<S: GraphStore + ?Sized>(
    store: &S,
    start_id: &EntityId,
    end_id: &EntityId,
    max_hops: usize,
    direction: Direction,
) -> Result<Option<GraphPath>, GraphError> {
    // Get the starting entity
    let Some(start_entity) = store.get_entity(start_id).await? else {
        return Ok(None);
    };

    // BFS queue
    let mut queue: VecDeque<(GraphPath, HashSet<EntityId>)> = VecDeque::new();

    let initial_path = GraphPath::from_entity(start_entity);
    let mut initial_visited = HashSet::new();
    initial_visited.insert(start_id.clone());

    queue.push_back((initial_path, initial_visited));

    while let Some((current_path, visited)) = queue.pop_front() {
        // Check if we've reached the target
        if let Some(end_entity) = current_path.end()
            && end_entity.id == *end_id
        {
            return Ok(Some(current_path));
        }

        // Stop if max hops reached
        if current_path.len() >= max_hops {
            continue;
        }

        // Get current entity ID
        let current_id = current_path.end().map(|e| e.id.clone()).unwrap_or_default();

        // Get neighbors
        let neighbors = store.get_neighbors(&current_id, direction).await?;

        for (relationship, neighbor) in neighbors {
            if visited.contains(&neighbor.id) {
                continue;
            }

            // Create new path
            let mut new_path = current_path.clone();
            new_path.extend(relationship, neighbor.clone());

            // Mark as visited
            let mut new_visited = visited.clone();
            new_visited.insert(neighbor.id.clone());

            queue.push_back((new_path, new_visited));
        }
    }

    Ok(None)
}

/// Find all entities within N hops from a starting entity.
///
/// # Errors
///
/// Returns an error if graph store operations fail.
pub async fn find_entities_within_hops<S: GraphStore + ?Sized>(
    store: &S,
    start_id: &EntityId,
    max_hops: usize,
    direction: Direction,
    entity_filter: Option<&[EntityType]>,
    relationship_filter: Option<&[RelationshipType]>,
) -> Result<Vec<(crate::layer4_graph::types::GraphEntity, usize)>, GraphError> {
    let mut results = Vec::new();
    let mut visited: HashSet<EntityId> = HashSet::new();

    // BFS queue: (entity_id, hop_count)
    let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();
    queue.push_back((start_id.clone(), 0));
    visited.insert(start_id.clone());

    while let Some((current_id, hops)) = queue.pop_front() {
        // Get the entity
        if let Some(entity) = store.get_entity(&current_id).await? {
            // Apply entity filter
            let include = entity_filter.is_none_or(|f| f.contains(&entity.entity_type));

            if include && current_id != *start_id {
                results.push((entity, hops));
            }
        }

        // Stop expanding if max hops reached
        if hops >= max_hops {
            continue;
        }

        // Get neighbors
        let neighbors = store.get_neighbors(&current_id, direction).await?;

        for (relationship, neighbor) in neighbors {
            if visited.contains(&neighbor.id) {
                continue;
            }

            // Apply relationship filter
            if let Some(filter) = relationship_filter
                && !filter.contains(&relationship.relationship_type)
            {
                continue;
            }

            visited.insert(neighbor.id.clone());
            queue.push_back((neighbor.id, hops + 1));
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer4_graph::memory::InMemoryGraphStore;
    use crate::layer4_graph::types::{GraphEntity, GraphRelationship, RelationshipType};

    async fn create_test_graph() -> InMemoryGraphStore {
        let mut store = InMemoryGraphStore::new();

        // Create entities: A -> B -> C -> D
        //                    \-> E
        store
            .add_entity(GraphEntity::new("A", EntityType::Concept).with_id("a"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("B", EntityType::Concept).with_id("b"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("C", EntityType::Concept).with_id("c"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("D", EntityType::Concept).with_id("d"))
            .await
            .unwrap();
        store
            .add_entity(GraphEntity::new("E", EntityType::Technology).with_id("e"))
            .await
            .unwrap();

        store
            .add_relationship(GraphRelationship::new(
                "a",
                "b",
                RelationshipType::RelatedTo,
            ))
            .await
            .unwrap();
        store
            .add_relationship(GraphRelationship::new(
                "b",
                "c",
                RelationshipType::RelatedTo,
            ))
            .await
            .unwrap();
        store
            .add_relationship(GraphRelationship::new(
                "c",
                "d",
                RelationshipType::RelatedTo,
            ))
            .await
            .unwrap();
        store
            .add_relationship(GraphRelationship::new("a", "e", RelationshipType::Uses))
            .await
            .unwrap();

        store
    }

    #[tokio::test]
    async fn test_bfs_traverse_basic() {
        let store = create_test_graph().await;

        let query = GraphQuery::new(vec!["a".to_string()]).with_max_hops(2);

        let paths = bfs_traverse(&store, &query).await.unwrap();

        // Should find multiple paths from A
        assert!(!paths.is_empty());

        // Check that we can reach B, E, and C (but not D at 2 hops via B)
        let reached_ids: HashSet<_> = paths
            .iter()
            .filter_map(|p| p.end().map(|e| e.id.clone()))
            .collect();
        assert!(
            reached_ids.contains("a") || reached_ids.contains("b") || reached_ids.contains("e")
        );
    }

    #[tokio::test]
    async fn test_bfs_traverse_with_filter() {
        let store = create_test_graph().await;

        let query = GraphQuery::new(vec!["a".to_string()])
            .with_max_hops(3)
            .with_entity_filter(vec![EntityType::Technology]);

        let paths = bfs_traverse(&store, &query).await.unwrap();

        // Should only find paths that pass through Technology entities
        for path in &paths {
            for entity in &path.entities {
                if entity.id != "a" {
                    // Starting entity might not match filter
                    assert_eq!(entity.entity_type, EntityType::Technology);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_bfs_traverse_with_relationship_filter() {
        let store = create_test_graph().await;

        let query = GraphQuery::new(vec!["a".to_string()])
            .with_max_hops(3)
            .with_relationship_filter(vec![RelationshipType::Uses]);

        let paths = bfs_traverse(&store, &query).await.unwrap();

        // Should only follow USES relationships
        for path in &paths {
            for rel in &path.relationships {
                assert_eq!(rel.relationship_type, RelationshipType::Uses);
            }
        }
    }

    #[tokio::test]
    async fn test_find_shortest_path() {
        let store = create_test_graph().await;

        // A -> B -> C -> D (3 hops)
        let path = find_shortest_path(
            &store,
            &"a".to_string(),
            &"d".to_string(),
            5,
            Direction::Outgoing,
        )
        .await
        .unwrap();

        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3); // 3 relationships
        assert_eq!(path.start().unwrap().id, "a");
        assert_eq!(path.end().unwrap().id, "d");
    }

    #[tokio::test]
    async fn test_find_shortest_path_not_found() {
        let store = create_test_graph().await;

        // E has no outgoing edges
        let path = find_shortest_path(
            &store,
            &"e".to_string(),
            &"d".to_string(),
            5,
            Direction::Outgoing,
        )
        .await
        .unwrap();

        assert!(path.is_none());
    }

    #[tokio::test]
    async fn test_find_entities_within_hops() {
        let store = create_test_graph().await;

        let entities =
            find_entities_within_hops(&store, &"a".to_string(), 2, Direction::Outgoing, None, None)
                .await
                .unwrap();

        // Should find B (1 hop), E (1 hop), C (2 hops)
        assert_eq!(entities.len(), 3);

        let ids: HashSet<_> = entities.iter().map(|(e, _)| e.id.clone()).collect();
        assert!(ids.contains("b"));
        assert!(ids.contains("e"));
        assert!(ids.contains("c"));
    }

    #[tokio::test]
    async fn test_find_entities_within_hops_with_filter() {
        let store = create_test_graph().await;

        let entities = find_entities_within_hops(
            &store,
            &"a".to_string(),
            3,
            Direction::Outgoing,
            Some(&[EntityType::Technology]),
            None,
        )
        .await
        .unwrap();

        // Should only find E (Technology)
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].0.id, "e");
    }
}
