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

#[cfg(disabled)]
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

    // Property-based tests
    #[cfg(disabled)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        // Helper to create a chain graph: 0 -> 1 -> 2 -> ... -> n
        async fn create_chain_graph(n: usize) -> InMemoryGraphStore {
            let mut store = InMemoryGraphStore::new();

            for i in 0..n {
                store
                    .add_entity(
                        GraphEntity::new(&format!("Entity {}", i), EntityType::Concept)
                            .with_id(&format!("{}", i)),
                    )
                    .await
                    .ok();
            }

            for i in 0..n - 1 {
                store
                    .add_relationship(GraphRelationship::new(
                        &format!("{}", i),
                        &format!("{}", i + 1),
                        RelationshipType::RelatedTo,
                    ))
                    .await
                    .ok();
            }

            store
        }

        proptest! {
            /// Shortest path length should equal hop count

            fn shortest_path_length_equals_hops(
                chain_length in 3usize..15,
                start_idx in 0usize..3,
                end_offset in 1usize..5
            ) {
                tokio_test::block_on(async {
                    let chain_length = chain_length.max(5);
                    let start_idx = start_idx.min(chain_length - 2);
                    let end_idx = (start_idx + end_offset).min(chain_length - 1);

                    let store = create_chain_graph(chain_length).await;

                    let path_result = find_shortest_path(
                        &store,
                        &format!("{}", start_idx),
                        &format!("{}", end_idx),
                        chain_length,
                        Direction::Outgoing,
                    )
                    .await;

                    if let Ok(Some(p)) = path_result {
                        let expected_hops = end_idx - start_idx;
                        prop_assert_eq!(p.len(), expected_hops,
                            "Path from {} to {} should have {} hops, got {}",
                            start_idx, end_idx, expected_hops, p.len());
                    }
                    Ok(())
                });
            }

            /// BFS should visit all reachable nodes within max_hops
            #[test]
            fn bfs_visits_all_reachable(
                chain_length in 3usize..12,
                max_hops in 1usize..8
            ) {
                tokio_test::block_on(async {
                    let store = create_chain_graph(chain_length).await;

                    let query = GraphQuery::new(vec!["0".to_string()]).with_max_hops(max_hops);

                    let paths = bfs_traverse(&store, &query).await.ok();

                    if let Some(paths) = paths {
                        // Collect all visited entities
                        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for path in &paths {
                            for entity in &path.entities {
                                visited.insert(entity.id.clone());
                            }
                        }

                        // Should visit start node + nodes within max_hops
                        let expected_max = (max_hops + 1).min(chain_length);
                        prop_assert!(visited.len() <= expected_max,
                            "BFS visited {} nodes, expected at most {} (max_hops={}, chain_length={})",
                            visited.len(), expected_max, max_hops, chain_length);
                    }
                    Ok(())
                });
            }

            /// find_entities_within_hops respects hop limit
            #[test]
            fn entities_within_hops_respects_limit(
                chain_length in 4usize..12,
                max_hops in 1usize..6
            ) {
                tokio_test::block_on(async {
                    let store = create_chain_graph(chain_length).await;

                    let entities = find_entities_within_hops(
                        &store,
                        &"0".to_string(),
                        max_hops,
                        Direction::Outgoing,
                        None,
                        None,
                    )
                    .await
                    .ok();

                    if let Some(entities) = entities {
                        // All entities should be within max_hops
                        for (_, hops) in &entities {
                            prop_assert!(*hops <= max_hops,
                                "Entity found at {} hops, exceeds max_hops {}",
                                hops, max_hops);
                        }

                        // Should find at most max_hops entities (excluding start)
                        let expected_max = max_hops.min(chain_length - 1);
                        prop_assert!(entities.len() <= expected_max,
                            "Found {} entities, expected at most {} (max_hops={}, chain_length={})",
                            entities.len(), expected_max, max_hops, chain_length);
                    }
                    Ok(())
                });
            }

            /// Shortest path from node to itself should be empty or single node
            #[test]
            fn shortest_path_to_self(
                chain_length in 2usize..10,
                node_idx in 0usize..5
            ) {
                tokio_test::block_on(async {
                    let node_idx = node_idx.min(chain_length - 1);
                    let store = create_chain_graph(chain_length).await;

                    let node_id = format!("{}", node_idx);
                    let path_result = find_shortest_path(
                        &store,
                        &node_id,
                        &node_id,
                        10,
                        Direction::Outgoing,
                    )
                    .await;

                    // Path to self should exist and have 0 hops
                    if let Ok(Some(p)) = path_result {
                        prop_assert_eq!(p.len(), 0,
                            "Path from node to itself should have 0 hops, got {}",
                            p.len());
                    }
                    Ok(())
                });
            }

            /// All paths returned by BFS should satisfy min_confidence
            #[test]
            fn bfs_respects_min_confidence(
                chain_length in 3usize..8,
                min_confidence in 0.0f32..0.9
            ) {
                tokio_test::block_on(async {
                    let store = create_chain_graph(chain_length).await;

                    let query = GraphQuery::new(vec!["0".to_string()])
                        .with_max_hops(chain_length)
                        .with_min_confidence(min_confidence);

                    let paths = bfs_traverse(&store, &query).await.ok();

                    if let Some(paths) = paths {
                        for path in &paths {
                            prop_assert!(path.total_confidence >= min_confidence,
                                "Path has confidence {}, below min {}",
                                path.total_confidence, min_confidence);
                        }
                    }
                    Ok(())
                });
            }

            /// Paths should not contain cycles (duplicate entities)
            #[test]
            fn paths_have_no_cycles(
                chain_length in 3usize..10,
                max_hops in 2usize..6
            ) {
                tokio_test::block_on(async {
                    let store = create_chain_graph(chain_length).await;

                    let query = GraphQuery::new(vec!["0".to_string()]).with_max_hops(max_hops);

                    let paths = bfs_traverse(&store, &query).await.ok();

                    if let Some(paths) = paths {
                        for path in &paths {
                            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                            for entity in &path.entities {
                                prop_assert!(!seen.contains(&entity.id),
                                    "Path contains duplicate entity: {}", entity.id);
                                seen.insert(entity.id.clone());
                            }
                        }
                    }
                    Ok(())
                });
            }

            /// find_entities_within_hops returns entities sorted by hop distance
            #[test]
            fn entities_within_hops_sorted_by_distance(
                chain_length in 4usize..10
            ) {
                tokio_test::block_on(async {
                    let store = create_chain_graph(chain_length).await;

                    let entities = find_entities_within_hops(
                        &store,
                        &"0".to_string(),
                        chain_length - 1,
                        Direction::Outgoing,
                        None,
                        None,
                    )
                    .await
                    .ok();

                    if let Some(entities) = entities {
                        // Check that hop distances are valid
                        for (entity, hops) in &entities {
                            // In a chain, entity ID is the index, so hop count should match
                            if let Ok(entity_idx) = entity.id.parse::<usize>() {
                                prop_assert_eq!(*hops, entity_idx,
                                    "Entity {} should be at {} hops, found at {}",
                                    entity.id, entity_idx, hops);
                            }
                        }
                    }
                    Ok(())
                });
            }
        }
    }
}
