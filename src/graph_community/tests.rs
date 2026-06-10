//! Tests for the `graph_community` module.

use super::types::{CommunityId, GraphCommunityConfig, GraphCommunityError};

#[test]
fn test_community_id_new_and_display() {
    let id = CommunityId::new(3);
    assert_eq!(id.as_usize(), 3);
    assert_eq!(id.to_string(), "community:3");
}

#[test]
fn test_community_id_equality() {
    assert_eq!(CommunityId::new(1), CommunityId::new(1));
    assert_ne!(CommunityId::new(1), CommunityId::new(2));
}

#[test]
fn test_config_defaults() {
    let cfg = GraphCommunityConfig::default();
    assert!((cfg.resolution - 1.0).abs() < 1e-10);
    assert_eq!(cfg.max_levels, 5);
}

#[test]
fn test_config_builders() {
    let cfg = GraphCommunityConfig::default()
        .with_resolution(2.0)
        .with_max_levels(3);
    assert!((cfg.resolution - 2.0).abs() < 1e-10);
    assert_eq!(cfg.max_levels, 3);
}

#[test]
fn test_error_display() {
    let e = GraphCommunityError::EmptyGraph;
    assert!(!e.to_string().is_empty());
}

#[cfg(feature = "graphrag")]
mod graphrag_tests {
    use crate::layer4_graph::types::{
        EntityType, GraphEntity, GraphRelationship, RelationshipType,
    };

    use super::super::louvain::LouvainDetector;
    use super::super::types::{
        Community, CommunityDetector, GraphCommunityConfig, GraphCommunityError,
    };

    fn make_entity(id: &str, name: &str) -> GraphEntity {
        GraphEntity::new(name, EntityType::Concept).with_id(id)
    }

    fn make_rel(src: &str, tgt: &str) -> GraphRelationship {
        GraphRelationship::new(src, tgt, RelationshipType::RelatedTo)
            .with_id(format!("{src}-{tgt}"))
    }

    fn make_rel_conf(src: &str, tgt: &str, conf: f32) -> GraphRelationship {
        GraphRelationship::new(src, tgt, RelationshipType::RelatedTo)
            .with_id(format!("{src}-{tgt}"))
            .with_confidence(conf)
    }

    #[test]
    fn test_detect_empty_entities_error() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        let err = d.detect(&[], &[], &cfg).expect_err("should fail");
        assert!(matches!(err, GraphCommunityError::EmptyGraph));
    }

    #[test]
    fn test_detect_single_entity_singleton() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        let entities = vec![make_entity("e1", "Alpha")];
        let result = d.detect(&entities, &[], &cfg).expect("ok");
        assert_eq!(result.len(), 1);
        assert!(result.communities[0].is_singleton());
    }

    #[test]
    fn test_detect_no_edges_each_own_community() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        let entities = vec![
            make_entity("e1", "A"),
            make_entity("e2", "B"),
            make_entity("e3", "C"),
        ];
        let result = d.detect(&entities, &[], &cfg).expect("ok");
        // With no edges, every node is its own community
        assert_eq!(result.len(), 3);
        assert!(
            (result.modularity).abs() < 1e-9,
            "zero-edge modularity should be ~0"
        );
    }

    #[test]
    fn test_detect_two_cliques() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        // Two dense cliques: {A,B,C} and {D,E,F}, one weak bridge B-D
        let entities = vec![
            make_entity("a", "A"),
            make_entity("b", "B"),
            make_entity("c", "C"),
            make_entity("d", "D"),
            make_entity("e", "E"),
            make_entity("f", "F"),
        ];
        let rels = vec![
            make_rel_conf("a", "b", 1.0),
            make_rel_conf("b", "c", 1.0),
            make_rel_conf("a", "c", 1.0),
            make_rel_conf("d", "e", 1.0),
            make_rel_conf("e", "f", 1.0),
            make_rel_conf("d", "f", 1.0),
            make_rel_conf("c", "d", 0.01), // weak bridge
        ];
        let result = d.detect(&entities, &rels, &cfg).expect("ok");
        assert!(!result.is_empty(), "should detect at least one community");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_community_size() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        let entities = vec![make_entity("e1", "A"), make_entity("e2", "B")];
        let rels = vec![make_rel("e1", "e2")];
        let result = d.detect(&entities, &rels, &cfg).expect("ok");
        let total_members: usize = result.communities.iter().map(Community::size).sum();
        assert_eq!(total_members, 2, "all entities should be assigned");
    }

    #[test]
    fn test_community_graph_len_is_empty() {
        let d = LouvainDetector::new();
        let cfg = GraphCommunityConfig::default();
        let entities = vec![make_entity("e1", "A")];
        let result = d.detect(&entities, &[], &cfg).expect("ok");
        assert!(!result.is_empty());
        assert_eq!(result.len(), result.communities.len());
    }

    #[test]
    fn test_detect_resolution_high_splits_more() {
        let d = LouvainDetector::new();
        let cfg_low = GraphCommunityConfig::default().with_resolution(0.01);
        let cfg_high = GraphCommunityConfig::default().with_resolution(10.0);
        let entities: Vec<_> = (0..6_usize)
            .map(|i| make_entity(&i.to_string(), &format!("E{i}")))
            .collect();
        let rels = vec![
            make_rel_conf("0", "1", 1.0),
            make_rel_conf("2", "3", 1.0),
            make_rel_conf("4", "5", 1.0),
            make_rel_conf("1", "2", 0.1),
            make_rel_conf("3", "4", 0.1),
        ];
        let low = d.detect(&entities, &rels, &cfg_low).expect("ok");
        let high = d.detect(&entities, &rels, &cfg_high).expect("ok");
        assert!(
            high.len() >= low.len(),
            "higher resolution should produce >= communities: low={} high={}",
            low.len(),
            high.len()
        );
    }
}
