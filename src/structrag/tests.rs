#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::format_push_string
)]
//! Tests for the `structrag` module.

use crate::structrag::engine::StructRagEngine;
use crate::structrag::reason::StructRagReasoner;
use crate::structrag::restructure::StructRagRestructurer;
use crate::structrag::router::StructRagRouter;
use crate::structrag::types::{
    StructRagAlgorithm, StructRagCatalogue, StructRagCatalogueItem, StructRagConfig,
    StructRagError, StructRagGraph, StructRagGraphEdge, StructRagGraphNode,
    StructRagKnowledgeStructure, StructRagPassage, StructRagResult, StructRagRoutingDecision,
    StructRagStep, StructRagStructureKind, StructRagTable, StructRagTableRow, StructRagTree,
    StructRagTreeNode,
};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.01
}

// ── StructRagPassage ═════════════════════════════════════════════════════════

#[test]
fn passage_new_constructor() {
    let p = StructRagPassage::new("id1", "some text");
    assert_eq!(p.id, "id1");
    assert_eq!(p.text, "some text");
}

// ── StructRagStructureKind ═══════════════════════════════════════════════════

#[test]
fn structure_kind_as_str_and_display() {
    assert_eq!(StructRagStructureKind::Table.as_str(), "table");
    assert_eq!(StructRagStructureKind::Graph.as_str(), "graph");
    assert_eq!(StructRagStructureKind::Tree.as_str(), "tree");
    assert_eq!(StructRagStructureKind::Catalogue.as_str(), "catalogue");
    assert_eq!(StructRagStructureKind::Algorithm.as_str(), "algorithm");
    assert_eq!(format!("{}", StructRagStructureKind::Table), "table");
    assert_eq!(
        format!("{}", StructRagStructureKind::Algorithm),
        "algorithm"
    );
}

#[test]
fn structure_kind_all_returns_five_distinct_kinds() {
    let all = StructRagStructureKind::all();
    assert_eq!(all.len(), 5);
    let mut seen: Vec<&str> = all.iter().map(|k| k.as_str()).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 5, "all five kinds must be distinct");
}

// ── StructRagKnowledgeStructure ═══════════════════════════════════════════════

#[test]
fn knowledge_structure_table_kind_len_and_downcast() {
    let ks = StructRagKnowledgeStructure::Table(StructRagTable {
        columns: vec!["a".to_string()],
        rows: vec![StructRagTableRow {
            cells: vec!["1".to_string()],
            source_passage_id: "p1".to_string(),
        }],
    });
    assert_eq!(ks.kind(), StructRagStructureKind::Table);
    assert_eq!(ks.len(), 1);
    assert!(!ks.is_empty());
    assert!(ks.as_table().is_some());
    assert!(ks.as_graph().is_none());
    assert!(ks.as_tree().is_none());
    assert!(ks.as_catalogue().is_none());
    assert!(ks.as_algorithm().is_none());
}

#[test]
fn knowledge_structure_graph_kind_len_and_downcast() {
    let ks = StructRagKnowledgeStructure::Graph(StructRagGraph {
        nodes: vec![
            StructRagGraphNode {
                id: "a".to_string(),
                label: "A".to_string(),
                mentions: 1,
            },
            StructRagGraphNode {
                id: "b".to_string(),
                label: "B".to_string(),
                mentions: 1,
            },
        ],
        edges: vec![StructRagGraphEdge {
            source: "a".to_string(),
            target: "b".to_string(),
            relation: "related_to".to_string(),
            source_passage_id: "p1".to_string(),
        }],
    });
    assert_eq!(ks.kind(), StructRagStructureKind::Graph);
    assert_eq!(ks.len(), 2);
    assert!(ks.as_graph().is_some());
    assert!(ks.as_table().is_none());
}

#[test]
fn knowledge_structure_tree_kind_len_and_downcast() {
    let ks = StructRagKnowledgeStructure::Tree(StructRagTree {
        nodes: vec![StructRagTreeNode {
            id: 0,
            label: "Root".to_string(),
            parent: None,
            children: Vec::new(),
            depth: 0,
            source_passage_id: "p1".to_string(),
        }],
        root_ids: vec![0],
    });
    assert_eq!(ks.kind(), StructRagStructureKind::Tree);
    assert_eq!(ks.len(), 1);
    assert!(ks.as_tree().is_some());
    assert!(ks.as_catalogue().is_none());
}

#[test]
fn knowledge_structure_catalogue_kind_len_and_downcast() {
    let ks = StructRagKnowledgeStructure::Catalogue(StructRagCatalogue {
        items: vec![StructRagCatalogueItem {
            key: "Item".to_string(),
            attributes: vec![("color".to_string(), "red".to_string())],
            source_passage_id: "p1".to_string(),
        }],
    });
    assert_eq!(ks.kind(), StructRagStructureKind::Catalogue);
    assert_eq!(ks.len(), 1);
    assert!(ks.as_catalogue().is_some());
    assert!(ks.as_algorithm().is_none());
}

#[test]
fn knowledge_structure_algorithm_kind_len_and_downcast() {
    let ks = StructRagKnowledgeStructure::Algorithm(StructRagAlgorithm {
        steps: vec![StructRagStep {
            index: 1,
            action: "Do something".to_string(),
            source_passage_id: "p1".to_string(),
        }],
    });
    assert_eq!(ks.kind(), StructRagStructureKind::Algorithm);
    assert_eq!(ks.len(), 1);
    assert!(ks.as_algorithm().is_some());
    assert!(ks.as_tree().is_none());
}

#[test]
fn knowledge_structure_is_empty_for_empty_variant() {
    let ks = StructRagKnowledgeStructure::Table(StructRagTable::default());
    assert!(ks.is_empty());
    assert_eq!(ks.len(), 0);
}

// ── StructRagRoutingDecision ═════════════════════════════════════════════════

#[test]
fn routing_decision_score_of_present_and_absent() {
    let decision = StructRagRoutingDecision {
        kind: StructRagStructureKind::Table,
        confidence: 0.8,
        rationale: "test".to_string(),
        scores: vec![(StructRagStructureKind::Table, 0.8)],
    };
    assert!(approx(
        decision.score_of(StructRagStructureKind::Table),
        0.8
    ));
    assert!(approx(
        decision.score_of(StructRagStructureKind::Graph),
        0.0
    ));
}

// ── StructRagTable accessors ═════════════════════════════════════════════════

#[test]
fn table_column_index_is_case_insensitive() {
    let table = StructRagTable {
        columns: vec!["Name".to_string(), "Price".to_string()],
        rows: vec![],
    };
    assert_eq!(table.column_index("name"), Some(0));
    assert_eq!(table.column_index("PRICE"), Some(1));
    assert_eq!(table.column_index("missing"), None);
}

#[test]
fn table_cell_out_of_range_is_none() {
    let table = StructRagTable {
        columns: vec!["name".to_string()],
        rows: vec![StructRagTableRow {
            cells: vec!["Gizmo".to_string()],
            source_passage_id: "p1".to_string(),
        }],
    };
    assert_eq!(table.cell(0, "name"), Some("Gizmo"));
    assert_eq!(table.cell(1, "name"), None);
    assert_eq!(table.cell(0, "missing"), None);
}

#[test]
fn table_numeric_column_on_missing_column_is_empty() {
    let table = StructRagTable::default();
    assert!(table.numeric_column("value").is_empty());
}

// ── StructRagGraph accessors ═════════════════════════════════════════════════

#[test]
fn graph_find_node_neighbors_and_edges_from() {
    let graph = StructRagGraph {
        nodes: vec![
            StructRagGraphNode {
                id: "alpha".to_string(),
                label: "Alpha".to_string(),
                mentions: 2,
            },
            StructRagGraphNode {
                id: "beta".to_string(),
                label: "Beta".to_string(),
                mentions: 1,
            },
        ],
        edges: vec![StructRagGraphEdge {
            source: "alpha".to_string(),
            target: "beta".to_string(),
            relation: "meets".to_string(),
            source_passage_id: "p1".to_string(),
        }],
    };
    assert_eq!(graph.find_node("Alpha").map(|n| n.mentions), Some(2));
    assert_eq!(graph.find_node("nonexistent"), None);
    assert_eq!(graph.neighbors("alpha"), vec!["beta".to_string()]);
    assert_eq!(graph.neighbors("beta"), vec!["alpha".to_string()]);
    assert_eq!(graph.edges_from("alpha").len(), 1);
    assert_eq!(graph.edges_from("beta").len(), 0);
}

// ── StructRagTree accessors ══════════════════════════════════════════════════

#[test]
fn tree_ancestors_and_descendants_manual() {
    let tree = StructRagTree {
        nodes: vec![
            StructRagTreeNode {
                id: 0,
                label: "Root".to_string(),
                parent: None,
                children: vec![1],
                depth: 0,
                source_passage_id: "p1".to_string(),
            },
            StructRagTreeNode {
                id: 1,
                label: "Mid".to_string(),
                parent: Some(0),
                children: vec![2],
                depth: 1,
                source_passage_id: "p1".to_string(),
            },
            StructRagTreeNode {
                id: 2,
                label: "Leaf".to_string(),
                parent: Some(1),
                children: vec![],
                depth: 2,
                source_passage_id: "p1".to_string(),
            },
        ],
        root_ids: vec![0],
    };
    assert_eq!(tree.ancestors(2), vec![0, 1]);
    assert_eq!(tree.ancestors(0), Vec::<usize>::new());
    assert_eq!(tree.descendants(0), vec![1, 2]);
    assert_eq!(tree.descendants(2), Vec::<usize>::new());
    assert_eq!(tree.find("leaf").map(|n| n.id), Some(2));
}

// ── StructRagCatalogue accessors ═════════════════════════════════════════════

#[test]
fn catalogue_find_is_case_insensitive() {
    let catalogue = StructRagCatalogue {
        items: vec![StructRagCatalogueItem {
            key: "Gizmo".to_string(),
            attributes: vec![],
            source_passage_id: "p1".to_string(),
        }],
    };
    assert!(catalogue.find("gizmo").is_some());
    assert!(catalogue.find("GIZMO").is_some());
    assert!(catalogue.find("widget").is_none());
}

// ── StructRagConfig ═══════════════════════════════════════════════════════════

#[test]
fn config_default_values() {
    let config = StructRagConfig::default();
    assert_eq!(config.enabled_kinds, StructRagStructureKind::all().to_vec());
    assert!(approx(config.min_confidence, 0.15));
    assert_eq!(config.default_kind, StructRagStructureKind::Catalogue);
    assert_eq!(config.max_structure_size, 256);
    assert_eq!(
        config.tie_break_order,
        StructRagStructureKind::all().to_vec()
    );
}

#[test]
fn config_new_equals_default() {
    assert_eq!(StructRagConfig::new(), StructRagConfig::default());
}

#[test]
fn config_builders_chain() {
    let config = StructRagConfig::new()
        .with_min_confidence(0.42)
        .with_default_kind(StructRagStructureKind::Tree)
        .with_max_structure_size(10)
        .with_tie_break_order(vec![StructRagStructureKind::Graph]);
    assert!(approx(config.min_confidence, 0.42));
    assert_eq!(config.default_kind, StructRagStructureKind::Tree);
    assert_eq!(config.max_structure_size, 10);
    assert_eq!(config.tie_break_order, vec![StructRagStructureKind::Graph]);
}

#[test]
fn config_with_enabled_kinds_replaces_set() {
    let config = StructRagConfig::new().with_enabled_kinds(vec![
        StructRagStructureKind::Table,
        StructRagStructureKind::Graph,
    ]);
    assert_eq!(
        config.enabled_kinds,
        vec![StructRagStructureKind::Table, StructRagStructureKind::Graph]
    );
}

#[test]
fn config_with_disabled_kind_removes_only_that_kind() {
    let config = StructRagConfig::new().with_disabled_kind(StructRagStructureKind::Table);
    assert!(!config.is_enabled(StructRagStructureKind::Table));
    assert!(config.is_enabled(StructRagStructureKind::Graph));
    assert!(config.is_enabled(StructRagStructureKind::Tree));
    assert!(config.is_enabled(StructRagStructureKind::Catalogue));
    assert!(config.is_enabled(StructRagStructureKind::Algorithm));
    assert_eq!(config.enabled_kinds.len(), 4);
}

// ── StructRagError ════════════════════════════════════════════════════════════

#[test]
fn error_messages_are_specific() {
    assert!(StructRagError::EmptyQuery.to_string().contains("empty"));
    assert!(
        StructRagError::EmptyPassages
            .to_string()
            .contains("no passages")
    );
    assert!(
        StructRagError::NoEnabledStructureKinds
            .to_string()
            .contains("no structure kinds are enabled")
    );
    let disabled = StructRagError::DisabledStructureKind(StructRagStructureKind::Table);
    let msg = disabled.to_string();
    assert!(msg.contains("table"));
    assert!(msg.contains("disabled"));
}

// ══════════════════════════════════════════════════════════════════════════
// Router
// ══════════════════════════════════════════════════════════════════════════

fn default_router() -> StructRagRouter {
    StructRagRouter::new()
}

#[test]
fn router_picks_table_for_comparison_query() {
    let router = default_router();
    let decision = router
        .route("Which product is cheapest?", &StructRagConfig::default())
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Table);
    assert!(approx(decision.confidence, 1.0));
}

#[test]
fn router_picks_table_for_statistics_query() {
    let router = default_router();
    let decision = router
        .route(
            "Show me the statistics comparison between the two datasets.",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Table);
}

#[test]
fn router_picks_graph_for_relationship_query() {
    let router = default_router();
    let decision = router
        .route(
            "What is the relationship between the two companies?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Graph);
    assert!(approx(decision.confidence, 1.0));
}

#[test]
fn router_picks_graph_for_connection_query() {
    let router = default_router();
    let decision = router
        .route(
            "How are these two events connected to each other?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Graph);
}

#[test]
fn router_picks_tree_for_hierarchy_query() {
    let router = default_router();
    let decision = router
        .route(
            "What is the hierarchy of the animal kingdom?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Tree);
    assert!(approx(decision.confidence, 1.0));
}

#[test]
fn router_picks_tree_for_taxonomy_query() {
    let router = default_router();
    let decision = router
        .route(
            "Describe the taxonomy and classification of species.",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Tree);
}

#[test]
fn router_picks_catalogue_for_enumeration_query() {
    let router = default_router();
    let decision = router
        .route(
            "Please list all the ingredients in this recipe.",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Catalogue);
    assert!(approx(decision.confidence, 1.0));
}

#[test]
fn router_picks_catalogue_for_list_query() {
    let router = default_router();
    let decision = router
        .route(
            "What are the different types of tea available?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Catalogue);
}

#[test]
fn router_picks_algorithm_for_procedure_query() {
    let router = default_router();
    let decision = router
        .route(
            "What is the procedure for installing this software?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Algorithm);
}

#[test]
fn router_picks_algorithm_for_stepbystep_query() {
    let router = default_router();
    let decision = router
        .route(
            "Can you give me step by step instructions for this task?",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Algorithm);
    assert!(approx(decision.confidence, 1.0));
}

#[test]
fn router_confidence_and_rationale_present() {
    let router = default_router();
    let decision = router
        .route(
            "Compare the relationship between prices.",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    // Table = "compare" (1.5); Graph = "relationship between" (2.0) +
    // "relationship" (1.0) = 3.0; Graph should win.
    assert_eq!(decision.kind, StructRagStructureKind::Graph);
    assert!(decision.confidence > 0.0 && decision.confidence < 1.0);
    assert!(approx(
        decision.score_of(StructRagStructureKind::Graph),
        0.667
    ));
    assert!(approx(
        decision.score_of(StructRagStructureKind::Table),
        0.333
    ));
    assert!(!decision.rationale.is_empty());
    assert!(decision.rationale.contains("relationship"));
    assert!(decision.rationale.contains("confidence="));
}

#[test]
fn router_scores_cover_all_five_kinds() {
    let router = default_router();
    let decision = router
        .route("Which product is cheapest?", &StructRagConfig::default())
        .expect("route should succeed");
    assert_eq!(decision.scores.len(), 5);
    for kind in StructRagStructureKind::all() {
        assert!(decision.scores.iter().any(|(k, _)| *k == kind));
    }
}

#[test]
fn router_rejects_empty_query() {
    let router = default_router();
    let err = router
        .route("   ", &StructRagConfig::default())
        .unwrap_err();
    assert!(matches!(err, StructRagError::EmptyQuery));
    let err2 = router.route("", &StructRagConfig::default()).unwrap_err();
    assert!(matches!(err2, StructRagError::EmptyQuery));
}

#[test]
fn router_is_deterministic_same_query_same_result() {
    let router = default_router();
    let config = StructRagConfig::default();
    let d1 = router.route("Which product is cheapest?", &config).unwrap();
    let d2 = router.route("Which product is cheapest?", &config).unwrap();
    assert_eq!(d1, d2);
}

#[test]
fn router_tie_break_order_used_on_equal_scores() {
    let router = default_router();
    // Table = "compare" (1.5); Graph = "related to" (1.5): an exact tie.
    let decision = router
        .route(
            "Compare how X is related to Y.",
            &StructRagConfig::default(),
        )
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Table);
    assert!(approx(decision.confidence, 0.5));
}

#[test]
fn router_custom_tie_break_order_changes_winner_on_tie() {
    let router = default_router();
    let config = StructRagConfig::new().with_tie_break_order(vec![
        StructRagStructureKind::Graph,
        StructRagStructureKind::Table,
        StructRagStructureKind::Tree,
        StructRagStructureKind::Catalogue,
        StructRagStructureKind::Algorithm,
    ]);
    let decision = router
        .route("Compare how X is related to Y.", &config)
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Graph);
    assert!(approx(decision.confidence, 0.5));
}

#[test]
fn router_skips_disabled_top_kind_and_picks_next_enabled() {
    let router = default_router();
    let config = StructRagConfig::new().with_disabled_kind(StructRagStructureKind::Table);
    let decision = router
        .route("Compare how X is related to Y.", &config)
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Graph);
    assert!(approx(decision.confidence, 0.5));
    assert!(!decision.rationale.contains("falling back"));
}

#[test]
fn router_default_kind_used_when_no_cues_match() {
    let router = default_router();
    let decision = router
        .route("zzzqx wvbnm plugh trqz", &StructRagConfig::default())
        .expect("route should succeed");
    assert_eq!(decision.kind, StructRagStructureKind::Catalogue);
    assert!(decision.rationale.contains("below threshold"));
    assert!(decision.rationale.contains("falling back"));
}

#[test]
fn router_default_kind_used_below_custom_min_confidence() {
    let router = default_router();
    let config = StructRagConfig::new().with_min_confidence(0.7);
    let decision = router
        .route("Compare the relationship between prices.", &config)
        .expect("route should succeed");
    // Graph would normally win at confidence ~0.667, which is now below the
    // raised 0.7 threshold, so the configured default kind takes over.
    assert_eq!(decision.kind, StructRagStructureKind::Catalogue);
    assert!(decision.rationale.contains("below threshold"));
}

#[test]
fn router_errors_when_all_kinds_disabled() {
    let router = default_router();
    let config = StructRagConfig::new().with_enabled_kinds(vec![]);
    let err = router
        .route("Which product is cheapest?", &config)
        .unwrap_err();
    assert!(matches!(err, StructRagError::NoEnabledStructureKinds));
}

// ══════════════════════════════════════════════════════════════════════════
// Restructurer
// ══════════════════════════════════════════════════════════════════════════

fn default_restructurer() -> StructRagRestructurer {
    StructRagRestructurer::new()
}

// ── Table ─────────────────────────────────────────────────────────────────────

#[test]
fn restructure_table_builds_columns_and_rows_from_key_value_passages() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "name: Gizmo, price: 19.99, stock: 120"),
        StructRagPassage::new("p2", "name: Widget, price: 24.50, stock: 45"),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Table,
        &passages,
        &StructRagConfig::default(),
    );
    let table = structure.as_table().expect("table variant");
    assert_eq!(table.columns, vec!["name", "price", "stock"]);
    assert_eq!(table.rows.len(), 2);
    assert_eq!(table.cell(0, "name"), Some("Gizmo"));
    assert_eq!(table.cell(1, "name"), Some("Widget"));
    assert_eq!(
        table.numeric_column("price"),
        vec![Some(19.99), Some(24.50)]
    );
}

#[test]
fn restructure_table_fallback_extracts_subject_value_detail_when_no_explicit_pairs() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Zenith costs $19.99 and has a rating of 4.5 stars."),
        StructRagPassage::new("p2", "Nimbus costs $9.99 and has a rating of 4.0 stars."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Table,
        &passages,
        &StructRagConfig::default(),
    );
    let table = structure.as_table().expect("table variant");
    assert_eq!(table.columns, vec!["subject", "value", "detail"]);
    assert_eq!(table.cell(0, "subject"), Some("Zenith"));
    assert_eq!(table.cell(0, "value"), Some("$19.99"));
    assert_eq!(table.cell(1, "subject"), Some("Nimbus"));
    assert_eq!(table.cell(1, "value"), Some("$9.99"));
}

#[test]
fn restructure_table_numeric_column_parses_values_and_skips_non_numeric() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Zenith costs $19.99 and has a rating of 4.5 stars."),
        StructRagPassage::new("p2", "Nimbus costs $9.99 and has a rating of 4.0 stars."),
        StructRagPassage::new("p3", "Vertex is currently out of stock."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Table,
        &passages,
        &StructRagConfig::default(),
    );
    let table = structure.as_table().expect("table variant");
    assert_eq!(
        table.numeric_column("value"),
        vec![Some(19.99), Some(9.99), None]
    );
}

// ── Graph ─────────────────────────────────────────────────────────────────────

#[test]
fn restructure_graph_builds_nodes_and_edges_from_entity_sentences() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Marie Curie discovered Polonium.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Graph,
        &passages,
        &StructRagConfig::default(),
    );
    let graph = structure.as_graph().expect("graph variant");
    assert_eq!(graph.nodes.len(), 2);
    assert!(graph.find_node("Marie Curie").is_some());
    assert!(graph.find_node("Polonium").is_some());
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.edges[0].source, "marie curie");
    assert_eq!(graph.edges[0].target, "polonium");
    assert_eq!(graph.edges[0].relation, "discovered");
    assert_eq!(graph.edges[0].source_passage_id, "p1");
}

#[test]
fn restructure_graph_dedupes_nodes_and_counts_mentions() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Paris is the capital of France."),
        StructRagPassage::new("p2", "Many tourists visit Paris every year."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Graph,
        &passages,
        &StructRagConfig::default(),
    );
    let graph = structure.as_graph().expect("graph variant");
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.find_node("Paris").expect("Paris node").mentions, 2);
    assert_eq!(graph.find_node("France").expect("France node").mentions, 1);
    assert_eq!(graph.edges.len(), 1);
}

#[test]
fn restructure_graph_isolated_entity_has_no_edges() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Photosynthesis occurs in plants.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Graph,
        &passages,
        &StructRagConfig::default(),
    );
    let graph = structure.as_graph().expect("graph variant");
    assert_eq!(graph.nodes.len(), 1);
    assert!(graph.edges.is_empty());
    assert!(graph.edges_from("photosynthesis").is_empty());
}

// ── Tree ──────────────────────────────────────────────────────────────────────

#[test]
fn restructure_tree_builds_parent_child_from_is_a_phrase_and_strips_article() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new("p1", "A Sparrow is a type of Bird.")];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let tree = structure.as_tree().expect("tree variant");
    let bird = tree.find("Bird").expect("Bird node");
    let sparrow = tree.find("Sparrow").expect("Sparrow node");
    assert_eq!(sparrow.parent, Some(bird.id));
    assert!(bird.children.contains(&sparrow.id));
    assert_eq!(tree.root_ids, vec![bird.id]);
    assert_eq!(bird.depth, 0);
    assert_eq!(sparrow.depth, 1);
}

#[test]
fn restructure_tree_builds_parent_children_from_consists_of_phrase() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Aves consists of Sparrow, Eagle, and Penguin.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let tree = structure.as_tree().expect("tree variant");
    assert_eq!(tree.nodes.len(), 4);
    let aves = tree.find("Aves").expect("Aves node");
    assert_eq!(tree.root_ids, vec![aves.id]);
    for child_label in ["Sparrow", "Eagle", "Penguin"] {
        let child = tree
            .find(child_label)
            .unwrap_or_else(|| panic!("{child_label} node"));
        assert_eq!(child.parent, Some(aves.id));
        assert!(aves.children.contains(&child.id));
    }
}

#[test]
fn restructure_tree_indentation_fallback_builds_hierarchy() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Animals"),
        StructRagPassage::new("p2", "- Mammals"),
        StructRagPassage::new("p3", "- Birds"),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let tree = structure.as_tree().expect("tree variant");
    let animals = tree.find("Animals").expect("Animals node");
    let mammals = tree.find("Mammals").expect("Mammals node");
    let birds = tree.find("Birds").expect("Birds node");
    assert_eq!(tree.root_ids, vec![animals.id]);
    assert_eq!(mammals.parent, Some(animals.id));
    assert_eq!(birds.parent, Some(animals.id));
    assert_eq!(mammals.depth, 1);
    assert_eq!(birds.depth, 1);
}

#[test]
fn restructure_tree_avoids_cycle_when_relations_contradict() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Bird is a type of Animal."),
        // Contradicts the first relation; must not create a cycle.
        StructRagPassage::new("p2", "Animal is a type of Bird."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let tree = structure.as_tree().expect("tree variant");
    assert_eq!(tree.nodes.len(), 2);
    let animal = tree.find("Animal").expect("Animal node");
    let bird = tree.find("Bird").expect("Bird node");
    assert_eq!(bird.parent, Some(animal.id));
    assert_eq!(
        animal.parent, None,
        "the contradictory relation must be rejected"
    );
    assert_eq!(tree.ancestors(bird.id), vec![animal.id]);
    assert!(
        tree.ancestors(animal.id).is_empty(),
        "no cycle: Animal must not be its own ancestor"
    );
}

// ── Catalogue ─────────────────────────────────────────────────────────────────

#[test]
fn restructure_catalogue_builds_heterogeneous_keyed_items() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "name: Gizmo, price: 19.99, stock: 120"),
        StructRagPassage::new("p2", "name: Widget, color: red"),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Catalogue,
        &passages,
        &StructRagConfig::default(),
    );
    let catalogue = structure.as_catalogue().expect("catalogue variant");
    assert_eq!(catalogue.items.len(), 2);
    let gizmo = catalogue.find("Gizmo").expect("Gizmo item");
    assert!(
        gizmo
            .attributes
            .contains(&("price".to_string(), "19.99".to_string()))
    );
    let widget = catalogue.find("Widget").expect("Widget item");
    assert!(
        widget
            .attributes
            .contains(&("color".to_string(), "red".to_string()))
    );
    // Heterogeneous: Gizmo has 3 attributes, Widget has 2 — unlike a table,
    // items are not forced onto a shared column set.
    assert_eq!(gizmo.attributes.len(), 3);
    assert_eq!(widget.attributes.len(), 2);
}

#[test]
fn restructure_catalogue_fallback_item_from_plain_passage() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Ingredients: flour, sugar, and eggs.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Catalogue,
        &passages,
        &StructRagConfig::default(),
    );
    let catalogue = structure.as_catalogue().expect("catalogue variant");
    assert_eq!(catalogue.items.len(), 1);
    assert_eq!(catalogue.items[0].key, "Ingredients");
    assert_eq!(
        catalogue.items[0].attributes,
        vec![("items".to_string(), "flour, sugar, and eggs".to_string())]
    );
}

// ── Algorithm ─────────────────────────────────────────────────────────────────

#[test]
fn restructure_algorithm_orders_steps_by_explicit_numbering() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Step 2: Preheat the oven."),
        StructRagPassage::new("p2", "Step 1: Gather the ingredients."),
        StructRagPassage::new("p3", "Step 3: Bake for 20 minutes."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Algorithm,
        &passages,
        &StructRagConfig::default(),
    );
    let algorithm = structure.as_algorithm().expect("algorithm variant");
    assert_eq!(algorithm.steps.len(), 3);
    assert_eq!(algorithm.steps[0].index, 1);
    assert_eq!(algorithm.steps[0].action, "Gather the ingredients.");
    assert_eq!(algorithm.steps[1].index, 2);
    assert_eq!(algorithm.steps[1].action, "Preheat the oven.");
    assert_eq!(algorithm.steps[2].index, 3);
    assert_eq!(algorithm.steps[2].action, "Bake for 20 minutes.");
}

#[test]
fn restructure_algorithm_preserves_encounter_order_without_explicit_numbers() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "First, gather the ingredients."),
        StructRagPassage::new("p2", "Then, preheat the oven."),
        StructRagPassage::new("p3", "Finally, bake for 20 minutes."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Algorithm,
        &passages,
        &StructRagConfig::default(),
    );
    let algorithm = structure.as_algorithm().expect("algorithm variant");
    assert_eq!(algorithm.steps.len(), 3);
    assert_eq!(algorithm.steps[0].action, "gather the ingredients.");
    assert_eq!(algorithm.steps[1].action, "preheat the oven.");
    assert_eq!(algorithm.steps[2].action, "bake for 20 minutes.");
}

// ── max_structure_size truncation ────────────────────────────────────────────

#[test]
fn restructure_catalogue_respects_max_structure_size() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "name: Alpha"),
        StructRagPassage::new("p2", "name: Beta"),
        StructRagPassage::new("p3", "name: Gamma"),
        StructRagPassage::new("p4", "name: Delta"),
        StructRagPassage::new("p5", "name: Epsilon"),
    ];
    let config = StructRagConfig::new().with_max_structure_size(2);
    let structure = restructurer.restructure(StructRagStructureKind::Catalogue, &passages, &config);
    let catalogue = structure.as_catalogue().expect("catalogue variant");
    assert_eq!(catalogue.items.len(), 2);
    assert_eq!(catalogue.items[0].key, "Alpha");
    assert_eq!(catalogue.items[1].key, "Beta");
}

#[test]
fn restructure_graph_respects_max_structure_size_and_drops_dangling_edges() {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Alpha meets Beta."),
        StructRagPassage::new("p2", "Beta meets Gamma."),
        StructRagPassage::new("p3", "Gamma meets Delta."),
    ];
    let config = StructRagConfig::new().with_max_structure_size(2);
    let structure = restructurer.restructure(StructRagStructureKind::Graph, &passages, &config);
    let graph = structure.as_graph().expect("graph variant");
    assert_eq!(graph.nodes.len(), 2);
    assert!(graph.find_node("Alpha").is_some());
    assert!(graph.find_node("Beta").is_some());
    assert!(graph.find_node("Gamma").is_none());
    assert!(graph.find_node("Delta").is_none());
    // The Beta->Gamma edge must be dropped since Gamma no longer exists.
    assert_eq!(graph.edges.len(), 1);
}

#[test]
fn restructure_tree_respects_max_structure_size_and_repairs_children() {
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Animals consists of Mammals, Birds, Fish, and Reptiles.",
    )];
    let config = StructRagConfig::new().with_max_structure_size(3);
    let structure = restructurer.restructure(StructRagStructureKind::Tree, &passages, &config);
    let tree = structure.as_tree().expect("tree variant");
    assert_eq!(tree.nodes.len(), 3);
    let animals = tree.find("Animals").expect("Animals node");
    assert_eq!(animals.children.len(), 2);
    assert!(tree.find("Fish").is_none());
    assert!(tree.find("Reptiles").is_none());
    assert_eq!(tree.root_ids, vec![animals.id]);
}

// ── restructurer robustness ──────────────────────────────────────────────────

#[test]
fn restructure_handles_empty_passages_gracefully() {
    let restructurer = default_restructurer();
    let config = StructRagConfig::default();
    for kind in StructRagStructureKind::all() {
        let structure = restructurer.restructure(kind, &[], &config);
        assert_eq!(structure.kind(), kind);
        assert!(structure.is_empty());
    }
}

// ══════════════════════════════════════════════════════════════════════════
// Reasoner
// ══════════════════════════════════════════════════════════════════════════

fn default_reasoner() -> StructRagReasoner {
    StructRagReasoner::new()
}

fn zenith_nimbus_table() -> StructRagKnowledgeStructure {
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Zenith costs $19.99 and has a rating of 4.5 stars."),
        StructRagPassage::new("p2", "Nimbus costs $9.99 and has a rating of 4.0 stars."),
    ];
    restructurer.restructure(
        StructRagStructureKind::Table,
        &passages,
        &StructRagConfig::default(),
    )
}

#[test]
fn reason_table_answers_min_query() {
    let reasoner = default_reasoner();
    let structure = zenith_nimbus_table();
    let answer = reasoner.reason("Which product is cheapest?", &structure);
    assert!(answer.contains("Nimbus"));
    assert!(answer.contains("lowest"));
}

#[test]
fn reason_table_answers_max_query() {
    let reasoner = default_reasoner();
    let structure = zenith_nimbus_table();
    let answer = reasoner.reason("Which product has the highest price?", &structure);
    assert!(answer.contains("Zenith"));
    assert!(answer.contains("highest"));
}

#[test]
fn reason_table_falls_back_to_full_table_summary() {
    let reasoner = default_reasoner();
    let structure = zenith_nimbus_table();
    let answer = reasoner.reason("Tell me something unrelated to any row.", &structure);
    assert!(answer.contains("Restructured into a table"));
    assert!(answer.contains("Zenith"));
    assert!(answer.contains("Nimbus"));
}

#[test]
fn reason_table_empty_rows_reports_no_rows() {
    let reasoner = default_reasoner();
    let structure = StructRagKnowledgeStructure::Table(StructRagTable::default());
    let answer = reasoner.reason("anything", &structure);
    assert!(answer.contains("no rows"));
}

#[test]
fn reason_graph_answers_relationship_between_two_entities() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Marie Curie discovered Polonium.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Graph,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("How is Marie Curie related to Polonium?", &structure);
    assert!(answer.contains("Marie Curie"));
    assert!(answer.contains("Polonium"));
    assert!(answer.contains("discovered"));
}

#[test]
fn reason_graph_summarizes_when_no_entity_matches() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Marie Curie discovered Polonium.",
    )];
    let structure = restructurer.restructure(
        StructRagStructureKind::Graph,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("What is the boiling point of water?", &structure);
    assert!(answer.contains("Restructured into a graph"));
    assert!(answer.contains("Marie Curie"));
}

#[test]
fn reason_tree_answers_hierarchy_lookup_for_matched_node() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new("p1", "A Sparrow is a type of Bird.")];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("Tell me about Sparrow.", &structure);
    assert!(answer.contains("Sparrow"));
    assert!(answer.contains("Bird"));
    assert!(answer.contains("leaf node"));
}

#[test]
fn reason_tree_summarizes_when_no_node_matches() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![StructRagPassage::new("p1", "A Sparrow is a type of Bird.")];
    let structure = restructurer.restructure(
        StructRagStructureKind::Tree,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("Completely unrelated question here.", &structure);
    assert!(answer.contains("Restructured into a tree"));
    assert!(answer.contains("Bird"));
}

#[test]
fn reason_catalogue_answers_lookup_for_matched_item() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "name: Gizmo, price: 19.99, stock: 120"),
        StructRagPassage::new("p2", "name: Widget, color: red"),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Catalogue,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("Tell me about Gizmo.", &structure);
    assert!(answer.contains("Gizmo"));
    assert!(answer.contains("price=19.99"));
}

#[test]
fn reason_catalogue_lists_all_items_when_none_matches() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "name: Gizmo, price: 19.99, stock: 120"),
        StructRagPassage::new("p2", "name: Widget, color: red"),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Catalogue,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("List all the items available.", &structure);
    assert!(answer.contains("Gizmo"));
    assert!(answer.contains("Widget"));
    assert!(answer.contains("2 items"));
}

#[test]
fn reason_algorithm_walks_ordered_steps() {
    let reasoner = default_reasoner();
    let restructurer = default_restructurer();
    let passages = vec![
        StructRagPassage::new("p1", "Step 1: Gather the ingredients."),
        StructRagPassage::new("p2", "Step 2: Preheat the oven."),
        StructRagPassage::new("p3", "Step 3: Bake for 20 minutes."),
    ];
    let structure = restructurer.restructure(
        StructRagStructureKind::Algorithm,
        &passages,
        &StructRagConfig::default(),
    );
    let answer = reasoner.reason("How do I bake this?", &structure);
    let step1 = answer.find("Step 1:").expect("Step 1 present");
    let step2 = answer.find("Step 2:").expect("Step 2 present");
    let step3 = answer.find("Step 3:").expect("Step 3 present");
    assert!(step1 < step2 && step2 < step3, "steps must appear in order");
    assert!(answer.contains("Bake for 20 minutes."));
}

// ══════════════════════════════════════════════════════════════════════════
// Engine
// ══════════════════════════════════════════════════════════════════════════

fn default_engine() -> StructRagEngine {
    StructRagEngine::new(StructRagConfig::default())
}

#[test]
fn engine_run_end_to_end_table() {
    let engine = default_engine();
    let passages = vec![
        StructRagPassage::new("p1", "Zenith costs $19.99 and has a rating of 4.5 stars."),
        StructRagPassage::new("p2", "Nimbus costs $9.99 and has a rating of 4.0 stars."),
    ];
    let result = engine
        .run("Which product is cheapest?", &passages)
        .expect("run should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Table);
    assert_eq!(result.structure.kind(), StructRagStructureKind::Table);
    assert!(result.answer.contains("Nimbus"));
}

#[test]
fn engine_run_end_to_end_graph() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Marie Curie discovered Polonium.",
    )];
    let result = engine
        .run("How is Marie Curie related to Polonium?", &passages)
        .expect("run should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Graph);
    assert!(result.answer.contains("discovered"));
}

#[test]
fn engine_run_end_to_end_tree() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new("p1", "A Sparrow is a type of Bird.")];
    let result = engine
        .run("What is the hierarchy of Sparrow?", &passages)
        .expect("run should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Tree);
    assert!(result.answer.contains("Sparrow"));
    assert!(result.answer.contains("Bird"));
}

#[test]
fn engine_run_end_to_end_catalogue() {
    let engine = default_engine();
    let passages = vec![
        StructRagPassage::new("p1", "name: Gizmo, price: 19.99, stock: 120"),
        StructRagPassage::new("p2", "name: Widget, color: red"),
    ];
    let result = engine
        .run("Please list all the items available.", &passages)
        .expect("run should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Catalogue);
    assert!(result.answer.contains("Gizmo"));
    assert!(result.answer.contains("Widget"));
}

#[test]
fn engine_run_end_to_end_algorithm() {
    let engine = default_engine();
    let passages = vec![
        StructRagPassage::new("p1", "Step 1: Gather the ingredients."),
        StructRagPassage::new("p2", "Step 2: Preheat the oven."),
        StructRagPassage::new("p3", "Step 3: Bake for 20 minutes."),
    ];
    let result = engine
        .run(
            "What is the step by step procedure to bake a cake?",
            &passages,
        )
        .expect("run should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Algorithm);
    assert!(result.answer.contains("Step 1"));
    assert!(result.answer.contains("Step 3"));
    assert!(result.answer.contains("Bake for 20 minutes."));
}

#[test]
fn engine_run_rejects_empty_query() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new("p1", "some text")];
    let err = engine.run("   ", &passages).unwrap_err();
    assert!(matches!(err, StructRagError::EmptyQuery));
}

#[test]
fn engine_run_rejects_empty_passages() {
    let engine = default_engine();
    let err = engine.run("a valid query", &[]).unwrap_err();
    assert!(matches!(err, StructRagError::EmptyPassages));
}

#[test]
fn engine_run_errors_when_all_kinds_disabled() {
    let engine = StructRagEngine::new(StructRagConfig::new().with_enabled_kinds(vec![]));
    let passages = vec![StructRagPassage::new("p1", "some text")];
    let err = engine.run("a valid query", &passages).unwrap_err();
    assert!(matches!(err, StructRagError::NoEnabledStructureKinds));
}

#[test]
fn engine_run_with_kind_forces_structure_bypassing_router() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new("p1", "name: Gizmo, price: 19.99")];
    // A nonsense query would normally fall back to the default kind
    // (Catalogue); forcing Table must bypass that entirely.
    let result = engine
        .run_with_kind("zzzqx wvbnm", &passages, StructRagStructureKind::Table)
        .expect("run_with_kind should succeed");
    assert_eq!(result.routing.kind, StructRagStructureKind::Table);
    assert!(approx(result.routing.confidence, 1.0));
    assert!(result.routing.rationale.contains("forced"));
    let table = result.structure.as_table().expect("table variant");
    assert!(table.columns.contains(&"name".to_string()));
    assert!(table.columns.contains(&"price".to_string()));
}

#[test]
fn engine_run_with_kind_errors_on_disabled_kind() {
    let engine = StructRagEngine::new(
        StructRagConfig::new().with_disabled_kind(StructRagStructureKind::Table),
    );
    let passages = vec![StructRagPassage::new("p1", "name: Gizmo")];
    let err = engine
        .run_with_kind("a query", &passages, StructRagStructureKind::Table)
        .unwrap_err();
    match err {
        StructRagError::DisabledStructureKind(kind) => {
            assert_eq!(kind, StructRagStructureKind::Table);
        }
        other => panic!("expected DisabledStructureKind, got {other:?}"),
    }
}

#[test]
fn engine_run_is_deterministic() {
    let engine = default_engine();
    let passages = vec![
        StructRagPassage::new("p1", "Zenith costs $19.99 and has a rating of 4.5 stars."),
        StructRagPassage::new("p2", "Nimbus costs $9.99 and has a rating of 4.0 stars."),
    ];
    let r1 = engine.run("Which product is cheapest?", &passages).unwrap();
    let r2 = engine.run("Which product is cheapest?", &passages).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn engine_result_carries_routing_and_structure_for_transparency() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Zenith costs $19.99 and has a rating of 4.5 stars.",
    )];
    let result: StructRagResult = engine
        .run("Which product is cheapest?", &passages)
        .expect("run should succeed");
    assert_eq!(result.structure.kind(), result.routing.kind);
    assert_eq!(result.query, "Which product is cheapest?");
}

#[test]
fn engine_route_standalone_matches_run_routing() {
    let engine = default_engine();
    let passages = vec![StructRagPassage::new(
        "p1",
        "Zenith costs $19.99 and has a rating of 4.5 stars.",
    )];
    let query = "Which product is cheapest?";
    let standalone = engine.route(query).expect("route should succeed");
    let full = engine.run(query, &passages).expect("run should succeed");
    assert_eq!(standalone, full.routing);
}

#[test]
fn engine_default_uses_default_config() {
    let engine = StructRagEngine::default();
    assert_eq!(engine.config, StructRagConfig::default());
}
