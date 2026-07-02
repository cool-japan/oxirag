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
    clippy::too_many_lines
)]

use crate::think_on_graph::engine::TogEngine;
use crate::think_on_graph::types::{
    TogBeamPath, TogConfig, TogDecision, TogEntity, TogError, TogExploration, TogKnowledgeGraph,
    TogRelation, TogResult, TogTriple,
};

// ── Graph fixtures ──────────────────────────────────────────────────────────

/// `France -[capital]-> Paris`.
fn single_hop_graph() -> TogKnowledgeGraph {
    TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("France", "France"),
            TogEntity::new("Paris", "Paris"),
        ],
        vec![TogTriple::new("France", "capital", "Paris")],
    )
    .expect("single-hop graph is well-formed")
}

/// `alpha -[linksto]-> beta -[reaches]-> gamma`.
fn linear_chain_graph() -> TogKnowledgeGraph {
    TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("alpha", "alpha"),
            TogEntity::new("beta", "beta"),
            TogEntity::new("gamma", "gamma"),
        ],
        vec![
            TogTriple::new("alpha", "linksto", "beta"),
            TogTriple::new("beta", "reaches", "gamma"),
        ],
    )
    .expect("linear chain graph is well-formed")
}

/// `alpha -[linksto]-> beta -[reaches]-> gamma -[attains]-> delta`.
fn three_hop_graph() -> TogKnowledgeGraph {
    TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("alpha", "alpha"),
            TogEntity::new("beta", "beta"),
            TogEntity::new("gamma", "gamma"),
            TogEntity::new("delta", "delta"),
        ],
        vec![
            TogTriple::new("alpha", "linksto", "beta"),
            TogTriple::new("beta", "reaches", "gamma"),
            TogTriple::new("gamma", "attains", "delta"),
        ],
    )
    .expect("three-hop graph is well-formed")
}

/// A star: `hub` connects out to four leaves via four distinct relations.
fn star_graph() -> TogKnowledgeGraph {
    TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("hub", "hub"),
            TogEntity::new("nodea", "nodea"),
            TogEntity::new("nodeb", "nodeb"),
            TogEntity::new("nodec", "nodec"),
            TogEntity::new("noded", "noded"),
        ],
        vec![
            TogTriple::new("hub", "rela", "nodea"),
            TogTriple::new("hub", "relb", "nodeb"),
            TogTriple::new("hub", "relc", "nodec"),
            TogTriple::new("hub", "reld", "noded"),
        ],
    )
    .expect("star graph is well-formed")
}

// ── TogConfig ───────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = TogConfig::default();
    assert_eq!(config.beam_width, 3);
    assert_eq!(config.max_depth, 3);
    assert_eq!(config.relations_per_expansion, 3);
    assert_eq!(config.entities_per_relation, 3);
    assert_eq!(config.embedding_dim, 64);
    assert_eq!(config.sufficiency_threshold, 0.6);
    assert_eq!(config.lexical_weight, 0.5);
    assert_eq!(config.min_relation_score, 0.0);
    assert_eq!(TogConfig::new(), TogConfig::default());
}

#[test]
fn config_builders_chain() {
    let config = TogConfig::new()
        .with_beam_width(7)
        .with_max_depth(9)
        .with_relations_per_expansion(4)
        .with_entities_per_relation(5)
        .with_embedding_dim(128)
        .with_sufficiency_threshold(0.8)
        .with_lexical_weight(0.25)
        .with_min_relation_score(0.1);
    assert_eq!(config.beam_width, 7);
    assert_eq!(config.max_depth, 9);
    assert_eq!(config.relations_per_expansion, 4);
    assert_eq!(config.entities_per_relation, 5);
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.sufficiency_threshold, 0.8);
    assert_eq!(config.lexical_weight, 0.25);
    assert_eq!(config.min_relation_score, 0.1);
}

#[test]
fn config_validate_accepts_default() {
    assert!(TogConfig::default().validate().is_ok());
}

#[test]
fn config_validate_rejects_zero_beam_width() {
    let err = TogConfig::default().with_beam_width(0).validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

#[test]
fn config_validate_rejects_zero_max_depth() {
    let err = TogConfig::default().with_max_depth(0).validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

#[test]
fn config_validate_rejects_zero_relations_per_expansion() {
    let err = TogConfig::default()
        .with_relations_per_expansion(0)
        .validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

#[test]
fn config_validate_rejects_zero_entities_per_relation() {
    let err = TogConfig::default()
        .with_entities_per_relation(0)
        .validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

#[test]
fn config_validate_rejects_zero_embedding_dim() {
    let err = TogConfig::default().with_embedding_dim(0).validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

#[test]
fn config_validate_rejects_out_of_range_lexical_weight() {
    assert!(matches!(
        TogConfig::default().with_lexical_weight(1.5).validate(),
        Err(TogError::InvalidConfig { .. })
    ));
    assert!(matches!(
        TogConfig::default().with_lexical_weight(-0.1).validate(),
        Err(TogError::InvalidConfig { .. })
    ));
}

#[test]
fn config_validate_rejects_non_finite_threshold() {
    let err = TogConfig::default()
        .with_sufficiency_threshold(f32::NAN)
        .validate();
    assert!(matches!(err, Err(TogError::InvalidConfig { .. })));
}

// ── TogEntity / TogRelation / TogTriple ─────────────────────────────────────

#[test]
fn entity_new_fields() {
    let entity = TogEntity::new("id1", "Name One");
    assert_eq!(entity.id, "id1");
    assert_eq!(entity.name, "Name One");
}

#[test]
fn relation_new_and_accessors() {
    let relation = TogRelation::new("capital_of");
    assert_eq!(relation.as_str(), "capital_of");
    assert_eq!(relation.label(), "capital_of");
    assert_eq!(relation, TogRelation::new("capital_of"));
}

#[test]
fn triple_new_fields() {
    let triple = TogTriple::new("h", "r", "t");
    assert_eq!(triple.head, "h");
    assert_eq!(triple.relation, "r");
    assert_eq!(triple.tail, "t");
}

// ── TogBeamPath ─────────────────────────────────────────────────────────────

#[test]
fn beam_path_seed_defaults() {
    let path = TogBeamPath::seed("topic", 0.5);
    assert_eq!(path.topic_entity_id, "topic");
    assert_eq!(path.current_entity_id(), "topic");
    assert_eq!(path.depth(), 0);
    assert!(path.is_seed());
    assert_eq!(path.entities(), &["topic".to_string()]);
    assert!(path.triples().is_empty());
    assert_eq!(path.score, 0.5);
}

#[test]
fn beam_path_extended_advances_frontier_and_score() {
    let seed = TogBeamPath::seed("a", 1.0);
    let extended = seed.extended(TogTriple::new("a", "r", "b"), "b", 0.25);
    assert_eq!(extended.depth(), 1);
    assert!(!extended.is_seed());
    assert_eq!(extended.current_entity_id(), "b");
    assert_eq!(extended.topic_entity_id, "a");
    assert_eq!(extended.entities(), &["a".to_string(), "b".to_string()]);
    assert_eq!(extended.score, 1.25);
    // Original path is unmodified.
    assert_eq!(seed.depth(), 0);
}

#[test]
fn beam_path_contains_entity_detects_cycle() {
    let path = TogBeamPath::seed("a", 0.0).extended(TogTriple::new("a", "r", "b"), "b", 0.0);
    assert!(path.contains_entity("a"));
    assert!(path.contains_entity("b"));
    assert!(!path.contains_entity("c"));
}

#[test]
fn beam_path_score_monotonic_along_hops() {
    let p0 = TogBeamPath::seed("a", 0.2);
    let p1 = p0.extended(TogTriple::new("a", "r", "b"), "b", 0.3);
    let p2 = p1.extended(TogTriple::new("b", "r", "c"), "c", 0.4);
    assert!(p1.score >= p0.score);
    assert!(p2.score >= p1.score);
    assert_eq!(p2.score, 0.9);
}

// ── TogDecision ─────────────────────────────────────────────────────────────

#[test]
fn decision_accessors() {
    let sufficient = TogDecision::Sufficient {
        depth: 2,
        coverage: 0.9,
    };
    assert!(sufficient.is_sufficient());
    assert!(sufficient.is_terminal());
    assert_eq!(sufficient.depth(), 2);
    assert_eq!(sufficient.coverage(), 0.9);
    assert_eq!(sufficient.as_str(), "sufficient");

    let cont = TogDecision::Continue {
        depth: 1,
        coverage: 0.3,
    };
    assert!(!cont.is_sufficient());
    assert!(!cont.is_terminal());
    assert_eq!(cont.as_str(), "continue");

    let maxed = TogDecision::MaxDepthReached {
        depth: 3,
        coverage: 0.4,
    };
    assert!(!maxed.is_sufficient());
    assert!(maxed.is_terminal());
    assert_eq!(maxed.as_str(), "max_depth_reached");

    let exhausted = TogDecision::Exhausted {
        depth: 0,
        coverage: 0.0,
    };
    assert!(exhausted.is_terminal());
    assert!(!exhausted.is_sufficient());
    assert_eq!(exhausted.as_str(), "exhausted");
}

// ── TogExploration / TogResult ──────────────────────────────────────────────

#[test]
fn exploration_new_fields() {
    let exploration = TogExploration::new(
        1,
        4,
        vec![TogBeamPath::seed("a", 1.0)],
        vec![(TogRelation::new("r"), 0.5)],
        TogDecision::Continue {
            depth: 1,
            coverage: 0.2,
        },
    );
    assert_eq!(exploration.depth, 1);
    assert_eq!(exploration.candidates_generated, 4);
    assert_eq!(exploration.beam.len(), 1);
    assert_eq!(exploration.scored_relations.len(), 1);
    assert!(!exploration.decision.is_terminal());
}

#[test]
fn result_helpers() {
    let result = TogResult {
        query: "q".to_string(),
        topic_entity_ids: vec!["a".to_string()],
        paths: vec![TogBeamPath::seed("a", 2.0), TogBeamPath::seed("b", 1.0)],
        evidence: vec![TogTriple::new("a", "r", "b")],
        answer: "Answer: X".to_string(),
        stopped_early: true,
        depth_reached: 1,
        coverage: 0.75,
        explorations: Vec::new(),
    };
    assert!(result.has_answer());
    assert_eq!(result.evidence_count(), 1);
    assert_eq!(result.path_count(), 2);
    assert_eq!(
        result.best_path().map(|p| p.topic_entity_id.as_str()),
        Some("a")
    );
}

#[test]
fn result_has_answer_false_for_blank() {
    let result = TogResult {
        query: "q".to_string(),
        topic_entity_ids: Vec::new(),
        paths: Vec::new(),
        evidence: Vec::new(),
        answer: "   ".to_string(),
        stopped_early: false,
        depth_reached: 0,
        coverage: 0.0,
        explorations: Vec::new(),
    };
    assert!(!result.has_answer());
    assert!(result.best_path().is_none());
}

// ── TogKnowledgeGraph ───────────────────────────────────────────────────────

#[test]
fn graph_new_is_empty() {
    let graph = TogKnowledgeGraph::new();
    assert!(graph.is_empty());
    assert_eq!(graph.entity_count(), 0);
    assert_eq!(graph.triple_count(), 0);
}

#[test]
fn graph_add_entity_and_lookup() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("a", "Alpha"));
    assert!(!graph.is_empty());
    assert!(graph.contains_entity("a"));
    assert_eq!(graph.entity_name("a"), Some("Alpha"));
    assert_eq!(graph.entity("a").map(|e| e.name.as_str()), Some("Alpha"));
    assert!(graph.entity("missing").is_none());
    assert_eq!(graph.entity_name("missing"), None);
}

#[test]
fn graph_add_entity_dedup_updates_name() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("a", "First"));
    graph.add_entity(TogEntity::new("a", "Second"));
    assert_eq!(graph.entity_count(), 1);
    assert_eq!(graph.entity_name("a"), Some("Second"));
}

#[test]
fn graph_add_triple_updates_adjacency() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("a", "a"));
    graph.add_entity(TogEntity::new("b", "b"));
    graph.add_triple(TogTriple::new("a", "r", "b")).expect("ok");
    assert_eq!(graph.triple_count(), 1);
    assert_eq!(graph.outgoing_triples("a").len(), 1);
    assert_eq!(graph.incoming_triples("b").len(), 1);
    assert!(graph.outgoing_triples("b").is_empty());
    assert!(graph.incoming_triples("a").is_empty());
}

#[test]
fn graph_add_triple_unknown_head_errors() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("b", "b"));
    let err = graph.add_triple(TogTriple::new("a", "r", "b"));
    assert!(matches!(err, Err(TogError::UnknownEntity { id }) if id == "a"));
}

#[test]
fn graph_add_triple_unknown_tail_errors() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("a", "a"));
    let err = graph.add_triple(TogTriple::new("a", "r", "b"));
    assert!(matches!(err, Err(TogError::UnknownEntity { id }) if id == "b"));
}

#[test]
fn graph_from_parts_ok() {
    let graph = single_hop_graph();
    assert_eq!(graph.entity_count(), 2);
    assert_eq!(graph.triple_count(), 1);
    assert_eq!(graph.entities().len(), 2);
    assert_eq!(graph.triples().len(), 1);
}

#[test]
fn graph_from_parts_unknown_entity_errors() {
    let err = TogKnowledgeGraph::from_parts(
        vec![TogEntity::new("a", "a")],
        vec![TogTriple::new("a", "r", "missing")],
    );
    assert!(matches!(err, Err(TogError::UnknownEntity { id }) if id == "missing"));
}

#[test]
fn graph_multiple_edges_same_node() {
    let graph = star_graph();
    assert_eq!(graph.outgoing_triples("hub").len(), 4);
    assert!(graph.incoming_triples("hub").is_empty());
    assert_eq!(graph.incoming_triples("nodea").len(), 1);
}

// ── Topic-entity matching ───────────────────────────────────────────────────

#[test]
fn topic_matching_substring() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    let matched = engine.match_topic_entities("What is the capital of France?", &graph);
    assert!(matched.contains(&"France".to_string()));
    assert!(!matched.contains(&"Paris".to_string()));
}

#[test]
fn topic_matching_token_for_multiword_name() {
    let mut graph = TogKnowledgeGraph::new();
    graph.add_entity(TogEntity::new("ae", "Albert Einstein"));
    graph.add_entity(TogEntity::new("nt", "Isaac Newton"));
    let engine = TogEngine::default();
    // "albert einstein" is not a substring, but the token "einstein" matches.
    let matched = engine.match_topic_entities("einstein relativity theory", &graph);
    assert!(matched.contains(&"ae".to_string()));
    assert!(!matched.contains(&"nt".to_string()));
}

#[test]
fn topic_matching_none_returns_empty() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    let matched = engine.match_topic_entities("completely unrelated words here", &graph);
    assert!(matched.is_empty());
}

// ── Error paths from run ────────────────────────────────────────────────────

#[test]
fn run_empty_query_errors() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    assert!(matches!(
        engine.run("   ", &graph),
        Err(TogError::EmptyQuery)
    ));
}

#[test]
fn run_empty_graph_errors() {
    let engine = TogEngine::default();
    let graph = TogKnowledgeGraph::new();
    assert!(matches!(
        engine.run("anything", &graph),
        Err(TogError::EmptyGraph)
    ));
}

#[test]
fn run_no_topic_entity_errors() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    assert!(matches!(
        engine.run("nothing relevant matches", &graph),
        Err(TogError::NoTopicEntity)
    ));
}

#[test]
fn run_invalid_config_errors() {
    let engine = TogEngine::new(TogConfig::default().with_beam_width(0));
    let graph = single_hop_graph();
    assert!(matches!(
        engine.run("capital of France", &graph),
        Err(TogError::InvalidConfig { .. })
    ));
}

// ── Single-hop early stop ───────────────────────────────────────────────────

#[test]
fn single_hop_answer_early_stop_at_depth_one() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    let result = engine
        .run("What is the capital of France?", &graph)
        .expect("run should succeed");
    assert!(result.stopped_early);
    assert_eq!(result.depth_reached, 1);
    assert_eq!(result.topic_entity_ids, vec!["France".to_string()]);
    let best = result.best_path().expect("a path");
    assert_eq!(best.depth(), 1);
    assert_eq!(best.current_entity_id(), "Paris");
    assert_eq!(best.hops[0].relation, "capital");
    assert!(result.coverage >= 0.6);
    let last = result.explorations.last().expect("an exploration");
    assert!(last.decision.is_sufficient());
    assert!(result.answer.contains("Paris"));
}

// ── Multi-hop chains ────────────────────────────────────────────────────────

#[test]
fn multi_hop_chain_reaches_depth_two() {
    let engine = TogEngine::default();
    let graph = linear_chain_graph();
    let result = engine
        .run("alpha reaches", &graph)
        .expect("run should succeed");
    assert_eq!(result.depth_reached, 2);
    assert!(result.stopped_early);
    let best = result.best_path().expect("a path");
    assert_eq!(best.depth(), 2);
    assert_eq!(best.current_entity_id(), "gamma");
    // Depth 1 was not yet sufficient; depth 2 was.
    assert_eq!(result.explorations.len(), 2);
    assert!(matches!(
        result.explorations[0].decision,
        TogDecision::Continue { .. }
    ));
    assert!(result.explorations[1].decision.is_sufficient());
}

#[test]
fn multi_hop_chain_reaches_depth_three() {
    let engine = TogEngine::default();
    let graph = three_hop_graph();
    let result = engine
        .run("alpha attains", &graph)
        .expect("run should succeed");
    assert_eq!(result.depth_reached, 3);
    assert!(result.stopped_early);
    let best = result.best_path().expect("a path");
    assert_eq!(best.depth(), 3);
    assert_eq!(best.current_entity_id(), "delta");
    assert_eq!(result.explorations.len(), 3);
}

// ── Beam width bounds ───────────────────────────────────────────────────────

#[test]
fn beam_width_one_retains_single_path() {
    let engine = TogEngine::new(
        TogConfig::default()
            .with_beam_width(1)
            .with_relations_per_expansion(10)
            .with_entities_per_relation(10)
            .with_max_depth(1)
            .with_sufficiency_threshold(2.0),
    );
    let graph = star_graph();
    let result = engine.run("hub", &graph).expect("run should succeed");
    assert_eq!(result.paths.len(), 1);
    assert_eq!(result.paths[0].depth(), 1);
}

#[test]
fn beam_width_three_retains_more_paths_than_width_one() {
    let base = TogConfig::default()
        .with_relations_per_expansion(10)
        .with_entities_per_relation(10)
        .with_max_depth(1)
        .with_sufficiency_threshold(2.0);
    let graph = star_graph();

    let narrow = TogEngine::new(base.with_beam_width(1))
        .run("hub", &graph)
        .expect("ok");
    let wide = TogEngine::new(base.with_beam_width(3))
        .run("hub", &graph)
        .expect("ok");

    assert_eq!(narrow.paths.len(), 1);
    assert_eq!(wide.paths.len(), 3);
    assert!(wide.paths.len() > narrow.paths.len());
}

#[test]
fn beam_never_exceeds_configured_width() {
    for width in 1..=4 {
        let engine = TogEngine::new(
            TogConfig::default()
                .with_beam_width(width)
                .with_relations_per_expansion(10)
                .with_entities_per_relation(10)
                .with_max_depth(1)
                .with_sufficiency_threshold(2.0),
        );
        let graph = star_graph();
        let result = engine.run("hub", &graph).expect("ok");
        assert!(result.paths.len() <= width);
        for exploration in &result.explorations {
            assert!(exploration.beam.len() <= width);
        }
    }
}

// ── Relation pruning ────────────────────────────────────────────────────────

#[test]
fn relation_pruning_keeps_relevant_drops_irrelevant() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("hub", "hub"),
            TogEntity::new("paris", "Paris"),
            TogEntity::new("pop", "populationvalue"),
        ],
        vec![
            TogTriple::new("hub", "capital", "paris"),
            TogTriple::new("hub", "population", "pop"),
        ],
    )
    .expect("ok");
    let engine = TogEngine::new(TogConfig::default().with_relations_per_expansion(1));
    let result = engine.run("hub capital", &graph).expect("ok");

    // Every surviving path went through the query-relevant relation only.
    for path in &result.paths {
        for hop in path.triples() {
            assert_eq!(hop.relation, "capital");
            assert_ne!(hop.relation, "population");
        }
    }
    for triple in &result.evidence {
        assert_ne!(triple.relation, "population");
    }

    // Both relations were considered; the relevant one scored strictly higher.
    let scored = &result.explorations[0].scored_relations;
    let capital = scored
        .iter()
        .find(|(r, _)| r.label() == "capital")
        .map(|(_, s)| *s)
        .expect("capital scored");
    let population = scored
        .iter()
        .find(|(r, _)| r.label() == "population")
        .map(|(_, s)| *s)
        .expect("population scored");
    assert!(capital > population);
}

// ── Sufficiency threshold behaviour ─────────────────────────────────────────

#[test]
fn sufficiency_threshold_zero_stops_at_depth_one() {
    let engine = TogEngine::new(TogConfig::default().with_sufficiency_threshold(0.0));
    let graph = linear_chain_graph();
    let result = engine.run("alpha reaches", &graph).expect("ok");
    // With a zero threshold the very first depth is already "sufficient".
    assert!(result.stopped_early);
    assert_eq!(result.depth_reached, 1);
}

#[test]
fn sufficiency_default_threshold_needs_depth_two() {
    // Same graph and query, contrasting with the zero-threshold case above.
    let engine = TogEngine::default();
    let graph = linear_chain_graph();
    let result = engine.run("alpha reaches", &graph).expect("ok");
    assert!(result.stopped_early);
    assert_eq!(result.depth_reached, 2);
}

#[test]
fn impossible_threshold_runs_to_max_depth() {
    let engine = TogEngine::new(
        TogConfig::default()
            .with_sufficiency_threshold(2.0)
            .with_max_depth(2),
    );
    let graph = linear_chain_graph();
    let result = engine.run("alpha reaches", &graph).expect("ok");
    assert!(!result.stopped_early);
    assert_eq!(result.depth_reached, 2);
    let last = result.explorations.last().expect("exploration");
    assert!(matches!(last.decision, TogDecision::MaxDepthReached { .. }));
}

// ── Disconnected topic entity ───────────────────────────────────────────────

#[test]
fn disconnected_topic_returns_seed_only() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("island", "island"),
            TogEntity::new("cityx", "cityx"),
            TogEntity::new("cityy", "cityy"),
        ],
        vec![TogTriple::new("cityx", "rel", "cityy")],
    )
    .expect("ok");
    let engine = TogEngine::default();
    let result = engine.run("island", &graph).expect("ok");
    assert!(result.evidence.is_empty());
    assert_eq!(result.depth_reached, 0);
    assert_eq!(result.paths.len(), 1);
    assert!(result.paths[0].is_seed());
    assert!(!result.stopped_early);
    let last = result.explorations.last().expect("exploration");
    assert!(matches!(last.decision, TogDecision::Exhausted { .. }));
    assert!(result.answer.contains("island"));
}

// ── Cycle avoidance ─────────────────────────────────────────────────────────

#[test]
fn cycle_avoidance_never_revisits_entity() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("nodea", "nodea"),
            TogEntity::new("nodeb", "nodeb"),
        ],
        vec![
            TogTriple::new("nodea", "rel", "nodeb"),
            TogTriple::new("nodeb", "rel", "nodea"),
        ],
    )
    .expect("ok");
    let engine = TogEngine::new(
        TogConfig::default()
            .with_beam_width(1)
            .with_max_depth(3)
            .with_sufficiency_threshold(2.0),
    );
    let result = engine.run("nodea", &graph).expect("ok");
    let best = result.best_path().expect("a path");
    // No entity appears twice on the path.
    let mut seen = std::collections::HashSet::new();
    for id in best.entities() {
        assert!(seen.insert(id.clone()), "entity {id} was revisited");
    }
    assert!(best.depth() <= 1);
}

// ── Evidence de-duplication ─────────────────────────────────────────────────

#[test]
fn evidence_is_deduplicated_across_paths() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("xxx", "xxx"),
            TogEntity::new("yyy", "yyy"),
            TogEntity::new("mmm", "mmm"),
            TogEntity::new("zzz", "zzz"),
        ],
        vec![
            TogTriple::new("xxx", "linksto", "mmm"),
            TogTriple::new("yyy", "linksto", "mmm"),
            TogTriple::new("mmm", "reaches", "zzz"),
        ],
    )
    .expect("ok");
    let engine = TogEngine::new(
        TogConfig::default()
            .with_sufficiency_threshold(2.0)
            .with_max_depth(2)
            .with_beam_width(5)
            .with_relations_per_expansion(5)
            .with_entities_per_relation(5),
    );
    let result = engine.run("xxx yyy", &graph).expect("ok");

    // The shared edge mmm -[reaches]-> zzz must appear at most once.
    let shared = TogTriple::new("mmm", "reaches", "zzz");
    let occurrences = result.evidence.iter().filter(|t| **t == shared).count();
    assert_eq!(occurrences, 1);

    // No duplicate triples at all.
    let unique: std::collections::HashSet<_> = result
        .evidence
        .iter()
        .map(|t| (t.head.clone(), t.relation.clone(), t.tail.clone()))
        .collect();
    assert_eq!(unique.len(), result.evidence.len());
}

// ── Scoring monotonicity ────────────────────────────────────────────────────

#[test]
fn scoring_relevant_hop_outscores_irrelevant_hop() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("start", "start"),
            TogEntity::new("target", "target"),
            TogEntity::new("other", "other"),
        ],
        vec![
            TogTriple::new("start", "reach", "target"),
            TogTriple::new("start", "avoid", "other"),
        ],
    )
    .expect("ok");
    let engine = TogEngine::new(
        TogConfig::default()
            .with_relations_per_expansion(5)
            .with_entities_per_relation(5)
            .with_beam_width(5)
            .with_max_depth(1)
            .with_sufficiency_threshold(2.0),
    );
    let result = engine.run("start reach target", &graph).expect("ok");

    let target_path = result
        .paths
        .iter()
        .find(|p| p.current_entity_id() == "target")
        .expect("target path");
    let other_path = result
        .paths
        .iter()
        .find(|p| p.current_entity_id() == "other")
        .expect("other path");
    assert!(target_path.score > other_path.score);
    assert!(target_path.score > 0.0);
}

#[test]
fn paths_are_ranked_best_first() {
    let engine = TogEngine::new(
        TogConfig::default()
            .with_relations_per_expansion(10)
            .with_entities_per_relation(10)
            .with_beam_width(4)
            .with_max_depth(1)
            .with_sufficiency_threshold(2.0),
    );
    let graph = star_graph();
    let result = engine.run("hub", &graph).expect("ok");
    for window in result.paths.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

// ── Determinism ─────────────────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let engine = TogEngine::default();
    let graph = linear_chain_graph();
    let first = engine.run("alpha reaches", &graph).expect("ok");
    let second = engine.run("alpha reaches", &graph).expect("ok");
    assert_eq!(first, second);
}

#[test]
fn run_is_deterministic_on_star() {
    let engine = TogEngine::new(
        TogConfig::default()
            .with_beam_width(2)
            .with_relations_per_expansion(10)
            .with_entities_per_relation(10)
            .with_max_depth(1)
            .with_sufficiency_threshold(2.0),
    );
    let graph = star_graph();
    let first = engine.run("hub", &graph).expect("ok");
    let second = engine.run("hub", &graph).expect("ok");
    assert_eq!(first.paths, second.paths);
    assert_eq!(first.evidence, second.evidence);
    assert_eq!(first.answer, second.answer);
}

// ── Multiple topic entities ─────────────────────────────────────────────────

#[test]
fn multiple_topic_entities_are_seeded() {
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("alpha", "alpha"),
            TogEntity::new("gamma", "gamma"),
            TogEntity::new("beta", "beta"),
        ],
        vec![
            TogTriple::new("alpha", "linksto", "beta"),
            TogTriple::new("gamma", "linksto", "beta"),
        ],
    )
    .expect("ok");
    let engine = TogEngine::default();
    let result = engine.run("alpha and gamma together", &graph).expect("ok");
    assert!(result.topic_entity_ids.contains(&"alpha".to_string()));
    assert!(result.topic_entity_ids.contains(&"gamma".to_string()));
    assert!(result.topic_entity_ids.len() >= 2);
}

// ── Bidirectional traversal ─────────────────────────────────────────────────

#[test]
fn traversal_follows_incoming_edges() {
    // The topic entity is only reachable *into*; the answer lies through an
    // incoming edge, which bidirectional exploration must still follow.
    let graph = TogKnowledgeGraph::from_parts(
        vec![
            TogEntity::new("author", "author"),
            TogEntity::new("book", "book"),
        ],
        vec![TogTriple::new("author", "wrote", "book")],
    )
    .expect("ok");
    let engine = TogEngine::new(
        TogConfig::default()
            .with_max_depth(1)
            .with_sufficiency_threshold(2.0),
    );
    // Seed on "book"; the only edge is incoming (author -> book).
    let result = engine.run("book", &graph).expect("ok");
    let best = result.best_path().expect("path");
    assert_eq!(best.current_entity_id(), "author");
    assert_eq!(best.hops[0].relation, "wrote");
}

// ── Answer synthesis ────────────────────────────────────────────────────────

#[test]
fn answer_is_non_empty_and_names_target() {
    let engine = TogEngine::default();
    let graph = single_hop_graph();
    let result = engine.run("capital of France", &graph).expect("ok");
    assert!(result.has_answer());
    assert!(result.answer.contains("Paris"));
    assert!(result.answer.contains("Reasoning path"));
}

#[test]
fn explorations_have_ascending_depths() {
    let engine = TogEngine::default();
    let graph = three_hop_graph();
    let result = engine.run("alpha attains", &graph).expect("ok");
    for (index, exploration) in result.explorations.iter().enumerate() {
        assert_eq!(exploration.depth, index + 1);
    }
}

#[test]
fn min_relation_score_floor_drops_low_relations() {
    // A very high floor drops all relations, so no expansion happens and the
    // search terminates as exhausted at depth 1 with only the seed.
    let engine = TogEngine::new(
        TogConfig::default()
            .with_min_relation_score(1.5)
            .with_sufficiency_threshold(2.0),
    );
    let graph = single_hop_graph();
    let result = engine.run("capital of France", &graph).expect("ok");
    assert!(result.evidence.is_empty());
    assert_eq!(result.depth_reached, 0);
    assert!(result.paths[0].is_seed());
}
