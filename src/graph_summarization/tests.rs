//! Tests for the `graph_summarization` module.

use super::search::GlobalSearchEngine;
use super::types::{CommunitySummary, GraphSummarizationConfig, GraphSummarizationError};
use crate::graph_community::types::CommunityId;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_summary(id: usize, title: &str, summary: &str, key: &[&str]) -> CommunitySummary {
    CommunitySummary {
        community_id: CommunityId::new(id),
        title: title.to_string(),
        summary: summary.to_string(),
        key_entities: key.iter().map(ToString::to_string).collect(),
    }
}

// ── GraphSummarizationConfig tests ────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = GraphSummarizationConfig::default();
    assert_eq!(cfg.max_summary_sentences, 3);
    assert_eq!(cfg.top_communities, 5);
}

#[test]
fn test_config_builders() {
    let cfg = GraphSummarizationConfig::default()
        .with_max_summary_sentences(2)
        .with_top_communities(10);
    assert_eq!(cfg.max_summary_sentences, 2);
    assert_eq!(cfg.top_communities, 10);
}

// ── CommunitySummary tests ────────────────────────────────────────────────────

#[test]
fn test_summary_is_empty() {
    let s = make_summary(0, "title", "", &[]);
    assert!(s.is_empty());
}

#[test]
fn test_summary_not_empty() {
    let s = make_summary(1, "title", "Some content", &["Entity"]);
    assert!(!s.is_empty());
}

// ── SummaryReport tests ───────────────────────────────────────────────────────

#[test]
fn test_summary_report_is_empty() {
    use super::types::SummaryReport;
    let r = SummaryReport {
        answer: String::new(),
        used_communities: Vec::new(),
    };
    assert!(r.is_empty());
}

// ── GlobalSearchEngine tests ──────────────────────────────────────────────────

#[test]
fn test_global_search_empty_summaries_error() {
    let engine = GlobalSearchEngine::new();
    let cfg = GraphSummarizationConfig::default();
    let err = engine.search("query", &[], &cfg).expect_err("should fail");
    assert!(matches!(err, GraphSummarizationError::NoCommunities));
}

#[test]
fn test_global_search_no_match_returns_empty_answer() {
    let engine = GlobalSearchEngine::new();
    let cfg = GraphSummarizationConfig::default();
    let summaries = vec![make_summary(0, "XYZ", "xylophone and zebra fauna", &["X"])];
    let report = engine
        .search("completely unrelated query", &summaries, &cfg)
        .expect("ok");
    assert!(report.is_empty(), "no matching community → empty answer");
}

#[test]
fn test_global_search_matching_returns_answer() {
    let engine = GlobalSearchEngine::new();
    let cfg = GraphSummarizationConfig::default();
    let summaries = vec![make_summary(
        0,
        "Rust",
        "Rust is a systems programming language focused on safety.",
        &["Rust"],
    )];
    let report = engine
        .search("rust programming", &summaries, &cfg)
        .expect("ok");
    assert!(
        !report.answer.is_empty(),
        "matching summary should contribute"
    );
    assert!(!report.used_communities.is_empty());
}

#[test]
fn test_global_search_top_communities_limit() {
    let engine = GlobalSearchEngine::new();
    let cfg = GraphSummarizationConfig::default().with_top_communities(2);
    let summaries: Vec<_> = (0..6)
        .map(|i| make_summary(i, &format!("Comm{i}"), &format!("rust content {i}"), &[]))
        .collect();
    let report = engine.search("rust", &summaries, &cfg).expect("ok");
    assert!(
        report.used_communities.len() <= 2,
        "should cap at top_communities=2"
    );
}

#[test]
fn test_global_search_multiple_summaries() {
    let engine = GlobalSearchEngine::new();
    let cfg = GraphSummarizationConfig::default();
    let summaries = vec![
        make_summary(0, "Languages", "Rust is safe and fast.", &["Rust"]),
        make_summary(
            1,
            "Patterns",
            "Ownership pattern prevents data races.",
            &["Ownership"],
        ),
        make_summary(2, "Tools", "Cargo manages Rust packages.", &["Cargo"]),
    ];
    let report = engine
        .search("rust ownership", &summaries, &cfg)
        .expect("ok");
    assert!(!report.answer.is_empty());
}

// ── graphrag-gated tests ──────────────────────────────────────────────────────

#[cfg(feature = "graphrag")]
mod graphrag_tests {
    use super::super::search::LocalSearchEngine;
    use super::super::summarizer::CommunitySummarizer;
    use super::super::types::GraphSummarizationConfig;
    use super::make_summary;
    use crate::graph_community::types::{Community, CommunityGraph, CommunityId};
    use crate::layer4_graph::types::{
        EntityType, GraphEntity, GraphRelationship, RelationshipType,
    };

    fn make_entity(id: &str, name: &str) -> GraphEntity {
        GraphEntity::new(name, EntityType::Concept).with_id(id)
    }

    fn make_rel(src: &str, tgt: &str, rel_type: RelationshipType) -> GraphRelationship {
        GraphRelationship::new(src, tgt, rel_type).with_id(format!("{src}-{tgt}"))
    }

    fn make_community_graph(ids: &[&str]) -> (CommunityGraph, Vec<GraphEntity>) {
        let entities: Vec<GraphEntity> = ids.iter().map(|id| make_entity(id, id)).collect();
        let community = Community {
            id: CommunityId::new(0),
            members: ids.iter().map(ToString::to_string).collect(),
            level: 0,
        };
        let graph = CommunityGraph {
            communities: vec![community],
            modularity: 0.5,
        };
        (graph, entities)
    }

    #[test]
    fn test_summarizer_produces_summaries() {
        let (graph, entities) = make_community_graph(&["rust", "cargo"]);
        let rels = vec![make_rel("rust", "cargo", RelationshipType::Uses)];
        let cfg = GraphSummarizationConfig::default();
        let summarizer = CommunitySummarizer::new();
        let summaries = summarizer.summarize(&graph, &entities, &rels, &cfg);
        assert_eq!(summaries.len(), 1);
        assert!(!summaries[0].summary.is_empty());
    }

    #[test]
    fn test_summarizer_key_entities_present() {
        let (graph, entities) = make_community_graph(&["alpha", "beta", "gamma"]);
        let rels = vec![
            make_rel("alpha", "beta", RelationshipType::RelatedTo),
            make_rel("beta", "gamma", RelationshipType::RelatedTo),
        ];
        let cfg = GraphSummarizationConfig::default();
        let summarizer = CommunitySummarizer::new();
        let summaries = summarizer.summarize(&graph, &entities, &rels, &cfg);
        assert!(!summaries[0].key_entities.is_empty());
    }

    #[test]
    fn test_summarizer_no_internal_rels_fallback() {
        let (graph, entities) = make_community_graph(&["only_node"]);
        let cfg = GraphSummarizationConfig::default();
        let summarizer = CommunitySummarizer::new();
        let summaries = summarizer.summarize(&graph, &entities, &[], &cfg);
        assert!(
            !summaries[0].summary.is_empty(),
            "fallback summary when no rels"
        );
    }

    #[test]
    fn test_local_search_no_summaries_error() {
        let engine = LocalSearchEngine::new();
        let cfg = GraphSummarizationConfig::default();
        let entities = vec![make_entity("e1", "Rust")];
        let err = engine
            .search("rust", &entities, &[], &[], &cfg)
            .expect_err("should fail");
        assert!(matches!(
            err,
            super::super::types::GraphSummarizationError::NoCommunities
        ));
    }

    #[test]
    fn test_local_search_entity_match() {
        let engine = LocalSearchEngine::new();
        let cfg = GraphSummarizationConfig::default();
        let entities = vec![make_entity("e1", "Rust"), make_entity("e2", "Cargo")];
        let rels = vec![make_rel("e1", "e2", RelationshipType::Uses)];
        let summaries = vec![make_summary(
            0,
            "Rust",
            "Rust uses Cargo.",
            &["Rust", "Cargo"],
        )];
        let report = engine
            .search("rust", &entities, &rels, &summaries, &cfg)
            .expect("ok");
        assert!(!report.answer.is_empty(), "entity match should contribute");
    }
}
