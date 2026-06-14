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
//! Tests for the `drift_search` module.

use super::engine::DriftSearchEngine;
use super::types::{
    CommunityReport, DriftAnswer, DriftConfig, DriftError, DriftStep, DriftStepKind,
};
use crate::types::{Document, DocumentId};

// ── helpers ─────────────────────────────────────────────────────────────────────

fn pid(id: &str) -> DocumentId {
    DocumentId::from_string(id)
}

fn passage(id: &str, content: &str) -> Document {
    Document::new(content).with_id(pid(id))
}

/// A small corpus with three thematically distinct communities:
/// community 0 (rust/safety), community 1 (ocean/marine), community 2 (cooking).
fn rust_passages() -> Vec<Document> {
    vec![
        passage(
            "p1",
            "Rust guarantees memory safety without a garbage collector at runtime.",
        ),
        passage(
            "p2",
            "Ownership and borrowing rules enforce safety guarantees during compilation.",
        ),
        passage(
            "p3",
            "The borrow checker validates lifetimes and prevents data races statically.",
        ),
        passage(
            "o1",
            "Coral reefs host diverse marine fish populations in warm ocean waters.",
        ),
        passage(
            "o2",
            "Plankton drift through ocean currents feeding whales and other marine life.",
        ),
        passage(
            "c1",
            "Simmer onions garlic and tomatoes to build a rich pasta sauce flavour.",
        ),
    ]
}

fn rust_reports() -> Vec<CommunityReport> {
    vec![
        CommunityReport::new(
            0,
            "The Rust safety community covers ownership borrowing lifetimes and the borrow checker enforcing memory safety guarantees.",
            vec!["Rust".into(), "ownership".into(), "borrow".into()],
            vec![pid("p1"), pid("p2"), pid("p3")],
        ),
        CommunityReport::new(
            1,
            "The marine ecology community describes coral reefs plankton ocean currents fish and whales.",
            vec!["coral".into(), "plankton".into(), "ocean".into()],
            vec![pid("o1"), pid("o2")],
        ),
        CommunityReport::new(
            2,
            "The cooking community discusses simmering onions garlic tomatoes and pasta sauce.",
            vec!["onions".into(), "garlic".into(), "pasta".into()],
            vec![pid("c1")],
        ),
    ]
}

fn built_engine(config: DriftConfig) -> DriftSearchEngine {
    let mut engine = DriftSearchEngine::new(config);
    engine
        .build(rust_reports(), &rust_passages())
        .expect("build should succeed");
    engine
}

// ── DriftConfig defaults ────────────────────────────────────────────────────────

#[test]
fn test_config_default_top_communities() {
    assert_eq!(DriftConfig::default().top_communities, 3);
}

#[test]
fn test_config_default_follow_ups() {
    assert_eq!(DriftConfig::default().follow_ups, 2);
}

#[test]
fn test_config_default_top_local() {
    assert_eq!(DriftConfig::default().top_local, 5);
}

#[test]
fn test_config_default_dim() {
    assert_eq!(DriftConfig::default().dim, 128);
}

// ── DriftConfig builders ────────────────────────────────────────────────────────

#[test]
fn test_config_with_top_communities() {
    let cfg = DriftConfig::new().with_top_communities(7);
    assert_eq!(cfg.top_communities, 7);
}

#[test]
fn test_config_with_follow_ups() {
    let cfg = DriftConfig::new().with_follow_ups(9);
    assert_eq!(cfg.follow_ups, 9);
}

#[test]
fn test_config_with_top_local() {
    let cfg = DriftConfig::new().with_top_local(11);
    assert_eq!(cfg.top_local, 11);
}

#[test]
fn test_config_with_dim() {
    let cfg = DriftConfig::new().with_dim(256);
    assert_eq!(cfg.dim, 256);
}

// ── CommunityReport ─────────────────────────────────────────────────────────────

#[test]
fn test_community_report_new_fields() {
    let r = CommunityReport::new(5, "summary text", vec!["E".into()], vec![pid("x")]);
    assert_eq!(r.id, 5);
    assert_eq!(r.summary, "summary text");
    assert_eq!(r.entities, vec!["E".to_string()]);
    assert_eq!(r.passage_ids, vec![pid("x")]);
}

#[test]
fn test_community_report_is_empty_true() {
    let r = CommunityReport::new(0, "   ", vec![], vec![]);
    assert!(r.is_empty());
}

#[test]
fn test_community_report_is_empty_false() {
    let r = CommunityReport::new(0, "non-empty", vec![], vec![]);
    assert!(!r.is_empty());
}

// ── DriftStepKind ───────────────────────────────────────────────────────────────

#[test]
fn test_step_kind_is_global() {
    assert!(DriftStepKind::Global.is_global());
    assert!(!DriftStepKind::Local.is_global());
}

#[test]
fn test_step_kind_is_local() {
    assert!(DriftStepKind::Local.is_local());
    assert!(!DriftStepKind::Global.is_local());
}

#[test]
fn test_step_kind_label() {
    assert_eq!(DriftStepKind::Global.label(), "global");
    assert_eq!(DriftStepKind::Local.label(), "local");
}

// ── DriftStep ───────────────────────────────────────────────────────────────────

#[test]
fn test_step_new_and_len() {
    let step = DriftStep::new(
        "q",
        DriftStepKind::Local,
        vec![(pid("a"), 0.5), (pid("b"), 0.25)],
    );
    assert_eq!(step.query, "q");
    assert_eq!(step.kind, DriftStepKind::Local);
    assert_eq!(step.len(), 2);
    assert!(!step.is_empty());
}

#[test]
fn test_step_is_empty() {
    let step = DriftStep::new("q", DriftStepKind::Global, Vec::new());
    assert!(step.is_empty());
    assert_eq!(step.len(), 0);
}

// ── build / is_built ────────────────────────────────────────────────────────────

#[test]
fn test_engine_not_built_initially() {
    let engine = DriftSearchEngine::new(DriftConfig::default());
    assert!(!engine.is_built());
}

#[test]
fn test_engine_built_after_build() {
    let engine = built_engine(DriftConfig::default());
    assert!(engine.is_built());
}

#[test]
fn test_engine_build_records_counts() {
    let engine = built_engine(DriftConfig::default());
    assert_eq!(engine.community_count(), 3);
    assert_eq!(engine.passage_count(), 6);
}

#[test]
fn test_engine_build_empty_communities_errors() {
    let mut engine = DriftSearchEngine::new(DriftConfig::default());
    let err = engine.build(Vec::new(), &rust_passages()).unwrap_err();
    assert!(matches!(err, DriftError::EmptyCommunities));
    assert!(!engine.is_built());
}

#[test]
fn test_engine_rebuild_replaces_state() {
    let mut engine = DriftSearchEngine::new(DriftConfig::default());
    engine.build(rust_reports(), &rust_passages()).unwrap();
    assert_eq!(engine.community_count(), 3);

    let smaller = vec![CommunityReport::new(
        0,
        "only one community here",
        vec!["one".into()],
        vec![pid("p1")],
    )];
    engine
        .build(smaller, &[passage("p1", "single passage")])
        .unwrap();
    assert_eq!(engine.community_count(), 1);
    assert_eq!(engine.passage_count(), 1);
}

// ── global step selection ───────────────────────────────────────────────────────

#[test]
fn test_global_selects_most_relevant_community_rust() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine
        .search("memory safety ownership borrow checker")
        .unwrap();
    assert_eq!(answer.communities, vec![0]);
}

#[test]
fn test_global_selects_most_relevant_community_marine() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine
        .search("coral reefs plankton ocean currents")
        .unwrap();
    assert_eq!(answer.communities, vec![1]);
}

#[test]
fn test_global_selects_most_relevant_community_cooking() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine.search("simmer onions garlic pasta sauce").unwrap();
    assert_eq!(answer.communities, vec![2]);
}

#[test]
fn test_global_step_results_use_community_ids() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine.search("memory safety ownership").unwrap();
    let global = &answer.steps[0];
    assert_eq!(global.results.len(), 1);
    assert_eq!(global.results[0].0.as_str(), "community:0");
}

#[test]
fn test_global_respects_top_communities_cap() {
    let engine = built_engine(DriftConfig::new().with_top_communities(2));
    let answer = engine.search("safety ocean").unwrap();
    assert_eq!(answer.communities.len(), 2);
    assert_eq!(answer.steps[0].results.len(), 2);
}

// ── follow-up generation ────────────────────────────────────────────────────────

#[test]
fn test_follow_ups_count_respects_config() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("safety").unwrap();
    let local_count = answer.local_steps().count();
    assert!(
        local_count <= 2,
        "expected at most 2 local steps, got {local_count}"
    );
}

#[test]
fn test_follow_ups_zero_yields_no_local_steps() {
    let engine = built_engine(DriftConfig::new().with_follow_ups(0));
    let answer = engine.search("memory safety").unwrap();
    assert_eq!(answer.local_steps().count(), 0);
}

#[test]
fn test_follow_ups_derived_from_salient_terms() {
    // Selecting only the rust community, follow-ups should mention its entities.
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(3),
    );
    let answer = engine.search("safety").unwrap();
    let joined: String = answer
        .local_steps()
        .map(|s| s.query.clone())
        .collect::<Vec<_>>()
        .join(" | ");
    // Salient terms come from the rust community entities/summary.
    assert!(
        joined.contains("rust")
            || joined.contains("ownership")
            || joined.contains("borrow")
            || joined.contains("lifetimes")
            || joined.contains("checker"),
        "follow-ups should derive from salient terms, got: {joined}"
    );
}

#[test]
fn test_follow_ups_include_original_query() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("safety").unwrap();
    for step in answer.local_steps() {
        assert!(
            step.query.starts_with("safety"),
            "follow-up should extend the original query: {}",
            step.query
        );
    }
}

#[test]
fn test_follow_ups_are_distinct() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(3),
    );
    let answer = engine.search("safety").unwrap();
    let queries: Vec<String> = answer.local_steps().map(|s| s.query.clone()).collect();
    let mut sorted = queries.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        queries.len(),
        "follow-up queries must be distinct"
    );
}

// ── local step scoping ──────────────────────────────────────────────────────────

#[test]
fn test_local_searches_only_selected_communities() {
    // Select only the rust community; local results must be rust passages.
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("memory safety ownership").unwrap();
    assert_eq!(answer.communities, vec![0]);
    for step in answer.local_steps() {
        for (id, _) in &step.results {
            assert!(
                id.as_str().starts_with('p'),
                "local result {} escaped the selected rust community",
                id.as_str()
            );
        }
    }
}

#[test]
fn test_local_excludes_other_community_passages() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("coral plankton ocean").unwrap();
    assert_eq!(answer.communities, vec![1]);
    for step in answer.local_steps() {
        for (id, _) in &step.results {
            assert!(
                id.as_str().starts_with('o'),
                "marine search should not retrieve {}",
                id.as_str()
            );
        }
    }
}

#[test]
fn test_local_respects_top_local_cap() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2)
            .with_top_local(1),
    );
    let answer = engine.search("memory safety").unwrap();
    for step in answer.local_steps() {
        assert!(step.results.len() <= 1, "top_local cap violated");
    }
}

#[test]
fn test_local_results_sorted_descending() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("memory safety ownership").unwrap();
    for step in answer.local_steps() {
        for w in step.results.windows(2) {
            assert!(w[0].1 >= w[1].1, "local results not sorted by score");
        }
    }
}

// ── DriftAnswer structure ───────────────────────────────────────────────────────

#[test]
fn test_answer_has_global_and_local_steps() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("memory safety ownership").unwrap();
    assert_eq!(answer.global_steps().count(), 1);
    assert!(answer.local_steps().count() >= 1);
}

#[test]
fn test_answer_final_context_aggregates_passages() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(2),
    );
    let answer = engine.search("memory safety ownership borrow").unwrap();
    assert!(!answer.final_context.is_empty());
    for id in &answer.final_context {
        assert!(id.as_str().starts_with('p'));
    }
}

#[test]
fn test_answer_final_context_deduplicated() {
    let engine = built_engine(
        DriftConfig::new()
            .with_top_communities(1)
            .with_follow_ups(3),
    );
    let answer = engine.search("memory safety ownership borrow").unwrap();
    let mut sorted: Vec<String> = answer
        .final_context
        .iter()
        .map(|d| d.as_str().to_string())
        .collect();
    let n = sorted.len();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), n, "final_context must not contain duplicates");
}

#[test]
fn test_answer_communities_match_selection() {
    let engine = built_engine(DriftConfig::new().with_top_communities(2));
    let answer = engine.search("safety ocean coral").unwrap();
    assert!(answer.communities.contains(&0) || answer.communities.contains(&1));
    assert_eq!(answer.communities.len(), 2);
}

#[test]
fn test_answer_summary_non_empty_on_match() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine.search("memory safety").unwrap();
    assert!(!answer.summary.is_empty());
}

#[test]
fn test_answer_summary_from_selected_community() {
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine.search("coral plankton ocean").unwrap();
    assert!(answer.summary.contains("marine") || answer.summary.contains("coral"));
}

#[test]
fn test_answer_is_empty_helper() {
    let answer = DriftAnswer {
        steps: Vec::new(),
        communities: Vec::new(),
        final_context: Vec::new(),
        summary: String::new(),
    };
    assert!(answer.is_empty());
}

// ── errors ──────────────────────────────────────────────────────────────────────

#[test]
fn test_search_not_built_errors() {
    let engine = DriftSearchEngine::new(DriftConfig::default());
    let err = engine.search("anything").unwrap_err();
    assert!(matches!(err, DriftError::NotBuilt));
}

#[test]
fn test_search_empty_query_errors() {
    let engine = built_engine(DriftConfig::default());
    let err = engine.search("   ").unwrap_err();
    assert!(matches!(err, DriftError::EmptyQuery));
}

#[test]
fn test_error_display_messages() {
    assert_eq!(DriftError::EmptyCommunities.to_string(), "no communities");
    assert_eq!(
        DriftError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(DriftError::NotBuilt.to_string(), "engine not built");
}

#[test]
fn test_not_built_takes_precedence_over_empty_query() {
    let engine = DriftSearchEngine::new(DriftConfig::default());
    let err = engine.search("").unwrap_err();
    assert!(matches!(err, DriftError::NotBuilt));
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn test_search_is_deterministic_communities() {
    let engine = built_engine(DriftConfig::default());
    let a = engine.search("memory safety ownership").unwrap();
    let b = engine.search("memory safety ownership").unwrap();
    assert_eq!(a.communities, b.communities);
}

#[test]
fn test_search_is_deterministic_context() {
    let engine = built_engine(DriftConfig::default());
    let a = engine.search("memory safety ownership").unwrap();
    let b = engine.search("memory safety ownership").unwrap();
    let ai: Vec<&str> = a.final_context.iter().map(DocumentId::as_str).collect();
    let bi: Vec<&str> = b.final_context.iter().map(DocumentId::as_str).collect();
    assert_eq!(ai, bi);
}

#[test]
fn test_rebuild_then_search_deterministic() {
    let a = built_engine(DriftConfig::default())
        .search("memory safety")
        .unwrap()
        .communities;
    let b = built_engine(DriftConfig::default())
        .search("memory safety")
        .unwrap()
        .communities;
    assert_eq!(a, b);
}

// ── edge cases ──────────────────────────────────────────────────────────────────

#[test]
fn test_community_with_missing_passages_skipped() {
    // Report references a passage that is not supplied; local scope must skip it.
    let mut engine = DriftSearchEngine::new(DriftConfig::new().with_top_communities(1));
    let reports = vec![CommunityReport::new(
        0,
        "topic about gardens flowers and soil nutrients for plants",
        vec!["gardens".into(), "flowers".into()],
        vec![pid("present"), pid("absent")],
    )];
    let passages = vec![passage(
        "present",
        "Flowers in the garden need rich soil and water.",
    )];
    engine.build(reports, &passages).unwrap();
    let answer = engine.search("gardens flowers soil").unwrap();
    for id in &answer.final_context {
        assert_ne!(id.as_str(), "absent");
    }
}

#[test]
fn test_search_with_no_passages_yields_empty_context() {
    let mut engine = DriftSearchEngine::new(DriftConfig::default());
    let reports = vec![CommunityReport::new(
        0,
        "a community summary with entities but no passages attached",
        vec!["alpha".into()],
        Vec::new(),
    )];
    engine.build(reports, &[]).unwrap();
    let answer = engine.search("alpha community summary").unwrap();
    assert!(answer.final_context.is_empty());
    // A global step is still recorded.
    assert_eq!(answer.global_steps().count(), 1);
}

#[test]
fn test_top_communities_larger_than_corpus() {
    let engine = built_engine(DriftConfig::new().with_top_communities(100));
    let answer = engine.search("safety ocean cooking").unwrap();
    assert_eq!(answer.communities.len(), 3);
}

#[test]
fn test_irrelevant_query_still_returns_answer() {
    // A query with no shared vocabulary still selects communities (best-effort)
    // and returns a well-formed answer without panicking.
    let engine = built_engine(DriftConfig::new().with_top_communities(1));
    let answer = engine.search("xyzzy quux blorp").unwrap();
    assert_eq!(answer.global_steps().count(), 1);
    assert_eq!(answer.communities.len(), 1);
}

#[test]
fn test_single_community_single_passage() {
    let mut engine = DriftSearchEngine::new(DriftConfig::new().with_top_communities(1));
    let reports = vec![CommunityReport::new(
        7,
        "quantum entanglement links particles across distance instantaneously",
        vec!["quantum".into(), "entanglement".into()],
        vec![pid("q1")],
    )];
    let passages = vec![passage(
        "q1",
        "Entangled quantum particles share state regardless of separation distance.",
    )];
    engine.build(reports, &passages).unwrap();
    let answer = engine.search("quantum entanglement particles").unwrap();
    assert_eq!(answer.communities, vec![7]);
    assert!(answer.final_context.iter().any(|d| d.as_str() == "q1"));
}
