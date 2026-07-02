#![allow(
    clippy::similar_names,
    clippy::useless_vec,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::redundant_clone,
    clippy::manual_range_contains
)]

//! Tests for the `rq_rag` module.

use super::engine::{MockRefiner, QueryRefinementEngine, QueryRefiner};
use super::types::{
    AMBIGUITY_SENSE_TABLE, DEFAULT_COLLOQUIAL_MARKERS, RefinementAction, RefinementPlan,
    RqRagConfig, RqRagError,
};

use std::collections::HashSet;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn default_refiner() -> MockRefiner {
    MockRefiner::default()
}

fn default_engine() -> QueryRefinementEngine<MockRefiner> {
    QueryRefinementEngine::new(RqRagConfig::default(), MockRefiner::default())
}

/// Lower-cased, punctuation-stripped whitespace tokens of `text`.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .collect()
}

// ── RefinementAction ──────────────────────────────────────────────────────────

#[test]
fn action_as_str_rewrite() {
    assert_eq!(RefinementAction::Rewrite.as_str(), "rewrite");
}

#[test]
fn action_as_str_decompose() {
    assert_eq!(RefinementAction::Decompose.as_str(), "decompose");
}

#[test]
fn action_as_str_disambiguate() {
    assert_eq!(RefinementAction::Disambiguate.as_str(), "disambiguate");
}

#[test]
fn action_as_str_respond() {
    assert_eq!(RefinementAction::Respond.as_str(), "respond");
}

#[test]
fn action_display_matches_as_str() {
    for action in [
        RefinementAction::Rewrite,
        RefinementAction::Decompose,
        RefinementAction::Disambiguate,
        RefinementAction::Respond,
    ] {
        assert_eq!(format!("{action}"), action.as_str());
    }
}

#[test]
fn action_equality() {
    assert_eq!(RefinementAction::Respond, RefinementAction::Respond);
    assert_ne!(RefinementAction::Respond, RefinementAction::Rewrite);
}

#[test]
fn action_copy_semantics() {
    let a = RefinementAction::Decompose;
    let b = a; // Copy, not move.
    assert_eq!(a, b);
}

#[test]
fn action_clone() {
    let a = RefinementAction::Disambiguate;
    #[allow(clippy::clone_on_copy)]
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn action_hashable_in_set() {
    let mut set = HashSet::new();
    set.insert(RefinementAction::Rewrite);
    set.insert(RefinementAction::Rewrite);
    set.insert(RefinementAction::Decompose);
    assert_eq!(set.len(), 2);
}

#[test]
fn action_debug_format_nonempty() {
    let s = format!("{:?}", RefinementAction::Respond);
    assert!(!s.is_empty());
}

// ── RefinementPlan ────────────────────────────────────────────────────────────

#[test]
fn plan_query_count_single() {
    let plan = RefinementPlan {
        original_query: "q".to_string(),
        action: RefinementAction::Respond,
        refined_queries: vec!["q".to_string()],
    };
    assert_eq!(plan.query_count(), 1);
    assert!(plan.is_single());
}

#[test]
fn plan_query_count_multiple() {
    let plan = RefinementPlan {
        original_query: "q".to_string(),
        action: RefinementAction::Decompose,
        refined_queries: vec!["a".to_string(), "b".to_string()],
    };
    assert_eq!(plan.query_count(), 2);
    assert!(!plan.is_single());
}

#[test]
fn plan_clone_and_eq() {
    let plan = RefinementPlan {
        original_query: "q".to_string(),
        action: RefinementAction::Rewrite,
        refined_queries: vec!["q2".to_string()],
    };
    let cloned = plan.clone();
    assert_eq!(plan, cloned);
}

// ── RqRagConfig ───────────────────────────────────────────────────────────────

#[test]
fn config_default_min_decompose_parts() {
    assert_eq!(RqRagConfig::default().min_decompose_parts, 2);
}

#[test]
fn config_default_ambiguity_markers_nonempty() {
    assert!(!RqRagConfig::default().ambiguity_markers.is_empty());
}

#[test]
fn config_default_colloquial_markers_nonempty() {
    assert!(!RqRagConfig::default().colloquial_markers.is_empty());
}

#[test]
fn config_new_equals_default() {
    assert_eq!(RqRagConfig::new(), RqRagConfig::default());
}

#[test]
fn config_builder_min_decompose_parts() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(3);
    assert_eq!(cfg.min_decompose_parts, 3);
}

#[test]
fn config_builder_ambiguity_markers() {
    let cfg = RqRagConfig::new().with_ambiguity_markers(vec!["widget".to_string()]);
    assert_eq!(cfg.ambiguity_markers, vec!["widget".to_string()]);
}

#[test]
fn config_builder_colloquial_markers() {
    let cfg = RqRagConfig::new().with_colloquial_markers(vec!["meh".to_string()]);
    assert_eq!(cfg.colloquial_markers, vec!["meh".to_string()]);
}

#[test]
fn config_builder_extra_ambiguity_marker() {
    let base_len = RqRagConfig::default().ambiguity_markers.len();
    let cfg = RqRagConfig::new().with_extra_ambiguity_marker("gizmo");
    assert_eq!(cfg.ambiguity_markers.len(), base_len + 1);
    assert!(cfg.ambiguity_markers.contains(&"gizmo".to_string()));
}

#[test]
fn config_builder_extra_colloquial_marker() {
    let base_len = RqRagConfig::default().colloquial_markers.len();
    let cfg = RqRagConfig::new().with_extra_colloquial_marker("meh");
    assert_eq!(cfg.colloquial_markers.len(), base_len + 1);
    assert!(cfg.colloquial_markers.contains(&"meh".to_string()));
}

#[test]
fn config_validate_ok_for_default() {
    assert!(RqRagConfig::default().validate().is_ok());
}

#[test]
fn config_validate_err_for_zero() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(0);
    assert!(matches!(cfg.validate(), Err(RqRagError::InvalidConfig(_))));
}

#[test]
fn config_validate_err_for_one() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(1);
    assert!(matches!(cfg.validate(), Err(RqRagError::InvalidConfig(_))));
}

#[test]
fn config_validate_ok_for_higher_threshold() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(5);
    assert!(cfg.validate().is_ok());
}

// ── RqRagError ────────────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    assert_eq!(RqRagError::EmptyQuery.to_string(), "query is empty");
}

#[test]
fn error_invalid_config_display() {
    let err = RqRagError::InvalidConfig("bad value".to_string());
    assert!(err.to_string().contains("bad value"));
}

#[test]
fn error_equality() {
    assert_eq!(RqRagError::EmptyQuery, RqRagError::EmptyQuery);
    assert_ne!(
        RqRagError::EmptyQuery,
        RqRagError::InvalidConfig("x".to_string())
    );
}

#[test]
fn error_clone() {
    let err = RqRagError::InvalidConfig("y".to_string());
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

// ── AMBIGUITY_SENSE_TABLE / DEFAULT_COLLOQUIAL_MARKERS invariants ────────────

#[test]
fn sense_table_contains_bank() {
    assert!(AMBIGUITY_SENSE_TABLE.iter().any(|entry| entry.0 == "bank"));
}

#[test]
fn sense_table_contains_mercury_with_three_senses() {
    let entry = AMBIGUITY_SENSE_TABLE
        .iter()
        .find(|entry| entry.0 == "mercury");
    assert_eq!(entry.map(|entry| entry.1.len()), Some(3));
}

#[test]
fn sense_table_every_entry_has_multiple_senses() {
    for entry in AMBIGUITY_SENSE_TABLE {
        assert!(
            entry.1.len() >= 2,
            "term {} has fewer than 2 listed senses",
            entry.0
        );
    }
}

#[test]
fn sense_table_at_least_ten_entries() {
    assert!(AMBIGUITY_SENSE_TABLE.len() >= 10);
}

#[test]
fn default_colloquial_markers_at_least_ten() {
    assert!(DEFAULT_COLLOQUIAL_MARKERS.len() >= 10);
}

// ── MockRefiner::classify — Respond ───────────────────────────────────────────

#[test]
fn classify_respond_atomic_question() {
    assert_eq!(
        default_refiner().classify("What is Rust?"),
        RefinementAction::Respond
    );
}

#[test]
fn classify_respond_single_word() {
    assert_eq!(
        default_refiner().classify("Rust"),
        RefinementAction::Respond
    );
}

#[test]
fn classify_respond_empty_query() {
    assert_eq!(default_refiner().classify(""), RefinementAction::Respond);
}

#[test]
fn classify_respond_whitespace_only_query() {
    assert_eq!(default_refiner().classify("   "), RefinementAction::Respond);
}

// ── MockRefiner::classify — Decompose ─────────────────────────────────────────

#[test]
fn classify_decompose_and_conjunction() {
    let q = "Tell me about Rust and tell me about C++";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_decompose_or_conjunction() {
    let q = "Should I use Rust or should I use Go";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_decompose_multi_question_marks() {
    let q = "What is Rust? What is Go?";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_decompose_semicolon() {
    let q = "Explain Rust; explain Go";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_decompose_as_well_as() {
    let q = "I want to learn Rust as well as Go";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_decompose_requires_min_parts() {
    // With min_decompose_parts raised to 3, a single "and" split (2 parts)
    // must no longer classify as Decompose.
    let cfg = RqRagConfig::new().with_min_decompose_parts(3);
    let refiner = MockRefiner::new(cfg);
    let q = "Tell me about Rust and tell me about C++";
    assert_ne!(refiner.classify(q), RefinementAction::Decompose);
}

// ── MockRefiner::classify — Disambiguate ──────────────────────────────────────

#[test]
fn classify_disambiguate_mercury() {
    assert_eq!(
        default_refiner().classify("Tell me about mercury"),
        RefinementAction::Disambiguate
    );
}

#[test]
fn classify_disambiguate_python() {
    assert_eq!(
        default_refiner().classify("I need information about python"),
        RefinementAction::Disambiguate
    );
}

#[test]
fn classify_disambiguate_bank() {
    let q = "What can you tell me about bank interest rates";
    assert_eq!(
        default_refiner().classify(q),
        RefinementAction::Disambiguate
    );
}

#[test]
fn classify_disambiguate_custom_marker() {
    let cfg = RqRagConfig::new().with_extra_ambiguity_marker("gizmo");
    let refiner = MockRefiner::new(cfg);
    assert_eq!(
        refiner.classify("Tell me about the gizmo"),
        RefinementAction::Disambiguate
    );
}

#[test]
fn classify_no_disambiguate_when_marker_is_substring_not_word() {
    // "banking" must not match the "bank" marker at a word boundary.
    let q = "Explain modern banking regulations";
    assert_ne!(
        default_refiner().classify(q),
        RefinementAction::Disambiguate
    );
}

// ── MockRefiner::classify — Rewrite ───────────────────────────────────────────

#[test]
fn classify_rewrite_um() {
    assert_eq!(
        default_refiner().classify("um can you tell me what Rust is"),
        RefinementAction::Rewrite
    );
}

#[test]
fn classify_rewrite_kinda() {
    assert_eq!(
        default_refiner().classify("kinda need to know what Rust does"),
        RefinementAction::Rewrite
    );
}

#[test]
fn classify_rewrite_basically() {
    assert_eq!(
        default_refiner().classify("what is Rust basically"),
        RefinementAction::Rewrite
    );
}

// ── MockRefiner::classify — priority ordering ─────────────────────────────────

#[test]
fn classify_priority_decompose_over_disambiguate() {
    // Contains both an "and" conjunction and ambiguity-table terms; Decompose
    // must win.
    let q = "Tell me about python and tell me about java";
    assert_eq!(default_refiner().classify(q), RefinementAction::Decompose);
}

#[test]
fn classify_priority_disambiguate_over_rewrite() {
    // Contains both a colloquial filler ("um") and an ambiguity marker
    // ("python"), with no compound-clause separator; Disambiguate must win.
    let q = "um tell me about python";
    assert_eq!(
        default_refiner().classify(q),
        RefinementAction::Disambiguate
    );
}

// ── MockRefiner::classify — determinism ───────────────────────────────────────

#[test]
fn classify_is_deterministic() {
    let refiner = default_refiner();
    let q = "Tell me about mercury";
    assert_eq!(refiner.classify(q), refiner.classify(q));
}

#[test]
fn classify_two_instances_agree() {
    let q = "What is Rust and what is Python?";
    assert_eq!(
        default_refiner().classify(q),
        MockRefiner::default().classify(q)
    );
}

// ── MockRefiner::rewrite ──────────────────────────────────────────────────────

#[test]
fn rewrite_removes_um_marker() {
    let result = default_refiner().rewrite("um can you tell me what Rust is");
    assert!(!words(&result).contains(&"um".to_string()));
}

#[test]
fn rewrite_removes_basically_marker() {
    let result = default_refiner().rewrite("what is Rust basically");
    assert!(!words(&result).contains(&"basically".to_string()));
}

#[test]
fn rewrite_capitalizes_first_letter() {
    let result = default_refiner().rewrite("gonna check the rust docs");
    let first = result.chars().next().expect("non-empty result");
    assert!(first.is_uppercase());
}

#[test]
fn rewrite_trims_whitespace() {
    let result = default_refiner().rewrite("   um hello there   ");
    assert_eq!(result.trim(), result);
}

#[test]
fn rewrite_empty_query_returns_empty_string() {
    assert_eq!(default_refiner().rewrite(""), "");
}

#[test]
fn rewrite_preserves_words_when_no_marker_present() {
    let result = default_refiner().rewrite("Rust is fast");
    assert_eq!(words(&result), vec!["rust", "is", "fast"]);
}

#[test]
fn rewrite_is_deterministic() {
    let refiner = default_refiner();
    let q = "um, like, what is rust basically";
    assert_eq!(refiner.rewrite(q), refiner.rewrite(q));
}

// ── MockRefiner::decompose ────────────────────────────────────────────────────

#[test]
fn decompose_splits_and() {
    let parts = default_refiner().decompose("buy milk and buy bread");
    assert_eq!(parts.len(), 2);
    assert!(parts[0].to_lowercase().contains("milk"));
    assert!(parts[1].to_lowercase().contains("bread"));
}

#[test]
fn decompose_splits_or() {
    let parts = default_refiner().decompose("Should I use Rust or should I use Go");
    assert_eq!(parts.len(), 2);
}

#[test]
fn decompose_splits_multi_question_marks() {
    let parts = default_refiner().decompose("What is Rust? What is Go?");
    assert_eq!(parts.len(), 2);
    assert!(parts[0].ends_with('?'));
    assert!(parts[1].ends_with('?'));
}

#[test]
fn decompose_fallback_single_part_when_no_separator() {
    let parts = default_refiner().decompose("Rust is great");
    assert_eq!(parts, vec!["Rust is great".to_string()]);
}

#[test]
fn decompose_empty_query_returns_empty_vec() {
    assert!(default_refiner().decompose("").is_empty());
}

#[test]
fn decompose_no_empty_segments() {
    let parts = default_refiner().decompose("Explain Rust; explain Go");
    assert!(parts.iter().all(|p| !p.trim().is_empty()));
}

// ── MockRefiner::disambiguate ──────────────────────────────────────────────────

#[test]
fn disambiguate_mercury_returns_three_senses() {
    let variants = default_refiner().disambiguate("Tell me about mercury");
    assert_eq!(variants.len(), 3);
}

#[test]
fn disambiguate_variants_contain_original_query() {
    let variants = default_refiner().disambiguate("Tell me about mercury");
    for v in &variants {
        assert!(v.contains("Tell me about mercury"));
    }
}

#[test]
fn disambiguate_variants_are_distinct() {
    let variants = default_refiner().disambiguate("Tell me about bank loans");
    let unique: HashSet<&String> = variants.iter().collect();
    assert_eq!(unique.len(), variants.len());
}

#[test]
fn disambiguate_custom_marker_returns_two_generic_variants() {
    let cfg = RqRagConfig::new().with_extra_ambiguity_marker("gizmo");
    let refiner = MockRefiner::new(cfg);
    let variants = refiner.disambiguate("I bought a gizmo");
    assert_eq!(variants.len(), 2);
    assert!(variants.iter().all(|v| v.contains("gizmo")));
}

#[test]
fn disambiguate_no_marker_returns_single_original() {
    let variants = default_refiner().disambiguate("Rust is fast");
    assert_eq!(variants, vec!["Rust is fast".to_string()]);
}

#[test]
fn disambiguate_empty_query_returns_empty_vec() {
    assert!(default_refiner().disambiguate("").is_empty());
}

// ── MockRefiner::new / Default / config accessor ─────────────────────────────

#[test]
fn mock_refiner_default_uses_default_config() {
    assert_eq!(MockRefiner::default().config(), &RqRagConfig::default());
}

#[test]
fn mock_refiner_new_stores_config() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(4);
    let refiner = MockRefiner::new(cfg.clone());
    assert_eq!(refiner.config(), &cfg);
}

#[test]
fn mock_refiner_clone_behaves_identically() {
    let refiner = default_refiner();
    let cloned = refiner.clone();
    let q = "Tell me about mercury";
    assert_eq!(refiner.classify(q), cloned.classify(q));
}

#[test]
fn mock_refiner_debug_format_nonempty() {
    let s = format!("{:?}", default_refiner());
    assert!(!s.is_empty());
}

// ── QueryRefinementEngine::run — plan shape per action ───────────────────────

#[test]
fn engine_run_respond_plan_shape() {
    let plan = default_engine().run("What is Rust?").expect("run succeeds");
    assert_eq!(plan.action, RefinementAction::Respond);
    assert_eq!(plan.refined_queries, vec!["What is Rust?".to_string()]);
    assert_eq!(plan.original_query, "What is Rust?");
}

#[test]
fn engine_run_rewrite_plan_shape() {
    let plan = default_engine()
        .run("um what is Rust")
        .expect("run succeeds");
    assert_eq!(plan.action, RefinementAction::Rewrite);
    assert_eq!(plan.refined_queries.len(), 1);
}

#[test]
fn engine_run_decompose_plan_shape() {
    let plan = default_engine()
        .run("Tell me about Rust and tell me about C++")
        .expect("run succeeds");
    assert_eq!(plan.action, RefinementAction::Decompose);
    assert!(plan.refined_queries.len() >= 2);
}

#[test]
fn engine_run_disambiguate_plan_shape() {
    let plan = default_engine()
        .run("Tell me about mercury")
        .expect("run succeeds");
    assert_eq!(plan.action, RefinementAction::Disambiguate);
    assert!(plan.refined_queries.len() >= 2);
}

#[test]
fn engine_run_refined_queries_never_empty_respond() {
    let plan = default_engine().run("Rust").expect("run succeeds");
    assert!(!plan.refined_queries.is_empty());
}

// ── QueryRefinementEngine::run — errors ───────────────────────────────────────

#[test]
fn engine_run_empty_query_error() {
    assert_eq!(default_engine().run(""), Err(RqRagError::EmptyQuery));
}

#[test]
fn engine_run_whitespace_only_error() {
    assert_eq!(default_engine().run("   "), Err(RqRagError::EmptyQuery));
}

#[test]
fn engine_run_invalid_config_error_zero() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(0);
    let engine = QueryRefinementEngine::new(cfg, MockRefiner::default());
    assert!(matches!(
        engine.run("Rust"),
        Err(RqRagError::InvalidConfig(_))
    ));
}

#[test]
fn engine_run_invalid_config_error_one() {
    let cfg = RqRagConfig::new().with_min_decompose_parts(1);
    let engine = QueryRefinementEngine::new(cfg, MockRefiner::default());
    assert!(matches!(
        engine.run("Rust"),
        Err(RqRagError::InvalidConfig(_))
    ));
}

#[test]
fn engine_run_invalid_config_takes_priority_over_empty_query() {
    // Even an empty query should surface InvalidConfig first, since config
    // validation happens before the empty-query check.
    let cfg = RqRagConfig::new().with_min_decompose_parts(0);
    let engine = QueryRefinementEngine::new(cfg, MockRefiner::default());
    assert!(matches!(engine.run(""), Err(RqRagError::InvalidConfig(_))));
}

// ── QueryRefinementEngine — construction & accessors ─────────────────────────

#[test]
fn engine_run_trims_query() {
    let plan = default_engine()
        .run("   What is Rust?   ")
        .expect("run succeeds");
    assert_eq!(plan.original_query, "What is Rust?");
}

#[test]
fn engine_with_config_builder_replaces_config() {
    let engine = default_engine().with_config(RqRagConfig::new().with_min_decompose_parts(3));
    assert_eq!(engine.config().min_decompose_parts, 3);
}

#[test]
fn engine_config_accessor_returns_current_config() {
    assert_eq!(default_engine().config().min_decompose_parts, 2);
}

#[test]
fn engine_refiner_accessor_returns_refiner() {
    let engine = default_engine();
    assert_eq!(engine.refiner().config(), &RqRagConfig::default());
}

#[test]
fn engine_run_is_deterministic() {
    let engine = default_engine();
    let q = "Tell me about mercury";
    assert_eq!(engine.run(q), engine.run(q));
}

#[test]
fn engine_run_original_query_preserved_for_decompose() {
    let q = "Tell me about Rust and tell me about C++";
    let plan = default_engine().run(q).expect("run succeeds");
    assert_eq!(plan.original_query, q);
}

// ── Trait-object usability ─────────────────────────────────────────────────────

#[test]
fn mock_refiner_usable_as_trait_object() {
    let boxed: Box<dyn QueryRefiner> = Box::new(MockRefiner::default());
    assert_eq!(boxed.classify("What is Rust?"), RefinementAction::Respond);
    assert_eq!(
        boxed.decompose("Rust is great"),
        vec!["Rust is great".to_string()]
    );
    assert_eq!(
        boxed.disambiguate("Rust is great"),
        vec!["Rust is great".to_string()]
    );
    assert_eq!(boxed.rewrite("Rust is great"), "Rust is great".to_string());
}

#[test]
fn trait_object_classify_decompose() {
    let boxed: Box<dyn QueryRefiner> = Box::new(MockRefiner::default());
    let action = boxed.classify("Tell me about Rust and tell me about C++");
    assert_eq!(action, RefinementAction::Decompose);
}

/// A minimal custom [`QueryRefiner`] implementation used to prove that
/// [`QueryRefinementEngine`] is generic over any conforming strategy, not
/// just [`MockRefiner`].
struct AlwaysRespond;

impl QueryRefiner for AlwaysRespond {
    fn classify(&self, _query: &str) -> RefinementAction {
        RefinementAction::Respond
    }

    fn rewrite(&self, query: &str) -> String {
        query.to_string()
    }

    fn decompose(&self, query: &str) -> Vec<String> {
        vec![query.to_string()]
    }

    fn disambiguate(&self, query: &str) -> Vec<String> {
        vec![query.to_string()]
    }
}

#[test]
fn engine_generic_over_custom_refiner() {
    let engine = QueryRefinementEngine::new(RqRagConfig::default(), AlwaysRespond);
    // Even though the query contains "and", the custom refiner always
    // classifies as Respond — proving the engine defers entirely to the
    // refiner's own classification rather than re-deriving the action.
    let plan = engine.run("anything and everything").expect("run succeeds");
    assert_eq!(plan.action, RefinementAction::Respond);
    assert_eq!(
        plan.refined_queries,
        vec!["anything and everything".to_string()]
    );
}

#[test]
fn custom_refiner_usable_as_trait_object() {
    let boxed: Box<dyn QueryRefiner> = Box::new(AlwaysRespond);
    assert_eq!(boxed.classify("anything"), RefinementAction::Respond);
}
