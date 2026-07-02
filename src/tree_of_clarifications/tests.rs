#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::redundant_clone,
    clippy::many_single_char_names
)]
//! Tests for the `tree_of_clarifications` module.

use super::engine::{ToCEngine, aggregate};
use super::tree::{ClarificationNode, ClarificationTree};
use super::types::{
    ClarificationAnswerer, ClarificationRetriever, Disambiguator, MockClarificationAnswerer,
    MockClarificationRetriever, MockDisambiguator, ToCConfig, ToCError,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build a [`ClarificationNode`] with explicit fields for direct tree tests.
fn node(
    question: &str,
    relevance_score: f32,
    answer: &str,
    children: Vec<ClarificationNode>,
) -> ClarificationNode {
    ClarificationNode {
        question: question.to_string(),
        relevance_score,
        answer: Some(answer.to_string()),
        children,
    }
}

fn default_engine()
-> ToCEngine<MockDisambiguator, MockClarificationRetriever, MockClarificationAnswerer> {
    // `prune_threshold` is relaxed to `0.0` because the paired
    // `MockClarificationRetriever::empty()` never returns supporting passages,
    // which would otherwise make every node's `relevance_score` `0.0` and get
    // pruned away — these tests exercise tree *construction*, not pruning.
    ToCEngine::new(
        ToCConfig::default().with_prune_threshold(0.0),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    )
}

// ── ToCConfig: defaults & builders ────────────────────────────────────────────

#[test]
fn config_default_max_depth_is_two() {
    assert_eq!(ToCConfig::default().max_depth, 2);
}

#[test]
fn config_default_top_k_is_three() {
    assert_eq!(ToCConfig::default().top_k, 3);
}

#[test]
fn config_default_prune_threshold_is_point_two() {
    assert!((ToCConfig::default().prune_threshold - 0.2).abs() < 1e-6);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(ToCConfig::new(), ToCConfig::default());
}

#[test]
fn config_with_max_depth_builder() {
    let cfg = ToCConfig::new().with_max_depth(5);
    assert_eq!(cfg.max_depth, 5);
}

#[test]
fn config_with_top_k_builder() {
    let cfg = ToCConfig::new().with_top_k(10);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn config_with_prune_threshold_builder() {
    let cfg = ToCConfig::new().with_prune_threshold(0.75);
    assert!((cfg.prune_threshold - 0.75).abs() < 1e-6);
}

#[test]
fn config_builders_chain_independently() {
    let cfg = ToCConfig::new()
        .with_max_depth(4)
        .with_top_k(7)
        .with_prune_threshold(0.5);
    assert_eq!(cfg.max_depth, 4);
    assert_eq!(cfg.top_k, 7);
    assert!((cfg.prune_threshold - 0.5).abs() < 1e-6);
}

// ── ToCConfig: validate ────────────────────────────────────────────────────────

#[test]
fn config_validate_default_ok() {
    assert!(ToCConfig::default().validate().is_ok());
}

#[test]
fn config_validate_zero_max_depth_is_invalid() {
    let cfg = ToCConfig::new().with_max_depth(0);
    assert!(matches!(cfg.validate(), Err(ToCError::InvalidConfig(_))));
}

#[test]
fn config_validate_zero_top_k_is_invalid() {
    let cfg = ToCConfig::new().with_top_k(0);
    assert!(matches!(cfg.validate(), Err(ToCError::InvalidConfig(_))));
}

#[test]
fn config_validate_negative_prune_threshold_is_invalid() {
    let cfg = ToCConfig::new().with_prune_threshold(-0.1);
    assert!(matches!(cfg.validate(), Err(ToCError::InvalidConfig(_))));
}

#[test]
fn config_validate_prune_threshold_above_one_is_invalid() {
    let cfg = ToCConfig::new().with_prune_threshold(1.1);
    assert!(matches!(cfg.validate(), Err(ToCError::InvalidConfig(_))));
}

#[test]
fn config_validate_nan_prune_threshold_is_invalid() {
    let cfg = ToCConfig::new().with_prune_threshold(f32::NAN);
    assert!(matches!(cfg.validate(), Err(ToCError::InvalidConfig(_))));
}

#[test]
fn config_validate_boundary_zero_prune_threshold_ok() {
    let cfg = ToCConfig::new().with_prune_threshold(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validate_boundary_one_prune_threshold_ok() {
    let cfg = ToCConfig::new().with_prune_threshold(1.0);
    assert!(cfg.validate().is_ok());
}

// ── ToCError ───────────────────────────────────────────────────────────────────

#[test]
fn error_empty_question_display() {
    assert_eq!(
        ToCError::EmptyQuestion.to_string(),
        "question must not be empty"
    );
}

#[test]
fn error_invalid_config_display_contains_message() {
    let err = ToCError::InvalidConfig("top_k must be at least 1".to_string());
    assert!(err.to_string().contains("top_k must be at least 1"));
}

#[test]
fn error_equality() {
    assert_eq!(ToCError::EmptyQuestion, ToCError::EmptyQuestion);
    assert_ne!(
        ToCError::EmptyQuestion,
        ToCError::InvalidConfig("x".to_string())
    );
}

#[test]
fn error_clone() {
    let err = ToCError::InvalidConfig("bad".to_string());
    assert_eq!(err.clone(), err);
}

// ── MockDisambiguator ──────────────────────────────────────────────────────────

#[test]
fn mock_disambiguator_unambiguous_never_matches() {
    let d = MockDisambiguator::unambiguous();
    assert!(d.disambiguate("What is the capital?").is_empty());
}

#[test]
fn mock_disambiguator_matches_substring() {
    let d = MockDisambiguator::new(vec![(
        "capital".to_string(),
        vec![
            "capital of France".to_string(),
            "capital of a company".to_string(),
        ],
    )]);
    let candidates = d.disambiguate("What is the capital?");
    assert_eq!(candidates.len(), 2);
    assert!(candidates.contains(&"capital of France".to_string()));
}

#[test]
fn mock_disambiguator_case_insensitive_match() {
    let d = MockDisambiguator::new(vec![("CAPITAL".to_string(), vec!["reading".to_string()])]);
    let candidates = d.disambiguate("what is the CAPITAL?");
    assert_eq!(candidates, vec!["reading".to_string()]);
}

#[test]
fn mock_disambiguator_no_match_returns_empty() {
    let d = MockDisambiguator::new(vec![("capital".to_string(), vec!["reading".to_string()])]);
    assert!(d.disambiguate("What is Rust?").is_empty());
}

#[test]
fn mock_disambiguator_empty_needle_never_matches() {
    let d = MockDisambiguator::new(vec![(String::new(), vec!["reading".to_string()])]);
    assert!(d.disambiguate("anything at all").is_empty());
}

#[test]
fn mock_disambiguator_first_match_wins() {
    let d = MockDisambiguator::new(vec![
        ("bank".to_string(), vec!["river bank".to_string()]),
        ("bank".to_string(), vec!["money bank".to_string()]),
    ]);
    assert_eq!(d.disambiguate("bank"), vec!["river bank".to_string()]);
}

#[test]
fn mock_disambiguator_default_is_unambiguous() {
    assert!(
        MockDisambiguator::default()
            .disambiguate("anything")
            .is_empty()
    );
}

// ── MockClarificationRetriever ─────────────────────────────────────────────────

#[test]
fn mock_retriever_empty_never_matches() {
    let r = MockClarificationRetriever::empty();
    assert!(r.retrieve("query", 3).is_empty());
}

#[test]
fn mock_retriever_matches_substring() {
    let r = MockClarificationRetriever::new(vec![(
        "france".to_string(),
        vec!["Paris is the capital.".to_string()],
    )]);
    let passages = r.retrieve("capital of France", 3);
    assert_eq!(passages, vec!["Paris is the capital.".to_string()]);
}

#[test]
fn mock_retriever_truncates_to_top_k() {
    let r = MockClarificationRetriever::new(vec![(
        "france".to_string(),
        vec!["p1".to_string(), "p2".to_string(), "p3".to_string()],
    )]);
    let passages = r.retrieve("about France", 2);
    assert_eq!(passages, vec!["p1".to_string(), "p2".to_string()]);
}

#[test]
fn mock_retriever_top_k_larger_than_corpus() {
    let r = MockClarificationRetriever::new(vec![("france".to_string(), vec!["p1".to_string()])]);
    let passages = r.retrieve("about France", 10);
    assert_eq!(passages.len(), 1);
}

#[test]
fn mock_retriever_case_insensitive() {
    let r = MockClarificationRetriever::new(vec![("FRANCE".to_string(), vec!["p1".to_string()])]);
    let passages = r.retrieve("france is nice", 3);
    assert_eq!(passages, vec!["p1".to_string()]);
}

#[test]
fn mock_retriever_no_match_returns_empty() {
    let r = MockClarificationRetriever::new(vec![("france".to_string(), vec!["p1".to_string()])]);
    assert!(r.retrieve("something else entirely", 3).is_empty());
}

#[test]
fn mock_retriever_empty_needle_never_matches() {
    let r = MockClarificationRetriever::new(vec![(String::new(), vec!["p1".to_string()])]);
    assert!(r.retrieve("anything", 3).is_empty());
}

// ── MockClarificationAnswerer ──────────────────────────────────────────────────

#[test]
fn mock_answerer_with_passages() {
    let a = MockClarificationAnswerer;
    let passages = vec!["Paris is the capital.".to_string()];
    let answer = a.answer("What is the capital of France?", &passages);
    assert!(answer.contains("What is the capital of France?"));
    assert!(answer.contains("Paris is the capital."));
}

#[test]
fn mock_answerer_without_passages() {
    let a = MockClarificationAnswerer;
    let answer = a.answer("What is the capital?", &[]);
    assert!(answer.contains("No supporting evidence"));
    assert!(answer.contains("What is the capital?"));
}

#[test]
fn mock_answerer_deterministic() {
    let a = MockClarificationAnswerer;
    let passages = vec!["evidence".to_string()];
    assert_eq!(a.answer("q", &passages), a.answer("q", &passages),);
}

#[test]
fn mock_answerer_joins_multiple_passages() {
    let a = MockClarificationAnswerer;
    let passages = vec!["p1".to_string(), "p2".to_string()];
    let answer = a.answer("q", &passages);
    assert!(answer.contains("p1"));
    assert!(answer.contains("p2"));
}

// ── ClarificationNode ──────────────────────────────────────────────────────────

#[test]
fn clarification_node_new_has_no_children() {
    let n = ClarificationNode::new("q", 0.5, "a");
    assert!(n.children.is_empty());
    assert_eq!(n.answer, Some("a".to_string()));
    assert!((n.relevance_score - 0.5).abs() < 1e-6);
}

#[test]
fn clarification_node_new_is_leaf() {
    let n = ClarificationNode::new("q", 0.5, "a");
    assert!(n.is_leaf());
}

#[test]
fn clarification_node_with_children_is_not_leaf() {
    let child = ClarificationNode::new("child", 0.5, "a");
    let n = node("parent", 0.9, "pa", vec![child]);
    assert!(!n.is_leaf());
}

#[test]
fn clarification_node_count_single() {
    let n = ClarificationNode::new("q", 0.5, "a");
    assert_eq!(n.node_count(), 1);
}

#[test]
fn clarification_node_count_with_children() {
    let c1 = ClarificationNode::new("c1", 0.5, "a1");
    let c2 = ClarificationNode::new("c2", 0.5, "a2");
    let n = node("root", 0.9, "ra", vec![c1, c2]);
    assert_eq!(n.node_count(), 3);
}

#[test]
fn clarification_node_count_nested() {
    let gc = ClarificationNode::new("gc", 0.5, "ga");
    let c = node("c", 0.6, "ca", vec![gc]);
    let root = node("root", 0.9, "ra", vec![c]);
    assert_eq!(root.node_count(), 3);
}

// ── ClarificationTree: basic structure ─────────────────────────────────────────

#[test]
fn tree_default_is_empty() {
    let tree = ClarificationTree::default();
    assert!(tree.is_empty());
    assert_eq!(tree.node_count(), 0);
}

#[test]
fn tree_new_equals_default() {
    assert_eq!(ClarificationTree::new(), ClarificationTree::default());
}

#[test]
fn tree_with_one_root_not_empty() {
    let tree = ClarificationTree {
        root_questions: vec![ClarificationNode::new("q", 0.5, "a")],
    };
    assert!(!tree.is_empty());
}

#[test]
fn tree_node_count_sums_across_roots() {
    let tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("r1", 0.5, "a1"),
            ClarificationNode::new("r2", 0.5, "a2"),
        ],
    };
    assert_eq!(tree.node_count(), 2);
}

// ── ClarificationTree::leaves ──────────────────────────────────────────────────

#[test]
fn tree_leaves_empty_tree() {
    let tree = ClarificationTree::default();
    assert!(tree.leaves().is_empty());
}

#[test]
fn tree_leaves_single_leaf_root() {
    let tree = ClarificationTree {
        root_questions: vec![ClarificationNode::new("q", 0.5, "a")],
    };
    let leaves = tree.leaves();
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].question, "q");
}

#[test]
fn tree_leaves_multiple_root_leaves() {
    let tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("r1", 0.5, "a1"),
            ClarificationNode::new("r2", 0.6, "a2"),
        ],
    };
    assert_eq!(tree.leaves().len(), 2);
}

#[test]
fn tree_leaves_only_returns_leaf_nodes_not_internal() {
    let child1 = ClarificationNode::new("c1", 0.5, "a1");
    let child2 = ClarificationNode::new("c2", 0.5, "a2");
    let root = node("root", 0.9, "ra", vec![child1, child2]);
    let tree = ClarificationTree {
        root_questions: vec![root],
    };
    let leaves = tree.leaves();
    assert_eq!(leaves.len(), 2);
    assert!(leaves.iter().all(|l| l.question != "root"));
}

#[test]
fn tree_leaves_deep_tree() {
    let gc1 = ClarificationNode::new("gc1", 0.5, "ga1");
    let gc2 = ClarificationNode::new("gc2", 0.5, "ga2");
    let c = node("c", 0.6, "ca", vec![gc1, gc2]);
    let root = node("root", 0.9, "ra", vec![c]);
    let tree = ClarificationTree {
        root_questions: vec![root],
    };
    let leaves = tree.leaves();
    assert_eq!(leaves.len(), 2);
    assert!(leaves.iter().any(|l| l.question == "gc1"));
    assert!(leaves.iter().any(|l| l.question == "gc2"));
}

// ── ClarificationTree::prune ────────────────────────────────────────────────────

#[test]
fn prune_keeps_all_when_threshold_is_zero() {
    let mut tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("r1", 0.0, "a1"),
            ClarificationNode::new("r2", 0.1, "a2"),
        ],
    };
    tree.prune(0.0);
    assert_eq!(tree.root_questions.len(), 2);
}

#[test]
fn prune_removes_low_relevance_root() {
    let mut tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("low", 0.05, "a1"),
            ClarificationNode::new("high", 0.9, "a2"),
        ],
    };
    tree.prune(0.2);
    assert_eq!(tree.root_questions.len(), 1);
    assert_eq!(tree.root_questions[0].question, "high");
}

#[test]
fn prune_removes_low_relevance_child_keeps_sibling() {
    let low_child = ClarificationNode::new("low_child", 0.05, "la");
    let high_child = ClarificationNode::new("high_child", 0.8, "ha");
    let root = node("root", 0.9, "ra", vec![low_child, high_child]);
    let mut tree = ClarificationTree {
        root_questions: vec![root],
    };
    tree.prune(0.2);
    assert_eq!(tree.root_questions[0].children.len(), 1);
    assert_eq!(tree.root_questions[0].children[0].question, "high_child");
}

#[test]
fn prune_all_children_pruned_becomes_leaf_with_own_answer() {
    let low_child_a = ClarificationNode::new("child_a", 0.01, "child a answer");
    let low_child_b = ClarificationNode::new("child_b", 0.02, "child b answer");
    let root = node(
        "root question",
        0.9,
        "root's own best-effort answer",
        vec![low_child_a, low_child_b],
    );
    let mut tree = ClarificationTree {
        root_questions: vec![root],
    };
    tree.prune(0.2);

    assert_eq!(tree.root_questions.len(), 1);
    let survivor = &tree.root_questions[0];
    assert!(survivor.is_leaf());
    assert_eq!(survivor.question, "root question");
    assert_eq!(
        survivor.answer.as_deref(),
        Some("root's own best-effort answer")
    );
}

#[test]
fn prune_nested_fallback_grandchildren_pruned_child_becomes_leaf() {
    let gc1 = ClarificationNode::new("gc1", 0.0, "ga1");
    let gc2 = ClarificationNode::new("gc2", 0.0, "ga2");
    let c = node("c", 0.7, "c own answer", vec![gc1, gc2]);
    let root = node("root", 0.9, "root own answer", vec![c]);
    let mut tree = ClarificationTree {
        root_questions: vec![root],
    };
    tree.prune(0.2);

    assert_eq!(tree.root_questions[0].children.len(), 1);
    let survivor_child = &tree.root_questions[0].children[0];
    assert!(survivor_child.is_leaf());
    assert_eq!(survivor_child.answer.as_deref(), Some("c own answer"));
}

#[test]
fn prune_high_threshold_removes_everything_below_one() {
    let mut tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("r1", 0.99, "a1"),
            ClarificationNode::new("r2", 1.0, "a2"),
        ],
    };
    tree.prune(1.0);
    assert_eq!(tree.root_questions.len(), 1);
    assert_eq!(tree.root_questions[0].question, "r2");
}

#[test]
fn prune_boundary_equal_to_threshold_survives() {
    let mut tree = ClarificationTree {
        root_questions: vec![ClarificationNode::new("r1", 0.2, "a1")],
    };
    tree.prune(0.2);
    assert_eq!(tree.root_questions.len(), 1);
}

#[test]
fn prune_empty_tree_stays_empty() {
    let mut tree = ClarificationTree::default();
    tree.prune(0.2);
    assert!(tree.is_empty());
}

// ── aggregate ──────────────────────────────────────────────────────────────────

#[test]
fn aggregate_empty_tree_returns_fallback_message() {
    let tree = ClarificationTree::default();
    let answer = aggregate(&tree);
    assert!(answer.contains("No disambiguated reading survived pruning"));
}

#[test]
fn aggregate_single_leaf() {
    let tree = ClarificationTree {
        root_questions: vec![ClarificationNode::new("q1", 0.5, "answer1")],
    };
    let answer = aggregate(&tree);
    assert_eq!(answer, "Regarding q1: answer1");
}

#[test]
fn aggregate_multiple_leaves_each_attributed() {
    let tree = ClarificationTree {
        root_questions: vec![
            ClarificationNode::new("q1", 0.5, "answer1"),
            ClarificationNode::new("q2", 0.5, "answer2"),
        ],
    };
    let answer = aggregate(&tree);
    assert!(answer.contains("Regarding q1: answer1"));
    assert!(answer.contains("Regarding q2: answer2"));
    assert_eq!(answer.lines().count(), 2);
}

#[test]
fn aggregate_missing_answer_uses_placeholder() {
    let node_without_answer = ClarificationNode {
        question: "q".to_string(),
        relevance_score: 0.5,
        answer: None,
        children: Vec::new(),
    };
    let tree = ClarificationTree {
        root_questions: vec![node_without_answer],
    };
    let answer = aggregate(&tree);
    assert!(answer.contains("No answer was produced"));
}

// ── ToCEngine: input validation ────────────────────────────────────────────────

#[test]
fn engine_run_empty_question_errors() {
    let engine = default_engine();
    assert_eq!(engine.run(""), Err(ToCError::EmptyQuestion));
}

#[test]
fn engine_run_whitespace_question_errors() {
    let engine = default_engine();
    assert_eq!(engine.run("   \t"), Err(ToCError::EmptyQuestion));
}

#[test]
fn engine_run_invalid_config_max_depth_zero_errors() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(0),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    assert!(matches!(
        engine.run("question"),
        Err(ToCError::InvalidConfig(_))
    ));
}

#[test]
fn engine_run_invalid_config_top_k_zero_errors() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_top_k(0),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    assert!(matches!(
        engine.run("question"),
        Err(ToCError::InvalidConfig(_))
    ));
}

#[test]
fn engine_run_invalid_config_bad_prune_threshold_errors() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_prune_threshold(2.0),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    assert!(matches!(
        engine.run("question"),
        Err(ToCError::InvalidConfig(_))
    ));
}

// ── ToCEngine: non-ambiguous single-leaf case ─────────────────────────────────

#[test]
fn engine_run_unambiguous_question_produces_single_root_equals_leaf() {
    let engine = default_engine();
    let (tree, _answer) = engine.run("What is Rust?").unwrap();
    assert_eq!(tree.root_questions.len(), 1);
    assert_eq!(tree.root_questions[0].question, "What is Rust?");
    assert!(tree.root_questions[0].is_leaf());
}

#[test]
fn engine_run_unambiguous_question_answer_mentions_original_question() {
    let engine = default_engine();
    let (_tree, answer) = engine.run("What is Rust?").unwrap();
    assert!(answer.contains("What is Rust?"));
}

#[test]
fn engine_run_unambiguous_question_trims_whitespace() {
    let engine = default_engine();
    let (tree, _answer) = engine.run("  What is Rust?  ").unwrap();
    assert_eq!(tree.root_questions[0].question, "What is Rust?");
}

// ── ToCEngine: ambiguous multi-branch case ────────────────────────────────────

fn ambiguous_engine(
    max_depth: usize,
) -> ToCEngine<MockDisambiguator, MockClarificationRetriever, MockClarificationAnswerer> {
    let disambiguator = MockDisambiguator::new(vec![(
        "capital".to_string(),
        vec![
            "capital of France".to_string(),
            "capital of a company".to_string(),
        ],
    )]);
    let retriever = MockClarificationRetriever::new(vec![
        (
            "capital of france".to_string(),
            vec!["Paris is the capital of France city.".to_string()],
        ),
        (
            "capital of a company".to_string(),
            vec!["The capital of a company is called shareholder capital funds.".to_string()],
        ),
    ]);
    ToCEngine::new(
        ToCConfig::new().with_max_depth(max_depth),
        disambiguator,
        retriever,
        MockClarificationAnswerer,
    )
}

#[test]
fn engine_run_ambiguous_question_produces_two_roots() {
    let engine = ambiguous_engine(1);
    let (tree, _answer) = engine.run("What is the capital?").unwrap();
    assert_eq!(tree.root_questions.len(), 2);
}

#[test]
fn engine_run_ambiguous_question_roots_are_leaves_at_max_depth_one() {
    let engine = ambiguous_engine(1);
    let (tree, _answer) = engine.run("What is the capital?").unwrap();
    assert!(tree.root_questions.iter().all(ClarificationNode::is_leaf));
}

#[test]
fn engine_run_ambiguous_question_aggregation_covers_both_leaves() {
    let engine = ambiguous_engine(1);
    let (_tree, answer) = engine.run("What is the capital?").unwrap();
    assert!(answer.contains("Regarding capital of France"));
    assert!(answer.contains("Regarding capital of a company"));
}

#[test]
fn engine_run_ambiguous_question_answers_include_retrieved_passage_content() {
    let engine = ambiguous_engine(1);
    let (_tree, answer) = engine.run("What is the capital?").unwrap();
    assert!(answer.contains("Paris is the capital of France city."));
}

// ── ToCEngine: max_depth bounds recursion ─────────────────────────────────────

fn recursive_disambiguator() -> MockDisambiguator {
    // Every reading containing "topic" is further split into two sub-readings,
    // each of which still contains "topic" and could be split again.
    MockDisambiguator::new(vec![(
        "topic".to_string(),
        vec![
            "topic branch one".to_string(),
            "topic branch two".to_string(),
        ],
    )])
}

#[test]
fn engine_run_max_depth_one_produces_shallow_tree() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(1).with_prune_threshold(0.0),
        recursive_disambiguator(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    let (tree, _answer) = engine.run("topic").unwrap();
    // depth-1 roots exist but cannot recurse further.
    assert_eq!(tree.root_questions.len(), 2);
    assert!(tree.root_questions.iter().all(ClarificationNode::is_leaf));
    assert_eq!(tree.node_count(), 2);
}

#[test]
fn engine_run_max_depth_two_produces_deeper_tree() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(2).with_prune_threshold(0.0),
        recursive_disambiguator(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    let (tree, _answer) = engine.run("topic").unwrap();
    // Each of the 2 depth-1 roots gets 2 depth-2 children: 2 + 2*2 = 6 nodes.
    assert_eq!(tree.node_count(), 6);
    assert!(tree.root_questions.iter().all(|r| !r.is_leaf()));
}

#[test]
fn engine_run_max_depth_bounds_are_monotonic() {
    let shallow = ToCEngine::new(
        ToCConfig::new().with_max_depth(1).with_prune_threshold(0.0),
        recursive_disambiguator(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    let deep = ToCEngine::new(
        ToCConfig::new().with_max_depth(3).with_prune_threshold(0.0),
        recursive_disambiguator(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    let (shallow_tree, _) = shallow.run("topic").unwrap();
    let (deep_tree, _) = deep.run("topic").unwrap();
    assert!(deep_tree.node_count() > shallow_tree.node_count());
}

// ── ToCEngine: pruning integration through run() ──────────────────────────────

#[test]
fn engine_run_prunes_branch_with_no_supporting_passages() {
    let disambiguator = MockDisambiguator::new(vec![(
        "capital".to_string(),
        vec![
            "capital of France".to_string(),
            "capital of nowhere".to_string(),
        ],
    )]);
    // Only "capital of france" has a retriever match; "capital of nowhere"
    // gets no passages at all, so its relevance_score is 0.0.
    let retriever = MockClarificationRetriever::new(vec![(
        "capital of france".to_string(),
        vec!["Paris is the capital of France city.".to_string()],
    )]);
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(1),
        disambiguator,
        retriever,
        MockClarificationAnswerer,
    );
    let (tree, answer) = engine.run("What is the capital?").unwrap();

    assert_eq!(tree.root_questions.len(), 1);
    assert_eq!(tree.root_questions[0].question, "capital of France");
    assert!(answer.contains("Regarding capital of France"));
    assert!(!answer.contains("capital of nowhere"));
}

#[test]
fn engine_run_zero_prune_threshold_keeps_zero_relevance_branch() {
    let disambiguator = MockDisambiguator::new(vec![(
        "capital".to_string(),
        vec!["capital of nowhere".to_string()],
    )]);
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(1).with_prune_threshold(0.0),
        disambiguator,
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    let (tree, _answer) = engine.run("What is the capital?").unwrap();
    assert_eq!(tree.root_questions.len(), 1);
}

#[test]
fn engine_run_all_children_pruned_falls_back_to_own_answer() {
    // Top level: one candidate reading ("reading X").
    // "reading X" is itself further disambiguated into two weakly supported
    // sub-readings, both of which get pruned, so "reading X" reverts to being
    // a leaf using its own (well supported) answer.
    let disambiguator = MockDisambiguator::new(vec![
        ("ambiguous root".to_string(), vec!["reading X".to_string()]),
        (
            "reading x".to_string(),
            vec!["reading X sub A".to_string(), "reading X sub B".to_string()],
        ),
    ]);
    let retriever = MockClarificationRetriever::new(vec![
        ("sub a".to_string(), vec!["zzz zzz zzz".to_string()]),
        ("sub b".to_string(), vec!["zzz zzz zzz".to_string()]),
        (
            "reading x".to_string(),
            vec!["reading X content here".to_string()],
        ),
    ]);
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(2),
        disambiguator,
        retriever,
        MockClarificationAnswerer,
    );
    let (tree, answer) = engine.run("ambiguous root").unwrap();

    assert_eq!(tree.root_questions.len(), 1);
    let survivor = &tree.root_questions[0];
    assert_eq!(survivor.question, "reading X");
    assert!(survivor.is_leaf());
    assert!(answer.contains("Regarding reading X"));
    assert!(answer.contains("reading X content here"));
    assert!(!answer.contains("sub A"));
    assert!(!answer.contains("sub B"));
}

// ── ToCEngine: determinism ─────────────────────────────────────────────────────

#[test]
fn engine_run_is_deterministic() {
    let engine = ambiguous_engine(2);
    let (tree1, answer1) = engine.run("What is the capital?").unwrap();
    let (tree2, answer2) = engine.run("What is the capital?").unwrap();
    assert_eq!(tree1, tree2);
    assert_eq!(answer1, answer2);
}

#[test]
fn engine_clone_produces_identical_results() {
    let engine = ambiguous_engine(1);
    let cloned = engine.clone();
    let (tree1, answer1) = engine.run("What is the capital?").unwrap();
    let (tree2, answer2) = cloned.run("What is the capital?").unwrap();
    assert_eq!(tree1, tree2);
    assert_eq!(answer1, answer2);
}

// ── ToCEngine: config field accessible ─────────────────────────────────────────

#[test]
fn engine_new_stores_config() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(9),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    assert_eq!(engine.config.max_depth, 9);
}

#[test]
fn engine_run_high_prune_threshold_can_empty_the_tree() {
    let engine = ToCEngine::new(
        ToCConfig::new().with_max_depth(1).with_prune_threshold(1.0),
        MockDisambiguator::unambiguous(),
        MockClarificationRetriever::empty(),
        MockClarificationAnswerer,
    );
    // MockClarificationRetriever::empty() never returns passages, so
    // relevance_score is always 0.0, which is below a threshold of 1.0.
    let (tree, answer) = engine.run("What is Rust?").unwrap();
    assert!(tree.is_empty());
    assert!(answer.contains("No disambiguated reading survived pruning"));
}
