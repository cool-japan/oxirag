#![allow(clippy::too_many_lines)]

use crate::searchain::engine::SearChainEngine;
use crate::searchain::types::{
    AnswerAssemblyStrategy, ChainGenerator, ChainNode, Generator, MockChainGenerator,
    MockSearchainGenerator, MockSearchainRetriever, NodeVerdict, Retriever, SearChainConfig,
    SearChainError, SearChainStatus,
};

// ── Test helpers ─────────────────────────────────────────────────────────────

/// A [`ChainGenerator`] that ignores the query and always returns a fixed,
/// caller-supplied chain — used to exercise `build_chain` validation paths
/// that `MockChainGenerator`'s substring-matching fallback cannot reach
/// (e.g. an empty chain, or malformed ids/dependencies).
struct FixedChainGenerator(Vec<ChainNode>);

impl ChainGenerator for FixedChainGenerator {
    fn generate_chain(&self, _query: &str) -> Vec<ChainNode> {
        self.0.clone()
    }
}

fn default_engine() -> SearChainEngine {
    SearChainEngine::new(SearChainConfig::default())
}

// ── NodeVerdict ───────────────────────────────────────────────────────────────

#[test]
fn node_verdict_as_str() {
    assert_eq!(NodeVerdict::Verified.as_str(), "verified");
    assert_eq!(NodeVerdict::Unverified.as_str(), "unverified");
    assert_eq!(NodeVerdict::Conflicting.as_str(), "conflicting");
}

// ── ChainNode ─────────────────────────────────────────────────────────────────

#[test]
fn chain_node_new_defaults() {
    let node = ChainNode::new(0, "sub-query", "tentative");
    assert_eq!(node.id, 0);
    assert_eq!(node.sub_query, "sub-query");
    assert_eq!(node.tentative_answer, "tentative");
    assert_eq!(node.final_answer, "tentative");
    assert!(node.depends_on.is_empty());
    assert!(node.evidence.is_empty());
    assert_eq!(node.verdict, None);
    assert!(!node.revised);
}

#[test]
fn chain_node_with_depends_on() {
    let node = ChainNode::new(2, "q", "a").with_depends_on(vec![0, 1]);
    assert_eq!(node.depends_on, vec![0, 1]);
    assert!(!node.is_root());
}

#[test]
fn chain_node_is_root_true_without_dependencies() {
    let node = ChainNode::new(0, "q", "a");
    assert!(node.is_root());
}

#[test]
fn chain_node_is_conflicting_and_is_terminal() {
    let mut node = ChainNode::new(0, "q", "a");
    assert!(!node.is_conflicting());
    assert!(!node.is_terminal());

    node.verdict = Some(NodeVerdict::Conflicting);
    assert!(node.is_conflicting());
    assert!(!node.is_terminal());

    node.verdict = Some(NodeVerdict::Verified);
    assert!(!node.is_conflicting());
    assert!(node.is_terminal());

    node.verdict = Some(NodeVerdict::Unverified);
    assert!(!node.is_conflicting());
    assert!(node.is_terminal());
}

// ── MockChainGenerator ────────────────────────────────────────────────────────

#[test]
fn mock_chain_generator_matches_substring() {
    let chain = vec![ChainNode::new(0, "capital of France", "Paris")];
    let generator = MockChainGenerator::new(vec![("France".to_string(), chain.clone())]);
    assert_eq!(
        generator.generate_chain("What is the capital of France?"),
        chain
    );
}

#[test]
fn mock_chain_generator_fallback_echoes_query() {
    let generator = MockChainGenerator::default();
    let chain = generator.generate_chain("unmapped query");
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].id, 0);
    assert_eq!(chain[0].sub_query, "unmapped query");
    assert_eq!(chain[0].tentative_answer, "unmapped query");
}

// ── MockSearchainRetriever ───────────────────────────────────────────────────

#[test]
fn mock_retriever_matches_substring() {
    let retriever = MockSearchainRetriever::new(vec![(
        "France".to_string(),
        vec!["Paris is the capital of France.".to_string()],
    )]);
    assert_eq!(
        retriever.retrieve("capital of France"),
        vec!["Paris is the capital of France.".to_string()]
    );
}

#[test]
fn mock_retriever_no_match_returns_empty() {
    let retriever = MockSearchainRetriever::default();
    assert!(retriever.retrieve("anything").is_empty());
}

// ── MockSearchainGenerator ───────────────────────────────────────────────────

#[test]
fn mock_generator_last_node_strategy() {
    let chain = vec![
        ChainNode::new(0, "q0", "a0"),
        ChainNode::new(1, "q1", "a1").with_depends_on(vec![0]),
    ];
    let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::LastNode);
    assert_eq!(generator.generate("orig", &chain), "a1");
}

#[test]
fn mock_generator_join_all_strategy() {
    let chain = vec![
        ChainNode::new(0, "q0", "a0"),
        ChainNode::new(1, "q1", "a1").with_depends_on(vec![0]),
    ];
    let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::JoinAll);
    assert_eq!(generator.generate("orig", &chain), "a0; a1");
}

#[test]
fn mock_generator_last_node_on_empty_chain_is_empty_string() {
    let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::LastNode);
    assert_eq!(generator.generate("orig", &[]), "");
}

// ── SearChainConfig ───────────────────────────────────────────────────────────

#[test]
fn config_defaults() {
    let config = SearChainConfig::default();
    assert_eq!(config.min_shared_terms, 2);
    assert!((config.min_support_overlap - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.max_backtrack_iterations, 8);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SearChainConfig::new(), SearChainConfig::default());
}

#[test]
fn config_builders() {
    let config = SearChainConfig::new()
        .with_min_shared_terms(3)
        .with_min_support_overlap(0.75)
        .with_max_backtrack_iterations(2);
    assert_eq!(config.min_shared_terms, 3);
    assert!((config.min_support_overlap - 0.75).abs() < f32::EPSILON);
    assert_eq!(config.max_backtrack_iterations, 2);
}

// ── SearChainStatus / SearChainResult ─────────────────────────────────────────

#[test]
fn status_is_complete() {
    assert!(SearChainStatus::Complete.is_complete());
    assert!(
        !SearChainStatus::PartialBacktrackExhausted {
            unresolved_node_ids: vec![1]
        }
        .is_complete()
    );
}

// ── build_chain: validation ───────────────────────────────────────────────────

#[test]
fn build_chain_rejects_empty_query() {
    let engine = default_engine();
    let generator = MockChainGenerator::default();
    let err = engine.build_chain("   ", &generator).unwrap_err();
    assert!(matches!(err, SearChainError::EmptyQuery));
}

#[test]
fn build_chain_rejects_empty_chain() {
    let engine = default_engine();
    let generator = FixedChainGenerator(vec![]);
    let err = engine.build_chain("query", &generator).unwrap_err();
    assert!(matches!(err, SearChainError::EmptyChain));
}

#[test]
fn build_chain_rejects_non_sequential_ids() {
    let engine = default_engine();
    let generator = FixedChainGenerator(vec![ChainNode::new(1, "q", "a")]);
    let err = engine.build_chain("query", &generator).unwrap_err();
    match err {
        SearChainError::NonSequentialIds { expected, found } => {
            assert_eq!(expected, 0);
            assert_eq!(found, 1);
        }
        other => panic!("expected NonSequentialIds, got {other:?}"),
    }
}

#[test]
fn build_chain_rejects_self_dependency() {
    let engine = default_engine();
    let generator = FixedChainGenerator(vec![
        ChainNode::new(0, "q0", "a0"),
        ChainNode::new(1, "q1", "a1").with_depends_on(vec![1]),
    ]);
    let err = engine.build_chain("query", &generator).unwrap_err();
    match err {
        SearChainError::InvalidDependency { node_id, dep_id } => {
            assert_eq!(node_id, 1);
            assert_eq!(dep_id, 1);
        }
        other => panic!("expected InvalidDependency, got {other:?}"),
    }
}

#[test]
fn build_chain_rejects_forward_dependency() {
    let engine = default_engine();
    let generator = FixedChainGenerator(vec![
        ChainNode::new(0, "q0", "a0").with_depends_on(vec![1]),
        ChainNode::new(1, "q1", "a1"),
    ]);
    let err = engine.build_chain("query", &generator).unwrap_err();
    match err {
        SearChainError::InvalidDependency { node_id, dep_id } => {
            assert_eq!(node_id, 0);
            assert_eq!(dep_id, 1);
        }
        other => panic!("expected InvalidDependency, got {other:?}"),
    }
}

#[test]
fn build_chain_accepts_valid_chain() {
    let engine = default_engine();
    let generator = FixedChainGenerator(vec![
        ChainNode::new(0, "q0", "a0"),
        ChainNode::new(1, "q1", "a1").with_depends_on(vec![0]),
    ]);
    let chain = engine.build_chain("query", &generator).unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[1].depends_on, vec![0]);
}

// ── verify_chain: no conflicts ────────────────────────────────────────────────

#[test]
fn verify_chain_all_verified_straight_through() {
    let engine = default_engine();
    let mut chain = vec![
        ChainNode::new(0, "What is the capital of France?", "Paris"),
        ChainNode::new(
            1,
            "What river runs through the capital of France?",
            "The Seine runs through Paris.",
        )
        .with_depends_on(vec![0]),
    ];

    let retriever = MockSearchainRetriever::new(vec![
        (
            // Unique to node 0's sub-query; node 1's sub-query ("What river
            // runs through the capital of France?") also ends in "the
            // capital of France", so the needle must be specific enough not
            // to accidentally match node 1 too.
            "What is the capital of France".to_string(),
            vec!["Paris is the capital of France.".to_string()],
        ),
        (
            "river runs through".to_string(),
            vec!["The Seine runs through Paris, the capital.".to_string()],
        ),
    ]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(status, SearChainStatus::Complete);
    assert_eq!(backtrack_events, 0);
    for node in &chain {
        assert_eq!(node.verdict, Some(NodeVerdict::Verified));
        assert!(!node.revised);
    }
}

// ── verify_chain: isolated conflicting node, no dependents ───────────────────

#[test]
fn verify_chain_isolated_conflict_needs_no_cascade() {
    let engine = default_engine();
    // A single node with no dependents: it can still conflict on its own
    // merits, but there is nothing downstream to backtrack into.
    let mut chain = vec![ChainNode::new(
        0,
        "Was the treaty ratified?",
        "The treaty was ratified.",
    )];

    let retriever = MockSearchainRetriever::new(vec![(
        "Was the treaty ratified".to_string(),
        vec!["The treaty was not ratified.".to_string()],
    )]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(status, SearChainStatus::Complete);
    assert_eq!(
        backtrack_events, 0,
        "an isolated conflicting node has no dependents to cascade into"
    );
    assert_eq!(chain[0].verdict, Some(NodeVerdict::Conflicting));
    assert!(chain[0].revised);
    assert_eq!(chain[0].final_answer, "The treaty was not ratified.");
}

#[test]
fn verify_chain_isolated_conflict_amid_siblings_does_not_touch_unrelated_nodes() {
    let engine = default_engine();
    // Node 1 conflicts but has no dependents; node 2 is unrelated (depends on
    // node 0, not node 1) and must be verified independently, unaffected by
    // node 1's conflict.
    let mut chain = vec![
        ChainNode::new(0, "capital of Spain", "Madrid"),
        ChainNode::new(1, "Was the treaty ratified?", "The treaty was ratified."),
        // Deliberately does not textually contain node 0's sub-query
        // ("capital of Spain"), so the mock retriever's substring matching
        // can distinguish it from node 0's own mapping.
        ChainNode::new(2, "language spoken in Madrid", "Spanish").with_depends_on(vec![0]),
    ];

    let retriever = MockSearchainRetriever::new(vec![
        (
            "capital of Spain".to_string(),
            vec!["Madrid is the capital of Spain.".to_string()],
        ),
        (
            "Was the treaty ratified".to_string(),
            vec!["The treaty was not ratified.".to_string()],
        ),
        (
            "language spoken".to_string(),
            vec!["Spanish is spoken in Madrid, the capital of Spain.".to_string()],
        ),
    ]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(status, SearChainStatus::Complete);
    assert_eq!(backtrack_events, 0);
    assert_eq!(chain[0].verdict, Some(NodeVerdict::Verified));
    assert!(!chain[0].revised);
    assert_eq!(chain[1].verdict, Some(NodeVerdict::Conflicting));
    assert!(chain[1].revised);
    assert_eq!(chain[2].verdict, Some(NodeVerdict::Verified));
    assert!(!chain[2].revised);
}

// ── verify_chain: cascading backtrack (the centerpiece) ───────────────────────

#[test]
fn verify_chain_conflict_cascades_to_direct_dependent() {
    let engine = default_engine();
    // Node 1's tentative answer was built assuming node 0's (wrong)
    // tentative answer; node 0 turns out to be Conflicting, which must
    // trigger an explicit re-verification of node 1.
    let mut chain = vec![
        ChainNode::new(0, "Was the treaty ratified?", "The treaty was ratified."),
        ChainNode::new(
            1,
            "Did trade resume after the treaty?",
            "Trade resumed because The treaty was ratified.",
        )
        .with_depends_on(vec![0]),
    ];

    let retriever = MockSearchainRetriever::new(vec![
        (
            "Was the treaty ratified".to_string(),
            vec!["The treaty was not ratified.".to_string()],
        ),
        (
            "Did trade resume".to_string(),
            vec!["Trade resumed because the treaty was not ratified after all.".to_string()],
        ),
    ]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(status, SearChainStatus::Complete);
    assert!(
        backtrack_events >= 1,
        "node 1 must have been explicitly re-verified as node 0's dependent"
    );

    assert_eq!(chain[0].verdict, Some(NodeVerdict::Conflicting));
    assert!(chain[0].revised);
    assert_eq!(chain[0].final_answer, "The treaty was not ratified.");

    // Node 1's evidence was fetched twice: once on the first sweep, once as
    // part of the backtrack cascade. Either way it must have settled on a
    // terminal verdict and reflect the corrected evidence.
    assert!(chain[1].is_terminal());
    assert!(chain[1].revised);
    assert_eq!(
        chain[1].final_answer,
        "Trade resumed because the treaty was not ratified after all."
    );
}

#[test]
fn verify_chain_conflict_cascades_transitively_two_levels() {
    let engine = default_engine();
    // A -> B -> C: A conflicts, forcing an explicit re-verification of B.
    // Node B's tentative answer *textually embeds* node A's original
    // tentative answer verbatim, so the backtracking patch swaps in A's
    // corrected text; that patched candidate now disagrees with B's own
    // (year-sensitive) evidence, so B *itself* comes out Conflicting on the
    // cascade recheck -- which must in turn pull in C, which depends on B
    // (not directly on A). A plain single pass over dependents-of-A would
    // never reach C: this is the "downstream nodes of a downstream node"
    // case.
    let mut chain = vec![
        ChainNode::new(
            0,
            "founding year of the observatory",
            "The observatory was founded in 1990.",
        ),
        ChainNode::new(
            1,
            "director appointed the year the observatory was founded",
            "Dr. Lee was appointed director. The observatory was founded in 1990.",
        )
        .with_depends_on(vec![0]),
        ChainNode::new(
            2,
            "first publication under the appointed director",
            "The first publication credits the appointed director and the wrong founding year.",
        )
        .with_depends_on(vec![1]),
    ];

    let retriever = MockSearchainRetriever::new(vec![
        (
            "founding year of the observatory".to_string(),
            vec!["The observatory was founded in 2005.".to_string()],
        ),
        (
            "director appointed the year".to_string(),
            vec!["Dr. Lee was appointed director in 1990.".to_string()],
        ),
        (
            "first publication under the appointed director".to_string(),
            vec!["The first publication under the correct director came out in 2006.".to_string()],
        ),
    ]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(status, SearChainStatus::Complete);
    // Node 1 is re-verified as node 0's direct dependent, and node 2 is
    // re-verified as node 1's direct dependent once node 1 itself comes out
    // Conflicting on its cascade recheck -- two distinct backtracking
    // events, not one.
    assert!(
        backtrack_events >= 2,
        "expected cascading re-verification of both downstream nodes, got {backtrack_events}"
    );

    assert_eq!(chain[0].verdict, Some(NodeVerdict::Conflicting));
    assert!(chain[0].revised);
    assert_eq!(
        chain[0].final_answer,
        "The observatory was founded in 2005."
    );

    assert_eq!(
        chain[1].verdict,
        Some(NodeVerdict::Conflicting),
        "node 1's patched (year-corrected) premise must itself conflict with its own evidence"
    );
    assert!(chain[1].revised, "node 1 must have been patched/revised");

    assert!(chain[2].revised, "node 2 must have been patched/revised");
    // Confirm node 2 was genuinely reached via node 1's cascade: its final
    // answer must reflect the corrected publication evidence.
    assert!(chain[2].final_answer.contains("2006"));
}

// ── verify_chain: max_backtrack_iterations exhaustion ─────────────────────────

#[test]
fn verify_chain_reports_partial_status_when_backtrack_budget_exhausted() {
    // A single conflicting ancestor fans out to three dependents, but the
    // backtracking budget only allows one re-verification: two dependents
    // must be honestly reported as unresolved rather than silently skipped
    // or silently trusted.
    let engine = SearChainEngine::new(SearChainConfig::new().with_max_backtrack_iterations(1));

    let mut chain = vec![
        ChainNode::new(0, "Was the treaty ratified?", "The treaty was ratified."),
        ChainNode::new(
            1,
            "consequence A of the treaty",
            "A depends on The treaty was ratified.",
        )
        .with_depends_on(vec![0]),
        ChainNode::new(
            2,
            "consequence B of the treaty",
            "B depends on The treaty was ratified.",
        )
        .with_depends_on(vec![0]),
        ChainNode::new(
            3,
            "consequence C of the treaty",
            "C depends on The treaty was ratified.",
        )
        .with_depends_on(vec![0]),
    ];

    let retriever = MockSearchainRetriever::new(vec![(
        "Was the treaty ratified".to_string(),
        vec!["The treaty was not ratified.".to_string()],
    )]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(backtrack_events, 1, "budget caps re-verifications at 1");
    match status {
        SearChainStatus::PartialBacktrackExhausted {
            unresolved_node_ids,
        } => {
            assert_eq!(
                unresolved_node_ids.len(),
                2,
                "two of the three dependents never got a chance to be re-verified: {unresolved_node_ids:?}"
            );
            for id in &unresolved_node_ids {
                assert!((1..=3).contains(id));
            }
        }
        SearChainStatus::Complete => panic!("expected a partial, budget-exhausted status"),
    }

    // The ancestor itself is not "unresolved" — its own conflict was fully
    // processed; only its dependents' re-verification was cut short.
    assert_eq!(chain[0].verdict, Some(NodeVerdict::Conflicting));
    assert!(chain[0].revised);
}

#[test]
fn verify_chain_partial_status_unresolved_includes_transitive_dependents() {
    // A -> B -> C, with a budget of zero: B never even gets a chance to be
    // re-verified, so C (which depends on B) must also be reported as
    // unresolved even though the cascade never literally reached it.
    let engine = SearChainEngine::new(SearChainConfig::new().with_max_backtrack_iterations(0));

    let mut chain = vec![
        ChainNode::new(0, "Was the treaty ratified?", "The treaty was ratified."),
        ChainNode::new(
            1,
            "consequence of the treaty",
            "Consequence depends on The treaty was ratified.",
        )
        .with_depends_on(vec![0]),
        ChainNode::new(
            2,
            "second-order consequence",
            "Second order depends on the first consequence.",
        )
        .with_depends_on(vec![1]),
    ];

    let retriever = MockSearchainRetriever::new(vec![(
        "Was the treaty ratified".to_string(),
        vec!["The treaty was not ratified.".to_string()],
    )]);

    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);

    assert_eq!(backtrack_events, 0);
    match status {
        SearChainStatus::PartialBacktrackExhausted {
            unresolved_node_ids,
        } => {
            assert_eq!(unresolved_node_ids, vec![1, 2]);
        }
        SearChainStatus::Complete => panic!("expected a partial, budget-exhausted status"),
    }
}

// ── SearChainResult helpers via SearChainEngine::run ──────────────────────────

#[test]
fn run_full_pipeline_success_last_node_strategy() {
    let engine = default_engine();
    let chain = vec![
        ChainNode::new(0, "capital of Japan", "Tokyo"),
        ChainNode::new(1, "population of the capital of Japan", "About 14 million")
            .with_depends_on(vec![0]),
    ];
    let chain_generator = MockChainGenerator::new(vec![("Japan".to_string(), chain)]);
    // Node 1's sub-query ("population of the capital of Japan") also
    // contains node 0's sub-query text ("capital of Japan"); listing the
    // more specific mapping first ensures the mock retriever's
    // first-match-wins substring search resolves each node to its own
    // evidence.
    let retriever = MockSearchainRetriever::new(vec![
        (
            "population of the capital".to_string(),
            vec!["Tokyo, the capital, has a population of about 14 million.".to_string()],
        ),
        (
            "capital of Japan".to_string(),
            vec!["Tokyo is the capital of Japan.".to_string()],
        ),
    ]);
    let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::LastNode);

    let result = engine
        .run(
            "What is the population of the capital of Japan?",
            &chain_generator,
            &retriever,
            &generator,
        )
        .unwrap();

    assert_eq!(
        result.original_query,
        "What is the population of the capital of Japan?"
    );
    assert!(result.is_complete());
    assert_eq!(result.chain.len(), 2);
    assert_eq!(result.final_answer, result.chain[1].final_answer);
    assert!(result.revised_node_ids().is_empty());
    assert!(result.unresolved_node_ids().is_empty());
}

#[test]
fn run_propagates_build_chain_errors() {
    let engine = default_engine();
    let chain_generator = MockChainGenerator::default();
    let retriever = MockSearchainRetriever::default();
    let generator = MockSearchainGenerator::default();

    let err = engine
        .run("   ", &chain_generator, &retriever, &generator)
        .unwrap_err();
    assert!(matches!(err, SearChainError::EmptyQuery));
}

#[test]
fn run_join_all_strategy_reflects_every_node() {
    let engine = default_engine();
    let chain = vec![
        ChainNode::new(0, "q0 about widgets", "widgets are useful"),
        ChainNode::new(1, "q1 about gadgets", "gadgets are useful too").with_depends_on(vec![0]),
    ];
    let chain_generator = MockChainGenerator::new(vec![("widgets".to_string(), chain)]);
    let retriever = MockSearchainRetriever::default(); // no evidence anywhere -> Unverified, kept as-is
    let generator = MockSearchainGenerator::new(AnswerAssemblyStrategy::JoinAll);

    let result = engine
        .run(
            "tell me about widgets and gadgets",
            &chain_generator,
            &retriever,
            &generator,
        )
        .unwrap();

    assert!(result.is_complete());
    assert_eq!(
        result.final_answer,
        "widgets are useful; gadgets are useful too"
    );
    // No evidence anywhere means every node stayed Unverified with its
    // original tentative answer -- nothing was "revised" from evidence.
    for node in &result.chain {
        assert_eq!(node.verdict, Some(NodeVerdict::Unverified));
        assert!(!node.revised);
    }
}

#[test]
fn result_revised_node_ids_lists_only_changed_nodes() {
    let engine = default_engine();
    let mut chain = vec![
        ChainNode::new(0, "capital of Italy", "Rome"),
        ChainNode::new(1, "Was the treaty ratified?", "The treaty was ratified."),
    ];
    let retriever = MockSearchainRetriever::new(vec![
        (
            "capital of Italy".to_string(),
            vec!["Rome is the capital of Italy.".to_string()],
        ),
        (
            "Was the treaty ratified".to_string(),
            vec!["The treaty was not ratified.".to_string()],
        ),
    ]);

    let (_status, _events) = engine.verify_chain(&mut chain, &retriever);

    let revised: Vec<usize> = chain.iter().filter(|n| n.revised).map(|n| n.id).collect();
    assert_eq!(revised, vec![1]);
}

// ── SearChainError Display ────────────────────────────────────────────────────

#[test]
fn error_messages_are_non_empty_and_specific() {
    assert!(SearChainError::EmptyQuery.to_string().contains("empty"));
    assert!(
        SearChainError::EmptyChain
            .to_string()
            .contains("empty chain")
    );
    let non_seq = SearChainError::NonSequentialIds {
        expected: 0,
        found: 3,
    };
    assert!(non_seq.to_string().contains('0') && non_seq.to_string().contains('3'));
    let invalid_dep = SearChainError::InvalidDependency {
        node_id: 2,
        dep_id: 5,
    };
    assert!(invalid_dep.to_string().contains('2') && invalid_dep.to_string().contains('5'));
}

// ── verify_chain: empty chain edge case ───────────────────────────────────────

#[test]
fn verify_chain_on_empty_slice_is_trivially_complete() {
    let engine = default_engine();
    let mut chain: Vec<ChainNode> = Vec::new();
    let retriever = MockSearchainRetriever::default();
    let (status, backtrack_events) = engine.verify_chain(&mut chain, &retriever);
    assert_eq!(status, SearChainStatus::Complete);
    assert_eq!(backtrack_events, 0);
}
