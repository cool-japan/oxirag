//! Robustness, configuration, tree/RNG helpers, and the async retrieval path.
#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::doc_markdown
)]
use super::*;

// ═════════════════════════════════════════════════════════════════════════════
// Robustness: what a broken generator or evaluator does to the search
// ═════════════════════════════════════════════════════════════════════════════

/// A `NaN`-returning evaluator must not poison the tree.
///
/// This is the failure mode the clamp exists for. Untreated, a `NaN` return flows into
/// `total_value`, out of `mean_value`, and into every selection score — where it loses
/// every `>` comparison silently, turning the tree policy into "always take the first
/// child". The search keeps running, keeps counting visits, and keeps returning a
/// confident answer produced by a policy that has stopped working.
#[test]
fn a_nan_evaluator_cannot_poison_the_tree() {
    struct Nan;
    impl MctsTerminalEvaluator for Nan {
        fn score(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> f64 {
            f64::NAN
        }
    }
    let world = GoldenPath {
        depth: 3,
        branching: 3,
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(200)
            .with_max_depth(3),
    )
    .search_with("q", &world, &Nan, &[])
    .unwrap();

    for node in &out.tree.nodes {
        assert!(
            node.total_value.is_finite(),
            "node {} accumulated a non-finite value",
            node.id
        );
        assert!(node.mean_value().is_finite());
        assert_eq!(node.mean_value(), 0.0, "NaN maps to the worst legal return");
    }
    assert!(out.stats.rollout_returns.iter().all(|g| g.is_finite()));
    assert!(out.tree.check_visit_invariant());
}

/// Out-of-range scores are clamped into `[0, 1]`, so a miscalibrated evaluator cannot
/// blow up the exploration/exploitation balance.
#[test]
fn out_of_range_scores_are_clamped_into_the_unit_interval() {
    struct Wild;
    impl MctsTerminalEvaluator for Wild {
        fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
            if path.len().is_multiple_of(2) {
                500.0
            } else {
                -300.0
            }
        }
    }
    let world = GoldenPath {
        depth: 3,
        branching: 2,
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(100)
            .with_max_depth(3),
    )
    .search_with("q", &world, &Wild, &[])
    .unwrap();

    for &g in &out.stats.rollout_returns {
        assert!((0.0..=1.0).contains(&g), "return {g} escaped the clamp");
    }
    for node in &out.tree.nodes {
        assert!((0.0..=1.0).contains(&node.mean_value()));
    }
    assert!(out.tree.check_visit_invariant());
}

/// Degenerate priors — negative, zero, `NaN`, infinite — normalize to uniform rather
/// than propagating into the selection score.
#[test]
fn degenerate_priors_fall_back_to_uniform() {
    struct BadPriors;
    impl MctsStepGenerator for BadPriors {
        fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            if path.len() > 1 {
                return Vec::new();
            }
            vec![
                MctsCandidate::with_prior("a", f64::NAN),
                MctsCandidate::with_prior("b", -5.0),
                MctsCandidate::with_prior("c", 0.0),
                MctsCandidate::with_prior("d", f64::INFINITY),
            ]
        }
    }
    struct Half;
    impl MctsTerminalEvaluator for Half {
        fn score(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> f64 {
            0.5
        }
    }

    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(60)
            .with_max_depth(1)
            .with_widening(None)
            .with_selection(MctsSelectionPolicy::Puct { exploration: 2.0 }),
    )
    .search_with("q", &BadPriors, &Half, &[])
    .unwrap();

    let root = out.tree.root().unwrap();
    assert_eq!(root.action_count(), 4);
    for candidate in &root.candidates {
        assert_eq!(
            candidate.prior, 0.25,
            "every degenerate weight must fall back to the uniform 1/4"
        );
    }
    for &id in &root.children {
        assert!(out.tree.nodes[id].prior.is_finite());
    }
    assert!(out.tree.check_visit_invariant());
}

/// A *partially* degenerate prior set keeps the usable weights and floors the rest.
#[test]
fn mixed_priors_normalize_to_a_probability_distribution() {
    struct MixedPriors;
    impl MctsStepGenerator for MixedPriors {
        fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            if path.len() > 1 {
                return Vec::new();
            }
            vec![
                MctsCandidate::with_prior("a", 3.0),
                MctsCandidate::with_prior("b", 1.0),
                MctsCandidate::with_prior("c", -2.0), // floored to 0
            ]
        }
    }
    struct Half;
    impl MctsTerminalEvaluator for Half {
        fn score(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> f64 {
            0.5
        }
    }

    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(20)
            .with_max_depth(1)
            .with_widening(None),
    )
    .search_with("q", &MixedPriors, &Half, &[])
    .unwrap();

    let root = out.tree.root().unwrap();
    let priors: Vec<f64> = root.candidates.iter().map(|c| c.prior).collect();
    assert_eq!(priors, vec![0.75, 0.25, 0.0]);
    let total: f64 = priors.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "priors must sum to 1, got {total}"
    );
}

/// A prior-weighted rollout must actually follow the prior: with a heavily skewed
/// prior, the playout concentrates on the favoured step.
///
/// The measurement isolates the rollout policy cleanly by reading only the **root's
/// creation rollout** — the one rollout that starts from the root before any tree
/// selection has happened, so its every step is chosen by the rollout policy and
/// nothing else. Averaged over many seeds it is a direct estimate of the policy's
/// behaviour, with none of the tree policy's prefix bleeding in. (A rollout taken
/// deeper in the tree is contaminated by the tree-selected prefix above it, which is
/// exactly what made the naive version of this test measure the wrong thing.)
#[test]
fn prior_weighted_rollouts_follow_the_prior() {
    // At *every* level, an almost-certain "common" and an almost-impossible "rare".
    struct Skewed;
    impl MctsStepGenerator for Skewed {
        fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            if path.len() > 5 {
                return Vec::new();
            }
            vec![
                MctsCandidate::with_prior("rare", 1.0),
                MctsCandidate::with_prior("common", 99.0),
            ]
        }
    }
    // The fraction of the rollout's steps that chose "common".
    struct FractionCommon;
    impl MctsTerminalEvaluator for FractionCommon {
        fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
            let steps = &path[1..];
            if steps.is_empty() {
                return 0.0;
            }
            steps.iter().filter(|s| s.as_str() == "common").count() as f64 / steps.len() as f64
        }
    }

    // One simulation => the recorded rollout is exactly the root's creation rollout, a
    // pure five-step sample of the rollout policy. Average it over many seeds.
    let root_rollout = |policy: MctsRolloutPolicy, seed: u64| -> f64 {
        MctsEngine::new(
            MctsConfig::default()
                .with_simulations(1)
                .with_max_depth(5)
                .with_rollout(policy)
                .with_seed(seed),
        )
        .search_with("q", &Skewed, &FractionCommon, &[])
        .unwrap()
        .stats
        .rollout_returns[0]
    };

    let samples = 300u64;
    let mean = |policy: MctsRolloutPolicy| -> f64 {
        (0..samples).map(|s| root_rollout(policy, s)).sum::<f64>() / samples as f64
    };
    let prior_mean = mean(MctsRolloutPolicy::PriorWeighted);
    let uniform_mean = mean(MctsRolloutPolicy::Uniform);

    // A 99:1 prior at every step must make almost every step "common".
    assert!(
        prior_mean > 0.9,
        "a 99:1 prior-weighted rollout must be almost all 'common', got {prior_mean:.3}"
    );
    // A uniform rollout ignores the prior entirely: ~50/50.
    assert!(
        (0.4..0.6).contains(&uniform_mean),
        "a uniform rollout must be a coin flip regardless of the prior, got {uniform_mean:.3}"
    );
    assert!(
        prior_mean > uniform_mean + 0.3,
        "the prior must move the rollouts substantially: {prior_mean:.3} vs {uniform_mean:.3}"
    );
}

/// `rollout_depth_cap: Some(0)` turns the rollout off: the return is the evaluator's
/// score of the node's own path, and no simulated steps are taken.
#[test]
fn a_zero_rollout_cap_scores_the_node_in_place() {
    // Returns the path length, so a rollout that took extra steps would show up.
    struct PathLen;
    impl MctsTerminalEvaluator for PathLen {
        fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
            (path.len() - 1) as f64 / 4.0
        }
    }
    let world = GoldenPath {
        depth: 4,
        branching: 2,
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(60)
            .with_max_depth(4)
            .with_rollout_depth_cap(Some(0))
            .with_widening(None),
    )
    .search_with("q", &world, &PathLen, &[])
    .unwrap();

    // The root's creation rollout took no steps, so its return is its own depth: 0.
    assert_eq!(out.stats.rollout_returns[0], 0.0);

    // For an *internal* node, its own creation rollout is its value residue — its
    // accumulated value minus its children's (see `a_parents_value_residue...`). With
    // the rollout capped to zero steps that residue must be exactly the evaluator's
    // in-place score of the node's own path, `depth/4`, because the creation rollout
    // advanced no further than the node it started from. (The node's *mean* value is
    // not `depth/4`: it also averages in every deeper rollout backed up through it —
    // that is what backup is. And this holds only for internal nodes: a terminal leaf
    // re-scored on every visit has residue `visits * depth/4`, handled just below.)
    for node in &out.tree.nodes {
        if node.children.is_empty() {
            continue;
        }
        let children_total: f64 = node
            .children
            .iter()
            .map(|&c| out.tree.nodes[c].total_value)
            .sum();
        let own_rollout = node.total_value - children_total;
        let expected = node.depth as f64 / 4.0;
        // `total_value` and the children's totals are sums formed in different orders,
        // so the residue equals the in-place score up to accumulated rounding.
        assert!(
            (own_rollout - expected).abs() < 1e-9,
            "node {} at depth {} has a creation-rollout residue of {own_rollout}, but a \
             zero-cap rollout must score it in place at {expected}",
            node.id,
            node.depth,
        );
    }

    // Every childless node was scored *only* in place, on every one of its visits, so
    // its mean is exactly `depth/4` — no rollout ever stepped past it. `depth/4` is a
    // multiple of `0.25`, hence exact, so this is a bit-for-bit equality.
    for node in &out.tree.nodes {
        if node.children.is_empty() {
            assert_eq!(
                node.mean_value(),
                node.depth as f64 / 4.0,
                "a childless node's mean is exactly its single in-place score"
            );
        }
    }
    assert!(out.tree.check_visit_invariant());
}

// ═════════════════════════════════════════════════════════════════════════════
// Configuration and errors
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn config_validation_rejects_unrunnable_searches() {
    let world = FlatArms { values: vec![1.0] };
    let bad: Vec<(&str, MctsConfig)> = vec![
        ("zero budget", MctsConfig::default().with_simulations(0)),
        (
            "u32-overflowing budget",
            MctsConfig::default().with_simulations(u32::MAX as usize),
        ),
        ("zero depth", MctsConfig::default().with_max_depth(0)),
        (
            "negative exploration",
            MctsConfig::default().with_selection(MctsSelectionPolicy::Uct { exploration: -1.0 }),
        ),
        (
            "non-finite exploration",
            MctsConfig::default().with_selection(MctsSelectionPolicy::Puct {
                exploration: f64::NAN,
            }),
        ),
        (
            "zero widening coefficient",
            MctsConfig::default().with_widening(Some(MctsWidening::new(0.0, 0.5))),
        ),
        (
            "widening exponent of 1",
            MctsConfig::default().with_widening(Some(MctsWidening::new(1.0, 1.0))),
        ),
        (
            "widening exponent above 1",
            MctsConfig::default().with_widening(Some(MctsWidening::new(1.0, 1.5))),
        ),
        (
            "widening exponent of 0",
            MctsConfig::default().with_widening(Some(MctsWidening::new(1.0, 0.0))),
        ),
    ];

    for (name, config) in bad {
        assert!(
            config.validate().is_err(),
            "[{name}] must be rejected by validate()"
        );
        let err = MctsEngine::new(config)
            .search_with("q", &world, &world, &[])
            .unwrap_err();
        assert!(
            matches!(err, MctsError::InvalidConfig(_)),
            "[{name}] search_with must surface InvalidConfig, got {err:?}"
        );
    }

    assert!(MctsConfig::default().validate().is_ok());
}

#[test]
fn an_empty_query_is_rejected() {
    let world = FlatArms { values: vec![1.0] };
    let engine = MctsEngine::default();
    for query in ["", "   ", "\t\n"] {
        let err = engine.search_with(query, &world, &world, &[]).unwrap_err();
        assert!(matches!(err, MctsError::EmptyQuery), "got {err:?}");
    }
}

#[test]
fn a_generator_with_no_root_candidates_is_an_error() {
    struct Nothing;
    impl MctsStepGenerator for Nothing {
        fn propose(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            Vec::new()
        }
    }
    struct Half;
    impl MctsTerminalEvaluator for Half {
        fn score(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> f64 {
            0.5
        }
    }
    let err = MctsEngine::default()
        .search_with("q", &Nothing, &Half, &[])
        .unwrap_err();
    assert!(matches!(err, MctsError::NoCandidates), "got {err:?}");
}

/// A single-candidate, single-simulation search is the smallest legal search, and it
/// must still satisfy every invariant.
#[test]
fn the_smallest_legal_search_is_well_formed() {
    let world = FlatArms { values: vec![0.7] };
    let out = MctsEngine::new(MctsConfig::default().with_simulations(1).with_max_depth(1))
        .search_with("q", &world, &world, &[])
        .unwrap();

    assert_eq!(out.tree.nodes.len(), 2, "root plus its single child");
    assert_eq!(
        out.tree.root().unwrap().visits,
        2,
        "1 simulation + creation"
    );
    assert_eq!(out.tree.nodes[1].visits, 1);
    assert!(out.tree.check_visit_invariant());
    assert_eq!(out.best_root_action, Some(1));
    assert_eq!(out.stats.rollout_returns.len(), 2);
    assert_eq!(out.stats.root_action_history, vec![1]);
}

/// Builder methods round-trip into the config they claim to set.
#[test]
fn config_builders_round_trip() {
    let config = MctsConfig::default()
        .with_simulations(11)
        .with_max_depth(7)
        .with_rollout_depth_cap(Some(2))
        .with_selection(MctsSelectionPolicy::Puct { exploration: 1.25 })
        .with_rollout(MctsRolloutPolicy::PriorWeighted)
        .with_widening(Some(MctsWidening::new(2.0, 0.25)))
        .with_seed(99)
        .with_top_k(3);

    assert_eq!(config.simulations, 11);
    assert_eq!(config.max_depth, 7);
    assert_eq!(config.rollout_depth_cap, Some(2));
    assert_eq!(
        config.selection,
        MctsSelectionPolicy::Puct { exploration: 1.25 }
    );
    assert_eq!(config.rollout, MctsRolloutPolicy::PriorWeighted);
    assert_eq!(config.widening, Some(MctsWidening::new(2.0, 0.25)));
    assert_eq!(config.seed, 99);
    assert_eq!(config.top_k, 3);
    assert!(config.validate().is_ok());
}

// ═════════════════════════════════════════════════════════════════════════════
// Tree helpers
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn tree_paths_run_from_the_root_downward() {
    let world = GoldenPath {
        depth: 3,
        branching: 2,
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(80)
            .with_max_depth(3)
            .with_widening(None),
    )
    .search_with("what", &world, &world, &[])
    .unwrap();

    let deepest = out
        .tree
        .nodes
        .iter()
        .max_by_key(|n| n.depth)
        .map(|n| n.id)
        .unwrap();
    let path = out.tree.path_from_root(deepest);
    assert_eq!(path[0], 0, "the root comes first");
    assert_eq!(*path.last().unwrap(), deepest);
    assert_eq!(path.len(), out.tree.nodes[deepest].depth + 1);

    let contents = out.tree.path_contents(deepest);
    assert_eq!(contents[0], "Question: what");
    assert_eq!(contents.len(), path.len());

    assert_eq!(out.tree.node(0).map(|n| n.id), Some(0));
    assert!(out.tree.node(9999).is_none());
    assert!(MctsTree::default().root().is_none());
    assert!(MctsTree::default().principal_variation().is_empty());
    assert!(MctsTree::default().check_visit_invariant());
    assert!(MctsTree::default().best_child_by_visits(0).is_none());
}

#[test]
fn node_accessors_describe_the_expansion_state() {
    let mut node = make_node(0, 4, 2.0, 1.0, 3);
    assert_eq!(node.mean_value(), 0.5);
    assert_eq!(node.action_count(), 3);
    assert_eq!(node.untried_count(), 3);
    assert!(!node.fully_expanded());

    node.children = vec![1, 2];
    assert_eq!(node.untried_count(), 1);
    assert!(!node.fully_expanded());

    node.children = vec![1, 2, 3];
    assert_eq!(node.untried_count(), 0);
    assert!(node.fully_expanded());

    let unvisited = make_node(1, 0, 0.0, 0.5, 0);
    assert_eq!(
        unvisited.mean_value(),
        0.0,
        "an unvisited node has no estimate"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The RNG
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn the_rng_is_deterministic_and_well_ranged() {
    let mut a = MctsRng::new(42);
    let mut b = MctsRng::new(42);
    let mut c = MctsRng::new(43);

    let stream_a: Vec<u64> = (0..64).map(|_| a.next_u64()).collect();
    let stream_b: Vec<u64> = (0..64).map(|_| b.next_u64()).collect();
    assert_eq!(stream_a, stream_b, "the same seed must replay exactly");
    let stream_c: Vec<u64> = (0..64).map(|_| c.next_u64()).collect();
    assert_ne!(stream_a, stream_c);

    // Even seed 0 is a full-quality seed — SplitMix64 has no degenerate state.
    let mut zero = MctsRng::new(0);
    let first: Vec<u64> = (0..8).map(|_| zero.next_u64()).collect();
    assert!(first.iter().all(|&x| x != 0));

    let mut u = MctsRng::new(7);
    for _ in 0..10_000 {
        let x = u.next_f64();
        assert!(
            (0.0..1.0).contains(&x),
            "next_f64 must stay in [0, 1), got {x}"
        );
    }
}

#[test]
fn next_usize_below_covers_its_range_and_refuses_zero() {
    let mut rng = MctsRng::new(2024);
    assert_eq!(
        rng.next_usize_below(0),
        None,
        "there is no index below zero"
    );
    assert_eq!(rng.next_usize_below(1), Some(0));

    let mut counts = [0usize; 5];
    for _ in 0..50_000 {
        let index = rng.next_usize_below(5).unwrap();
        assert!(index < 5);
        counts[index] += 1;
    }
    // Roughly uniform: every bucket within 10% of 10_000.
    for (i, &count) in counts.iter().enumerate() {
        assert!(
            (9_000..11_000).contains(&count),
            "bucket {i} got {count}, which is not close to uniform"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The default heuristics, and the async retrieval path
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn the_heuristic_generator_proposes_context_terms_by_frequency() {
    let context = vec![
        make_result("photosynthesis converts sunlight into chemical energy", 0.9),
        make_result("photosynthesis occurs in chloroplasts", 0.8),
    ];
    let generator = MctsHeuristicGenerator::new(3);
    let candidates = generator.propose(
        "how does photosynthesis work",
        &["Question: how does photosynthesis work".to_string()],
        &context,
    );

    assert!(!candidates.is_empty());
    assert!(candidates.len() <= 3);
    // "photosynthesis" appears three times (twice in context, once in the query), more
    // than anything else, so it must lead — and carry the largest prior.
    assert_eq!(candidates[0].content, "Examine: photosynthesis");
    for other in &candidates[1..] {
        assert!(candidates[0].prior >= other.prior);
    }

    // A term already on the path is not proposed again.
    let follow_up = generator.propose(
        "how does photosynthesis work",
        &[
            "Question: how does photosynthesis work".to_string(),
            "Examine: photosynthesis".to_string(),
        ],
        &context,
    );
    assert!(
        follow_up
            .iter()
            .all(|c| c.content != "Examine: photosynthesis"),
        "a covered term must not be proposed twice on the same path"
    );
}

#[test]
fn the_heuristic_evaluator_rewards_query_term_coverage() {
    let evaluator = MctsHeuristicEvaluator;
    let query = "chloroplast sunlight";

    let none = evaluator.score(query, &["Question: chloroplast sunlight".into()], &[]);
    let half = evaluator.score(
        query,
        &["Question: x".into(), "Examine: chloroplast".into()],
        &[],
    );
    let full = evaluator.score(
        query,
        &[
            "Question: x".into(),
            "Examine: chloroplast".into(),
            "Examine: sunlight".into(),
        ],
        &[],
    );

    // The root's own content restates the query, so it already covers both terms.
    assert_eq!(none, 1.0);
    assert_eq!(half, 0.5);
    assert_eq!(full, 1.0);
    assert!(full > half, "more coverage must score higher");
    assert_eq!(evaluator.score("", &["anything".into()], &[]), 0.0);
}

#[tokio::test]
async fn run_retrieves_context_and_searches_with_the_default_heuristics() {
    let echo = MockEcho {
        results: vec![
            make_result("photosynthesis converts sunlight into chemical energy", 0.9),
            make_result("chloroplasts contain chlorophyll pigment", 0.8),
        ],
    };
    let engine = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(100)
            .with_max_depth(3),
    );
    let out = engine
        .run("how does photosynthesis work", &echo)
        .await
        .unwrap();

    assert!(!out.answer.is_empty());
    assert!(out.best_path.len() >= 2, "the root plus at least one step");
    assert_eq!(out.best_path[0], "Question: how does photosynthesis work");
    assert!(out.best_root_action.is_some());
    assert_eq!(out.stats.simulations, 100);
    assert!(out.tree.check_visit_invariant());
    assert!(
        out.answer.contains("photosynthesis"),
        "the answer must weave in the retrieved evidence: {}",
        out.answer
    );
}

#[tokio::test]
async fn run_with_accepts_a_custom_generator_and_evaluator() {
    let echo = MockEcho {
        results: vec![make_result("irrelevant", 0.5)],
    };
    let world = FlatArms {
        values: vec![0.2, 0.95, 0.4],
    };
    let engine = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(120)
            .with_max_depth(1)
            .with_widening(None),
    );
    let out = engine
        .run_with("pick the best", &echo, &world, &world)
        .await
        .unwrap();

    assert_eq!(out.best_path[1], "arm1", "must find the 0.95 arm");
    assert!(out.tree.check_visit_invariant());
}

#[tokio::test]
async fn run_rejects_an_empty_query_before_retrieving() {
    let echo = MockEcho { results: vec![] };
    let err = MctsEngine::default().run("  ", &echo).await.unwrap_err();
    assert!(matches!(err, MctsError::EmptyQuery), "got {err:?}");
}

#[tokio::test]
async fn a_search_with_no_retrieved_context_still_runs() {
    let echo = MockEcho { results: vec![] };
    // With no context the heuristic generator still has the query's own terms to work
    // with, so the search is well-formed.
    let out = MctsEngine::new(MctsConfig::default().with_simulations(40).with_max_depth(2))
        .run("explain cellular respiration", &echo)
        .await
        .unwrap();

    assert!(out.tree.check_visit_invariant());
    assert!(!out.answer.is_empty());
}

/// The engine is `Send`/`Sync` and cloneable, so it can be shared across a request
/// handler without ceremony.
#[test]
fn the_engine_is_shareable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MctsEngine>();
    assert_send_sync::<MctsConfig>();
    assert_send_sync::<MctsTree>();

    let engine = MctsEngine::new(MctsConfig::default().with_simulations(7));
    let clone = engine.clone();
    assert_eq!(clone.config.simulations, 7);
    assert_eq!(MctsEngine::default().config, MctsConfig::default());
}
