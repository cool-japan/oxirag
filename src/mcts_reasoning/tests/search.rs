//! End-to-end search tests: widening, backprop invariants, convergence, regret, determinism.
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
// Progressive widening
// ═════════════════════════════════════════════════════════════════════════════

/// The widening limit `k(N) = ceil(C * N^alpha)`, pinned exactly.
///
/// These constants are the guard on the `powf`/`ceil` rounding described in
/// [`MctsWidening`]: a platform whose `pow` overshoots `9^0.5` to `3.0000000000000004`
/// would return `4` here instead of `3` and silently widen the tree faster than the
/// schedule allows.
#[test]
fn widening_limits_are_exact_at_the_integer_boundaries() {
    let sqrt_schedule = MctsWidening::default();
    // ceil(sqrt(N)): the k-th child unlocks at N = (k-1)^2 + 1.
    let expected: [(u32, usize); 15] = [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2), // sqrt(4) = 2 exactly -> ceil = 2, NOT 3
        (5, 3),
        (8, 3),
        (9, 3), // sqrt(9) = 3 exactly -> ceil = 3, NOT 4
        (10, 4),
        (16, 4), // exact square again
        (17, 5),
        (25, 5),
        (26, 6),
        (81, 9),
        (82, 10),
        (100, 10),
    ];
    for (visits, want) in expected {
        assert_eq!(
            sqrt_schedule.limit(visits),
            want,
            "ceil(sqrt({visits})) must be {want}"
        );
    }

    // A fourth-root schedule, whose exact powers are the other classic trap.
    let quarter = MctsWidening::new(1.0, 0.25);
    for (visits, want) in [(16u32, 2usize), (81, 3), (256, 4), (257, 5), (625, 5)] {
        assert_eq!(quarter.limit(visits), want, "ceil({visits}^0.25)");
    }

    // Never zero — a visited non-terminal node must always be able to acquire a child.
    assert_eq!(MctsWidening::new(0.01, 0.5).limit(1), 1);
    assert_eq!(MctsWidening::new(0.01, 0.5).limit(4), 1);
}

/// Widening actually bounds the branching factor of a real search: with 20 candidates
/// on offer, the root ends up with far fewer children than that — and its children
/// count matches the schedule's allowance for the visits it received.
#[test]
fn progressive_widening_bounds_the_branching_factor_of_a_real_search() {
    // A generator that offers 20 continuations at every node.
    struct Wide;
    impl MctsStepGenerator for Wide {
        fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            if path.len() > 2 {
                return Vec::new();
            }
            (0..20)
                .map(|i| MctsCandidate::new(format!("w{i}")))
                .collect()
        }
    }
    struct Flat;
    impl MctsTerminalEvaluator for Flat {
        fn score(&self, _q: &str, _p: &[String], _c: &[SearchResult]) -> f64 {
            0.5
        }
    }

    let simulations = 64usize;
    let widened = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(simulations)
            .with_max_depth(2)
            .with_widening(Some(MctsWidening::default())),
    )
    .search_with("q", &Wide, &Flat, &[])
    .unwrap();

    let root = widened.tree.root().unwrap();
    assert_eq!(
        root.visits as usize,
        simulations + 1,
        "the root is visited once per simulation, plus its own creation rollout"
    );
    // The root holds no more children than the schedule allowed at its final visit
    // count, and (because a child is only added on a visit) exactly the allowance at
    // the point of its last widening.
    let allowed = MctsWidening::default().limit(root.visits);
    assert!(
        root.children.len() <= allowed,
        "children ({}) must not exceed the schedule's allowance ({allowed})",
        root.children.len()
    );
    assert!(
        root.children.len() < 20,
        "the whole point: 20 candidates were on offer and widening admitted {}",
        root.children.len()
    );
    assert!(
        root.children.len() >= 8,
        "ceil(sqrt(65)) = 9, so the root should be near-fully widened: {}",
        root.children.len()
    );
    assert_eq!(root.action_count(), 20, "all 20 candidates are still known");
    assert!(!root.fully_expanded());

    // With widening disabled, all 20 candidates become children.
    let unwidened = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(simulations)
            .with_max_depth(2)
            .with_widening(None),
    )
    .search_with("q", &Wide, &Flat, &[])
    .unwrap();
    let root = unwidened.tree.root().unwrap();
    assert_eq!(
        root.children.len(),
        20,
        "without widening, one new child per visit until fully expanded"
    );
    assert!(root.fully_expanded());
}

// ═════════════════════════════════════════════════════════════════════════════
// (d) Backpropagation invariants — exact
// ═════════════════════════════════════════════════════════════════════════════

/// **Headline.** The backup identity, checked exactly, on every internal node of every
/// tree the engine builds — plus the value-side identity, checked *bit for bit*.
///
/// Three separate claims:
///
/// 1. `N(s) == 1 + sum_c N(c)` at every node with children (the `1` is the node's own
///    creation rollout), so `N(root) == simulations + 1`.
/// 2. `Q(s) == W(s) / N(s)` exactly.
/// 3. The root's `W` equals the **in-order** sum of every rollout return the search
///    made — bit for bit, since backpropagation adds them to the root in exactly that
///    order. This is the one that a backup bug cannot survive: it ties the tree's
///    accumulated state to an independently recorded log of what actually happened.
#[test]
fn visit_invariant_holds_exactly_after_every_search() {
    let worlds: Vec<(&str, MctsConfig)> = vec![
        ("uct", MctsConfig::default().with_simulations(300)),
        (
            "puct",
            MctsConfig::default()
                .with_simulations(300)
                .with_selection(MctsSelectionPolicy::Puct { exploration: 1.5 }),
        ),
        (
            "prior-uct",
            MctsConfig::default().with_simulations(300).with_selection(
                MctsSelectionPolicy::PriorUct {
                    exploration: SQRT_2,
                },
            ),
        ),
        (
            "random",
            MctsConfig::default()
                .with_simulations(300)
                .with_selection(MctsSelectionPolicy::UniformRandom),
        ),
        (
            "no-widening",
            MctsConfig::default()
                .with_simulations(300)
                .with_widening(None),
        ),
        (
            "prior-weighted-rollout",
            MctsConfig::default()
                .with_simulations(300)
                .with_rollout(MctsRolloutPolicy::PriorWeighted),
        ),
        (
            "truncated-rollout",
            MctsConfig::default()
                .with_simulations(300)
                .with_rollout_depth_cap(Some(0)),
        ),
    ];

    let world = GoldenPath {
        depth: 5,
        branching: 3,
    };

    for (name, config) in worlds {
        let simulations = config.simulations;
        let out = MctsEngine::new(config.with_max_depth(5))
            .search_with("q", &world, &world, &[])
            .unwrap();

        // 1. The visit identity, at every internal node.
        assert!(
            out.tree.check_visit_invariant(),
            "[{name}] N(s) = 1 + sum_c N(c) must hold at every internal node"
        );
        assert_eq!(
            out.tree.root().unwrap().visits as usize,
            simulations + 1,
            "[{name}] the root sees every simulation, plus its own creation rollout"
        );

        // Re-derive it here rather than trusting the helper.
        for node in &out.tree.nodes {
            if node.children.is_empty() {
                continue;
            }
            let sum: u32 = node
                .children
                .iter()
                .map(|&c| out.tree.nodes[c].visits)
                .sum();
            assert_eq!(
                node.visits,
                1 + sum,
                "[{name}] node {} has N={} but its children sum to {sum}",
                node.id,
                node.visits
            );
        }

        // 2. Q == W / N, exactly.
        for node in &out.tree.nodes {
            assert!(
                node.visits >= 1,
                "[{name}] every node is rolled out on creation"
            );
            let expected = node.total_value / f64::from(node.visits);
            assert_eq!(
                node.mean_value().to_bits(),
                expected.to_bits(),
                "[{name}] Q must be exactly W/N at node {}",
                node.id
            );
            assert!(
                (0.0..=1.0).contains(&node.mean_value()),
                "[{name}] returns are clamped to [0,1], so Q must be too"
            );
        }

        // 3. The root's accumulated value IS the log of returns, in order, bit for bit.
        assert_eq!(
            out.stats.rollout_returns.len(),
            simulations + 1,
            "[{name}] one rollout per simulation, plus the root's"
        );
        let replayed = out
            .stats
            .rollout_returns
            .iter()
            .fold(0.0f64, |acc, &g| acc + g);
        assert_eq!(
            out.tree.root().unwrap().total_value.to_bits(),
            replayed.to_bits(),
            "[{name}] the root's W must be the in-order sum of every rollout return"
        );

        // Every simulation took exactly one root action.
        assert_eq!(out.stats.root_action_history.len(), simulations, "[{name}]");
        // And the tree grew by at most one node per simulation.
        assert!(out.stats.nodes_created <= simulations + 1, "[{name}]");
        assert_eq!(out.stats.nodes_created, out.tree.nodes.len(), "[{name}]");
    }
}

/// The value-side analogue of the visit identity: a parent's accumulated value, minus
/// its children's, is exactly the return of its own creation rollout — which must be a
/// legal return, i.e. in `[0, 1]`.
///
/// A backup that double-counted the expanded node, or that failed to stop at the root,
/// would leave a residue outside that range.
#[test]
fn a_parents_value_residue_is_exactly_one_legal_rollout_return() {
    let world = GoldenPath {
        depth: 4,
        branching: 3,
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(400)
            .with_max_depth(4),
    )
    .search_with("q", &world, &world, &[])
    .unwrap();

    for node in &out.tree.nodes {
        if node.children.is_empty() {
            continue;
        }
        let children_total: f64 = node
            .children
            .iter()
            .map(|&c| out.tree.nodes[c].total_value)
            .sum();
        let residue = node.total_value - children_total;
        assert!(
            (-1e-9..=1.0 + 1e-9).contains(&residue),
            "node {}: W={} minus children's {} leaves {residue}, which is not a legal \
             single rollout return",
            node.id,
            node.total_value,
            children_total
        );
    }
}

/// A terminal node keeps accruing visits with no children — the documented exception
/// to the invariant, and the reason it is stated for *internal* nodes.
#[test]
fn terminal_nodes_accrue_visits_without_children() {
    let world = FlatArms {
        values: vec![0.9, 0.1],
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(100)
            .with_max_depth(1)
            .with_widening(None),
    )
    .search_with("q", &world, &world, &[])
    .unwrap();

    let best = out.best_root_action.unwrap();
    let arm = &out.tree.nodes[best];
    assert!(arm.terminal);
    assert!(arm.children.is_empty());
    assert!(
        arm.visits > 50,
        "the good arm must be re-selected many times: N={}",
        arm.visits
    );
    // `0.9` is not exactly representable, and a mean is that value summed N times then
    // divided by N — so it is `0.9` only up to accumulated rounding, not bit-for-bit.
    assert!(
        (arm.mean_value() - 0.9).abs() < 1e-9,
        "a deterministic arm's mean is its value: {}",
        arm.mean_value()
    );
    assert!(out.tree.check_visit_invariant());
}

/// A generator that dead-ends part way down marks those nodes terminal, and the search
/// still satisfies every invariant.
#[test]
fn dead_end_states_are_terminal_and_do_not_break_the_invariant() {
    // Branch "x" dead-ends immediately; branch "y" continues to full depth.
    struct DeadEnd;
    impl MctsStepGenerator for DeadEnd {
        fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
            if path.len() > 3 {
                return Vec::new();
            }
            if path.last().is_some_and(|s| s.starts_with('x')) {
                return Vec::new(); // dead end, well short of max_depth
            }
            vec![MctsCandidate::new("x"), MctsCandidate::new("y")]
        }
    }
    struct Len;
    impl MctsTerminalEvaluator for Len {
        fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
            (path.len() - 1) as f64 / 3.0
        }
    }

    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(200)
            .with_max_depth(3)
            .with_widening(None),
    )
    .search_with("q", &DeadEnd, &Len, &[])
    .unwrap();

    let dead: Vec<&MctsNode> = out
        .tree
        .nodes
        .iter()
        .filter(|n| n.content == "x" && n.depth < 3)
        .collect();
    assert!(
        !dead.is_empty(),
        "the fixture must actually produce dead ends"
    );
    for node in dead {
        assert!(node.terminal, "a state with no proposals is terminal");
        assert!(node.children.is_empty());
        assert_eq!(node.action_count(), 0);
    }
    assert!(out.tree.check_visit_invariant());
    // The search should prefer the branch that can actually go somewhere.
    assert_eq!(out.best_path[1], "y");
}

// ═════════════════════════════════════════════════════════════════════════════
// (b) Convergence to a known optimum, and the ablation against random descent
// ═════════════════════════════════════════════════════════════════════════════

/// Run `GoldenPath` under `policy` and report `(pv_is_golden, first_optimal_sim,
/// optimal_rate)`.
fn run_golden(
    world: &GoldenPath,
    policy: MctsSelectionPolicy,
    simulations: usize,
    seed: u64,
) -> (bool, Option<usize>, f64) {
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(simulations)
            .with_max_depth(world.depth)
            .with_widening(None)
            .with_selection(policy)
            .with_seed(seed),
    )
    .search_with("q", world, world, &[])
    .unwrap();

    let pv: Vec<String> = out.best_path.iter().skip(1).cloned().collect();
    let pv_is_golden = pv == world.golden_steps();

    let first_optimal = out.stats.rollout_returns.iter().position(|&g| g == 1.0);
    let optimal_count = out
        .stats
        .rollout_returns
        .iter()
        .filter(|&&g| g == 1.0)
        .count();
    let rate = optimal_count as f64 / out.stats.rollout_returns.len() as f64;

    (pv_is_golden, first_optimal, rate)
}

/// **Headline.** On a tree whose optimal leaf is known by construction, `UCT` finds it
/// — and its endorsed reasoning path *is* that leaf's path — once the simulation budget
/// exceeds a stated threshold, and stays there as the budget grows.
///
/// The world: depth 5, branching 3, exactly one correct chain of choices, and the
/// correct choice at every level is the **last** candidate proposed (so no tie-break
/// and no widening order can luck into it). The optimal leaf has value exactly `1.0`.
/// A uniformly random playout reaches it with probability `(1/3)^5 = 1/243`.
#[test]
fn uct_converges_to_the_known_optimal_leaf() {
    let world = GoldenPath {
        depth: 5,
        branching: 3,
    };
    assert!((world.random_hit_rate() - 1.0 / 243.0).abs() < 1e-12);

    let policy = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    };

    // Below the threshold the search has not necessarily converged; at and above it,
    // it must be right, and must *stay* right.
    let converged_budget: usize = 400;
    for simulations in [converged_budget, 800, 1600, 3200] {
        let (golden, first, rate) = run_golden(&world, policy.clone(), simulations, 0x5EED);
        assert!(
            golden,
            "at {simulations} simulations the principal variation must be the golden \
             path (first optimum at {first:?}, optimal-return rate {rate:.3})"
        );
        assert!(
            first.is_some(),
            "the search must actually have reached the optimal leaf at least once"
        );
    }

    // The root action, specifically: the golden first step.
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(converged_budget)
            .with_max_depth(5)
            .with_widening(None),
    )
    .search_with("q", &world, &world, &[])
    .unwrap();
    let action = out.best_root_action.unwrap();
    assert_eq!(
        out.tree.nodes[action].content,
        world.golden_step(0),
        "the chosen first step must be the golden one"
    );
    assert!(out.tree.check_visit_invariant());
}

/// **Headline (the ablation).** `UCT` reaches the optimal leaf far sooner, and far more
/// often, than the *same* Monte-Carlo estimator with the tree policy replaced by
/// uniformly random descent.
///
/// This is the test that distinguishes a search from an expensive random sampler that
/// keeps statistics. Both configurations expand identically, roll out identically,
/// back up identically, and differ in exactly one thing: how selection chooses a child.
///
/// Two readout-free measures, neither of which depends on how the final answer is
/// extracted:
///
/// * **Time to first optimum** — the simulation index at which a rollout first returns
///   `1.0`.
/// * **Optimal-return rate** — the fraction of all rollouts that returned `1.0`. For
///   uniformly random descent this is pinned by the geometry at `(1/3)^5 = 0.41%`;
///   anything a tree policy earns above that, it earned by searching.
#[test]
fn uct_finds_the_optimum_with_far_fewer_simulations_than_random_descent() {
    let world = GoldenPath {
        depth: 5,
        branching: 3,
    };
    let budget = 3200usize;

    let (uct_golden, uct_first, uct_rate) = run_golden(
        &world,
        MctsSelectionPolicy::Uct {
            exploration: SQRT_2,
        },
        budget,
        0x5EED,
    );
    let (rand_golden, rand_first, rand_rate) =
        run_golden(&world, MctsSelectionPolicy::UniformRandom, budget, 0x5EED);

    let uct_first = uct_first.expect("UCT must find the optimal leaf");

    // 1. Time to first optimum. UCT climbs the gradient; random descent has to stumble
    //    onto a 1-in-243 leaf.
    let rand_first = rand_first.expect("random descent does eventually stumble on it");
    // Time-to-first-discovery is the *weaker* of the two signals: UCT still has to
    // explore five levels deep before it can concentrate, so its edge here is real but
    // modest. The optimal-return *rate* below is where the gulf is. A 2x margin is what
    // this deterministic seed actually delivers (73 vs 244); the rate margin is 20x.
    assert!(
        uct_first * 2 < rand_first,
        "UCT must reach the optimum sooner: UCT first hit at simulation {uct_first}, \
         random at {rand_first}"
    );

    // 2. Optimal-return rate, against the rate the geometry hands out for free.
    let free = world.random_hit_rate(); // 1/243 = 0.00412
    assert!(
        rand_rate < 4.0 * free,
        "uniformly random descent cannot beat its own geometry by much: rate \
         {rand_rate:.4} vs the free {free:.4}"
    );
    assert!(
        uct_rate > 20.0 * rand_rate,
        "UCT's optimal-return rate must dwarf random descent's: {uct_rate:.4} vs \
         {rand_rate:.4}"
    );
    assert!(
        uct_rate > 0.10,
        "UCT should spend a large share of its budget on the optimal leaf once it has \
         found it, got {uct_rate:.4}"
    );

    // 3. And the answer itself.
    assert!(
        uct_golden,
        "UCT's principal variation must be the golden path"
    );
    assert!(
        !rand_golden,
        "random descent's visit counts carry no signal, so its principal variation \
         must not be the golden path (that would be a 1-in-243 coincidence)"
    );
}

/// The same ablation, expressed the way the claim is usually stated: **how many
/// simulations does each policy need** before its endorsed path is the optimal one, and
/// stays optimal as the budget grows.
#[test]
fn uct_needs_a_smaller_budget_than_random_descent_to_endorse_the_optimal_path() {
    let world = GoldenPath {
        depth: 5,
        branching: 3,
    };
    let ladder = [50usize, 100, 200, 400, 800, 1600, 3200];

    // The smallest budget on the ladder from which the policy is golden at that budget
    // and at every larger one.
    let smallest_stable = |policy: MctsSelectionPolicy| -> Option<usize> {
        let golden: Vec<bool> = ladder
            .iter()
            .map(|&n| run_golden(&world, policy.clone(), n, 0x5EED).0)
            .collect();
        (0..ladder.len())
            .find(|&i| golden[i..].iter().all(|&g| g))
            .map(|i| ladder[i])
    };

    let uct = smallest_stable(MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    })
    .expect("UCT must converge somewhere on this ladder");
    let random = smallest_stable(MctsSelectionPolicy::UniformRandom);

    assert!(
        uct <= 400,
        "UCT should converge within 400 simulations, needed {uct}"
    );
    assert!(
        random.is_none() || random.unwrap() > 4 * uct,
        "random descent must need a budget at least 4x larger (or never converge on \
         this ladder): UCT {uct}, random {random:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (c) Sublinear regret
// ═════════════════════════════════════════════════════════════════════════════

/// Cumulative regret of the root's action sequence over the first `n` simulations.
fn cumulative_regret(world: &FlatArms, tree: &MctsTree, history: &[usize], n: usize) -> f64 {
    let best = world.best();
    history
        .iter()
        .take(n)
        .map(|&id| best - world.value_of(&tree.nodes[id].content))
        .sum()
}

/// **Headline.** `UCT`'s cumulative regret is sublinear: `regret(4n) < 4 * regret(n)`.
///
/// Theory (Kocsis & Szepesvári, via `UCB1`) says a suboptimal arm is pulled
/// `O(c^2 * ln n / gap^2)` times, so cumulative regret grows like
/// `c^2 * ln n * sum_a (1/gap_a)` — i.e. **logarithmically**. Quadrupling the budget
/// should therefore multiply regret by only `ln(4n)/ln(n)`, which at `n = 500` is about
/// `1.22`, not `4`.
///
/// The contrast is the point. A policy with **linear** regret — one that keeps pulling
/// bad arms at a constant rate, i.e. one that is not learning — multiplies its regret
/// by exactly `4`. Uniformly random descent is that policy, and it is measured here
/// alongside, on the same arms, so the claim is a comparison and not an assertion about
/// a number in a vacuum.
///
/// The arms are deterministic (`0.9, 0.6, 0.5, 0.4, 0.3`), so every pull of a
/// suboptimal arm is *pure* regret with no noise to hide in.
#[test]
fn uct_cumulative_regret_is_sublinear() {
    let world = FlatArms {
        values: vec![0.9, 0.6, 0.5, 0.4, 0.3],
    };
    let n = 500usize;
    let big = 4 * n;

    let measure = |policy: MctsSelectionPolicy| -> (f64, f64) {
        let out = MctsEngine::new(
            MctsConfig::default()
                .with_simulations(big)
                .with_max_depth(1)
                .with_widening(None)
                .with_selection(policy)
                .with_seed(0x5EED),
        )
        .search_with("q", &world, &world, &[])
        .unwrap();
        assert!(out.tree.check_visit_invariant());
        let history = &out.stats.root_action_history;
        (
            cumulative_regret(&world, &out.tree, history, n),
            cumulative_regret(&world, &out.tree, history, big),
        )
    };

    let (uct_n, uct_4n) = measure(MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    });
    let (rand_n, rand_4n) = measure(MctsSelectionPolicy::UniformRandom);

    assert!(uct_n > 0.0, "some exploration must happen");
    let uct_ratio = uct_4n / uct_n;
    let rand_ratio = rand_4n / rand_n;

    // The required bound.
    assert!(
        uct_ratio < 4.0,
        "UCT's regret must be sublinear: regret({big})/regret({n}) = {uct_ratio:.3} \
         (regret {uct_n:.1} -> {uct_4n:.1})"
    );
    // A much tighter one, which is what logarithmic growth actually looks like:
    // ln(2000)/ln(500) = 1.223.
    assert!(
        uct_ratio < 2.0,
        "logarithmic growth should give a ratio near ln(4n)/ln(n) = 1.22, got \
         {uct_ratio:.3}"
    );

    // The linear-regret baseline, on the same arms: ~4x, on the nose.
    assert!(
        rand_ratio > 3.5,
        "uniformly random descent has linear regret, so its ratio must be ~4: got \
         {rand_ratio:.3}"
    );
    assert!(
        uct_4n < rand_4n / 2.0,
        "UCT must accumulate far less regret in absolute terms: {uct_4n:.1} vs \
         {rand_4n:.1}"
    );

    // And it must actually converge on the best arm.
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(big)
            .with_max_depth(1)
            .with_widening(None),
    )
    .search_with("q", &world, &world, &[])
    .unwrap();
    assert_eq!(out.best_path[1], "arm0", "arm0 has the highest value, 0.9");
}

/// The per-arm pull counts must follow `UCB1`'s prediction: a suboptimal arm is pulled
/// about `c^2 * ln(n) / gap^2` times, so **the bigger the gap, the fewer the pulls** —
/// strictly monotone in the gap.
#[test]
fn uct_allocates_pulls_in_inverse_proportion_to_the_squared_gap() {
    let world = FlatArms {
        values: vec![0.9, 0.6, 0.5, 0.4, 0.3],
    };
    let out = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(4000)
            .with_max_depth(1)
            .with_widening(None),
    )
    .search_with("q", &world, &world, &[])
    .unwrap();

    let root = out.tree.root().unwrap();
    let mut pulls: Vec<(f64, u32)> = root
        .children
        .iter()
        .map(|&id| {
            let node = &out.tree.nodes[id];
            (world.value_of(&node.content), node.visits)
        })
        .collect();
    pulls.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // The best arm dominates.
    assert_eq!(pulls[0].0, 0.9);
    assert!(
        pulls[0].1 > pulls[1].1 * 4,
        "the optimal arm must be pulled far more than the runner-up: {pulls:?}"
    );

    // And among the suboptimal arms, pulls fall monotonically as the gap widens.
    for window in pulls[1..].windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "a wider gap must earn fewer pulls: {pulls:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// (e) Determinism
// ═════════════════════════════════════════════════════════════════════════════

/// **Headline.** The same seed reproduces the same tree, bit for bit — every node,
/// every visit count, every accumulated value.
#[test]
fn same_seed_reproduces_the_tree_bit_for_bit() {
    let world = GoldenPath {
        depth: 5,
        branching: 3,
    };
    let config = MctsConfig::default()
        .with_simulations(500)
        .with_max_depth(5)
        .with_rollout(MctsRolloutPolicy::PriorWeighted)
        .with_seed(0xC0FFEE);

    let a = MctsEngine::new(config.clone())
        .search_with("q", &world, &world, &[])
        .unwrap();
    let b = MctsEngine::new(config.clone())
        .search_with("q", &world, &world, &[])
        .unwrap();

    assert_eq!(a.tree, b.tree, "the trees must be structurally identical");
    assert_eq!(a.stats, b.stats);
    assert_eq!(a.answer, b.answer);

    // `PartialEq` on f64 is not bit equality (it conflates 0.0 and -0.0), so check the
    // bits explicitly — that is the contract the module actually promises.
    assert_eq!(a.tree.nodes.len(), b.tree.nodes.len());
    for (x, y) in a.tree.nodes.iter().zip(&b.tree.nodes) {
        assert_eq!(x.total_value.to_bits(), y.total_value.to_bits());
        assert_eq!(x.prior.to_bits(), y.prior.to_bits());
        assert_eq!(x.visits, y.visits);
        assert_eq!(x.content, y.content);
        assert_eq!(x.children, y.children);
    }
    for (x, y) in a.stats.rollout_returns.iter().zip(&b.stats.rollout_returns) {
        assert_eq!(x.to_bits(), y.to_bits());
    }

    // A different seed must actually change the search — otherwise the seed is a lie.
    let c = MctsEngine::new(config.with_seed(0xBEEF))
        .search_with("q", &world, &world, &[])
        .unwrap();
    assert_ne!(
        c.stats.rollout_returns, a.stats.rollout_returns,
        "a different seed must produce a different rollout stream"
    );
}

/// `UniformRandom` selection is seeded too — the ablation baseline has to be
/// reproducible or it cannot be a baseline.
#[test]
fn the_random_ablation_baseline_is_also_reproducible() {
    let world = GoldenPath {
        depth: 4,
        branching: 3,
    };
    let config = MctsConfig::default()
        .with_simulations(200)
        .with_max_depth(4)
        .with_selection(MctsSelectionPolicy::UniformRandom)
        .with_seed(11);

    let a = MctsEngine::new(config.clone())
        .search_with("q", &world, &world, &[])
        .unwrap();
    let b = MctsEngine::new(config)
        .search_with("q", &world, &world, &[])
        .unwrap();
    assert_eq!(a.tree, b.tree);
    assert_eq!(a.stats.root_action_history, b.stats.root_action_history);
}
