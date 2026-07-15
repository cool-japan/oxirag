//! Selection-policy tests: exact UCT/PUCT arithmetic, the reductions, tie-breaks.
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
// (a) Exact UCT arithmetic
// ═════════════════════════════════════════════════════════════════════════════

/// **Headline.** The `UCT` score of every child of the canonical fixture, checked
/// against constants computed offline, by `to_bits` equality — not "close to", *the
/// same bits*.
///
/// The derivation, for `c = sqrt(2) = 1.4142135623730951` and `ln(6) = 1.791759469228055`:
///
/// ```text
/// child a:  Q = 2.4 / 3 = 0.7999999999999999   <- note: NOT 0.8; 2.4 is inexact in binary
///           sqrt(ln(6) / 3) = sqrt(0.5972531564093516) = 0.7728215553472558
///           bonus = 1.4142135623730951 * 0.7728215553472558 = 1.0929347248663588
///           UCT   = 0.7999999999999999 + 1.0929347248663588 = 1.8929347248663588
///
/// child b:  Q = 0.5 / 1 = 0.5
///           sqrt(ln(6) / 1) = 1.3385661990458504
///           bonus = 1.4142135623730951 * 1.3385661990458504 = 1.8930184728248456
///           UCT   = 0.5 + 1.8930184728248456 = 2.3930184728248456
///
/// child c:  Q = 0.2 / 1 = 0.2
///           bonus = same as b (same visit count) = 1.8930184728248456
///           UCT   = 0.2 + 1.8930184728248456 = 2.0930184728248458
/// ```
///
/// The point of *these* numbers: the winner is **b**, the mediocre, barely-tried
/// child — not **a**, which has the best mean by a wide margin and has been visited
/// three times as often. Even **c**, the *worst* child by mean, outranks `a`. That is
/// the exploration term doing its job, and it means an implementation that quietly
/// dropped it, or that inverted `N(s)` and `N(s,a)`, cannot pass this test while still
/// looking like a working search.
#[test]
fn uct_scores_match_hand_computed_arithmetic_bit_for_bit() {
    let tree = canonical_fixture();
    let parent = &tree.nodes[0];
    let policy = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    };

    // Sanity: the fixture is a tree the engine could have built.
    assert!(
        tree.check_visit_invariant(),
        "the fixture must itself satisfy N(s) = 1 + sum_c N(c)"
    );

    let expected: [(usize, f64, f64); 3] = [
        (1, 0.7999999999999999, 1.8929347248663588),
        (2, 0.5, 2.3930184728248456),
        (3, 0.2, 2.0930184728248458),
    ];

    for (id, expected_q, expected_uct) in expected {
        let child = &tree.nodes[id];
        assert_eq!(
            child.mean_value().to_bits(),
            expected_q.to_bits(),
            "Q of child {id}: got {}, want {expected_q}",
            child.mean_value()
        );
        let score = policy.score(parent, child);
        assert_eq!(
            score.to_bits(),
            expected_uct.to_bits(),
            "UCT of child {id}: got {score}, want {expected_uct}"
        );
    }

    // The whole point of the fixture: exploration overrides exploitation here.
    assert_eq!(
        argmax_child(&tree, &policy),
        2,
        "UCT must pick the under-explored child b, not the best-mean child a"
    );
    let score_a = policy.score(parent, &tree.nodes[1]);
    let score_c = policy.score(parent, &tree.nodes[3]);
    assert!(
        score_c > score_a,
        "even the worst-mean child must outrank the over-visited one here: \
         c={score_c} vs a={score_a}"
    );
}

/// `c = 0` removes the bonus entirely, and `UCT` collapses to greedy exploitation of
/// `Q`. This is the other end of the same dial, and it pins the sign and placement of
/// the exploration term.
#[test]
fn zero_exploration_reduces_uct_to_greedy_exploitation() {
    let tree = canonical_fixture();
    let greedy = MctsSelectionPolicy::Uct { exploration: 0.0 };
    for id in [1usize, 2, 3] {
        let child = &tree.nodes[id];
        assert_eq!(
            greedy.score(&tree.nodes[0], child).to_bits(),
            child.mean_value().to_bits(),
            "with c = 0 the score must be exactly Q"
        );
    }
    assert_eq!(
        argmax_child(&tree, &greedy),
        1,
        "greedy must pick the best-mean child a — the exact opposite of UCT's choice"
    );
}

/// The bonus must shrink as evidence accumulates: `1/sqrt(N(s,a))`, monotonically.
#[test]
fn uct_bonus_decays_with_child_visits_and_grows_with_parent_visits() {
    let policy = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    };
    let parent = make_node(0, 1000, 0.0, 1.0, 4);

    // Same Q, increasing visits => strictly decreasing score.
    let mut previous = f64::INFINITY;
    for visits in [1u32, 2, 4, 8, 64, 512] {
        let child = make_node(1, visits, 0.5 * f64::from(visits), 0.25, 0);
        assert_eq!(child.mean_value(), 0.5, "held Q fixed at 0.5");
        let score = policy.score(&parent, &child);
        assert!(
            score < previous,
            "bonus must strictly decay with visits: N={visits} gave {score}, previous {previous}"
        );
        previous = score;
    }

    // Same child, more parent evidence => the bonus grows (like sqrt(ln N)).
    let child = make_node(1, 4, 2.0, 0.25, 0);
    let small = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    }
    .score(&make_node(0, 10, 0.0, 1.0, 4), &child);
    let large = policy.score(&make_node(0, 10_000, 0.0, 1.0, 4), &child);
    assert!(
        large > small,
        "an unvisited-relative-to-parent child must look more attractive as the \
         parent gathers evidence: {large} vs {small}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// (f) PUCT, PriorUct, and the reduction that does — and does not — hold
// ═════════════════════════════════════════════════════════════════════════════

/// **Headline.** [`MctsSelectionPolicy::PriorUct`] with a **uniform** prior is
/// [`MctsSelectionPolicy::Uct`], bit for bit, over a grid of exploration constants,
/// visit counts and values — for every candidate count `K` below `49`.
///
/// `49` is not arbitrary. The reduction rests on `K * fl(1/K) == 1.0`, and `49` is the
/// first `K` for which that is false in `f64` (`49 * fl(1/49) = 0.9999999999999999`);
/// there are 82 such `K` below 1000. The next test pins what happens there.
#[test]
fn prior_uct_reduces_to_uct_exactly_under_a_uniform_prior() {
    let mut checked = 0usize;
    for k in 1..49usize {
        let uniform_prior = 1.0 / k as f64;
        for &c in &[SQRT_2, 0.0, 0.5, 1.0, 2.0, 3.7] {
            let uct = MctsSelectionPolicy::Uct { exploration: c };
            let prior_uct = MctsSelectionPolicy::PriorUct { exploration: c };
            for parent_visits in [2u32, 3, 7, 50, 999] {
                for child_visits in [1u32, 2, 6, 49] {
                    if child_visits >= parent_visits {
                        continue;
                    }
                    for &total in &[0.0, 0.5, 2.4, 7.0] {
                        let parent = make_node(0, parent_visits, 0.0, 1.0, k);
                        let child = make_node(1, child_visits, total, uniform_prior, 0);
                        let a = uct.score(&parent, &child);
                        let b = prior_uct.score(&parent, &child);
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "K={k} c={c} N(s)={parent_visits} N(s,a)={child_visits} W={total}: \
                             UCT={a} but PriorUct={b}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 4000, "grid should be substantial, was {checked}");
}

/// The documented **exception**, pinned rather than papered over: at `K = 49` the
/// uniform prior no longer round-trips, and the two scores can differ — by one ulp.
///
/// This test exists so that the day someone "fixes" the reduction by inserting an
/// epsilon comparison, they have to come here and read why they should not.
#[test]
fn prior_uct_differs_from_uct_by_one_ulp_at_the_documented_pathological_k() {
    let k = 49usize;
    let uniform_prior = 1.0 / k as f64;
    let relative = uniform_prior * k as f64;
    assert_ne!(
        relative, 1.0,
        "49 * fl(1/49) is the first K where the reciprocal fails to round-trip"
    );
    assert_eq!(relative, 0.9999999999999999);

    // The exact grid point at which the discrepancy surfaces.
    let parent = make_node(0, 2, 0.0, 1.0, k);
    let child = make_node(1, 1, 0.0, uniform_prior, 0);
    let uct = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    }
    .score(&parent, &child);
    let prior_uct = MctsSelectionPolicy::PriorUct {
        exploration: SQRT_2,
    }
    .score(&parent, &child);

    let ulps = (uct.to_bits() as i64 - prior_uct.to_bits() as i64).abs();
    assert_eq!(
        ulps, 1,
        "expected exactly the documented one-ulp gap, got {ulps} ({uct} vs {prior_uct})"
    );

    // And it is never worse than 2 ulps anywhere on the grid, for any K.
    let mut worst = 0i64;
    for k in 1..=200usize {
        let prior = 1.0 / k as f64;
        for &c in &[SQRT_2, 1.0, 2.0, 3.7] {
            for parent_visits in [2u32, 5, 40, 900] {
                for child_visits in [1u32, 3, 39] {
                    if child_visits >= parent_visits {
                        continue;
                    }
                    let parent = make_node(0, parent_visits, 0.0, 1.0, k);
                    let child = make_node(1, child_visits, 1.0, prior, 0);
                    let a = MctsSelectionPolicy::Uct { exploration: c }.score(&parent, &child);
                    let b = MctsSelectionPolicy::PriorUct { exploration: c }.score(&parent, &child);
                    worst = worst.max((a.to_bits() as i64 - b.to_bits() as i64).abs());
                }
            }
        }
    }
    assert!(
        worst <= 2,
        "the reduction must never be off by more than 2 ulps, saw {worst}"
    );
}

/// **The negative claim, asserted.** `AlphaZero`'s `PUCT` does **not** reduce to `UCT`
/// under a uniform prior — for any exploration constant.
///
/// This is not a subtle numerical point. On the canonical fixture, with a uniform
/// prior and the very same `c = sqrt(2)`, the two policies choose **different
/// children**: `UCT` takes `b` (the under-explored one), `PUCT` takes `a` (the
/// best-mean one). `PUCT`'s bonus decays like `1/(1 + N(s,a))`, which at `N(s,a) = 3`
/// has already collapsed to a quarter of its value at `N(s,a) = 1`, while `UCT`'s
/// `1/sqrt(N(s,a))` has only fallen to `0.58` of it — so `a`'s three visits buy it far
/// more relief from exploration pressure under `PUCT` than under `UCT`.
///
/// A uniform prior makes `PUCT`'s bonus *constant across a node's children*, which is
/// emphatically not the same thing as making it `UCT`'s bonus. Any claim that "`PUCT`
/// with a flat prior is just `UCT`" is false, and this is the counterexample.
#[test]
fn alphazero_puct_does_not_reduce_to_uct_under_a_uniform_prior() {
    let tree = canonical_fixture();
    let parent = &tree.nodes[0];

    let uct = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    };
    let puct = MctsSelectionPolicy::Puct {
        exploration: SQRT_2,
    };

    // Every child carries the uniform prior 1/3 in the canonical fixture.
    for id in [1usize, 2, 3] {
        assert_eq!(tree.nodes[id].prior, 1.0 / 3.0);
    }

    // They disagree on the answer, not merely on the numbers.
    assert_eq!(
        argmax_child(&tree, &uct),
        2,
        "UCT picks the under-explored b"
    );
    assert_eq!(argmax_child(&tree, &puct), 1, "PUCT picks the best-mean a");

    // The PUCT scores, derived offline:
    //   bonus = c * P * sqrt(N(s)) / (1 + N(s,a)),  c = sqrt(2), P = 1/3, N(s) = 6
    //   a: 1.4142135623730951 * (1/3) * 2.449489742783178 / 4 = 0.28867513459481287
    //   b: same, / 2                                          = 0.5773502691896257
    //   c: same, / 2                                          = 0.5773502691896257
    let expected: [(usize, f64); 3] = [
        (1, 1.088675134594813),
        (2, 1.0773502691896257),
        (3, 0.7773502691896257),
    ];
    for (id, want) in expected {
        let got = puct.score(parent, &tree.nodes[id]);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "PUCT of child {id}: got {got}, want {want}"
        );
    }

    // The disagreement is not just in the argmax: the two policies induce genuinely
    // different *total orders* on the children. At c = sqrt(2), UCT ranks b > c > a
    // (exploration-dominated), while PUCT ranks a > b > c (exploitation-dominated). Two
    // different orderings cannot be a monotone rescaling of one another — and "reduces
    // to" would require exactly such a rescaling. This is the precise sense in which
    // PUCT does not reduce to UCT.
    let ordering = |policy: &MctsSelectionPolicy| -> Vec<usize> {
        let mut ids = vec![1usize, 2, 3];
        ids.sort_by(|&x, &y| {
            policy
                .score(parent, &tree.nodes[y])
                .partial_cmp(&policy.score(parent, &tree.nodes[x]))
                .unwrap()
        });
        ids
    };
    assert_eq!(ordering(&uct), vec![2, 3, 1], "UCT ranks b > c > a");
    assert_eq!(ordering(&puct), vec![1, 2, 3], "PUCT ranks a > b > c");
    assert_ne!(
        ordering(&uct),
        ordering(&puct),
        "the orderings differ, so no monotone rescaling maps one policy to the other"
    );

    // For honesty about the boundary: at a *large enough* constant PUCT does become
    // exploration-dominated and picks b too — so the claim is emphatically NOT "PUCT
    // never picks what UCT picks", it is "at the matched constant they differ, because
    // the bonuses are differently shaped". The crossover is where a's exploitation edge
    // (0.3) equals its exploration deficit, at c ~= 1.47 > sqrt(2).
    assert_eq!(
        argmax_child(&tree, &MctsSelectionPolicy::Puct { exploration: 3.0 }),
        2,
        "at c = 3.0 PUCT is exploration-dominated and picks b, like UCT — the two \
         agreeing at some c is not the same as one reducing to the other"
    );
}

/// [`MctsSelectionPolicy::uses_prior`] and [`MctsSelectionPolicy::exploration`] must
/// tell the truth about the variants.
#[test]
fn selection_policy_accessors_report_the_right_shape() {
    assert!(!MctsSelectionPolicy::Uct { exploration: 1.0 }.uses_prior());
    assert!(MctsSelectionPolicy::PriorUct { exploration: 1.0 }.uses_prior());
    assert!(MctsSelectionPolicy::Puct { exploration: 1.0 }.uses_prior());
    assert!(!MctsSelectionPolicy::UniformRandom.uses_prior());

    assert_eq!(
        MctsSelectionPolicy::Puct { exploration: 2.5 }.exploration(),
        Some(2.5)
    );
    assert_eq!(MctsSelectionPolicy::UniformRandom.exploration(), None);
    assert_eq!(MctsSelectionPolicy::default().exploration(), Some(SQRT_2));
}

// ═════════════════════════════════════════════════════════════════════════════
// Final action selection: max-visit, and the tie-break
// ═════════════════════════════════════════════════════════════════════════════

/// The final action is the **most-visited** child, even when another child has a
/// strictly better mean. This is the whole "robust child" argument, and it is a real
/// behavioural choice that a test must pin: the two rules disagree here, and the code
/// must implement the first.
#[test]
fn final_selection_takes_max_visits_not_max_value() {
    let mut parent = make_node(0, 1 + 400 + 2, 0.0, 1.0, 2);
    parent.children = vec![1, 2];
    let tree = MctsTree {
        nodes: vec![
            parent,
            // Visited 400 times, converged mean 0.60.
            make_node(1, 400, 240.0, 0.5, 0),
            // Visited twice, got lucky twice: mean 0.95.
            make_node(2, 2, 1.9, 0.5, 0),
        ],
    };
    assert!(tree.check_visit_invariant());
    assert_eq!(tree.nodes[1].mean_value(), 0.6);
    assert_eq!(tree.nodes[2].mean_value(), 0.95);

    assert_eq!(
        tree.best_child_by_visits(0),
        Some(1),
        "must take the 400-visit child with the *lower* mean, not the lucky 2-visit one"
    );
}

/// **The `max_by_key` trap.** On an exact tie of both visits and value, the documented
/// rule is *lowest id wins*. `children.iter().max_by_key(|c| c.visits)` returns the
/// **last** maximal element, so an implementation written that way silently returns
/// child `3` here — a reproducible, plausible, wrong answer that depends on nothing
/// but arena fill order.
#[test]
fn visit_ties_are_broken_by_value_then_by_lowest_id() {
    // All three tied on visits; child 2 has the best mean => it wins on rule 2.
    let mut parent = make_node(0, 1 + 15, 0.0, 1.0, 3);
    parent.children = vec![1, 2, 3];
    let by_value = MctsTree {
        nodes: vec![
            parent.clone(),
            make_node(1, 5, 2.0, 1.0 / 3.0, 0), // Q = 0.40
            make_node(2, 5, 4.5, 1.0 / 3.0, 0), // Q = 0.90  <- best
            make_node(3, 5, 2.0, 1.0 / 3.0, 0), // Q = 0.40
        ],
    };
    assert!(by_value.check_visit_invariant());
    assert_eq!(
        by_value.best_child_by_visits(0),
        Some(2),
        "equal visits must be broken by the higher mean value"
    );

    // Fully tied on visits *and* value => lowest id wins.
    let fully_tied = MctsTree {
        nodes: vec![
            parent,
            make_node(1, 5, 3.5, 1.0 / 3.0, 0), // Q = 0.7
            make_node(2, 5, 3.5, 1.0 / 3.0, 0), // Q = 0.7
            make_node(3, 5, 3.5, 1.0 / 3.0, 0), // Q = 0.7
        ],
    };
    let winner = fully_tied.best_child_by_visits(0);
    assert_eq!(
        winner,
        Some(1),
        "a total tie must resolve to the lowest id (the earliest-proposed candidate)"
    );
    assert_ne!(
        winner,
        Some(3),
        "returning the LAST maximal child is the max_by_key bug this rule exists to avoid"
    );
}

/// The selection argmax has the same tie-break, and for the same reason: on a total
/// tie of score, the earliest-proposed child is descended into.
#[test]
fn selection_ties_resolve_to_the_earliest_proposed_child() {
    // Three children, identical statistics => identical UCT scores.
    let mut parent = make_node(0, 1 + 6, 0.0, 1.0, 3);
    parent.children = vec![1, 2, 3];
    let tree = MctsTree {
        nodes: vec![
            parent,
            make_node(1, 2, 1.0, 1.0 / 3.0, 0),
            make_node(2, 2, 1.0, 1.0 / 3.0, 0),
            make_node(3, 2, 1.0, 1.0 / 3.0, 0),
        ],
    };
    let policy = MctsSelectionPolicy::Uct {
        exploration: SQRT_2,
    };
    let s1 = policy.score(&tree.nodes[0], &tree.nodes[1]);
    let s2 = policy.score(&tree.nodes[0], &tree.nodes[2]);
    let s3 = policy.score(&tree.nodes[0], &tree.nodes[3]);
    assert_eq!(s1.to_bits(), s2.to_bits());
    assert_eq!(s2.to_bits(), s3.to_bits());
    assert_eq!(
        argmax_child(&tree, &policy),
        1,
        "the strict-greater fold must keep the first maximal child"
    );
}

/// The principal variation is a max-visit descent, and it stops at a leaf.
#[test]
fn principal_variation_descends_by_visits_to_a_leaf() {
    let world = FlatArms {
        values: vec![0.1, 0.9, 0.2],
    };
    let engine = MctsEngine::new(
        MctsConfig::default()
            .with_simulations(150)
            .with_max_depth(1)
            .with_widening(None),
    );
    let out = engine.search_with("q", &world, &world, &[]).unwrap();

    let pv = out.tree.principal_variation();
    assert_eq!(pv.len(), 2, "root plus one arm");
    assert_eq!(pv[0], 0);
    assert_eq!(
        out.tree.nodes[pv[1]].content, "arm1",
        "the PV must end on the best arm"
    );
    assert_eq!(out.best_root_action, Some(pv[1]));
    assert_eq!(out.best_path.len(), 2);
    assert_eq!(out.best_path[1], "arm1");
}
