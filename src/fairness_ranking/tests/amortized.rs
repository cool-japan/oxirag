//! Tests for amortized equity of attention.
//!
//! The headline is [`amortization_converges_where_a_single_ranking_cannot`]: over
//! a sequence of queries the cross-group attention-per-relevance gap collapses,
//! while the greedy relevance baseline's gap does not. A single fair ranking
//! cannot do this — that is the whole reason the construction spans a sequence.

use crate::fairness_ranking::amortized::EquityOfAttention;
use crate::fairness_ranking::assignment::solve_assignment;
use crate::fairness_ranking::exposure::exposure_at_rank;
use crate::fairness_ranking::rng::FairnessRng;
use crate::fairness_ranking::types::GroupId;

/// Six subjects of **equal relevance**, three per group, three positions per page.
/// This is Biega et al.'s motivating case: no single ranking can be fair (someone
/// must take the smaller payout), so any fairness must come from amortization
/// across the sequence.
fn groups() -> Vec<GroupId> {
    vec![
        GroupId(0),
        GroupId(0),
        GroupId(0),
        GroupId(1),
        GroupId(1),
        GroupId(1),
    ]
}

// ── (e) amortization actually amortizes ──────────────────────────────────────

#[test]
fn amortization_converges_where_a_single_ranking_cannot() {
    let relevances = vec![1.0; 6]; // all equal, every query
    let queries = 50;

    // Pure equity (lambda = 0), amortized across the sequence.
    let mut amortized = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");
    // The greedy relevance ranker, folded into the identical accounting.
    let mut greedy = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");

    let mut amortized_gap_at_1 = 0.0;
    let mut greedy_gap_at_1 = 0.0;

    for query in 1..=queries {
        amortized.serve(&relevances).expect("serves");
        greedy.serve_by_relevance(&relevances).expect("serves");
        if query == 1 {
            amortized_gap_at_1 = amortized.group_gap();
            greedy_gap_at_1 = greedy.group_gap();
        }
    }

    let amortized_gap_at_50 = amortized.group_gap();
    let greedy_gap_at_50 = greedy.group_gap();

    println!(
        "amortized group gap: T=1 {amortized_gap_at_1:.6} -> T=50 {amortized_gap_at_50:.6}\n\
         greedy    group gap: T=1 {greedy_gap_at_1:.6} -> T=50 {greedy_gap_at_50:.6}\n\
         amortized individual inequity at T=50: {:.6}",
        amortized.individual_inequity()
    );

    // The amortized sequence converges toward parity: the gap at T = 50 is a small
    // fraction of the gap it (necessarily) had at T = 1.
    assert!(
        amortized_gap_at_50 < 0.25 * amortized_gap_at_1,
        "amortized gap should collapse: T=1 {amortized_gap_at_1}, T=50 {amortized_gap_at_50}"
    );

    // The greedy baseline does NOT converge: it shows the same three subjects
    // every query, so one group keeps all the exposure and its gap stays large.
    assert!(
        greedy_gap_at_50 > 0.8 * greedy_gap_at_1,
        "greedy gap should persist: T=1 {greedy_gap_at_1}, T=50 {greedy_gap_at_50}"
    );

    // And amortization ends far fairer than greedy — the entire point.
    assert!(
        amortized_gap_at_50 < 0.2 * greedy_gap_at_50,
        "amortized ({amortized_gap_at_50}) must end much fairer than greedy ({greedy_gap_at_50})"
    );
}

#[test]
fn individual_inequity_falls_over_the_sequence() {
    // sum_i |A_i - R_i| — Biega's L1 objective — must trend down as the amortized
    // sequence lengthens.
    let relevances = vec![1.0; 6];
    let mut equity = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");

    equity.serve(&relevances).expect("serves");
    let inequity_early = equity.individual_inequity();
    for _ in 0..49 {
        equity.serve(&relevances).expect("serves");
    }
    let inequity_late = equity.individual_inequity();

    // Per-query inequity: the cumulative L1 grows sublinearly, so its per-query
    // average must shrink.
    let per_query_early = inequity_early / 1.0;
    let per_query_late = inequity_late / 50.0;
    println!(
        "individual inequity per query: early {per_query_early:.6} -> late {per_query_late:.6}"
    );
    assert!(per_query_late < per_query_early);
}

// ── the quality dial ─────────────────────────────────────────────────────────

#[test]
fn lambda_trades_equity_for_quality() {
    // lambda = 0 is pure equity; lambda = 1 is pure relevance (a plain sort).
    // Increasing lambda must not *improve* fairness and must not *worsen* utility.
    let relevances = vec![0.9, 0.7, 0.5, 0.8, 0.6, 0.4];
    let queries = 30;

    let mut equity_gap = 0.0;
    let mut quality_gap = 0.0;
    let mut equity_ndcg = 0.0;
    let mut quality_ndcg = 0.0;

    for (lambda, gap, ndcg) in [
        (0.0, &mut equity_gap, &mut equity_ndcg),
        (1.0, &mut quality_gap, &mut quality_ndcg),
    ] {
        let mut sequence = EquityOfAttention::new(groups(), 2, 3, lambda).expect("valid");
        for _ in 0..queries {
            sequence.serve(&relevances).expect("serves");
        }
        let report = sequence.report();
        *gap = report.group_gap;
        *ndcg = report.mean_ndcg;
    }

    println!(
        "lambda=0: gap {equity_gap:.6}, nDCG {equity_ndcg:.6}\n\
         lambda=1: gap {quality_gap:.6}, nDCG {quality_ndcg:.6}"
    );
    // Pure relevance yields the highest utility...
    assert!(quality_ndcg >= equity_ndcg - 1e-9);
    // ...and pure equity yields the smallest (or equal) group gap.
    assert!(equity_gap <= quality_gap + 1e-9);
}

// ── lambda = 1 is exactly the relevance sort ─────────────────────────────────

#[test]
fn lambda_one_assignment_equals_the_relevance_sort() {
    // At lambda = 1 the cost is -r_i * a_j; by the rearrangement inequality the
    // minimizer pairs the largest attention with the largest relevance, i.e. the
    // ranking is the relevance sort. Check the assignment solver against that
    // closed form directly.
    let relevances = [0.2, 0.9, 0.5, 0.7, 0.1, 0.8];
    let n = relevances.len();
    let positions = 3;

    let attention: Vec<f64> = {
        let raw: Vec<f64> = (0..positions).map(exposure_at_rank).collect();
        let total: f64 = raw.iter().sum();
        raw.iter().map(|value| value / total).collect()
    };
    let shares: Vec<f64> = {
        let total: f64 = relevances.iter().sum();
        relevances.iter().map(|value| value / total).collect()
    };

    // Cost = -share_i * attention_j (the lambda = 1 term).
    let mut cost = vec![0.0; positions * n];
    for j in 0..positions {
        for i in 0..n {
            cost[j * n + i] = -shares[i] * attention[j];
        }
    }
    let solution = solve_assignment(&cost, positions, n).expect("ok");

    // The relevance sort: the three highest-relevance subjects, best first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| relevances[b].partial_cmp(&relevances[a]).unwrap());
    let expected = &order[..positions];

    assert_eq!(
        solution.column_of_row(),
        expected,
        "lambda=1 must reproduce the relevance sort"
    );
}

// ── determinism and guards ───────────────────────────────────────────────────

#[test]
fn serving_is_deterministic() {
    let relevances = vec![0.5, 0.9, 0.1, 0.7, 0.3, 0.6];
    let mut a = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");
    let mut b = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");
    for _ in 0..10 {
        assert_eq!(
            a.serve(&relevances).expect("ok"),
            b.serve(&relevances).expect("ok")
        );
    }
    assert_eq!(a.subject_attention(), b.subject_attention());
}

#[test]
fn zero_relevance_query_is_rejected() {
    let mut sequence = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");
    let all_zero = vec![0.0; 6];
    assert!(sequence.serve(&all_zero).is_err());
}

#[test]
fn construction_guards() {
    // More positions than subjects.
    assert!(EquityOfAttention::new(vec![GroupId(0), GroupId(1)], 2, 5, 0.0).is_err());
    // lambda outside [0, 1].
    assert!(EquityOfAttention::new(groups(), 2, 3, 1.5).is_err());
    // Out-of-range group label.
    assert!(EquityOfAttention::new(vec![GroupId(0), GroupId(9)], 2, 1, 0.0).is_err());
}

/// A seeded random sequence must still drive the group gap down — the convergence
/// is not an artifact of the all-equal fixture.
#[test]
fn amortization_converges_on_random_relevances() {
    let mut rng = FairnessRng::new(2024);
    let mut sequence = EquityOfAttention::new(groups(), 2, 3, 0.0).expect("valid");

    let mut gap_at_1 = 0.0;
    for query in 1..=60 {
        // Group-symmetric relevances: both groups drawn from the same
        // distribution, so a fair allocation is equal exposure per relevance.
        let relevances: Vec<f64> = (0..6).map(|_| rng.next_range(0.1, 1.0)).collect();
        sequence.serve(&relevances).expect("serves");
        if query == 1 {
            gap_at_1 = sequence.group_gap();
        }
    }
    let gap_at_60 = sequence.group_gap();
    println!("random-relevance amortization: T=1 {gap_at_1:.6} -> T=60 {gap_at_60:.6}");
    assert!(
        gap_at_60 < gap_at_1,
        "the gap must shrink over the sequence"
    );
}
