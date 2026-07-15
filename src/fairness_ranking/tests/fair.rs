//! Tests for `FA*IR` ranked group fairness.
//!
//! Two independent ground truths are used. The `m-table` is checked against
//! `m_alpha(k)` **worked out by hand** in the comments of
//! [`m_table_matches_hand_derivation`]. The multiple-test correction is checked
//! two ways: its exact family-wise failure probability is compared against a
//! **Monte-Carlo** estimate, and the corrected table is asserted to control the
//! error rate where the uncorrected one does not.

use crate::fairness_ranking::fair::{
    MTable, adjusted_significance, audit_ranking, failure_probability, fair_top_k, raw_table,
};
use crate::fairness_ranking::rng::FairnessRng;

/// Draw one null ranking of `k` positions, each protected independently with
/// probability `p`, and report whether it violates `required` at some prefix.
fn null_ranking_violates(rng: &mut FairnessRng, required: &[usize], k: usize, p: f64) -> bool {
    let mut count = 0usize;
    for prefix in 1..=k {
        if rng.next_bernoulli(p) {
            count += 1;
        }
        if count < required[prefix] {
            return true;
        }
    }
    false
}

/// The empirical family-wise violation rate of `required` under the null model,
/// over `trials` seeded draws.
fn monte_carlo_failure_rate(required: &[usize], k: usize, p: f64, trials: usize, seed: u64) -> f64 {
    let mut rng = FairnessRng::new(seed);
    let mut failures = 0usize;
    for _ in 0..trials {
        if null_ranking_violates(&mut rng, required, k, p) {
            failures += 1;
        }
    }
    (failures as f64) / (trials as f64)
}

// ── (c) the m-table, worked out by hand ──────────────────────────────────────

#[test]
fn m_table_matches_hand_derivation() {
    // p = 0.5, per-prefix alpha = 0.1. m_alpha(k) = min { m : P(Bin(k,0.5) <= m) >= 0.1 },
    // computed by hand from P(Bin(k,0.5) <= m) = (sum_{i<=m} C(k,i)) / 2^k:
    //
    //   k=1: P(<=0) = 1/2   = 0.5    >= 0.1        -> m = 0
    //   k=2: P(<=0) = 1/4   = 0.25   >= 0.1        -> m = 0
    //   k=3: P(<=0) = 1/8   = 0.125  >= 0.1        -> m = 0
    //   k=4: P(<=0) = 1/16  = 0.0625 <  0.1
    //        P(<=1) = 5/16  = 0.3125 >= 0.1        -> m = 1
    //   k=5: P(<=0) = 1/32  = 0.031  <  0.1
    //        P(<=1) = 6/32  = 0.1875 >= 0.1        -> m = 1
    //   k=6: P(<=0) = 1/64  = 0.0156 <  0.1
    //        P(<=1) = 7/64  = 0.1094 >= 0.1        -> m = 1
    //   k=7: P(<=1) = 8/128 = 0.0625 <  0.1
    //        P(<=2) = 29/128= 0.2266 >= 0.1        -> m = 2
    //   k=8: P(<=1) = 9/256 = 0.0352 <  0.1
    //        P(<=2) = 37/256= 0.1445 >= 0.1        -> m = 2
    //
    // so the raw m-table (index 0 is the ignored padding slot) is:
    let expected = vec![0usize, 0, 0, 0, 1, 1, 1, 2, 2];
    assert_eq!(raw_table(8, 0.5, 0.1), expected);

    // The critical off-by-one: m_alpha(1) = 0. The top slot is NOT forced
    // protected at alpha = 0.1 — a single Bernoulli trial cannot be significantly
    // unbalanced. A min/strict-`<` mixture would put a 1 here.
    assert_eq!(raw_table(1, 0.5, 0.1)[1], 0);
}

// ── (c) acceptance / rejection exactly as derived ────────────────────────────

#[test]
fn ranker_accepts_and_rejects_exactly_as_derived() {
    // Use the hand-derived raw table above (unadjusted, so the acceptance test is
    // against exactly the numbers derived by hand). Required at prefixes 1..8:
    //   [0, 0, 0, 1, 1, 1, 2, 2].
    let table = MTable::unadjusted(8, 0.5, 0.1).expect("valid");
    assert_eq!(table.entries(), &[0, 0, 0, 0, 1, 1, 1, 2, 2]);

    // Ranking A: protected at 1-based positions 4 and 7 -> counts
    //   pos: 1 2 3 4 5 6 7 8
    //   cnt: 0 0 0 1 1 1 2 2   which meets [_,0,0,0,1,1,1,2,2] everywhere. ACCEPT.
    let ranking_a = [false, false, false, true, false, false, true, false];
    let audit_a = audit_ranking(&ranking_a, &table);
    assert!(audit_a.satisfied, "ranking A should pass: {audit_a:?}");
    assert_eq!(audit_a.first_violation, None);

    // Ranking B: no protected candidates at all. At prefix 4 the table needs 1
    // and the ranking has 0. REJECT, first violation at prefix 4, short by 1.
    let ranking_b = [false; 8];
    let audit_b = audit_ranking(&ranking_b, &table);
    assert!(!audit_b.satisfied);
    assert_eq!(audit_b.first_violation, Some(4));
    assert_eq!(audit_b.shortfall, 1);

    // Ranking C: protected first appears at position 5, so prefix 4 has 0 < 1.
    // REJECT at 4, even though later prefixes recover.
    let ranking_c = [false, false, false, false, true, true, false, false];
    let audit_c = audit_ranking(&ranking_c, &table);
    assert!(!audit_c.satisfied);
    assert_eq!(audit_c.first_violation, Some(4));

    // Ranking D: protected at 4 and 6 -> counts 0,0,0,1,1,2,2,2, meets the table.
    let ranking_d = [false, false, false, true, false, true, false, false];
    assert!(audit_ranking(&ranking_d, &table).satisfied);
}

// ── the exact failure probability, checked against Monte Carlo ───────────────

#[test]
fn exact_failure_probability_matches_monte_carlo() {
    // The exact O(k^2) forward recursion must agree with the empirical violation
    // rate of the same table under the null model.
    for &(k, p, alpha) in &[(8usize, 0.5, 0.1), (20, 0.5, 0.2), (30, 0.3, 0.15)] {
        let table = raw_table(k, p, alpha);
        let exact = failure_probability(&table, p);
        let empirical = monte_carlo_failure_rate(&table, k, p, 200_000, 0xF00D);
        assert!(
            (exact - empirical).abs() < 0.01,
            "k={k} p={p} alpha={alpha}: exact failure {exact} vs monte carlo {empirical}"
        );
    }
}

// ── the correction does real work ────────────────────────────────────────────

#[test]
fn correction_controls_family_wise_error_where_naive_does_not() {
    // At k = 100, applying alpha = 0.1 at every one of 100 prefixes without
    // correction inflates the family-wise error far past 0.1. The correction must
    // pull it back to at most 0.1.
    let k = 100;
    let p = 0.5;
    let alpha = 0.1;

    let unadjusted = MTable::unadjusted(k, p, alpha).expect("valid");
    let corrected = MTable::new(k, p, alpha).expect("valid");

    // The naive procedure overshoots the nominal level...
    assert!(
        unadjusted.failure_probability() > alpha,
        "uncorrected family-wise error should exceed alpha, was {}",
        unadjusted.failure_probability()
    );
    // ...and the correction controls it.
    assert!(
        corrected.failure_probability() <= alpha + 1e-9,
        "corrected family-wise error should be <= alpha, was {}",
        corrected.failure_probability()
    );
    // The correction is a genuine tightening of the per-prefix significance.
    assert!(corrected.adjusted_significance() < alpha);
    assert!(corrected.adjusted_significance() > 0.0);

    // And it is not *more* conservative than it must be: the failure probability
    // sits just under alpha, not far below it. (If it were e.g. 0.001 the test
    // would have lost most of its power.)
    assert!(
        corrected.failure_probability() > 0.5 * alpha,
        "correction is over-conservative: failure {} vs alpha {alpha}",
        corrected.failure_probability()
    );

    // Independent Monte-Carlo confirmation of the corrected table's real FWER.
    let empirical = monte_carlo_failure_rate(corrected.entries(), k, p, 100_000, 0xBEEF);
    assert!(
        empirical <= alpha + 0.01,
        "empirical FWER of the corrected table should be ~<= alpha, was {empirical}"
    );
}

#[test]
fn adjusted_significance_is_a_monotone_step_below_alpha() {
    // Short rankings need no correction: too few prefixes for the multiplicity to
    // bite, so alpha comes back unchanged.
    assert_eq!(adjusted_significance(2, 0.5, 0.1), 0.1);
    // Long rankings do.
    let adjusted = adjusted_significance(200, 0.5, 0.1);
    assert!(adjusted < 0.1 && adjusted > 0.0);
    // The corrected table's own reported adjusted significance never exceeds the
    // requested one, at any length.
    for k in [1usize, 5, 25, 75, 150] {
        let table = MTable::new(k, 0.5, 0.1).expect("valid");
        assert!(table.adjusted_significance() <= 0.1 + 1e-15);
    }
}

// ── the m-table shape lemmas fair_top_k depends on ───────────────────────────

#[test]
fn m_table_is_non_decreasing_and_one_lipschitz() {
    for &(k, p, alpha) in &[(60usize, 0.5, 0.1), (60, 0.25, 0.05), (60, 0.7, 0.2)] {
        let table = raw_table(k, p, alpha);
        for prefix in 1..k {
            let step = table[prefix + 1] as i64 - table[prefix] as i64;
            assert!(
                (0..=1).contains(&step),
                "m-table must increase by 0 or 1: at prefix {prefix}, {} -> {} (p={p})",
                table[prefix],
                table[prefix + 1]
            );
        }
    }
}

// ── the greedy fair ranking ──────────────────────────────────────────────────

#[test]
fn fair_top_k_satisfies_the_table_and_preserves_in_group_order() {
    // Ten candidates: indices 0..4 protected (scores descending), 5..9 not.
    // The fair ranking must (i) satisfy the m-table, (ii) keep each group in its
    // given order, and (iii) be a permutation of a prefix of the candidates.
    let protected_order = vec![0usize, 1, 2, 3, 4];
    let unprotected_order = vec![5usize, 6, 7, 8, 9];
    let scores = vec![0.95, 0.85, 0.75, 0.65, 0.55, 0.9, 0.8, 0.7, 0.6, 0.5];

    let table = MTable::new(10, 0.5, 0.1).expect("valid");
    let ranking = fair_top_k(&protected_order, &unprotected_order, &scores, &table).expect("ok");
    assert_eq!(ranking.len(), 10);

    // The protected candidates appear in the order 0,1,2,3,4 and the unprotected
    // in 5,6,7,8,9 (in-group monotonicity).
    let protected_seen: Vec<usize> = ranking.iter().copied().filter(|&i| i < 5).collect();
    let unprotected_seen: Vec<usize> = ranking.iter().copied().filter(|&i| i >= 5).collect();
    assert_eq!(protected_seen, protected_order);
    assert_eq!(unprotected_seen, unprotected_order);

    // The m-table is satisfied at every prefix.
    let flags: Vec<bool> = ranking.iter().map(|&i| i < 5).collect();
    assert!(audit_ranking(&flags, &table).satisfied);
}

#[test]
fn fair_top_k_reports_insufficient_protected() {
    // A table needing protected candidates the pool does not have must be a loud
    // error, never a silently-unfair ranking.
    let protected_order = vec![0usize]; // only one protected candidate
    let unprotected_order = vec![1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let scores = vec![0.5; 11];
    // A strict setting that forces several protected candidates in the top-k.
    let table = MTable::new(10, 0.6, 0.2).expect("valid");
    let result = fair_top_k(&protected_order, &unprotected_order, &scores, &table);
    assert!(
        matches!(
            result,
            Err(crate::fairness_ranking::types::FairnessError::InsufficientProtected { .. })
        ),
        "expected InsufficientProtected, got {result:?}"
    );
}
