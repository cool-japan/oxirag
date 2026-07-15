//! Tests for the [`FairnessRanker`] façade, including the module's central
//! ablation.

use crate::fairness_ranking::ranker::FairnessRanker;
use crate::fairness_ranking::types::{
    ExposureTarget, FairnessConfig, FairnessMetric, FairnessPolicy, GroupId, ProtectedAttribute,
};
use crate::types::{Document, SearchResult};

/// A worst-case pool for exposure fairness: the protected group holds every
/// low-scoring candidate, so the unconstrained ranking buries it.
fn segregated_pool() -> (Vec<f64>, Vec<GroupId>) {
    // Twelve candidates. The unprotected group (1) holds the six best scores; the
    // protected group (0) holds the six worst.
    let scores = vec![
        0.95, 0.90, 0.85, 0.80, 0.75, 0.70, // unprotected
        0.65, 0.60, 0.55, 0.50, 0.45, 0.40, // protected
    ];
    let groups = vec![
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(1),
        GroupId(0),
        GroupId(0),
        GroupId(0),
        GroupId(0),
        GroupId(0),
        GroupId(0),
    ];
    (scores, groups)
}

fn attribute() -> ProtectedAttribute {
    ProtectedAttribute::binary("gender", "female", "male").expect("valid")
}

// ── (d) the ablation: disparity strictly falls, utility cost bounded ─────────

#[test]
fn fair_ranking_strictly_reduces_exposure_disparity_at_bounded_utility_cost() {
    let (scores, groups) = segregated_pool();

    // FA*IR aiming for parity (p = 0.5) at alpha = 0.2. The m-table then demands a
    // real protected presence in the early prefixes, so the fair ranking has to
    // interleave the two groups substantially.
    let policy = FairnessPolicy::RankedGroupFairness {
        target_proportion: 0.5,
        significance: 0.2,
    };
    let ranker =
        FairnessRanker::new(FairnessConfig::new(attribute(), policy, 0)).expect("valid config");
    let report = ranker.report(&scores, &groups).expect("well-formed");

    // Both headline numbers, reported.
    println!(
        "FA*IR ablation: baseline exposure disparity = {:.6}, fair = {:.6} (reduction {:.6}); \
         baseline nDCG = {:.6}, fair = {:.6} (utility cost {:.6})",
        report.baseline_gap,
        report.adjusted_gap,
        report.disparity_reduction(),
        report.baseline_ndcg,
        report.adjusted_ndcg,
        report.utility_cost(),
    );

    // The exposure disparity STRICTLY decreases.
    assert!(
        report.adjusted_gap < report.baseline_gap,
        "fair disparity {} must be below baseline {}",
        report.adjusted_gap,
        report.baseline_gap
    );
    // The margin is real, not a rounding artifact: measured reduction ~0.072 on a
    // baseline disparity of ~0.100 — the fair ranking removes over two-thirds of
    // the exposure gap.
    assert!(
        report.disparity_reduction() > 0.05,
        "disparity reduction {} is too small to be meaningful",
        report.disparity_reduction()
    );
    // Utility degrades by no more than a stated bound: measured nDCG cost ~0.004,
    // so a 0.03 bound is comfortable and still tight enough to catch a regression.
    assert!(
        report.utility_cost() < 0.03,
        "utility cost {} exceeds the stated bound",
        report.utility_cost()
    );
    // The fair ranking passes its own audit.
    assert!(report.audit.as_ref().unwrap().satisfied);
    assert_eq!(report.metric(FairnessMetric::RankedGroupFairness), 1.0);
}

#[test]
fn proportional_exposure_reduces_the_disparity_it_targets() {
    // The two targets optimize *different* fairness criteria, so each is measured
    // against the one it aims at — testing Population against exposure-per-relevance
    // would be testing it against a goal it explicitly does not pursue.
    let (scores, groups) = segregated_pool();

    // Relevance target -> equalize exposure per unit relevance (the EUR gap).
    let relevance_ranker = FairnessRanker::new(FairnessConfig::new(
        attribute(),
        FairnessPolicy::ProportionalExposure {
            target: ExposureTarget::Relevance,
        },
        0,
    ))
    .expect("valid");
    let relevance_report = relevance_ranker.report(&scores, &groups).expect("ok");
    println!(
        "Proportional(Relevance): EUR gap {:.6} -> {:.6}, utility cost {:.6}",
        relevance_report.baseline_gap,
        relevance_report.adjusted_gap,
        relevance_report.utility_cost()
    );
    assert!(
        relevance_report.adjusted_gap < relevance_report.baseline_gap,
        "relevance-targeted exposure must reduce the EUR gap: {} -> {}",
        relevance_report.baseline_gap,
        relevance_report.adjusted_gap
    );

    // Population target -> equalize raw exposure across groups (demographic parity
    // of exposure), which is measured by the demographic-parity gap, NOT the EUR
    // gap (which it will happily worsen, since it ignores relevance by design).
    let population_ranker = FairnessRanker::new(FairnessConfig::new(
        attribute(),
        FairnessPolicy::ProportionalExposure {
            target: ExposureTarget::Population,
        },
        0,
    ))
    .expect("valid");
    let population_report = population_ranker.report(&scores, &groups).expect("ok");
    let baseline_dp = population_report.baseline.demographic_parity_gap();
    let fair_dp = population_report.adjusted.demographic_parity_gap();
    println!("Proportional(Population): demographic-parity gap {baseline_dp:.6} -> {fair_dp:.6}");
    assert!(
        fair_dp < baseline_dp,
        "population-targeted exposure must reduce the demographic-parity gap: {baseline_dp} -> {fair_dp}"
    );
}

#[test]
fn unconstrained_baseline_is_the_plain_sort() {
    let (scores, groups) = segregated_pool();
    let ranker = FairnessRanker::new(FairnessConfig::new(
        attribute(),
        FairnessPolicy::Unconstrained,
        0,
    ))
    .expect("valid");
    let order = ranker.order(&scores, &groups).expect("ok");
    // Descending score, so 0,1,2,...,11.
    assert_eq!(order, (0..12).collect::<Vec<_>>());
    // Under the unconstrained policy the baseline equals the adjusted ranking, so
    // the disparity does not move.
    let report = ranker.report(&scores, &groups).expect("ok");
    assert!((report.adjusted_gap - report.baseline_gap).abs() < 1e-12);
}

// ── the SearchResult path ────────────────────────────────────────────────────

#[test]
fn rerank_search_results_reassigns_ranks() {
    let (scores, groups) = segregated_pool();
    let results: Vec<SearchResult> = scores
        .iter()
        .enumerate()
        .map(|(index, &score)| {
            let mut document = Document::new(format!("doc {index}"));
            document.id = crate::types::DocumentId::from_string(format!("d{index:02}"));
            SearchResult::new(document, score as f32, index)
        })
        .collect();

    let ranker = FairnessRanker::new(FairnessConfig::new(
        attribute(),
        FairnessPolicy::ProportionalExposure {
            target: ExposureTarget::Population,
        },
        0,
    ))
    .expect("valid");
    let reranked = ranker.rerank(&results, &groups).expect("ok");

    assert_eq!(reranked.len(), results.len());
    // Ranks are re-assigned 0..n in output order.
    for (position, result) in reranked.iter().enumerate() {
        assert_eq!(result.rank, position);
    }
    // The protected group (document ids d06..d11) is no longer entirely at the
    // back: at least one protected document appears in the top half.
    let protected_in_top_half = reranked
        .iter()
        .take(6)
        .filter(|result| {
            let id: usize = result.document.id.as_str()[1..].parse().unwrap();
            id >= 6
        })
        .count();
    assert!(
        protected_in_top_half > 0,
        "fairness should lift the protected group"
    );
}

// ── guards ───────────────────────────────────────────────────────────────────

#[test]
fn invalid_policy_parameters_are_rejected() {
    // Proportion outside (0, 1).
    let policy = FairnessPolicy::RankedGroupFairness {
        target_proportion: 1.0,
        significance: 0.1,
    };
    assert!(FairnessRanker::new(FairnessConfig::new(attribute(), policy, 0)).is_err());
    // Significance outside (0, 1).
    let policy = FairnessPolicy::RankedGroupFairness {
        target_proportion: 0.5,
        significance: 0.0,
    };
    assert!(FairnessRanker::new(FairnessConfig::new(attribute(), policy, 0)).is_err());
}

#[test]
fn label_mismatch_is_rejected() {
    let ranker = FairnessRanker::new(FairnessConfig::new(
        attribute(),
        FairnessPolicy::Unconstrained,
        0,
    ))
    .expect("valid");
    let scores = vec![0.5, 0.4, 0.3];
    let groups = vec![GroupId(0), GroupId(1)]; // one short
    assert!(ranker.order(&scores, &groups).is_err());
    // Out-of-range group.
    let groups = vec![GroupId(0), GroupId(1), GroupId(5)];
    assert!(ranker.order(&scores, &groups).is_err());
}
