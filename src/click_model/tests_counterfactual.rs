#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::manual_midpoint,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::items_after_statements,
    clippy::doc_markdown
)]
//! Tests for the cross-model EM-monotonicity invariant and for the
//! *counterfactual* half of `click_model`: propensity estimates and their
//! clipping, ranking policies, the IPS / SNIPS / doubly-robust estimators, the
//! `DebiasedRanker`, the deterministic PRNG, and the simulator.
//!
//! The three click models' own E-step / M-step derivations and their
//! parameter-recovery tests live in the sibling `tests` module.
//!
//! The headline test here is `naive_ctr_is_fooled_but_ips_recovers_the_true_order`:
//! the entire justification for the module, end to end.

use std::collections::HashMap;

use super::cascade::{CascadeClickModel, cascade_examined_depth};
use super::counterfactual::{
    ClickRankingPolicy, ClickRewardModel, CounterfactualStrategy, DoublyRobustEstimator,
    IpsEstimator, PropensityEstimates, SnipsEstimator, TableClickPolicy, naive_click_through_rates,
    policy_positions,
};
use super::dbn::DbnClickModel;
use super::debias::{DebiasedRanker, DebiasedRankerConfig};
use super::pbm::PositionBasedModel;
use super::synthetic::{ClickLogSimulator, ClickSimulationModel, ClickSplitMix64};
use super::tests::{close, ids, log_of, max_abs_error, session};
use super::types::{
    ClickLog, ClickModel, ClickModelConfig, ClickModelError, ClickSession, position_gain,
};

// ═══════════════════════════════════════════════════════════════════════════
// EM monotonicity — the correctness invariant, for all three models
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn em_log_likelihood_is_monotone_for_all_three_models() {
    // A deliberately awkward log: multi-click sessions, all-zero sessions,
    // single-document sessions, varying depths. If any E-step or M-step is wrong,
    // one of these will drive the log-likelihood downhill.
    let mut rng = ClickSplitMix64::new(0x4E00_7011);
    let doc_ids = ids(&["a", "b", "c", "d", "e"]);
    let mut sessions = Vec::new();
    for _ in 0..300 {
        let mut pool = doc_ids.clone();
        rng.shuffle(&mut pool);
        let depth = 1 + rng.below(5);
        let slate: Vec<String> = pool[..depth].to_vec();
        let clicks: Vec<bool> = (0..depth).map(|_| rng.bernoulli(0.35)).collect();
        sessions.push(ClickSession::new("q", slate, clicks));
    }
    let log = log_of(sessions);

    let config = ClickModelConfig::default()
        .with_max_iterations(150)
        .with_tolerance(0.0); // never stop early: check EVERY iteration

    let mut models: Vec<Box<dyn ClickModel>> = vec![
        Box::new(PositionBasedModel::new(config.clone())),
        Box::new(CascadeClickModel::new(config.clone())),
        Box::new(DbnClickModel::new(config)),
    ];

    for model in &mut models {
        let name = model.model_name();
        let fit = model.fit(&log).expect("fit succeeds");
        assert_eq!(fit.log_likelihood_history.len(), fit.iterations + 1);
        assert!(
            fit.is_monotone(1e-9),
            "{name}: log-likelihood DECREASED by {} — the E-step or M-step is wrong",
            fit.worst_decrease()
        );
        assert!(
            fit.total_improvement() > 0.0,
            "{name}: EM should improve on its initialisation"
        );
        // The reported final log-likelihood is reproducible from the parameters.
        assert!(close(
            model.log_likelihood(&log).expect("fitted"),
            fit.final_log_likelihood,
            1e-9
        ));
        // Every model exposes a usable examination curve.
        let curve = model.examination_curve();
        assert_eq!(curve.len(), log.depth());
        for &value in curve {
            assert!(
                value.is_finite() && value > 0.0 && value <= 1.0,
                "{name}: γ = {value}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Propensities
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn propensity_clipping_actually_engages() {
    let propensities =
        PropensityEstimates::new(vec![1.0, 0.5, 0.05, 0.001], 0.01).expect("valid curve");

    assert_eq!(propensities.depth(), 4);
    assert_eq!(propensities.clip_floor(), 0.01);
    assert!(!propensities.is_clipped(0));
    assert!(!propensities.is_clipped(2));
    assert!(propensities.is_clipped(3), "0.001 < 0.01 must clip");
    assert_eq!(propensities.clipped_rank_count(), 1);

    // Unclipped ranks pass through; the clipped one is floored.
    assert!(close(propensities.propensity(2), 0.05, 1e-12));
    assert!(close(propensities.propensity(3), 0.01, 1e-12));
    assert!(close(
        propensities.raw_examination(3).expect("in range"),
        0.001,
        1e-12
    ));

    // The whole point: the importance weight is CAPPED at 1/clip.
    assert!(close(propensities.importance_weight(3), 100.0, 1e-9));
    assert!(
        propensities.importance_weight(3) < 1000.0,
        "without clipping this would have been 1/0.001 = 1000"
    );

    // Beyond the curve: a documented fallback, and an error if you want one.
    assert!(close(propensities.propensity(99), 0.01, 1e-12));
    assert!(matches!(
        propensities.try_propensity(99),
        Err(ClickModelError::UnknownRank { rank: 99, depth: 4 })
    ));
    assert!(close(
        propensities.try_propensity(2).expect("in range"),
        0.05,
        1e-12
    ));
}

#[test]
fn propensity_curve_validation_rejects_impossible_values() {
    assert!(PropensityEstimates::new(vec![1.0, 0.0], 0.01).is_err());
    assert!(PropensityEstimates::new(vec![1.0, 1.5], 0.01).is_err());
    assert!(PropensityEstimates::new(vec![1.0, f64::NAN], 0.01).is_err());
    assert!(PropensityEstimates::new(vec![1.0], 0.0).is_err());
    assert!(PropensityEstimates::new(vec![1.0], 1.5).is_err());
    assert!(PropensityEstimates::new(vec![1.0, 0.5], 1.0).is_ok());
}

#[test]
fn uniform_propensities_are_the_no_bias_control() {
    let uniform = PropensityEstimates::uniform(5);
    for rank in 0..5 {
        assert_eq!(uniform.propensity(rank), 1.0);
        assert_eq!(uniform.importance_weight(rank), 1.0);
    }
}

#[test]
fn propensities_come_straight_off_a_fitted_click_model() {
    let log = ClickLogSimulator::new(ids(&["a", "b", "c"]))
        .with_attractiveness(vec![0.8, 0.5, 0.3])
        .with_examination(vec![1.0, 0.5, 0.2])
        .with_seed(5)
        .simulate(ClickSimulationModel::PositionBased, 3_000)
        .expect("valid simulation");

    let mut model = PositionBasedModel::new(ClickModelConfig::default().with_max_iterations(200));
    assert!(matches!(
        PropensityEstimates::from_click_model(&model, 0.01),
        Err(ClickModelError::NotFitted { .. })
    ));

    model.fit(&log).expect("fit succeeds");
    let propensities = PropensityEstimates::from_click_model(&model, 0.01).expect("usable curve");
    assert_eq!(propensities.depth(), 3);
    assert!(close(propensities.propensity(0), 1.0, 1e-9));
    assert!(close(propensities.propensity(1), 0.5, 0.05));
    assert!(close(propensities.propensity(2), 0.2, 0.05));
}

// ═══════════════════════════════════════════════════════════════════════════
// Policies
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn policy_positions_permute_deterministically_with_logged_order_tiebreak() {
    let s = session("q", &["a", "b", "c"], &[false, false, false]);
    let policy = TableClickPolicy::new()
        .with_doc_score("a", 0.1)
        .with_doc_score("b", 0.9)
        .with_doc_score("c", 0.1);
    // "b" goes first; "a" and "c" tie and keep their logged order.
    assert_eq!(policy_positions(&policy, &s), vec![1, 0, 2]);

    // A per-query override beats the query-independent score.
    let scoped = policy.with_query_doc_score("q", "c", 5.0);
    assert_eq!(policy_positions(&scoped, &s), vec![2, 1, 0]);
}

#[test]
fn table_policy_falls_back_to_its_default_score() {
    let policy = TableClickPolicy::from_doc_scores(HashMap::from([("a".to_owned(), 1.0)]))
        .with_default_score(-1.0);
    assert_eq!(policy.score("q", "a"), 1.0);
    assert_eq!(policy.score("q", "unknown"), -1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Counterfactual estimators
// ═══════════════════════════════════════════════════════════════════════════

/// A simulated PBM log plus everything needed to score a policy against the
/// GROUND TRUTH — which is what makes the unbiasedness tests below real
/// measurements rather than self-consistency checks.
struct Bench {
    log: ClickLog,
    true_attractiveness: HashMap<String, f64>,
    true_propensities: PropensityEstimates,
    policy: TableClickPolicy,
}

fn build_bench(sessions: usize, seed: u64) -> Bench {
    let examination = vec![1.0, 0.6, 0.4, 0.25, 0.12, 0.05];
    let attractiveness = vec![0.8, 0.65, 0.5, 0.35, 0.2, 0.1, 0.45, 0.3];
    let doc_ids: Vec<String> = (0..attractiveness.len())
        .map(|index| format!("d{index}"))
        .collect();

    let log = ClickLogSimulator::new(doc_ids.clone())
        .with_attractiveness(attractiveness.clone())
        .with_examination(examination.clone())
        .with_slate_size(examination.len())
        .with_seed(seed)
        .simulate(ClickSimulationModel::PositionBased, sessions)
        .expect("valid simulation");

    // The candidate policy: rank by true relevance (the "oracle" reordering).
    let mut policy = TableClickPolicy::new();
    for (doc, &alpha) in doc_ids.iter().zip(attractiveness.iter()) {
        policy = policy.with_doc_score(doc.clone(), alpha);
    }

    Bench {
        true_attractiveness: doc_ids.iter().cloned().zip(attractiveness).collect(),
        true_propensities: PropensityEstimates::new(examination, 0.01).expect("valid"),
        log,
        policy,
    }
}

impl Bench {
    /// The estimand: `(1/|S|) Σ_s Σ_d gain(pos_π(d)) · α_d`, computed from the
    /// ground-truth `α` — the number every unbiased estimator must land on.
    fn true_policy_value(&self) -> f64 {
        let mut total = 0.0_f64;
        for s in self.log.sessions() {
            let positions = policy_positions(&self.policy, s);
            for (rank, doc) in s.ranked_doc_ids.iter().enumerate() {
                total += position_gain(positions[rank]) * self.true_attractiveness[doc];
            }
        }
        total / self.log.len() as f64
    }

    /// The same log with the ground-truth examination indicators stripped out —
    /// i.e. what a *real* click log looks like.
    fn without_examinations(&self) -> ClickLog {
        ClickLog::from_sessions(
            self.log
                .sessions()
                .iter()
                .map(|s| ClickSession::new(&s.query_id, s.ranked_doc_ids.clone(), s.clicks.clone()))
                .collect(),
        )
        .expect("valid log")
    }
}

#[test]
fn ips_is_unbiased_when_the_propensities_are_right() {
    let bench = build_bench(20_000, 0xBEEF);
    let truth = bench.true_policy_value();
    let estimate = IpsEstimator::new(bench.true_propensities.clone())
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");

    let relative_error = (estimate.value - truth).abs() / truth;
    println!(
        "IPS: true = {truth:.4}, estimate = {:.4}, relative error = {:.2}%",
        estimate.value,
        relative_error * 100.0
    );
    assert!(
        relative_error < 0.03,
        "IPS should be unbiased: {} vs {truth}",
        estimate.value
    );

    assert_eq!(estimate.strategy, CounterfactualStrategy::Ips);
    assert_eq!(estimate.strategy.name(), "IPS");
    assert_eq!(estimate.session_count, 20_000);
    assert!(estimate.clicked_impression_count > 0);
    assert!(estimate.total_importance_weight > 0.0);
    assert!(estimate.max_importance_weight >= 1.0);
    assert!(estimate.effective_sample_size > 0.0);
    assert!(
        estimate.effective_sample_size <= estimate.clicked_impression_count as f64 + 1e-6,
        "Kish's ESS can never exceed the raw count"
    );
    assert_eq!(
        estimate.clipped_impression_count, 0,
        "no rank is below 0.01"
    );
}

#[test]
fn a_naive_unweighted_estimate_is_badly_biased() {
    // The control: the SAME estimator with propensities of 1 everywhere — i.e.
    // "assume no position bias". It must be visibly wrong.
    let bench = build_bench(20_000, 0xBEEF);
    let truth = bench.true_policy_value();
    let naive = IpsEstimator::new(PropensityEstimates::uniform(6))
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");
    let naive_error = (naive.value - truth).abs() / truth;
    println!(
        "Naive (uniform propensity): true = {truth:.4}, estimate = {:.4}, relative error = {:.1}%",
        naive.value,
        naive_error * 100.0
    );
    assert!(
        naive_error > 0.25,
        "ignoring position bias must be badly wrong, not slightly wrong: {naive_error}"
    );
    // And it under-counts, because deep clicks are never up-weighted.
    assert!(naive.value < truth);
}

#[test]
fn doubly_robust_collapses_onto_ips_when_the_reward_model_is_zero() {
    // The algebraic sanity bound: ŷ ≡ 0 ⟹ ŷ + (c − ô·ŷ)/γ = c/γ, exactly IPS.
    let bench = build_bench(3_000, 0x1234);
    let ips = IpsEstimator::new(bench.true_propensities.clone())
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");
    let dr = DoublyRobustEstimator::new(
        bench.true_propensities.clone(),
        ClickRewardModel::constant(0.0),
    )
    .estimate(&bench.log, &bench.policy)
    .expect("non-empty log");
    assert!(
        close(dr.value, ips.value, 1e-9),
        "DR with a zero reward model must be IPS: {} vs {}",
        dr.value,
        ips.value
    );
    assert_eq!(dr.strategy, CounterfactualStrategy::DoublyRobust);
}

#[test]
fn doubly_robust_is_unbiased_when_the_reward_model_is_wrong_but_propensities_are_right() {
    let bench = build_bench(20_000, 0xD0D0);
    let truth = bench.true_policy_value();

    // A hopelessly wrong reward model: every document is "excellent".
    let wrong_reward = ClickRewardModel::constant(0.95);

    // (a) With the ground-truth examination indicator the simulator logged.
    let exact = DoublyRobustEstimator::new(bench.true_propensities.clone(), wrong_reward.clone())
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");
    let exact_error = (exact.value - truth).abs() / truth;
    println!(
        "DR (reward model WRONG, propensities RIGHT, examinations LOGGED): true = {truth:.4}, DR = {:.4} ({:.2}%)",
        exact.value,
        exact_error * 100.0
    );
    assert!(
        exact_error < 0.03,
        "DR must survive a wrong reward model: {} vs {truth}",
        exact.value
    );

    // (b) With the examination indicator IMPUTED from the click model's own
    //     posterior, ô = γ(1−α)/(1−γα). E[ô] = γ exactly when the click model is
    //     right — so the propensity-correct branch survives the imputation too.
    let stripped = bench.without_examinations();
    let imputed = DoublyRobustEstimator::new(bench.true_propensities.clone(), wrong_reward)
        .with_attractiveness(bench.true_attractiveness.clone())
        .estimate(&stripped, &bench.policy)
        .expect("non-empty log");
    let imputed_error = (imputed.value - truth).abs() / truth;
    println!(
        "DR (same, but examinations IMPUTED from the posterior): DR = {:.4} ({:.2}%)",
        imputed.value,
        imputed_error * 100.0
    );
    assert!(
        imputed_error < 0.03,
        "E[ô] = γ makes the imputed DR unbiased too, when the click model is right"
    );

    // For contrast: the *direct method* alone — the same wrong reward model with
    // no correction at all — is hopeless.
    let direct_only = DoublyRobustEstimator::new(
        bench.true_propensities.clone(),
        ClickRewardModel::constant(0.95),
    );
    let _ = direct_only; // (the DM has no estimator of its own; see below)
    let mut direct_value = 0.0_f64;
    for s in bench.log.sessions() {
        let positions = policy_positions(&bench.policy, s);
        for rank in 0..s.len() {
            direct_value += position_gain(positions[rank]) * 0.95;
        }
    }
    direct_value /= bench.log.len() as f64;
    let direct_error = (direct_value - truth).abs() / truth;
    println!(
        "Direct method with the same wrong reward model: {direct_value:.4} ({:.1}%)",
        direct_error * 100.0
    );
    assert!(
        direct_error > 0.5,
        "the DM inherits the reward model's error wholesale; DR does not"
    );
}

#[test]
fn doubly_robust_is_unbiased_when_the_propensities_are_wrong_but_the_reward_model_is_right() {
    // The OTHER robustness branch. It needs a truthful examination indicator —
    // which is exactly the caveat the module documents — so this uses the
    // ground-truth `examinations` the simulator logs.
    let bench = build_bench(20_000, 0x00C0_FFEE);
    let truth = bench.true_policy_value();

    // Deliberately, badly wrong propensities: a flat 0.9 curve — "almost everyone
    // sees everything" — against a true curve that runs from 1.0 down to 0.05.
    let wrong_propensities = PropensityEstimates::new(vec![0.9; 6], 0.01).expect("valid");
    // ...but a perfect reward model.
    let right_reward = ClickRewardModel::from_table(bench.true_attractiveness.clone(), 0.0);

    let dr = DoublyRobustEstimator::new(wrong_propensities.clone(), right_reward.clone())
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");
    let dr_error = (dr.value - truth).abs() / truth;

    // And, for contrast, IPS with the SAME wrong propensities.
    let ips = IpsEstimator::new(wrong_propensities.clone())
        .estimate(&bench.log, &bench.policy)
        .expect("non-empty log");
    let ips_error = (ips.value - truth).abs() / truth;

    println!(
        "Propensities WRONG, reward model RIGHT: true = {truth:.4}, DR = {:.4} ({:.2}%), IPS = {:.4} ({:.1}%)",
        dr.value,
        dr_error * 100.0,
        ips.value,
        ips_error * 100.0
    );
    assert!(
        dr_error < 0.02,
        "a correct reward model must rescue DR from wrong propensities: {} vs {truth}",
        dr.value
    );
    assert!(
        ips_error > 0.3,
        "IPS has no second chance — it must be badly wrong here: {ips_error}"
    );

    // The documented caveat, demonstrated rather than swept away: strip the
    // ground-truth examinations, and this branch of the guarantee is LOST,
    // because the imputed ô is computed from the same wrong propensities.
    let stripped = bench.without_examinations();
    let imputed = DoublyRobustEstimator::new(wrong_propensities, right_reward)
        .with_attractiveness(bench.true_attractiveness.clone())
        .estimate(&stripped, &bench.policy)
        .expect("non-empty log");
    let imputed_error = (imputed.value - truth).abs() / truth;
    println!(
        "...and with the examination indicator imputed from the WRONG click model: {:.4} ({:.1}%) — the reward-correct branch does NOT survive",
        imputed.value,
        imputed_error * 100.0
    );
    assert!(
        imputed_error > 0.05,
        "the module documents that the reward-correct branch needs a truthful ô; \
         this asserts the documentation is HONEST rather than optimistic \
         (with a logged ô the error is {dr_error}; with an imputed one it is {imputed_error})"
    );
}

#[test]
fn snips_has_lower_variance_than_ips() {
    // Both estimators, repeatedly, on sub-samples of one big log. IPS's spread is
    // driven by the fluctuation in Σw; SNIPS divides that same Σw out. The
    // estimators live on different scales, so we compare the scale-free
    // coefficient of variation.
    let bench = build_bench(20_000, 0x5417_5008);
    let ips = IpsEstimator::new(bench.true_propensities.clone());
    let snips = SnipsEstimator::new(bench.true_propensities.clone());

    let mut rng = ClickSplitMix64::new(0x0A11_7CE5);
    let subsample_size = 60_usize;
    let replications = 200_usize;

    let mut ips_values = Vec::with_capacity(replications);
    let mut snips_values = Vec::with_capacity(replications);
    for _ in 0..replications {
        let indices: Vec<usize> = (0..subsample_size)
            .map(|_| rng.below(bench.log.len()))
            .collect();
        let subsample = bench.log.subset(&indices);
        ips_values.push(
            ips.estimate(&subsample, &bench.policy)
                .expect("non-empty")
                .value,
        );
        snips_values.push(
            snips
                .estimate(&subsample, &bench.policy)
                .expect("non-empty")
                .value,
        );
    }

    fn coefficient_of_variation(values: &[f64]) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt() / mean.abs()
    }

    let ips_cv = coefficient_of_variation(&ips_values);
    let snips_cv = coefficient_of_variation(&snips_values);
    println!(
        "Across {replications} sub-samples of {subsample_size} sessions: \
         CV(IPS) = {ips_cv:.4}, CV(SNIPS) = {snips_cv:.4} \
         → SNIPS cuts relative spread by {:.0}%",
        (1.0 - snips_cv / ips_cv) * 100.0
    );
    assert!(
        snips_cv < ips_cv,
        "self-normalisation must reduce relative variance: CV(SNIPS) = {snips_cv}, CV(IPS) = {ips_cv}"
    );
}

#[test]
fn snips_and_ips_survive_a_click_free_log() {
    let log = log_of(vec![
        session("q", &["a", "b"], &[false, false]),
        session("q", &["b", "a"], &[false, false]),
    ]);
    let propensities = PropensityEstimates::new(vec![1.0, 0.4], 0.01).expect("valid");
    let policy = TableClickPolicy::new().with_doc_score("a", 1.0);

    let snips = SnipsEstimator::new(propensities.clone())
        .estimate(&log, &policy)
        .expect("non-empty log");
    // No clicks ⇒ no importance mass ⇒ the honest answer is zero, not NaN.
    assert_eq!(snips.value, 0.0);
    assert_eq!(snips.strategy.name(), "SNIPS");
    assert_eq!(snips.clicked_impression_count, 0);
    assert_eq!(snips.total_importance_weight, 0.0);
    assert_eq!(snips.effective_sample_size, 0.0);

    let ips = IpsEstimator::new(propensities)
        .estimate(&log, &policy)
        .expect("non-empty log");
    assert_eq!(ips.value, 0.0);
}

#[test]
fn estimators_reject_an_empty_log() {
    let propensities = PropensityEstimates::uniform(3);
    let policy = TableClickPolicy::new();
    let empty = ClickLog::new();
    assert!(matches!(
        IpsEstimator::new(propensities.clone()).estimate(&empty, &policy),
        Err(ClickModelError::EmptyLog)
    ));
    assert!(matches!(
        SnipsEstimator::new(propensities.clone()).estimate(&empty, &policy),
        Err(ClickModelError::EmptyLog)
    ));
    assert!(matches!(
        DoublyRobustEstimator::new(propensities, ClickRewardModel::constant(0.5))
            .estimate(&empty, &policy),
        Err(ClickModelError::EmptyLog)
    ));
}

#[test]
fn reward_model_shrinks_thin_evidence_toward_the_pooled_mean() {
    // "solo" is seen once and clicked once at a deep, heavily-weighted rank. A raw
    // IPS estimate would hand it a reward of 1.0 on a single observation;
    // shrinkage refuses to.
    let mut sessions = Vec::new();
    for index in 0..100 {
        sessions.push(session("q", &["a", "b"], &[index % 4 == 0, index % 8 == 0]));
    }
    sessions.push(session("q", &["a", "b", "solo"], &[false, false, true]));
    let log = log_of(sessions);
    let propensities = PropensityEstimates::new(vec![1.0, 0.5, 0.1], 0.01).expect("valid");

    // With no shrinkage, that single click at γ = 0.1 carries an importance weight
    // of 10, so the raw IPS relevance estimate is 10 — clamped to a confident,
    // absurd 1.0 on the strength of one observation.
    let unshrunk = ClickRewardModel::fit_ips(&log, &propensities, 0.0);
    assert!(
        close(unshrunk.reward("solo"), 1.0, 1e-9),
        "1 click / γ=0.1 → 10, clamped to 1: {}",
        unshrunk.reward("solo")
    );

    // Shrinkage pulls it back toward the pooled mean, monotonically in the
    // equivalent-sample-size prior κ.
    // (κ = 10 is not enough to matter here: the raw weighted click is 10, so
    // `(10 + 10·ȳ) / (1 + 10)` is still above 1 and clamps. The prior has to be
    // comparable to the importance weight it is fighting.)
    let rewards: Vec<f64> = [0.0, 20.0, 50.0, 100.0]
        .iter()
        .map(|&prior| ClickRewardModel::fit_ips(&log, &propensities, prior).reward("solo"))
        .collect();
    for window in rewards.windows(2) {
        assert!(
            window[1] < window[0],
            "a stronger prior must shrink harder: {rewards:?}"
        );
    }

    let shrunk = ClickRewardModel::fit_ips(&log, &propensities, 50.0);
    assert!(
        shrunk.reward("solo") < 0.6,
        "a single observation must not buy a reward of 1.0: {}",
        shrunk.reward("solo")
    );
    // Documents with plenty of evidence (101 impressions apiece) are barely moved.
    assert!(
        close(shrunk.reward("a"), unshrunk.reward("a"), 0.05),
        "{} vs {}",
        shrunk.reward("a"),
        unshrunk.reward("a")
    );
    for doc in ["a", "b", "solo"] {
        assert!((0.0..=1.0).contains(&shrunk.reward(doc)));
    }
    assert!((0.0..=1.0).contains(&shrunk.default_reward()));
    // An unknown document falls back on the pooled mean.
    assert_eq!(shrunk.reward("unheard-of"), shrunk.default_reward());
}

#[test]
fn naive_click_through_rates_are_exactly_clicks_over_impressions() {
    let log = log_of(vec![
        session("q", &["a", "b"], &[true, false]),
        session("q", &["a", "b"], &[false, true]),
        session("q", &["a", "b"], &[true, true]),
    ]);
    let ctr = naive_click_through_rates(&log);
    assert!(close(ctr["a"], 2.0 / 3.0, 1e-12));
    assert!(close(ctr["b"], 2.0 / 3.0, 1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// THE headline demonstration: naive CTR is fooled, IPS recovers the truth
// ═══════════════════════════════════════════════════════════════════════════

/// The rigged log: 90% of traffic sees the production ranking (which puts a
/// mediocre document on top and buries an excellent one), 10% is randomised —
/// the swap intervention that makes propensities estimable at all.
fn rigged_bias_log() -> (ClickLog, Vec<f64>, HashMap<String, f64>) {
    let docs = ids(&["mediocre", "filler", "excellent"]);
    let true_attractiveness = vec![0.3, 0.15, 0.9];
    let true_examination = vec![1.0, 0.4, 0.1];

    let mut rankings = Vec::new();
    for index in 0..8_000 {
        rankings.push(match index % 20 {
            18 => ids(&["excellent", "mediocre", "filler"]),
            19 => ids(&["filler", "excellent", "mediocre"]),
            _ => docs.clone(),
        });
    }
    let log = ClickLogSimulator::new(docs.clone())
        .with_attractiveness(true_attractiveness.clone())
        .with_examination(true_examination.clone())
        .with_seed(0xB1A5)
        .simulate_rankings(ClickSimulationModel::PositionBased, &rankings)
        .expect("valid simulation");

    let truth = docs.into_iter().zip(true_attractiveness).collect();
    (log, true_examination, truth)
}

#[test]
fn naive_ctr_is_fooled_but_ips_recovers_the_true_order() {
    let (log, true_examination, truth) = rigged_bias_log();
    assert!(
        truth["excellent"] > truth["mediocre"],
        "ground truth: excellent really IS the better document"
    );

    // ── 1. The naive click-through rate is FOOLED. ───────────────────────────
    let ctr = naive_click_through_rates(&log);
    println!(
        "Raw CTR:  mediocre = {:.4}, excellent = {:.4}",
        ctr["mediocre"], ctr["excellent"]
    );
    assert!(
        ctr["mediocre"] > ctr["excellent"],
        "a raw-CTR ranker must be fooled by the position bias: {ctr:?}"
    );

    // ── 2. So is a naive PAIRWISE learner — "no position bias" = uniform γ. ──
    let mut naive_ranker = DebiasedRanker::new(DebiasedRankerConfig::default());
    naive_ranker
        .fit_tabular(&log, &PropensityEstimates::uniform(3))
        .expect("fit succeeds");
    println!(
        "Naive pairwise scores: mediocre = {:.4}, excellent = {:.4}",
        naive_ranker.score_doc("mediocre"),
        naive_ranker.score_doc("excellent")
    );
    assert!(
        naive_ranker.score_doc("mediocre") > naive_ranker.score_doc("excellent"),
        "an unweighted pairwise learner must inherit the logging policy's bias"
    );

    // ── 3. The PBM recovers the examination curve from the randomised slice. ─
    let mut pbm = PositionBasedModel::new(
        ClickModelConfig::default()
            .with_max_iterations(400)
            .with_tolerance(1e-10),
    );
    let fit = pbm.fit(&log).expect("fit succeeds");
    assert!(
        fit.is_monotone_relative(1e-10),
        "worst decrease {}",
        fit.worst_decrease()
    );
    let identifiability = pbm.identifiability().expect("fitted");
    assert!(
        identifiability.fully_identified,
        "the 10% randomised traffic is what makes this log identifiable: {}",
        identifiability.summary()
    );
    let examination_error = max_abs_error(pbm.examination_curve(), &true_examination);
    println!(
        "PBM examination curve: true = {true_examination:?}, fitted = {:?} (max |Δγ| = {examination_error:.4})",
        pbm.examination_curve()
    );
    assert!(examination_error < 0.03);

    // The PBM also recovers the true relevance ORDER, which raw CTR inverted.
    let alpha_mediocre = pbm.attractiveness("mediocre").expect("known");
    let alpha_excellent = pbm.attractiveness("excellent").expect("known");
    println!("PBM α: mediocre = {alpha_mediocre:.4}, excellent = {alpha_excellent:.4}");
    assert!(close(alpha_mediocre, truth["mediocre"], 0.03));
    assert!(close(alpha_excellent, truth["excellent"], 0.03));
    assert!(alpha_excellent > alpha_mediocre);

    // ── 4. The IPS-debiased ranker RECOVERS the true order. ──────────────────
    let propensities = PropensityEstimates::from_click_model(&pbm, 0.01).expect("usable curve");
    let mut debiased = DebiasedRanker::new(DebiasedRankerConfig::default());
    let debiased_fit = debiased
        .fit_tabular(&log, &propensities)
        .expect("fit succeeds");
    assert!(
        debiased_fit.is_monotone(1e-9),
        "the pairwise logistic loss is convex; descent must go downhill (worst rise {})",
        debiased_fit
            .loss_history
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(0.0_f64, f64::max)
    );
    println!(
        "IPS-debiased scores: mediocre = {:.4}, excellent = {:.4}",
        debiased.score_doc("mediocre"),
        debiased.score_doc("excellent")
    );
    assert!(
        debiased.score_doc("excellent") > debiased.score_doc("mediocre"),
        "the IPS-weighted pairwise loss must recover the TRUE order from the SAME biased clicks"
    );
    // The whole ranking, not just the pair.
    assert_eq!(
        debiased.rank(&ids(&["mediocre", "filler", "excellent"]))[0],
        "excellent"
    );

    // ── 5. And off-policy evaluation agrees: the debiased policy beats the
    //       logged one, once position bias is divided out. ────────────────────
    let logged_policy = TableClickPolicy::new()
        .with_doc_score("mediocre", 3.0)
        .with_doc_score("filler", 2.0)
        .with_doc_score("excellent", 1.0);
    let ips = IpsEstimator::new(propensities);
    let logged_value = ips.estimate(&log, &logged_policy).expect("non-empty").value;
    let debiased_value = ips.estimate(&log, &debiased).expect("non-empty").value;
    println!("IPS policy value: logged = {logged_value:.4}, debiased = {debiased_value:.4}");
    assert!(
        debiased_value > logged_value,
        "the debiased ranking must evaluate BETTER than the production one"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// DebiasedRanker
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn debiased_ranker_loss_decreases_monotonically() {
    let (log, _, _) = rigged_bias_log();
    let propensities = PropensityEstimates::new(vec![1.0, 0.4, 0.1], 0.01).expect("valid");
    let mut ranker = DebiasedRanker::new(
        DebiasedRankerConfig::default()
            .with_learning_rate(0.4)
            .with_epochs(300)
            .with_l2_regularization(1e-3)
            .with_tolerance(0.0), // never stop early: check every epoch
    );
    let fit = ranker
        .fit_tabular(&log, &propensities)
        .expect("fit succeeds");

    assert_eq!(fit.loss_history.len(), fit.epochs + 1);
    assert!(fit.is_monotone(1e-12), "convex loss + full-batch descent");
    assert!(fit.final_loss < fit.loss_history[0]);
    assert!(fit.total_pair_weight > 0.0);
    // Pair aggregation is exact: 3 documents ⇒ at most 3·2 = 6 ordered pairs.
    assert!(fit.distinct_pair_count <= 6);
    assert_eq!(ranker.feature_dim(), 3);
    assert!(ranker.is_fitted());
    assert_eq!(ranker.config().epochs, 300);
}

#[test]
fn debiased_ranker_accepts_custom_features_and_generalises_across_documents() {
    // Two features: [decisive, distractor]. The clicked document scores on the
    // first, so the learner should put its weight there.
    let mut sessions = Vec::new();
    for index in 0..200 {
        sessions.push(session(
            "q",
            &["hi", "lo"],
            &[index % 3 != 0, index % 9 == 0],
        ));
    }
    let log = log_of(sessions);
    let features = HashMap::from([
        ("hi".to_owned(), vec![1.0, 0.0]),
        ("lo".to_owned(), vec![0.0, 1.0]),
    ]);
    let propensities = PropensityEstimates::new(vec![1.0, 0.5], 0.01).expect("valid");

    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default());
    let fit = ranker
        .fit(&log, &propensities, &features)
        .expect("fit succeeds");
    assert!(fit.is_monotone(1e-12));
    assert_eq!(ranker.feature_dim(), 2);
    assert_eq!(ranker.weights().len(), 2);
    assert!(
        ranker.weights()[0] > ranker.weights()[1],
        "the decisive feature should carry the larger weight: {:?}",
        ranker.weights()
    );
    // A document the fit never saw has no features, hence no opinion.
    assert_eq!(ranker.score_doc("never-seen"), 0.0);
    assert_eq!(ranker.score_map().len(), 2);
}

#[test]
fn debiased_ranker_rejects_unusable_inputs() {
    let log = log_of(vec![session("q", &["a", "b"], &[true, false])]);
    let propensities = PropensityEstimates::uniform(2);

    // Bad hyper-parameters.
    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default().with_epochs(0));
    assert!(matches!(
        ranker.fit_tabular(&log, &propensities),
        Err(ClickModelError::InvalidConfig { .. })
    ));
    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default().with_learning_rate(-1.0));
    assert!(matches!(
        ranker.fit_tabular(&log, &propensities),
        Err(ClickModelError::InvalidConfig { .. })
    ));
    let mut ranker =
        DebiasedRanker::new(DebiasedRankerConfig::default().with_l2_regularization(-1.0));
    assert!(matches!(
        ranker.fit_tabular(&log, &propensities),
        Err(ClickModelError::InvalidConfig { .. })
    ));

    // Empty log.
    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default());
    assert!(matches!(
        ranker.fit_tabular(&ClickLog::new(), &propensities),
        Err(ClickModelError::EmptyLog)
    ));

    // A missing feature vector.
    let features = HashMap::from([("a".to_owned(), vec![1.0])]);
    assert!(matches!(
        ranker.fit(&log, &propensities, &features),
        Err(ClickModelError::UnknownDocument { doc_id }) if doc_id == "b"
    ));

    // A ragged feature vector.
    let features = HashMap::from([
        ("a".to_owned(), vec![1.0, 2.0]),
        ("b".to_owned(), vec![1.0]),
    ]);
    assert!(matches!(
        ranker.fit(&log, &propensities, &features),
        Err(ClickModelError::FeatureDimensionMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));

    // An empty feature vector.
    let features = HashMap::from([("a".to_owned(), Vec::new()), ("b".to_owned(), Vec::new())]);
    assert!(matches!(
        ranker.fit(&log, &propensities, &features),
        Err(ClickModelError::FeatureDimensionMismatch { actual: 0, .. })
    ));
}

#[test]
fn debiased_ranker_refuses_a_log_with_no_preference_signal() {
    // Every session is either all-clicked or all-unclicked, so no (clicked,
    // unclicked) pair exists anywhere. There is nothing to learn, and pretending
    // otherwise would return an untrained zero vector dressed up as a model.
    let log = log_of(vec![
        session("q", &["a", "b"], &[true, true]),
        session("q", &["a", "b"], &[false, false]),
    ]);
    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default());
    assert!(matches!(
        ranker.fit_tabular(&log, &PropensityEstimates::uniform(2)),
        Err(ClickModelError::NoPreferencePairs)
    ));
    assert!(!ranker.is_fitted());
}

#[test]
fn debiased_ranker_is_a_ranking_policy() {
    // It plugs straight into the counterfactual estimators, which is the point:
    // fit a policy on a log, then evaluate it without ever deploying it.
    let (log, _, _) = rigged_bias_log();
    let propensities = PropensityEstimates::new(vec![1.0, 0.4, 0.1], 0.01).expect("valid");
    let mut ranker = DebiasedRanker::new(DebiasedRankerConfig::default());
    ranker
        .fit_tabular(&log, &propensities)
        .expect("fit succeeds");

    let policy: &dyn ClickRankingPolicy = &ranker;
    assert_eq!(
        policy.score("q0", "excellent"),
        ranker.score_doc("excellent")
    );
    let estimate = IpsEstimator::new(propensities)
        .estimate(&log, policy)
        .expect("non-empty log");
    assert!(estimate.value > 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// The deterministic PRNG and the simulator
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn splitmix64_is_deterministic_and_reproducible() {
    let mut first = ClickSplitMix64::new(42);
    let mut second = ClickSplitMix64::new(42);
    let mut third = ClickSplitMix64::new(43);
    let a: Vec<u64> = (0..8).map(|_| first.next_u64()).collect();
    let b: Vec<u64> = (0..8).map(|_| second.next_u64()).collect();
    let c: Vec<u64> = (0..8).map(|_| third.next_u64()).collect();
    assert_eq!(a, b, "the same seed must produce the same stream");
    assert_ne!(a, c, "a different seed must produce a different stream");
}

#[test]
fn splitmix64_draws_are_well_distributed() {
    let mut rng = ClickSplitMix64::new(7);
    for _ in 0..10_000 {
        let value = rng.next_f64();
        assert!(
            (0.0..1.0).contains(&value),
            "next_f64 out of range: {value}"
        );
    }
    let mut rng = ClickSplitMix64::new(9);
    let successes = (0..20_000).filter(|_| rng.bernoulli(0.3)).count();
    let rate = successes as f64 / 20_000.0;
    assert!(close(rate, 0.3, 0.02), "Bernoulli(0.3) produced {rate}");

    assert!(!ClickSplitMix64::new(1).bernoulli(0.0));
    assert!(ClickSplitMix64::new(1).bernoulli(1.0));
    assert_eq!(ClickSplitMix64::new(1).below(0), 0);
}

#[test]
fn splitmix64_shuffle_is_a_permutation() {
    let mut rng = ClickSplitMix64::new(3);
    for _ in 0..50 {
        let mut items: Vec<usize> = (0..10).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<usize>>());
    }
}

#[test]
fn simulator_validates_its_parameters() {
    let simulator = ClickLogSimulator::new(ids(&["a", "b"]));
    assert!(
        simulator
            .clone()
            .with_attractiveness(vec![0.5])
            .simulate(ClickSimulationModel::PositionBased, 10)
            .is_err(),
        "one attractiveness for two documents"
    );
    assert!(
        simulator
            .clone()
            .with_slate_size(5)
            .simulate(ClickSimulationModel::PositionBased, 10)
            .is_err(),
        "a slate deeper than the document pool"
    );
    assert!(
        simulator
            .clone()
            .with_examination(vec![1.0])
            .simulate(ClickSimulationModel::PositionBased, 10)
            .is_err(),
        "an examination curve shallower than the slate"
    );
    assert!(
        simulator
            .clone()
            .with_satisfaction(vec![0.5])
            .simulate(ClickSimulationModel::Dbn, 10)
            .is_err(),
        "one satisfaction for two documents"
    );
    assert!(
        ClickLogSimulator::new(Vec::new())
            .simulate(ClickSimulationModel::Cascade, 10)
            .is_err(),
        "no documents at all"
    );
    assert!(matches!(
        simulator
            .clone()
            .simulate_rankings(ClickSimulationModel::PositionBased, &[ids(&["ghost"])]),
        Err(ClickModelError::UnknownDocument { .. })
    ));
}

#[test]
fn simulator_obeys_the_generative_stories_it_claims() {
    let simulator = ClickLogSimulator::new(ids(&["a", "b", "c"]))
        .with_attractiveness(vec![0.6, 0.5, 0.4])
        .with_satisfaction(vec![0.5, 0.5, 0.5])
        .with_persistence(0.8)
        .with_examination(vec![1.0, 0.7, 0.4])
        .with_query_count(3)
        .with_seed(17);

    // Cascade: at most one click, and nothing below it is examined.
    let cascade = simulator
        .simulate(ClickSimulationModel::Cascade, 500)
        .expect("valid");
    for s in cascade.sessions() {
        assert!(
            s.click_count() <= 1,
            "a cascade session clicks at most once"
        );
        let examinations = s.examinations.as_ref().expect("simulator records them");
        let examined_depth = cascade_examined_depth(s);
        for rank in 0..s.len() {
            assert_eq!(
                examinations[rank],
                rank < examined_depth,
                "cascade examination must stop at the first click"
            );
        }
    }

    // PBM: every position is drawn independently, so multiple clicks are ordinary.
    let pbm = simulator
        .simulate(ClickSimulationModel::PositionBased, 500)
        .expect("valid");
    assert!(pbm.sessions().iter().any(|s| s.click_count() > 1));
    for s in pbm.sessions() {
        let examinations = s.examinations.as_ref().expect("recorded");
        for rank in 0..s.len() {
            assert!(!s.clicks[rank] || examinations[rank], "click ⇒ examined");
        }
    }

    // DBN: examination is a prefix (monotone), and a satisfied click ends it.
    let dbn = simulator
        .simulate(ClickSimulationModel::Dbn, 500)
        .expect("valid");
    for s in dbn.sessions() {
        let examinations = s.examinations.as_ref().expect("recorded");
        assert!(examinations[0], "the top rank is always examined");
        let mut seen_unexamined = false;
        for &examined in examinations {
            if examined {
                assert!(!seen_unexamined, "DBN examination must be a prefix");
            } else {
                seen_unexamined = true;
            }
        }
    }

    // Query ids cycle as configured.
    assert_eq!(pbm.sessions()[0].query_id, "q0");
    assert_eq!(pbm.sessions()[1].query_id, "q1");
    assert_eq!(pbm.sessions()[3].query_id, "q0");
}

#[test]
fn simulator_exposes_the_ground_truth_it_generated_from() {
    let simulator = ClickLogSimulator::new(ids(&["a", "b"]))
        .with_attractiveness(vec![0.7, 0.2])
        .with_satisfaction(vec![0.4, 0.6])
        .with_examination(vec![1.0, 0.3]);
    assert_eq!(simulator.true_attractiveness(), &[0.7, 0.2]);
    assert_eq!(simulator.true_satisfaction(), &[0.4, 0.6]);
    assert_eq!(simulator.true_examination(), &[1.0, 0.3]);
}
