#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::doc_markdown
)]
//! Tests for contextual-bandit online ranking.
//! Edge cases: zero arms, one arm, a zero context, `d = 1`, the ε extremes, and
//! every invalid-input path.

use crate::bandit_ranker::epsilon_greedy::EpsilonGreedyRanker;
use crate::bandit_ranker::linucb::LinUcbRanker;
use crate::bandit_ranker::thompson::ThompsonSamplingRanker;
use crate::bandit_ranker::types::{BanditConfig, BanditContext, BanditError, BanditRanker};

// ═════════════════════════════════════════════════════════════════════════════
// Edge cases
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn a_bandit_with_no_arms_refuses_to_select() {
    let config = BanditConfig::with_dimension(2).expect("valid");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    let empty: Vec<String> = Vec::new();
    let mut linucb = LinUcbRanker::from_ids(config.clone(), empty.clone()).expect("empty is legal");
    let mut thompson =
        ThompsonSamplingRanker::from_ids(config.clone(), empty.clone()).expect("empty is legal");
    let mut greedy = EpsilonGreedyRanker::from_ids(config, empty).expect("empty is legal");

    // "The best of nothing" is not a question with an answer, so it is an error --
    // not an empty ranking a caller would silently act on.
    assert!(matches!(linucb.select(&context), Err(BanditError::NoArms)));
    assert!(matches!(
        thompson.select(&context),
        Err(BanditError::NoArms)
    ));
    assert!(matches!(greedy.select(&context), Err(BanditError::NoArms)));

    assert!(matches!(
        linucb.update("anything", &context, 1.0),
        Err(BanditError::UnknownArm { .. })
    ));
    assert!(linucb.arm_ids().is_empty());
    assert_eq!(linucb.stats().total_pulls, 0);
    assert_eq!(linucb.stats().most_pulled_arm(), None);
    assert_eq!(linucb.stats().concentration(), 0.0);
}

#[test]
fn a_single_arm_bandit_always_returns_that_arm() {
    let config = BanditConfig::with_dimension(2).expect("valid");
    let context = BanditContext::new(vec![0.5, -0.5]).expect("finite");

    let mut linucb = LinUcbRanker::from_ids(config.clone(), ["only"]).expect("one arm");
    let mut thompson = ThompsonSamplingRanker::from_ids(config.clone(), ["only"]).expect("one arm");
    let mut greedy =
        EpsilonGreedyRanker::from_ids(config.set_epsilon(1.0).set_decay_epsilon(false), ["only"])
            .expect("one arm");

    for _ in 0..50 {
        for ranking in [
            linucb.select(&context).expect("one arm"),
            thompson.select(&context).expect("one arm"),
            greedy.select(&context).expect("one arm"),
        ] {
            assert_eq!(ranking.len(), 1);
            assert_eq!(ranking.top_arm_id(), Some("only"));
            assert_eq!(ranking.ranked[0].rank, 0);
        }
        linucb.update("only", &context, 1.0).expect("known arm");
        thompson.update("only", &context, 1.0).expect("known arm");
        greedy.update("only", &context, 1.0).expect("known arm");
    }
    // Even with epsilon = 1, "explore uniformly over one arm" is still that arm.
    assert_eq!(greedy.exploration_rounds(), 50);
}

#[test]
fn a_zero_context_produces_finite_tied_scores_and_learns_nothing() {
    let config = BanditConfig::with_dimension(3)
        .expect("valid")
        .set_alpha(1.5)
        .set_exploration_variance(0.5)
        .set_seed(2);
    let zero = BanditContext::new(vec![0.0, 0.0, 0.0]).expect("a zero context is legal");

    let mut linucb = LinUcbRanker::from_ids(config.clone(), ["a", "b", "c"]).expect("distinct ids");
    let ranking = linucb.select(&zero).expect("arms exist");
    for arm in &ranking.ranked {
        assert_eq!(arm.score, 0.0, "theta^T 0 + alpha * sqrt(0^T A^-1 0) = 0");
        assert_eq!(arm.mean_estimate, 0.0);
        assert!(arm.score.is_finite());
    }
    assert_eq!(
        ranking.arm_ids(),
        vec!["a", "b", "c"],
        "an exact tie must break by registration order, deterministically"
    );

    // Updating on a zero context is a legal, information-free no-op.
    linucb.update("a", &zero, 42.0).expect("legal");
    assert_eq!(linucb.theta("a").expect("known arm"), &[0.0, 0.0, 0.0]);
    assert_eq!(linucb.stats().arms[0].pulls, 1, "the pull still happened");

    // Thompson's posterior draw at a zero context has zero variance, so its score
    // is exactly 0 too -- no NaN from the Cholesky, no jitter needed.
    let mut thompson =
        ThompsonSamplingRanker::from_ids(config, ["a", "b", "c"]).expect("distinct ids");
    let ranking = thompson.select(&zero).expect("arms exist");
    for arm in &ranking.ranked {
        assert_eq!(arm.score, 0.0, "theta_tilde^T 0 = 0 for *any* draw");
    }
    thompson.update("a", &zero, 1.0).expect("legal");
    assert_eq!(thompson.jitter_events(), 0);
}

#[test]
fn every_policy_rejects_a_context_of_the_wrong_dimension() {
    let config = BanditConfig::with_dimension(3).expect("valid");
    let wrong = BanditContext::new(vec![1.0, 2.0]).expect("finite");

    let mut linucb = LinUcbRanker::from_ids(config.clone(), ["a"]).expect("one arm");
    let mut thompson = ThompsonSamplingRanker::from_ids(config.clone(), ["a"]).expect("one arm");
    let mut greedy = EpsilonGreedyRanker::from_ids(config, ["a"]).expect("one arm");

    for result in [
        linucb.select(&wrong).err(),
        thompson.select(&wrong).err(),
        greedy.select(&wrong).err(),
        linucb.update("a", &wrong, 1.0).err(),
        thompson.update("a", &wrong, 1.0).err(),
        greedy.update("a", &wrong, 1.0).err(),
    ] {
        assert!(matches!(
            result,
            Some(BanditError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }
}

#[test]
fn every_policy_rejects_a_non_finite_reward() {
    let config = BanditConfig::with_dimension(2).expect("valid");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    let mut linucb = LinUcbRanker::from_ids(config.clone(), ["a"]).expect("one arm");
    let mut thompson = ThompsonSamplingRanker::from_ids(config.clone(), ["a"]).expect("one arm");
    let mut greedy = EpsilonGreedyRanker::from_ids(config, ["a"]).expect("one arm");

    for result in [
        linucb.update("a", &context, f64::NAN).err(),
        thompson.update("a", &context, f64::INFINITY).err(),
        greedy.update("a", &context, f64::NEG_INFINITY).err(),
    ] {
        assert!(matches!(result, Some(BanditError::NonFinite { .. })));
    }
    // And nothing was corrupted by the rejected updates.
    assert_eq!(linucb.theta("a").expect("known arm"), &[0.0, 0.0]);
    assert_eq!(linucb.stats().total_pulls, 0);
}

#[test]
fn every_policy_rejects_a_duplicate_arm_and_an_unknown_arm() {
    let config = BanditConfig::with_dimension(2).expect("valid");
    let context = BanditContext::new(vec![1.0, 1.0]).expect("finite");

    assert!(matches!(
        LinUcbRanker::from_ids(config.clone(), ["a", "a"]),
        Err(BanditError::DuplicateArm { .. })
    ));
    assert!(matches!(
        ThompsonSamplingRanker::from_ids(config.clone(), ["a", "a"]),
        Err(BanditError::DuplicateArm { .. })
    ));
    assert!(matches!(
        EpsilonGreedyRanker::from_ids(config.clone(), ["a", "a"]),
        Err(BanditError::DuplicateArm { .. })
    ));

    let mut ranker = LinUcbRanker::from_ids(config, ["a"]).expect("one arm");
    assert!(matches!(
        ranker.update("ghost", &context, 1.0),
        Err(BanditError::UnknownArm { .. })
    ));
}

#[test]
fn an_invalid_config_is_rejected_by_every_constructor() {
    let bad = BanditConfig {
        dimension: 0,
        ..BanditConfig::default()
    };
    assert!(matches!(
        LinUcbRanker::from_ids(bad.clone(), ["a"]),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        ThompsonSamplingRanker::from_ids(bad.clone(), ["a"]),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        EpsilonGreedyRanker::from_ids(bad, ["a"]),
        Err(BanditError::InvalidConfig { .. })
    ));
}

#[test]
fn a_larger_ridge_lambda_shrinks_the_prior_and_the_bonus() {
    // lambda is the prior precision: A = lambda I, so A^-1 = (1/lambda) I and the
    // prior confidence width is 1/sqrt(lambda). A bigger lambda is a *more*
    // confident prior, hence a smaller exploration bonus at t = 0.
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");
    for lambda in [0.25_f64, 1.0, 4.0] {
        let config = BanditConfig::with_dimension(2)
            .expect("valid")
            .set_alpha(1.0)
            .set_ridge_lambda(lambda);
        let ranker = LinUcbRanker::from_ids(config, ["a"]).expect("one arm");
        let bonus = ranker.exploration_bonus("a", &context).expect("known arm");
        assert!(
            (bonus - 1.0 / lambda.sqrt()).abs() < 1e-12,
            "the prior bonus must be alpha / sqrt(lambda), got {bonus} at lambda = {lambda}"
        );
    }
}
