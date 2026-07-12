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
//! Tests for the public data types and for the shared per-arm ridge model.

use super::max_abs_diff;
use crate::bandit_ranker::linalg::scaled_identity;
use crate::bandit_ranker::linucb::LinUcbRanker;
use crate::bandit_ranker::model::{BanditArmSet, BanditLinearModel};
use crate::bandit_ranker::types::{
    BanditArm, BanditConfig, BanditContext, BanditError, BanditRanker,
};

// ═════════════════════════════════════════════════════════════════════════════
// types
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn bandit_config_validates_every_field() {
    assert!(matches!(
        BanditConfig::with_dimension(0),
        Err(BanditError::InvalidConfig { .. })
    ));

    let base = BanditConfig::with_dimension(3).expect("valid");
    assert!(base.validate().is_ok());

    assert!(matches!(
        base.clone().set_alpha(-1.0).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(
        base.clone().set_alpha(0.0).validate().is_ok(),
        "alpha = 0 is the pure-greedy LinUCB, which is legal and instructive"
    );
    assert!(matches!(
        base.clone().set_alpha(f64::NAN).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        base.clone().set_exploration_variance(0.0).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        base.clone().set_epsilon(1.5).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        base.clone().set_epsilon(-0.1).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(base.clone().set_epsilon(0.0).validate().is_ok());
    assert!(base.clone().set_epsilon(1.0).validate().is_ok());
    assert!(matches!(
        base.clone().set_ridge_lambda(0.0).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
    assert!(matches!(
        base.set_ridge_lambda(-1.0).validate(),
        Err(BanditError::InvalidConfig { .. })
    ));
}

#[test]
fn bandit_context_rejects_non_finite_features() {
    assert!(matches!(
        BanditContext::new(vec![1.0, f64::NAN]),
        Err(BanditError::NonFinite { .. })
    ));
    assert!(matches!(
        BanditContext::new(vec![f64::INFINITY]),
        Err(BanditError::NonFinite { .. })
    ));

    let context = BanditContext::new(vec![1.0, 2.0])
        .expect("finite")
        .with_query_id("q-7");
    assert_eq!(context.dimension(), 2);
    assert_eq!(context.features(), &[1.0, 2.0]);
    assert_eq!(context.query_id(), Some("q-7"));
    assert!(context.require_dimension(2).is_ok());
    assert!(matches!(
        context.require_dimension(3),
        Err(BanditError::DimensionMismatch {
            expected: 3,
            actual: 2
        })
    ));
}

#[test]
fn bandit_arm_carries_an_optional_label() {
    let arm = BanditArm::new("dense").with_label("Dense vector retrieval");
    assert_eq!(arm.id, "dense");
    assert_eq!(arm.label.as_deref(), Some("Dense vector retrieval"));
    assert_eq!(BanditArm::new("bm25").label, None);
}

#[test]
fn bandit_stats_summarize_concentration() {
    let config = BanditConfig::with_dimension(2).expect("valid");
    let mut ranker = LinUcbRanker::from_ids(config, ["a", "b"]).expect("distinct ids");
    let context = BanditContext::new(vec![1.0, 0.0]).expect("finite");

    assert_eq!(
        ranker.stats().concentration(),
        0.0,
        "no pulls, no concentration"
    );
    assert_eq!(
        ranker.stats().most_pulled_arm(),
        Some("a"),
        "ties break by order"
    );

    for _ in 0..9 {
        ranker.update("a", &context, 1.0).expect("known arm");
    }
    ranker.update("b", &context, 0.0).expect("known arm");

    let stats = ranker.stats();
    assert_eq!(stats.policy, "linucb");
    assert_eq!(stats.total_pulls, 10);
    assert_eq!(stats.total_reward, 9.0);
    assert_eq!(stats.mean_reward, 0.9);
    assert_eq!(stats.most_pulled_arm(), Some("a"));
    assert!((stats.concentration() - 0.9).abs() < 1e-12);
    assert_eq!(stats.arms[0].pulls, 9);
    assert_eq!(stats.arms[1].pulls, 1);
}

// ═════════════════════════════════════════════════════════════════════════════
// model
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn linear_model_starts_at_the_prior() {
    let model = BanditLinearModel::new(BanditArm::new("a"), 3, 2.0);
    assert_eq!(model.theta(), &[0.0, 0.0, 0.0]);
    assert_eq!(model.pulls(), 0);
    assert_eq!(model.mean_reward(), 0.0);
    assert_eq!(model.design_matrix(), &scaled_identity(3, 2.0));
    // A = 2I  =>  A^-1 = 0.5I, known in closed form: no inversion at startup.
    assert_eq!(model.design_inverse(), &scaled_identity(3, 0.5));
}

#[test]
fn linear_model_update_matches_the_ridge_normal_equations() {
    // One observation (x, r) with lambda = 1: A = I + x x^T, b = r x, and the
    // closed-form ridge solution is theta = r x / (1 + ||x||^2).
    let mut model = BanditLinearModel::new(BanditArm::new("a"), 2, 1.0);
    let x = [3.0, 4.0]; // ||x||^2 = 25
    model.update(&x, 2.0).expect("valid update");

    let denominator = 1.0 + 25.0;
    let expected = [2.0 * 3.0 / denominator, 2.0 * 4.0 / denominator];
    assert!(max_abs_diff(model.theta(), &expected) < 1e-12);
    assert_eq!(model.pulls(), 1);
    assert_eq!(model.total_reward(), 2.0);
    assert_eq!(model.mean_reward(), 2.0);

    // And the design matrix really is I + x x^T.
    assert_eq!(model.design_matrix(), &[1.0 + 9.0, 12.0, 12.0, 1.0 + 16.0]);
}

#[test]
fn linear_model_zero_context_is_an_information_free_pull() {
    let mut model = BanditLinearModel::new(BanditArm::new("a"), 3, 1.0);
    let before_inverse = model.design_inverse().to_vec();
    model
        .update(&[0.0, 0.0, 0.0], 5.0)
        .expect("a zero context is legal");

    assert_eq!(
        model.theta(),
        &[0.0, 0.0, 0.0],
        "no direction, nothing learned"
    );
    assert_eq!(model.design_inverse(), before_inverse.as_slice());
    assert_eq!(
        model.pulls(),
        1,
        "the pull still happened and is still counted"
    );
    assert_eq!(model.total_reward(), 5.0);
    assert!(model.theta().iter().all(|value| value.is_finite()));
}

#[test]
fn linear_model_confidence_width_shrinks_where_it_has_looked() {
    let mut model = BanditLinearModel::new(BanditArm::new("a"), 2, 1.0);
    let observed = [1.0, 0.0];
    let unobserved = [0.0, 1.0];

    let width_before = model.confidence_width(&observed).expect("dims");
    let orthogonal_before = model.confidence_width(&unobserved).expect("dims");
    assert!(
        (width_before - 1.0).abs() < 1e-12,
        "prior width is sqrt(1/lambda)"
    );

    for _ in 0..100 {
        model.update(&observed, 1.0).expect("valid");
    }

    let width_after = model.confidence_width(&observed).expect("dims");
    let orthogonal_after = model.confidence_width(&unobserved).expect("dims");

    // Exactly: A = (1 + 100) in the observed direction, so the width is 1/sqrt(101).
    assert!((width_after - (1.0_f64 / 101.0).sqrt()).abs() < 1e-12);
    assert!(
        width_after < width_before * 0.11,
        "the width must collapse in the direction that was observed"
    );
    // The *orthogonal* direction is untouched -- this is what makes it a
    // *contextual* bandit rather than K independent scalar ones.
    assert!((orthogonal_after - orthogonal_before).abs() < 1e-12);
}

#[test]
fn linear_model_rejects_bad_input() {
    let mut model = BanditLinearModel::new(BanditArm::new("a"), 2, 1.0);
    assert!(matches!(
        model.update(&[1.0, 1.0], f64::NAN),
        Err(BanditError::NonFinite { what: "reward", .. })
    ));
    assert!(matches!(
        model.update(&[1.0, 1.0], f64::INFINITY),
        Err(BanditError::NonFinite { .. })
    ));
    assert!(matches!(
        model.update(&[1.0], 1.0),
        Err(BanditError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        model.mean_estimate(&[1.0, 2.0, 3.0]),
        Err(BanditError::DimensionMismatch { .. })
    ));
    assert_eq!(model.pulls(), 0, "a rejected update must not be counted");
}

#[test]
fn arm_set_rejects_duplicates_and_looks_up_by_id() {
    let mut set = BanditArmSet::new([BanditArm::new("a"), BanditArm::new("b")], 2, 1.0)
        .expect("distinct ids");
    assert_eq!(set.len(), 2);
    assert!(!set.is_empty());
    assert_eq!(set.arm_ids(), vec!["a", "b"]);
    assert!(set.get("a").is_some());
    assert!(set.get("zzz").is_none());
    assert!(matches!(
        set.get_mut("zzz"),
        Err(BanditError::UnknownArm { .. })
    ));
    assert!(matches!(
        set.add(BanditArm::new("a")),
        Err(BanditError::DuplicateArm { .. })
    ));
    assert!(set.add(BanditArm::new("c")).is_ok());
    assert_eq!(
        set.arm_ids(),
        vec!["a", "b", "c"],
        "registration order is preserved"
    );

    let empty = BanditArmSet::new(Vec::new(), 2, 1.0).expect("empty is legal");
    assert!(empty.is_empty());
}
