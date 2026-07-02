//! Comprehensive tests for the `learning_to_rank` module.
//!
//! Coverage spans: feature-extraction determinism and correctness, feature
//! vector / config / training-set validation, the numerically stable sigmoid
//! and softplus, the training loop's monotonic loss decrease on a separable
//! synthetic dataset, closed-form gradient checks, learned held-out ranking,
//! early stopping, determinism, and every error path.

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::default_trait_access,
    clippy::default_constructed_unit_structs
)]

use crate::learning_to_rank::engine::LtrEngine;
use crate::learning_to_rank::features::LtrFeatureExtractor;
use crate::learning_to_rank::train::{sigmoid, softplus};
use crate::learning_to_rank::types::{
    LTR_DEFAULT_FEATURE_DIM, LTR_FEATURE_BM25, LTR_FEATURE_EMBEDDING_SIM, LTR_FEATURE_LENGTH,
    LTR_FEATURE_POPULARITY, LTR_FEATURE_RECENCY, LtrConfig, LtrDocument, LtrError,
    LtrFeatureVector, LtrModel, LtrTrainingPair, LtrTrainingSet,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

/// A 1-D preference pair `(higher, lower)` with label `1.0`.
fn pair_1d(query_id: u64, higher: f64, lower: f64) -> LtrTrainingPair {
    LtrTrainingPair::preferred(
        query_id,
        LtrFeatureVector::new(vec![higher]),
        LtrFeatureVector::new(vec![lower]),
    )
}

/// A separable multi-feature training set: dims 0 and 1 both favour `doc_a`,
/// dim 2 is identical between the two documents (pure, non-discriminative
/// noise). The unique-optimum objective is convex, so full-batch gradient
/// descent must decrease the loss monotonically.
fn separable_set() -> LtrTrainingSet {
    let rows = [
        (0.90, 0.80, 0.10, 0.05, 0.5),
        (0.85, 0.70, 0.20, 0.15, 0.3),
        (0.75, 0.60, 0.30, 0.20, 0.7),
        (0.95, 0.88, 0.05, 0.02, 0.9),
        (0.70, 0.55, 0.40, 0.30, 0.1),
        (0.80, 0.66, 0.25, 0.12, 0.4),
    ];
    let mut set = LtrTrainingSet::new();
    for (i, &(a0, a1, b0, b1, noise)) in rows.iter().enumerate() {
        set.push(LtrTrainingPair::preferred(
            i as u64,
            LtrFeatureVector::new(vec![a0, a1, noise]),
            LtrFeatureVector::new(vec![b0, b1, noise]),
        ));
    }
    set
}

// ── numerically stable sigmoid / softplus ────────────────────────────────────

#[test]
fn sigmoid_at_zero_is_half() {
    assert_eq!(sigmoid(0.0), 0.5);
}

#[test]
fn sigmoid_is_monotonic() {
    assert!(sigmoid(-1.0) < sigmoid(0.0));
    assert!(sigmoid(0.0) < sigmoid(1.0));
    assert!(sigmoid(1.0) < sigmoid(2.0));
}

#[test]
fn sigmoid_stable_on_large_positive() {
    let s = sigmoid(1_000.0);
    assert!(s.is_finite());
    assert!(close(s, 1.0, 1e-9));
}

#[test]
fn sigmoid_stable_on_large_negative() {
    let s = sigmoid(-1_000.0);
    assert!(s.is_finite());
    assert!(close(s, 0.0, 1e-9));
    assert!(s >= 0.0);
}

#[test]
fn sigmoid_never_nan_or_inf_on_extremes() {
    for &x in &[-1e300, -1e9, -50.0, 50.0, 1e9, 1e300] {
        let s = sigmoid(x);
        assert!(s.is_finite(), "sigmoid({x}) was not finite");
        assert!((0.0..=1.0).contains(&s));
    }
}

#[test]
fn softplus_matches_reference_for_small_values() {
    // softplus(0) = ln 2.
    assert!(close(softplus(0.0), 2.0_f64.ln(), 1e-12));
    // softplus(1) = ln(1 + e).
    assert!(close(softplus(1.0), (1.0 + 1.0_f64.exp()).ln(), 1e-12));
}

#[test]
fn softplus_stable_on_extremes() {
    // For large x, softplus(x) ~= x.
    assert!(close(softplus(1_000.0), 1_000.0, 1e-6));
    // For large negative x, softplus(x) ~= 0.
    assert!(close(softplus(-1_000.0), 0.0, 1e-9));
    assert!(softplus(1e300).is_finite());
    assert!(softplus(-1e300).is_finite());
}

// ── LtrFeatureVector ─────────────────────────────────────────────────────────

#[test]
fn feature_vector_dim_and_accessors() {
    let v = LtrFeatureVector::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.dim(), 3);
    assert!(!v.is_empty());
    assert_eq!(v.get(1), Some(2.0));
    assert_eq!(v.get(9), None);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn feature_vector_from_slice_and_signals() {
    let v = LtrFeatureVector::from_slice(&[0.1, 0.2]);
    assert_eq!(v.values, vec![0.1, 0.2]);
    let s = LtrFeatureVector::from_signals(0.5, 0.6, 0.7, 0.8, 0.9);
    assert_eq!(s.dim(), 5);
    assert_eq!(s.get(LTR_FEATURE_BM25), Some(0.5));
    assert_eq!(s.get(LTR_FEATURE_RECENCY), Some(0.6));
    assert_eq!(s.get(LTR_FEATURE_EMBEDDING_SIM), Some(0.7));
    assert_eq!(s.get(LTR_FEATURE_POPULARITY), Some(0.8));
    assert_eq!(s.get(LTR_FEATURE_LENGTH), Some(0.9));
}

#[test]
fn feature_vector_dot_product() {
    let v = LtrFeatureVector::new(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.dot(&[1.0, 1.0, 1.0]), 6.0);
    assert_eq!(v.dot(&[2.0, 0.0, 1.0]), 5.0);
    // Common-prefix dot never panics on mismatch.
    assert_eq!(v.dot(&[1.0]), 1.0);
}

#[test]
fn feature_vector_validate_ok() {
    let v = LtrFeatureVector::new(vec![0.1, 0.2, 0.3]);
    assert!(v.validate(3).is_ok());
}

#[test]
fn feature_vector_validate_dim_mismatch() {
    let v = LtrFeatureVector::new(vec![0.1, 0.2]);
    assert_eq!(
        v.validate(5),
        Err(LtrError::DimensionMismatch {
            expected: 5,
            found: 2
        })
    );
}

#[test]
fn feature_vector_validate_non_finite() {
    let v = LtrFeatureVector::new(vec![0.1, f64::NAN, 0.3]);
    assert_eq!(v.validate(3), Err(LtrError::NonFiniteFeature { index: 1 }));
    let w = LtrFeatureVector::new(vec![f64::INFINITY]);
    assert_eq!(w.validate(1), Err(LtrError::NonFiniteFeature { index: 0 }));
}

// ── LtrDocument ──────────────────────────────────────────────────────────────

#[test]
fn document_builder() {
    let d = LtrDocument::new("d1", "hello world")
        .with_timestamp(42)
        .with_popularity(7);
    assert_eq!(d.id, "d1");
    assert_eq!(d.content, "hello world");
    assert_eq!(d.timestamp, 42);
    assert_eq!(d.popularity, 7);
}

// ── LtrTrainingPair ──────────────────────────────────────────────────────────

#[test]
fn training_pair_new_and_preferred() {
    let a = LtrFeatureVector::new(vec![1.0, 2.0]);
    let b = LtrFeatureVector::new(vec![0.0, 1.0]);
    let p = LtrTrainingPair::new(3, a.clone(), b.clone(), 0.75);
    assert_eq!(p.query_id, 3);
    assert_eq!(p.label, 0.75);
    let q = LtrTrainingPair::preferred(3, a, b);
    assert_eq!(q.label, 1.0);
}

#[test]
fn training_pair_difference() {
    let p = LtrTrainingPair::preferred(
        0,
        LtrFeatureVector::new(vec![0.9, 0.2, 0.5]),
        LtrFeatureVector::new(vec![0.1, 0.8, 0.5]),
    );
    let diff = p.difference();
    assert!(close(diff[0], 0.8, 1e-12));
    assert!(close(diff[1], -0.6, 1e-12));
    assert!(close(diff[2], 0.0, 1e-12));
}

#[test]
fn training_pair_validate_paths() {
    let good = LtrTrainingPair::new(
        0,
        LtrFeatureVector::new(vec![0.5, 0.5]),
        LtrFeatureVector::new(vec![0.4, 0.4]),
        0.5,
    );
    assert!(good.validate(2).is_ok());

    let bad_label = LtrTrainingPair::new(
        0,
        LtrFeatureVector::new(vec![0.5, 0.5]),
        LtrFeatureVector::new(vec![0.4, 0.4]),
        1.5,
    );
    assert_eq!(bad_label.validate(2), Err(LtrError::InvalidLabel(1.5)));

    let bad_dim = LtrTrainingPair::preferred(
        0,
        LtrFeatureVector::new(vec![0.5]),
        LtrFeatureVector::new(vec![0.4]),
    );
    assert_eq!(
        bad_dim.validate(2),
        Err(LtrError::DimensionMismatch {
            expected: 2,
            found: 1
        })
    );
}

// ── LtrTrainingSet ───────────────────────────────────────────────────────────

#[test]
fn training_set_construction_and_len() {
    let mut set = LtrTrainingSet::new();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert_eq!(set.feature_dim(), None);

    set.push(pair_1d(0, 0.9, 0.1));
    assert_eq!(set.len(), 1);
    assert!(!set.is_empty());
    assert_eq!(set.feature_dim(), Some(1));

    let set2 = LtrTrainingSet::new()
        .with_pair(pair_1d(0, 0.8, 0.2))
        .with_pair(pair_1d(1, 0.7, 0.3));
    assert_eq!(set2.len(), 2);

    let set3 = LtrTrainingSet::from_pairs(vec![pair_1d(0, 0.6, 0.4)]);
    assert_eq!(set3.len(), 1);
}

#[test]
fn training_set_validate_empty() {
    let set = LtrTrainingSet::new();
    assert_eq!(set.validate(3), Err(LtrError::EmptyTrainingSet));
}

#[test]
fn training_set_validate_dim_mismatch() {
    let set = LtrTrainingSet::from_pairs(vec![pair_1d(0, 0.9, 0.1)]);
    assert_eq!(
        set.validate(3),
        Err(LtrError::DimensionMismatch {
            expected: 3,
            found: 1
        })
    );
}

#[test]
fn training_set_validate_bad_label() {
    let set = LtrTrainingSet::from_pairs(vec![LtrTrainingPair::new(
        0,
        LtrFeatureVector::new(vec![0.9]),
        LtrFeatureVector::new(vec![0.1]),
        -0.2,
    )]);
    assert_eq!(set.validate(1), Err(LtrError::InvalidLabel(-0.2)));
}

// ── LtrConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let c = LtrConfig::default();
    assert_eq!(c.learning_rate, 0.1);
    assert_eq!(c.epochs, 200);
    assert_eq!(c.feature_dim, LTR_DEFAULT_FEATURE_DIM);
    assert_eq!(c.l2_regularization, 0.0);
    assert_eq!(c.weight_init_scale, 0.0);
    assert!(c.validate().is_ok());
}

#[test]
fn config_builders() {
    let c = LtrConfig::new()
        .with_learning_rate(0.5)
        .with_epochs(42)
        .with_tolerance(1e-4)
        .with_feature_dim(7)
        .with_seed(99)
        .with_l2_regularization(0.01)
        .with_weight_init_scale(0.25);
    assert_eq!(c.learning_rate, 0.5);
    assert_eq!(c.epochs, 42);
    assert_eq!(c.tolerance, 1e-4);
    assert_eq!(c.feature_dim, 7);
    assert_eq!(c.seed, 99);
    assert_eq!(c.l2_regularization, 0.01);
    assert_eq!(c.weight_init_scale, 0.25);
    assert!(c.validate().is_ok());
}

#[test]
fn config_invalid_learning_rate() {
    assert_eq!(
        LtrConfig::new().with_learning_rate(-1.0).validate(),
        Err(LtrError::InvalidLearningRate(-1.0))
    );
    assert_eq!(
        LtrConfig::new().with_learning_rate(0.0).validate(),
        Err(LtrError::InvalidLearningRate(0.0))
    );
    assert!(matches!(
        LtrConfig::new().with_learning_rate(f64::NAN).validate(),
        Err(LtrError::InvalidLearningRate(_))
    ));
}

#[test]
fn config_zero_epochs() {
    assert_eq!(
        LtrConfig::new().with_epochs(0).validate(),
        Err(LtrError::ZeroEpochs)
    );
}

#[test]
fn config_invalid_tolerance() {
    assert_eq!(
        LtrConfig::new().with_tolerance(-1e-3).validate(),
        Err(LtrError::InvalidTolerance(-1e-3))
    );
    assert!(matches!(
        LtrConfig::new().with_tolerance(f64::INFINITY).validate(),
        Err(LtrError::InvalidTolerance(_))
    ));
}

#[test]
fn config_zero_feature_dim() {
    assert_eq!(
        LtrConfig::new().with_feature_dim(0).validate(),
        Err(LtrError::ZeroFeatureDimension)
    );
}

#[test]
fn config_invalid_regularization_and_scale() {
    assert_eq!(
        LtrConfig::new().with_l2_regularization(-0.5).validate(),
        Err(LtrError::InvalidRegularization(-0.5))
    );
    assert_eq!(
        LtrConfig::new().with_weight_init_scale(-0.5).validate(),
        Err(LtrError::InvalidInitScale(-0.5))
    );
}

// ── training: error paths ────────────────────────────────────────────────────

#[test]
fn train_rejects_invalid_config() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let bad = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(-1.0);
    assert_eq!(
        engine.train(&set, &bad),
        Err(LtrError::InvalidLearningRate(-1.0))
    );
}

#[test]
fn train_rejects_empty_set() {
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::new();
    let config = LtrConfig::new().with_feature_dim(3);
    assert_eq!(engine.train(&set, &config), Err(LtrError::EmptyTrainingSet));
}

#[test]
fn train_rejects_dimension_mismatch() {
    let engine = LtrEngine::new();
    let set = separable_set(); // 3-dim
    let config = LtrConfig::new().with_feature_dim(5);
    assert_eq!(
        engine.train(&set, &config),
        Err(LtrError::DimensionMismatch {
            expected: 5,
            found: 3
        })
    );
}

#[test]
fn train_rejects_non_finite_feature() {
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![LtrTrainingPair::preferred(
        0,
        LtrFeatureVector::new(vec![f64::NAN]),
        LtrFeatureVector::new(vec![0.1]),
    )]);
    let config = LtrConfig::new().with_feature_dim(1);
    assert_eq!(
        engine.train(&set, &config),
        Err(LtrError::NonFiniteFeature { index: 0 })
    );
}

#[test]
fn train_rejects_bad_label() {
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![LtrTrainingPair::new(
        0,
        LtrFeatureVector::new(vec![0.9]),
        LtrFeatureVector::new(vec![0.1]),
        2.0,
    )]);
    let config = LtrConfig::new().with_feature_dim(1);
    assert_eq!(
        engine.train(&set, &config),
        Err(LtrError::InvalidLabel(2.0))
    );
}

// ── training: convergence & correctness ──────────────────────────────────────

#[test]
fn loss_decreases_monotonically_on_separable_data() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.2)
        .with_epochs(120)
        .with_tolerance(1e-12);
    let model = engine.train(&set, &config).expect("training succeeds");

    assert!(model.loss_history.len() >= 2);
    // Convex objective + small step => the recorded per-epoch losses form a
    // non-increasing sequence (real convergence, not just "did not panic").
    for window in model.loss_history.windows(2) {
        assert!(
            window[1] <= window[0] + 1e-9,
            "loss increased: {} -> {}",
            window[0],
            window[1]
        );
    }
    // And it fell substantially: the model actually learned.
    assert!(model.final_loss < model.loss_history[0] - 0.1);
    assert!(model.final_loss.is_finite());
}

#[test]
fn closed_form_single_gradient_step() {
    // One 1-D pair, zero init, one epoch. RankNet gradient at w=0 is
    // (sigmoid(0) - 1) * d = -0.5 * d, so w1 = -lr * (-0.5 d) = 0.5 * lr * d.
    let engine = LtrEngine::new();
    let d_a = 0.4;
    let set = LtrTrainingSet::from_pairs(vec![LtrTrainingPair::preferred(
        0,
        LtrFeatureVector::new(vec![d_a]),
        LtrFeatureVector::new(vec![0.0]),
    )]);
    let config = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(1.0)
        .with_epochs(1)
        .with_tolerance(0.0);
    let model = engine.train(&set, &config).expect("training succeeds");

    let expected = 0.5 * 1.0 * d_a;
    assert!(
        close(model.weights[0], expected, 1e-12),
        "weight {} != {}",
        model.weights[0],
        expected
    );
    // The recorded (pre-update) loss is exactly ln 2 = softplus(0).
    assert!(close(model.loss_history[0], 2.0_f64.ln(), 1e-12));
}

#[test]
fn closed_form_symmetric_optimum_is_zero() {
    // Two 1-D pairs with the same feature difference but opposite labels:
    // the unique optimum is w = 0 (and gradient is exactly 0 there).
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![
        LtrTrainingPair::new(
            0,
            LtrFeatureVector::new(vec![1.0]),
            LtrFeatureVector::new(vec![0.0]),
            1.0,
        ),
        LtrTrainingPair::new(
            0,
            LtrFeatureVector::new(vec![1.0]),
            LtrFeatureVector::new(vec![0.0]),
            0.0,
        ),
    ]);
    let config = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(0.5)
        .with_epochs(100)
        .with_tolerance(0.0);
    let model = engine.train(&set, &config).expect("training succeeds");
    assert!(close(model.weights[0], 0.0, 1e-12));
    // Minimum loss for this symmetric set is softplus(0) = ln 2.
    assert!(close(model.final_loss, 2.0_f64.ln(), 1e-12));
}

#[test]
fn single_feature_separable_learns_positive_weight() {
    // Every pair prefers the larger feature value => the learned weight must
    // be positive and score(higher) > score(lower).
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![
        pair_1d(0, 0.9, 0.1),
        pair_1d(1, 0.8, 0.2),
        pair_1d(2, 0.7, 0.3),
    ]);
    let config = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(0.5)
        .with_epochs(400);
    let model = engine.train(&set, &config).expect("training succeeds");

    assert!(model.weights[0] > 0.0);
    let hi = model.score(&LtrFeatureVector::new(vec![0.95]));
    let lo = model.score(&LtrFeatureVector::new(vec![0.05]));
    assert!(hi > lo);
}

#[test]
fn model_learns_signed_weights_and_ranks_held_out_pair() {
    // Dim 0 correlates with preference (higher is better); dim 1 anti-correlates
    // (higher is worse). The model must learn w0 > 0 and w1 < 0.
    let engine = LtrEngine::new();
    let rows = [
        (0.9, 0.1, 0.1, 0.9),
        (0.8, 0.2, 0.2, 0.8),
        (0.7, 0.3, 0.3, 0.7),
        (0.95, 0.05, 0.05, 0.95),
        (0.6, 0.4, 0.4, 0.6),
    ];
    let mut set = LtrTrainingSet::new();
    for (i, &(a0, a1, b0, b1)) in rows.iter().enumerate() {
        set.push(LtrTrainingPair::preferred(
            i as u64,
            LtrFeatureVector::new(vec![a0, a1]),
            LtrFeatureVector::new(vec![b0, b1]),
        ));
    }
    let config = LtrConfig::new()
        .with_feature_dim(2)
        .with_learning_rate(0.3)
        .with_epochs(800);
    let model = engine.train(&set, &config).expect("training succeeds");

    assert!(model.weight(0).expect("w0") > 0.0);
    assert!(model.weight(1).expect("w1") < 0.0);

    // Held-out preference consistent with training: A should outrank B.
    let a = LtrFeatureVector::new(vec![0.9, 0.1]);
    let b = LtrFeatureVector::new(vec![0.1, 0.9]);
    assert!(model.score(&a) > model.score(&b));
    let ranked = engine.rank(&model, &[b.clone(), a.clone()]);
    assert_eq!(ranked[0].0, 1); // index of `a`
}

#[test]
fn training_is_deterministic() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.25)
        .with_epochs(60);
    let m1 = engine.train(&set, &config).expect("train 1");
    let m2 = engine.train(&set, &config).expect("train 2");
    assert_eq!(m1.weights, m2.weights);
    assert_eq!(m1.loss_history, m2.loss_history);
    assert_eq!(m1.final_loss, m2.final_loss);
    assert_eq!(m1.epochs_run, m2.epochs_run);
}

#[test]
fn seeded_init_is_deterministic() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.2)
        .with_epochs(30)
        .with_weight_init_scale(0.5)
        .with_seed(1234);
    let m1 = engine.train(&set, &config).expect("train 1");
    let m2 = engine.train(&set, &config).expect("train 2");
    assert_eq!(m1.weights, m2.weights);
}

#[test]
fn seed_is_irrelevant_with_zero_init_scale() {
    // With the default zero init scale, the seed does not affect the result.
    let engine = LtrEngine::new();
    let set = separable_set();
    let base = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.2)
        .with_epochs(30);
    let m1 = engine.train(&set, &base.clone().with_seed(1)).expect("m1");
    let m2 = engine.train(&set, &base.with_seed(999_999)).expect("m2");
    assert_eq!(m1.weights, m2.weights);
}

#[test]
fn early_stop_triggers_on_large_tolerance() {
    // A tolerance larger than any epoch-to-epoch loss change stops the loop on
    // the second epoch.
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.2)
        .with_epochs(100)
        .with_tolerance(10.0);
    let model = engine.train(&set, &config).expect("training succeeds");
    assert!(model.converged);
    assert!(model.epochs_run < 100);
    assert_eq!(model.epochs_run, 2);
}

#[test]
fn early_stop_triggers_on_plateau() {
    // The symmetric set sits at its optimum from the first step, so the loss
    // change hits zero and early stopping fires well before the epoch budget.
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![
        LtrTrainingPair::new(
            0,
            LtrFeatureVector::new(vec![1.0]),
            LtrFeatureVector::new(vec![0.0]),
            1.0,
        ),
        LtrTrainingPair::new(
            0,
            LtrFeatureVector::new(vec![1.0]),
            LtrFeatureVector::new(vec![0.0]),
            0.0,
        ),
    ]);
    let config = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(0.5)
        .with_epochs(500)
        .with_tolerance(1e-9);
    let model = engine.train(&set, &config).expect("training succeeds");
    assert!(model.converged);
    assert!(model.epochs_run < 500);
}

#[test]
fn training_stays_finite_on_extreme_features() {
    // Extreme-magnitude features must not produce NaN/Inf thanks to the stable
    // sigmoid/softplus.
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![
        LtrTrainingPair::preferred(
            0,
            LtrFeatureVector::new(vec![1e6]),
            LtrFeatureVector::new(vec![-1e6]),
        ),
        LtrTrainingPair::preferred(
            1,
            LtrFeatureVector::new(vec![1e5]),
            LtrFeatureVector::new(vec![-1e5]),
        ),
    ]);
    let config = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(1e-6)
        .with_epochs(20);
    let model = engine.train(&set, &config).expect("training succeeds");
    assert!(model.final_loss.is_finite());
    assert!(model.weights.iter().all(|w| w.is_finite()));
    assert!(model.loss_history.iter().all(|l| l.is_finite()));
}

#[test]
fn l2_regularization_shrinks_weights() {
    let engine = LtrEngine::new();
    let set = LtrTrainingSet::from_pairs(vec![
        pair_1d(0, 0.9, 0.1),
        pair_1d(1, 0.8, 0.2),
        pair_1d(2, 0.7, 0.3),
    ]);
    let base = LtrConfig::new()
        .with_feature_dim(1)
        .with_learning_rate(0.5)
        .with_epochs(300)
        .with_tolerance(0.0);
    let unregularized = engine.train(&set, &base.clone()).expect("plain");
    let regularized = engine
        .train(&set, &base.with_l2_regularization(0.5))
        .expect("l2");
    assert!(regularized.weights[0].abs() < unregularized.weights[0].abs());
    assert!(regularized.weights[0] > 0.0);
}

#[test]
fn epochs_run_matches_budget_without_early_stop() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new()
        .with_feature_dim(3)
        .with_learning_rate(0.2)
        .with_epochs(15)
        .with_tolerance(0.0); // never early-stops
    let model = engine.train(&set, &config).expect("training succeeds");
    assert_eq!(model.epochs_run, 15);
    assert!(!model.converged);
    assert_eq!(model.loss_history.len(), 15);
}

// ── LtrModel scoring ─────────────────────────────────────────────────────────

#[test]
fn model_score_is_dot_product() {
    let model = LtrModel {
        weights: vec![1.0, 2.0, 3.0],
        feature_dim: 3,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let v = LtrFeatureVector::new(vec![1.0, 1.0, 1.0]);
    assert_eq!(model.score(&v), 6.0);
    let w = LtrFeatureVector::new(vec![0.0, 0.5, 1.0]);
    assert_eq!(model.score(&w), 4.0);
}

#[test]
fn model_try_score_validates() {
    let model = LtrModel {
        weights: vec![1.0, 2.0],
        feature_dim: 2,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    assert_eq!(
        model.try_score(&LtrFeatureVector::new(vec![1.0, 1.0])),
        Ok(3.0)
    );
    assert_eq!(
        model.try_score(&LtrFeatureVector::new(vec![1.0])),
        Err(LtrError::DimensionMismatch {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn model_weight_accessor() {
    let model = LtrModel {
        weights: vec![0.5, -0.5],
        feature_dim: 2,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    assert_eq!(model.weight(0), Some(0.5));
    assert_eq!(model.weight(1), Some(-0.5));
    assert_eq!(model.weight(2), None);
}

// ── ranking ──────────────────────────────────────────────────────────────────

#[test]
fn rank_orders_by_descending_score() {
    let model = LtrModel {
        weights: vec![1.0, 0.0],
        feature_dim: 2,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let engine = LtrEngine::new();
    let candidates = vec![
        LtrFeatureVector::new(vec![0.2, 9.0]),
        LtrFeatureVector::new(vec![0.9, 0.0]),
        LtrFeatureVector::new(vec![0.5, 3.0]),
    ];
    let ranked = engine.rank(&model, &candidates);
    assert_eq!(
        ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![1, 2, 0]
    );
    // Scores are sorted descending.
    for w in ranked.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

#[test]
fn rank_matches_score_ordering() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new().with_feature_dim(3).with_epochs(100);
    let model = engine.train(&set, &config).expect("training succeeds");
    let candidates = vec![
        LtrFeatureVector::new(vec![0.1, 0.1, 0.5]),
        LtrFeatureVector::new(vec![0.9, 0.9, 0.5]),
        LtrFeatureVector::new(vec![0.5, 0.5, 0.5]),
    ];
    let ranked = engine.rank(&model, &candidates);
    let mut manual: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, model.score(c)))
        .collect();
    manual.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(
        ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        manual.iter().map(|(i, _)| *i).collect::<Vec<_>>()
    );
}

#[test]
fn rank_breaks_ties_by_ascending_index() {
    let model = LtrModel {
        weights: vec![1.0],
        feature_dim: 1,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let engine = LtrEngine::new();
    // All identical scores -> stable ascending-index order.
    let candidates = vec![
        LtrFeatureVector::new(vec![0.5]),
        LtrFeatureVector::new(vec![0.5]),
        LtrFeatureVector::new(vec![0.5]),
    ];
    let ranked = engine.rank(&model, &candidates);
    assert_eq!(
        ranked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn rank_empty_candidates() {
    let model = LtrModel {
        weights: vec![1.0],
        feature_dim: 1,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let engine = LtrEngine::new();
    assert!(engine.rank(&model, &[]).is_empty());
}

#[test]
fn rank_checked_rejects_dim_mismatch() {
    let model = LtrModel {
        weights: vec![1.0, 1.0],
        feature_dim: 2,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let engine = LtrEngine::new();
    let candidates = vec![
        LtrFeatureVector::new(vec![0.5, 0.5]),
        LtrFeatureVector::new(vec![0.5]),
    ];
    assert_eq!(
        engine.rank_checked(&model, &candidates),
        Err(LtrError::DimensionMismatch {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn rank_checked_ok() {
    let model = LtrModel {
        weights: vec![1.0, 1.0],
        feature_dim: 2,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    let engine = LtrEngine::new();
    let candidates = vec![
        LtrFeatureVector::new(vec![0.1, 0.1]),
        LtrFeatureVector::new(vec![0.9, 0.9]),
    ];
    let ranked = engine.rank_checked(&model, &candidates).expect("ok");
    assert_eq!(ranked[0].0, 1);
}

#[test]
fn rank_indices_matches_rank() {
    let engine = LtrEngine::new();
    let set = separable_set();
    let config = LtrConfig::new().with_feature_dim(3).with_epochs(100);
    let model = engine.train(&set, &config).expect("training succeeds");
    let candidates = vec![
        LtrFeatureVector::new(vec![0.2, 0.2, 0.5]),
        LtrFeatureVector::new(vec![0.8, 0.8, 0.5]),
    ];
    let idx = engine.rank_indices(&model, &candidates);
    let full = engine.rank(&model, &candidates);
    assert_eq!(idx, full.iter().map(|(i, _)| *i).collect::<Vec<_>>());
}

#[test]
fn engine_default_equals_new() {
    let a = LtrEngine::default();
    let b = LtrEngine::new();
    // Both are unit structs; construct and use them to confirm they behave the
    // same on an empty candidate list.
    let model = LtrModel {
        weights: vec![1.0],
        feature_dim: 1,
        epochs_run: 0,
        final_loss: 0.0,
        converged: false,
        loss_history: vec![],
    };
    assert_eq!(a.rank(&model, &[]), b.rank(&model, &[]));
}

// ── feature extraction ───────────────────────────────────────────────────────

#[test]
fn extractor_output_dimension_and_range() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let doc = LtrDocument::new("d", "rust systems programming language")
        .with_timestamp(999_000_000)
        .with_popularity(20);
    let features = extractor.extract("rust programming", &doc);
    assert_eq!(features.dim(), LTR_DEFAULT_FEATURE_DIM);
    for &v in features.as_slice() {
        assert!((0.0..=1.0).contains(&v), "feature {v} out of [0,1]");
    }
}

#[test]
fn extractor_is_deterministic() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let doc = LtrDocument::new("d", "the quick brown fox jumps")
        .with_timestamp(999_500_000)
        .with_popularity(5);
    let f1 = extractor.extract("quick fox", &doc);
    let f2 = extractor.extract("quick fox", &doc);
    assert_eq!(f1, f2);
}

#[test]
fn extractor_recency_newer_scores_higher() {
    let reference = 1_000_000_000;
    let extractor = LtrFeatureExtractor::new(reference).with_recency_scale_days(30.0);
    let newer = LtrDocument::new("new", "hello world").with_timestamp(reference);
    let older = LtrDocument::new("old", "hello world").with_timestamp(reference - 30 * 86_400);
    let f_new = extractor.extract("hello", &newer);
    let f_old = extractor.extract("hello", &older);
    // Newest possible => recency 1.0; exactly one half-scale old => 0.5.
    assert!(close(f_new.get(LTR_FEATURE_RECENCY).unwrap(), 1.0, 1e-12));
    assert!(close(f_old.get(LTR_FEATURE_RECENCY).unwrap(), 0.5, 1e-9));
    assert!(f_new.get(LTR_FEATURE_RECENCY) > f_old.get(LTR_FEATURE_RECENCY));
}

#[test]
fn extractor_popularity_monotonic() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000).with_popularity_scale(10.0);
    let low = LtrDocument::new("a", "hello world").with_popularity(0);
    let mid = LtrDocument::new("b", "hello world").with_popularity(10);
    let high = LtrDocument::new("c", "hello world").with_popularity(90);
    let f_low = extractor
        .extract("hello", &low)
        .get(LTR_FEATURE_POPULARITY)
        .unwrap();
    let f_mid = extractor
        .extract("hello", &mid)
        .get(LTR_FEATURE_POPULARITY)
        .unwrap();
    let f_high = extractor
        .extract("hello", &high)
        .get(LTR_FEATURE_POPULARITY)
        .unwrap();
    assert!(close(f_low, 0.0, 1e-12));
    assert!(close(f_mid, 0.5, 1e-12));
    assert!(f_low < f_mid && f_mid < f_high);
    assert!(close(f_high, 0.9, 1e-12));
}

#[test]
fn extractor_length_monotonic() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let short = LtrDocument::new("a", "one two");
    let long = LtrDocument::new("b", "one two three four five six seven eight nine ten");
    let f_short = extractor
        .extract("one", &short)
        .get(LTR_FEATURE_LENGTH)
        .unwrap();
    let f_long = extractor
        .extract("one", &long)
        .get(LTR_FEATURE_LENGTH)
        .unwrap();
    assert!(f_long > f_short);
}

#[test]
fn extractor_bm25_present_beats_absent() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let present = LtrDocument::new("a", "quantum physics research paper");
    let absent = LtrDocument::new("b", "cooking recipes and food");
    let f_present = extractor
        .extract("quantum", &present)
        .get(LTR_FEATURE_BM25)
        .unwrap();
    let f_absent = extractor
        .extract("quantum", &absent)
        .get(LTR_FEATURE_BM25)
        .unwrap();
    assert!(f_present > 0.0);
    assert!(close(f_absent, 0.0, 1e-12));
    assert!(f_present > f_absent);
}

#[test]
fn extractor_bm25_more_matches_scores_higher() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let twice = LtrDocument::new("a", "rust rust");
    let once = LtrDocument::new("b", "rust");
    let f_twice = extractor
        .extract("rust", &twice)
        .get(LTR_FEATURE_BM25)
        .unwrap();
    let f_once = extractor
        .extract("rust", &once)
        .get(LTR_FEATURE_BM25)
        .unwrap();
    assert!(f_twice > f_once);
}

#[test]
fn extractor_embedding_similarity_related_beats_unrelated() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let query = "machine learning models";
    let same = LtrDocument::new("a", "machine learning models");
    let unrelated = LtrDocument::new("b", "banana orange cherry grape");
    let f_same = extractor
        .extract(query, &same)
        .get(LTR_FEATURE_EMBEDDING_SIM)
        .unwrap();
    let f_unrelated = extractor
        .extract(query, &unrelated)
        .get(LTR_FEATURE_EMBEDDING_SIM)
        .unwrap();
    assert!(f_same > 0.9); // identical token multiset => cosine ~ 1.0
    assert!(f_same > f_unrelated);
    assert!(f_unrelated < 0.5);
}

#[test]
fn extractor_from_corpus_sets_idf_and_avg_len() {
    let corpus = vec![
        LtrDocument::new("a", "quantum physics research"),
        LtrDocument::new("b", "classical physics notes"),
        LtrDocument::new("c", "cooking recipes food"),
    ];
    let extractor = LtrFeatureExtractor::from_corpus(&corpus, 1_000_000_000);
    // A rare term ("quantum", df=1) should still yield a positive BM25 signal.
    let f = extractor
        .extract("quantum", &corpus[0])
        .get(LTR_FEATURE_BM25)
        .unwrap();
    assert!(f > 0.0);
    // A document without the term scores zero.
    let f_absent = extractor
        .extract("quantum", &corpus[2])
        .get(LTR_FEATURE_BM25)
        .unwrap();
    assert!(close(f_absent, 0.0, 1e-12));
}

#[test]
fn extractor_empty_query_gives_zero_relevance() {
    let extractor = LtrFeatureExtractor::new(1_000_000_000);
    let doc = LtrDocument::new("a", "some content here");
    let features = extractor.extract("", &doc);
    assert!(close(features.get(LTR_FEATURE_BM25).unwrap(), 0.0, 1e-12));
    assert!(close(
        features.get(LTR_FEATURE_EMBEDDING_SIM).unwrap(),
        0.0,
        1e-12
    ));
}

#[test]
fn extractor_end_to_end_training() {
    // Build a training set entirely from extracted features and confirm the
    // pipeline trains to a finite, converging model.
    let reference = 1_000_000_000;
    let extractor = LtrFeatureExtractor::new(reference);
    let query = "rust programming";
    let relevant = LtrDocument::new("rel", "rust programming systems language")
        .with_timestamp(reference)
        .with_popularity(100);
    let irrelevant = LtrDocument::new("irr", "gardening tips for spring")
        .with_timestamp(reference - 200 * 86_400)
        .with_popularity(1);
    let f_rel = extractor.extract(query, &relevant);
    let f_irr = extractor.extract(query, &irrelevant);

    let set = LtrTrainingSet::from_pairs(vec![
        LtrTrainingPair::preferred(0, f_rel.clone(), f_irr.clone()),
        LtrTrainingPair::preferred(1, f_rel.clone(), f_irr.clone()),
    ]);
    let config = LtrConfig::new()
        .with_feature_dim(LTR_DEFAULT_FEATURE_DIM)
        .with_learning_rate(0.3)
        .with_epochs(200);
    let engine = LtrEngine::new();
    let model = engine.train(&set, &config).expect("training succeeds");
    assert!(model.final_loss.is_finite());
    assert!(model.final_loss < model.loss_history[0]);
    // The relevant document now outranks the irrelevant one.
    assert!(model.score(&f_rel) > model.score(&f_irr));
}

// ── error type ───────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(
        LtrError::EmptyTrainingSet.to_string(),
        "training set must contain at least one preference pair"
    );
    assert_eq!(
        LtrError::ZeroEpochs.to_string(),
        "epoch count must be greater than zero"
    );
    assert!(
        LtrError::InvalidLearningRate(-1.0)
            .to_string()
            .contains("learning rate")
    );
    assert!(
        LtrError::DimensionMismatch {
            expected: 5,
            found: 3
        }
        .to_string()
        .contains("expected 5")
    );
}

#[test]
fn error_is_clone_and_eq() {
    let e = LtrError::NonFiniteFeature { index: 2 };
    assert_eq!(e.clone(), e);
    assert_ne!(e, LtrError::NonFiniteFeature { index: 3 });
}
