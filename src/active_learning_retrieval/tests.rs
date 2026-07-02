use crate::active_learning_retrieval::engine::ActiveLearningSelector;
use crate::active_learning_retrieval::types::{
    ActiveLearningConfig, ActiveLearningError, PoolItem, SelectionBatch, UncertaintyMeasure,
    UncertaintySample,
};

/// Float tolerance for hand-computed `f32` assertions.
const EPS: f32 = 1e-4;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

fn default_selector() -> ActiveLearningSelector {
    ActiveLearningSelector::new(ActiveLearningConfig::default()).expect("default config is valid")
}

// ── PoolItem ──────────────────────────────────────────────────────────────

#[test]
fn pool_item_new_defaults_to_empty_features() {
    let item = PoolItem::new("doc-1", vec![0.9, 0.1]);
    assert_eq!(item.id, "doc-1");
    assert_eq!(item.scores, vec![0.9, 0.1]);
    assert!(item.features.is_empty());
}

#[test]
fn pool_item_with_features_builder() {
    let item = PoolItem::new("doc-1", vec![0.9, 0.1]).with_features(vec![1.0, 2.0, 3.0]);
    assert_eq!(item.features, vec![1.0, 2.0, 3.0]);
}

// ── UncertaintyMeasure ────────────────────────────────────────────────────

#[test]
fn uncertainty_measure_as_str_is_snake_case() {
    assert_eq!(
        UncertaintyMeasure::MarginSampling.as_str(),
        "margin_sampling"
    );
    assert_eq!(
        UncertaintyMeasure::EntropySampling.as_str(),
        "entropy_sampling"
    );
}

#[test]
fn uncertainty_measure_display_matches_as_str() {
    assert_eq!(
        UncertaintyMeasure::MarginSampling.to_string(),
        UncertaintyMeasure::MarginSampling.as_str()
    );
    assert_eq!(
        UncertaintyMeasure::EntropySampling.to_string(),
        UncertaintyMeasure::EntropySampling.as_str()
    );
}

#[test]
fn uncertainty_measure_default_is_margin_sampling() {
    assert_eq!(
        UncertaintyMeasure::default(),
        UncertaintyMeasure::MarginSampling
    );
}

// ── ActiveLearningConfig ──────────────────────────────────────────────────

#[test]
fn config_defaults() {
    let config = ActiveLearningConfig::default();
    assert_eq!(config.measure, UncertaintyMeasure::MarginSampling);
    assert_eq!(config.batch_size, 10);
    assert!(!config.diversity_enabled);
    assert!(approx(config.diversity_threshold, 0.9));
}

#[test]
fn config_new_equals_default() {
    assert_eq!(ActiveLearningConfig::new(), ActiveLearningConfig::default());
}

#[test]
fn config_builders_chain() {
    let config = ActiveLearningConfig::new()
        .with_measure(UncertaintyMeasure::EntropySampling)
        .with_batch_size(4)
        .with_diversity_enabled(true)
        .with_diversity_threshold(0.5);
    assert_eq!(config.measure, UncertaintyMeasure::EntropySampling);
    assert_eq!(config.batch_size, 4);
    assert!(config.diversity_enabled);
    assert!(approx(config.diversity_threshold, 0.5));
}

#[test]
fn config_validate_accepts_default() {
    assert!(ActiveLearningConfig::default().validate().is_ok());
}

#[test]
fn config_validate_accepts_boundary_thresholds() {
    assert!(
        ActiveLearningConfig::new()
            .with_diversity_threshold(1.0)
            .validate()
            .is_ok()
    );
    assert!(
        ActiveLearningConfig::new()
            .with_diversity_threshold(-1.0)
            .validate()
            .is_ok()
    );
}

#[test]
fn config_validate_rejects_out_of_range_threshold() {
    let err = ActiveLearningConfig::new()
        .with_diversity_threshold(1.5)
        .validate()
        .unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));

    let err = ActiveLearningConfig::new()
        .with_diversity_threshold(-1.5)
        .validate()
        .unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));
}

#[test]
fn config_validate_rejects_non_finite_threshold() {
    let err = ActiveLearningConfig::new()
        .with_diversity_threshold(f32::NAN)
        .validate()
        .unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));

    let err = ActiveLearningConfig::new()
        .with_diversity_threshold(f32::INFINITY)
        .validate()
        .unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));
}

// ── ActiveLearningSelector::new ───────────────────────────────────────────

#[test]
fn selector_new_accepts_valid_config() {
    assert!(ActiveLearningSelector::new(ActiveLearningConfig::default()).is_ok());
}

#[test]
fn selector_new_propagates_invalid_config() {
    let config = ActiveLearningConfig::new().with_diversity_threshold(2.0);
    let err = ActiveLearningSelector::new(config).unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));
}

#[test]
fn selector_default_is_valid() {
    let selector = ActiveLearningSelector::default();
    assert_eq!(selector.config, ActiveLearningConfig::default());
}

// ── compute_uncertainty: validation errors ────────────────────────────────

#[test]
fn compute_uncertainty_rejects_empty_scores() {
    let err =
        ActiveLearningSelector::compute_uncertainty("x", &[], UncertaintyMeasure::MarginSampling)
            .unwrap_err();
    match err {
        ActiveLearningError::EmptyItemScores { item_id } => assert_eq!(item_id, "x"),
        other => panic!("expected EmptyItemScores, got {other:?}"),
    }
}

#[test]
fn compute_uncertainty_rejects_nan_score() {
    let err = ActiveLearningSelector::compute_uncertainty(
        "y",
        &[1.0, f32::NAN],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap_err();
    match err {
        ActiveLearningError::NonFiniteScore { item_id, index } => {
            assert_eq!(item_id, "y");
            assert_eq!(index, 1);
        }
        other => panic!("expected NonFiniteScore, got {other:?}"),
    }
}

#[test]
fn compute_uncertainty_rejects_infinite_score() {
    let err = ActiveLearningSelector::compute_uncertainty(
        "z",
        &[f32::INFINITY, 1.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap_err();
    match err {
        ActiveLearningError::NonFiniteScore { item_id, index } => {
            assert_eq!(item_id, "z");
            assert_eq!(index, 0);
        }
        other => panic!("expected NonFiniteScore, got {other:?}"),
    }
}

// ── margin sampling: hand-computed ground truth ───────────────────────────
//
// Normalization: clamp negatives to 0, divide by sum (uniform fallback when
// the sum is ~0). Margin uncertainty = 1 - (p_top1 - p_top2), p_top2 = 0
// when there is no second candidate.

#[test]
fn margin_sampling_tied_scores_is_maximally_uncertain() {
    // probs = [0.5, 0.5] => margin = 0 => uncertainty = 1.0.
    let score = ActiveLearningSelector::compute_uncertainty(
        "tied",
        &[5.0, 5.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(approx(score, 1.0), "expected 1.0, got {score}");
}

#[test]
fn margin_sampling_dominant_top1_is_low_uncertainty() {
    // probs = [0.9, 0.1] => margin = 0.8 => uncertainty = 0.2.
    let score = ActiveLearningSelector::compute_uncertainty(
        "confident",
        &[9.0, 1.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(approx(score, 0.2), "expected 0.2, got {score}");
}

#[test]
fn margin_sampling_close_call_hand_computed() {
    // probs = [6/11, 5/11] => margin = 1/11 => uncertainty = 10/11.
    let score = ActiveLearningSelector::compute_uncertainty(
        "close",
        &[6.0, 5.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(
        approx(score, 10.0 / 11.0),
        "expected {}, got {score}",
        10.0 / 11.0
    );
}

#[test]
fn margin_sampling_small_margin_ranks_above_large_margin() {
    // The core margin-sampling correctness claim: an ambiguous (small-margin)
    // item must score strictly higher than a confident (large-margin) item.
    let ambiguous = ActiveLearningSelector::compute_uncertainty(
        "ambiguous",
        &[5.0, 4.9],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    let confident = ActiveLearningSelector::compute_uncertainty(
        "confident",
        &[9.0, 1.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(
        ambiguous > confident,
        "ambiguous ({ambiguous}) must rank above confident ({confident})"
    );
}

#[test]
fn margin_sampling_single_score_is_fully_confident() {
    // Only one candidate: nothing to be ambiguous against => uncertainty 0.
    let score = ActiveLearningSelector::compute_uncertainty(
        "solo",
        &[7.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(approx(score, 0.0), "expected 0.0, got {score}");
}

#[test]
fn margin_sampling_all_zero_scores_falls_back_to_uniform_tie() {
    // Sum clamps to 0 => uniform fallback [0.5, 0.5] => uncertainty 1.0.
    let score = ActiveLearningSelector::compute_uncertainty(
        "blank",
        &[0.0, 0.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!(approx(score, 1.0), "expected 1.0, got {score}");
}

#[test]
fn margin_sampling_negative_scores_are_clamped_not_erroring() {
    // Negative "relevance" clamps to 0 before normalizing; this must not be
    // treated as non-finite (which would error) and must produce a bounded
    // result.
    let score = ActiveLearningSelector::compute_uncertainty(
        "neg",
        &[-3.0, 9.0],
        UncertaintyMeasure::MarginSampling,
    )
    .unwrap();
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn margin_sampling_is_bounded_in_zero_one_across_examples() {
    let examples: &[&[f32]] = &[
        &[5.0, 5.0],
        &[9.0, 1.0],
        &[7.0],
        &[0.0, 0.0],
        &[100.0, 99.9, 0.1],
    ];
    for scores in examples {
        let score = ActiveLearningSelector::compute_uncertainty(
            "x",
            scores,
            UncertaintyMeasure::MarginSampling,
        )
        .unwrap();
        assert!(
            (0.0..=1.0).contains(&score),
            "margin score {score} out of bounds for {scores:?}"
        );
    }
}

// ── entropy sampling: hand-computed ground truth ──────────────────────────

#[test]
fn entropy_sampling_uniform_three_way_equals_ln_3() {
    let score = ActiveLearningSelector::compute_uncertainty(
        "uniform3",
        &[1.0, 1.0, 1.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    assert!(
        approx(score, 3.0_f32.ln()),
        "expected ln(3)={}, got {score}",
        3.0_f32.ln()
    );
}

#[test]
fn entropy_sampling_all_zero_three_way_falls_back_to_uniform_ln_3() {
    // Sum clamps to 0 => uniform fallback over 3 entries => ln(3), same as
    // the explicit uniform case above.
    let score = ActiveLearningSelector::compute_uncertainty(
        "blank3",
        &[0.0, 0.0, 0.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    assert!(approx(score, 3.0_f32.ln()));
}

#[test]
fn entropy_sampling_fully_peaked_is_zero() {
    // probs = [1.0, 0.0, 0.0] => H = -(1*ln 1) = 0.
    let score = ActiveLearningSelector::compute_uncertainty(
        "peaked",
        &[10.0, 0.0, 0.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    assert!(approx(score, 0.0), "expected 0.0, got {score}");
}

#[test]
fn entropy_sampling_partially_peaked_hand_computed() {
    // probs = [0.8, 0.2, 0.0] => H = -(0.8 ln 0.8 + 0.2 ln 0.2) ≈ 0.5004024.
    let score = ActiveLearningSelector::compute_uncertainty(
        "mid",
        &[8.0, 2.0, 0.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    let expected = -(0.8_f32 * 0.8_f32.ln() + 0.2_f32 * 0.2_f32.ln());
    assert!(approx(score, expected), "expected {expected}, got {score}");
    assert!(approx(score, 0.500_402_4));
}

#[test]
fn entropy_sampling_single_score_is_zero() {
    // Only one candidate: probs = [1.0] => H = 0.
    let score = ActiveLearningSelector::compute_uncertainty(
        "solo",
        &[7.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    assert!(approx(score, 0.0));
}

#[test]
fn entropy_sampling_uniform_ranks_above_peaked() {
    let uniform = ActiveLearningSelector::compute_uncertainty(
        "uniform",
        &[1.0, 1.0, 1.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    let peaked = ActiveLearningSelector::compute_uncertainty(
        "peaked",
        &[10.0, 0.0, 0.0],
        UncertaintyMeasure::EntropySampling,
    )
    .unwrap();
    assert!(
        uniform > peaked,
        "uniform ({uniform}) must rank above peaked ({peaked})"
    );
}

#[test]
fn entropy_sampling_is_never_negative() {
    let examples: &[&[f32]] = &[
        &[5.0, 5.0],
        &[9.0, 1.0],
        &[7.0],
        &[0.0, 0.0],
        &[100.0, 99.9, 0.1],
    ];
    for scores in examples {
        let score = ActiveLearningSelector::compute_uncertainty(
            "x",
            scores,
            UncertaintyMeasure::EntropySampling,
        )
        .unwrap();
        assert!(score >= 0.0, "entropy {score} negative for {scores:?}");
    }
}

// ── margin vs. entropy: genuinely different computations ─────────────────

#[test]
fn margin_and_entropy_disagree_on_a_crafted_example() {
    // X: a clean two-way tie plus one dead candidate. Y: a mild top-1 lead
    // spread over three live candidates.
    //
    // Margin only looks at the top two entries, so X's exact tie (margin=0)
    // makes it *more* uncertain than Y's 0.1 gap:
    //   margin(X) = 1.0, margin(Y) = 0.9  =>  X > Y under margin.
    //
    // Entropy looks at the whole distribution, so Y's mass being spread
    // across three live candidates (instead of concentrated on two, with a
    // dead third) makes it *more* uncertain than X:
    //   entropy(X) = ln(2) ≈ 0.6931, entropy(Y) ≈ 1.0889  =>  Y > X under entropy.
    //
    // The ranking flips between measures: they are genuinely different
    // computations, not aliases of one another.
    let x = [0.5_f32, 0.5, 0.0];
    let y = [0.4_f32, 0.3, 0.3];

    let margin_x =
        ActiveLearningSelector::compute_uncertainty("x", &x, UncertaintyMeasure::MarginSampling)
            .unwrap();
    let margin_y =
        ActiveLearningSelector::compute_uncertainty("y", &y, UncertaintyMeasure::MarginSampling)
            .unwrap();
    assert!(approx(margin_x, 1.0));
    assert!(approx(margin_y, 0.9));
    assert!(margin_x > margin_y, "expected margin to rank X above Y");

    let entropy_x =
        ActiveLearningSelector::compute_uncertainty("x", &x, UncertaintyMeasure::EntropySampling)
            .unwrap();
    let entropy_y =
        ActiveLearningSelector::compute_uncertainty("y", &y, UncertaintyMeasure::EntropySampling)
            .unwrap();
    assert!(approx(entropy_x, 2.0_f32.ln()));
    assert!(approx(entropy_y, 1.088_9));
    assert!(entropy_y > entropy_x, "expected entropy to rank Y above X");

    // The headline assertion: the two measures produce opposite orderings.
    assert!(margin_x > margin_y && entropy_y > entropy_x);
}

// ── select_batch: ranking correctness ─────────────────────────────────────

fn ranking_pool() -> Vec<PoolItem> {
    vec![
        PoolItem::new("ambiguous", vec![5.0, 4.9]),
        PoolItem::new("confident", vec![9.0, 1.0]),
        PoolItem::new("midpack", vec![6.0, 4.0]),
    ]
}

#[test]
fn select_batch_margin_ranks_ambiguous_above_confident() {
    let selector = default_selector();
    let batch = selector
        .select_batch(&ranking_pool(), 3, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.ranked[0].item_id, "ambiguous");
    assert_eq!(batch.ranked[2].item_id, "confident");
}

#[test]
fn select_batch_entropy_ranks_uniform_above_peaked() {
    let pool = vec![
        PoolItem::new("uniform", vec![1.0, 1.0, 1.0]),
        PoolItem::new("peaked", vec![10.0, 0.0, 0.0]),
        PoolItem::new("mid", vec![8.0, 2.0, 0.0]),
    ];
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 3, UncertaintyMeasure::EntropySampling)
        .unwrap();
    assert_eq!(batch.ranked[0].item_id, "uniform");
    assert_eq!(batch.ranked[1].item_id, "mid");
    assert_eq!(batch.ranked[2].item_id, "peaked");
}

#[test]
fn select_batch_result_is_fully_descending() {
    let pool = vec![
        PoolItem::new("a", vec![1.0, 9.0]),
        PoolItem::new("b", vec![5.0, 5.0]),
        PoolItem::new("c", vec![3.0, 1.0]),
        PoolItem::new("d", vec![2.0, 2.0, 2.0]),
    ];
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 4, UncertaintyMeasure::MarginSampling)
        .unwrap();
    for window in batch.ranked.windows(2) {
        assert!(
            window[0].uncertainty_score >= window[1].uncertainty_score,
            "ranked list not descending: {:?}",
            batch.ranked
        );
    }
}

#[test]
fn select_batch_measure_argument_overrides_config_measure() {
    // Selector's own config defaults to MarginSampling, but the explicit
    // `measure` argument to `select_batch` must win.
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new().with_measure(UncertaintyMeasure::MarginSampling),
    )
    .unwrap();
    let pool = vec![
        PoolItem::new("uniform", vec![1.0, 1.0, 1.0]),
        PoolItem::new("peaked", vec![10.0, 0.0, 0.0]),
    ];
    let batch = selector
        .select_batch(&pool, 2, UncertaintyMeasure::EntropySampling)
        .unwrap();
    assert_eq!(batch.measure, UncertaintyMeasure::EntropySampling);
    assert_eq!(batch.ranked[0].item_id, "uniform");
}

// ── select_batch: batch-size bounds ────────────────────────────────────────

#[test]
fn select_batch_returns_exactly_batch_size_when_pool_is_larger() {
    let pool = ranking_pool();
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.selected.len(), 2);
}

#[test]
fn select_batch_returns_fewer_when_batch_size_exceeds_pool() {
    let pool = ranking_pool();
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 100, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.selected.len(), pool.len());
    assert_eq!(batch.ranked.len(), pool.len());
}

#[test]
fn select_batch_size_zero_selects_nothing_but_ranks_everything() {
    let pool = ranking_pool();
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 0, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert!(batch.is_empty());
    assert_eq!(batch.selected.len(), 0);
    assert_eq!(batch.ranked.len(), pool.len());
}

#[test]
fn select_batch_empty_pool_is_ok_not_err() {
    let selector = default_selector();
    let batch = selector
        .select_batch(&[], 5, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert!(batch.is_empty());
    assert!(batch.ranked.is_empty());
    assert!(batch.most_uncertain().is_none());
}

#[test]
fn select_batch_single_item_pool() {
    let pool = vec![PoolItem::new("solo", vec![3.0, 1.0])];
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 5, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.selected.len(), 1);
    assert_eq!(batch.selected[0].item_id, "solo");
}

// ── select_batch: error propagation ────────────────────────────────────────

#[test]
fn select_batch_rejects_duplicate_ids() {
    let pool = vec![
        PoolItem::new("dup", vec![1.0, 2.0]),
        PoolItem::new("dup", vec![3.0, 4.0]),
    ];
    let selector = default_selector();
    let err = selector
        .select_batch(&pool, 1, UncertaintyMeasure::MarginSampling)
        .unwrap_err();
    match err {
        ActiveLearningError::DuplicateItemId { item_id } => assert_eq!(item_id, "dup"),
        other => panic!("expected DuplicateItemId, got {other:?}"),
    }
}

#[test]
fn select_batch_propagates_empty_scores_error() {
    let pool = vec![PoolItem::new("bad", vec![]), PoolItem::new("ok", vec![1.0])];
    let selector = default_selector();
    let err = selector
        .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
        .unwrap_err();
    match err {
        ActiveLearningError::EmptyItemScores { item_id } => assert_eq!(item_id, "bad"),
        other => panic!("expected EmptyItemScores, got {other:?}"),
    }
}

#[test]
fn select_batch_propagates_non_finite_score_error() {
    let pool = vec![PoolItem::new("bad", vec![1.0, f32::NAN])];
    let selector = default_selector();
    let err = selector
        .select_batch(&pool, 1, UncertaintyMeasure::EntropySampling)
        .unwrap_err();
    assert!(matches!(err, ActiveLearningError::NonFiniteScore { .. }));
}

#[test]
fn select_batch_rejects_invalid_selector_config() {
    // Config is mutated to an invalid state after construction (the field is
    // public); `select_batch` must re-validate defensively rather than trust
    // the state captured at `new`.
    let mut selector = default_selector();
    selector.config.diversity_threshold = 5.0;
    let err = selector
        .select_batch(&ranking_pool(), 1, UncertaintyMeasure::MarginSampling)
        .unwrap_err();
    assert!(matches!(
        err,
        ActiveLearningError::InvalidDiversityThreshold(_)
    ));
}

// ── select_batch: determinism ──────────────────────────────────────────────

#[test]
fn select_batch_is_deterministic_across_repeated_calls() {
    let pool = vec![
        PoolItem::new("a", vec![5.0, 4.0]),
        PoolItem::new("b", vec![5.0, 4.0]),
        PoolItem::new("c", vec![9.0, 1.0]),
        PoolItem::new("d", vec![1.0, 1.0]),
    ];
    let selector = default_selector();
    let first = selector
        .select_batch(&pool, 3, UncertaintyMeasure::MarginSampling)
        .unwrap();
    let second = selector
        .select_batch(&pool, 3, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(first, second);
}

#[test]
fn select_batch_ties_preserve_original_pool_order() {
    // "a" and "b" carry identical scores (tied uncertainty); the stable sort
    // must keep "a" (which appears first in the input pool) ahead of "b".
    let pool = vec![
        PoolItem::new("a", vec![5.0, 5.0]),
        PoolItem::new("b", vec![5.0, 5.0]),
        PoolItem::new("c", vec![9.0, 1.0]),
    ];
    let selector = default_selector();
    let batch = selector
        .select_batch(&pool, 3, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.ranked[0].item_id, "a");
    assert_eq!(batch.ranked[1].item_id, "b");
    assert_eq!(batch.ranked[2].item_id, "c");
}

// ── select_batch: diversity-aware selection ────────────────────────────────

fn near_duplicate_pool() -> Vec<PoolItem> {
    vec![
        PoolItem::new("a", vec![5.0, 5.0]).with_features(vec![1.0, 0.0, 0.0]),
        // "a2" is a near-duplicate of "a": identical uncertainty score and
        // identical feature vector (cosine similarity 1.0).
        PoolItem::new("a2", vec![5.0, 5.0]).with_features(vec![1.0, 0.0, 0.0]),
        // "b" is distinct from "a"/"a2" (orthogonal features) but less
        // uncertain.
        PoolItem::new("b", vec![9.0, 1.0]).with_features(vec![0.0, 1.0, 0.0]),
    ]
}

#[test]
fn diversity_disabled_selects_both_near_duplicates() {
    let selector =
        ActiveLearningSelector::new(ActiveLearningConfig::new().with_diversity_enabled(false))
            .unwrap();
    let batch = selector
        .select_batch(
            &near_duplicate_pool(),
            2,
            UncertaintyMeasure::MarginSampling,
        )
        .unwrap();
    assert_eq!(batch.selected_ids(), vec!["a", "a2"]);
}

#[test]
fn diversity_enabled_skips_near_duplicate_in_favor_of_distinct_item() {
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new()
            .with_diversity_enabled(true)
            .with_diversity_threshold(0.9),
    )
    .unwrap();
    let batch = selector
        .select_batch(
            &near_duplicate_pool(),
            2,
            UncertaintyMeasure::MarginSampling,
        )
        .unwrap();
    // "a2" is skipped as a near-duplicate of "a"; "b" fills the second slot
    // even though it is less uncertain than "a2" would have been.
    assert_eq!(batch.selected_ids(), vec!["a", "b"]);
    // The full ranking is untouched by the diversity filter -- it still
    // shows "a2" as the (tied) second-most-uncertain item overall.
    assert_eq!(batch.ranked[1].item_id, "a2");
}

#[test]
fn diversity_enabled_returns_fewer_than_batch_size_when_pool_lacks_distinct_items() {
    // Three mutually near-identical items and nothing else: the diversity
    // filter can honestly fill only one slot rather than backfilling with
    // near-duplicates to force the count up to `target`.
    let pool = vec![
        PoolItem::new("a", vec![5.0, 5.0]).with_features(vec![1.0, 0.0, 0.0]),
        PoolItem::new("a2", vec![5.0, 5.0]).with_features(vec![1.0, 0.0, 0.0]),
        PoolItem::new("a3", vec![5.0, 5.0]).with_features(vec![1.0, 0.0, 0.0]),
    ];
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new()
            .with_diversity_enabled(true)
            .with_diversity_threshold(0.9),
    )
    .unwrap();
    let batch = selector
        .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(
        batch.selected.len(),
        1,
        "expected only one mutually-distinct item"
    );
    assert_eq!(batch.selected[0].item_id, "a");
    // The ranking still reports all three items -- only the *selection* is
    // diversity-filtered.
    assert_eq!(batch.ranked.len(), 3);
}

#[test]
fn diversity_enabled_with_empty_features_never_filters() {
    // Neither item supplies a feature vector; cosine similarity between two
    // empty vectors is defined as 0.0, so even an aggressive threshold of
    // 0.0 (anything > 0.0 is filtered) must not treat them as duplicates.
    let pool = vec![
        PoolItem::new("a", vec![5.0, 5.0]),
        PoolItem::new("a2", vec![5.0, 5.0]),
    ];
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new()
            .with_diversity_enabled(true)
            .with_diversity_threshold(0.0),
    )
    .unwrap();
    let batch = selector
        .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.selected.len(), 2);
}

#[test]
fn diversity_threshold_below_actual_similarity_still_filters() {
    // Features are similar but not identical (cosine similarity strictly
    // between 0 and 1); a low threshold must still catch it.
    let pool = vec![
        PoolItem::new("a", vec![5.0, 5.0]).with_features(vec![1.0, 0.1, 0.0]),
        PoolItem::new("a2", vec![5.0, 5.0]).with_features(vec![1.0, 0.2, 0.0]),
    ];
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new()
            .with_diversity_enabled(true)
            .with_diversity_threshold(0.1),
    )
    .unwrap();
    let batch = selector
        .select_batch(&pool, 2, UncertaintyMeasure::MarginSampling)
        .unwrap();
    assert_eq!(batch.selected.len(), 1);
}

// ── select_default_batch convenience ───────────────────────────────────────

#[test]
fn select_default_batch_uses_config_measure_and_batch_size() {
    let pool = vec![
        PoolItem::new("uniform", vec![1.0, 1.0, 1.0]),
        PoolItem::new("peaked", vec![10.0, 0.0, 0.0]),
        PoolItem::new("mid", vec![8.0, 2.0, 0.0]),
    ];
    let selector = ActiveLearningSelector::new(
        ActiveLearningConfig::new()
            .with_measure(UncertaintyMeasure::EntropySampling)
            .with_batch_size(1),
    )
    .unwrap();
    let batch = selector.select_default_batch(&pool).unwrap();
    assert_eq!(batch.measure, UncertaintyMeasure::EntropySampling);
    assert_eq!(batch.selected.len(), 1);
    assert_eq!(batch.selected[0].item_id, "uniform");
}

// ── SelectionBatch accessors ────────────────────────────────────────────────

#[test]
fn selection_batch_is_empty_and_len() {
    let batch = SelectionBatch {
        selected: vec![],
        ranked: vec![],
        measure: UncertaintyMeasure::MarginSampling,
    };
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
}

#[test]
fn selection_batch_selected_ids_preserves_order() {
    let batch = SelectionBatch {
        selected: vec![
            UncertaintySample {
                item_id: "first".to_string(),
                uncertainty_score: 0.9,
            },
            UncertaintySample {
                item_id: "second".to_string(),
                uncertainty_score: 0.5,
            },
        ],
        ranked: vec![],
        measure: UncertaintyMeasure::MarginSampling,
    };
    assert_eq!(batch.selected_ids(), vec!["first", "second"]);
    assert_eq!(batch.len(), 2);
    assert!(!batch.is_empty());
}

#[test]
fn selection_batch_most_uncertain_is_head_of_ranked() {
    let batch = SelectionBatch {
        selected: vec![],
        ranked: vec![
            UncertaintySample {
                item_id: "top".to_string(),
                uncertainty_score: 0.95,
            },
            UncertaintySample {
                item_id: "next".to_string(),
                uncertainty_score: 0.4,
            },
        ],
        measure: UncertaintyMeasure::EntropySampling,
    };
    assert_eq!(batch.most_uncertain().unwrap().item_id, "top");
}

// ── ActiveLearningError Display ─────────────────────────────────────────────

#[test]
fn error_messages_are_specific() {
    let empty = ActiveLearningError::EmptyItemScores {
        item_id: "abc".to_string(),
    };
    assert!(empty.to_string().contains("abc"));
    assert!(empty.to_string().contains("empty"));

    let non_finite = ActiveLearningError::NonFiniteScore {
        item_id: "abc".to_string(),
        index: 3,
    };
    assert!(non_finite.to_string().contains("abc"));
    assert!(non_finite.to_string().contains('3'));

    let dup = ActiveLearningError::DuplicateItemId {
        item_id: "dup-id".to_string(),
    };
    assert!(dup.to_string().contains("dup-id"));

    let bad_threshold = ActiveLearningError::InvalidDiversityThreshold(4.2);
    assert!(bad_threshold.to_string().contains("4.2"));
}

// ── PoolItem / UncertaintySample / SelectionBatch trait smoke tests ────────

#[test]
fn types_implement_debug_and_clone() {
    let item = PoolItem::new("a", vec![1.0]);
    let cloned = item.clone();
    assert_eq!(item, cloned);
    assert!(!format!("{item:?}").is_empty());

    let sample = UncertaintySample {
        item_id: "a".to_string(),
        uncertainty_score: 0.5,
    };
    assert_eq!(sample.clone(), sample);

    let measure = UncertaintyMeasure::MarginSampling;
    assert_eq!(measure, measure);
}
