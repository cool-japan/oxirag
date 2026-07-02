#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::uninlined_format_args
)]
//! Tests for the `conformal_rag` module.
//!
//! The centerpiece is [`empirical_coverage_matches_target`]: a fully
//! deterministic Monte-Carlo simulation that *proves* the calibrated threshold
//! delivers the promised `1 - alpha` marginal coverage. If the finite-sample
//! quantile index (`ceil((n + 1) * (1 - alpha))`) were computed incorrectly, the
//! measured empirical coverage would drift away from `1 - alpha` and the test
//! would fail. The remaining tests pin the index arithmetic itself (including
//! the floating-point traps and the `> n` boundary), threshold monotonicity, the
//! abstain/singleton/ambiguous prediction-set behaviour, the scorers, the error
//! paths, and the Mondrian group-conditional variant.

use super::calibrator::{ConformalCalibrator, conformal_rank};
use super::mondrian::MondrianConformalCalibrator;
use super::types::{
    ConformalConfig, ConformalError, LexicalOverlapScorer, NonconformityScorer, PredictionKind,
};

// ── Deterministic pseudo-randomness (FNV-1a fold + integer finalizer) ─────────
//
// The SciRS2 policy forbids the `rand` crate. These helpers mirror the crate's
// established FNV-1a-seeded pattern (see `rabitq::rotation`) and add a
// splitmix64-style integer finalizer for better equidistribution — all pure
// arithmetic, no dependencies, and fully deterministic so the coverage test is
// never flaky.

/// FNV-1a fold of `(seed, index)` followed by a splitmix64 finalizer.
fn det_bits(seed: u64, index: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in seed.to_le_bytes().iter().chain(index.to_le_bytes().iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    let mut z = h;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A deterministic pseudo-uniform in the open interval `(0, 1)`.
fn det_unit(seed: u64, index: u64) -> f64 {
    (det_bits(seed, index) as f64 + 1.0) / (u64::MAX as f64 + 2.0)
}

/// A deterministic pseudo-random exponential draw with the given `scale`
/// (mean `scale`). Continuous and unbounded — a decidedly non-uniform shape,
/// which is the point: conformal coverage is distribution-free.
fn det_exponential(seed: u64, index: u64, scale: f64) -> f64 {
    -scale * det_unit(seed, index).ln()
}

/// The empirical coverage of `test` scores under `calibrator`: the fraction of
/// test scores admitted by the calibrated threshold.
fn empirical_coverage(calibrator: &ConformalCalibrator, test: &[f64]) -> f64 {
    let admitted = test.iter().filter(|&&s| calibrator.admits(s)).count();
    admitted as f64 / test.len() as f64
}

// ── THE coverage validation test (mandatory centerpiece) ──────────────────────

/// Empirically proves the marginal coverage guarantee `P(s <= threshold) >= 1
/// - alpha` holds — and holds *tightly around* `1 - alpha`, not merely above it.
///
/// Design:
/// * Simulate a large population of i.i.d. nonconformity scores drawn from a
///   known **exponential** distribution (skewed, unbounded — emphatically not
///   the shape any naive method assumes).
/// * Split into a calibration set (`n_cal = 8000`) and a disjoint held-out test
///   set (`n_test = 40000`). Because the draws are i.i.d., calibration and test
///   are exchangeable, satisfying the theorem's only hypothesis.
/// * Calibrate a threshold at several `alpha`, then measure the fraction of test
///   scores at/under the threshold — the exact quantity the theorem bounds.
///
/// Tolerance: the coverage of a *single* split, conditioned on the calibration
/// draw, is Beta-distributed with mean `~1 - alpha` and standard deviation
/// `~sqrt((1-alpha)*alpha / n_cal)` (`~0.0034` at `n_cal = 8000, alpha = 0.1`);
/// the additional binomial noise from a finite test set of `n_test = 40000` adds
/// `~sqrt((1-alpha)*alpha / n_test)` (`~0.0015`). The combined standard
/// deviation is `~0.0037`, so a `±0.02` band around `1 - alpha` is `> 5` sigma —
/// wide enough that a *correct* implementation never trips it, yet tight enough
/// that any real quantile bug (wrong direction, wrong index, `>=` vs `<=`,
/// unsorted scores) fails loudly. The test is deterministic, so this either
/// always passes or always fails.
#[test]
fn empirical_coverage_matches_target() {
    const N_CAL: usize = 8000;
    const N_TEST: usize = 40000;
    const SEED: u64 = 0x00c0_ffee_1234_5678;
    const SCALE: f64 = 1.7;
    const TOL: f64 = 0.02;

    // One shared i.i.d. exponential population, split into calibration + test.
    let pool: Vec<f64> = (0..(N_CAL + N_TEST) as u64)
        .map(|i| det_exponential(SEED, i, SCALE))
        .collect();
    let (cal, test) = pool.split_at(N_CAL);

    for &alpha in &[0.05_f64, 0.10, 0.20] {
        let cfg = ConformalConfig::new().with_alpha(alpha);
        let calibrator = ConformalCalibrator::calibrate(cfg, cal).expect("calibration succeeds");
        let coverage = empirical_coverage(&calibrator, test);
        let target = 1.0 - alpha;

        println!(
            "[conformal coverage] alpha={alpha:.2} target={target:.4} \
             measured={coverage:.4} threshold={:.4} rank={}/{}",
            calibrator.threshold(),
            calibrator.rank(),
            calibrator.calibration_size(),
        );

        assert!(
            (coverage - target).abs() <= TOL,
            "alpha={alpha}: empirical coverage {coverage:.4} strays from target \
             {target:.4} by more than {TOL} — the finite-sample quantile is wrong",
        );
        // The theorem's core promise: coverage is at least 1 - alpha (up to the
        // documented sampling tolerance).
        assert!(
            coverage >= target - TOL,
            "alpha={alpha}: coverage {coverage:.4} under-covers {target:.4}",
        );
    }
}

/// Averaging the empirical coverage over many independent calibration/test
/// splits estimates the *marginal* coverage, which converges to the exact
/// closed form `ceil((n + 1) * (1 - alpha)) / (n + 1)`. For `n_cal = 1000,
/// alpha = 0.1` that is `901 / 1001 = 0.90010`. This cross-checks the coverage
/// theorem from the averaged angle and is far more sensitive to a directional
/// mistake than any single split.
#[test]
fn marginal_coverage_converges_to_closed_form() {
    const TRIALS: u64 = 120;
    const N_CAL: usize = 1000;
    const N_TEST: usize = 1000;
    const ALPHA: f64 = 0.10;
    const BASE_SEED: u64 = 0xa5a5_0f0f_dead_c0de;

    let expected = conformal_rank(N_CAL, ALPHA) as f64 / (N_CAL as f64 + 1.0);

    let mut coverage_sum = 0.0;
    for trial in 0..TRIALS {
        // Independent data per trial via a per-trial seed.
        let seed = det_bits(BASE_SEED, trial);
        let pool: Vec<f64> = (0..(N_CAL + N_TEST) as u64)
            .map(|i| det_exponential(seed, i, 2.3))
            .collect();
        let (cal, test) = pool.split_at(N_CAL);
        let cfg = ConformalConfig::new().with_alpha(ALPHA);
        let calibrator = ConformalCalibrator::calibrate(cfg, cal).expect("calibration succeeds");
        coverage_sum += empirical_coverage(&calibrator, test);
    }
    let avg = coverage_sum / TRIALS as f64;

    println!("[conformal marginal] trials={TRIALS} expected={expected:.5} measured_avg={avg:.5}");
    assert!(
        (avg - expected).abs() < 0.008,
        "averaged marginal coverage {avg:.5} deviates from closed form {expected:.5}",
    );
}

/// Coverage is *distribution-free*: it holds at `1 - alpha` regardless of the
/// score distribution. We re-run the coverage measurement on a uniform
/// distribution and on a bimodal mixture and confirm the same target holds.
#[test]
fn coverage_is_distribution_free() {
    const N_CAL: usize = 8000;
    const N_TEST: usize = 40000;
    const ALPHA: f64 = 0.10;
    const TOL: f64 = 0.02;
    let target = 1.0 - ALPHA;

    // (a) Uniform on (0, 1).
    let uniform: Vec<f64> = (0..(N_CAL + N_TEST) as u64)
        .map(|i| det_unit(0x1111_2222_3333_4444, i))
        .collect();

    // (b) Bimodal mixture: half in [0, 0.3), half in [0.6, 1.0). Continuous
    // within each band, so effectively tie-free.
    let bimodal: Vec<f64> = (0..(N_CAL + N_TEST) as u64)
        .map(|i| {
            let u = det_unit(0x5555_6666_7777_8888, i);
            let which = det_unit(0x9999_aaaa_bbbb_cccc, i);
            if which < 0.5 { u * 0.3 } else { 0.6 + u * 0.4 }
        })
        .collect();

    for (name, pool) in [("uniform", &uniform), ("bimodal", &bimodal)] {
        let (cal, test) = pool.split_at(N_CAL);
        let cfg = ConformalConfig::new().with_alpha(ALPHA);
        let calibrator = ConformalCalibrator::calibrate(cfg, cal).expect("calibration succeeds");
        let coverage = empirical_coverage(&calibrator, test);
        println!("[conformal dist-free] {name}: coverage={coverage:.4} target={target:.4}");
        assert!(
            (coverage - target).abs() <= TOL,
            "{name}: coverage {coverage:.4} strays from {target:.4}",
        );
    }
}

// ── Finite-sample quantile index arithmetic ───────────────────────────────────

/// The finite-sample rank `k = ceil((n + 1) * (1 - alpha))` — including the
/// floating-point traps where `(n + 1) * (1 - alpha)` is mathematically an
/// integer but the `f64` product lands at `integer ± epsilon`. A naive `ceil`
/// over-counts by one on those; the snap-to-integer guard fixes it. These
/// assertions are the precise proof that the `(n + 1)` finite-sample correction
/// is implemented exactly (a `1 / (n + 1)` effect too small for the coverage
/// test to resolve).
#[test]
fn conformal_rank_is_exact_including_float_traps() {
    // Ordinary non-integer targets.
    assert_eq!(conformal_rank(100, 0.1), 91); // 101 * 0.9 = 90.9  -> 91
    assert_eq!(conformal_rank(10, 0.2), 9); //  11 * 0.8 = 8.8   -> 9

    // Float traps: exact integers that the f64 product perturbs upward.
    assert_eq!(conformal_rank(99, 0.1), 90); // 100 * 0.9 = 90.0  -> 90, NOT 91
    assert_eq!(conformal_rank(999, 0.1), 900); // 1000 * 0.9 = 900 -> 900, NOT 901
    assert_eq!(conformal_rank(9, 0.2), 8); //  10 * 0.8 = 8.0   -> 8, NOT 9
    assert_eq!(conformal_rank(19, 0.05), 19); // 20 * 0.95 = 19.0 -> 19, NOT 20

    // Boundary regime: k > n  (alpha < 1 / (n + 1))  =>  k == n + 1.
    assert_eq!(conformal_rank(5, 0.05), 6); // 6 * 0.95 = 5.7 -> 6 > 5
    assert_eq!(conformal_rank(18, 0.05), 19); // 19 * 0.95 = 18.05 -> 19 > 18
    assert_eq!(conformal_rank(1, 0.4), 2); // 2 * 0.6 = 1.2 -> 2 > 1

    // Smallest calibration set that still yields a finite threshold.
    assert_eq!(conformal_rank(1, 0.5), 1); // 2 * 0.5 = 1.0 -> 1
}

/// When `ceil((n + 1) * (1 - alpha)) > n`, the threshold must be `+infinity` and
/// the calibrator must admit *every* candidate — the boundary case must be
/// handled explicitly, never index out of bounds or panic.
#[test]
fn boundary_case_admits_everything() {
    // n = 5, alpha = 0.05 -> rank 6 > 5 -> trivial (include-all).
    let cal = [0.1, 0.2, 0.3, 0.4, 0.5];
    let cfg = ConformalConfig::new().with_alpha(0.05);
    let calibrator = ConformalCalibrator::calibrate(cfg, &cal).expect("calibrates");

    assert!(calibrator.is_trivial());
    assert!(calibrator.threshold().is_infinite());
    assert_eq!(calibrator.rank(), 6);
    assert_eq!(calibrator.calibration_size(), 5);

    // Even an absurdly nonconforming candidate is admitted.
    assert!(calibrator.admits(1.0e9));
    let set = calibrator.predict_set_from_scores(&[0.0, 5.0, 1.0e12]);
    assert_eq!(set.len(), 3);
    assert_eq!(set.kind, PredictionKind::Ambiguous);
    assert_eq!(set.threshold, f64::INFINITY);

    // n = 1, alpha = 0.4 -> rank 2 > 1 -> trivial.
    let tiny = ConformalCalibrator::calibrate(ConformalConfig::new().with_alpha(0.4), &[0.7])
        .expect("calibrates");
    assert!(tiny.is_trivial());
    assert!(tiny.admits(f64::MAX));
}

/// A finite threshold selects exactly the k-th smallest calibration score.
#[test]
fn threshold_is_the_kth_order_statistic() {
    // Unsorted on purpose to exercise the internal sort.
    let cal = [0.5, 0.1, 1.0, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6];
    let cfg = ConformalConfig::new().with_alpha(0.2);
    let calibrator = ConformalCalibrator::calibrate(cfg, &cal).expect("calibrates");

    // n = 10, alpha = 0.2 -> rank = ceil(11 * 0.8) = ceil(8.8) = 9.
    assert_eq!(calibrator.rank(), 9);
    // 9th smallest of 0.1..=1.0 is 0.9.
    assert_eq!(calibrator.threshold(), 0.9);
    assert!(!calibrator.is_trivial());
}

/// Threshold (hence prediction-set size) is monotone non-increasing in `alpha`:
/// tolerating more miscoverage can only shrink the admissible set.
#[test]
fn threshold_is_monotone_in_alpha() {
    let cal: Vec<f64> = (1..=50).map(|i| f64::from(i) / 50.0).collect();
    let alphas = [0.01_f64, 0.02, 0.05, 0.1, 0.2, 0.5];

    let mut prev = f64::INFINITY;
    let candidates: Vec<f64> = (0..20).map(|i| f64::from(i) / 20.0).collect();
    let mut prev_admitted = usize::MAX;

    for &alpha in &alphas {
        let cfg = ConformalConfig::new().with_alpha(alpha);
        let calibrator = ConformalCalibrator::calibrate(cfg, &cal).expect("calibrates");
        let thr = calibrator.threshold();
        assert!(
            thr <= prev,
            "threshold rose from {prev} to {thr} as alpha grew to {alpha}",
        );
        prev = thr;

        let admitted = calibrator.predict_set_from_scores(&candidates).len();
        assert!(
            admitted <= prev_admitted,
            "admitted count rose from {prev_admitted} to {admitted} at alpha {alpha}",
        );
        prev_admitted = admitted;
    }
    // The smallest alpha here (0.01, n = 50) is in the include-all regime.
    let trivial = ConformalCalibrator::calibrate(ConformalConfig::new().with_alpha(0.01), &cal)
        .expect("calibrates");
    assert!(trivial.is_trivial());
}

// ── Prediction-set shape (abstain / singleton / ambiguous) ────────────────────

/// Crafted examples covering every prediction-set shape, including the `<=`
/// boundary (a candidate exactly at the threshold is admitted).
#[test]
fn prediction_set_shapes() {
    // Threshold = 0.9 (n = 10, alpha = 0.2 -> 9th order statistic).
    let cal: Vec<f64> = (1..=10).map(|i| f64::from(i) / 10.0).collect();
    let cfg = ConformalConfig::new().with_alpha(0.2);
    let calibrator = ConformalCalibrator::calibrate(cfg, &cal).expect("calibrates");
    assert_eq!(calibrator.threshold(), 0.9);

    // Multiple admitted -> Ambiguous; members sorted ascending; best() is min.
    let multi = calibrator.predict_set_from_scores(&[0.5, 0.05, 0.95, 0.3]);
    assert_eq!(multi.kind, PredictionKind::Ambiguous);
    assert!(multi.is_ambiguous());
    assert_eq!(multi.len(), 3); // 0.95 excluded
    assert_eq!(multi.total_candidates, 4);
    assert_eq!(multi.best().expect("non-empty").nonconformity, 0.05);
    assert_eq!(multi.best().expect("non-empty").index, 1);
    // Ascending order.
    let scores: Vec<f64> = multi.members.iter().map(|m| m.nonconformity).collect();
    assert_eq!(scores, vec![0.05, 0.3, 0.5]);

    // Exactly one admitted -> Singleton.
    let single = calibrator.predict_set_from_scores(&[0.99, 0.3, 1.0]);
    assert_eq!(single.kind, PredictionKind::Singleton);
    assert!(single.is_singleton());
    assert_eq!(single.len(), 1);
    assert_eq!(single.best().expect("non-empty").nonconformity, 0.3);

    // None admitted -> Abstain.
    let none = calibrator.predict_set_from_scores(&[0.95, 0.99, 1.0]);
    assert_eq!(none.kind, PredictionKind::Abstain);
    assert!(none.is_abstain());
    assert!(none.is_empty());
    assert!(none.best().is_none());

    // The `<=` boundary: a candidate exactly at the threshold is admitted.
    let boundary = calibrator.predict_set_from_scores(&[0.9]);
    assert_eq!(boundary.kind, PredictionKind::Singleton);
    assert_eq!(boundary.len(), 1);

    // Empty candidate list -> Abstain, no panic.
    let empty = calibrator.predict_set_from_scores(&[]);
    assert!(empty.is_abstain());
    assert_eq!(empty.total_candidates, 0);
}

/// A non-finite candidate score is never admitted (`NaN <= x` is false).
#[test]
fn non_finite_candidate_scores_are_rejected() {
    let cal: Vec<f64> = (1..=10).map(|i| f64::from(i) / 10.0).collect();
    let calibrator = ConformalCalibrator::calibrate(ConformalConfig::new().with_alpha(0.2), &cal)
        .expect("calibrates");
    let set = calibrator.predict_set_from_scores(&[0.1, f64::NAN, f64::INFINITY]);
    assert_eq!(set.len(), 1);
    assert_eq!(set.best().expect("non-empty").nonconformity, 0.1);
}

// ── Scorers ───────────────────────────────────────────────────────────────────

#[test]
fn lexical_overlap_scorer_semantics() {
    let s = LexicalOverlapScorer::new();
    // Identical token sets -> Jaccard 1 -> nonconformity 0.
    assert_eq!(s.nonconformity("the cat sat", "the cat sat"), 0.0);
    // Disjoint token sets -> Jaccard 0 -> nonconformity 1.
    assert_eq!(s.nonconformity("alpha beta", "gamma delta"), 1.0);
    // Both empty -> defined as maximally nonconforming.
    assert_eq!(s.nonconformity("", ""), 1.0);
    // Query non-empty, candidate empty -> no overlap -> 1.
    assert_eq!(s.nonconformity("hello", ""), 1.0);
    // Partial overlap lies strictly between 0 and 1 and is case/punct-robust.
    let partial = s.nonconformity("capital of France", "The capital city, France!");
    assert!(partial > 0.0 && partial < 1.0, "got {partial}");
    // "capital","of","france" vs "the","capital","city","france":
    // intersection {capital, france} = 2, union = 5 -> Jaccard 0.4 -> s = 0.6.
    assert!((partial - 0.6).abs() < 1e-12, "got {partial}");
}

#[test]
fn predict_set_with_lexical_scorer_filters_candidates() {
    // Calibrate a threshold that admits fairly-relevant but not irrelevant text.
    let cal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let calibrator = ConformalCalibrator::calibrate(ConformalConfig::new().with_alpha(0.2), &cal)
        .expect("calibrates"); // threshold = 9th smallest = 0.8

    let scorer = LexicalOverlapScorer::new();
    let query = "capital of france";
    let candidates = [
        "The capital of France is Paris.",
        "Bananas are an excellent source of potassium.",
    ];
    let set = calibrator.predict_set(&scorer, query, &candidates);
    assert_eq!(set.len(), 1);
    let best = set.best().expect("non-empty");
    assert_eq!(best.index, 0);
    assert_eq!(
        best.label.as_deref(),
        Some("The capital of France is Paris.")
    );
}

#[test]
fn closure_implements_nonconformity_scorer() {
    // Any Fn(&str, &str) -> f64 is a NonconformityScorer.
    let scorer = |_query: &str, candidate: &str| -> f64 {
        // Nonconformity proportional to candidate length (toy example).
        candidate.len() as f64 / 100.0
    };
    let calibrator = ConformalCalibrator::calibrate(
        ConformalConfig::new().with_alpha(0.2),
        &(1..=10).map(|i| f64::from(i) / 20.0).collect::<Vec<_>>(),
    )
    .expect("calibrates"); // threshold = 9th smallest of 0.05..=0.5 = 0.45

    let candidates = [
        "short",
        "a_much_longer_candidate_string_exceeding_the_bar_by_far",
    ];
    let set = calibrator.predict_set(&scorer, "q", &candidates);
    assert_eq!(set.len(), 1);
    assert_eq!(set.best().expect("non-empty").index, 0);
}

// ── calibrate_labeled ─────────────────────────────────────────────────────────

#[test]
fn calibrate_labeled_uses_only_correct_examples() {
    let labeled = [
        (0.1, true),
        (0.9, false), // ignored
        (0.2, true),
        (0.8, false), // ignored
        (0.3, true),
        (0.4, true),
        (0.5, true),
    ];
    let from_labeled =
        ConformalCalibrator::calibrate_labeled(ConformalConfig::new().with_alpha(0.2), &labeled)
            .expect("calibrates");
    // Equivalent to calibrating directly on the true-labelled scores.
    let direct = ConformalCalibrator::calibrate(
        ConformalConfig::new().with_alpha(0.2),
        &[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    .expect("calibrates");
    assert_eq!(from_labeled.threshold(), direct.threshold());
    assert_eq!(from_labeled.calibration_size(), 5);
}

#[test]
fn calibrate_labeled_with_no_correct_examples_errors() {
    let labeled = [(0.1, false), (0.2, false)];
    let err = ConformalCalibrator::calibrate_labeled(ConformalConfig::default(), &labeled)
        .expect_err("must error");
    assert_eq!(err, ConformalError::EmptyCalibration);
}

// ── Error paths ───────────────────────────────────────────────────────────────

#[test]
fn invalid_alpha_is_rejected() {
    for bad in [0.0_f64, 1.0, 1.5, -0.1, f64::NAN, f64::INFINITY] {
        let cfg = ConformalConfig::new().with_alpha(bad);
        let err = ConformalCalibrator::calibrate(cfg, &[0.1, 0.2, 0.3]).expect_err("must error");
        assert!(
            matches!(err, ConformalError::InvalidAlpha(_)),
            "alpha={bad}"
        );
    }
}

#[test]
fn empty_calibration_is_rejected() {
    let err = ConformalCalibrator::calibrate(ConformalConfig::default(), &[]).expect_err("errors");
    assert_eq!(err, ConformalError::EmptyCalibration);
}

#[test]
fn non_finite_calibration_score_is_rejected() {
    let nan = ConformalCalibrator::calibrate(ConformalConfig::default(), &[0.1, f64::NAN, 0.3])
        .expect_err("errors");
    assert_eq!(nan, ConformalError::NonFiniteScore { index: 1 });
    let inf =
        ConformalCalibrator::calibrate(ConformalConfig::default(), &[0.1, 0.2, f64::INFINITY])
            .expect_err("errors");
    assert_eq!(inf, ConformalError::NonFiniteScore { index: 2 });
}

// ── Config ────────────────────────────────────────────────────────────────────

#[test]
fn config_defaults_and_builders() {
    let d = ConformalConfig::default();
    assert_eq!(d.alpha, 0.1);
    assert_eq!(d.min_group_size, 1);
    assert!(d.validate().is_ok());

    let c = ConformalConfig::new()
        .with_alpha(0.05)
        .with_min_group_size(50);
    assert_eq!(c.alpha, 0.05);
    assert_eq!(c.min_group_size, 50);
    assert!(c.validate().is_ok());

    assert!(ConformalConfig::new().with_alpha(0.0).validate().is_err());
    assert!(ConformalConfig::new().with_alpha(1.0).validate().is_err());
}

#[test]
fn prediction_kind_helpers() {
    assert_eq!(PredictionKind::from_len(0), PredictionKind::Abstain);
    assert_eq!(PredictionKind::from_len(1), PredictionKind::Singleton);
    assert_eq!(PredictionKind::from_len(2), PredictionKind::Ambiguous);
    assert_eq!(PredictionKind::from_len(9), PredictionKind::Ambiguous);
    assert_eq!(PredictionKind::Abstain.as_str(), "abstain");
    assert_eq!(PredictionKind::Singleton.to_string(), "singleton");
    assert_eq!(PredictionKind::Ambiguous.as_str(), "ambiguous");
}

// ── Mondrian (group-conditional) ──────────────────────────────────────────────

#[test]
fn mondrian_calibrates_per_group_thresholds() {
    // "easy" group has small scores; "hard" group has large scores.
    let mut examples: Vec<(String, f64)> = Vec::new();
    for i in 1..=20 {
        examples.push(("easy".to_string(), f64::from(i) / 100.0)); // 0.01..=0.20
        examples.push(("hard".to_string(), f64::from(i) / 20.0)); // 0.05..=1.00
    }
    let cfg = ConformalConfig::new().with_alpha(0.2);
    let mondrian = MondrianConformalCalibrator::calibrate(cfg, &examples).expect("calibrates");

    assert_eq!(mondrian.group_count(), 2);
    let groups: Vec<&str> = mondrian.groups().collect();
    assert_eq!(groups, vec!["easy", "hard"]); // BTreeMap => sorted, deterministic

    let easy_thr = mondrian.threshold_for("easy");
    let hard_thr = mondrian.threshold_for("hard");
    assert!(
        easy_thr < hard_thr,
        "expected tighter easy threshold ({easy_thr}) < hard ({hard_thr})",
    );

    // An unseen group falls back to the pooled global threshold.
    let unseen = mondrian.threshold_for("medium");
    assert_eq!(unseen, mondrian.global().threshold());
    assert!(mondrian.group_calibrator("medium").is_none());

    // Group-aware prediction uses the per-group threshold.
    let easy_set = mondrian.predict_set_from_scores("easy", &[0.05, 0.5]);
    assert_eq!(easy_set.len(), 1); // 0.5 far exceeds the easy threshold
    let hard_set = mondrian.predict_set_from_scores("hard", &[0.05, 0.5]);
    assert_eq!(hard_set.len(), 2); // both within the looser hard threshold
}

#[test]
fn mondrian_min_group_size_falls_back_to_global() {
    let mut examples: Vec<(String, f64)> = Vec::new();
    for i in 1..=30 {
        examples.push(("big".to_string(), f64::from(i) / 30.0));
    }
    examples.push(("tiny".to_string(), 0.5)); // single sample
    examples.push(("tiny".to_string(), 0.6)); // two samples total

    let cfg = ConformalConfig::new()
        .with_alpha(0.1)
        .with_min_group_size(5);
    let mondrian = MondrianConformalCalibrator::calibrate(cfg, &examples).expect("calibrates");

    // "tiny" (2 < 5) does not earn its own calibrator.
    assert!(mondrian.group_calibrator("tiny").is_none());
    assert_eq!(
        mondrian.threshold_for("tiny"),
        mondrian.global().threshold()
    );
    // "big" (30 >= 5) does.
    assert!(mondrian.group_calibrator("big").is_some());
    assert_eq!(mondrian.group_count(), 1);
}

/// Mondrian delivers the `1 - alpha` guarantee *conditionally within each group*
/// even when the groups have wildly different score distributions.
#[test]
fn mondrian_group_conditional_coverage() {
    const N_CAL: usize = 4000;
    const N_TEST: usize = 8000;
    const ALPHA: f64 = 0.10;
    const TOL: f64 = 0.02;
    let target = 1.0 - ALPHA;

    // Group A: exponential scale 1. Group B: exponential scale 6.
    let group_specs: [(&str, u64, f64); 2] = [
        ("a", 0x0101_0101_0101_0101, 1.0),
        ("b", 0x0202_0202_0202_0202, 6.0),
    ];

    // Build a combined calibration set (group, score).
    let mut cal_examples: Vec<(String, f64)> = Vec::new();
    let mut test_by_group: Vec<(&str, Vec<f64>)> = Vec::new();
    for &(name, seed, scale) in &group_specs {
        for i in 0..N_CAL as u64 {
            cal_examples.push((name.to_string(), det_exponential(seed, i, scale)));
        }
        let test: Vec<f64> = (N_CAL as u64..(N_CAL + N_TEST) as u64)
            .map(|i| det_exponential(seed, i, scale))
            .collect();
        test_by_group.push((name, test));
    }

    let cfg = ConformalConfig::new().with_alpha(ALPHA);
    let mondrian = MondrianConformalCalibrator::calibrate(cfg, &cal_examples).expect("calibrates");

    for (name, test) in &test_by_group {
        let calibrator = mondrian.calibrator_for(name);
        let coverage = empirical_coverage(calibrator, test);
        println!(
            "[mondrian coverage] group={name} coverage={coverage:.4} target={target:.4} \
             threshold={:.4}",
            calibrator.threshold(),
        );
        assert!(
            (coverage - target).abs() <= TOL,
            "group {name}: coverage {coverage:.4} strays from {target:.4}",
        );
    }

    // The two groups genuinely learned different thresholds.
    assert!(mondrian.threshold_for("a") < mondrian.threshold_for("b"));
}
