#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use super::detector::WatermarkDetector;
use super::generator::WatermarkGenerator;
use super::hasher::{SplitMix64, WatermarkHasher};
use super::stats::{erf, normal_cdf, p_value_from_z, z_score, z_score_and_p_value};
use super::types::{WatermarkConfig, WatermarkError, WatermarkMode, WatermarkTokenId};

// ── Test helpers (no `rand`; hand-rolled `splitmix64`, mirrors production code) ─

/// Deterministic synthetic logits for one decoding step, with real entropy
/// (no single dominant token by construction): a `splitmix64` stream keyed
/// on `(seed, step)` fills `vocab_size` values spread over `[-5.0, 5.0)`.
fn synthetic_logits(step: usize, vocab_size: usize, seed: u64) -> Vec<f64> {
    let mixed = seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut rng = SplitMix64::new(mixed);
    (0..vocab_size)
        .map(|_| {
            let bits = rng.next_u64();
            let unit = (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            unit * 10.0 - 5.0
        })
        .collect()
}

/// Plain arg-max decoding with **no** watermark bias at all: the
/// "non-watermarked" baseline used to build true-negative / calibration
/// text.
fn unbiased_argmax_sequence(logits_per_step: &[Vec<f64>]) -> Vec<WatermarkTokenId> {
    logits_per_step
        .iter()
        .map(|logits| {
            let mut best_idx = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for (idx, &value) in logits.iter().enumerate() {
                if value.total_cmp(&best_val) == std::cmp::Ordering::Greater {
                    best_val = value;
                    best_idx = idx;
                }
            }
            best_idx as WatermarkTokenId
        })
        .collect()
}

/// Deterministically substitute an `edit_fraction` of `tokens` with a
/// different token id, decided position-by-position from a `splitmix64`
/// stream seeded by `edit_seed`. The substitution decisions depend only on
/// position (not on token content), so calling this with the same
/// `(edit_fraction, edit_seed)` against two different token sequences edits
/// the *same relative positions* in both — what the robustness comparison
/// below relies on.
fn apply_edits(
    tokens: &[WatermarkTokenId],
    edit_fraction: f64,
    vocab_size: usize,
    edit_seed: u64,
) -> Vec<WatermarkTokenId> {
    let mut rng = SplitMix64::new(edit_seed);
    tokens
        .iter()
        .map(|&tok| {
            let r = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let replacement_bits = rng.next_u64();
            if r < edit_fraction {
                let mut new_tok = (replacement_bits % vocab_size as u64) as WatermarkTokenId;
                if new_tok == tok {
                    new_tok = (new_tok + 1) % vocab_size as WatermarkTokenId;
                }
                new_tok
            } else {
                tok
            }
        })
        .collect()
}

// ── WatermarkConfig: defaults, builders, validation ──────────────────────────

#[test]
fn config_new_has_documented_defaults() {
    let cfg = WatermarkConfig::new(1000, 42);
    assert_eq!(cfg.gamma, 0.25);
    assert_eq!(cfg.delta, 2.0);
    assert_eq!(cfg.context_width, 1);
    assert_eq!(cfg.secret_key, 42);
    assert_eq!(cfg.vocab_size, 1000);
    assert_eq!(cfg.z_threshold, 4.0);
}

#[test]
fn config_builders_override_fields() {
    let cfg = WatermarkConfig::new(500, 7)
        .with_gamma(0.5)
        .with_delta(3.5)
        .with_context_width(4)
        .with_secret_key(99)
        .with_vocab_size(600)
        .with_z_threshold(3.0);
    assert_eq!(cfg.gamma, 0.5);
    assert_eq!(cfg.delta, 3.5);
    assert_eq!(cfg.context_width, 4);
    assert_eq!(cfg.secret_key, 99);
    assert_eq!(cfg.vocab_size, 600);
    assert_eq!(cfg.z_threshold, 3.0);
}

#[test]
fn config_validate_accepts_defaults() {
    assert!(WatermarkConfig::new(100, 1).validate().is_ok());
}

#[test]
fn config_validate_rejects_gamma_out_of_range() {
    let cfg = WatermarkConfig::new(100, 1).with_gamma(-0.1);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidGamma { .. })
    ));
    let cfg = WatermarkConfig::new(100, 1).with_gamma(1.1);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidGamma { .. })
    ));
    let cfg = WatermarkConfig::new(100, 1).with_gamma(f64::NAN);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidGamma { .. })
    ));
}

#[test]
fn config_validate_accepts_gamma_extremes() {
    assert!(
        WatermarkConfig::new(100, 1)
            .with_gamma(0.0)
            .validate()
            .is_ok()
    );
    assert!(
        WatermarkConfig::new(100, 1)
            .with_gamma(1.0)
            .validate()
            .is_ok()
    );
}

#[test]
fn config_validate_rejects_non_finite_delta() {
    let cfg = WatermarkConfig::new(100, 1).with_delta(f64::INFINITY);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidDelta { .. })
    ));
}

#[test]
fn config_validate_rejects_zero_context_width() {
    let cfg = WatermarkConfig::new(100, 1).with_context_width(0);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidContextWidth { .. })
    ));
}

#[test]
fn config_validate_rejects_zero_vocab_size() {
    let cfg = WatermarkConfig::new(0, 1);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidVocabSize { .. })
    ));
}

#[test]
fn config_validate_rejects_non_finite_z_threshold() {
    let cfg = WatermarkConfig::new(100, 1).with_z_threshold(f64::NAN);
    assert!(matches!(
        cfg.validate(),
        Err(WatermarkError::InvalidZThreshold { .. })
    ));
}

// ── WatermarkHasher: determinism ─────────────────────────────────────────────

#[test]
fn hasher_same_key_same_context_is_identical() {
    let cfg = WatermarkConfig::new(500, 0xDEAD_BEEF);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");

    let context = [17u32];
    let mask_a = hasher.green_mask(&context).expect("valid context");
    let mask_b = hasher.green_mask(&context).expect("valid context");
    assert_eq!(mask_a, mask_b, "same key + same context must be identical");

    let seed_a = hasher.context_seed(&context).expect("valid context");
    let seed_b = hasher.context_seed(&context).expect("valid context");
    assert_eq!(seed_a, seed_b);
}

#[test]
fn hasher_different_key_yields_different_partition() {
    let context = [17u32];
    let hasher_a = WatermarkHasher::new(&WatermarkConfig::new(500, 1)).expect("valid config");
    let hasher_b = WatermarkHasher::new(&WatermarkConfig::new(500, 2)).expect("valid config");

    let mask_a = hasher_a.green_mask(&context).expect("valid context");
    let mask_b = hasher_b.green_mask(&context).expect("valid context");
    assert_ne!(
        mask_a, mask_b,
        "different secret keys must (overwhelmingly likely) yield different partitions"
    );
}

#[test]
fn hasher_different_context_yields_different_partition() {
    let cfg = WatermarkConfig::new(500, 12345);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");

    let mask_a = hasher.green_mask(&[1]).expect("valid context");
    let mask_b = hasher.green_mask(&[2]).expect("valid context");
    assert_ne!(mask_a, mask_b);
}

#[test]
fn hasher_context_length_mismatch_errors() {
    let cfg = WatermarkConfig::new(500, 1).with_context_width(2);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    assert!(matches!(
        hasher.green_mask(&[1]),
        Err(WatermarkError::ContextLengthMismatch { .. })
    ));
    assert!(matches!(
        hasher.green_mask(&[1, 2, 3]),
        Err(WatermarkError::ContextLengthMismatch { .. })
    ));
}

#[test]
fn hasher_is_green_rejects_out_of_range_token() {
    let cfg = WatermarkConfig::new(10, 1);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    assert!(matches!(
        hasher.is_green(&[0], 10),
        Err(WatermarkError::TokenOutOfRange { .. })
    ));
    assert!(hasher.is_green(&[0], 9).is_ok());
}

#[test]
fn hasher_multi_token_context_width_changes_seed() {
    // h=2: swapping the order of the two context tokens must change the
    // seed (order matters, this isn't a commutative hash).
    let cfg = WatermarkConfig::new(200, 7).with_context_width(2);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    let seed_ab = hasher.context_seed(&[3, 9]).expect("valid context");
    let seed_ba = hasher.context_seed(&[9, 3]).expect("valid context");
    assert_ne!(seed_ab, seed_ba);
}

// ── WatermarkHasher: green-list size and unbiasedness ────────────────────────

#[test]
fn hasher_green_list_size_is_exact_for_every_context() {
    // vocab_size chosen so gamma * vocab_size is an exact integer for every
    // gamma below, so there is no rounding artifact to confound the check.
    let vocab_size = 200;
    for &gamma in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let cfg = WatermarkConfig::new(vocab_size, 555).with_gamma(gamma);
        let hasher = WatermarkHasher::new(&cfg).expect("valid config");
        let expected_green_size = (gamma * vocab_size as f64).round() as usize;
        assert_eq!(hasher.green_size(), expected_green_size);

        for context in [[0u32], [1], [500], [999_999]] {
            let mask = hasher.green_mask(&context).expect("valid context");
            let count = mask.iter().filter(|&&g| g).count();
            assert_eq!(
                count, expected_green_size,
                "gamma={gamma} context={context:?}: green count must be exact, not just approximate"
            );
        }
    }
}

#[test]
fn hasher_green_membership_is_unbiased_across_contexts() {
    // For a *fixed* token id, its green-membership rate across many
    // *different* predecessor contexts should converge to gamma. This is
    // the statistical property the exact-per-context partition size alone
    // does not prove: it rules out a broken hash that (say) always marks
    // low-numbered contexts green regardless of the token being tested.
    let vocab_size = 200;
    let num_contexts = 4000u32;
    let fixed_token_id: WatermarkTokenId = 17;

    for &gamma in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let cfg = WatermarkConfig::new(vocab_size, 2024).with_gamma(gamma);
        let hasher = WatermarkHasher::new(&cfg).expect("valid config");

        let mut green_hits = 0u32;
        for context_token in 0..num_contexts {
            if hasher
                .is_green(&[context_token], fixed_token_id)
                .expect("valid inputs")
            {
                green_hits += 1;
            }
        }
        let empirical = f64::from(green_hits) / f64::from(num_contexts);
        // Binomial std error at n=4000 is well under 0.01 for every gamma
        // tested here; 0.05 gives a very comfortable margin against flakes
        // while still catching a badly biased hash.
        assert!(
            (empirical - gamma).abs() < 0.05,
            "gamma={gamma}: empirical green rate {empirical} too far from gamma"
        );
    }
}

// ── stats: erf / normal_cdf against known values ─────────────────────────────

#[test]
fn erf_is_odd_and_zero_at_origin() {
    assert!(erf(0.0).abs() < 1e-9);
    for &x in &[0.3, 1.0, 2.0, 3.5] {
        assert!((erf(-x) + erf(x)).abs() < 1e-9, "erf must be odd: x={x}");
    }
}

#[test]
fn erf_matches_known_values_to_documented_accuracy() {
    // Reference values (to 1e-9) from standard tables / high-precision erf.
    let cases = [
        (1.0_f64, 0.842_700_792_949_714_9),
        (2.0_f64, 0.995_322_265_018_952_7),
        (0.5_f64, 0.520_499_877_813_046_5),
    ];
    for (x, expected) in cases {
        let got = erf(x);
        assert!(
            (got - expected).abs() < 2e-7,
            "erf({x}) = {got}, expected ~{expected}"
        );
    }
}

#[test]
fn normal_cdf_matches_textbook_quantiles() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
    assert!((normal_cdf(1.645) - 0.95).abs() < 1e-3);
    assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
    assert!((normal_cdf(2.326) - 0.99).abs() < 1e-3);
    assert!((normal_cdf(2.576) - 0.995).abs() < 1e-3);
    // Symmetry: Phi(-x) == 1 - Phi(x).
    assert!((normal_cdf(-1.96) - (1.0 - normal_cdf(1.96))).abs() < 1e-9);
}

// ── stats: z-score hand-computed exact agreement ─────────────────────────────

#[test]
fn z_score_hand_computed_exact_cases() {
    // expected = gamma*T = 50, variance = T*gamma*(1-gamma) = 25, sqrt = 5.
    assert_eq!(z_score(60, 100, 0.5), 2.0);
    assert_eq!(z_score(40, 100, 0.5), -2.0);
    // expected = 25, observed = 25 -> exactly on the null mean.
    assert_eq!(z_score(25, 100, 0.25), 0.0);
}

#[test]
fn z_score_degenerate_gamma_extremes_are_zero() {
    // gamma=0 or gamma=1 has zero null variance: no statistical test is
    // possible, so this must be a defined 0.0, never NaN/inf.
    assert_eq!(z_score(0, 100, 0.0), 0.0);
    assert_eq!(z_score(100, 100, 1.0), 0.0);
    // Even a "surprising" count (e.g. green_count=100 when gamma=0, which
    // cannot actually arise from this module's own hasher but must still be
    // handled honestly by the formula alone) does not produce NaN/inf.
    let z = z_score(100, 100, 0.0);
    assert!(z.is_finite());
}

#[test]
fn p_value_from_z_basic_properties() {
    assert!((p_value_from_z(0.0) - 0.5).abs() < 1e-6);
    assert!(p_value_from_z(10.0) < 1e-10);
    assert!((p_value_from_z(-10.0) - 1.0).abs() < 1e-9);
    // Always clamped into a valid probability range.
    for &z in &[-100.0, -1.0, 0.0, 1.0, 100.0] {
        let p = p_value_from_z(z);
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn z_score_and_p_value_are_consistent() {
    let (z, p) = z_score_and_p_value(60, 100, 0.5);
    assert_eq!(z, 2.0);
    assert!((p - p_value_from_z(2.0)).abs() < 1e-12);
}

// ── WatermarkGenerator: soft/hard biasing mechanics ───────────────────────────

#[test]
fn generator_soft_adds_delta_only_to_green_logits() {
    let cfg = WatermarkConfig::new(20, 999)
        .with_gamma(0.5)
        .with_delta(3.0);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    let generator =
        WatermarkGenerator::new(cfg.clone(), WatermarkMode::Soft).expect("valid config");

    let context = [4u32];
    let mask = hasher.green_mask(&context).expect("valid context");
    let original = vec![1.0_f64; 20];
    let mut biased = original.clone();
    generator
        .bias_logits(&mut biased, &context)
        .expect("valid inputs");

    for (idx, &is_green) in mask.iter().enumerate() {
        if is_green {
            assert!((biased[idx] - (original[idx] + 3.0)).abs() < 1e-12);
        } else {
            assert_eq!(biased[idx], original[idx]);
        }
    }
}

#[test]
fn generator_hard_forces_red_logits_to_neg_infinity() {
    let cfg = WatermarkConfig::new(20, 111).with_gamma(0.3);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Hard).expect("valid config");

    let context = [2u32];
    let mask = hasher.green_mask(&context).expect("valid context");
    let mut logits = vec![0.5_f64; 20];
    generator
        .bias_logits(&mut logits, &context)
        .expect("valid inputs");

    for (idx, &is_green) in mask.iter().enumerate() {
        if is_green {
            assert_eq!(logits[idx], 0.5);
        } else {
            assert_eq!(logits[idx], f64::NEG_INFINITY);
        }
    }
}

#[test]
fn generator_no_bias_before_full_context_available() {
    let cfg = WatermarkConfig::new(10, 1).with_context_width(3);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");
    let original = vec![0.0_f64, 1.0, 2.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    for context_len in 0..3 {
        let context: Vec<WatermarkTokenId> = (0..context_len as u32).collect();
        let mut logits = original.clone();
        generator
            .bias_logits(&mut logits, &context)
            .expect("valid inputs");
        assert_eq!(logits, original, "context_len={context_len} must not bias");
    }
}

#[test]
fn generator_rejects_wrong_logits_length() {
    let cfg = WatermarkConfig::new(10, 1);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");
    let mut logits = vec![0.0_f64; 5];
    assert!(matches!(
        generator.bias_logits(&mut logits, &[0]),
        Err(WatermarkError::LogitsVocabMismatch { .. })
    ));
}

#[test]
fn generator_hard_mode_empty_green_list_errors() {
    let cfg = WatermarkConfig::new(10, 1).with_gamma(0.0);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Hard).expect("valid config");
    let mut logits = vec![0.0_f64; 10];
    assert!(matches!(
        generator.bias_logits(&mut logits, &[0]),
        Err(WatermarkError::EmptyGreenList { .. })
    ));
}

#[test]
fn generator_soft_preserves_forced_low_entropy_tokens() {
    // The core quality-preservation claim: when one token's logit exceeds
    // every other by more than delta, biasing cannot change the arg-max,
    // whether or not that token happens to be green.
    let vocab_size = 30;
    let delta = 2.0;
    let cfg = WatermarkConfig::new(vocab_size, 4242)
        .with_gamma(0.4)
        .with_delta(delta);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");

    for forced_token in [0u32, 5, 29] {
        for context_seed in [1u32, 2, 3, 4, 5] {
            let mut logits = vec![0.0_f64; vocab_size];
            logits[forced_token as usize] = 100.0; // margin (100) >> delta (2.0)
            let context = [context_seed];
            let chosen = generator
                .generate_token(&logits, &context)
                .expect("valid inputs");
            assert_eq!(
                chosen, forced_token,
                "forced token must survive watermarking regardless of its green/red status"
            );
        }
    }
}

#[test]
fn generator_sequence_is_deterministic_and_correct_length() {
    let cfg = WatermarkConfig::new(16, 77).with_gamma(0.5).with_delta(2.0);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");
    let steps: Vec<Vec<f64>> = (0..40).map(|s| synthetic_logits(s, 16, 555)).collect();

    let seq_a = generator.generate_sequence(&steps).expect("valid steps");
    let seq_b = generator.generate_sequence(&steps).expect("valid steps");
    assert_eq!(seq_a.len(), 40);
    assert_eq!(seq_a, seq_b);
}

// ── WatermarkDetector: basic mechanics and errors ─────────────────────────────

#[test]
fn detector_empty_text_errors() {
    let detector = WatermarkDetector::new(WatermarkConfig::new(10, 1)).expect("valid config");
    assert!(matches!(
        detector.detect(&[]),
        Err(WatermarkError::EmptyText)
    ));
}

#[test]
fn detector_text_shorter_than_context_width_errors() {
    let cfg = WatermarkConfig::new(10, 1).with_context_width(3);
    let detector = WatermarkDetector::new(cfg).expect("valid config");

    // len < h
    match detector.detect(&[1, 2]) {
        Err(WatermarkError::InsufficientContext { required, actual }) => {
            assert_eq!(required, 4);
            assert_eq!(actual, 2);
        }
        other => panic!("expected InsufficientContext, got {other:?}"),
    }
    // len == h exactly: still zero scoreable tokens.
    match detector.detect(&[1, 2, 3]) {
        Err(WatermarkError::InsufficientContext { required, actual }) => {
            assert_eq!(required, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("expected InsufficientContext, got {other:?}"),
    }
    // len == h + 1: exactly one scoreable token, must succeed.
    assert!(detector.detect(&[1, 2, 3, 4]).is_ok());
}

#[test]
fn detector_rejects_out_of_range_token() {
    let detector = WatermarkDetector::new(WatermarkConfig::new(5, 1)).expect("valid config");
    assert!(matches!(
        detector.detect(&[0, 1, 5]),
        Err(WatermarkError::TokenOutOfRange { .. })
    ));
}

#[test]
fn detector_is_deterministic() {
    let cfg = WatermarkConfig::new(32, 88).with_gamma(0.5).with_delta(3.0);
    let generator =
        WatermarkGenerator::new(cfg.clone(), WatermarkMode::Soft).expect("valid config");
    let steps: Vec<Vec<f64>> = (0..60).map(|s| synthetic_logits(s, 32, 21)).collect();
    let tokens = generator.generate_sequence(&steps).expect("valid steps");

    let detector = WatermarkDetector::new(cfg).expect("valid config");
    let a = detector.detect(&tokens).expect("valid tokens");
    let b = detector.detect(&tokens).expect("valid tokens");
    assert_eq!(a, b);
}

// ── Headline test: the statistics actually work ───────────────────────────────

#[test]
fn headline_true_positive_watermarked_text_is_detected() {
    let vocab_size = 64;
    let cfg = WatermarkConfig::new(vocab_size, 0xF00D_CAFE)
        .with_gamma(0.25)
        .with_delta(4.0)
        .with_context_width(1);
    let generator =
        WatermarkGenerator::new(cfg.clone(), WatermarkMode::Soft).expect("valid config");
    let detector = WatermarkDetector::new(cfg).expect("valid config");

    let steps: Vec<Vec<f64>> = (0..400)
        .map(|s| synthetic_logits(s, vocab_size, 0xA11CE))
        .collect();
    let tokens = generator.generate_sequence(&steps).expect("valid steps");

    let detection = detector.detect(&tokens).expect("enough tokens to score");
    eprintln!(
        "[true_positive] green={}/{} z={:.3} p={:.3e}",
        detection.green_count, detection.total_scored, detection.z_score, detection.p_value
    );

    // Watermarked text over a high-entropy synthetic distribution should
    // land the detector far into "watermarked" territory.
    assert!(
        detection.z_score > 8.0,
        "expected a strong z-score, got {}",
        detection.z_score
    );
    assert!(detection.p_value < 1e-10);
    assert!(detection.is_watermarked);
    assert!(detection.green_fraction() > 0.35);
}

#[test]
fn headline_true_negative_unwatermarked_text_is_not_detected() {
    let vocab_size = 64;
    let cfg = WatermarkConfig::new(vocab_size, 0xF00D_CAFE)
        .with_gamma(0.25)
        .with_context_width(1);
    let detector = WatermarkDetector::new(cfg).expect("valid config");

    let steps: Vec<Vec<f64>> = (0..400)
        .map(|s| synthetic_logits(s, vocab_size, 0xB0B))
        .collect();
    let tokens = unbiased_argmax_sequence(&steps);

    let detection = detector.detect(&tokens).expect("enough tokens to score");
    eprintln!(
        "[true_negative] green={}/{} z={:.3} p={:.3}",
        detection.green_count, detection.total_scored, detection.z_score, detection.p_value
    );

    // A single non-watermarked draw should almost never exceed the default
    // threshold: P(|Z| > 4) ~ 6e-5 under the (approximately) standard-normal
    // null, so this is not a flaky bound.
    assert!(
        detection.z_score.abs() < 4.0,
        "unwatermarked text scored a suspiciously large |z|={}",
        detection.z_score
    );
    assert!(!detection.is_watermarked);
}

#[test]
fn headline_false_positive_rate_is_calibrated_to_the_normal_tail() {
    // The real test: run many *independent* non-watermarked sequences and
    // check that the empirical rate of exceeding a threshold matches the
    // theoretical normal-tail rate `1 - Phi(threshold)`. A detector that
    // over- or under-flags would pass the true-positive/negative tests
    // above but fail here.
    let vocab_size = 64; // gamma*vocab_size is an exact integer for gamma=0.25 -> no rounding bias
    let num_trials = 3000u64;
    let tokens_per_trial = 61; // context_width=1 -> 60 scored tokens/trial
    let cfg = WatermarkConfig::new(vocab_size, 0x5EED_5EED)
        .with_gamma(0.25)
        .with_context_width(1);
    let detector = WatermarkDetector::new(cfg).expect("valid config");

    let thresholds = [1.0_f64, 1.645, 2.0];
    let mut exceed_counts = [0u64; 3];
    let mut high_threshold_exceed_count = 0u64; // z > 4.0, theoretical ~3.2e-5

    let mut trial_seed_rng = SplitMix64::new(0xABCD_1234_5678_9EF0);
    for _ in 0..num_trials {
        let trial_seed = trial_seed_rng.next_u64();
        let steps: Vec<Vec<f64>> = (0..tokens_per_trial)
            .map(|s| synthetic_logits(s, vocab_size, trial_seed))
            .collect();
        let tokens = unbiased_argmax_sequence(&steps);
        let detection = detector.detect(&tokens).expect("enough tokens to score");

        for (slot, &threshold) in thresholds.iter().enumerate() {
            if detection.z_score >= threshold {
                exceed_counts[slot] += 1;
            }
        }
        if detection.z_score > 4.0 {
            high_threshold_exceed_count += 1;
        }
    }

    eprintln!("[fpr_calibration] trials={num_trials}");
    for (&threshold, &count) in thresholds.iter().zip(exceed_counts.iter()) {
        let empirical = count as f64 / num_trials as f64;
        let theoretical = 1.0 - normal_cdf(threshold);
        eprintln!(
            "  z>={threshold}: empirical={empirical:.4} theoretical={theoretical:.4} (n={count})"
        );
        // n=3000 gives a binomial std error well under 0.01 at every one of
        // these thresholds; 0.04 is a very generous margin that still
        // catches a genuinely broken statistic (e.g. one off by a constant
        // factor, or the wrong tail).
        assert!(
            (empirical - theoretical).abs() < 0.04,
            "z>={threshold}: empirical FPR {empirical} too far from theoretical {theoretical}"
        );
    }

    let high_empirical = high_threshold_exceed_count as f64 / num_trials as f64;
    let high_theoretical = 1.0 - normal_cdf(4.0);
    eprintln!(
        "  z>4.0: empirical={high_empirical:.6} theoretical={high_theoretical:.6} (n={high_threshold_exceed_count})"
    );
    // Expected count at n=3000 is ~0.095; requiring <=3 hits keeps the
    // false-flake probability astronomically small (Poisson(0.095)) while
    // still verifying the tail really is thin, not just "small-ish".
    assert!(
        high_threshold_exceed_count <= 3,
        "z>4.0 fired {high_threshold_exceed_count} times out of {num_trials}, expected ~0"
    );
}

// ── Robustness / attack: graceful degradation under edits ────────────────────

#[test]
fn robustness_z_score_degrades_gracefully_under_substitution_edits() {
    let vocab_size = 64;
    let steps_len = 260;
    let delta = 5.0;
    let logit_seed = 0x1357_9BDF;
    let edit_seed = 0x2468_ACE0;

    let cfg_h1 = WatermarkConfig::new(vocab_size, 0x900D_5EED)
        .with_gamma(0.25)
        .with_delta(delta)
        .with_context_width(1);
    let steps: Vec<Vec<f64>> = (0..steps_len)
        .map(|s| synthetic_logits(s, vocab_size, logit_seed))
        .collect();

    let generator_h1 =
        WatermarkGenerator::new(cfg_h1.clone(), WatermarkMode::Soft).expect("valid config");
    let detector_h1 = WatermarkDetector::new(cfg_h1).expect("valid config");
    let sequence_h1 = generator_h1.generate_sequence(&steps).expect("valid steps");
    let baseline_z1 = detector_h1
        .detect(&sequence_h1)
        .expect("enough tokens to score")
        .z_score;

    assert!(
        baseline_z1 > 4.0,
        "baseline watermark (h=1) too weak to test: z={baseline_z1}"
    );

    let edit_fractions = [0.0_f64, 0.02, 0.05, 0.1, 0.2, 0.3];
    let mut curve_h1 = Vec::with_capacity(edit_fractions.len());
    for &fraction in &edit_fractions {
        let edited = apply_edits(&sequence_h1, fraction, vocab_size, edit_seed);
        let z = detector_h1
            .detect(&edited)
            .expect("enough tokens to score")
            .z_score;
        curve_h1.push((fraction, z));
    }

    eprintln!("[robustness h=1] baseline_z={baseline_z1:.3}");
    for (fraction, z) in &curve_h1 {
        eprintln!("  edit_fraction={fraction:.2} z={z:.3}");
    }

    // No edits at all reproduces the baseline exactly.
    assert_eq!(curve_h1[0], (0.0, baseline_z1));

    // Graceful, not a cliff: a small edit fraction should retain a
    // substantial majority of the signal.
    let z_at_5pct = curve_h1[2].1;
    assert!(
        z_at_5pct > 0.5 * baseline_z1,
        "5% edits collapsed the signal: baseline={baseline_z1} at_5pct={z_at_5pct}"
    );

    // But real degradation: the largest edit fraction tested is clearly
    // lower than the baseline.
    let z_at_30pct = curve_h1.last().expect("non-empty").1;
    assert!(
        z_at_30pct < baseline_z1,
        "30% edits did not degrade the score at all: baseline={baseline_z1} at_30pct={z_at_30pct}"
    );
}

#[test]
fn robustness_wider_context_is_more_fragile_to_edits() {
    // Same synthetic "document" (per-step logits), watermarked once with
    // h=1 and once with h=3, then hit with the *same* relative edit pattern
    // at several fractions. Average the relative signal drop across
    // fractions and documents for a low-noise comparison: h=3 must degrade
    // proportionally faster, because each edit corrupts up to h+1 scored
    // positions instead of up to 2.
    let vocab_size = 64;
    let steps_len = 260;
    let delta = 5.0;
    let edit_fractions = [0.05_f64, 0.1, 0.15, 0.2, 0.3];
    let document_seeds = [0x1357_9BDFu64, 0x2222_3333, 0x777A_BCDE];

    let mut relative_drops_h1 = Vec::new();
    let mut relative_drops_h3 = Vec::new();

    for (doc_idx, &logit_seed) in document_seeds.iter().enumerate() {
        let edit_seed = 0xE001_0000 ^ (doc_idx as u64);
        let steps: Vec<Vec<f64>> = (0..steps_len)
            .map(|s| synthetic_logits(s, vocab_size, logit_seed))
            .collect();

        let cfg_h1 = WatermarkConfig::new(vocab_size, 0xAAAA_BBBB)
            .with_gamma(0.25)
            .with_delta(delta)
            .with_context_width(1);
        let cfg_h3 = cfg_h1.clone().with_context_width(3);

        let gen1 =
            WatermarkGenerator::new(cfg_h1.clone(), WatermarkMode::Soft).expect("valid config");
        let gen3 =
            WatermarkGenerator::new(cfg_h3.clone(), WatermarkMode::Soft).expect("valid config");
        let det1 = WatermarkDetector::new(cfg_h1).expect("valid config");
        let det3 = WatermarkDetector::new(cfg_h3).expect("valid config");

        let seq1 = gen1.generate_sequence(&steps).expect("valid steps");
        let seq3 = gen3.generate_sequence(&steps).expect("valid steps");

        let baseline_z1 = det1.detect(&seq1).expect("enough tokens").z_score;
        let baseline_z3 = det3.detect(&seq3).expect("enough tokens").z_score;
        assert!(
            baseline_z1 > 4.0 && baseline_z3 > 4.0,
            "baseline too weak to compare"
        );

        for &fraction in &edit_fractions {
            let edited1 = apply_edits(&seq1, fraction, vocab_size, edit_seed);
            let edited3 = apply_edits(&seq3, fraction, vocab_size, edit_seed);
            let z1 = det1.detect(&edited1).expect("enough tokens").z_score;
            let z3 = det3.detect(&edited3).expect("enough tokens").z_score;

            let drop1 = (baseline_z1 - z1) / baseline_z1;
            let drop3 = (baseline_z3 - z3) / baseline_z3;
            relative_drops_h1.push(drop1);
            relative_drops_h3.push(drop3);
        }
    }

    let avg_drop1 = relative_drops_h1.iter().sum::<f64>() / relative_drops_h1.len() as f64;
    let avg_drop3 = relative_drops_h3.iter().sum::<f64>() / relative_drops_h3.len() as f64;

    eprintln!(
        "[robustness h1 vs h3] avg relative drop h=1: {avg_drop1:.4}, h=3: {avg_drop3:.4} \
         (averaged over {} document x fraction combinations)",
        relative_drops_h1.len()
    );

    assert!(
        avg_drop3 > avg_drop1,
        "expected h=3 to degrade proportionally faster than h=1: avg_drop1={avg_drop1} avg_drop3={avg_drop3}"
    );
}

// ── Edge cases ─────────────────────────────────────────────────────────────────

#[test]
fn edge_case_gamma_zero_soft_mode_is_unbiased_generation() {
    let vocab_size = 20;
    let cfg = WatermarkConfig::new(vocab_size, 1)
        .with_gamma(0.0)
        .with_delta(9.0);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    assert_eq!(hasher.green_size(), 0);

    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");
    let logits = synthetic_logits(0, vocab_size, 1);
    let mut biased = logits.clone();
    generator
        .bias_logits(&mut biased, &[0])
        .expect("valid inputs");
    assert_eq!(biased, logits, "gamma=0 has no green tokens to bias");
}

#[test]
fn edge_case_gamma_zero_hard_mode_errors() {
    let cfg = WatermarkConfig::new(20, 1).with_gamma(0.0);
    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Hard).expect("valid config");
    let mut logits = vec![0.0_f64; 20];
    assert!(matches!(
        generator.bias_logits(&mut logits, &[0]),
        Err(WatermarkError::EmptyGreenList { .. })
    ));
}

#[test]
fn edge_case_gamma_zero_detection_has_no_signal() {
    let cfg = WatermarkConfig::new(20, 1).with_gamma(0.0);
    let detector = WatermarkDetector::new(cfg).expect("valid config");
    let detection = detector.detect(&[0, 1, 2, 3, 4, 5]).expect("enough tokens");
    assert_eq!(detection.green_count, 0);
    assert_eq!(detection.z_score, 0.0);
    assert_eq!(detection.p_value, 1.0);
    assert!(!detection.is_watermarked);
}

#[test]
fn edge_case_gamma_one_soft_mode_does_not_change_argmax() {
    // Adding delta to *every* logit is a uniform shift: the arg-max is
    // unchanged from the fully-unbiased decision.
    let vocab_size = 20;
    let cfg = WatermarkConfig::new(vocab_size, 1)
        .with_gamma(1.0)
        .with_delta(50.0);
    let hasher = WatermarkHasher::new(&cfg).expect("valid config");
    assert_eq!(hasher.green_size(), vocab_size);

    let generator = WatermarkGenerator::new(cfg, WatermarkMode::Soft).expect("valid config");
    let logits = synthetic_logits(3, vocab_size, 77);
    let unbiased_choice = unbiased_argmax_sequence(std::slice::from_ref(&logits))[0];
    let watermarked_choice = generator
        .generate_token(&logits, &[0])
        .expect("valid inputs");
    assert_eq!(unbiased_choice, watermarked_choice);
}

#[test]
fn edge_case_gamma_one_detection_has_no_signal() {
    // Every token is unconditionally green (green_size == vocab_size), so
    // observing "all green" carries zero evidence either way: z must be the
    // defined degenerate 0.0, not a spuriously huge value.
    let cfg = WatermarkConfig::new(20, 1).with_gamma(1.0);
    let detector = WatermarkDetector::new(cfg).expect("valid config");
    let detection = detector
        .detect(&[0, 5, 19, 3, 3, 3])
        .expect("enough tokens");
    assert_eq!(detection.green_count, detection.total_scored);
    assert_eq!(detection.z_score, 0.0);
    assert_eq!(detection.p_value, 1.0);
    assert!(!detection.is_watermarked);
}

#[test]
fn edge_case_vocab_size_one_never_panics() {
    for &gamma in &[0.0, 0.4, 0.5, 0.6, 1.0] {
        let cfg = WatermarkConfig::new(1, 1).with_gamma(gamma);
        let generator =
            WatermarkGenerator::new(cfg.clone(), WatermarkMode::Soft).expect("valid config");
        let token = generator
            .generate_token(&[0.0], &[0])
            .expect("only one token exists");
        assert_eq!(token, 0);

        let detector = WatermarkDetector::new(cfg).expect("valid config");
        let detection = detector.detect(&[0, 0, 0, 0]).expect("enough tokens");
        assert!(detection.z_score.is_finite());
        assert!((0.0..=1.0).contains(&detection.p_value));
    }
}

#[test]
fn edge_case_empty_text_is_a_distinct_error_from_insufficient_context() {
    let detector = WatermarkDetector::new(WatermarkConfig::new(10, 1)).expect("valid config");
    assert!(matches!(
        detector.detect(&[]),
        Err(WatermarkError::EmptyText)
    ));
    assert!(matches!(
        detector.detect(&[0]),
        Err(WatermarkError::InsufficientContext { .. })
    ));
}
