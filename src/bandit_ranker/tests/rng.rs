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
//! Tests for [`SplitMix64Rng`]: reproducibility, uniformity, the Box–Muller
//! normals, and the unbiasedness of the rejection-sampled integer draws.

use crate::bandit_ranker::rng::SplitMix64Rng;

// ═════════════════════════════════════════════════════════════════════════════
// SplitMix64Rng
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn splitmix64_is_reproducible_from_its_seed() {
    let mut first = SplitMix64Rng::new(0xDEAD_BEEF);
    let mut second = SplitMix64Rng::new(0xDEAD_BEEF);
    let a: Vec<u64> = (0..64).map(|_| first.next_u64()).collect();
    let b: Vec<u64> = (0..64).map(|_| second.next_u64()).collect();
    assert_eq!(a, b, "the same seed must reproduce the same stream exactly");
}

#[test]
fn splitmix64_different_seeds_give_different_streams() {
    let mut first = SplitMix64Rng::new(1);
    let mut second = SplitMix64Rng::new(2);
    let a: Vec<u64> = (0..32).map(|_| first.next_u64()).collect();
    let b: Vec<u64> = (0..32).map(|_| second.next_u64()).collect();
    assert_ne!(a, b);
}

#[test]
fn splitmix64_seed_zero_is_not_degenerate() {
    // SplitMix64 has no bad seeds: the increment is odd, so even a zero seed walks
    // the full 2^64 cycle. A generator that returned a constant (or a short cycle)
    // from seed zero would silently destroy every default-seeded bandit.
    let mut rng = SplitMix64Rng::new(0);
    let draws: Vec<u64> = (0..16).map(|_| rng.next_u64()).collect();
    assert!(draws[0] != 0, "seed zero must not emit a zero first draw");
    for window in draws.windows(2) {
        assert_ne!(window[0], window[1], "consecutive draws must differ");
    }
}

#[test]
fn splitmix64_uniform_lies_in_half_open_unit_interval() {
    let mut rng = SplitMix64Rng::new(7);
    let mut sum = 0.0;
    let draws = 200_000;
    for _ in 0..draws {
        let u = rng.next_f64();
        assert!(
            (0.0..1.0).contains(&u),
            "next_f64 must be in [0, 1), got {u}"
        );
        sum += u;
    }
    let mean = sum / f64::from(draws);
    // Mean of U(0,1) is 0.5, sd of the mean is 1/sqrt(12 * 200000) ~ 6.5e-4.
    assert!(
        (mean - 0.5).abs() < 0.01,
        "uniform mean should be ~0.5, got {mean}"
    );
}

#[test]
fn splitmix64_open_uniform_excludes_both_endpoints() {
    let mut rng = SplitMix64Rng::new(99);
    for _ in 0..100_000 {
        let u = rng.next_f64_open();
        assert!(
            u > 0.0,
            "next_f64_open must exclude 0 so that ln(u) is finite"
        );
        assert!(u < 1.0, "next_f64_open must exclude 1");
    }
}

#[test]
fn splitmix64_box_muller_normals_have_the_right_moments() {
    let mut rng = SplitMix64Rng::new(2024);
    let draws = 400_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut sum_cube = 0.0;
    for _ in 0..draws {
        let z = rng.next_standard_normal();
        assert!(
            z.is_finite(),
            "Box-Muller must never emit a non-finite value"
        );
        sum += z;
        sum_sq += z * z;
        sum_cube += z * z * z;
    }
    let n = f64::from(draws);
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;
    let skew = sum_cube / n;

    // sd of the sample mean is 1/sqrt(400000) ~ 1.6e-3.
    assert!(mean.abs() < 0.01, "normal mean should be ~0, got {mean}");
    assert!(
        (variance - 1.0).abs() < 0.02,
        "normal variance should be ~1, got {variance}"
    );
    assert!(
        skew.abs() < 0.03,
        "the normal distribution is symmetric, so its third moment should be ~0, got {skew}"
    );
}

#[test]
fn splitmix64_usize_below_is_bounded_and_roughly_uniform() {
    let mut rng = SplitMix64Rng::new(5);
    assert_eq!(rng.next_usize_below(0), None, "a zero bound has no draw");
    assert_eq!(
        rng.next_usize_below(1),
        Some(0),
        "a unit bound has exactly one draw"
    );

    let bound = 7;
    let draws = 140_000;
    let mut counts = vec![0_u32; bound];
    for _ in 0..draws {
        let value = rng.next_usize_below(bound).expect("bound is non-zero");
        assert!(value < bound);
        counts[value] += 1;
    }
    // Expect 20000 each; sd ~ sqrt(140000 * (1/7) * (6/7)) ~ 131. A 10% band is
    // ~15 sd wide, so this catches modulo bias without being flaky.
    for count in counts {
        let ratio = f64::from(count) / f64::from(draws);
        assert!(
            (ratio - 1.0 / 7.0).abs() < 0.01,
            "rejection sampling should be unbiased, got a bucket at ratio {ratio}"
        );
    }
}

#[test]
fn splitmix64_shuffle_puts_every_item_first_equally_often() {
    // The first element of a Fisher-Yates shuffle is what epsilon-greedy's uniform
    // exploration actually uses, so its marginal must be uniform.
    let mut rng = SplitMix64Rng::new(31337);
    let trials = 120_000;
    let mut first_counts = [0_u32; 4];
    for _ in 0..trials {
        let mut items = [0_usize, 1, 2, 3];
        rng.shuffle(&mut items);
        first_counts[items[0]] += 1;
        let mut seen = items;
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3], "a shuffle must be a permutation");
    }
    for count in first_counts {
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (ratio - 0.25).abs() < 0.01,
            "each item should lead a quarter of the time, got {ratio}"
        );
    }
}

#[test]
fn splitmix64_shuffle_of_tiny_slices_is_a_noop() {
    let mut rng = SplitMix64Rng::new(1);
    let mut empty: [u8; 0] = [];
    rng.shuffle(&mut empty);
    let mut single = [42_u8];
    rng.shuffle(&mut single);
    assert_eq!(single, [42]);
}
