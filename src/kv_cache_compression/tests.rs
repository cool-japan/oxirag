#![allow(
    // Attention weights and hand-derived softmax values are compared against
    // exactly-computed constants; the comparisons are deliberate and are all
    // either exact-by-construction (a masked weight is exactly 0.0, an
    // untouched cache is bitwise identical) or tolerance-guarded.
    clippy::float_cmp,
    // `cache`/`caches`, `keys`/`key`, `stats`/`state` and similar pairs appear
    // constantly in this domain and renaming them would obscure the maths.
    clippy::similar_names,
    // Token/step/slot counts are `usize` and are divided into `f64` masses.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // Hand-computed attention fixtures name their scalars `q`, `k`, `v`, `d`.
    clippy::many_single_char_names,
    // The headline experiment is one long, linear narrative; splitting it would
    // make the measurement harder to follow, not easier.
    clippy::too_many_lines,
    clippy::unreadable_literal,
    // Fixture builders index parallel `[head][dim]` buffers by hand.
    clippy::needless_range_loop,
    clippy::items_after_statements
)]
//! Tests for attention-score-driven KV-cache eviction.
//!
//! The suite is ordered by dependency, because the later results are only
//! meaningful if the earlier ones hold:
//!
//! 1. **The attention kernel itself** — hand-computed softmax weights and
//!    outputs, row sums, the causal mask, the degenerate all-masked row, and
//!    numerical stability under extreme logits. If the attention is wrong, every
//!    eviction policy built on it is measuring noise, so this is tested first
//!    and hardest.
//! 2. **Score accumulation** — that the accumulated column sums really are the
//!    column sums, and the exact algebraic relation between the two
//!    normalizations that quantifies the early-token bias.
//! 3. **The policies**, individually.
//! 4. **The headline experiment** — fidelity under a budget, measured: H2O and
//!    `SnapKV` versus the recency baseline on a workload whose attention mass
//!    genuinely sits on old tokens.
//! 5. **The `StreamingLLM` sink demonstration** — that dropping the first few
//!    tokens is catastrophic while dropping the same number of middle tokens is
//!    not.
//! 6. Invariants (budget adherence for every policy at every budget) and edge
//!    cases.
//!
//! Sections 1-2 live in [`kernel`], sections 3-6 in [`policies`]. This file holds
//! the lint allowances (which propagate to both submodules) and the two things
//! they share: a deterministic PRNG and the deviation metric.

// ── Deterministic PRNG (SplitMix64) ──────────────────────────────────────────

/// `SplitMix64` — Steele, Lea & Flood (2014). A tiny, fast, well-distributed
/// generator, hand-rolled here so the synthetic tensors are reproducible without
/// pulling in a dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn next_symmetric_unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 11) as f64;
        let unit = mantissa / (1u64 << 53) as f64;
        (unit.mul_add(2.0, -1.0)) as f32
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// Relative L2 deviation `‖candidate - reference‖ / ‖reference‖`.
///
/// This is *the* metric of the headline experiment: how far a compressed cache's
/// attention output moves from the uncompressed one. `1.0` means the output
/// moved by as much as its own magnitude — i.e. it was destroyed.
fn relative_deviation(reference: &[f32], candidate: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "deviation is only defined between equally-shaped outputs"
    );
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    for (&expected, &actual) in reference.iter().zip(candidate.iter()) {
        let difference = f64::from(expected) - f64::from(actual);
        squared_error += difference * difference;
        squared_reference += f64::from(expected) * f64::from(expected);
    }
    if squared_reference == 0.0 {
        return if squared_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (squared_error / squared_reference).sqrt()
}

/// Bit patterns of a float buffer, for genuinely *bitwise* comparison.
fn bit_pattern(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

mod kernel;
mod policies;
