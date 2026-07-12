//! [`SplitMix64Rng`] — the module's deterministic, dependency-free pseudo-random
//! number generator.
//!
//! Thompson sampling and ε-greedy are *randomized* policies: their behaviour is
//! only defined relative to a source of randomness. This crate takes no
//! dependency on `rand` / `rand_distr` (see the project's `SciRS2` policy), and
//! the existing [`ab_eval`](crate::ab_eval) module already establishes the
//! convention that stochastic procedures here must be *deterministic given their
//! inputs* (it derives its bootstrap resample indices from an FNV-1a hash rather
//! than from an RNG). This module follows the same spirit but needs a genuine
//! *stream* of independent variates rather than a hash of a fixed index, so it
//! ships a small, well-understood generator instead.
//!
//! # `SplitMix64`
//!
//! `SplitMix64` (Steele, Lea & Flood, 2014, "Fast Splittable Pseudorandom Number
//! Generators"; also the standard seeding routine for the xoshiro/xoroshiro
//! family) is a *fixed-increment* Weyl sequence followed by an MurmurHash3-style
//! avalanche finalizer:
//!
//! ```text
//! state += 0x9E37_79B9_7F4A_7C15          (the 64-bit golden-ratio odd constant)
//! z  = state
//! z  = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9
//! z  = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB
//! z ^= z >> 31
//! ```
//!
//! All arithmetic is `wrapping` (mod 2^64). Two properties matter here:
//!
//! * **Full period.** The increment is odd, so `state` walks every one of the
//!   2^64 residues before repeating; the finalizer is a *bijection* on `u64`,
//!   so the output sequence is a permutation of the state sequence and inherits
//!   the full 2^64 period. There is no "bad seed" — every seed, including `0`,
//!   produces a full-period stream.
//! * **Statistical quality.** The two multiply-xorshift rounds avalanche the
//!   low-entropy increments; `SplitMix64` passes `BigCrush`. That is far more than
//!   this module needs (uniform arm choices, Fisher–Yates shuffles, and
//!   standard normals for posterior sampling), and it is ~5 lines of code.
//!
//! # Reproducibility contract
//!
//! Everything downstream of a [`SplitMix64Rng`] is a pure function of the seed
//! and the call sequence: the same seed replayed against the same contexts and
//! rewards yields a byte-identical action sequence. This is what makes the
//! module's determinism test meaningful, and it is what lets an operator
//! *reproduce* a production exploration trajectory from a logged seed.
//!
//! One subtlety worth stating explicitly: [`SplitMix64Rng::next_standard_normal`]
//! generates normals in *pairs* (Box–Muller emits two independent variates from
//! two uniforms) and caches the second. The normal stream is therefore
//! reproducible but its alignment with the underlying `u64` stream depends on
//! the *parity* of the calls made so far. Interleaving `next_standard_normal`
//! with `next_f64` still gives a deterministic overall stream, just not the same
//! one you would get by drawing the normals up front.

// The RNG's whole job is bit-twiddling `u64`s into `f64`s and `usize`s. The
// casts below are all deliberate and individually justified in comments; the
// `u64 -> f64` mantissa cast is *exactly* representable by construction (the
// value is < 2^53), and the `u64 -> usize` cast is guarded by a modulus that is
// itself a `usize`.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::f64::consts::TAU;

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First avalanche multiplier of the MurmurHash3-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second avalanche multiplier of the MurmurHash3-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// `2^-53` — the spacing of the `f64` grid this generator samples on.
///
/// A `f64` has a 53-bit significand, so the 2^53 values `k * 2^-53` for
/// `k in [0, 2^53)` are *all* exactly representable. Taking the top 53 bits of a
/// `SplitMix64` output as `k` therefore yields a uniform draw on that grid with no
/// rounding bias at all.
const F64_ULP: f64 = 1.0 / 9_007_199_254_740_992.0; // 1 / 2^53

/// A deterministic `SplitMix64` pseudo-random number generator.
///
/// Seeded once, it reproduces its entire stream exactly. See the
/// [module documentation](self) for the algorithm, its period, and the
/// reproducibility contract.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "bandit-ranker")]
/// # {
/// use oxirag::bandit_ranker::SplitMix64Rng;
///
/// let mut a = SplitMix64Rng::new(42);
/// let mut b = SplitMix64Rng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// // Uniforms land in [0, 1).
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SplitMix64Rng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
    /// The second variate of the most recent Box–Muller pair, kept until it is
    /// consumed by the next call to [`SplitMix64Rng::next_standard_normal`].
    cached_normal: Option<f64>,
}

impl SplitMix64Rng {
    /// Create a generator from `seed`.
    ///
    /// Every seed is equally good — `SplitMix64` has no degenerate states (see the
    /// [module documentation](self)), so `0` is a perfectly ordinary seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    /// Draw the next uniformly distributed `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
        z ^ (z >> 31)
    }

    /// Draw a uniform `f64` in the **half-open** interval `[0, 1)`.
    ///
    /// Implemented by taking the top 53 bits of a `u64` draw (the high bits are
    /// the best-avalanched ones) and scaling by `2^-53`, so the result is a
    /// uniform draw over the 2^53 exactly-representable multiples of `2^-53` in
    /// `[0, 1)`. `0.0` *is* attainable; `1.0` is not.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_ULP
    }

    /// Draw a uniform `f64` in the **open** interval `(0, 1)`.
    ///
    /// Box–Muller evaluates `ln(u)`, which is `-inf` at `u == 0`. Rather than
    /// rejection-sampling the (astronomically rare, but not impossible —
    /// `2^-53`) zero, this shifts the 53-bit grid by half an ulp:
    /// `(k + 0.5) * 2^-53` for `k in [0, 2^53)`, which is bounded strictly
    /// inside `(0, 1)` and is still exactly uniform on its grid. That makes
    /// `next_standard_normal` *total* — it cannot produce a non-finite value.
    pub fn next_f64_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * F64_ULP
    }

    /// Draw a standard normal variate, `z ~ N(0, 1)`, by the **Box–Muller**
    /// transform.
    ///
    /// Given two independent uniforms `u1, u2 ~ U(0, 1)`,
    ///
    /// ```text
    /// r  = sqrt(-2 * ln(u1))
    /// t  = 2 * pi * u2
    /// z0 = r * cos(t)
    /// z1 = r * sin(t)
    /// ```
    ///
    /// yields **two** independent standard normals. (The polar coordinates
    /// `(r, t)` of a standard bivariate normal are independent, with `t` uniform
    /// on `[0, 2pi)` and `r^2 ~ Exp(1/2)`; `-2 ln(u1)` is exactly an `Exp(1/2)`
    /// draw by inverse-transform sampling.)
    ///
    /// Both variates are used: `z1` is cached and returned by the next call, so
    /// the generator costs one `ln`/`sin`/`cos` triple per *two* normals rather
    /// than per normal. Because `u1` comes from [`SplitMix64Rng::next_f64_open`]
    /// it is strictly positive, so `ln(u1)` is finite and the result is always a
    /// finite `f64`.
    pub fn next_standard_normal(&mut self) -> f64 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = TAU * u2;
        self.cached_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }

    /// Draw a uniform `usize` in `[0, bound)`, or `None` when `bound == 0`.
    ///
    /// Uses **rejection sampling** rather than a bare `next_u64() % bound`. The
    /// naive modulo is biased whenever `bound` does not divide `2^64`: the
    /// residues `0 .. (2^64 mod bound)` each have one extra pre-image. Here the
    /// first `2^64 mod bound` outputs are simply rejected and redrawn, which
    /// makes the remaining range an exact multiple of `bound` and the reduction
    /// exactly uniform. The expected number of redraws is below `1` for any
    /// `bound` a bandit would plausibly use (rejection probability is at most
    /// `bound / 2^64`).
    pub fn next_usize_below(&mut self, bound: usize) -> Option<usize> {
        if bound == 0 {
            return None;
        }
        let bound_u64 = bound as u64;
        // `2^64 mod bound`, computed without 128-bit arithmetic: in wrapping
        // `u64` arithmetic, `0u64.wrapping_sub(bound) == 2^64 - bound`, and
        // `(2^64 - bound) mod bound == 2^64 mod bound`.
        let reject_below = bound_u64.wrapping_neg() % bound_u64;
        loop {
            let draw = self.next_u64();
            if draw >= reject_below {
                return Some((draw % bound_u64) as usize);
            }
        }
    }

    /// Shuffle `items` uniformly at random with an in-place **Fisher–Yates**
    /// (Durstenfeld) pass.
    ///
    /// Walks from the back, swapping each position `i` with a uniformly chosen
    /// position in `[0, i]`. Every one of the `n!` permutations is equally
    /// likely (given an unbiased `next_usize_below`, which the rejection
    /// sampling above guarantees). In particular the *first* element of the
    /// result is uniform over all items — which is exactly what ε-greedy's
    /// "explore uniformly over arms" step needs from a *ranking* policy.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            // `i + 1` is at least 2, so `next_usize_below` always yields a draw.
            // Written as an `if let` rather than an `expect` so that this method
            // is *total*: it has no panic path at all, only an unreachable one.
            if let Some(j) = self.next_usize_below(i + 1) {
                items.swap(i, j);
            }
        }
    }
}

impl Default for SplitMix64Rng {
    /// A generator seeded with `0`.
    ///
    /// Deliberately *not* entropy-seeded: a default that silently varied run to
    /// run would break the module's reproducibility contract. Callers who want a
    /// fresh trajectory pass a fresh seed.
    fn default() -> Self {
        Self::new(0)
    }
}
