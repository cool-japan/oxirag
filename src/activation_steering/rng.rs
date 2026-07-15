//! [`SteeringRng`] — the module's deterministic, dependency-free pseudo-random
//! number generator.
//!
//! Fitting a probe requires a **train/validation split**, and a split is only
//! meaningful if it is *random* — a split that took the first 80% of the pairs
//! would inherit whatever ordering the caller's dataset happened to have, and a
//! validation accuracy measured against it would say more about that ordering
//! than about the probe. This crate takes no dependency on `rand` / `rand_distr`
//! (see the project's `SciRS2` policy), so the module ships its own generator.
//!
//! # `SplitMix64`
//!
//! `SplitMix64` (Steele, Lea & Flood, 2014, *Fast Splittable Pseudorandom Number
//! Generators*) is a fixed-increment Weyl sequence followed by a
//! `MurmurHash3`-style avalanche finalizer:
//!
//! ```text
//! state += 0x9E37_79B9_7F4A_7C15          (the 64-bit golden-ratio odd constant)
//! z  = state
//! z  = (z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9
//! z  = (z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB
//! z ^= z >> 31
//! ```
//!
//! All arithmetic is `wrapping` (mod 2^64). The increment is odd, so the state
//! walks every one of the 2^64 residues before repeating, and the finalizer is a
//! *bijection* on `u64`, so the output stream inherits the full period. There is
//! no bad seed — `0` is as good as any other.
//!
//! # Why this is not `bandit_ranker::SplitMix64Rng`
//!
//! It is the same algorithm, and that duplication is deliberate rather than
//! careless. `bandit_ranker` sits behind its own Cargo feature; this module's
//! feature (`activation-steering`) depends only on `hidden-states`. Reaching
//! across to `bandit_ranker` would silently widen this module's feature
//! requirements to import ~50 lines of bit-twiddling. The generator is stated
//! here instead, and the two are free to diverge (this one, for instance, needs
//! no Box–Muller normals in `src/` — only the tests draw them).
//!
//! # Reproducibility contract
//!
//! Everything downstream of a [`SteeringRng`] is a pure function of the seed and
//! the call sequence. The same [`SteeringConfig::seed`](super::SteeringConfig::seed)
//! replayed against the same contrastive pairs yields the *same* train/validation
//! split, hence the same probes, the same head ranking, and the same steering
//! vectors — which is what makes a fitted [`ActivationSteering`](super::ActivationSteering)
//! an auditable artifact rather than a lucky draw.

// The generator's whole job is bit-twiddling `u64`s into `f64`s and `usize`s.
// The `u64 -> f64` mantissa cast is *exactly* representable by construction (the
// value is < 2^53), and the `u64 -> usize` cast is guarded by a modulus that is
// itself a `usize`.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::f64::consts::TAU;

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// `2^-53` — the spacing of the `f64` grid this generator samples on.
///
/// An `f64` has a 53-bit significand, so the 2^53 values `k * 2^-53` for
/// `k in [0, 2^53)` are all exactly representable. Taking the top 53 bits of a
/// `SplitMix64` output as `k` yields a uniform draw on that grid with no rounding
/// bias at all.
const F64_ULP: f64 = 1.0 / 9_007_199_254_740_992.0; // 1 / 2^53

/// A deterministic `SplitMix64` pseudo-random number generator.
///
/// Seeded once, it reproduces its entire stream exactly. See the
/// [module documentation](self) for the algorithm and the reproducibility
/// contract.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "activation-steering")]
/// # {
/// use oxirag::activation_steering::SteeringRng;
///
/// let mut a = SteeringRng::new(42);
/// let mut b = SteeringRng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// // Uniforms land in [0, 1).
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SteeringRng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
    /// The second variate of the most recent Box–Muller pair, kept until it is
    /// consumed by the next call to [`SteeringRng::next_standard_normal`].
    cached_normal: Option<f64>,
}

impl SteeringRng {
    /// Create a generator from `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    /// Create a generator whose seed is derived from `seed` and a `label`, so
    /// that independent streams can be spun off a single configured seed without
    /// correlating them.
    ///
    /// The label is folded in with the same avalanche the generator itself uses,
    /// which is more than enough to decorrelate the resulting streams.
    #[must_use]
    pub fn derive(seed: u64, label: u64) -> Self {
        let mut mixer = Self::new(seed ^ label.wrapping_mul(GOLDEN_GAMMA));
        let derived = mixer.next_u64();
        Self::new(derived)
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
    /// Takes the top 53 bits of a `u64` draw (the best-avalanched ones) and
    /// scales by `2^-53`. `0.0` is attainable; `1.0` is not.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_ULP
    }

    /// Draw a uniform `f64` in the **open** interval `(0, 1)`.
    ///
    /// Box–Muller evaluates `ln(u)`, which is `-inf` at `u == 0`. Rather than
    /// rejection-sampling the (astronomically rare, but not impossible — `2^-53`)
    /// zero, this shifts the grid by half an ulp: `(k + 0.5) * 2^-53`, bounded
    /// strictly inside `(0, 1)` and still exactly uniform on its grid. That makes
    /// [`SteeringRng::next_standard_normal`] *total*: it cannot produce a
    /// non-finite value.
    pub fn next_f64_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * F64_ULP
    }

    /// Draw a uniform `f64` in `[-1, 1)`.
    ///
    /// The fixture model ([`SteeringFixtureModel`](super::SteeringFixtureModel))
    /// builds its embeddings and weight matrices from this.
    pub fn next_symmetric(&mut self) -> f64 {
        self.next_f64().mul_add(2.0, -1.0)
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
    /// yields **two** independent standard normals. Both are used: `z1` is cached
    /// and returned by the next call. Because `u1` comes from
    /// [`SteeringRng::next_f64_open`] it is strictly positive, so `ln(u1)` is
    /// finite and the result is always a finite `f64`.
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
    /// Uses **rejection sampling** rather than a bare `next_u64() % bound`: the
    /// naive modulo is biased whenever `bound` does not divide `2^64`, because
    /// the residues `0 .. (2^64 mod bound)` each have one extra pre-image. Here
    /// the first `2^64 mod bound` outputs are rejected and redrawn, which makes
    /// the remaining range an exact multiple of `bound` and the reduction exactly
    /// uniform.
    pub fn next_usize_below(&mut self, bound: usize) -> Option<usize> {
        if bound == 0 {
            return None;
        }
        let bound_u64 = bound as u64;
        // `2^64 mod bound`, without 128-bit arithmetic: in wrapping `u64`
        // arithmetic `0u64.wrapping_sub(bound) == 2^64 - bound`, and
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
    /// position in `[0, i]`. Every one of the `n!` permutations is equally likely,
    /// given the unbiased [`SteeringRng::next_usize_below`] above. This is what
    /// produces the train/validation split of the contrastive pairs.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            // `i + 1 >= 2`, so `next_usize_below` always yields a draw. Written
            // as an `if let` rather than an `expect` so the method is *total*: it
            // has no panic path at all, only an unreachable one.
            if let Some(j) = self.next_usize_below(i + 1) {
                items.swap(i, j);
            }
        }
    }
}

impl Default for SteeringRng {
    /// A generator seeded with `0`.
    ///
    /// Deliberately *not* entropy-seeded: a default that silently varied run to
    /// run would break the module's reproducibility contract.
    fn default() -> Self {
        Self::new(0)
    }
}
