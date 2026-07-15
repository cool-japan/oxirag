//! [`MctsRng`] — the module's deterministic, dependency-free pseudo-random number
//! generator.
//!
//! Monte-Carlo tree search is *Monte-Carlo*: the "simulation" phase is a random
//! playout, and without a source of randomness the algorithm is not defined. This
//! crate takes no dependency on `rand` / `rand_distr` (see the project's `SciRS2`
//! policy), so the generator ships here.
//!
//! # Why a *separate* generator
//!
//! `bandit_ranker` already exports a `SplitMix64Rng`, and
//! `click_model` already exports a `ClickSplitMix64`. Both are
//! the same well-known algorithm. The crate's prelude is **flat**, so each module
//! carries its own, module-prefixed copy rather than reaching across a module
//! boundary for one — the cost is thirty lines, and the benefit is that
//! `mcts_reasoning`'s reproducibility contract does not depend on a type owned by an
//! unrelated module that is free to change its stream at any time. A search whose
//! recorded seed no longer replays is worse than useless.
//!
//! # `SplitMix64`
//!
//! `SplitMix64` (Steele, Lea & Flood, 2014, "Fast Splittable Pseudorandom Number
//! Generators") is a fixed-increment Weyl sequence followed by a `MurmurHash3`-style
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
//! All arithmetic is wrapping (mod 2^64). The increment is odd, so the state walks
//! every one of the 2^64 residues before repeating, and the finalizer is a bijection
//! on `u64`, so the output inherits the full period. **Every** seed is a good seed,
//! including `0`. It passes `BigCrush`, which is far more than a rollout policy
//! needs.
//!
//! # Reproducibility contract
//!
//! Everything downstream of an [`MctsRng`] is a pure function of the seed and the
//! call sequence. Given the same [`MctsConfig::seed`](crate::mcts_reasoning::MctsConfig::seed),
//! the same generator, and the same evaluator, a search reproduces its tree
//! **bit-for-bit** — every visit count, every accumulated value, every node, in
//! order. That is what makes `mcts_reasoning`'s determinism test meaningful, and it
//! is what lets an operator replay a production search from a logged seed.
//!
//! Note that the *tie-breaks* in selection and in final action choice are **not**
//! randomized (they are resolved by node id — see
//! [`MctsTree::best_child_by_visits`](crate::mcts_reasoning::MctsTree::best_child_by_visits)).
//! The `RNG` is consumed by exactly two things: the rollout policy, and the
//! [`UniformRandom`](crate::mcts_reasoning::MctsSelectionPolicy::UniformRandom)
//! ablation baseline. Nothing else in the search is stochastic.

// The generator's whole job is bit-twiddling `u64`s into `f64`s and `usize`s. The
// `u64 -> f64` mantissa cast is *exactly* representable by construction (the value
// is < 2^53), and the `u64 -> usize` cast is guarded by a modulus that is itself a
// `usize`.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// `2^-53` — the spacing of the `f64` grid this generator samples on.
///
/// An `f64` has a 53-bit significand, so the 2^53 values `k * 2^-53` for
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
/// # #[cfg(feature = "mcts-reasoning")]
/// # {
/// use oxirag::mcts_reasoning::MctsRng;
///
/// let mut a = MctsRng::new(7);
/// let mut b = MctsRng::new(7);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// // Uniforms land in [0, 1).
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MctsRng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl MctsRng {
    /// Create a generator from `seed`.
    ///
    /// Every seed is equally good — `SplitMix64` has no degenerate states (see the
    /// [module documentation](self)), so `0` is a perfectly ordinary seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
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
    /// Implemented by taking the top 53 bits of a `u64` draw (the high bits are the
    /// best-avalanched ones) and scaling by `2^-53`, so the result is a uniform draw
    /// over the 2^53 exactly-representable multiples of `2^-53` in `[0, 1)`. `0.0`
    /// *is* attainable; `1.0` is not — which is what makes it safe to use as the
    /// inverse-transform coordinate of a cumulative distribution whose total mass is
    /// `1.0`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_ULP
    }

    /// Draw a uniformly distributed index in `[0, n)`, or `None` when `n == 0`.
    ///
    /// Uses **Lemire's** multiply-shift reduction (`(x * n) >> 64` on the full
    /// 128-bit product) rather than `x % n`. The modulo is not merely slower; it is
    /// *biased* whenever `n` does not divide 2^64, because the `2^64 mod n` lowest
    /// residues get one extra preimage. Lemire's map has the same defect in
    /// principle but its bias is bounded by `n / 2^64` — for the handful of
    /// reasoning steps this module ever chooses between, that is zero for all
    /// practical purposes and it costs one multiply.
    pub fn next_usize_below(&mut self, n: usize) -> Option<usize> {
        if n == 0 {
            return None;
        }
        let product = u128::from(self.next_u64()) * (n as u128);
        Some((product >> 64) as usize)
    }
}
