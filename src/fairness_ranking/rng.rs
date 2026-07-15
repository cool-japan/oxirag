//! A seeded `SplitMix64` generator, so that every randomized part of this module
//! is **reproducible from an integer**.
//!
//! This crate takes no dependency on `rand`, and the fairness machinery needs
//! pseudo-randomness in exactly two places:
//!
//! * [`DeltrModel`](crate::fairness_ranking::DeltrModel) initializes its weight
//!   vector from a small symmetric interval rather than from zeros. A zero
//!   initialization is not *wrong* for a linear softmax model — the listwise
//!   gradient at `w = 0` is non-zero, so descent does start moving — but it puts
//!   every feature on an exactly symmetric footing, and on a degenerate design
//!   (duplicated or perfectly anti-correlated columns) the iterates can stay
//!   pinned on that symmetry. A tiny seeded perturbation breaks it, and being
//!   *seeded* means a training run is replayable byte for byte from
//!   [`DeltrConfig::seed`](crate::fairness_ranking::DeltrConfig::seed).
//! * The module's tests generate seeded cost matrices to check the assignment
//!   solver against brute force, and seeded null rankings to check that the
//!   `FA*IR` multiple-test correction actually attains its nominal family-wise
//!   error rate.
//!
//! `SplitMix64` (Steele, Lea & Flood, 2014) is the right generator for this job:
//! it is four lines, it has no degenerate seeds — the state is advanced by a
//! fixed odd increment, so *every* seed enters the same full-period Weyl
//! sequence — and it passes `BigCrush`. It is not cryptographic and is not meant
//! to be.
//!
//! The type is deliberately **not** called `SplitMix64Rng`: that name is already
//! exported by `bandit_ranker`, and this crate's prelude is flat.

/// The `SplitMix64` Weyl increment: `floor(2^64 / phi)` rounded to the nearest
/// odd integer, where `phi` is the golden ratio. Being odd makes the additive
/// sequence `state += GAMMA` hit every one of the `2^64` states before
/// repeating, which is what makes every seed equally good.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First of the two mixing multipliers in the `SplitMix64` finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second of the two mixing multipliers in the `SplitMix64` finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// `2^-53`: the spacing of the uniform grid a 53-bit mantissa can represent in
/// `[0, 1)`. `2^53 = 9_007_199_254_740_992`, written as a literal so there is no
/// lossy `u64 -> f64` cast (the value is exactly representable, but clippy cannot
/// see that through a cast).
const F64_ULP: f64 = 1.0 / 9_007_199_254_740_992.0;

/// A seeded `SplitMix64` pseudo-random generator.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "fairness-ranking")]
/// # {
/// use oxirag::fairness_ranking::FairnessRng;
///
/// let mut a = FairnessRng::new(7);
/// let mut b = FairnessRng::new(7);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FairnessRng {
    /// The Weyl-sequence state, advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl FairnessRng {
    /// Create a generator from `seed`.
    ///
    /// Every seed is equally good, `0` included: `SplitMix64` has no degenerate
    /// states (see the [module documentation](self)).
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

    /// Draw a uniform `f64` in the half-open interval `[0, 1)`.
    ///
    /// Built from the top 53 bits of a `u64` draw (the best-avalanched ones),
    /// scaled by `2^-53`, so the result is uniform over the `2^53` exactly
    /// representable multiples of `2^-53` in `[0, 1)`. `0.0` is attainable;
    /// `1.0` is not.
    #[allow(clippy::cast_precision_loss)] // 53-bit value into a 53-bit mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_ULP
    }

    /// Draw a uniform `f64` in `[low, high)`.
    ///
    /// If `high <= low` the interval is empty and `low` is returned; if either
    /// bound is non-finite the result is whatever the arithmetic produces, which
    /// is the caller's problem to have created.
    pub fn next_range(&mut self, low: f64, high: f64) -> f64 {
        if high <= low {
            return low;
        }
        low + self.next_f64() * (high - low)
    }

    /// Draw a `bool` that is `true` with probability `p`.
    ///
    /// `p <= 0` never fires and `p >= 1` always fires; this is exactly the
    /// Bernoulli trial the `FA*IR` null model is built from (each rank is
    /// occupied by a protected candidate independently with probability `p`).
    pub fn next_bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}
