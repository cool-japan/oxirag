//! [`StarRng`] — the module's deterministic, dependency-free pseudo-random
//! number generator, used to shuffle the problem order each bootstrapping round.
//!
//! `STaR` processes its training problems in some order every round. That order
//! is immaterial to the *fixed point* the loop reaches (the correctness filter
//! is order-independent), but reproducing a *specific* run — the exact sequence
//! of rounds an operator logged — needs a deterministic source of shuffles. This
//! crate takes no dependency on `rand` (see the project's `SciRS2` policy), so
//! the module ships a small, well-understood generator instead.
//!
//! # Why a module-prefixed name
//!
//! The crate funnels ~200 modules through a single flat prelude, and a bare
//! `SplitMix64Rng` is *already* exported there (by `bandit_ranker`). A second
//! bare `SplitMix64Rng` would collide, so this module's generator is
//! [`StarRng`]. It is the same algorithm — a `SplitMix64` fixed-increment Weyl
//! sequence with a `MurmurHash3`-style avalanche finalizer — reimplemented here
//! rather than imported, so the `self-taught-reasoner` feature stays independent
//! of `bandit-ranker`.
//!
//! # `SplitMix64`
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
//! bijection on `u64`, so the output inherits the full period. Every seed —
//! including `0` — produces a full-period stream; there is no degenerate seed.

// The generator's whole job is bit-twiddling `u64`s into `usize`s. The one cast
// below (`u64 -> usize`) is guarded by a modulus that is itself a `usize`.
#![allow(clippy::cast_possible_truncation)]

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// Apply the `SplitMix64` avalanche finalizer to `z`.
///
/// A bijection on `u64`: distinct inputs give distinct outputs. Exposed as
/// [`mix_seed`] so the engine can derive an independent, reproducible per-round
/// seed from `(seed, round_index)` without pulling in the whole generator.
#[must_use]
fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
    z ^ (z >> 31)
}

/// Derive a fresh, reproducible seed from a base `seed` and a `salt`.
///
/// Used to give each bootstrapping round its own shuffle stream, so that a round
/// can be reproduced from `(seed, round_index)` in isolation. Because the
/// `SplitMix64` finalizer is a bijection, distinct salts yield distinct derived
/// seeds for a fixed base seed.
#[must_use]
pub fn mix_seed(seed: u64, salt: u64) -> u64 {
    finalize(seed.wrapping_add(salt.wrapping_mul(GOLDEN_GAMMA)))
}

/// A deterministic `SplitMix64` pseudo-random number generator.
///
/// Seeded once, it reproduces its entire stream exactly. See the
/// [module documentation](self) for the algorithm and its full-period guarantee.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "self-taught-reasoner")]
/// # {
/// use oxirag::self_taught_reasoner::StarRng;
///
/// let mut a = StarRng::new(42);
/// let mut b = StarRng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StarRng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl StarRng {
    /// Create a generator from `seed`.
    ///
    /// Every seed is equally good — `SplitMix64` has no degenerate states — so
    /// `0` is a perfectly ordinary seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next uniformly distributed `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        finalize(self.state)
    }

    /// Draw a uniform `usize` in `[0, bound)`, or `None` when `bound == 0`.
    ///
    /// Uses **rejection sampling** rather than a biased `next_u64() % bound`. The
    /// first `2^64 mod bound` outputs are rejected and redrawn, which makes the
    /// remaining range an exact multiple of `bound` and the reduction exactly
    /// uniform. The expected number of redraws is far below `1` for any `bound` a
    /// problem set would plausibly reach.
    pub fn next_usize_below(&mut self, bound: usize) -> Option<usize> {
        if bound == 0 {
            return None;
        }
        let bound_u64 = bound as u64;
        // `2^64 mod bound`, computed without 128-bit arithmetic: in wrapping
        // `u64` arithmetic `0u64.wrapping_sub(bound) == 2^64 - bound`, and
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
    /// position in `[0, i]`. Every one of the `n!` permutations is equally likely
    /// (given the unbiased [`Self::next_usize_below`]). The method is *total*: it
    /// has no panic path, only an unreachable one written as an `if let`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            if let Some(j) = self.next_usize_below(i + 1) {
                items.swap(i, j);
            }
        }
    }
}

impl Default for StarRng {
    /// A generator seeded with `0`.
    ///
    /// Deliberately *not* entropy-seeded: a default that silently varied run to
    /// run would break the reproducibility contract. Callers who want a fresh
    /// trajectory pass a fresh seed.
    fn default() -> Self {
        Self::new(0)
    }
}
