//! [`SchedulerRng`] — the module's deterministic, dependency-free pseudo-random
//! number generator, used **only to synthesise workloads in tests**.
//!
//! The scheduler itself is entirely deterministic: every tie is broken by a total
//! order on `(key, RequestId)`, so no randomness ever enters a scheduling
//! decision. What *is* randomised is the stream of `(arrival, deadline, length,
//! weight)` tuples the tests throw at it — the adversarial arrival patterns, the
//! `n <= 8` instances brute-forced against `n!` permutations, the backlogged
//! flows whose fairness bound is measured. Those need a reproducible source of
//! variates, and this crate takes no dependency on `rand` (`SciRS2` policy).
//!
//! This is the same `SplitMix64` generator the `bandit_ranker` module's
//! `SplitMix64Rng` ships, re-derived here under a **module-prefixed name** so the
//! flat crate prelude never sees two bare `SplitMix64Rng`s. See that module for
//! the full
//! derivation of the algorithm, its full `2^64` period, and why every seed
//! (including `0`) is equally good.

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// A deterministic `SplitMix64` pseudo-random number generator.
///
/// Seeded once, it reproduces its entire stream exactly. Used by this module's
/// test suite to generate reproducible random workloads; a given seed replayed
/// against the same generation code yields a byte-identical workload, which is
/// what makes the module's determinism test (same seed, same workload, identical
/// schedule tick-for-tick) meaningful.
#[derive(Debug, Clone)]
pub struct SchedulerRng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl SchedulerRng {
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
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
        z ^ (z >> 31)
    }

    /// Draw a uniform `u64` in the inclusive range `[low, high]`.
    ///
    /// Uses rejection sampling so the reduction is exactly uniform (a bare modulo
    /// is biased whenever the span does not divide `2^64`). Returns `low` when
    /// `high <= low`, so the method is total.
    pub fn next_range(&mut self, low: u64, high: u64) -> u64 {
        if high <= low {
            return low;
        }
        // `span` is the number of admissible values; it cannot overflow here
        // because `high > low` guarantees `high - low >= 1`, and adding one to a
        // value strictly below `u64::MAX` is safe. When `high == u64::MAX` and
        // `low == 0` the span would be `2^64`; guard that as the whole range.
        let span = high - low;
        if span == u64::MAX {
            return low.wrapping_add(self.next_u64());
        }
        let span = span + 1;
        let reject_below = span.wrapping_neg() % span;
        loop {
            let draw = self.next_u64();
            if draw >= reject_below {
                return low + (draw % span);
            }
        }
    }

    /// Draw a `usize` in `[0, bound)`, or `None` when `bound == 0`.
    pub fn next_index(&mut self, bound: usize) -> Option<usize> {
        if bound == 0 {
            return None;
        }
        // `bound - 1` is a valid inclusive upper bound; the cast back to `usize`
        // is exact because the drawn value is `< bound <= usize::MAX`.
        let high = (bound - 1) as u64;
        Some(usize::try_from(self.next_range(0, high)).unwrap_or(0))
    }
}

impl Default for SchedulerRng {
    /// A generator seeded with `0`. Deliberately *not* entropy-seeded: a default
    /// that silently varied run to run would break reproducibility.
    fn default() -> Self {
        Self::new(0)
    }
}
