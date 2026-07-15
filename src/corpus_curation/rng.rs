//! [`CurationRng`] — the module's deterministic, dependency-free pseudo-random
//! number generator, plus the `FNV-1a` hash shared by the classifier's
//! feature-hashing and the near-duplicate pillar's shingle hashing.
//!
//! `corpus_curation` needs genuine randomness in three places: shuffling
//! documents for a train/test split ([`super::classifier::train_test_split`]),
//! optionally randomising a classifier's initial weights, and — in the test
//! suite — drawing synthetic "good" and "bad" documents from two distinct word
//! distributions so the classifier test measures generalisation rather than
//! memorisation of a hand-picked fixture. This crate takes no dependency on
//! `rand` / `rand_distr` (the project's `SciRS2` policy), so this module ships
//! its own small, well-understood generator.
//!
//! # Why a module-prefixed name
//!
//! `crate::bandit_ranker::SplitMix64Rng` already implements the identical
//! `SplitMix64` algorithm and sits in the crate's flat prelude. Re-using that
//! *name* here (even with an independent implementation) would collide the
//! moment both features are enabled together, so this type is named
//! [`CurationRng`] instead. The algorithm is intentionally the same well-proven
//! generator — there is no reason to invent a second one — only the name and
//! the (smaller) surface differ.
//!
//! # `SplitMix64`
//!
//! `SplitMix64` (Steele, Lea & Flood, 2014) is a fixed-increment Weyl sequence
//! followed by a `MurmurHash3`-style avalanche finalizer. All arithmetic is
//! `wrapping` (mod 2^64); the increment is odd so the state visits all 2^64
//! residues before repeating, and the finalizer is a bijection on `u64`, so
//! every seed — including `0` — produces a full-period, well-mixed stream.

/// The 64-bit odd "golden ratio" increment, `floor(2^64 / phi) | 1`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// First avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second avalanche multiplier of the `MurmurHash3`-style finalizer.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;
/// `2^-53`, the spacing of the `f64` grid this generator samples on.
const F64_ULP: f64 = 1.0 / 9_007_199_254_740_992.0; // 1 / 2^53

/// A deterministic `SplitMix64` pseudo-random number generator, scoped to
/// `corpus_curation`.
///
/// Seeded once, it reproduces its entire stream exactly: the same seed
/// replayed against the same call sequence always yields byte-identical
/// output. See the [module documentation](self) for the algorithm.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "corpus-curation")] {
/// use oxirag::corpus_curation::CurationRng;
///
/// let mut a = CurationRng::new(42);
/// let mut b = CurationRng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
///
/// let u = a.next_f64();
/// assert!((0.0..1.0).contains(&u));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CurationRng {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] on every draw.
    state: u64,
}

impl CurationRng {
    /// Create a generator from `seed`. Every seed is equally good; `0` is a
    /// perfectly ordinary seed (see the [module documentation](self)).
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
    /// Takes the top 53 bits of a `u64` draw (the best-avalanched bits) and
    /// scales by `2^-53`, giving a uniform draw over the 2^53 exactly
    /// representable grid points in `[0, 1)` with no rounding bias.
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * F64_ULP
    }

    /// Draw `true` with probability `p` (clamped to `[0, 1]`).
    pub fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p.clamp(0.0, 1.0)
    }

    /// Draw a uniform `usize` in `[0, bound)`, or `None` when `bound == 0`.
    ///
    /// Uses rejection sampling rather than a biased `next_u64() % bound`: the
    /// first `2^64 mod bound` draws are discarded so the surviving range is an
    /// exact multiple of `bound`, making the reduction exactly uniform.
    #[allow(clippy::cast_possible_truncation)]
    pub fn next_usize_below(&mut self, bound: usize) -> Option<usize> {
        if bound == 0 {
            return None;
        }
        let bound_u64 = bound as u64;
        let reject_below = bound_u64.wrapping_neg() % bound_u64;
        loop {
            let draw = self.next_u64();
            if draw >= reject_below {
                return Some((draw % bound_u64) as usize);
            }
        }
    }

    /// Shuffle `items` uniformly at random with an in-place Fisher-Yates
    /// (Durstenfeld) pass. Every one of the `n!` permutations is equally
    /// likely.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            if let Some(j) = self.next_usize_below(i + 1) {
                items.swap(i, j);
            }
        }
    }

    /// Choose a uniformly random element of `items`, or `None` when empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        self.next_usize_below(items.len()).map(|i| &items[i])
    }
}

impl Default for CurationRng {
    /// A generator seeded with `0`. Deliberately not entropy-seeded: a
    /// silently-varying default would break reproducibility. Callers who want
    /// a fresh trajectory should pass a fresh seed.
    fn default() -> Self {
        Self::new(0)
    }
}

// ── FNV-1a hashing ──────────────────────────────────────────────────────────

/// Compute the deterministic `FNV-1a` 64-bit hash of `bytes`.
///
/// Shared by the classifier's hashed bag-of-words feature extraction
/// ([`super::classifier`]) and the near-duplicate pillar's shingle hashing
/// ([`super::near_dup`]) so both use one, consistent, dependency-free hash
/// rather than two hand-rolled copies.
#[must_use]
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = CurationRng::new(7);
        let mut b = CurationRng::new(7);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = CurationRng::new(1);
        let mut b = CurationRng::new(2);
        let seq_a: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut rng = CurationRng::new(123);
        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn next_usize_below_zero_is_none() {
        let mut rng = CurationRng::new(1);
        assert_eq!(rng.next_usize_below(0), None);
    }

    #[test]
    fn next_usize_below_respects_bound() {
        let mut rng = CurationRng::new(99);
        for _ in 0..1000 {
            let v = rng.next_usize_below(7).expect("bound is nonzero");
            assert!(v < 7);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = CurationRng::new(55);
        let mut items: Vec<usize> = (0..20).collect();
        rng.shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn fnv1a_is_deterministic_and_sensitive() {
        assert_eq!(fnv1a(b"hello"), fnv1a(b"hello"));
        assert_ne!(fnv1a(b"hello"), fnv1a(b"hellp"));
    }
}
