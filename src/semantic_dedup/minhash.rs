//! `MinHash` signatures and `Jaccard` estimation over word shingles.
//!
//! A passage is reduced to the set of its `shingle_size`-word shingles. For
//! each of `num_perm` deterministic permutations (realised by seeding `FNV-1a`
//! with the permutation index) the minimum hash over all shingles is recorded.
//! The fraction of matching signature slots between two passages is an unbiased
//! estimate of the `Jaccard` similarity of their shingle sets.

use super::simhash::{fnv1a, tokenize};

/// Sentinel used for empty shingle sets so signatures stay comparable.
const EMPTY_SLOT: u64 = u64::MAX;

/// Build the set of `shingle_size`-word shingles from `text`.
///
/// When the token count is below `shingle_size` (but non-zero) the whole token
/// list is treated as a single shingle, so short passages still compare sensibly.
#[must_use]
pub(crate) fn shingles(text: &str, shingle_size: usize) -> Vec<String> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Vec::new();
    }
    let size = shingle_size.max(1);
    if tokens.len() < size {
        return vec![tokens.join(" ")];
    }
    tokens.windows(size).map(|w| w.join(" ")).collect()
}

/// Hash a shingle under permutation `seed` by xoring the seed into `FNV-1a`.
fn seeded_hash(shingle: &str, seed: u64) -> u64 {
    fnv1a(shingle.as_bytes()) ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Compute the `MinHash` signature of `text` with `num_perm` permutations.
///
/// The returned vector always has length `num_perm.max(1)`. Empty inputs yield
/// a signature filled with the [`EMPTY_SLOT`] sentinel.
#[must_use]
pub(crate) fn minhash_signature(text: &str, shingle_size: usize, num_perm: usize) -> Vec<u64> {
    let perms = num_perm.max(1);
    let grams = shingles(text, shingle_size);
    if grams.is_empty() {
        return vec![EMPTY_SLOT; perms];
    }

    let mut signature = vec![EMPTY_SLOT; perms];
    for (j, slot) in signature.iter_mut().enumerate() {
        let seed = j as u64;
        let mut min_hash = EMPTY_SLOT;
        for shingle in &grams {
            let h = seeded_hash(shingle, seed);
            if h < min_hash {
                min_hash = h;
            }
        }
        *slot = min_hash;
    }
    signature
}

/// Estimate the `Jaccard` similarity of two `MinHash` signatures.
///
/// Returns the fraction of equal slots. Mismatched lengths compare over the
/// shorter prefix; zero-length inputs yield `0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn jaccard(a: &[u64], b: &[u64]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let equal = a
        .iter()
        .zip(b.iter())
        .take(len)
        .filter(|(x, y)| x == y)
        .count();
    equal as f32 / len as f32
}
