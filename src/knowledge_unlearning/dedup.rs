//! Word-shingle `MinHash`/`Jaccard` near-duplicate detection for
//! [`crate::knowledge_unlearning`]'s scope resolution.
//!
//! A document is reduced to the set of its `shingle_size`-word shingles.
//! For each of `num_perm` deterministic permutations (realised by seeding
//! `FNV-1a` with the permutation index) the minimum hash over all shingles
//! is recorded. The fraction of matching signature slots between two
//! documents is an unbiased estimate of the `Jaccard` similarity of their
//! shingle sets — high similarity between two *different* documents is
//! exactly the signal that a paraphrased or re-chunked copy of deleted
//! content is still present elsewhere in the corpus.
//!
//! This is a self-contained, hand-rolled reimplementation in the same style
//! as [`crate::semantic_dedup::minhash`] — deliberately not imported from
//! there, since `knowledge_unlearning` must not depend on the
//! `semantic-dedup` feature (or any other optional feature) to keep its own
//! feature flag independent.

use std::collections::HashMap;

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Sentinel used for empty shingle sets so signatures stay comparable.
const EMPTY_SLOT: u64 = u64::MAX;

/// Compute the deterministic `FNV-1a` 64-bit hash of `bytes`.
#[must_use]
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Split `text` into lowercase alphanumeric tokens.
///
/// Unlike a stricter tokenizer, single-character tokens (e.g. standalone
/// digits) are kept: distinctive facts in unlearning targets are often
/// short numeric or code-like tokens that must not be silently dropped.
#[must_use]
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Build the set of `shingle_size`-word shingles from `text`.
///
/// When the token count is below `shingle_size` (but non-zero) the whole
/// token list is treated as a single shingle, so short passages still
/// compare sensibly.
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
/// The returned vector always has length `num_perm.max(1)`. Empty inputs
/// yield a signature filled with the [`EMPTY_SLOT`] sentinel, so two empty
/// (or shingle-less) texts compare as maximally similar to each other and
/// [`jaccard`] is well-defined for them.
#[must_use]
pub(crate) fn minhash_signature(text: &str, shingle_size: usize, num_perm: usize) -> Vec<u64> {
    let perms = num_perm.max(1);
    let grams = shingles(text, shingle_size);
    if grams.is_empty() {
        return vec![EMPTY_SLOT; perms];
    }

    let mut signature = vec![EMPTY_SLOT; perms];
    for (j, slot) in signature.iter_mut().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
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
pub(crate) fn jaccard(a: &[u64], b: &[u64]) -> f64 {
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
    equal as f64 / len as f64
}

/// A count of how many times each shingle occurs in a text, used by
/// [`UnlearningNearDuplicateDetector::exact_shingle_overlap`] for a cheap,
/// deterministic sanity check independent of the randomized-looking
/// `MinHash` estimate.
fn shingle_multiset(text: &str, shingle_size: usize) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for gram in shingles(text, shingle_size) {
        *counts.entry(gram).or_insert(0) += 1;
    }
    counts
}

// ── UnlearningNearDuplicateDetector ─────────────────────────────────────────

/// Detects near-duplicate documents via `MinHash`-estimated `Jaccard`
/// similarity over word shingles.
///
/// This is the primitive [`crate::knowledge_unlearning::scope`] uses to
/// find paraphrased or re-chunked copies of content named in an
/// [`crate::knowledge_unlearning::UnlearningRequest`] — content that a
/// naive "delete this one document id" operation would never find.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnlearningNearDuplicateDetector {
    /// Number of words per shingle. See
    /// [`crate::knowledge_unlearning::UnlearningConfig::shingle_size`].
    pub shingle_size: usize,
    /// Number of `MinHash` permutations (signature length). See
    /// [`crate::knowledge_unlearning::UnlearningConfig::num_perm`].
    pub num_perm: usize,
}

impl UnlearningNearDuplicateDetector {
    /// Create a new detector with the given shingle size and signature
    /// length. Both are clamped to a minimum of `1`.
    #[must_use]
    pub fn new(shingle_size: usize, num_perm: usize) -> Self {
        Self {
            shingle_size: shingle_size.max(1),
            num_perm: num_perm.max(1),
        }
    }

    /// Compute the `MinHash` signature of `text` under this detector's
    /// configuration.
    #[must_use]
    pub fn signature(&self, text: &str) -> Vec<u64> {
        minhash_signature(text, self.shingle_size, self.num_perm)
    }

    /// Estimate the `Jaccard` similarity of `a` and `b`'s shingle sets via
    /// their `MinHash` signatures, in `[0.0, 1.0]`.
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        jaccard(&self.signature(a), &self.signature(b))
    }

    /// The exact (non-estimated) `Jaccard` similarity of `a` and `b`'s
    /// shingle multisets, computed by direct set intersection rather than
    /// `MinHash` sampling.
    ///
    /// [`similarity`](Self::similarity) is the method scope resolution and
    /// auditing actually use (it is what stays cheap at corpus scale); this
    /// exact variant exists for tests that need a ground-truth comparison
    /// point independent of `MinHash`'s sampling error.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn exact_shingle_overlap(&self, a: &str, b: &str) -> f64 {
        let sa = shingle_multiset(a, self.shingle_size);
        let sb = shingle_multiset(b, self.shingle_size);
        if sa.is_empty() && sb.is_empty() {
            return 1.0;
        }
        if sa.is_empty() || sb.is_empty() {
            return 0.0;
        }
        let mut intersection = 0u64;
        for (gram, count_a) in &sa {
            if let Some(count_b) = sb.get(gram) {
                intersection += u64::from((*count_a).min(*count_b));
            }
        }
        let union: u64 = sa.values().map(|&c| u64::from(c)).sum::<u64>()
            + sb.values().map(|&c| u64::from(c)).sum::<u64>()
            - intersection;
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// `true` when [`similarity`](Self::similarity) meets or exceeds
    /// `threshold`.
    #[must_use]
    pub fn is_near_duplicate(&self, a: &str, b: &str, threshold: f64) -> bool {
        self.similarity(a, b) >= threshold
    }
}

impl Default for UnlearningNearDuplicateDetector {
    fn default() -> Self {
        Self::new(3, 64)
    }
}
