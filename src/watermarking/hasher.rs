//! [`WatermarkHasher`] — deterministic green/red vocabulary partitioning.
//!
//! Both generation and detection need the *exact same* answer to "given
//! this predecessor context, which tokens are on the green list?", computed
//! independently and without any shared state. This module is the one place
//! that answer is computed, so generation and detection can never drift
//! apart as long as they share a [`crate::watermarking::WatermarkConfig`].
//!
//! # Algorithm
//!
//! 1. Mix the secret key and the `context_width` predecessor tokens into a
//!    single 64-bit seed via FNV-1a followed by a `splitmix64` avalanche
//!    step (mirrors the pattern used elsewhere in this crate, e.g.
//!    `rp_tree_index::tree::SplitMix64` — a hand-rolled, dependency-free
//!    generator, never `rand`).
//! 2. Derive an independent 64-bit hash for *every* token id in
//!    `0..vocab_size` by mixing that seed with the token id (again FNV-1a +
//!    `splitmix64`).
//! 3. Sort `(hash, token_id)` pairs ascending and take the first
//!    `round(gamma * vocab_size)` token ids as the green list.
//!
//! Step 3 is what gives an *exact* green-list size of `round(gamma *
//! vocab_size)` (rather than each token independently flipping a
//! `gamma`-biased coin, which would only hit that size on average) — the
//! ties in step 3 are broken by ascending token id, so the result is a
//! total order and hence fully deterministic even in the vanishingly
//! unlikely event of a hash collision.
//!
//! This is the same "hash a pseudo-random permutation, take a prefix"
//! approach the original Kirchenbauer et al. (2023) reference implementation
//! uses, and it has the same computational profile: computing a step's green
//! list is `O(vocab_size log vocab_size)`, paid once per generated or
//! scored token.

use super::types::{WatermarkConfig, WatermarkError, WatermarkResult, WatermarkTokenId};

// ── SplitMix64 ──────────────────────────────────────────────────────────────

/// A minimal, dependency-free `splitmix64` pseudo-random generator.
///
/// Used only to avalanche an FNV-1a-mixed seed into a well-distributed
/// 64-bit value; never exposed outside this crate. Deterministic given its
/// seed, per this crate's no-`rand` convention (see e.g.
/// `rp_tree_index::tree::SplitMix64`, `muvera::fde`).
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator seeded with `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the generator and return the next 64-bit output.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

// ── FNV-1a ────────────────────────────────────────────────────────────────────

/// FNV-1a offset basis for 64-bit hashing.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime for 64-bit hashing.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Mix one byte into a running FNV-1a hash state.
const fn fnv1a_byte(state: u64, byte: u8) -> u64 {
    (state ^ byte as u64).wrapping_mul(FNV_PRIME)
}

/// Hash the eight bytes of a `u64` (little-endian) into a running FNV-1a
/// state.
const fn fnv1a_u64(mut state: u64, value: u64) -> u64 {
    let bytes = value.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        state = fnv1a_byte(state, bytes[i]);
        i += 1;
    }
    state
}

/// Derive the 64-bit seed for one predecessor `context` under `secret_key`:
/// FNV-1a-fold the key followed by every context token (in order, so
/// permuting the context changes the seed), then avalanche the result
/// through one `splitmix64` step.
///
/// Same key + same context (element-for-element) always yields the same
/// seed; changing either the key or any context token changes it.
fn context_seed(secret_key: u64, context: &[WatermarkTokenId]) -> u64 {
    let mut state = fnv1a_u64(FNV_OFFSET, secret_key);
    for &token in context {
        state = fnv1a_u64(state, u64::from(token));
    }
    SplitMix64::new(state).next_u64()
}

/// Derive the 64-bit hash used to rank `token_id` for green-list membership
/// under a step's `seed` (see [`context_seed`]).
fn token_hash(seed: u64, token_id: WatermarkTokenId) -> u64 {
    let mixed = fnv1a_u64(seed, u64::from(token_id));
    SplitMix64::new(mixed).next_u64()
}

// ── WatermarkHasher ───────────────────────────────────────────────────────────

/// Computes the deterministic green-list partition of the vocabulary for a
/// given predecessor context.
///
/// Holds the pieces of a [`WatermarkConfig`] that affect the partition
/// itself (`secret_key`, `gamma`, `vocab_size`, `context_width`); the
/// pre-derived `green_size` is cached at construction so it does not need
/// to be recomputed (and re-rounded) on every call.
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkHasher {
    secret_key: u64,
    gamma: f64,
    vocab_size: usize,
    context_width: usize,
    green_size: usize,
}

impl WatermarkHasher {
    /// Build a hasher from the partition-relevant fields of `config`.
    ///
    /// # Errors
    ///
    /// Propagates [`config.validate()`](WatermarkConfig::validate)'s error
    /// if `config` is invalid.
    pub fn new(config: &WatermarkConfig) -> WatermarkResult<Self> {
        config.validate()?;
        #[allow(clippy::cast_precision_loss)]
        let vocab_size_f = config.vocab_size as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let green_size = (config.gamma * vocab_size_f).round() as usize;
        let green_size = green_size.min(config.vocab_size);
        Ok(Self {
            secret_key: config.secret_key,
            gamma: config.gamma,
            vocab_size: config.vocab_size,
            context_width: config.context_width,
            green_size,
        })
    }

    /// The green-list fraction this hasher partitions with.
    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The configured vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The required predecessor-context length `h`.
    #[must_use]
    pub fn context_width(&self) -> usize {
        self.context_width
    }

    /// The secret key mixed into every seed.
    #[must_use]
    pub fn secret_key(&self) -> u64 {
        self.secret_key
    }

    /// The exact number of green tokens every partition has:
    /// `round(gamma * vocab_size)`, clamped to `vocab_size`.
    #[must_use]
    pub fn green_size(&self) -> usize {
        self.green_size
    }

    /// The 64-bit seed this hasher derives for `context`.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::ContextLengthMismatch`] unless
    /// `context.len() == context_width`.
    pub fn context_seed(&self, context: &[WatermarkTokenId]) -> WatermarkResult<u64> {
        self.check_context_len(context)?;
        Ok(context_seed(self.secret_key, context))
    }

    /// The green-list membership mask for `context`: a `vocab_size`-long
    /// vector where `mask[token_id] == true` iff `token_id` is green under
    /// this context.
    ///
    /// Exactly [`green_size`](Self::green_size) entries are `true`.
    /// Deterministic: the same context (and the same hasher configuration)
    /// always yields a bit-identical mask.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::ContextLengthMismatch`] unless
    /// `context.len() == context_width`.
    pub fn green_mask(&self, context: &[WatermarkTokenId]) -> WatermarkResult<Vec<bool>> {
        let seed = self.context_seed(context)?;
        Ok(self.green_mask_for_seed(seed))
    }

    /// Whether `token_id` is green under `context`.
    ///
    /// Convenience wrapper around [`green_mask`](Self::green_mask) for
    /// single-token queries; callers scoring many tokens against the same
    /// context (e.g. biasing a full logit vector) should call
    /// [`green_mask`](Self::green_mask) once instead.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::ContextLengthMismatch`] unless
    /// `context.len() == context_width`, or
    /// [`WatermarkError::TokenOutOfRange`] if `token_id >= vocab_size`.
    pub fn is_green(
        &self,
        context: &[WatermarkTokenId],
        token_id: WatermarkTokenId,
    ) -> WatermarkResult<bool> {
        if token_id as usize >= self.vocab_size {
            return Err(WatermarkError::TokenOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }
        let mask = self.green_mask(context)?;
        Ok(mask[token_id as usize])
    }

    /// Shared implementation of [`green_mask`](Self::green_mask) once the
    /// context has already been reduced to a seed.
    fn green_mask_for_seed(&self, seed: u64) -> Vec<bool> {
        #[allow(clippy::cast_possible_truncation)]
        let mut scored: Vec<(u64, WatermarkTokenId)> = (0..self.vocab_size as u32)
            .map(|id| (token_hash(seed, id), id))
            .collect();
        scored.sort_unstable();

        let mut mask = vec![false; self.vocab_size];
        for &(_, id) in scored.iter().take(self.green_size) {
            mask[id as usize] = true;
        }
        mask
    }

    /// Validate that `context` has exactly `context_width` entries.
    fn check_context_len(&self, context: &[WatermarkTokenId]) -> WatermarkResult<()> {
        if context.len() == self.context_width {
            Ok(())
        } else {
            Err(WatermarkError::ContextLengthMismatch {
                expected: self.context_width,
                actual: context.len(),
            })
        }
    }
}
