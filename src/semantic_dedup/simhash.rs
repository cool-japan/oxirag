//! `SimHash` fingerprinting over token streams.
//!
//! Each token is hashed to a 64-bit value with a deterministic `FNV-1a` hash.
//! For every bit position a signed vote is accumulated (`+1` when the bit is
//! set in the token hash, `-1` otherwise), optionally weighted by the token's
//! frequency. The fingerprint bit is `1` when the accumulated vote is positive.
//! Two near-duplicate passages therefore differ in only a handful of bits, so
//! the `Hamming` distance of their fingerprints is small.

use std::collections::HashMap;

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

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

/// Split `text` into lowercase alphanumeric tokens of length `>= 2`.
#[must_use]
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Compute the 64-bit `SimHash` fingerprint of `text`.
///
/// Tokens are frequency-weighted: a token appearing `k` times contributes a
/// vote of magnitude `k` to each bit position. An empty token stream yields `0`.
#[must_use]
pub(crate) fn simhash(text: &str) -> u64 {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return 0;
    }

    // Frequency-weight tokens so repeated terms dominate the fingerprint.
    let mut freq: HashMap<String, i64> = HashMap::new();
    for token in tokens {
        *freq.entry(token).or_insert(0) += 1;
    }

    let mut votes = [0i64; 64];
    for (token, weight) in &freq {
        let hash = fnv1a(token.as_bytes());
        for (bit, vote) in votes.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *vote += weight;
            } else {
                *vote -= weight;
            }
        }
    }

    let mut fingerprint = 0u64;
    for (bit, &vote) in votes.iter().enumerate() {
        if vote > 0 {
            fingerprint |= 1u64 << bit;
        }
    }
    fingerprint
}

/// Compute the `Hamming` distance between two 64-bit fingerprints.
#[must_use]
pub(crate) fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}
