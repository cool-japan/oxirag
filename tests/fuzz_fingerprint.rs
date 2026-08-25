//! Proptest-based fuzz harness for `ContextFingerprint` and
//! `ContextFingerprintGenerator`.
//!
//! Properties:
//! - `ContextFingerprint::new` never panics
//! - The `hash` and `prefix_length` fields are preserved exactly
//! - `ContextFingerprintGenerator::generate` is deterministic (same input → same output)
//! - `generate_with_length` stores the supplied prefix length verbatim
//! - `generate_prefix` produces a fingerprint whose `prefix_length` ≤ the input length
//! - `is_prefix_of` is reflexive: every fingerprint is a prefix of itself
//!
//! Note: requires the `prefix-cache` feature to be enabled, e.g.
//! `cargo nextest run --features prefix-cache --test fuzz_fingerprint`

#![cfg(not(target_arch = "wasm32"))]

use proptest::prelude::*;

#[cfg(feature = "prefix-cache")]
use oxirag::prefix_cache::{ContextFingerprint, ContextFingerprintGenerator};

proptest! {
    /// `ContextFingerprint::new` must never panic for any combination of inputs.
    #[test]
    fn fuzz_fingerprint_new_no_panic(
        hash in any::<u64>(),
        prefix_len in 0usize..=100_000,
        summary in any::<String>(),
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fp = ContextFingerprint::new(hash, prefix_len, summary.as_str());
            let _ = fp;
        }
        #[cfg(not(feature = "prefix-cache"))]
        {
            let _ = (hash, prefix_len, summary);
        }
    }

    /// The `hash` field stored in the fingerprint must equal what was passed in.
    #[test]
    fn fuzz_fingerprint_hash_preserved(
        hash in any::<u64>(),
        prefix_len in 0usize..=1_000,
        summary in any::<String>(),
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fp = ContextFingerprint::new(hash, prefix_len, summary.as_str());
            prop_assert_eq!(fp.hash, hash);
        }
        #[cfg(not(feature = "prefix-cache"))]
        {
            let _ = (hash, prefix_len, summary);
        }
    }

    /// The `prefix_length` field must equal what was passed to `new`.
    #[test]
    fn fuzz_fingerprint_prefix_length_preserved(
        hash in any::<u64>(),
        prefix_len in 0usize..=100_000,
        summary in any::<String>(),
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fp = ContextFingerprint::new(hash, prefix_len, summary.as_str());
            prop_assert_eq!(fp.prefix_length, prefix_len);
        }
        #[cfg(not(feature = "prefix-cache"))]
        {
            let _ = (hash, prefix_len, summary);
        }
    }

    /// `ContextFingerprintGenerator::generate` must be deterministic:
    /// calling it twice on the same content must yield identical fingerprints.
    #[test]
    fn fuzz_generator_deterministic(content in any::<String>()) {
        #[cfg(feature = "prefix-cache")]
        {
            let fgen = ContextFingerprintGenerator::new();
            let fp1 = fgen.generate(&content);
            let fp2 = fgen.generate(&content);
            prop_assert_eq!(fp1.hash, fp2.hash,
                "hash changed between two generate() calls on the same content");
            prop_assert_eq!(fp1.prefix_length, fp2.prefix_length,
                "prefix_length changed between two generate() calls");
            prop_assert_eq!(fp1.content_summary, fp2.content_summary,
                "content_summary changed between two generate() calls");
        }
        #[cfg(not(feature = "prefix-cache"))]
        let _ = content;
    }

    /// `generate_with_length` must store the explicit prefix length verbatim.
    #[test]
    fn fuzz_generator_with_length_stores_length(
        content in any::<String>(),
        explicit_len in 0usize..=100_000,
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fgen = ContextFingerprintGenerator::new();
            let fp = fgen.generate_with_length(&content, explicit_len);
            prop_assert_eq!(fp.prefix_length, explicit_len);
        }
        #[cfg(not(feature = "prefix-cache"))]
        let _ = (content, explicit_len);
    }

    /// `generate_prefix` must produce a fingerprint whose `prefix_length` is
    /// at most `content.len()` — never exceeds the content character count.
    #[test]
    fn fuzz_generator_prefix_length_bounded(
        content in any::<String>(),
        requested_len in 0usize..=100_000,
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fgen = ContextFingerprintGenerator::new();
            let fp = fgen.generate_prefix(&content, requested_len);
            let expected_max = content.len();
            prop_assert!(
                fp.prefix_length <= expected_max,
                "prefix_length {} > content.len() {}",
                fp.prefix_length, expected_max
            );
        }
        #[cfg(not(feature = "prefix-cache"))]
        let _ = (content, requested_len);
    }

    /// `is_prefix_of` must be reflexive: every fingerprint is a prefix of itself.
    #[test]
    fn fuzz_fingerprint_is_prefix_of_reflexive(
        hash in any::<u64>(),
        prefix_len in 0usize..=10_000,
        summary in any::<String>(),
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let fp = ContextFingerprint::new(hash, prefix_len, summary.as_str());
            prop_assert!(fp.is_prefix_of(&fp), "fingerprint is not a prefix of itself");
        }
        #[cfg(not(feature = "prefix-cache"))]
        let _ = (hash, prefix_len, summary);
    }

    /// Two fingerprints generated from purely "a+" vs "b+" content must differ
    /// in their `content_summary` (truncated copy of input).  Guards against
    /// trivial identity-hash implementations.
    #[test]
    fn fuzz_generator_different_content_differs(
        a_len in 1usize..=50,
        b_len in 1usize..=50,
    ) {
        #[cfg(feature = "prefix-cache")]
        {
            let content_a: String = "a".repeat(a_len);
            let content_b: String = "b".repeat(b_len);
            let fgen = ContextFingerprintGenerator::new();
            let fp_a = fgen.generate(&content_a);
            let fp_b = fgen.generate(&content_b);
            prop_assert_ne!(fp_a.content_summary, fp_b.content_summary,
                "content summaries must differ for 'a+' vs 'b+' inputs");
        }
        #[cfg(not(feature = "prefix-cache"))]
        let _ = (a_len, b_len);
    }
}
