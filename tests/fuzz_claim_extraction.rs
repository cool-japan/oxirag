//! Proptest-based fuzz harness for claim extraction and normalization.
//!
//! Properties under test:
//! - `AdvancedClaimExtractor::extract_claims` never panics on arbitrary string input
//! - `DefaultClaimNormalizer::normalize_text` is idempotent: normalize(normalize(s)) == normalize(s)
//! - SMT-LIB output from extracted claims contains no null bytes
//! - Extracting from the same input twice yields the same count

#![cfg(not(target_arch = "wasm32"))]

use oxirag::layer3_judge::normalizer::ClaimNormalizer;
use oxirag::layer3_judge::{AdvancedClaimExtractor, ClaimExtractor, DefaultClaimNormalizer};
use proptest::prelude::*;

proptest! {
    /// `extract_claims` must never panic regardless of the input string.
    #[test]
    fn fuzz_extractor_no_panic(text in any::<String>()) {
        let extractor = AdvancedClaimExtractor::default();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime build failed");
        let result = rt.block_on(extractor.extract_claims(&text, 20));
        // The method returns a Result; we just assert it didn't panic.
        // Errors are acceptable; panics are not.
        let _ = result;
    }

    /// `normalize_text` must be idempotent: applying it twice yields the same
    /// string as applying it once.
    #[test]
    fn fuzz_normalizer_idempotent(text in any::<String>()) {
        let normalizer = DefaultClaimNormalizer::default();
        let once = normalizer.normalize_text(&text);
        let twice = normalizer.normalize_text(&once);
        prop_assert_eq!(once, twice);
    }

    /// Every claim's `.text` field must not contain null bytes.
    #[test]
    fn fuzz_extractor_output_no_null_bytes(text in ".*") {
        let extractor = AdvancedClaimExtractor::default();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime build failed");
        if let Ok(claims) = rt.block_on(extractor.extract_claims(&text, 20)) {
            for claim in &claims {
                prop_assert!(
                    !claim.text.contains('\0'),
                    "claim text contains null byte: {:?}", claim.text
                );
            }
        }
    }

    /// Extracting claims from the same input string twice must yield the same
    /// number of claims (determinism / idempotency of extraction).
    #[test]
    fn fuzz_extractor_dedup_idempotent(text in any::<String>()) {
        let extractor = AdvancedClaimExtractor::default();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime build failed");
        let claims1 = rt.block_on(extractor.extract_claims(&text, 50));
        let claims2 = rt.block_on(extractor.extract_claims(&text, 50));
        if let (Ok(c1), Ok(c2)) = (claims1, claims2) {
            prop_assert_eq!(c1.len(), c2.len());
        }
        // Both erroring is fine; we only check counts when both succeed.
    }

    /// SMT-LIB strings produced by `to_smtlib` must not contain null bytes.
    #[test]
    fn fuzz_smtlib_no_null_bytes(text in "[ \\t\\na-zA-Z0-9.,!?;:'\"()-]{0,200}") {
        let extractor = AdvancedClaimExtractor::default();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime build failed");
        if let Ok(claims) = rt.block_on(extractor.extract_claims(&text, 10)) {
            for claim in &claims {
                if let Ok(smt) = extractor.to_smtlib(claim) {
                    prop_assert!(
                        !smt.contains('\0'),
                        "SMT-LIB output contains null byte for claim: {:?}", claim.text
                    );
                }
            }
        }
    }

    /// `normalize_text` on whitespace-only strings must produce an empty string.
    #[test]
    fn fuzz_normalizer_whitespace_collapses(n in 0usize..=64) {
        let spaces: String = " \t\n".chars().cycle().take(n).collect();
        let normalizer = DefaultClaimNormalizer::default();
        let result = normalizer.normalize_text(&spaces);
        prop_assert!(result.is_empty(), "expected empty, got {:?}", result);
    }
}
