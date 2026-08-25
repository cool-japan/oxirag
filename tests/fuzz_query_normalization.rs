//! Proptest-based fuzz harness for query building.
//!
//! Properties:
//! - `QueryBuilder::build` never panics on arbitrary text
//! - The built query's `text` field exactly preserves the input (no silent mutation)
//! - `build` without text always returns an error
//! - Whitespace-only input builds a query without panicking
//! - Unicode non-control characters are handled without panics

#![cfg(not(target_arch = "wasm32"))]

use oxirag::query_builder::QueryBuilder;
use proptest::prelude::*;

proptest! {
    /// `QueryBuilder::build` must never panic regardless of the input text.
    #[test]
    fn fuzz_query_builder_no_panic(text in any::<String>()) {
        let result = QueryBuilder::new().text(&text).build();
        prop_assert!(result.is_ok(), "build failed for text of len {}: {:?}", text.len(), result.err());
    }

    /// The `text` field of the built query must equal the string passed to
    /// `.text(...)`, i.e., the builder must not silently normalize / mutate input.
    #[test]
    fn fuzz_query_text_preserved(text in any::<String>()) {
        if let Ok(query) = QueryBuilder::new().text(&text).build() {
            prop_assert_eq!(&query.text, &text);
        }
    }

    /// Building without any text must always produce an error.
    #[test]
    fn fuzz_query_missing_text_is_error(_seed in 0u64..u64::MAX) {
        let result = QueryBuilder::new().build();
        prop_assert!(result.is_err(), "expected Err for missing text but got Ok");
    }

    /// Whitespace-only strings of varying length must not cause panics.
    #[test]
    fn fuzz_query_whitespace_no_panic(n in 0usize..=128) {
        let spaces = " ".repeat(n);
        let result = QueryBuilder::new().text(&spaces).build();
        // Whether this succeeds or errors is an implementation detail; no panic.
        let _ = result;
    }

    /// Unicode (non-control) input must not cause panics.
    #[test]
    fn fuzz_query_unicode_no_panic(text in "\\PC*") {
        let result = QueryBuilder::new().text(&text).build();
        let _ = result;
    }

    /// Build a query with top-k set; the value must be preserved.
    #[test]
    fn fuzz_query_top_k_preserved(text in "\\PC+", k in 1usize..=1000) {
        if let Ok(query) = QueryBuilder::new().text(&text).with_top_k(k).build() {
            prop_assert_eq!(query.top_k, k);
        }
    }

    /// Build a query with min_score; the value must round-trip without panicking.
    #[test]
    fn fuzz_query_min_score_no_panic(text in "\\PC+", score in 0.0f32..=1.0f32) {
        let result = QueryBuilder::new()
            .text(&text)
            .with_min_score(score)
            .build();
        prop_assert!(result.is_ok());
    }
}
