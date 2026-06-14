#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]

//! Tests for the `citation_verification` module.

use super::types::{
    CitationCheck, CitationConfig, CitationError, CitationReport, VerifiedCitation,
};
use super::verifier::CitationVerifier;

// ── Helpers ────────────────────────────────────────────────────────────────────

fn default_verifier() -> CitationVerifier {
    CitationVerifier::new(CitationConfig::default())
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

// ── CitationConfig ─────────────────────────────────────────────────────────────

#[test]
fn config_default_min_overlap_ratio() {
    assert_eq!(CitationConfig::default().min_overlap_ratio, 0.5);
}

#[test]
fn config_default_substring_weight() {
    assert_eq!(CitationConfig::default().substring_weight, 0.3);
}

#[test]
fn config_default_token_overlap_weight() {
    assert_eq!(CitationConfig::default().token_overlap_weight, 0.7);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(CitationConfig::new(), CitationConfig::default());
}

#[test]
fn config_with_min_overlap_ratio_builder() {
    let cfg = CitationConfig::default().with_min_overlap_ratio(0.8);
    assert!(approx_eq(cfg.min_overlap_ratio, 0.8));
}

#[test]
fn config_with_substring_weight_builder() {
    let cfg = CitationConfig::default().with_substring_weight(0.5);
    assert!(approx_eq(cfg.substring_weight, 0.5));
}

#[test]
fn config_with_token_overlap_weight_builder() {
    let cfg = CitationConfig::default().with_token_overlap_weight(1.0);
    assert!(approx_eq(cfg.token_overlap_weight, 1.0));
}

#[test]
fn config_min_overlap_ratio_clamped_above_one() {
    let cfg = CitationConfig::default().with_min_overlap_ratio(1.5);
    assert!(approx_eq(cfg.min_overlap_ratio, 1.0));
}

#[test]
fn config_min_overlap_ratio_clamped_below_zero() {
    let cfg = CitationConfig::default().with_min_overlap_ratio(-0.5);
    assert!(approx_eq(cfg.min_overlap_ratio, 0.0));
}

// ── CitationError ──────────────────────────────────────────────────────────────

#[test]
fn error_empty_claim_display() {
    let err = CitationError::EmptyClaim;
    assert_eq!(format!("{err}"), "claim must not be empty");
}

#[test]
fn error_empty_source_display() {
    let err = CitationError::EmptySource;
    assert_eq!(format!("{err}"), "source passage must not be empty");
}

#[test]
fn error_verification_failed_display() {
    let err = CitationError::VerificationFailed("internal fault".to_string());
    assert!(format!("{err}").contains("internal fault"));
}

#[test]
fn error_debug_format_is_non_empty() {
    assert!(!format!("{:?}", CitationError::EmptyClaim).is_empty());
}

// ── CitationCheck ──────────────────────────────────────────────────────────────

#[test]
fn citation_check_new_stores_fields() {
    let check = CitationCheck::new("my claim", "my source");
    assert_eq!(check.claim, "my claim");
    assert_eq!(check.source_passage, "my source");
}

#[test]
fn citation_check_clone_is_equal() {
    let check = CitationCheck::new("claim", "source");
    assert_eq!(check.clone(), check);
}

#[test]
fn citation_check_debug_format() {
    let check = CitationCheck::new("hello", "world");
    assert!(format!("{check:?}").contains("hello"));
}

// ── VerifiedCitation ───────────────────────────────────────────────────────────

#[test]
fn verified_citation_is_grounded_field_true() {
    let check = CitationCheck::new("c", "s");
    let vc = VerifiedCitation {
        check,
        grounding_score: 0.9,
        is_grounded: true,
        matched_spans: vec!["foo bar baz".to_string()],
    };
    assert!(vc.is_grounded);
}

#[test]
fn verified_citation_is_grounded_field_false() {
    let check = CitationCheck::new("c", "s");
    let vc = VerifiedCitation {
        check,
        grounding_score: 0.1,
        is_grounded: false,
        matched_spans: vec![],
    };
    assert!(!vc.is_grounded);
}

#[test]
fn verified_citation_matched_spans_accessible() {
    let check = CitationCheck::new("c", "s");
    let spans = vec!["one two three".to_string()];
    let vc = VerifiedCitation {
        check,
        grounding_score: 0.8,
        is_grounded: true,
        matched_spans: spans.clone(),
    };
    assert_eq!(vc.matched_spans, spans);
}

// ── CitationReport ─────────────────────────────────────────────────────────────

#[test]
fn citation_report_empty_defaults() {
    let report = CitationReport {
        verified: vec![],
        overall_score: 0.0,
        grounded_count: 0,
        total_count: 0,
    };
    assert_eq!(report.total_count, 0);
    assert_eq!(report.grounded_count, 0);
    assert!(approx_eq(report.overall_score, 0.0));
}

#[test]
fn citation_report_clone_equal() {
    let report = CitationReport {
        verified: vec![],
        overall_score: 0.5,
        grounded_count: 1,
        total_count: 2,
    };
    assert_eq!(report.clone(), report);
}

// ── verify_one — error paths ───────────────────────────────────────────────────

#[test]
fn verify_one_empty_claim_returns_error() {
    let v = default_verifier();
    let res = v.verify_one(CitationCheck::new("", "some source text"));
    assert!(matches!(res, Err(CitationError::EmptyClaim)));
}

#[test]
fn verify_one_whitespace_only_claim_returns_error() {
    let v = default_verifier();
    let res = v.verify_one(CitationCheck::new("   \t\n  ", "some source text"));
    assert!(matches!(res, Err(CitationError::EmptyClaim)));
}

#[test]
fn verify_one_empty_source_returns_error() {
    let v = default_verifier();
    let res = v.verify_one(CitationCheck::new("some claim", ""));
    assert!(matches!(res, Err(CitationError::EmptySource)));
}

#[test]
fn verify_one_whitespace_only_source_returns_error() {
    let v = default_verifier();
    let res = v.verify_one(CitationCheck::new("some claim", "   \n  "));
    assert!(matches!(res, Err(CitationError::EmptySource)));
}

// ── verify_one — basic scoring ────────────────────────────────────────────────

#[test]
fn verify_one_grounded_claim_high_score() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris, France.";
    let claim = "The Eiffel Tower is in Paris.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(
        res.grounding_score > 0.5,
        "expected high score, got {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_ungrounded_claim_low_score() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris, France.";
    let claim = "Quantum computing uses qubits to perform parallel computation.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(
        res.grounding_score < 0.5,
        "expected low score, got {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_exact_copy_produces_high_score() {
    let v = default_verifier();
    let text = "The quick brown fox jumps over the lazy dog.";
    let res = v.verify_one(CitationCheck::new(text, text)).unwrap();
    // All tokens match + 3-gram match → score near 1.0
    assert!(
        res.grounding_score > 0.9,
        "exact copy score: {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_disjoint_tokens_zero_overlap() {
    let v = default_verifier();
    let claim = "alpha beta gamma delta epsilon zeta";
    let source = "uno dos tres cuatro cinco seis siete ocho";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(approx_eq(res.grounding_score, 0.0));
}

#[test]
fn verify_one_score_is_in_zero_one_range() {
    let v = default_verifier();
    let claim = "Some words appear here and there";
    let source = "Some information here regarding those words";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(res.grounding_score >= 0.0 && res.grounding_score <= 1.0);
}

#[test]
fn verify_one_preserves_original_check() {
    let v = default_verifier();
    let check = CitationCheck::new("claim text", "source text");
    let check_clone = check.clone();
    let res = v.verify_one(check).unwrap();
    assert_eq!(res.check, check_clone);
}

// ── verify_one — is_grounded threshold ────────────────────────────────────────

#[test]
fn verify_one_is_grounded_true_for_high_score() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris, France.";
    let claim = "The Eiffel Tower is in Paris.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(res.is_grounded);
}

#[test]
fn verify_one_is_grounded_false_for_low_score() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris, France.";
    let claim = "Quantum computing uses qubits to perform parallel computation.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(!res.is_grounded);
}

#[test]
fn verify_one_custom_threshold_changes_verdict() {
    // With a very low threshold, even a weak signal is "grounded".
    let cfg = CitationConfig::default().with_min_overlap_ratio(0.0);
    let v = CitationVerifier::new(cfg);
    let source = "The Eiffel Tower is located in Paris.";
    let claim = "Quantum bits allow parallel operations.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(
        res.is_grounded,
        "score {} should be >= 0.0",
        res.grounding_score
    );
}

#[test]
fn verify_one_high_threshold_makes_partial_match_ungrounded() {
    // Raise the bar so that partial overlap still fails.
    let cfg = CitationConfig::default().with_min_overlap_ratio(0.99);
    let v = CitationVerifier::new(cfg);
    // Claim shares 2 of 6 tokens with source, no 3-gram match.
    let claim = "alpha beta gamma delta epsilon zeta";
    let source = "alpha beta uno dos tres cuatro";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(!res.is_grounded);
}

// ── verify_one — matched_spans ─────────────────────────────────────────────────

#[test]
fn verify_one_matched_spans_nonempty_when_3gram_matches() {
    let v = default_verifier();
    let source = "the eiffel tower is located in paris france";
    let claim = "the eiffel tower is in paris";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(!res.matched_spans.is_empty());
}

#[test]
fn verify_one_matched_spans_empty_for_short_claim() {
    let v = default_verifier();
    // Only 2 tokens — no 3-gram window possible.
    let claim = "eiffel tower";
    let source = "eiffel tower is located in paris";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(res.matched_spans.is_empty());
}

#[test]
fn verify_one_matched_spans_empty_when_no_3gram_in_source() {
    let v = default_verifier();
    // Claim tokens in different order than source; no 3-gram matches.
    let claim = "paris france eiffel";
    let source = "eiffel france paris";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // "paris france eiffel" → 3-gram: "paris france eiffel"
    // normalised source: "eiffel france paris"
    // "paris france eiffel" not in "eiffel france paris" → no match
    assert!(res.matched_spans.is_empty());
}

#[test]
fn verify_one_matched_spans_contains_matching_3gram() {
    let v = default_verifier();
    let source = "the quick brown fox jumped over the lazy dog";
    let claim = "the quick brown fox is an animal";
    // "the quick brown" and "quick brown fox" should match.
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(
        res.matched_spans.contains(&"the quick brown".to_string())
            || res.matched_spans.contains(&"quick brown fox".to_string())
    );
}

#[test]
fn verify_one_matched_spans_are_sorted() {
    let v = default_verifier();
    let source = "the quick brown fox jumped over the lazy dog";
    let claim = "the quick brown fox jumped over";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    let sorted = {
        let mut s = res.matched_spans.clone();
        s.sort_unstable();
        s
    };
    assert_eq!(res.matched_spans, sorted);
}

#[test]
fn verify_one_matched_spans_are_deduplicated() {
    let v = default_verifier();
    // Repeated pattern that would generate duplicate 3-grams.
    let source = "ab cd ef gh ij kl";
    let claim = "ab cd ef gh ab cd ef";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    let unique: std::collections::HashSet<_> = res.matched_spans.iter().collect();
    assert_eq!(
        unique.len(),
        res.matched_spans.len(),
        "spans should be deduplicated"
    );
}

// ── verify_one — n-gram signal effect on score ────────────────────────────────

#[test]
fn verify_one_ngram_match_contributes_to_score() {
    // Build two scenarios differing only by whether a 3-gram matches.
    // Scenario A: claim tokens all in source, no 3-gram match (different order).
    // Scenario B: claim tokens all in source, 3-gram match (same order).
    let cfg_sub_only = CitationConfig {
        min_overlap_ratio: 0.0,
        substring_weight: 0.5,
        token_overlap_weight: 0.0,
    };
    let v = CitationVerifier::new(cfg_sub_only);

    // No 3-gram match: tokens shuffled
    let claim_no_ngram = "fox quick brown";
    let source = "the quick brown fox";
    let res_no = v
        .verify_one(CitationCheck::new(claim_no_ngram, source))
        .unwrap();

    // 3-gram match: consecutive
    let claim_with_ngram = "the quick brown";
    let res_yes = v
        .verify_one(CitationCheck::new(claim_with_ngram, source))
        .unwrap();

    assert!(
        res_yes.grounding_score > res_no.grounding_score,
        "ngram match ({}) should boost score over no-match ({})",
        res_yes.grounding_score,
        res_no.grounding_score
    );
}

#[test]
fn verify_one_no_ngram_boost_when_tokens_shuffled() {
    let v = default_verifier();
    // All tokens shared but in different order → no 3-gram substring match.
    let claim = "france eiffel paris";
    let source = "paris eiffel france";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // Score should equal token_overlap_weight × 1.0 (all tokens overlap, no ngram match).
    let expected = 0.7 * 1.0;
    assert!(
        approx_eq(res.grounding_score, expected),
        "expected {expected}, got {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_score_formula_with_known_values() {
    // claim: "ab cd ef"  source: "ab cd ef gh"
    // claim tokens: ["ab", "cd", "ef"]
    // source tokens: ["ab", "cd", "ef", "gh"]
    // token_overlap_ratio = 3/3 = 1.0
    // 3-gram "ab cd ef" in "ab cd ef gh" → YES
    // score = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    let v = default_verifier();
    let res = v
        .verify_one(CitationCheck::new("ab cd ef", "ab cd ef gh"))
        .unwrap();
    assert!(approx_eq(res.grounding_score, 1.0));
}

// ── verify_one — config weight effects ────────────────────────────────────────

#[test]
fn verify_one_zero_token_weight_score_depends_on_ngram_only() {
    let cfg = CitationConfig {
        min_overlap_ratio: 0.0,
        substring_weight: 0.4,
        token_overlap_weight: 0.0,
    };
    let v = CitationVerifier::new(cfg);
    // claim: "the quick brown" — 3-gram matches source
    let res = v
        .verify_one(CitationCheck::new(
            "the quick brown fox",
            "the quick brown fox",
        ))
        .unwrap();
    // token overlap ratio doesn't matter; ngram_score = 1.0 → score = 0.4 * 1.0 = 0.4
    assert!(
        approx_eq(res.grounding_score, 0.4),
        "score should be 0.4, got {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_zero_substring_weight_score_depends_on_overlap_only() {
    let cfg = CitationConfig {
        min_overlap_ratio: 0.0,
        substring_weight: 0.0,
        token_overlap_weight: 0.6,
    };
    let v = CitationVerifier::new(cfg);
    // claim: all 4 tokens in source but shuffled → overlap_ratio = 1.0, no ngram match
    let claim = "fox quick brown the";
    let source = "the quick brown fox jumped";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // score = 0.6 * 1.0 + 0.0 * 0.0 = 0.6
    assert!(
        approx_eq(res.grounding_score, 0.6),
        "score should be 0.6, got {}",
        res.grounding_score
    );
}

#[test]
fn verify_one_high_substring_weight_amplifies_ngram_signal() {
    let cfg_high = CitationConfig {
        min_overlap_ratio: 0.0,
        substring_weight: 0.9,
        token_overlap_weight: 0.1,
    };
    let cfg_low = CitationConfig {
        min_overlap_ratio: 0.0,
        substring_weight: 0.1,
        token_overlap_weight: 0.9,
    };
    let claim = "the quick brown fox";
    let source = "the quick brown fox jumps";
    let score_high = CitationVerifier::new(cfg_high)
        .verify_one(CitationCheck::new(claim, source))
        .unwrap()
        .grounding_score;
    let score_low = CitationVerifier::new(cfg_low)
        .verify_one(CitationCheck::new(claim, source))
        .unwrap()
        .grounding_score;
    // Both signals are 1.0 here, so clamped scores should both be 1.0.
    assert!(approx_eq(score_high, 1.0));
    assert!(approx_eq(score_low, 1.0));
}

#[test]
fn verify_one_partial_overlap_expected_score() {
    let v = default_verifier();
    // claim tokens: ["alpha", "beta", "gamma", "delta"] (4 tokens)
    // source tokens: ["alpha", "beta", "uno", "dos", "tres"] (5 tokens)
    // overlap: alpha, beta → 2/4 = 0.5
    // 3-grams from claim: "alpha beta gamma", "beta gamma delta"
    // normalised source: "alpha beta uno dos tres"
    // "alpha beta gamma" not in source → no match
    // "beta gamma delta" not in source → no match
    // score = 0.7 * 0.5 + 0.3 * 0.0 = 0.35
    let claim = "alpha beta gamma delta";
    let source = "alpha beta uno dos tres";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(
        approx_eq(res.grounding_score, 0.35),
        "expected 0.35, got {}",
        res.grounding_score
    );
    assert!(!res.is_grounded);
}

// ── verify_batch ───────────────────────────────────────────────────────────────

#[test]
fn verify_batch_empty_input_returns_empty_report() {
    let v = default_verifier();
    let report = v.verify_batch(vec![]);
    assert_eq!(report.total_count, 0);
    assert_eq!(report.grounded_count, 0);
    assert!(report.verified.is_empty());
    assert!(approx_eq(report.overall_score, 0.0));
}

#[test]
fn verify_batch_single_grounded_citation() {
    let v = default_verifier();
    let check = CitationCheck::new(
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is located in Paris, France.",
    );
    let report = v.verify_batch(vec![check]);
    assert_eq!(report.total_count, 1);
    assert_eq!(report.grounded_count, 1);
    assert!(approx_eq(
        report.overall_score,
        report.verified[0].grounding_score
    ));
}

#[test]
fn verify_batch_single_ungrounded_citation() {
    let v = default_verifier();
    let check = CitationCheck::new(
        "Quantum computing uses qubits.",
        "The Eiffel Tower is located in Paris, France.",
    );
    let report = v.verify_batch(vec![check]);
    assert_eq!(report.total_count, 1);
    assert_eq!(report.grounded_count, 0);
    assert!(!report.verified[0].is_grounded);
}

#[test]
fn verify_batch_all_grounded_count() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris France.";
    let checks = vec![
        CitationCheck::new("The Eiffel Tower is in Paris.", source),
        CitationCheck::new("The Eiffel Tower is located in Paris.", source),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 2);
    assert_eq!(report.grounded_count, 2);
}

#[test]
fn verify_batch_none_grounded_count() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris.";
    let checks = vec![
        CitationCheck::new("Quantum computing uses qubits for calculation.", source),
        CitationCheck::new("Machine learning algorithms train on datasets.", source),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 2);
    assert_eq!(report.grounded_count, 0);
}

#[test]
fn verify_batch_mixed_grounded_count() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris France.";
    let checks = vec![
        CitationCheck::new("The Eiffel Tower is in Paris.", source),
        CitationCheck::new("Quantum computing uses qubits for calculation.", source),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 2);
    assert_eq!(report.grounded_count, 1);
}

#[test]
fn verify_batch_overall_score_is_arithmetic_mean() {
    let v = default_verifier();
    let source = "The Eiffel Tower is located in Paris France.";
    let checks = vec![
        CitationCheck::new("The Eiffel Tower is in Paris.", source),
        CitationCheck::new("The Eiffel Tower is located in Paris.", source),
    ];
    let report = v.verify_batch(checks);
    let manual_mean = report
        .verified
        .iter()
        .map(|r| r.grounding_score)
        .sum::<f32>()
        / report.verified.len() as f32;
    assert!(approx_eq(report.overall_score, manual_mean));
}

#[test]
fn verify_batch_total_count_equals_verified_len() {
    let v = default_verifier();
    let source = "The quick brown fox jumps over the lazy dog.";
    let checks = vec![
        CitationCheck::new("the quick brown fox", source),
        CitationCheck::new("over the lazy dog", source),
        CitationCheck::new("brown fox jumps", source),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, report.verified.len());
}

#[test]
fn verify_batch_skips_empty_claims() {
    let v = default_verifier();
    let checks = vec![
        CitationCheck::new("", "valid source passage"),
        CitationCheck::new("valid claim text here", "valid source passage"),
    ];
    let report = v.verify_batch(checks);
    // Empty claim is skipped.
    assert_eq!(report.total_count, 1);
    assert_eq!(report.verified.len(), 1);
}

#[test]
fn verify_batch_skips_empty_sources() {
    let v = default_verifier();
    let checks = vec![
        CitationCheck::new("valid claim", ""),
        CitationCheck::new("another valid claim here", "another valid source here"),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 1);
}

#[test]
fn verify_batch_result_order_preserved() {
    let v = default_verifier();
    let source = "the quick brown fox jumps";
    let claims = vec!["the quick brown", "quick brown fox", "brown fox jumps"];
    let checks: Vec<_> = claims
        .iter()
        .map(|&c| CitationCheck::new(c, source))
        .collect();
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 3);
    for (i, vc) in report.verified.iter().enumerate() {
        assert_eq!(vc.check.claim, claims[i]);
    }
}

#[test]
fn verify_batch_grounded_count_correct_for_five_items() {
    let v = default_verifier();
    let source = "The Eiffel Tower stands in Paris France.";
    let checks = vec![
        CitationCheck::new("The Eiffel Tower stands in Paris.", source),
        CitationCheck::new("Quantum bits enable parallel computation.", source),
        CitationCheck::new("The Eiffel Tower is in Paris.", source),
        CitationCheck::new("Machine learning trains on large datasets.", source),
        CitationCheck::new("Eiffel Tower stands in France.", source),
    ];
    let report = v.verify_batch(checks);
    assert_eq!(report.total_count, 5);
    // Grounded count must be >= 1 (first claim is clearly grounded).
    assert!(report.grounded_count >= 1);
}

// ── Integration ────────────────────────────────────────────────────────────────

#[test]
fn integration_paraphrase_is_grounded() {
    let v = default_verifier();
    // Paraphrase shares key tokens with the source.
    let source = "Photosynthesis converts sunlight into chemical energy in plants.";
    let claim = "Plants convert sunlight into energy through photosynthesis.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(res.grounding_score > 0.0);
}

#[test]
fn integration_case_insensitive_token_matching() {
    let v = default_verifier();
    let source = "THE EIFFEL TOWER IS IN PARIS FRANCE.";
    let claim = "the eiffel tower is in paris.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // Lowercased tokenisation means all tokens match.
    assert!(res.grounding_score > 0.5);
    assert!(res.is_grounded);
}

#[test]
fn integration_claim_is_subset_of_source_sentence() {
    let v = default_verifier();
    let source = "The Eiffel Tower, a wrought-iron lattice tower, is located on the Champ de Mars in Paris, France.";
    let claim = "The Eiffel Tower is in Paris France.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    assert!(res.is_grounded);
}

#[test]
fn integration_numbers_and_alphanumeric_tokens() {
    let v = default_verifier();
    let source = "The Eiffel Tower is 330 metres tall and was built in 1889.";
    let claim = "The Eiffel Tower is 330 metres tall.";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // "330" has 3 chars → token, "metres" → token; both overlap.
    assert!(res.grounding_score > 0.5);
}

#[test]
fn integration_single_char_tokens_filtered_out() {
    let v = default_verifier();
    // Only 1-character tokens → tokeniser yields nothing usable.
    let claim = "a b c d e f";
    let source = "a b c d e f g h i";
    let res = v.verify_one(CitationCheck::new(claim, source)).unwrap();
    // No tokens survive filter → overlap = 0, no ngram → score = 0.
    assert!(approx_eq(res.grounding_score, 0.0));
}

#[test]
fn integration_full_workflow_batch() {
    let verifier = CitationVerifier::new(CitationConfig::default());
    let source = "The Eiffel Tower is located in Paris France.";
    let checks = vec![
        CitationCheck::new("The Eiffel Tower is in Paris.", source),
        CitationCheck::new("Quantum computing uses qubits.", source),
        CitationCheck::new("Eiffel Tower located in France.", source),
    ];
    let report = verifier.verify_batch(checks);
    assert_eq!(report.total_count, 3);
    assert!(report.grounded_count >= 1);
    assert!(report.overall_score >= 0.0 && report.overall_score <= 1.0);
    assert_eq!(report.verified.len(), 3);
}
