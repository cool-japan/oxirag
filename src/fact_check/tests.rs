#![allow(clippy::float_cmp, clippy::similar_names)]
//! Tests for the `fact_check` module.

use crate::fact_check::{
    Evidence, FactCheckConfig, FactCheckError, FactCheckResult, FactChecker, Verdict,
};
use crate::types::{Document, DocumentId};

// ── helpers ─────────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

// ── Verdict ─────────────────────────────────────────────────────────────────────

#[test]
fn test_verdict_as_str_supports() {
    assert_eq!(Verdict::Supports.as_str(), "SUPPORTS");
}

#[test]
fn test_verdict_as_str_refutes() {
    assert_eq!(Verdict::Refutes.as_str(), "REFUTES");
}

#[test]
fn test_verdict_as_str_nei() {
    assert_eq!(Verdict::NotEnoughInfo.as_str(), "NOT_ENOUGH_INFO");
}

#[test]
fn test_verdict_equality() {
    assert_eq!(Verdict::Supports, Verdict::Supports);
    assert_ne!(Verdict::Supports, Verdict::Refutes);
    assert_ne!(Verdict::Refutes, Verdict::NotEnoughInfo);
}

// ── Evidence ────────────────────────────────────────────────────────────────────

#[test]
fn test_evidence_new() {
    let e = Evidence::new(DocumentId::from_string("d1"), "a sentence", 0.5, false);
    assert_eq!(e.doc_id.as_str(), "d1");
    assert_eq!(e.sentence, "a sentence");
    assert_eq!(e.score, 0.5);
    assert!(!e.contradicts);
}

#[test]
fn test_evidence_clone() {
    let e = Evidence::new(DocumentId::from_string("d1"), "s", 0.3, true);
    let c = e.clone();
    assert_eq!(c.sentence, e.sentence);
    assert_eq!(c.contradicts, e.contradicts);
}

// ── FactCheckConfig ─────────────────────────────────────────────────────────────

#[test]
fn test_config_default() {
    let cfg = FactCheckConfig::default();
    assert_eq!(cfg.evidence_top_k, 3);
    assert_eq!(cfg.support_threshold, 0.4);
    assert_eq!(cfg.nei_threshold, 0.15);
}

#[test]
fn test_config_new_matches_default() {
    let a = FactCheckConfig::new();
    let b = FactCheckConfig::default();
    assert_eq!(a.evidence_top_k, b.evidence_top_k);
    assert_eq!(a.support_threshold, b.support_threshold);
    assert_eq!(a.nei_threshold, b.nei_threshold);
}

#[test]
fn test_config_builders_chain() {
    let cfg = FactCheckConfig::new()
        .with_evidence_top_k(5)
        .with_support_threshold(0.5)
        .with_nei_threshold(0.1);
    assert_eq!(cfg.evidence_top_k, 5);
    assert_eq!(cfg.support_threshold, 0.5);
    assert_eq!(cfg.nei_threshold, 0.1);
}

// ── FactCheckResult ─────────────────────────────────────────────────────────────

#[test]
fn test_result_new() {
    let r = FactCheckResult::new(Verdict::Supports, 0.8, vec![], "because");
    assert_eq!(r.verdict, Verdict::Supports);
    assert_eq!(r.confidence, 0.8);
    assert!(r.evidence.is_empty());
    assert_eq!(r.rationale, "because");
}

#[test]
fn test_result_is_supported() {
    let r = FactCheckResult::new(Verdict::Supports, 0.8, vec![], "x");
    assert!(r.is_supported());
    assert!(!r.is_refuted());
}

#[test]
fn test_result_is_refuted() {
    let r = FactCheckResult::new(Verdict::Refutes, 0.8, vec![], "x");
    assert!(r.is_refuted());
    assert!(!r.is_supported());
}

// ── FactChecker construction ────────────────────────────────────────────────────

#[test]
fn test_checker_new() {
    let checker = FactChecker::new(FactCheckConfig::new().with_evidence_top_k(2));
    assert_eq!(checker.config.evidence_top_k, 2);
}

#[test]
fn test_checker_default() {
    let checker = FactChecker::default();
    assert_eq!(checker.config.evidence_top_k, 3);
}

#[test]
fn test_checker_clone() {
    let checker = FactChecker::default();
    let c = checker.clone();
    assert_eq!(c.config.support_threshold, 0.4);
}

// ── verify: Supports ────────────────────────────────────────────────────────────

#[test]
fn test_verify_supports_high_overlap() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "The Eiffel Tower is located in Paris, France.")];
    let result = checker
        .verify("The Eiffel Tower is located in Paris.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
}

#[test]
fn test_verify_supports_picks_best_sentence() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Cats are mammals. The capital of Japan is Tokyo. Birds can fly.",
    )];
    let result = checker
        .verify("The capital of Japan is Tokyo.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
    assert!(result.evidence[0].sentence.contains("Tokyo"));
}

#[test]
fn test_verify_supports_confidence_range() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Mount Everest is the tallest mountain on Earth.")];
    let result = checker
        .verify("Mount Everest is the tallest mountain on Earth.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
    assert!(result.confidence >= 0.5);
    assert!(result.confidence <= 1.0);
}

#[test]
fn test_verify_supports_matching_numbers_not_refuted() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Marie Curie was born in 1867 in Warsaw.")];
    let result = checker
        .verify("Marie Curie was born in 1867.", &docs)
        .unwrap();
    // Same number 1867 present in both → not a number mismatch → Supports.
    assert_eq!(result.verdict, Verdict::Supports);
}

// ── verify: Refutes (negation) ──────────────────────────────────────────────────

#[test]
fn test_verify_refutes_negation_mismatch() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The Great Wall of China is not visible from the Moon with the naked eye.",
    )];
    let result = checker
        .verify(
            "The Great Wall of China is visible from the Moon with the naked eye.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
}

#[test]
fn test_verify_refutes_negation_contraction() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Penguins aren't able to fly through the air.")];
    let result = checker
        .verify("Penguins are able to fly through the air.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
}

#[test]
fn test_verify_refutes_confidence_above_half() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Sound does not travel through the vacuum of empty outer space.",
    )];
    let result = checker
        .verify(
            "Sound does travel through the vacuum of empty outer space.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
    assert!(result.confidence >= 0.5);
    assert!(result.confidence <= 1.0);
}

// ── verify: Refutes (number mismatch) ───────────────────────────────────────────

#[test]
fn test_verify_refutes_number_mismatch_years() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "George Washington was born in 1732 in Virginia colony.",
    )];
    let result = checker
        .verify(
            "George Washington was born in 1799 in Virginia colony.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
}

#[test]
fn test_verify_refutes_number_mismatch_simple() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The committee elected 1900 new members this year.",
    )];
    let result = checker
        .verify("The committee elected 1950 new members this year.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
}

#[test]
fn test_verify_refutes_born_in_year_mismatch() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The famous composer was born in 1900 according to records.",
    )];
    let result = checker
        .verify(
            "The famous composer was born in 1950 according to records.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
}

// ── verify: NotEnoughInfo ───────────────────────────────────────────────────────

#[test]
fn test_verify_nei_unrelated_corpus() {
    let checker = FactChecker::default();
    let docs = vec![
        doc("d1", "Bananas are a popular tropical fruit."),
        doc("d2", "The orchestra performed a symphony last night."),
    ];
    let result = checker
        .verify("Quantum entanglement links distant particles.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
}

#[test]
fn test_verify_nei_empty_overlap() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Zebra giraffe elephant rhino hippo.")];
    let result = checker
        .verify(
            "Photosynthesis converts sunlight into chemical energy.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
}

#[test]
fn test_verify_nei_weak_partial_overlap() {
    // Tune thresholds so a partial-overlap claim lands between nei and support.
    let cfg = FactCheckConfig::new()
        .with_support_threshold(0.9)
        .with_nei_threshold(0.05);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc(
        "d1",
        "The river flows gently past the ancient stone bridge.",
    )];
    let result = checker
        .verify("The river is wide and deep near the bridge.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
}

#[test]
fn test_verify_nei_confidence_range() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Completely unrelated content about gardening tools.",
    )];
    let result = checker
        .verify("Neural networks approximate nonlinear functions.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
}

// ── retrieve_evidence ───────────────────────────────────────────────────────────

#[test]
fn test_retrieve_ranks_by_overlap() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Apples grow on trees. The capital city of France is Paris and it is beautiful. Dogs bark loudly.",
    )];
    let evidence = checker.retrieve_evidence("The capital city of France is Paris.", &docs);
    assert!(!evidence.is_empty());
    assert!(evidence[0].sentence.contains("Paris"));
    // Scores must be in non-increasing order.
    for w in evidence.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn test_retrieve_caps_at_top_k() {
    let cfg = FactCheckConfig::new().with_evidence_top_k(2);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc(
        "d1",
        "Alpha one. Beta two. Gamma three. Delta four. Epsilon five.",
    )];
    let evidence = checker.retrieve_evidence("one two three four five", &docs);
    assert_eq!(evidence.len(), 2);
}

#[test]
fn test_retrieve_across_multiple_docs() {
    let checker = FactChecker::default();
    let docs = vec![
        doc("d1", "The moon orbits the earth every month."),
        doc("d2", "The sun is a star at the center of the solar system."),
    ];
    let evidence = checker.retrieve_evidence("The sun is a star.", &docs);
    assert!(!evidence.is_empty());
    assert_eq!(evidence[0].doc_id.as_str(), "d2");
}

#[test]
fn test_retrieve_preserves_doc_id() {
    let checker = FactChecker::default();
    let docs = vec![doc("special-id-42", "Octopuses have three hearts.")];
    let evidence = checker.retrieve_evidence("Octopuses have three hearts.", &docs);
    assert_eq!(evidence[0].doc_id.as_str(), "special-id-42");
}

#[test]
fn test_retrieve_empty_when_no_sentences() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "   ")];
    let evidence = checker.retrieve_evidence("anything here", &docs);
    assert!(evidence.is_empty());
}

#[test]
fn test_retrieve_top_k_larger_than_sentences() {
    let cfg = FactCheckConfig::new().with_evidence_top_k(10);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc("d1", "Only one sentence exists here.")];
    let evidence = checker.retrieve_evidence("only one sentence", &docs);
    assert_eq!(evidence.len(), 1);
}

// ── contradicts flag ────────────────────────────────────────────────────────────

#[test]
fn test_contradicts_flag_set_negation() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The medication is not effective against the viral infection.",
    )];
    let evidence = checker.retrieve_evidence(
        "The medication is effective against the viral infection.",
        &docs,
    );
    assert!(evidence[0].contradicts);
}

#[test]
fn test_contradicts_flag_set_number_mismatch() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "The marathon distance is 42 kilometers long.")];
    let evidence = checker.retrieve_evidence("The marathon distance is 50 kilometers long.", &docs);
    assert!(evidence[0].contradicts);
}

#[test]
fn test_contradicts_flag_clear_when_agreeing() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The recipe requires three cups of flour exactly.",
    )];
    let evidence = checker.retrieve_evidence("The recipe requires three cups of flour.", &docs);
    assert!(!evidence[0].contradicts);
}

#[test]
fn test_contradicts_flag_clear_when_unrelated() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "No penguins live in the hot tropical desert.")];
    // Claim shares no content tokens with the negated sentence.
    let evidence = checker.retrieve_evidence("Computers process binary data.", &docs);
    assert!(!evidence[0].contradicts);
}

#[test]
fn test_contradicts_clear_same_polarity_no_numbers() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "The library opens early in the morning hours.")];
    let evidence = checker.retrieve_evidence("The library opens early in the morning.", &docs);
    assert!(!evidence[0].contradicts);
}

// ── confidence bounds across verdicts ───────────────────────────────────────────

#[test]
fn test_confidence_always_in_unit_interval() {
    let checker = FactChecker::default();
    let cases = vec![
        (
            "The Earth orbits the Sun once per year.",
            "The Earth orbits the Sun once per year.",
        ),
        (
            "Iron is not a metal element on the table.",
            "Iron is a metal element on the table.",
        ),
        (
            "The tower has 100 floors above ground.",
            "The tower has 200 floors above ground.",
        ),
        (
            "Random unrelated claim about nothing.",
            "Totally different topic entirely here.",
        ),
    ];
    for (evi, claim) in cases {
        let docs = vec![doc("d1", evi)];
        let result = checker.verify(claim, &docs).unwrap();
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "confidence {} out of range for claim {claim}",
            result.confidence
        );
    }
}

// ── error handling ──────────────────────────────────────────────────────────────

#[test]
fn test_verify_empty_claim_error() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Some content here.")];
    let err = checker.verify("", &docs).unwrap_err();
    assert!(matches!(err, FactCheckError::EmptyClaim));
}

#[test]
fn test_verify_whitespace_claim_error() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Some content here.")];
    let err = checker.verify("    \n\t  ", &docs).unwrap_err();
    assert!(matches!(err, FactCheckError::EmptyClaim));
}

#[test]
fn test_verify_empty_corpus_error() {
    let checker = FactChecker::default();
    let docs: Vec<Document> = vec![];
    let err = checker.verify("A valid claim.", &docs).unwrap_err();
    assert!(matches!(err, FactCheckError::EmptyCorpus));
}

#[test]
fn test_empty_claim_error_message() {
    let err = FactCheckError::EmptyClaim;
    assert_eq!(err.to_string(), "claim must not be empty");
}

#[test]
fn test_empty_corpus_error_message() {
    let err = FactCheckError::EmptyCorpus;
    assert_eq!(err.to_string(), "evidence corpus is empty");
}

// ── rationale ───────────────────────────────────────────────────────────────────

#[test]
fn test_rationale_non_empty_supports() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The Pacific Ocean is the largest ocean on Earth.",
    )];
    let result = checker
        .verify("The Pacific Ocean is the largest ocean on Earth.", &docs)
        .unwrap();
    assert!(!result.rationale.is_empty());
}

#[test]
fn test_rationale_non_empty_refutes() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "The bridge is not safe for heavy vehicles today.",
    )];
    let result = checker
        .verify("The bridge is safe for heavy vehicles today.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
    assert!(!result.rationale.is_empty());
}

#[test]
fn test_rationale_non_empty_nei() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "Unrelated text about cooking pasta dishes.")];
    let result = checker
        .verify("The stock market closed higher today.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
    assert!(!result.rationale.is_empty());
}

#[test]
fn test_rationale_mentions_verdict_context() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Honey never spoils when stored properly in jars.",
    )];
    let result = checker
        .verify("Honey does spoil when stored properly in jars.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Refutes);
    assert!(result.rationale.to_lowercase().contains("refut"));
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn test_determinism_verify_repeatable() {
    let checker = FactChecker::default();
    let docs = vec![
        doc(
            "d1",
            "The speed of light is approximately 300000 kilometers per second.",
        ),
        doc(
            "d2",
            "Light does not travel instantaneously across the universe.",
        ),
    ];
    let claim = "The speed of light is approximately 300000 kilometers per second.";
    let r1 = checker.verify(claim, &docs).unwrap();
    let r2 = checker.verify(claim, &docs).unwrap();
    assert_eq!(r1.verdict, r2.verdict);
    assert_eq!(r1.confidence, r2.confidence);
    assert_eq!(r1.rationale, r2.rationale);
    assert_eq!(r1.evidence.len(), r2.evidence.len());
}

#[test]
fn test_determinism_retrieve_repeatable() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "First fact here. Second fact here. Third fact here.",
    )];
    let e1 = checker.retrieve_evidence("second fact here", &docs);
    let e2 = checker.retrieve_evidence("second fact here", &docs);
    assert_eq!(e1.len(), e2.len());
    for (a, b) in e1.iter().zip(e2.iter()) {
        assert_eq!(a.sentence, b.sentence);
        assert_eq!(a.score, b.score);
        assert_eq!(a.contradicts, b.contradicts);
    }
}

#[test]
fn test_determinism_tie_ordering_stable() {
    let checker = FactChecker::default();
    // Two sentences with identical overlap; order must be stable by text.
    let docs = vec![doc("d1", "zebra runs fast. zebra runs fast slowly.")];
    let e1 = checker.retrieve_evidence("zebra runs", &docs);
    let e2 = checker.retrieve_evidence("zebra runs", &docs);
    let order1: Vec<&str> = e1.iter().map(|e| e.sentence.as_str()).collect();
    let order2: Vec<&str> = e2.iter().map(|e| e.sentence.as_str()).collect();
    assert_eq!(order1, order2);
}

// ── threshold sensitivity ───────────────────────────────────────────────────────

#[test]
fn test_high_support_threshold_downgrades_to_nei() {
    let cfg = FactCheckConfig::new()
        .with_support_threshold(0.99)
        .with_nei_threshold(0.01);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc(
        "d1",
        "The conference will be held in the large downtown hall.",
    )];
    // Partial overlap below 0.99 → not supported, above 0.01 → not nei-floor.
    let result = checker
        .verify(
            "The conference is held in the downtown hall building.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
}

#[test]
fn test_low_support_threshold_promotes_to_supports() {
    let cfg = FactCheckConfig::new().with_support_threshold(0.1);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc(
        "d1",
        "The annual festival attracts many visitors from neighboring towns.",
    )];
    let result = checker
        .verify("The festival attracts visitors.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
}

#[test]
fn test_refutes_requires_strong_overlap() {
    // A contradicting sentence with weak overlap should NOT trigger Refutes.
    let cfg = FactCheckConfig::new().with_support_threshold(0.95);
    let checker = FactChecker::new(cfg);
    let docs = vec![doc(
        "d1",
        "The experimental treatment did not improve outcomes in the small trial cohort.",
    )];
    // Overlap is moderate but below 0.95, so no Refutes despite contradiction.
    let result = checker
        .verify("The experimental treatment improved outcomes.", &docs)
        .unwrap();
    assert_ne!(result.verdict, Verdict::Refutes);
}

// ── multi-sentence document handling ────────────────────────────────────────────

#[test]
fn test_verify_with_punctuation_variety() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Is the sky blue? Yes, the sky appears blue due to scattering! Indeed it is blue.",
    )];
    let result = checker
        .verify("The sky appears blue due to scattering.", &docs)
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
}

#[test]
fn test_verify_evidence_is_populated() {
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "The library has many books on history.")];
    let result = checker
        .verify("The library has many books on history.", &docs)
        .unwrap();
    assert!(!result.evidence.is_empty());
    assert!(result.evidence[0].score > 0.0);
}

#[test]
fn test_verify_supports_with_extra_evidence_sentences() {
    let checker = FactChecker::default();
    let docs = vec![doc(
        "d1",
        "Random intro. The Amazon rainforest produces a large share of the world oxygen supply. Random outro.",
    )];
    let result = checker
        .verify(
            "The Amazon rainforest produces a large share of the world oxygen supply.",
            &docs,
        )
        .unwrap();
    assert_eq!(result.verdict, Verdict::Supports);
    // Best evidence is the matching middle sentence.
    assert!(result.evidence[0].sentence.contains("Amazon"));
}

#[test]
fn test_tokenizer_ignores_short_tokens() {
    // Single-char tokens (len < 2) are dropped, so "a" / "i" do not inflate overlap.
    let checker = FactChecker::default();
    let docs = vec![doc("d1", "a a a a a i i i o o o")];
    let result = checker.verify("a i o", &docs).unwrap();
    // No tokens of length >= 2 in the claim → zero overlap → NEI.
    assert_eq!(result.verdict, Verdict::NotEnoughInfo);
}
