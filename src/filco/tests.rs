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
    clippy::too_many_lines
)]

//! Tests for the `filco` module.

use std::collections::HashSet;

use super::filter::FilcoFilter;
use super::measures::{
    cxmi_lite_score, document_frequency, idf_weighted_relevance, jaccard, lexical_overlap_score,
    split_sentences, strinc_heuristic_score, strinc_strict_score, token_set, tokenize,
};
use super::types::{FilcoConfig, FilcoError, FilcoReport, FilterMeasure, ScoredSentence};

// ── Test helpers ────────────────────────────────────────────────────────────────

fn default_filter() -> FilcoFilter {
    FilcoFilter::new(FilcoConfig::default())
}

/// Tokenise `query` both as an ordered vector and as a lookup set, mirroring
/// what `FilcoFilter::run` precomputes before calling
/// [`strinc_heuristic_score`].
fn query_parts(query: &str) -> (Vec<String>, HashSet<String>) {
    let ordered = tokenize(query);
    let set: HashSet<String> = ordered.iter().cloned().collect();
    (ordered, set)
}

// ── FilterMeasure ─────────────────────────────────────────────────────────────

#[test]
fn measure_default_is_strinc() {
    assert_eq!(FilterMeasure::default(), FilterMeasure::StrInc);
}

#[test]
fn measure_as_str_values() {
    assert_eq!(FilterMeasure::StrInc.as_str(), "strinc");
    assert_eq!(FilterMeasure::LexicalOverlap.as_str(), "lexical_overlap");
    assert_eq!(FilterMeasure::CxmiLite.as_str(), "cxmi_lite");
}

#[test]
fn measure_display_matches_as_str() {
    for m in [
        FilterMeasure::StrInc,
        FilterMeasure::LexicalOverlap,
        FilterMeasure::CxmiLite,
    ] {
        assert_eq!(format!("{m}"), m.as_str());
    }
}

#[test]
fn measure_equality_and_copy() {
    let a = FilterMeasure::LexicalOverlap;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(FilterMeasure::StrInc, FilterMeasure::CxmiLite);
}

// ── FilcoConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = FilcoConfig::default();
    assert_eq!(cfg.lexical_threshold, 0.15);
    assert_eq!(cfg.cxmi_threshold, 0.05);
    assert_eq!(cfg.strinc_min_ngram, 2);
    assert_eq!(cfg.measure, FilterMeasure::StrInc);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(FilcoConfig::new(), FilcoConfig::default());
}

#[test]
fn config_builder_lexical_threshold() {
    let cfg = FilcoConfig::new().with_lexical_threshold(0.3);
    assert_eq!(cfg.lexical_threshold, 0.3);
}

#[test]
fn config_builder_cxmi_threshold() {
    let cfg = FilcoConfig::new().with_cxmi_threshold(0.2);
    assert_eq!(cfg.cxmi_threshold, 0.2);
}

#[test]
fn config_builder_strinc_min_ngram() {
    let cfg = FilcoConfig::new().with_strinc_min_ngram(3);
    assert_eq!(cfg.strinc_min_ngram, 3);
}

#[test]
fn config_builder_measure() {
    let cfg = FilcoConfig::new().with_measure(FilterMeasure::CxmiLite);
    assert_eq!(cfg.measure, FilterMeasure::CxmiLite);
}

#[test]
fn config_builder_chain() {
    let cfg = FilcoConfig::new()
        .with_lexical_threshold(0.2)
        .with_cxmi_threshold(0.1)
        .with_strinc_min_ngram(4)
        .with_measure(FilterMeasure::LexicalOverlap);
    assert_eq!(cfg.lexical_threshold, 0.2);
    assert_eq!(cfg.cxmi_threshold, 0.1);
    assert_eq!(cfg.strinc_min_ngram, 4);
    assert_eq!(cfg.measure, FilterMeasure::LexicalOverlap);
}

#[test]
fn config_clone_eq() {
    let cfg = FilcoConfig::new().with_lexical_threshold(0.4);
    assert_eq!(cfg.clone(), cfg);
}

#[test]
fn config_validate_default_ok() {
    assert!(FilcoConfig::default().validate().is_ok());
}

#[test]
fn config_validate_rejects_negative_lexical_threshold() {
    let cfg = FilcoConfig::new().with_lexical_threshold(-0.1);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_lexical_threshold_above_one() {
    let cfg = FilcoConfig::new().with_lexical_threshold(1.1);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_nan_lexical_threshold() {
    let cfg = FilcoConfig::new().with_lexical_threshold(f32::NAN);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_negative_cxmi_threshold() {
    let cfg = FilcoConfig::new().with_cxmi_threshold(-0.5);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_cxmi_threshold_above_one() {
    let cfg = FilcoConfig::new().with_cxmi_threshold(2.0);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_nan_cxmi_threshold() {
    let cfg = FilcoConfig::new().with_cxmi_threshold(f32::NAN);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_rejects_zero_min_ngram() {
    let cfg = FilcoConfig::new().with_strinc_min_ngram(0);
    assert!(matches!(cfg.validate(), Err(FilcoError::InvalidConfig(_))));
}

#[test]
fn config_validate_accepts_boundary_thresholds() {
    let cfg = FilcoConfig::new()
        .with_lexical_threshold(0.0)
        .with_cxmi_threshold(1.0);
    assert!(cfg.validate().is_ok());
}

// ── FilcoError ─────────────────────────────────────────────────────────────────

#[test]
fn error_empty_query_display() {
    assert_eq!(FilcoError::EmptyQuery.to_string(), "query is empty");
}

#[test]
fn error_empty_passages_display() {
    assert_eq!(
        FilcoError::EmptyPassages.to_string(),
        "passages must not be empty"
    );
}

#[test]
fn error_invalid_config_display() {
    let err = FilcoError::InvalidConfig("bad threshold".to_string());
    assert!(err.to_string().contains("bad threshold"));
    assert!(err.to_string().starts_with("invalid filco config"));
}

#[test]
fn error_equality() {
    assert_eq!(FilcoError::EmptyQuery, FilcoError::EmptyQuery);
    assert_ne!(FilcoError::EmptyQuery, FilcoError::EmptyPassages);
}

// ── ScoredSentence / FilcoReport ────────────────────────────────────────────────

#[test]
fn scored_sentence_fields_accessible() {
    let s = ScoredSentence {
        text: "hi".to_string(),
        score: 0.5,
        kept: true,
    };
    assert_eq!(s.text, "hi");
    assert_eq!(s.score, 0.5);
    assert!(s.kept);
}

#[test]
fn report_is_empty_true_when_zero_kept() {
    let report = FilcoReport {
        filtered_passages: vec![String::new()],
        scored_sentences: vec![],
        original_sentence_count: 0,
        kept_sentence_count: 0,
        compression_ratio: 0.0,
    };
    assert!(report.is_empty());
}

#[test]
fn report_is_empty_false_when_kept_present() {
    let report = FilcoReport {
        filtered_passages: vec!["hi".to_string()],
        scored_sentences: vec![ScoredSentence {
            text: "hi".to_string(),
            score: 1.0,
            kept: true,
        }],
        original_sentence_count: 1,
        kept_sentence_count: 1,
        compression_ratio: 1.0,
    };
    assert!(!report.is_empty());
}

// ── tokenize / token_set ─────────────────────────────────────────────────────────

#[test]
fn tokenize_lowercases_and_splits() {
    let tokens = tokenize("Hello, World! 2024");
    assert_eq!(
        tokens,
        vec!["hello".to_string(), "world".to_string(), "2024".to_string()]
    );
}

#[test]
fn tokenize_drops_single_char_tokens() {
    let tokens = tokenize("a I go to it");
    assert_eq!(
        tokens,
        vec!["go".to_string(), "to".to_string(), "it".to_string()]
    );
}

#[test]
fn token_set_is_case_insensitive() {
    let a = token_set("Rust RUST rust");
    assert_eq!(a.len(), 1);
    assert!(a.contains("rust"));
}

// ── split_sentences ───────────────────────────────────────────────────────────────

#[test]
fn split_sentences_period_boundaries() {
    let sentences = split_sentences("First sentence. Second sentence. Third one.");
    assert_eq!(sentences.len(), 3);
    assert_eq!(sentences[0], "First sentence.");
    assert_eq!(sentences[2], "Third one.");
}

#[test]
fn split_sentences_question_mark() {
    let sentences = split_sentences("What is Rust? It is a systems language.");
    assert_eq!(sentences.len(), 2);
}

#[test]
fn split_sentences_double_newline() {
    let sentences = split_sentences("Paragraph one text\n\nParagraph two text");
    assert_eq!(sentences.len(), 2);
}

#[test]
fn split_sentences_empty_input_yields_none() {
    assert!(split_sentences("").is_empty());
}

#[test]
fn split_sentences_whitespace_only_yields_none() {
    assert!(split_sentences("   \n  ").is_empty());
}

// ── jaccard / lexical overlap ───────────────────────────────────────────────────

#[test]
fn jaccard_basic_ratio() {
    let a = token_set("alpha beta");
    let b = token_set("alpha gamma");
    let score = jaccard(&a, &b);
    assert!((score - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn jaccard_both_empty_is_zero() {
    let a = token_set("");
    let b = token_set("");
    assert_eq!(jaccard(&a, &b), 0.0);
}

#[test]
fn jaccard_one_side_empty_is_zero() {
    let a = token_set("hello world");
    let b = token_set("");
    assert_eq!(jaccard(&a, &b), 0.0);
}

#[test]
fn jaccard_identical_sets_is_one() {
    let a = token_set("alpha beta gamma");
    let b = token_set("gamma beta alpha");
    assert_eq!(jaccard(&a, &b), 1.0);
}

#[test]
fn lexical_overlap_score_matches_jaccard() {
    let sentence_tokens = token_set("alpha gamma");
    let query_tokens = token_set("alpha beta");
    let score = lexical_overlap_score(&sentence_tokens, &query_tokens);
    let expected = jaccard(&sentence_tokens, &query_tokens);
    assert_eq!(score, expected);
}

// ── STRINC — strict (paper-faithful) ───────────────────────────────────────────

#[test]
fn strinc_strict_matches_case_insensitive_substring() {
    let score = strinc_strict_score("Paris is the capital of France.", "PARIS");
    assert_eq!(score, 1.0);
}

#[test]
fn strinc_strict_no_match_is_zero() {
    let score = strinc_strict_score("Paris is the capital of France.", "London");
    assert_eq!(score, 0.0);
}

#[test]
fn strinc_strict_blank_answer_is_zero() {
    let score = strinc_strict_score("Any sentence here.", "   ");
    assert_eq!(score, 0.0);
}

#[test]
fn strinc_strict_multi_word_phrase_substring() {
    let score = strinc_strict_score("Paris is the capital of France.", "capital of France");
    assert_eq!(score, 1.0);
}

// ── STRINC — heuristic (no known answer) ───────────────────────────────────────

#[test]
fn strinc_heuristic_ngram_match_scores_one() {
    let (ordered, set) = query_parts("improves battery life");
    let score =
        strinc_heuristic_score("The battery life extends significantly.", &ordered, &set, 2);
    assert_eq!(score, 1.0);
}

#[test]
fn strinc_heuristic_no_overlap_scores_zero() {
    let (ordered, set) = query_parts("improves battery life");
    let score = strinc_heuristic_score(
        "The weather today is sunny and warm outside.",
        &ordered,
        &set,
        2,
    );
    assert_eq!(score, 0.0);
}

#[test]
fn strinc_heuristic_single_token_overlap_is_partial_not_kept() {
    let (ordered, set) = query_parts("chocolate cake recipe");
    let score = strinc_heuristic_score("cake tastes great", &ordered, &set, 2);
    assert!((score - 0.5).abs() < 1e-6);
    assert!(score < 1.0);
}

#[test]
fn strinc_heuristic_numeric_entity_triggers_keep() {
    let (ordered, set) = query_parts("release version 2024 update");
    let score = strinc_heuristic_score(
        "The new update arrived in 2024 with fixes",
        &ordered,
        &set,
        2,
    );
    assert_eq!(score, 1.0);
}

#[test]
fn strinc_heuristic_capitalized_entity_triggers_keep() {
    let (ordered, set) = query_parts("who is Einstein");
    let score = strinc_heuristic_score("Einstein published his theory in 1905", &ordered, &set, 2);
    assert_eq!(score, 1.0);
}

#[test]
fn strinc_heuristic_min_ngram_three_requires_longer_overlap() {
    let (ordered, set) = query_parts("improves battery life dramatically");
    let score =
        strinc_heuristic_score("The battery life extends significantly.", &ordered, &set, 3);
    assert!((score - 2.0 / 3.0).abs() < 1e-6);
    assert!(score < 1.0);
}

#[test]
fn strinc_heuristic_zero_min_ngram_clamped_to_one() {
    let (ordered, set) = query_parts("chocolate cake recipe");
    let score_zero = strinc_heuristic_score("cake tastes great", &ordered, &set, 0);
    let score_one = strinc_heuristic_score("cake tastes great", &ordered, &set, 1);
    assert_eq!(score_zero, score_one);
}

#[test]
fn strinc_heuristic_empty_query_scores_zero() {
    let (ordered, set) = query_parts("a I"); // both tokens length 1, dropped -> empty
    assert!(ordered.is_empty());
    let score = strinc_heuristic_score("Any content sentence here.", &ordered, &set, 2);
    assert_eq!(score, 0.0);
}

// ── CXMI-lite ──────────────────────────────────────────────────────────────────

#[test]
fn document_frequency_counts_across_corpus() {
    let sets = vec![token_set("alpha beta"), token_set("alpha gamma")];
    let df = document_frequency(&sets);
    assert_eq!(df.get("alpha").copied(), Some(2));
    assert_eq!(df.get("beta").copied(), Some(1));
    assert_eq!(df.get("gamma").copied(), Some(1));
}

#[test]
fn idf_weighted_relevance_empty_query_is_zero() {
    let query_tokens: HashSet<String> = HashSet::new();
    let context_tokens = token_set("something here");
    let doc_freq = document_frequency(std::slice::from_ref(&context_tokens));
    let score = idf_weighted_relevance(&query_tokens, &context_tokens, &doc_freq, 1);
    assert_eq!(score, 0.0);
}

#[test]
fn cxmi_lite_relevance_full_coverage_scores_one() {
    let query_tokens = token_set("quantum computing");
    let sentence_a_tokens = token_set("Quantum computing uses qubits.");
    let sentence_b_tokens = token_set("Bananas are tasty fruit.");
    let corpus = vec![sentence_a_tokens.clone(), sentence_b_tokens.clone()];
    let doc_freq = document_frequency(&corpus);

    let score_a = cxmi_lite_score(&sentence_a_tokens, &query_tokens, &doc_freq, corpus.len());
    let score_b = cxmi_lite_score(&sentence_b_tokens, &query_tokens, &doc_freq, corpus.len());

    assert_eq!(score_a, 1.0);
    assert_eq!(score_b, 0.0);
}

#[test]
fn cxmi_lite_partial_coverage_between_zero_and_one() {
    let query_tokens = token_set("quantum computing breakthrough");
    let sentence_tokens = token_set("Quantum computing uses qubits.");
    let corpus = vec![
        sentence_tokens.clone(),
        token_set("Bananas are tasty fruit."),
    ];
    let doc_freq = document_frequency(&corpus);

    let score = cxmi_lite_score(&sentence_tokens, &query_tokens, &doc_freq, corpus.len());
    assert!(score > 0.0 && score < 1.0);
}

#[test]
fn cxmi_lite_empty_context_relative_to_itself_is_zero() {
    let query_tokens = token_set("alpha beta");
    let empty_tokens: HashSet<String> = HashSet::new();
    let doc_freq = document_frequency(std::slice::from_ref(&query_tokens));
    let score = cxmi_lite_score(&empty_tokens, &query_tokens, &doc_freq, 1);
    assert_eq!(score, 0.0);
}

// ── FilcoFilter — validation ────────────────────────────────────────────────────

#[test]
fn filter_empty_query_errors() {
    let f = default_filter();
    let passages = vec!["Some content.".to_string()];
    assert_eq!(f.filter("", &passages), Err(FilcoError::EmptyQuery));
}

#[test]
fn filter_whitespace_query_errors() {
    let f = default_filter();
    let passages = vec!["Some content.".to_string()];
    assert_eq!(f.filter("   ", &passages), Err(FilcoError::EmptyQuery));
}

#[test]
fn filter_empty_passages_slice_errors() {
    let f = default_filter();
    assert_eq!(f.filter("query", &[]), Err(FilcoError::EmptyPassages));
}

#[test]
fn filter_all_blank_passages_errors() {
    let f = default_filter();
    let passages = vec![String::new(), "   ".to_string()];
    assert_eq!(f.filter("query", &passages), Err(FilcoError::EmptyPassages));
}

#[test]
fn filter_one_blank_one_real_passage_ok() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let passages = vec![
        String::new(),
        "Rust provides memory safety guarantees.".to_string(),
    ];
    let report = f.filter("memory safety", &passages).expect("ok");
    assert_eq!(report.filtered_passages.len(), 2);
    assert_eq!(report.filtered_passages[0], "");
    assert_eq!(
        report.filtered_passages[1],
        "Rust provides memory safety guarantees."
    );
    assert_eq!(report.original_sentence_count, 1);
    assert_eq!(report.kept_sentence_count, 1);
    assert_eq!(report.compression_ratio, 1.0);
}

#[test]
fn filter_invalid_config_propagates_error() {
    let f = FilcoFilter::new(FilcoConfig::new().with_strinc_min_ngram(0));
    let passages = vec!["Some content here.".to_string()];
    assert!(matches!(
        f.filter("query", &passages),
        Err(FilcoError::InvalidConfig(_))
    ));
}

#[test]
fn filter_with_answer_empty_query_errors() {
    let f = default_filter();
    let passages = vec!["content".to_string()];
    assert_eq!(
        f.filter_with_answer("", &passages, "answer"),
        Err(FilcoError::EmptyQuery)
    );
}

// ── FilcoFilter — LexicalOverlap ───────────────────────────────────────────────

#[test]
fn lexical_overlap_scores_correctly_via_filter() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let passages = vec!["Alpha gamma.".to_string()];
    let report = f.filter("alpha beta", &passages).expect("ok");
    assert_eq!(report.scored_sentences.len(), 1);
    let score = report.scored_sentences[0].score;
    assert!((score - 1.0 / 3.0).abs() < 1e-6);
}

#[test]
fn lexical_overlap_boundary_equal_threshold_not_kept() {
    let query = "alpha beta";
    let passages = vec!["Alpha gamma.".to_string()];

    let probe = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let probe_report = probe.filter(query, &passages).expect("ok");
    let score = probe_report.scored_sentences[0].score;

    let exact = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(score),
    );
    let exact_report = exact.filter(query, &passages).expect("ok");
    assert!(!exact_report.scored_sentences[0].kept);

    let below = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(score - 0.01),
    );
    let below_report = below.filter(query, &passages).expect("ok");
    assert!(below_report.scored_sentences[0].kept);
}

// ── FilcoFilter — CxmiLite ──────────────────────────────────────────────────────

#[test]
fn cxmi_lite_scores_correctly_via_filter() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::CxmiLite)
            .with_cxmi_threshold(0.0),
    );
    let passages = vec![
        "Quantum computing uses qubits.".to_string(),
        "Bananas are tasty fruit.".to_string(),
    ];
    let report = f.filter("quantum computing", &passages).expect("ok");
    assert_eq!(report.scored_sentences[0].score, 1.0);
    assert_eq!(report.scored_sentences[1].score, 0.0);
    assert!(report.scored_sentences[0].kept);
    assert!(!report.scored_sentences[1].kept);
}

#[test]
fn cxmi_lite_boundary_equal_threshold_not_kept() {
    let query = "quantum computing";
    let passages = vec![
        "Quantum computing uses qubits.".to_string(),
        "Bananas are tasty fruit.".to_string(),
    ];

    let probe = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::CxmiLite)
            .with_cxmi_threshold(0.0),
    );
    let probe_report = probe.filter(query, &passages).expect("ok");
    let score = probe_report.scored_sentences[0].score;

    let exact = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::CxmiLite)
            .with_cxmi_threshold(score),
    );
    let exact_report = exact.filter(query, &passages).expect("ok");
    assert!(!exact_report.scored_sentences[0].kept);
}

// ── FilcoFilter — StrInc (default measure) ─────────────────────────────────────

#[test]
fn strinc_default_measure_keeps_matching_sentence() {
    let f = default_filter();
    let passages = vec!["Paris is the capital of France.".to_string()];
    let report = f
        .filter("What is the capital of France?", &passages)
        .expect("ok");
    assert!(report.scored_sentences[0].kept);
    assert_eq!(
        report.filtered_passages[0],
        "Paris is the capital of France."
    );
}

#[test]
fn strinc_default_measure_drops_unrelated_sentence() {
    let f = default_filter();
    let passages = vec!["Bananas are yellow and rich in potassium.".to_string()];
    let report = f
        .filter("What is the capital of France?", &passages)
        .expect("ok");
    assert!(!report.scored_sentences[0].kept);
    assert_eq!(report.filtered_passages[0], "");
}

// ── FilcoFilter — filter_with_answer (strict STRINC) ───────────────────────────

#[test]
fn filter_with_answer_strict_keeps_matching_sentence() {
    let f = FilcoFilter::new(FilcoConfig::new().with_measure(FilterMeasure::LexicalOverlap));
    let passages = vec![
        "Paris is the capital of France.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
    ];
    let report = f
        .filter_with_answer("What is the capital of France?", &passages, "Paris")
        .expect("ok");
    assert!(report.scored_sentences[0].kept);
    assert!(!report.scored_sentences[1].kept);
    assert_eq!(
        report.filtered_passages[0],
        "Paris is the capital of France."
    );
    assert_eq!(report.filtered_passages[1], "");
}

#[test]
fn filter_with_answer_overrides_configured_measure() {
    let f = FilcoFilter::new(FilcoConfig::new().with_measure(FilterMeasure::CxmiLite));
    let passages = vec!["The mitochondria is the powerhouse of the cell.".to_string()];
    let report = f
        .filter_with_answer(
            "What is the powerhouse of the cell?",
            &passages,
            "mitochondria",
        )
        .expect("ok");
    assert!(report.scored_sentences[0].kept);
}

#[test]
fn filter_with_answer_blank_answer_keeps_nothing() {
    let f = default_filter();
    let passages = vec!["Any content sentence here.".to_string()];
    let report = f
        .filter_with_answer("query text", &passages, "   ")
        .expect("ok");
    assert!(report.is_empty());
}

#[test]
fn filter_with_answer_case_insensitive() {
    let f = default_filter();
    let passages = vec!["The Eiffel Tower is in Paris.".to_string()];
    let report = f
        .filter_with_answer("Where is the Eiffel Tower?", &passages, "PARIS")
        .expect("ok");
    assert!(report.scored_sentences[0].kept);
}

// ── Sentence order preservation ────────────────────────────────────────────────

#[test]
fn sentence_order_preserved_across_passages() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let passages = vec![
        "Cats are mammals. Dogs are mammals too.".to_string(),
        "Fish live in water. Birds can fly high.".to_string(),
    ];
    let report = f.filter("mammals fish", &passages).expect("ok");

    assert_eq!(report.original_sentence_count, 4);
    assert_eq!(report.scored_sentences.len(), 4);
    assert_eq!(report.scored_sentences[0].text, "Cats are mammals.");
    assert_eq!(report.scored_sentences[1].text, "Dogs are mammals too.");
    assert_eq!(report.scored_sentences[2].text, "Fish live in water.");
    assert_eq!(report.scored_sentences[3].text, "Birds can fly high.");

    assert_eq!(
        report.filtered_passages[0],
        "Cats are mammals. Dogs are mammals too."
    );
    assert_eq!(report.filtered_passages[1], "Fish live in water.");
}

#[test]
fn middle_sentence_dropped_preserves_surrounding_order() {
    let f = FilcoFilter::new(FilcoConfig::new().with_measure(FilterMeasure::LexicalOverlap));
    let passages = vec![
        "Alpha reigns supreme. Beta is unrelated nonsense filler. Gamma finishes strong."
            .to_string(),
    ];
    let report = f.filter("alpha gamma", &passages).expect("ok");
    assert_eq!(
        report.filtered_passages[0],
        "Alpha reigns supreme. Gamma finishes strong."
    );
    assert_eq!(report.kept_sentence_count, 2);
    assert_eq!(report.original_sentence_count, 3);
}

// ── Compression ratio ───────────────────────────────────────────────────────────

#[test]
fn compression_ratio_matches_kept_over_original() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let passages = vec![
        "Cats are mammals. Dogs are mammals too.".to_string(),
        "Fish live in water. Birds can fly high.".to_string(),
    ];
    let report = f.filter("mammals fish", &passages).expect("ok");
    assert_eq!(report.kept_sentence_count, 3);
    assert_eq!(report.original_sentence_count, 4);
    assert!((report.compression_ratio - 0.75).abs() < 1e-6);
}

#[test]
fn compression_ratio_zero_when_nothing_kept() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(1.0),
    );
    let passages = vec!["Completely unrelated sentence here.".to_string()];
    let report = f.filter("something else entirely", &passages).expect("ok");
    assert_eq!(report.kept_sentence_count, 0);
    assert_eq!(report.compression_ratio, 0.0);
    assert!(report.is_empty());
}

#[test]
fn compression_ratio_one_when_everything_kept() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(0.0),
    );
    let passages = vec!["Rust is fast.".to_string()];
    let report = f.filter("Rust fast", &passages).expect("ok");
    assert_eq!(report.compression_ratio, 1.0);
}

// ── FilcoFilter — general ───────────────────────────────────────────────────────

#[test]
fn default_filter_uses_default_config() {
    let f = FilcoFilter::default();
    assert_eq!(f.config(), &FilcoConfig::default());
}

#[test]
fn filter_is_deterministic() {
    let f = FilcoFilter::new(FilcoConfig::new().with_measure(FilterMeasure::LexicalOverlap));
    let passages = vec!["The quick brown fox jumps over the lazy dog.".to_string()];
    let a = f.filter("quick fox", &passages).expect("ok");
    let b = f.filter("quick fox", &passages).expect("ok");
    assert_eq!(a, b);
}

#[test]
fn filtered_passages_len_matches_input_len_even_when_all_dropped() {
    let f = FilcoFilter::new(
        FilcoConfig::new()
            .with_measure(FilterMeasure::LexicalOverlap)
            .with_lexical_threshold(1.0),
    );
    let passages = vec![
        "First unrelated passage.".to_string(),
        "Second unrelated passage.".to_string(),
        "Third unrelated passage.".to_string(),
    ];
    let report = f.filter("something else entirely", &passages).expect("ok");
    assert_eq!(report.filtered_passages.len(), 3);
    assert!(report.filtered_passages.iter().all(String::is_empty));
}

#[test]
fn scored_sentences_len_matches_original_sentence_count() {
    let f = FilcoFilter::new(FilcoConfig::new().with_measure(FilterMeasure::LexicalOverlap));
    let passages = vec![
        "One sentence here. Two sentences here.".to_string(),
        "Three sentences here.".to_string(),
    ];
    let report = f.filter("sentences", &passages).expect("ok");
    assert_eq!(
        report.scored_sentences.len(),
        report.original_sentence_count
    );
}
