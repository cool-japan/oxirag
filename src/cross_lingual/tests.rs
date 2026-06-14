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

use crate::cross_lingual::lexicon::{BilingualLexicon, normalize};
use crate::cross_lingual::retriever::CrossLingualRetriever;
use crate::cross_lingual::types::{CrossLingualConfig, CrossLingualError};
use crate::types::Document;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn fr_en_lexicon() -> BilingualLexicon {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    lex.add("chat", "cat");
    lex.add("maison", "house");
    lex.add("voiture", "car");
    lex
}

// ── Config defaults & builders ─────────────────────────────────────────────────

#[test]
fn config_default_dim_is_128() {
    assert_eq!(CrossLingualConfig::default().dim, 128);
}

#[test]
fn config_default_expand_is_true() {
    assert!(CrossLingualConfig::default().expand_with_lexicon);
}

#[test]
fn config_new_matches_default() {
    let a = CrossLingualConfig::new();
    let b = CrossLingualConfig::default();
    assert_eq!(a.dim, b.dim);
    assert_eq!(a.expand_with_lexicon, b.expand_with_lexicon);
}

#[test]
fn config_with_dim_sets_dim() {
    let config = CrossLingualConfig::new().with_dim(64);
    assert_eq!(config.dim, 64);
}

#[test]
fn config_with_dim_preserves_expand() {
    let config = CrossLingualConfig::new().with_dim(256);
    assert!(config.expand_with_lexicon);
}

#[test]
fn config_with_expand_false() {
    let config = CrossLingualConfig::new().with_expand_with_lexicon(false);
    assert!(!config.expand_with_lexicon);
}

#[test]
fn config_with_expand_preserves_dim() {
    let config = CrossLingualConfig::new().with_expand_with_lexicon(false);
    assert_eq!(config.dim, 128);
}

#[test]
fn config_builders_chain() {
    let config = CrossLingualConfig::new()
        .with_dim(32)
        .with_expand_with_lexicon(false);
    assert_eq!(config.dim, 32);
    assert!(!config.expand_with_lexicon);
}

#[test]
fn config_is_clone() {
    let config = CrossLingualConfig::new().with_dim(16);
    let cloned = config.clone();
    assert_eq!(cloned.dim, 16);
}

// ── Lexicon add / translate ────────────────────────────────────────────────────

#[test]
fn lexicon_new_is_empty() {
    assert!(BilingualLexicon::new().is_empty());
}

#[test]
fn lexicon_new_len_zero() {
    assert_eq!(BilingualLexicon::new().len(), 0);
}

#[test]
fn lexicon_default_is_empty() {
    assert!(BilingualLexicon::default().is_empty());
}

#[test]
fn lexicon_add_increases_len() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    assert_eq!(lex.len(), 1);
}

#[test]
fn lexicon_add_not_empty() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    assert!(!lex.is_empty());
}

#[test]
fn lexicon_translate_known_token() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    assert_eq!(lex.translate("chien"), vec!["dog".to_string()]);
}

#[test]
fn lexicon_translate_unknown_is_empty() {
    let lex = fr_en_lexicon();
    assert!(lex.translate("oiseau").is_empty());
}

#[test]
fn lexicon_translate_one_to_many() {
    let mut lex = BilingualLexicon::new();
    lex.add("grand", "big");
    lex.add("grand", "large");
    lex.add("grand", "tall");
    let mut translations = lex.translate("grand");
    translations.sort();
    assert_eq!(
        translations,
        vec!["big".to_string(), "large".to_string(), "tall".to_string()]
    );
}

#[test]
fn lexicon_one_to_many_len_is_one() {
    let mut lex = BilingualLexicon::new();
    lex.add("grand", "big");
    lex.add("grand", "large");
    assert_eq!(lex.len(), 1);
}

#[test]
fn lexicon_add_lowercases_source() {
    let mut lex = BilingualLexicon::new();
    lex.add("CHIEN", "dog");
    assert_eq!(lex.translate("chien"), vec!["dog".to_string()]);
}

#[test]
fn lexicon_add_lowercases_target() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "DOG");
    assert_eq!(lex.translate("chien"), vec!["dog".to_string()]);
}

#[test]
fn lexicon_translate_case_insensitive() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    assert_eq!(lex.translate("CHIEN"), vec!["dog".to_string()]);
}

#[test]
fn lexicon_add_duplicate_is_idempotent() {
    let mut lex = BilingualLexicon::new();
    lex.add("chien", "dog");
    lex.add("chien", "dog");
    assert_eq!(lex.translate("chien"), vec!["dog".to_string()]);
}

#[test]
fn lexicon_multiple_sources() {
    let lex = fr_en_lexicon();
    assert_eq!(lex.len(), 4);
}

// ── Normalize strips diacritics ────────────────────────────────────────────────

#[test]
fn normalize_cafe() {
    assert_eq!(normalize("café"), "cafe");
}

#[test]
fn normalize_munchen() {
    assert_eq!(normalize("München"), "munchen");
}

#[test]
fn normalize_lowercases() {
    assert_eq!(normalize("HELLO"), "hello");
}

#[test]
fn normalize_e_acute() {
    assert_eq!(normalize("é"), "e");
}

#[test]
fn normalize_u_umlaut() {
    assert_eq!(normalize("ü"), "u");
}

#[test]
fn normalize_n_tilde() {
    assert_eq!(normalize("ñ"), "n");
}

#[test]
fn normalize_c_cedilla() {
    assert_eq!(normalize("ç"), "c");
}

#[test]
fn normalize_o_umlaut() {
    assert_eq!(normalize("ö"), "o");
}

#[test]
fn normalize_a_umlaut() {
    assert_eq!(normalize("ä"), "a");
}

#[test]
fn normalize_spanish_word() {
    assert_eq!(normalize("Niño"), "nino");
}

#[test]
fn normalize_preserves_ascii() {
    assert_eq!(normalize("hello world"), "hello world");
}

#[test]
fn normalize_mixed_sentence() {
    assert_eq!(normalize("Crème Brûlée"), "creme brulee");
}

#[test]
fn normalize_associated_fn_matches_free_fn() {
    assert_eq!(CrossLingualRetriever::normalize("café"), normalize("café"));
}

#[test]
fn normalize_eszett() {
    assert_eq!(normalize("Straße"), "strasse");
}

// ── expand_query ───────────────────────────────────────────────────────────────

#[test]
fn expand_query_includes_original_tokens() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let expanded = retriever.expand_query("chien");
    assert!(expanded.contains(&"chien".to_string()));
}

#[test]
fn expand_query_includes_translations() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let expanded = retriever.expand_query("chien");
    assert!(expanded.contains(&"dog".to_string()));
}

#[test]
fn expand_query_normalizes_original() {
    let mut lex = BilingualLexicon::new();
    lex.add("cafe", "coffee");
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), lex);
    let expanded = retriever.expand_query("café");
    assert!(expanded.contains(&"cafe".to_string()));
}

#[test]
fn expand_query_multi_token() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let expanded = retriever.expand_query("chien chat");
    assert!(expanded.contains(&"dog".to_string()));
    assert!(expanded.contains(&"cat".to_string()));
}

#[test]
fn expand_query_false_skips_translations() {
    let config = CrossLingualConfig::new().with_expand_with_lexicon(false);
    let retriever = CrossLingualRetriever::new(config, fr_en_lexicon());
    let expanded = retriever.expand_query("chien");
    assert!(!expanded.contains(&"dog".to_string()));
}

#[test]
fn expand_query_false_keeps_original() {
    let config = CrossLingualConfig::new().with_expand_with_lexicon(false);
    let retriever = CrossLingualRetriever::new(config, fr_en_lexicon());
    let expanded = retriever.expand_query("chien");
    assert_eq!(expanded, vec!["chien".to_string()]);
}

#[test]
fn expand_query_original_before_translations() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let expanded = retriever.expand_query("chien");
    let orig_pos = expanded.iter().position(|t| t == "chien").unwrap();
    let trans_pos = expanded.iter().position(|t| t == "dog").unwrap();
    assert!(orig_pos < trans_pos);
}

#[test]
fn expand_query_dedups() {
    let mut lex = BilingualLexicon::new();
    lex.add("dog", "dog");
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), lex);
    let expanded = retriever.expand_query("dog");
    assert_eq!(expanded.iter().filter(|t| *t == "dog").count(), 1);
}

#[test]
fn expand_query_one_to_many() {
    let mut lex = BilingualLexicon::new();
    lex.add("grand", "big");
    lex.add("grand", "large");
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), lex);
    let expanded = retriever.expand_query("grand");
    assert!(expanded.contains(&"big".to_string()));
    assert!(expanded.contains(&"large".to_string()));
}

#[test]
fn expand_query_unknown_token_only_original() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let expanded = retriever.expand_query("oiseau");
    assert_eq!(expanded, vec!["oiseau".to_string()]);
}

// ── Cross-lingual retrieval ────────────────────────────────────────────────────

#[test]
fn search_cross_lingual_match() {
    let retriever_docs = vec![
        doc("d1", "The dog runs fast in the park."),
        doc("d2", "A boat sails across the sea."),
    ];
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&retriever_docs);
    // French query "chien" translates to "dog" and retrieves the English doc.
    let hits = retriever.search("chien", 5).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "d1");
}

#[test]
fn search_cross_lingual_positive_score() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs fast.")]);
    let hits = retriever.search("chien", 5).unwrap();
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_without_lexicon_no_cross_match() {
    let config = CrossLingualConfig::new().with_expand_with_lexicon(false);
    let mut retriever = CrossLingualRetriever::new(config, fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs fast.")]);
    // Without expansion the French token cannot match the English document.
    let hits = retriever.search("chien", 5).unwrap();
    assert_eq!(hits[0].score, 0.0);
}

#[test]
fn search_diacritic_insensitive_match() {
    let mut retriever =
        CrossLingualRetriever::new(CrossLingualConfig::default(), BilingualLexicon::new());
    retriever.build(&[doc("d1", "We visited a cafe downtown.")]);
    // Accented query matches the unaccented document.
    let hits = retriever.search("café", 5).unwrap();
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_accented_doc_unaccented_query() {
    let mut retriever =
        CrossLingualRetriever::new(CrossLingualConfig::default(), BilingualLexicon::new());
    retriever.build(&[doc("d1", "We visited a café downtown.")]);
    let hits = retriever.search("cafe", 5).unwrap();
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_munchen_diacritic_match() {
    let mut retriever =
        CrossLingualRetriever::new(CrossLingualConfig::default(), BilingualLexicon::new());
    retriever.build(&[doc("d1", "The conference was held in Munchen.")]);
    let hits = retriever.search("München", 5).unwrap();
    assert!(hits[0].score > 0.0);
}

#[test]
fn search_picks_relevant_doc_among_many() {
    let docs = vec![
        doc("d1", "A house stood tall downtown."),
        doc("d2", "The car parked quietly."),
        doc("d3", "A cat napped quietly."),
    ];
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&docs);
    // "voiture" → "car" should retrieve d2.
    let hits = retriever.search("voiture", 5).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "d2");
}

#[test]
fn search_returns_hit_document_verbatim() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs fast.")]);
    let hits = retriever.search("chien", 5).unwrap();
    assert_eq!(hits[0].document.content, "The dog runs fast.");
}

// ── top_k ──────────────────────────────────────────────────────────────────────

#[test]
fn search_top_k_limits_results() {
    let docs = vec![
        doc("d1", "The dog runs."),
        doc("d2", "The dog barks."),
        doc("d3", "The dog sleeps."),
    ];
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&docs);
    let hits = retriever.search("chien", 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_top_k_zero_empty() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs.")]);
    let hits = retriever.search("chien", 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_top_k_larger_than_corpus() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs."), doc("d2", "The cat sleeps.")]);
    let hits = retriever.search("chien", 100).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_results_sorted_descending() {
    let docs = vec![
        doc("d1", "The dog runs and the dog barks loudly all day."),
        doc("d2", "A single mention of a dog somewhere."),
        doc("d3", "Nothing relevant about boats and seas here."),
    ];
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&docs);
    let hits = retriever.search("chien", 3).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score);
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

#[test]
fn search_empty_corpus_errors() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let result = retriever.search("chien", 5);
    assert!(matches!(result, Err(CrossLingualError::EmptyCorpus)));
}

#[test]
fn search_empty_query_errors() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs.")]);
    let result = retriever.search("", 5);
    assert!(matches!(result, Err(CrossLingualError::EmptyQuery)));
}

#[test]
fn search_whitespace_query_errors() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "The dog runs.")]);
    let result = retriever.search("   ", 5);
    assert!(matches!(result, Err(CrossLingualError::EmptyQuery)));
}

#[test]
fn search_empty_query_checked_before_corpus() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    // Empty corpus *and* empty query: query check wins.
    let result = retriever.search("", 5);
    assert!(matches!(result, Err(CrossLingualError::EmptyQuery)));
}

#[test]
fn error_display_empty_corpus() {
    assert_eq!(
        CrossLingualError::EmptyCorpus.to_string(),
        "corpus is empty"
    );
}

#[test]
fn error_display_empty_query() {
    assert_eq!(
        CrossLingualError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── len / is_empty ─────────────────────────────────────────────────────────────

#[test]
fn retriever_new_is_empty() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    assert!(retriever.is_empty());
}

#[test]
fn retriever_new_len_zero() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    assert_eq!(retriever.len(), 0);
}

#[test]
fn retriever_len_after_build() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "a"), doc("d2", "b"), doc("d3", "c")]);
    assert_eq!(retriever.len(), 3);
}

#[test]
fn retriever_not_empty_after_build() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "a")]);
    assert!(!retriever.is_empty());
}

#[test]
fn retriever_build_replaces_corpus() {
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&[doc("d1", "a"), doc("d2", "b")]);
    retriever.build(&[doc("d3", "c")]);
    assert_eq!(retriever.len(), 1);
}

// ── Determinism ────────────────────────────────────────────────────────────────

#[test]
fn search_is_deterministic() {
    let docs = vec![
        doc("d1", "The dog runs fast in the park."),
        doc("d2", "A cat sleeps on the warm sofa."),
    ];
    let mut a = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    a.build(&docs);
    let mut b = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    b.build(&docs);
    let hits_a = a.search("chien", 5).unwrap();
    let hits_b = b.search("chien", 5).unwrap();
    assert_eq!(hits_a.len(), hits_b.len());
    for (x, y) in hits_a.iter().zip(hits_b.iter()) {
        assert_eq!(x.document.id.as_str(), y.document.id.as_str());
        assert_eq!(x.score, y.score);
    }
}

#[test]
fn expand_query_is_deterministic() {
    let retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    let first = retriever.expand_query("chien chat maison");
    let second = retriever.expand_query("chien chat maison");
    assert_eq!(first, second);
}

#[test]
fn normalize_is_deterministic() {
    assert_eq!(
        normalize("Crème Brûlée café"),
        normalize("Crème Brûlée café")
    );
}

#[test]
fn search_tie_break_by_document_id() {
    // Two documents with identical content tie on score; id ordering decides.
    let docs = vec![doc("zeta", "The dog runs."), doc("alpha", "The dog runs.")];
    let mut retriever = CrossLingualRetriever::new(CrossLingualConfig::default(), fr_en_lexicon());
    retriever.build(&docs);
    let hits = retriever.search("chien", 2).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "alpha");
    assert_eq!(hits[1].document.id.as_str(), "zeta");
}
