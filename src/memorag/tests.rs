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
//! Tests for the `memorag` module.

use crate::memorag::engine::MemoRagEngine;
use crate::memorag::memory::{build_gist, embed, split_sentences, tokenize};
use crate::memorag::types::{MemoRagConfig, MemoRagError};
use crate::types::Document;

// ── helpers ───────────────────────────────────────────────────────────────────

fn doc(content: &str) -> Document {
    Document::new(content)
}

fn doc_with_id(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

/// A small, deterministic corpus about Rust memory management.
fn rust_corpus() -> Vec<Document> {
    vec![
        doc_with_id(
            "d0",
            "Rust prevents data races. The borrow checker enforces ownership rules. \
             Memory safety is guaranteed without a garbage collector.",
        ),
        doc_with_id(
            "d1",
            "Ownership in Rust governs how memory is freed. Lifetimes track references. \
             The borrow checker validates every borrow.",
        ),
        doc_with_id(
            "d2",
            "Concurrency in Rust is fearless. Threads share memory safely through ownership. \
             Channels pass messages between threads.",
        ),
        doc_with_id(
            "d3",
            "The standard library provides collections. Vectors store elements contiguously. \
             Hash maps index entries by key.",
        ),
    ]
}

fn built_engine() -> MemoRagEngine {
    let mut engine = MemoRagEngine::new(MemoRagConfig::default());
    engine.build_memory(&rust_corpus()).unwrap();
    engine
}

// ── MemoRagConfig: defaults & builders ──────────────────────────────────────────

#[test]
fn config_default_gist_sentences() {
    assert_eq!(MemoRagConfig::default().gist_sentences, 5);
}

#[test]
fn config_default_num_clues() {
    assert_eq!(MemoRagConfig::default().num_clues, 3);
}

#[test]
fn config_default_key_terms() {
    assert_eq!(MemoRagConfig::default().key_terms, 10);
}

#[test]
fn config_default_dim() {
    assert_eq!(MemoRagConfig::default().dim, 128);
}

#[test]
fn config_new_matches_default() {
    let a = MemoRagConfig::new();
    let b = MemoRagConfig::default();
    assert_eq!(a.gist_sentences, b.gist_sentences);
    assert_eq!(a.num_clues, b.num_clues);
    assert_eq!(a.key_terms, b.key_terms);
    assert_eq!(a.dim, b.dim);
}

#[test]
fn config_with_gist_sentences() {
    assert_eq!(
        MemoRagConfig::new().with_gist_sentences(8).gist_sentences,
        8
    );
}

#[test]
fn config_with_num_clues() {
    assert_eq!(MemoRagConfig::new().with_num_clues(5).num_clues, 5);
}

#[test]
fn config_with_key_terms() {
    assert_eq!(MemoRagConfig::new().with_key_terms(20).key_terms, 20);
}

#[test]
fn config_with_dim() {
    assert_eq!(MemoRagConfig::new().with_dim(64).dim, 64);
}

#[test]
fn config_builders_chain() {
    let c = MemoRagConfig::new()
        .with_gist_sentences(2)
        .with_num_clues(4)
        .with_key_terms(7)
        .with_dim(256);
    assert_eq!(c.gist_sentences, 2);
    assert_eq!(c.num_clues, 4);
    assert_eq!(c.key_terms, 7);
    assert_eq!(c.dim, 256);
}

#[test]
fn config_is_clone() {
    let a = MemoRagConfig::new().with_num_clues(9);
    let b = a.clone();
    assert_eq!(b.num_clues, 9);
}

// ── lexical primitives ──────────────────────────────────────────────────────────

#[test]
fn tokenize_lowercases_and_filters_short() {
    let toks = tokenize("Rust IS a Systems language, x");
    assert!(toks.contains(&"rust".to_string()));
    assert!(toks.contains(&"is".to_string()));
    assert!(toks.contains(&"systems".to_string()));
    assert!(toks.contains(&"language".to_string()));
    // Single-character tokens are dropped.
    assert!(!toks.contains(&"x".to_string()));
    assert!(!toks.contains(&"a".to_string()));
}

#[test]
fn split_sentences_handles_terminators() {
    let s = split_sentences("One. Two! Three? ");
    assert_eq!(s, vec!["One", "Two", "Three"]);
}

#[test]
fn split_sentences_empty() {
    assert!(split_sentences("   ").is_empty());
}

#[test]
fn embed_is_l2_normalised() {
    let v = embed("ownership borrow checker", 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn embed_zero_dim_is_empty() {
    assert!(embed("anything", 0).is_empty());
}

#[test]
fn embed_empty_text_is_zero() {
    let v = embed("", 64);
    assert_eq!(v.len(), 64);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn embed_is_deterministic() {
    assert_eq!(embed("memory safety", 128), embed("memory safety", 128));
}

// ── build_gist (free function) ──────────────────────────────────────────────────

#[test]
fn build_gist_respects_sentence_budget() {
    let gist = build_gist(&rust_corpus(), 5, 10);
    assert!(split_sentences(&gist.summary).len() <= 5);
}

#[test]
fn build_gist_respects_term_budget() {
    let gist = build_gist(&rust_corpus(), 5, 10);
    assert!(gist.key_terms.len() <= 10);
}

#[test]
fn build_gist_zero_sentences_empty_summary() {
    let gist = build_gist(&rust_corpus(), 0, 10);
    assert!(gist.summary.is_empty());
}

#[test]
fn build_gist_zero_terms_empty_terms() {
    let gist = build_gist(&rust_corpus(), 5, 0);
    assert!(gist.key_terms.is_empty());
}

#[test]
fn build_gist_keeps_short_corpus_whole() {
    let small = vec![doc("Alpha beta. Gamma delta.")];
    let gist = build_gist(&small, 5, 10);
    // Both sentences fit within the budget, so all of their content survives.
    // (Sentence-terminating punctuation is consumed by the splitter when the
    // selected sentences are re-joined.)
    assert!(gist.summary.contains("Alpha beta"));
    assert!(gist.summary.contains("Gamma delta"));
}

#[test]
fn build_gist_is_deterministic() {
    let a = build_gist(&rust_corpus(), 5, 10);
    let b = build_gist(&rust_corpus(), 5, 10);
    assert_eq!(a.summary, b.summary);
    assert_eq!(a.key_terms, b.key_terms);
}

// ── build_memory & gist ─────────────────────────────────────────────────────────

#[test]
fn build_memory_populates_gist() {
    let engine = built_engine();
    assert!(engine.gist().is_some());
}

#[test]
fn gist_none_before_build() {
    let engine = MemoRagEngine::new(MemoRagConfig::default());
    assert!(engine.gist().is_none());
}

#[test]
fn build_memory_summary_within_budget() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    assert!(split_sentences(&gist.summary).len() <= 5);
}

#[test]
fn build_memory_terms_within_budget() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    assert!(gist.key_terms.len() <= 10);
}

#[test]
fn key_terms_are_real_corpus_terms() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    // Build the set of all corpus tokens.
    let mut corpus_tokens: Vec<String> = Vec::new();
    for d in rust_corpus() {
        corpus_tokens.extend(tokenize(&d.content));
    }
    for term in &gist.key_terms {
        assert!(
            corpus_tokens.contains(term),
            "key term {term:?} is not a real corpus term",
        );
    }
}

#[test]
fn key_terms_are_distinct() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    let mut seen = std::collections::HashSet::new();
    for term in &gist.key_terms {
        assert!(seen.insert(term.clone()), "duplicate key term {term:?}");
    }
}

#[test]
fn gist_summary_content_comes_from_corpus() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    assert!(!gist.summary.is_empty());
    // The selected sentences are joined into the summary; each *original* corpus
    // sentence that was selected must appear verbatim as a substring.
    let mut corpus_sentences: Vec<String> = Vec::new();
    for d in rust_corpus() {
        corpus_sentences.extend(split_sentences(&d.content));
    }
    let selected: Vec<&String> = corpus_sentences
        .iter()
        .filter(|s| gist.summary.contains(s.as_str()))
        .collect();
    // At least `gist_sentences` distinct corpus sentences make up the summary.
    assert!(
        selected.len() >= 5,
        "expected >=5 corpus sentences, got {}",
        selected.len()
    );
}

#[test]
fn build_memory_indexes_corpus() {
    let engine = built_engine();
    assert_eq!(engine.len(), 4);
    assert!(!engine.is_empty());
}

#[test]
fn build_memory_empty_before() {
    let engine = MemoRagEngine::new(MemoRagConfig::default());
    assert!(engine.is_empty());
    assert_eq!(engine.len(), 0);
}

#[test]
fn build_memory_is_deterministic() {
    let a = built_engine();
    let b = built_engine();
    assert_eq!(a.gist().unwrap().summary, b.gist().unwrap().summary);
    assert_eq!(a.gist().unwrap().key_terms, b.gist().unwrap().key_terms);
}

#[test]
fn build_memory_rebuild_replaces_corpus() {
    let mut engine = built_engine();
    let small = vec![doc("Single document here.")];
    engine.build_memory(&small).unwrap();
    assert_eq!(engine.len(), 1);
}

// ── build_memory errors ─────────────────────────────────────────────────────────

#[test]
fn build_memory_empty_corpus_errors() {
    let mut engine = MemoRagEngine::new(MemoRagConfig::default());
    let err = engine.build_memory(&[]).unwrap_err();
    assert!(matches!(err, MemoRagError::EmptyCorpus));
}

// ── generate_clues ──────────────────────────────────────────────────────────────

#[test]
fn generate_clues_count_within_budget() {
    let engine = built_engine();
    let clues = engine.generate_clues("how is memory managed").unwrap();
    assert!(clues.len() <= MemoRagConfig::default().num_clues);
}

#[test]
fn generate_clues_each_contains_query() {
    let engine = built_engine();
    let query = "how is memory managed";
    let clues = engine.generate_clues(query).unwrap();
    assert!(!clues.is_empty());
    for clue in &clues {
        assert!(clue.contains(query), "clue {clue:?} missing query");
    }
}

#[test]
fn generate_clues_first_is_bare_query() {
    let engine = built_engine();
    let query = "ownership rules";
    let clues = engine.generate_clues(query).unwrap();
    assert_eq!(clues[0], query);
}

#[test]
fn generate_clues_incorporate_key_terms() {
    let engine = built_engine();
    let gist = engine.gist().unwrap();
    let clues = engine.generate_clues("how is memory managed").unwrap();
    // At least one clue beyond the bare query must add a gist key term.
    let augmented: Vec<&String> = clues.iter().skip(1).collect();
    if !augmented.is_empty() {
        let any_key_term = augmented.iter().any(|clue| {
            gist.key_terms
                .iter()
                .any(|term| clue.contains(term.as_str()))
        });
        assert!(any_key_term, "no clue incorporated a gist key term");
    }
}

#[test]
fn generate_clues_augmenting_terms_are_distinct() {
    let engine = built_engine();
    let clues = engine.generate_clues("memory").unwrap();
    // Each augmented clue should be unique (distinct augmenting term).
    let mut seen = std::collections::HashSet::new();
    for clue in &clues {
        assert!(seen.insert(clue.clone()), "duplicate clue {clue:?}");
    }
}

#[test]
fn generate_clues_trims_query() {
    let engine = built_engine();
    let clues = engine.generate_clues("   ownership   ").unwrap();
    // The bare-query clue should be the trimmed form.
    assert_eq!(clues[0], "ownership");
}

#[test]
fn generate_clues_respects_num_clues_one() {
    let mut engine = MemoRagEngine::new(MemoRagConfig::default().with_num_clues(1));
    engine.build_memory(&rust_corpus()).unwrap();
    let clues = engine.generate_clues("memory").unwrap();
    assert_eq!(clues.len(), 1);
    assert_eq!(clues[0], "memory");
}

#[test]
fn generate_clues_zero_num_clues_empty() {
    let mut engine = MemoRagEngine::new(MemoRagConfig::default().with_num_clues(0));
    engine.build_memory(&rust_corpus()).unwrap();
    assert!(engine.generate_clues("memory").unwrap().is_empty());
}

#[test]
fn generate_clues_is_deterministic() {
    let engine = built_engine();
    let a = engine.generate_clues("how is memory managed").unwrap();
    let b = engine.generate_clues("how is memory managed").unwrap();
    assert_eq!(a, b);
}

// ── generate_clues errors ───────────────────────────────────────────────────────

#[test]
fn generate_clues_empty_query_errors() {
    let engine = built_engine();
    let err = engine.generate_clues("   ").unwrap_err();
    assert!(matches!(err, MemoRagError::EmptyQuery));
}

#[test]
fn generate_clues_memory_not_built_errors() {
    let engine = MemoRagEngine::new(MemoRagConfig::default());
    let err = engine.generate_clues("memory").unwrap_err();
    assert!(matches!(err, MemoRagError::MemoryNotBuilt));
}

// ── retrieve ────────────────────────────────────────────────────────────────────

#[test]
fn retrieve_returns_top_k() {
    let engine = built_engine();
    let hits = engine.retrieve("how is memory managed", 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn retrieve_top_k_caps_at_corpus_size() {
    let engine = built_engine();
    let hits = engine.retrieve("memory", 100).unwrap();
    assert_eq!(hits.len(), 4);
}

#[test]
fn retrieve_hits_carry_winning_clue() {
    let engine = built_engine();
    let query = "how is memory managed";
    let clues = engine.generate_clues(query).unwrap();
    let hits = engine.retrieve(query, 3).unwrap();
    for hit in &hits {
        assert!(
            clues.contains(&hit.clue),
            "hit clue {:?} not among generated clues",
            hit.clue,
        );
    }
}

#[test]
fn retrieve_winning_clue_contains_query() {
    let engine = built_engine();
    let query = "ownership";
    let hits = engine.retrieve(query, 3).unwrap();
    for hit in &hits {
        assert!(hit.clue.contains(query));
    }
}

#[test]
fn retrieve_sorted_descending() {
    let engine = built_engine();
    let hits = engine.retrieve("memory ownership", 4).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].score >= pair[1].score, "hits not sorted by score");
    }
}

#[test]
fn retrieve_relevant_document_surfaces() {
    let engine = built_engine();
    // "channels" only appears in d2.
    let hits = engine.retrieve("channels messages threads", 1).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "d2");
}

#[test]
fn retrieve_scores_in_unit_range() {
    let engine = built_engine();
    let hits = engine.retrieve("memory", 4).unwrap();
    for hit in &hits {
        assert!(hit.score >= -1e-6 && hit.score <= 1.0 + 1e-6);
    }
}

#[test]
fn retrieve_is_deterministic() {
    let engine = built_engine();
    let a = engine.retrieve("how is memory managed", 3).unwrap();
    let b = engine.retrieve("how is memory managed", 3).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.document.id, y.document.id);
        assert_eq!(x.score, y.score);
        assert_eq!(x.clue, y.clue);
    }
}

#[test]
fn retrieve_zero_top_k_empty() {
    let engine = built_engine();
    assert!(engine.retrieve("memory", 0).unwrap().is_empty());
}

#[test]
fn retrieve_uses_clue_fusion() {
    // With clues disabled to a single bare-query clue, retrieval still works and
    // the winning clue is exactly the bare query.
    let mut engine = MemoRagEngine::new(MemoRagConfig::default().with_num_clues(1));
    engine.build_memory(&rust_corpus()).unwrap();
    let hits = engine.retrieve("ownership", 2).unwrap();
    for hit in &hits {
        assert_eq!(hit.clue, "ownership");
    }
}

// ── retrieve errors ─────────────────────────────────────────────────────────────

#[test]
fn retrieve_empty_query_errors() {
    let engine = built_engine();
    let err = engine.retrieve("  ", 3).unwrap_err();
    assert!(matches!(err, MemoRagError::EmptyQuery));
}

#[test]
fn retrieve_memory_not_built_errors() {
    let engine = MemoRagEngine::new(MemoRagConfig::default());
    let err = engine.retrieve("memory", 3).unwrap_err();
    assert!(matches!(err, MemoRagError::MemoryNotBuilt));
}

// ── error display ───────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(MemoRagError::EmptyCorpus.to_string(), "corpus is empty");
    assert_eq!(
        MemoRagError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(MemoRagError::MemoryNotBuilt.to_string(), "memory not built");
}
