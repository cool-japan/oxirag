//! Comprehensive tests for the `coil_retrieval` module (COIL — Contextualized
//! Inverted List, Gao, Dai, Callan 2021).

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::default_trait_access
)]

use crate::coil_retrieval::engine::CoilRetriever;
use crate::coil_retrieval::index::CoilInvertedIndex;
use crate::coil_retrieval::types::{
    CoilConfig, CoilDocument, CoilError, CoilPosting, CoilScoreMode, CoilTokenVector,
};
use crate::types::{Document, DocumentId};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

fn tok_config() -> CoilConfig {
    CoilConfig::new().with_score_mode(CoilScoreMode::Tok)
}

fn build(config: CoilConfig, docs: &[Document]) -> CoilRetriever {
    let mut retriever = CoilRetriever::new(config);
    retriever.index(docs).unwrap();
    retriever
}

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn find_score(hits: &[(DocumentId, f32)], id: &str) -> Option<f32> {
    hits.iter()
        .find(|(doc_id, _)| doc_id.as_str() == id)
        .map(|(_, score)| *score)
}

fn ids(hits: &[(DocumentId, f32)]) -> Vec<String> {
    hits.iter().map(|(id, _)| id.as_str().to_string()).collect()
}

fn gating_corpus() -> Vec<Document> {
    vec![
        doc("rust", "memory safety ownership"),
        doc("python", "dynamic scripting language"),
        doc("ocean", "coral reefs whales"),
    ]
}

// ── CoilScoreMode ─────────────────────────────────────────────────────────────

#[test]
fn score_mode_as_str() {
    assert_eq!(CoilScoreMode::Tok.as_str(), "tok");
    assert_eq!(CoilScoreMode::Full.as_str(), "full");
}

#[test]
fn score_mode_includes_cls() {
    assert!(!CoilScoreMode::Tok.includes_cls());
    assert!(CoilScoreMode::Full.includes_cls());
}

#[test]
fn score_mode_default_is_full() {
    assert_eq!(CoilScoreMode::default(), CoilScoreMode::Full);
}

// ── CoilConfig ────────────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = CoilConfig::default();
    assert_eq!(config.dim, 128);
    assert_eq!(config.context_window, 2);
    assert_eq!(config.context_weight, 0.5);
    assert_eq!(config.lambda, 1.0);
    assert_eq!(config.score_mode, CoilScoreMode::Full);
    assert_eq!(config.top_k, 10);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(CoilConfig::new(), CoilConfig::default());
}

#[test]
fn config_builders_set_each_field() {
    let config = CoilConfig::new()
        .with_dim(64)
        .with_context_window(3)
        .with_context_weight(0.25)
        .with_lambda(2.5)
        .with_score_mode(CoilScoreMode::Tok)
        .with_top_k(7);
    assert_eq!(config.dim, 64);
    assert_eq!(config.context_window, 3);
    assert_eq!(config.context_weight, 0.25);
    assert_eq!(config.lambda, 2.5);
    assert_eq!(config.score_mode, CoilScoreMode::Tok);
    assert_eq!(config.top_k, 7);
}

#[test]
fn config_builder_is_pure() {
    let base = CoilConfig::new();
    let modified = base.clone().with_dim(999);
    assert_eq!(base.dim, 128);
    assert_eq!(modified.dim, 999);
}

// ── CoilTokenVector ───────────────────────────────────────────────────────────

#[test]
fn token_vector_new_fields() {
    let tv = CoilTokenVector::new("cat", 3, vec![1.0, 2.0, 3.0]);
    assert_eq!(tv.surface, "cat");
    assert_eq!(tv.position, 3);
    assert_eq!(tv.vector, vec![1.0, 2.0, 3.0]);
}

#[test]
fn token_vector_dim() {
    let tv = CoilTokenVector::new("cat", 0, vec![0.0; 16]);
    assert_eq!(tv.dim(), 16);
}

#[test]
fn token_vector_dot_self_is_one() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta gamma")]);
    let (tokens, _) = retriever.encode_query("alpha").unwrap();
    let self_dot = tokens[0].dot(&tokens[0]);
    assert!((self_dot - 1.0).abs() < 1e-4, "self dot was {self_dot}");
}

#[test]
fn token_vector_dot_distinct_tokens_below_one() {
    let retriever = build(tok_config(), &[doc("d1", "alpha")]);
    let (a, _) = retriever.encode_query("alpha").unwrap();
    let (b, _) = retriever.encode_query("omega").unwrap();
    let cross = a[0].dot(&b[0]);
    assert!(cross < 0.9, "cross dot was {cross}");
}

#[test]
fn token_vector_dot_mismatched_length_uses_shorter() {
    let a = CoilTokenVector::new("a", 0, vec![1.0, 0.0, 0.0]);
    let b = CoilTokenVector::new("b", 0, vec![1.0, 2.0]);
    assert_eq!(a.dot(&b), 1.0);
}

// ── CoilPosting ───────────────────────────────────────────────────────────────

#[test]
fn posting_new_and_surface() {
    let tv = CoilTokenVector::new("dog", 2, vec![0.1, 0.2]);
    let posting = CoilPosting::new(DocumentId::from_string("d1"), tv.clone());
    assert_eq!(posting.doc_id, DocumentId::from_string("d1"));
    assert_eq!(posting.surface(), "dog");
    assert_eq!(posting.token_vector, tv);
}

// ── CoilDocument ──────────────────────────────────────────────────────────────

#[test]
fn document_new_len_and_is_empty() {
    let tv = CoilTokenVector::new("dog", 0, vec![1.0]);
    let d = CoilDocument::new(DocumentId::from_string("d1"), vec![tv], vec![1.0]);
    assert_eq!(d.len(), 1);
    assert!(!d.is_empty());
}

#[test]
fn document_empty_token_vectors() {
    let d = CoilDocument::new(DocumentId::from_string("d1"), Vec::new(), vec![0.0, 0.0]);
    assert_eq!(d.len(), 0);
    assert!(d.is_empty());
}

// ── CoilInvertedIndex ─────────────────────────────────────────────────────────

#[test]
fn index_new_is_empty() {
    let index = CoilInvertedIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.num_documents(), 0);
    assert_eq!(index.num_terms(), 0);
    assert_eq!(index.num_postings(), 0);
}

#[test]
fn index_insert_round_trips_postings() {
    let mut index = CoilInvertedIndex::new();
    let tv = CoilTokenVector::new("cat", 0, vec![1.0, 0.0]);
    let cls = vec![0.5, 0.5];
    let document = CoilDocument::new(DocumentId::from_string("d1"), vec![tv.clone()], cls.clone());
    index.insert_document(&document);

    let postings = index.postings_for("cat");
    assert_eq!(postings.len(), 1);
    assert_eq!(postings[0].doc_id, DocumentId::from_string("d1"));
    assert_eq!(postings[0].token_vector, tv);
    assert_eq!(
        index.cls_for(&DocumentId::from_string("d1")),
        Some(cls.as_slice())
    );
}

#[test]
fn index_counts_documents_terms_postings() {
    let mut index = CoilInvertedIndex::new();
    let d1 = CoilDocument::new(
        DocumentId::from_string("d1"),
        vec![
            CoilTokenVector::new("cat", 0, vec![1.0]),
            CoilTokenVector::new("dog", 1, vec![1.0]),
        ],
        vec![1.0],
    );
    let d2 = CoilDocument::new(
        DocumentId::from_string("d2"),
        vec![CoilTokenVector::new("cat", 0, vec![1.0])],
        vec![1.0],
    );
    index.insert_document(&d1);
    index.insert_document(&d2);
    assert_eq!(index.num_documents(), 2);
    assert_eq!(index.num_terms(), 2); // "cat", "dog"
    assert_eq!(index.num_postings(), 3); // cat x2 + dog x1
}

#[test]
fn index_contains_token() {
    let mut index = CoilInvertedIndex::new();
    index.insert_document(&CoilDocument::new(
        DocumentId::from_string("d1"),
        vec![CoilTokenVector::new("cat", 0, vec![1.0])],
        vec![1.0],
    ));
    assert!(index.contains_token("cat"));
    assert!(!index.contains_token("dog"));
}

#[test]
fn index_postings_for_missing_returns_empty() {
    let index = CoilInvertedIndex::new();
    assert!(index.postings_for("nothing").is_empty());
    assert_eq!(index.cls_for(&DocumentId::from_string("x")), None);
}

#[test]
fn index_doc_ids_insertion_order() {
    let mut index = CoilInvertedIndex::new();
    for id in ["zeta", "alpha", "mu"] {
        index.insert_document(&CoilDocument::new(
            DocumentId::from_string(id),
            vec![CoilTokenVector::new("cat", 0, vec![1.0])],
            vec![1.0],
        ));
    }
    let order: Vec<&str> = index.doc_ids().iter().map(DocumentId::as_str).collect();
    assert_eq!(order, vec!["zeta", "alpha", "mu"]);
}

#[test]
fn index_multiple_occurrences_yield_multiple_postings() {
    let mut index = CoilInvertedIndex::new();
    index.insert_document(&CoilDocument::new(
        DocumentId::from_string("d1"),
        vec![
            CoilTokenVector::new("cat", 0, vec![1.0]),
            CoilTokenVector::new("cat", 1, vec![0.0, 1.0]),
        ],
        vec![1.0],
    ));
    let postings = index.postings_for("cat");
    assert_eq!(postings.len(), 2);
    assert_eq!(postings[0].token_vector.position, 0);
    assert_eq!(postings[1].token_vector.position, 1);
    assert_eq!(index.num_terms(), 1);
    assert_eq!(index.num_postings(), 2);
}

#[test]
fn index_reinsert_same_id_keeps_single_document() {
    let mut index = CoilInvertedIndex::new();
    let document = CoilDocument::new(
        DocumentId::from_string("d1"),
        vec![CoilTokenVector::new("cat", 0, vec![1.0])],
        vec![1.0],
    );
    index.insert_document(&document);
    index.insert_document(&document);
    assert_eq!(index.num_documents(), 1);
    // Postings are appended (documented contract: clear before rebuilding).
    assert_eq!(index.postings_for("cat").len(), 2);
}

#[test]
fn index_clear_resets_everything() {
    let mut index = CoilInvertedIndex::new();
    index.insert_document(&CoilDocument::new(
        DocumentId::from_string("d1"),
        vec![CoilTokenVector::new("cat", 0, vec![1.0])],
        vec![1.0],
    ));
    assert!(!index.is_empty());
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.num_terms(), 0);
    assert_eq!(index.num_postings(), 0);
    assert!(index.doc_ids().is_empty());
}

// ── Encoding ──────────────────────────────────────────────────────────────────

#[test]
fn encode_document_token_count() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta gamma")]);
    let encoded = retriever.encode_document(&doc("d1", "alpha beta gamma"));
    assert_eq!(encoded.token_vectors.len(), 3);
    assert_eq!(encoded.doc_id, DocumentId::from_string("d1"));
}

#[test]
fn encode_document_positions_are_sequential() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta gamma")]);
    let encoded = retriever.encode_document(&doc("d1", "alpha beta gamma"));
    let positions: Vec<usize> = encoded.token_vectors.iter().map(|tv| tv.position).collect();
    assert_eq!(positions, vec![0, 1, 2]);
}

#[test]
fn encode_document_cls_is_normalized() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta gamma delta")]);
    let encoded = retriever.encode_document(&doc("d1", "alpha beta gamma delta"));
    assert!((norm(&encoded.cls_vector) - 1.0).abs() < 1e-4);
}

#[test]
fn encode_document_token_vectors_are_normalized() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta gamma delta")]);
    let encoded = retriever.encode_document(&doc("d1", "alpha beta gamma delta"));
    for tv in &encoded.token_vectors {
        assert!((norm(&tv.vector) - 1.0).abs() < 1e-4);
    }
}

#[test]
fn encode_query_empty_errors() {
    let retriever = build(tok_config(), &gating_corpus());
    assert_eq!(retriever.encode_query(""), Err(CoilError::EmptyQuery));
}

#[test]
fn encode_query_whitespace_and_punctuation_errors() {
    let retriever = build(tok_config(), &gating_corpus());
    assert_eq!(retriever.encode_query("   "), Err(CoilError::EmptyQuery));
    assert_eq!(retriever.encode_query("!!! ??"), Err(CoilError::EmptyQuery));
}

#[test]
fn encode_query_ok_returns_tokens_and_cls() {
    let retriever = build(tok_config(), &gating_corpus());
    let (tokens, cls) = retriever.encode_query("memory safety").unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].surface, "memory");
    assert_eq!(tokens[1].surface, "safety");
    assert_eq!(cls.len(), retriever.config().dim);
}

#[test]
fn encode_is_deterministic() {
    let retriever = build(tok_config(), &[doc("d1", "alpha beta")]);
    let first = retriever.encode_document(&doc("d1", "alpha beta gamma"));
    let second = retriever.encode_document(&doc("d1", "alpha beta gamma"));
    assert_eq!(first, second);
}

#[test]
fn encode_query_lowercases_surface() {
    let retriever = build(tok_config(), &[doc("d1", "alpha")]);
    let (tokens, _) = retriever.encode_query("ALPHA").unwrap();
    assert_eq!(tokens[0].surface, "alpha");
}

// ── Polysemy (contextualisation) ──────────────────────────────────────────────

fn bank_vector(retriever: &CoilRetriever, content: &str) -> CoilTokenVector {
    let encoded = retriever.encode_document(&doc("d", content));
    encoded
        .token_vectors
        .into_iter()
        .find(|tv| tv.surface == "bank")
        .expect("content must contain the surface token 'bank'")
}

#[test]
fn polysemy_same_token_different_context_differs() {
    let retriever = build(tok_config(), &[doc("d1", "seed")]);
    let river = bank_vector(&retriever, "river bank flows water");
    let money = bank_vector(&retriever, "money bank loan interest");
    assert_ne!(river.vector, money.vector);
    let similarity = river.dot(&money);
    assert!(similarity < 0.999, "similarity was {similarity}");
}

#[test]
fn polysemy_same_token_same_context_is_identical() {
    let retriever = build(tok_config(), &[doc("d1", "seed")]);
    let first = bank_vector(&retriever, "river bank flows water");
    let second = bank_vector(&retriever, "river bank flows water");
    assert_eq!(first.vector, second.vector);
    assert!((first.dot(&second) - 1.0).abs() < 1e-4);
}

#[test]
fn context_weight_zero_disables_contextualisation() {
    // With no context blending, the same surface token is context-independent.
    let retriever = build(
        CoilConfig::new()
            .with_score_mode(CoilScoreMode::Tok)
            .with_context_weight(0.0),
        &[doc("d1", "seed")],
    );
    let river = bank_vector(&retriever, "river bank flows water");
    let money = bank_vector(&retriever, "money bank loan interest");
    assert_eq!(river.vector, money.vector);
}

// ── Retriever lifecycle / error paths ─────────────────────────────────────────

#[test]
fn search_before_index_errors() {
    let retriever = CoilRetriever::new(CoilConfig::default());
    assert!(!retriever.is_indexed());
    assert_eq!(retriever.search("alpha", 5), Err(CoilError::NotIndexed));
}

#[test]
fn index_empty_corpus_errors() {
    let mut retriever = CoilRetriever::new(CoilConfig::default());
    assert_eq!(retriever.index(&[]), Err(CoilError::EmptyCorpus));
    assert!(!retriever.is_indexed());
}

#[test]
fn search_empty_query_errors() {
    let retriever = build(CoilConfig::default(), &gating_corpus());
    assert_eq!(retriever.search("", 5), Err(CoilError::EmptyQuery));
}

#[test]
fn search_whitespace_query_errors() {
    let retriever = build(CoilConfig::default(), &gating_corpus());
    assert_eq!(retriever.search("   ", 5), Err(CoilError::EmptyQuery));
}

#[test]
fn index_populates_retriever_state() {
    let retriever = build(CoilConfig::default(), &gating_corpus());
    assert!(retriever.is_indexed());
    assert_eq!(retriever.len(), 3);
    assert!(!retriever.is_empty());
    assert!(retriever.num_terms() > 0);
}

#[test]
fn reindex_is_idempotent_for_fixed_corpus() {
    let mut retriever = CoilRetriever::new(CoilConfig::default());
    retriever.index(&gating_corpus()).unwrap();
    let terms_first = retriever.num_terms();
    let postings_first = retriever.inverted_index().num_postings();
    retriever.index(&gating_corpus()).unwrap();
    assert_eq!(retriever.num_terms(), terms_first);
    assert_eq!(retriever.inverted_index().num_postings(), postings_first);
    assert_eq!(retriever.len(), 3);
}

// ── Exact-token gating ────────────────────────────────────────────────────────

#[test]
fn tok_gating_scores_only_docs_with_surface_token() {
    let retriever = build(tok_config(), &gating_corpus());
    let hits = retriever.search("memory safety", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0.as_str(), "rust");
}

#[test]
fn tok_oov_only_query_returns_empty() {
    let retriever = build(tok_config(), &gating_corpus());
    let hits = retriever.search("quantum entanglement", 10).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn tok_query_token_contributes_only_to_matching_docs() {
    let corpus = vec![
        doc("d1", "alpha beta"),
        doc("d2", "alpha gamma"),
        doc("d3", "delta epsilon"),
    ];
    let retriever = build(tok_config(), &corpus);
    // "beta" only occurs in d1.
    let hits = retriever.search("beta", 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0.as_str(), "d1");
}

#[test]
fn full_oov_only_query_returns_docs_via_cls() {
    let retriever = build(CoilConfig::new().with_lambda(3.0), &gating_corpus());
    let hits = retriever.search("quantum entanglement", 10).unwrap();
    // In COIL-full every document is scored through the CLS term.
    assert_eq!(hits.len(), 3);
}

#[test]
fn full_scores_non_lexical_document_that_tok_excludes() {
    let corpus = vec![
        doc("d1", "alpha beta gamma"),
        doc("d2", "alpha delta epsilon"),
        doc("d3", "zeta eta theta"),
    ];
    let tok = build(tok_config(), &corpus);
    let full = build(CoilConfig::new().with_lambda(5.0), &corpus);

    let tok_hits = tok.search("alpha", 10).unwrap();
    let full_hits = full.search("alpha", 10).unwrap();

    // d3 shares no surface token with the query.
    assert!(!ids(&tok_hits).contains(&"d3".to_string()));
    assert!(ids(&tok_hits).contains(&"d1".to_string()));
    assert!(ids(&full_hits).contains(&"d3".to_string()));
}

// ── Scoring correctness ───────────────────────────────────────────────────────

#[test]
fn multi_token_score_is_sum_of_per_token_maxsims() {
    // "alpha" occurs twice with different contexts, so its contribution is a
    // genuine max over two postings.
    let retriever = build(tok_config(), &[doc("d1", "alpha beta alpha gamma")]);
    let (query_tokens, _) = retriever.encode_query("alpha beta").unwrap();
    let index = retriever.inverted_index();

    let mut expected = 0.0f32;
    for query_token in &query_tokens {
        let mut best = f32::NEG_INFINITY;
        for posting in index.postings_for(&query_token.surface) {
            let similarity = query_token.dot(&posting.token_vector);
            if similarity > best {
                best = similarity;
            }
        }
        if best > f32::NEG_INFINITY {
            expected += best;
        }
    }

    let hits = retriever.search("alpha beta", 10).unwrap();
    let score = find_score(&hits, "d1").unwrap();
    assert!(
        (score - expected).abs() < 1e-5,
        "score {score} vs {expected}"
    );
}

#[test]
fn ranking_prefers_documents_matching_more_query_tokens() {
    let corpus = vec![
        doc("rust", "memory safety without garbage collector ownership"),
        doc(
            "borrow",
            "borrow checker enforces memory safety compile time",
        ),
        doc("python", "dynamic scripting language data science"),
        doc("java", "garbage collection reclaims memory java runtime"),
    ];
    let retriever = build(tok_config(), &corpus);
    let hits = retriever.search("memory safety", 10).unwrap();
    assert!(!hits.is_empty());
    let top = hits[0].0.as_str();
    assert!(top == "rust" || top == "borrow", "top was {top}");
    // python shares no query token; it must not appear.
    assert!(!ids(&hits).contains(&"python".to_string()));
}

#[test]
fn tok_and_full_scores_differ_by_cls_term() {
    let corpus = vec![doc("d1", "alpha beta gamma"), doc("d2", "alpha delta")];
    let tok = build(tok_config(), &corpus);
    let full = build(CoilConfig::new().with_lambda(5.0), &corpus);
    let tok_score = find_score(&tok.search("alpha", 10).unwrap(), "d1").unwrap();
    let full_score = find_score(&full.search("alpha", 10).unwrap(), "d1").unwrap();
    assert!(
        (tok_score - full_score).abs() > 1e-6,
        "tok {tok_score} vs full {full_score}"
    );
}

#[test]
fn search_respects_k_truncation() {
    let corpus = vec![
        doc("d1", "alpha one"),
        doc("d2", "alpha two"),
        doc("d3", "alpha three"),
        doc("d4", "alpha four"),
    ];
    let retriever = build(CoilConfig::default(), &corpus);
    assert_eq!(retriever.search("alpha", 2).unwrap().len(), 2);
    assert_eq!(retriever.search("alpha", 10).unwrap().len(), 4);
}

#[test]
fn search_results_are_sorted_descending() {
    let corpus = vec![
        doc("d1", "alpha one"),
        doc("d2", "alpha two"),
        doc("d3", "alpha three"),
    ];
    let retriever = build(CoilConfig::default(), &corpus);
    let hits = retriever.search("alpha", 10).unwrap();
    for pair in hits.windows(2) {
        assert!(pair[0].1 >= pair[1].1, "not sorted: {hits:?}");
    }
}

#[test]
fn search_default_uses_config_top_k() {
    let corpus = vec![
        doc("d1", "alpha one"),
        doc("d2", "alpha two"),
        doc("d3", "alpha three"),
        doc("d4", "alpha four"),
    ];
    let retriever = build(CoilConfig::new().with_top_k(2), &corpus);
    assert_eq!(retriever.search_default("alpha").unwrap().len(), 2);
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn search_is_deterministic_across_retrievers() {
    let corpus = vec![
        doc("d1", "alpha beta gamma"),
        doc("d2", "alpha delta epsilon"),
        doc("d3", "beta gamma zeta"),
    ];
    let first = build(CoilConfig::default(), &corpus);
    let second = build(CoilConfig::default(), &corpus);
    assert_eq!(
        first.search("alpha beta", 10).unwrap(),
        second.search("alpha beta", 10).unwrap()
    );
}

#[test]
fn search_is_deterministic_when_repeated() {
    let retriever = build(CoilConfig::default(), &gating_corpus());
    let first = retriever.search("memory safety", 10).unwrap();
    let second = retriever.search("memory safety", 10).unwrap();
    assert_eq!(first, second);
}

// ── CoilError ─────────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(CoilError::EmptyCorpus.to_string(), "corpus is empty");
    assert_eq!(CoilError::EmptyQuery.to_string(), "query must not be empty");
    assert_eq!(CoilError::NotIndexed.to_string(), "retriever not indexed");
}

#[test]
fn error_equality() {
    assert_eq!(CoilError::EmptyQuery, CoilError::EmptyQuery);
    assert_ne!(CoilError::EmptyQuery, CoilError::EmptyCorpus);
}

// ── End-to-end ────────────────────────────────────────────────────────────────

#[test]
fn end_to_end_full_pipeline() {
    let corpus = vec![
        doc(
            "rust",
            "Rust delivers memory safety without a garbage collector.",
        ),
        doc("python", "Python is a dynamic scripting language."),
        doc(
            "db",
            "The database index accelerates lookups over large tables.",
        ),
    ];
    let retriever = build(CoilConfig::default(), &corpus);
    let hits = retriever.search("memory safety", 3).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0.as_str(), "rust");
}

#[test]
fn search_handles_k_zero() {
    let retriever = build(CoilConfig::default(), &gating_corpus());
    assert!(retriever.search("memory", 0).unwrap().is_empty());
}
