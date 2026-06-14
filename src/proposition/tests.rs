//! Tests for the `proposition` module.
#![allow(clippy::float_cmp)]

use super::extractor::{
    HeuristicPropositionExtractor, PropositionExtractor, embed, split_clauses, split_sentences,
    tokenize,
};
use super::index::PropositionIndex;
use super::types::{PropositionConfig, PropositionError};
use crate::types::{Document, DocumentId};

// ── helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

// ── Config defaults & builders ─────────────────────────────────────────────────

#[test]
fn test_config_default_min_tokens() {
    let cfg = PropositionConfig::default();
    assert_eq!(cfg.min_tokens, 3, "default min_tokens should be 3");
}

#[test]
fn test_config_default_max_propositions() {
    let cfg = PropositionConfig::default();
    assert_eq!(
        cfg.max_propositions_per_doc, 64,
        "default max_propositions_per_doc should be 64"
    );
}

#[test]
fn test_config_default_resolve_pronouns() {
    let cfg = PropositionConfig::default();
    assert!(
        cfg.resolve_pronouns,
        "default resolve_pronouns should be true"
    );
}

#[test]
fn test_config_default_dim() {
    let cfg = PropositionConfig::default();
    assert_eq!(cfg.dim, 128, "default dim should be 128");
}

#[test]
fn test_config_new_equals_default() {
    let cfg = PropositionConfig::new();
    assert_eq!(cfg.dim, 128, "new() should match default dim");
}

#[test]
fn test_config_with_min_tokens() {
    let cfg = PropositionConfig::new().with_min_tokens(5);
    assert_eq!(cfg.min_tokens, 5, "with_min_tokens should set min_tokens");
}

#[test]
fn test_config_with_max_propositions() {
    let cfg = PropositionConfig::new().with_max_propositions_per_doc(7);
    assert_eq!(
        cfg.max_propositions_per_doc, 7,
        "with_max_propositions_per_doc should set the cap"
    );
}

#[test]
fn test_config_with_resolve_pronouns() {
    let cfg = PropositionConfig::new().with_resolve_pronouns(false);
    assert!(
        !cfg.resolve_pronouns,
        "with_resolve_pronouns(false) should disable resolution"
    );
}

#[test]
fn test_config_with_dim() {
    let cfg = PropositionConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64, "with_dim should set the dimension");
}

#[test]
fn test_config_builder_chaining() {
    let cfg = PropositionConfig::new()
        .with_min_tokens(2)
        .with_dim(32)
        .with_max_propositions_per_doc(10)
        .with_resolve_pronouns(false);
    assert_eq!(cfg.min_tokens, 2, "chained builders should compose");
}

// ── Embedding ──────────────────────────────────────────────────────────────────

#[test]
fn test_embed_dim() {
    let emb = embed("hello world foo", 128);
    assert_eq!(emb.len(), 128, "embedding length should equal dim");
}

#[test]
fn test_embed_l2_normalised() {
    let emb = embed("rust is a systems language", 64);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "non-empty embedding should be unit-normalised: norm={norm}"
    );
}

#[test]
fn test_embed_empty_is_zero() {
    let emb = embed("", 32);
    assert!(
        emb.iter().all(|x| *x == 0.0),
        "empty text yields zero vector"
    );
}

#[test]
fn test_embed_deterministic() {
    let a = embed("alpha beta gamma", 128);
    let b = embed("alpha beta gamma", 128);
    assert_eq!(a, b, "embedding must be deterministic for identical input");
}

#[test]
fn test_embed_zero_dim() {
    let emb = embed("anything", 0);
    assert!(emb.is_empty(), "dim=0 should yield an empty embedding");
}

// ── Tokeniser & splitters ──────────────────────────────────────────────────────

#[test]
fn test_tokenize_drops_short_tokens() {
    let toks = tokenize("a bb ccc d");
    assert_eq!(toks, vec!["bb", "ccc"], "tokens < 2 chars must be dropped");
}

#[test]
fn test_tokenize_lowercases() {
    let toks = tokenize("Rust LANG");
    assert_eq!(toks, vec!["rust", "lang"], "tokens must be lowercased");
}

#[test]
fn test_split_sentences_basic() {
    let s = split_sentences("First one. Second one! Third?");
    assert_eq!(
        s.len(),
        3,
        "three sentence terminators yield three sentences"
    );
}

#[test]
fn test_split_sentences_drops_empty() {
    let s = split_sentences("Hello...   ");
    assert_eq!(s, vec!["Hello"], "empty fragments must be dropped");
}

#[test]
fn test_split_clauses_on_and() {
    let c = split_clauses("rust is fast and rust is safe");
    assert_eq!(
        c.len(),
        2,
        "an ' and ' boundary should split into two clauses"
    );
}

#[test]
fn test_split_clauses_on_comma() {
    let c = split_clauses("rust is fast, python is slow");
    assert_eq!(c.len(), 2, "a ', ' boundary should split into two clauses");
}

#[test]
fn test_split_clauses_no_boundary() {
    let c = split_clauses("a single clause here");
    assert_eq!(c.len(), 1, "a clause with no boundary stays whole");
}

// ── Extractor: compound splitting ──────────────────────────────────────────────

#[test]
fn test_extract_compound_sentence_yields_multiple() {
    let cfg = PropositionConfig::new().with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc(
        "d1",
        "Rust ensures memory safety and Rust avoids data races.",
    );
    let props = ext.extract(&d);
    assert!(
        props.len() > 1,
        "a compound sentence should yield more than one proposition: {props:?}"
    );
}

#[test]
fn test_extract_which_clause_splits() {
    let cfg = PropositionConfig::new().with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc(
        "d1",
        "Caching speeds up retrieval which lowers total latency.",
    );
    let props = ext.extract(&d);
    assert_eq!(
        props.len(),
        2,
        "a ' which ' clause should split into two: {props:?}"
    );
}

#[test]
fn test_extract_min_tokens_drops_tiny_clause() {
    let cfg = PropositionConfig::new()
        .with_min_tokens(4)
        .with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    // Second clause "it works" has only 2 tokens and must be dropped.
    let d = doc(
        "d1",
        "The retrieval pipeline indexes documents efficiently and it works.",
    );
    let props = ext.extract(&d);
    assert!(
        props.iter().all(|p| tokenize(p).len() >= 4),
        "no proposition should have fewer than min_tokens tokens: {props:?}"
    );
}

#[test]
fn test_extract_min_tokens_count() {
    let cfg = PropositionConfig::new()
        .with_min_tokens(5)
        .with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "This long enough clause survives and tiny one.");
    let props = ext.extract(&d);
    assert_eq!(
        props.len(),
        1,
        "only the long clause should remain: {props:?}"
    );
}

// ── Extractor: pronoun resolution ──────────────────────────────────────────────

#[test]
fn test_extract_resolves_leading_pronoun() {
    let cfg = PropositionConfig::new().with_min_tokens(2);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "Bm25 ranks documents. It improves recall.");
    let props = ext.extract(&d);
    assert!(
        props
            .iter()
            .any(|p| p.starts_with("Bm25") && p.contains("improves recall")),
        "leading pronoun should be replaced by the subject: {props:?}"
    );
}

#[test]
fn test_extract_resolution_drops_pronoun_token() {
    let cfg = PropositionConfig::new().with_min_tokens(2);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "Rerankers refine results. It boosts precision.");
    let props = ext.extract(&d);
    assert!(
        props.iter().any(|p| p == "Rerankers boosts precision"),
        "the pronoun 'It' should be swapped for 'Rerankers': {props:?}"
    );
}

#[test]
fn test_extract_no_resolution_when_disabled() {
    let cfg = PropositionConfig::new()
        .with_min_tokens(2)
        .with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "Bm25 ranks documents. It improves recall.");
    let props = ext.extract(&d);
    assert!(
        props.iter().any(|p| p == "It improves recall"),
        "with resolution disabled the pronoun is kept verbatim: {props:?}"
    );
}

#[test]
fn test_extract_subject_falls_back_to_title() {
    let cfg = PropositionConfig::new().with_min_tokens(2);
    let ext = HeuristicPropositionExtractor::new(cfg);
    // No capitalised content token; subject must come from the title.
    let d = Document::new("it improves recall greatly")
        .with_id(DocumentId::from_string("d1"))
        .with_title("Retriever");
    let props = ext.extract(&d);
    assert!(
        props.iter().any(|p| p.starts_with("Retriever")),
        "subject should fall back to the document title: {props:?}"
    );
}

#[test]
fn test_extract_non_pronoun_start_unchanged() {
    let cfg = PropositionConfig::new().with_min_tokens(2);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "Vectors encode meaning compactly.");
    let props = ext.extract(&d);
    assert!(
        props
            .iter()
            .any(|p| p == "Vectors encode meaning compactly"),
        "clauses not starting with a pronoun stay unchanged: {props:?}"
    );
}

// ── Extractor: cap ─────────────────────────────────────────────────────────────

#[test]
fn test_extract_respects_max_cap() {
    let cfg = PropositionConfig::new()
        .with_min_tokens(2)
        .with_max_propositions_per_doc(2)
        .with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc(
        "d1",
        "Clause one here and clause two here and clause three here and clause four here.",
    );
    let props = ext.extract(&d);
    assert_eq!(
        props.len(),
        2,
        "extraction must honour the max cap: {props:?}"
    );
}

#[test]
fn test_extract_with_positions_tracks_sentence() {
    let cfg = PropositionConfig::new().with_resolve_pronouns(false);
    let ext = HeuristicPropositionExtractor::new(cfg);
    let d = doc("d1", "First sentence is here. Second sentence is here.");
    let pairs = ext.extract_with_positions(&d);
    assert!(
        pairs.iter().any(|(idx, _)| *idx == 1),
        "second-sentence propositions should carry index 1: {pairs:?}"
    );
}

// ── Index: structural ──────────────────────────────────────────────────────────

#[test]
fn test_index_starts_empty() {
    let idx = PropositionIndex::new(PropositionConfig::default());
    assert!(idx.is_empty(), "a fresh index should be empty");
}

#[test]
fn test_index_len_zero_initially() {
    let idx = PropositionIndex::new(PropositionConfig::default());
    assert_eq!(idx.len(), 0, "a fresh index should have zero documents");
}

#[test]
fn test_index_add_document_not_empty() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Rust is fast and Rust is safe."));
    assert!(
        !idx.is_empty(),
        "index should be non-empty after adding a document"
    );
}

#[test]
fn test_index_len_counts_distinct_docs() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Rust is fast and Rust is safe."));
    idx.add_document(&doc("d2", "Python is slow but Python is simple."));
    assert_eq!(idx.len(), 2, "len should count distinct parent documents");
}

#[test]
fn test_index_proposition_count_positive() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Rust is fast and Rust is safe."));
    assert!(
        idx.proposition_count() >= 2,
        "a compound document should produce multiple propositions"
    );
}

#[test]
fn test_index_add_documents_batch() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    let docs = vec![
        doc("d1", "Rust is fast and Rust is safe."),
        doc("d2", "Vectors encode meaning compactly here."),
    ];
    idx.add_documents(&docs);
    assert_eq!(idx.len(), 2, "add_documents should index every document");
}

#[test]
fn test_index_proposition_ids_sequential() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Alpha runs fast and beta runs slow."));
    let ids: Vec<usize> = idx.propositions().iter().map(|p| p.id).collect();
    assert_eq!(
        ids,
        (0..ids.len()).collect::<Vec<_>>(),
        "proposition ids should be assigned sequentially: {ids:?}"
    );
}

#[test]
fn test_index_proposition_parent_id_preserved() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "docX",
        "Retrieval improves answers and grounds them well.",
    ));
    assert!(
        idx.propositions()
            .iter()
            .all(|p| p.parent_id.as_str() == "docX"),
        "every proposition should record its parent document id"
    );
}

// ── Index: search ──────────────────────────────────────────────────────────────

#[test]
fn test_search_returns_hits() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "Caching reduces latency and improves throughput.",
    ));
    let hits = idx
        .search("latency caching", 5)
        .expect("search should succeed");
    assert!(!hits.is_empty(), "search should return at least one hit");
}

#[test]
fn test_search_sorted_descending() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "Caching reduces latency greatly and gardening grows tomatoes well.",
    ));
    let hits = idx.search("latency caching reduces", 5).expect("search ok");
    let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "hit scores must be sorted descending: {scores:?}"
    );
}

#[test]
fn test_search_top_k_respected() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "One clause here and two clause here and three clause here and four clause here.",
    ));
    let hits = idx.search("clause", 2).expect("search ok");
    assert_eq!(hits.len(), 2, "search must respect top_k");
}

#[test]
fn test_search_best_hit_is_relevant() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "Quantum entanglement links particles and breakfast cereal tastes sweet.",
    ));
    let hits = idx
        .search("quantum entanglement particles", 5)
        .expect("search ok");
    assert!(
        hits[0].proposition.text.to_lowercase().contains("quantum"),
        "the top hit should be the quantum clause: {:?}",
        hits[0].proposition.text
    );
}

#[test]
fn test_search_empty_query_errors() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Some content here is present."));
    let err = idx.search("   ", 5).expect_err("blank query should error");
    assert!(
        matches!(err, PropositionError::EmptyQuery),
        "blank query should yield EmptyQuery: {err:?}"
    );
}

#[test]
fn test_search_empty_corpus_errors() {
    let idx = PropositionIndex::new(PropositionConfig::default());
    let err = idx
        .search("anything", 5)
        .expect_err("empty corpus should error");
    assert!(
        matches!(err, PropositionError::EmptyCorpus),
        "empty corpus should yield EmptyCorpus: {err:?}"
    );
}

#[test]
fn test_search_top_k_zero_returns_none() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Some content here is present and useful."));
    let hits = idx.search("content", 0).expect("search ok");
    assert!(hits.is_empty(), "top_k=0 should return no hits");
}

// ── Index: search_documents ────────────────────────────────────────────────────

#[test]
fn test_search_documents_dedupes_to_best() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    // Two propositions about latency from the SAME document.
    idx.add_document(&doc(
        "d1",
        "Latency matters for retrieval and latency affects user experience.",
    ));
    let docs = idx.search_documents("latency", 5).expect("search ok");
    let count = docs.iter().filter(|(id, _)| id.as_str() == "d1").count();
    assert_eq!(count, 1, "a document must appear at most once: {docs:?}");
}

#[test]
fn test_search_documents_uses_max_score() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    let d = doc(
        "d1",
        "Latency matters for retrieval and bananas ripen in summer.",
    );
    idx.add_document(&d);
    // Best per-proposition score for the document.
    let hits = idx
        .search("latency retrieval matters", 10)
        .expect("search ok");
    let expected = hits
        .iter()
        .filter(|h| h.proposition.parent_id.as_str() == "d1")
        .map(|h| h.score)
        .fold(f32::MIN, f32::max);
    let docs = idx
        .search_documents("latency retrieval matters", 5)
        .expect("ok");
    let got = docs
        .iter()
        .find(|(id, _)| id.as_str() == "d1")
        .map(|(_, s)| *s);
    assert_eq!(
        got,
        Some(expected),
        "document score must equal its best proposition score"
    );
}

#[test]
fn test_search_documents_sorted_descending() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "Caching reduces retrieval latency substantially.",
    ));
    idx.add_document(&doc("d2", "Gardening grows fresh tomatoes every summer."));
    let docs = idx
        .search_documents("retrieval latency caching", 5)
        .expect("ok");
    let scores: Vec<f32> = docs.iter().map(|(_, s)| *s).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "document scores must be sorted descending: {scores:?}"
    );
}

#[test]
fn test_search_documents_top_k_respected() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "d1",
        "Alpha content discusses retrieval methods here.",
    ));
    idx.add_document(&doc("d2", "Beta content discusses retrieval methods here."));
    idx.add_document(&doc(
        "d3",
        "Gamma content discusses retrieval methods here.",
    ));
    let docs = idx.search_documents("retrieval methods", 2).expect("ok");
    assert_eq!(docs.len(), 2, "search_documents must respect top_k");
}

#[test]
fn test_search_documents_picks_correct_parent() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc(
        "astronomy",
        "Stars form inside vast molecular clouds slowly.",
    ));
    idx.add_document(&doc(
        "cooking",
        "Bread rises when yeast ferments the dough.",
    ));
    let docs = idx
        .search_documents("stars molecular clouds form", 1)
        .expect("ok");
    assert_eq!(
        docs[0].0.as_str(),
        "astronomy",
        "the astronomy document should rank first: {docs:?}"
    );
}

#[test]
fn test_search_documents_empty_query_errors() {
    let mut idx = PropositionIndex::new(PropositionConfig::default());
    idx.add_document(&doc("d1", "Some content here is present."));
    let err = idx.search_documents("", 5).expect_err("blank query errors");
    assert!(
        matches!(err, PropositionError::EmptyQuery),
        "blank query should yield EmptyQuery: {err:?}"
    );
}

#[test]
fn test_search_documents_empty_corpus_errors() {
    let idx = PropositionIndex::new(PropositionConfig::default());
    let err = idx
        .search_documents("anything", 5)
        .expect_err("empty corpus errors");
    assert!(
        matches!(err, PropositionError::EmptyCorpus),
        "empty corpus should yield EmptyCorpus: {err:?}"
    );
}

// ── Determinism ────────────────────────────────────────────────────────────────

#[test]
fn test_search_documents_deterministic() {
    let build = || {
        let mut idx = PropositionIndex::new(PropositionConfig::default());
        idx.add_document(&doc(
            "d1",
            "Retrieval improves answers and grounds responses well.",
        ));
        idx.add_document(&doc(
            "d2",
            "Embeddings capture semantic similarity between texts.",
        ));
        idx.search_documents("retrieval grounds answers", 5)
            .expect("ok")
    };
    let first = build();
    let second = build();
    assert_eq!(
        first, second,
        "search_documents must be deterministic across runs"
    );
}

#[test]
fn test_search_hits_deterministic() {
    let build = || {
        let mut idx = PropositionIndex::new(PropositionConfig::default());
        idx.add_document(&doc(
            "d1",
            "Caching reduces latency and improves throughput greatly.",
        ));
        let hits = idx.search("latency throughput caching", 5).expect("ok");
        hits.into_iter()
            .map(|h| (h.proposition.id, h.score))
            .collect::<Vec<_>>()
    };
    assert_eq!(build(), build(), "search ordering must be deterministic");
}

#[test]
fn test_extract_deterministic() {
    let ext = HeuristicPropositionExtractor::new(PropositionConfig::default());
    let d = doc("d1", "Rust is fast and Rust is safe and Rust is modern.");
    assert_eq!(
        ext.extract(&d),
        ext.extract(&d),
        "extraction must be deterministic"
    );
}

// ── Trait object usage ─────────────────────────────────────────────────────────

#[test]
fn test_extractor_via_trait_object() {
    let ext = HeuristicPropositionExtractor::new(PropositionConfig::default());
    let dyn_ext: &dyn PropositionExtractor = &ext;
    let d = doc(
        "d1",
        "Rerankers refine results and improve final precision.",
    );
    let props = dyn_ext.extract(&d);
    assert!(
        props.len() >= 2,
        "trait-object extraction should yield multiple propositions: {props:?}"
    );
}
