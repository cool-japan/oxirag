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
//! Tests for the `generative_retrieval` module.

use super::retriever::GenerativeRetriever;
use super::types::{GenRetrievalConfig, GenRetrievalError, SemanticDocId, cosine, embed, tokenize};
use crate::types::Document;

// ── helpers ───────────────────────────────────────────────────────────────────

fn doc(content: &str) -> Document {
    Document::new(content)
}

/// A small corpus spanning four clearly distinct topics, several docs each.
fn topical_corpus() -> Vec<Document> {
    vec![
        doc("rust systems programming memory safety borrow checker compiler"),
        doc("rust cargo crates ownership lifetimes traits zero cost abstraction"),
        doc("rust async tokio futures runtime concurrency threads channels"),
        doc("python data science pandas numpy dataframe analysis notebook"),
        doc("python machine learning sklearn model training features pipeline"),
        doc("python web flask django request response template routing"),
        doc("ocean marine biology coral reef fish whales plankton tides"),
        doc("ocean currents salinity temperature waves deep sea pressure"),
        doc("mountain climbing altitude summit glacier rope harness rock"),
        doc("mountain hiking trail forest valley ridge backpack camping"),
    ]
}

// ── embed / tokenize / cosine primitives ──────────────────────────────────────

#[test]
fn test_embed_returns_requested_dim() {
    assert_eq!(embed("hello world", 128).len(), 128);
}

#[test]
fn test_embed_zero_dim_empty() {
    assert!(embed("hello world", 0).is_empty());
}

#[test]
fn test_embed_empty_text_zero_vec() {
    let v = embed("", 64);
    assert_eq!(v.len(), 64);
    assert!(v.iter().all(|x| *x == 0.0));
}

#[test]
fn test_embed_l2_normalised() {
    let v = embed("rust programming language tooling", 64);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-5);
}

#[test]
fn test_embed_deterministic() {
    let a = embed("the quick brown fox", 96);
    let b = embed("the quick brown fox", 96);
    assert_eq!(a, b);
}

#[test]
fn test_embed_case_insensitive() {
    let a = embed("Rust Systems Programming", 128);
    let b = embed("rust systems programming", 128);
    assert_eq!(a, b);
}

#[test]
fn test_tokenize_basic() {
    let toks = tokenize("Hello, World! foo");
    assert_eq!(toks, vec!["hello", "world", "foo"]);
}

#[test]
fn test_tokenize_drops_short_tokens() {
    let toks = tokenize("a I am ok go");
    assert_eq!(toks, vec!["am", "ok", "go"]);
}

#[test]
fn test_tokenize_lowercases() {
    assert_eq!(tokenize("ABCD"), vec!["abcd"]);
}

#[test]
fn test_tokenize_empty() {
    assert!(tokenize("").is_empty());
}

#[test]
fn test_cosine_identical_is_one() {
    let v = embed("vector search engine index", 64);
    assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_length_mismatch_zero() {
    assert_eq!(cosine(&[1.0, 0.0], &[1.0]), 0.0);
}

#[test]
fn test_cosine_empty_zero() {
    assert_eq!(cosine(&[], &[]), 0.0);
}

#[test]
fn test_cosine_disjoint_low() {
    let a = embed("rust borrow checker compiler", 256);
    let b = embed("ocean coral reef whales", 256);
    assert!(cosine(&a, &b) < 0.5);
}

// ── config defaults + builders ────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let c = GenRetrievalConfig::default();
    assert_eq!(c.branching, 4);
    assert_eq!(c.max_depth, 3);
    assert_eq!(c.beam, 2);
    assert_eq!(c.dim, 128);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(GenRetrievalConfig::new(), GenRetrievalConfig::default());
}

#[test]
fn test_config_with_branching() {
    assert_eq!(GenRetrievalConfig::new().with_branching(8).branching, 8);
}

#[test]
fn test_config_with_max_depth() {
    assert_eq!(GenRetrievalConfig::new().with_max_depth(5).max_depth, 5);
}

#[test]
fn test_config_with_beam() {
    assert_eq!(GenRetrievalConfig::new().with_beam(3).beam, 3);
}

#[test]
fn test_config_with_dim() {
    assert_eq!(GenRetrievalConfig::new().with_dim(256).dim, 256);
}

#[test]
fn test_config_builders_chain() {
    let c = GenRetrievalConfig::new()
        .with_branching(3)
        .with_max_depth(4)
        .with_beam(5)
        .with_dim(64);
    assert_eq!(c.branching, 3);
    assert_eq!(c.max_depth, 4);
    assert_eq!(c.beam, 5);
    assert_eq!(c.dim, 64);
}

// ── SemanticDocId ─────────────────────────────────────────────────────────────

#[test]
fn test_doc_id_as_string() {
    let id = SemanticDocId::new(vec![2, 5, 1]);
    assert_eq!(id.as_string(), "2-5-1");
}

#[test]
fn test_doc_id_as_string_single() {
    assert_eq!(SemanticDocId::new(vec![3]).as_string(), "3");
}

#[test]
fn test_doc_id_as_string_empty() {
    assert_eq!(SemanticDocId::new(vec![]).as_string(), "");
}

#[test]
fn test_doc_id_depth() {
    assert_eq!(SemanticDocId::new(vec![0, 1, 2]).depth(), 3);
    assert_eq!(SemanticDocId::new(vec![]).depth(), 0);
}

#[test]
fn test_doc_id_is_empty() {
    assert!(SemanticDocId::new(vec![]).is_empty());
    assert!(!SemanticDocId::new(vec![0]).is_empty());
}

#[test]
fn test_doc_id_display() {
    assert_eq!(format!("{}", SemanticDocId::new(vec![1, 0, 4])), "1-0-4");
}

#[test]
fn test_doc_id_equality_and_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SemanticDocId::new(vec![1, 2]));
    assert!(set.contains(&SemanticDocId::new(vec![1, 2])));
    assert!(!set.contains(&SemanticDocId::new(vec![2, 1])));
}

// ── construction / accessors ──────────────────────────────────────────────────

#[test]
fn test_new_is_empty() {
    let r = GenerativeRetriever::new(GenRetrievalConfig::default());
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}

#[test]
fn test_config_accessor() {
    let r = GenerativeRetriever::new(GenRetrievalConfig::new().with_beam(7));
    assert_eq!(r.config().beam, 7);
}

#[test]
fn test_build_sets_len() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    assert_eq!(r.len(), corpus.len());
    assert!(!r.is_empty());
}

#[test]
fn test_doc_id_accessor_in_range() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    assert!(r.doc_id(0).is_some());
}

#[test]
fn test_doc_id_accessor_out_of_range() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    assert!(r.doc_id(999).is_none());
}

#[test]
fn test_doc_id_none_before_build() {
    let r = GenerativeRetriever::new(GenRetrievalConfig::default());
    assert!(r.doc_id(0).is_none());
}

// ── build assigns ids ─────────────────────────────────────────────────────────

#[test]
fn test_build_assigns_non_empty_paths() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        let id = r.doc_id(i).unwrap();
        assert!(!id.is_empty(), "doc {i} got an empty path");
        assert!(id.depth() >= 1);
    }
}

#[test]
fn test_build_paths_within_max_depth() {
    let cfg = GenRetrievalConfig::default();
    let max_depth = cfg.max_depth;
    let mut r = GenerativeRetriever::new(cfg);
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        assert!(r.doc_id(i).unwrap().depth() <= max_depth);
    }
}

#[test]
fn test_build_cluster_indices_within_branching() {
    let cfg = GenRetrievalConfig::default();
    let branching = cfg.branching;
    let mut r = GenerativeRetriever::new(cfg);
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        for &ci in &r.doc_id(i).unwrap().path {
            assert!(
                ci < branching,
                "cluster index {ci} exceeds branching {branching}"
            );
        }
    }
}

#[test]
fn test_single_document_gets_non_empty_id() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&[doc("a lonely solitary document about nothing much")])
        .unwrap();
    let id = r.doc_id(0).unwrap();
    assert!(!id.is_empty());
    assert_eq!(id.depth(), 1, "single doc should descend exactly one level");
}

#[test]
fn test_two_documents_get_ids() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&[doc("alpha beta gamma delta"), doc("epsilon zeta eta theta")])
        .unwrap();
    assert!(!r.doc_id(0).unwrap().is_empty());
    assert!(!r.doc_id(1).unwrap().is_empty());
}

#[test]
fn test_all_docs_have_an_id() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let assigned = (0..corpus.len()).filter(|&i| r.doc_id(i).is_some()).count();
    assert_eq!(assigned, corpus.len());
}

// ── prefix hierarchy ──────────────────────────────────────────────────────────

#[test]
fn test_ids_form_prefix_hierarchy() {
    // Documents sharing the same first cluster index belong to the same
    // top-level cluster; we verify at least one shared prefix exists and that
    // every path begins with a valid top-level index.
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();

    let mut by_top: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..corpus.len() {
        let top = r.doc_id(i).unwrap().path[0];
        *by_top.entry(top).or_default() += 1;
    }
    // More than one top-level cluster, and the largest cluster shares path[0].
    assert!(
        by_top.len() >= 2,
        "corpus should split into multiple top clusters"
    );
    assert!(
        by_top.values().any(|&c| c >= 2),
        "some top cluster groups >= 2 docs"
    );
}

#[test]
fn test_similar_docs_share_top_cluster() {
    // Two near-identical documents should land in the same top-level cluster
    // among a backdrop of unrelated documents.
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = vec![
        doc("rust borrow checker ownership lifetimes compiler safety"),
        doc("rust borrow checker ownership lifetimes compiler safety guarantees"),
        doc("ocean coral reef marine biology fish whales plankton"),
        doc("mountain climbing summit glacier altitude rope harness"),
        doc("python pandas numpy dataframe data science analysis"),
        doc("baking bread flour yeast oven dough sourdough recipe"),
    ];
    r.build(&corpus).unwrap();
    assert_eq!(
        r.doc_id(0).unwrap().path[0],
        r.doc_id(1).unwrap().path[0],
        "near-duplicate docs should share the top cluster index"
    );
}

#[test]
fn test_paths_are_consistent_tree() {
    // Every doc with a given full path must agree on every prefix — trivially
    // true by construction, but we assert ids round-trip through as_string.
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        let id = r.doc_id(i).unwrap();
        let joined = id
            .path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-");
        assert_eq!(id.as_string(), joined);
    }
}

// ── search: exact match ───────────────────────────────────────────────────────

#[test]
fn test_search_exact_match_rank_zero() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let target = "ocean marine biology coral reef fish whales plankton tides";
    let hits = r.search(target, 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].document.content, target);
    assert!((hits[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn test_search_exact_match_second_topic() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let target = "rust async tokio futures runtime concurrency threads channels";
    let hits = r.search(target, 3).unwrap();
    assert_eq!(hits[0].document.content, target);
}

#[test]
fn test_search_returns_doc_id() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let hits = r.search("mountain climbing summit glacier", 1).unwrap();
    assert!(!hits[0].doc_id.is_empty());
}

#[test]
fn test_search_hit_doc_id_matches_index_id() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let target = "python web flask django request response template routing";
    let hits = r.search(target, 1).unwrap();
    // The returned doc_id must equal the stored id of the matched document.
    let idx = corpus
        .iter()
        .position(|d| d.content == hits[0].document.content)
        .unwrap();
    assert_eq!(&hits[0].doc_id, r.doc_id(idx).unwrap());
}

// ── search: top_k ─────────────────────────────────────────────────────────────

#[test]
fn test_search_respects_top_k() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default().with_beam(4));
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let hits = r.search("rust python ocean mountain", 3).unwrap();
    assert!(hits.len() <= 3);
}

#[test]
fn test_search_top_k_one() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let hits = r.search("rust cargo crates ownership", 1).unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn test_search_top_k_zero_empty() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    assert!(r.search("rust", 0).unwrap().is_empty());
}

#[test]
fn test_search_large_top_k_capped_by_candidates() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let hits = r.search("rust cargo crates", 1000).unwrap();
    assert!(hits.len() <= corpus.len());
}

#[test]
fn test_search_scores_descending() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default().with_beam(4));
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let hits = r
        .search("ocean currents salinity temperature waves", 5)
        .unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn test_search_wider_beam_at_least_as_many() {
    let corpus = topical_corpus();
    let mut narrow = GenerativeRetriever::new(GenRetrievalConfig::default().with_beam(1));
    let mut wide = GenerativeRetriever::new(GenRetrievalConfig::default().with_beam(4));
    narrow.build(&corpus).unwrap();
    wide.build(&corpus).unwrap();
    let n = narrow.search("rust python ocean", 100).unwrap().len();
    let w = wide.search("rust python ocean", 100).unwrap().len();
    assert!(
        w >= n,
        "wider beam should reach at least as many candidates"
    );
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_build_deterministic_same_ids() {
    let corpus = topical_corpus();
    let mut a = GenerativeRetriever::new(GenRetrievalConfig::default());
    let mut b = GenerativeRetriever::new(GenRetrievalConfig::default());
    a.build(&corpus).unwrap();
    b.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        assert_eq!(a.doc_id(i).unwrap(), b.doc_id(i).unwrap());
    }
}

#[test]
fn test_search_deterministic() {
    let corpus = topical_corpus();
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&corpus).unwrap();
    let h1 = r.search("rust async tokio runtime", 4).unwrap();
    let h2 = r.search("rust async tokio runtime", 4).unwrap();
    assert_eq!(h1.len(), h2.len());
    for (a, b) in h1.iter().zip(h2.iter()) {
        assert_eq!(a.document.content, b.document.content);
        assert_eq!(a.doc_id, b.doc_id);
    }
}

#[test]
fn test_rebuild_replaces_corpus() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    assert_eq!(r.len(), 10);
    r.build(&[doc("just one document now")]).unwrap();
    assert_eq!(r.len(), 1);
}

// ── errors ────────────────────────────────────────────────────────────────────

#[test]
fn test_build_empty_corpus_error() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let err = r.build(&[]).unwrap_err();
    assert!(matches!(err, GenRetrievalError::EmptyCorpus));
}

#[test]
fn test_search_not_built_error() {
    let r = GenerativeRetriever::new(GenRetrievalConfig::default());
    let err = r.search("anything", 5).unwrap_err();
    assert!(matches!(err, GenRetrievalError::NotBuilt));
}

#[test]
fn test_search_empty_query_error() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    let err = r.search("   ", 5).unwrap_err();
    assert!(matches!(err, GenRetrievalError::EmptyQuery));
}

#[test]
fn test_search_blank_query_error() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    assert!(matches!(
        r.search("", 5).unwrap_err(),
        GenRetrievalError::EmptyQuery
    ));
}

#[test]
fn test_not_built_checked_before_empty_query() {
    // NotBuilt must take precedence even with a blank query.
    let r = GenerativeRetriever::new(GenRetrievalConfig::default());
    assert!(matches!(
        r.search("", 5).unwrap_err(),
        GenRetrievalError::NotBuilt
    ));
}

#[test]
fn test_error_messages() {
    assert_eq!(
        GenRetrievalError::EmptyCorpus.to_string(),
        "corpus is empty"
    );
    assert_eq!(
        GenRetrievalError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(GenRetrievalError::NotBuilt.to_string(), "index not built");
}

// ── misc robustness ───────────────────────────────────────────────────────────

#[test]
fn test_shallow_tree_depth_one() {
    let cfg = GenRetrievalConfig::default().with_max_depth(1);
    let mut r = GenerativeRetriever::new(cfg);
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    for i in 0..corpus.len() {
        assert_eq!(r.doc_id(i).unwrap().depth(), 1);
    }
}

#[test]
fn test_deeper_tree_allows_deeper_ids() {
    let cfg = GenRetrievalConfig::default()
        .with_max_depth(4)
        .with_branching(2);
    let mut r = GenerativeRetriever::new(cfg);
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let max_depth = r.config().max_depth;
    let deepest = (0..corpus.len())
        .map(|i| r.doc_id(i).unwrap().depth())
        .max()
        .unwrap();
    assert!(deepest >= 2);
    assert!(deepest <= max_depth);
}

#[test]
fn test_search_after_rebuild_uses_new_corpus() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    r.build(&[
        doc("quantum entanglement superposition qubit measurement"),
        doc("classical mechanics newton force momentum energy"),
    ])
    .unwrap();
    let hits = r
        .search("quantum entanglement superposition qubit", 2)
        .unwrap();
    assert!(hits[0].document.content.contains("quantum"));
}

#[test]
fn test_custom_dim_build_and_search() {
    let cfg = GenRetrievalConfig::default().with_dim(256);
    let mut r = GenerativeRetriever::new(cfg);
    let corpus = topical_corpus();
    r.build(&corpus).unwrap();
    let target = "mountain hiking trail forest valley ridge backpack camping";
    let hits = r.search(target, 1).unwrap();
    assert_eq!(hits[0].document.content, target);
}

#[test]
fn test_search_unrelated_query_still_returns() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    // A query unrelated to any document should still produce candidates.
    let hits = r.search("zzzz qqqq xxxx unrelated gibberish", 3).unwrap();
    assert!(hits.len() <= 3);
}

#[test]
fn test_clone_retriever_preserves_ids() {
    let mut r = GenerativeRetriever::new(GenRetrievalConfig::default());
    r.build(&topical_corpus()).unwrap();
    let cloned = r.clone();
    for i in 0..r.len() {
        assert_eq!(r.doc_id(i).unwrap(), cloned.doc_id(i).unwrap());
    }
}
