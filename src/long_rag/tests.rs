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
//! Tests for the `long_rag` module.

use super::grouper::{LongUnitGrouper, embed, token_count, tokenize};
use super::retriever::LongRagRetriever;
use super::types::{GroupingStrategy, LongRagConfig, LongRagError};
use crate::types::{Document, DocumentId};

// ── helpers ───────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

/// Build a document with `n` distinct words to control token counts precisely.
fn doc_words(id: &str, prefix: &str, n: usize) -> Document {
    let words: Vec<String> = (0..n).map(|i| format!("{prefix}{i}")).collect();
    doc(id, &words.join(" "))
}

// ── Config defaults ───────────────────────────────────────────────────────────

#[test]
fn test_config_default_max_unit_tokens() {
    let cfg = LongRagConfig::default();
    assert_eq!(
        cfg.max_unit_tokens, 2000,
        "default max_unit_tokens should be 2000"
    );
}

#[test]
fn test_config_default_group_by() {
    let cfg = LongRagConfig::default();
    assert_eq!(
        cfg.group_by,
        GroupingStrategy::ByDocument,
        "default grouping should be ByDocument"
    );
}

#[test]
fn test_config_default_dim() {
    let cfg = LongRagConfig::default();
    assert_eq!(cfg.dim, 128, "default dim should be 128");
}

#[test]
fn test_config_default_similarity_threshold() {
    let cfg = LongRagConfig::default();
    assert_eq!(
        cfg.similarity_threshold, 0.3,
        "default similarity_threshold should be 0.3"
    );
}

#[test]
fn test_config_new_equals_default() {
    let cfg = LongRagConfig::new();
    assert_eq!(
        cfg.max_unit_tokens, 2000,
        "new() should match default budget"
    );
    assert_eq!(cfg.dim, 128, "new() should match default dim");
}

#[test]
fn test_grouping_strategy_default_trait() {
    assert_eq!(
        GroupingStrategy::default(),
        GroupingStrategy::ByDocument,
        "GroupingStrategy::default() should be ByDocument"
    );
}

// ── Config builders ───────────────────────────────────────────────────────────

#[test]
fn test_config_with_max_unit_tokens() {
    let cfg = LongRagConfig::new().with_max_unit_tokens(500);
    assert_eq!(
        cfg.max_unit_tokens, 500,
        "with_max_unit_tokens should set budget"
    );
}

#[test]
fn test_config_with_group_by() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::FixedTokenWindow);
    assert_eq!(
        cfg.group_by,
        GroupingStrategy::FixedTokenWindow,
        "with_group_by should set the strategy"
    );
}

#[test]
fn test_config_with_dim() {
    let cfg = LongRagConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64, "with_dim should set the dimension");
}

#[test]
fn test_config_with_similarity_threshold() {
    let cfg = LongRagConfig::new().with_similarity_threshold(0.75);
    assert_eq!(
        cfg.similarity_threshold, 0.75,
        "with_similarity_threshold should set the threshold"
    );
}

#[test]
fn test_config_builder_chaining() {
    let cfg = LongRagConfig::new()
        .with_max_unit_tokens(1000)
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_dim(32)
        .with_similarity_threshold(0.5);
    assert_eq!(
        cfg.max_unit_tokens, 1000,
        "chained builder should set budget"
    );
    assert_eq!(cfg.group_by, GroupingStrategy::BySemanticAdjacency);
    assert_eq!(cfg.dim, 32, "chained builder should set dim");
    assert_eq!(
        cfg.similarity_threshold, 0.5,
        "chained builder should set threshold"
    );
}

// ── Tokeniser & token_count ───────────────────────────────────────────────────

#[test]
fn test_tokenize_lowercases() {
    let toks = tokenize("Rust Is FAST");
    assert_eq!(
        toks,
        vec!["rust", "is", "fast"],
        "tokens should be lowercased"
    );
}

#[test]
fn test_tokenize_drops_short_tokens() {
    let toks = tokenize("a I am ok");
    assert_eq!(
        toks,
        vec!["am", "ok"],
        "tokens shorter than 2 chars should drop"
    );
}

#[test]
fn test_token_count_whitespace_words() {
    assert_eq!(token_count("one two three"), 3, "token_count counts words");
}

#[test]
fn test_token_count_empty() {
    assert_eq!(token_count("   "), 0, "blank text has zero tokens");
}

// ── Embedding ─────────────────────────────────────────────────────────────────

#[test]
fn test_embed_dimension() {
    let e = embed("rust memory safety", 128);
    assert_eq!(e.len(), 128, "embedding length should equal dim");
}

#[test]
fn test_embed_zero_dim() {
    let e = embed("rust", 0);
    assert!(e.is_empty(), "zero-dim embedding should be empty");
}

#[test]
fn test_embed_normalised() {
    let e = embed("rust memory safety borrow checker", 128);
    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "non-empty embedding should be L2-normalised"
    );
}

#[test]
fn test_embed_deterministic() {
    let a = embed("rust is memory safe", 64);
    let b = embed("rust is memory safe", 64);
    assert_eq!(a, b, "embedding must be deterministic");
}

// ── ByDocument grouping ───────────────────────────────────────────────────────

#[test]
fn test_by_document_one_unit_per_doc() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::ByDocument);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc("a", "first document content here"),
        doc("b", "second document content here"),
        doc("c", "third document content here"),
    ];
    let units = grouper.group(&docs);
    assert_eq!(
        units.len(),
        3,
        "ByDocument should emit one unit per document"
    );
}

#[test]
fn test_by_document_source_ids_single() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::ByDocument);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc("a", "alpha beta gamma"), doc("b", "delta epsilon zeta")];
    let units = grouper.group(&docs);
    assert_eq!(units[0].source_ids, vec![DocumentId::from_string("a")]);
    assert_eq!(units[1].source_ids, vec![DocumentId::from_string("b")]);
}

#[test]
fn test_by_document_truncates_to_budget() {
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::ByDocument)
        .with_max_unit_tokens(5);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc_words("a", "w", 50)];
    let units = grouper.group(&docs);
    assert_eq!(units.len(), 1, "ByDocument still emits a single unit");
    assert_eq!(
        units[0].token_count, 5,
        "unit should be capped at the budget"
    );
}

#[test]
fn test_by_document_token_count_matches_text() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::ByDocument);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc("a", "alpha beta gamma delta")];
    let units = grouper.group(&docs);
    assert_eq!(
        units[0].token_count,
        token_count(&units[0].text),
        "token_count should match the unit text"
    );
    assert_eq!(units[0].token_count, 4, "four words expected");
}

#[test]
fn test_by_document_embedding_present() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::ByDocument);
    let dim = cfg.dim;
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc("a", "rust memory safety")];
    let units = grouper.group(&docs);
    assert_eq!(units[0].embedding.len(), dim, "embedding should be present");
}

#[test]
fn test_by_document_empty_corpus() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::ByDocument);
    let grouper = LongUnitGrouper::new(cfg);
    let units = grouper.group(&[]);
    assert!(units.is_empty(), "no docs yields no units");
}

// ── FixedTokenWindow grouping ─────────────────────────────────────────────────

#[test]
fn test_fixed_window_respects_budget() {
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(10);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc_words("a", "w", 25), doc_words("b", "v", 25)];
    let units = grouper.group(&docs);
    assert!(!units.is_empty(), "fixed-window should produce units");
    for u in &units {
        assert!(
            u.token_count <= 10,
            "every fixed-window unit must be within the budget, got {}",
            u.token_count
        );
    }
}

#[test]
fn test_fixed_window_count() {
    // 30 total words, budget 10 → exactly 3 windows.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(10);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc_words("a", "w", 30)];
    let units = grouper.group(&docs);
    assert_eq!(units.len(), 3, "30 words / 10 budget → 3 windows");
}

#[test]
fn test_fixed_window_partial_last() {
    // 25 words, budget 10 → 10, 10, 5.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(10);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc_words("a", "w", 25)];
    let units = grouper.group(&docs);
    assert_eq!(units.len(), 3, "25 words / 10 → 3 windows");
    assert_eq!(units[2].token_count, 5, "last window holds the remainder");
}

#[test]
fn test_fixed_window_crosses_document_boundaries() {
    // Two 5-word docs, budget 10 → one window spanning both documents.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(10);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![doc_words("a", "w", 5), doc_words("b", "v", 5)];
    let units = grouper.group(&docs);
    assert_eq!(units.len(), 1, "10 words fit in a single window");
    assert_eq!(
        units[0].source_ids,
        vec![DocumentId::from_string("a"), DocumentId::from_string("b")],
        "window source_ids should span both documents"
    );
}

#[test]
fn test_fixed_window_empty_corpus() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::FixedTokenWindow);
    let grouper = LongUnitGrouper::new(cfg);
    assert!(grouper.group(&[]).is_empty(), "no docs yields no windows");
}

// ── Grouping reduces unit count vs raw chunks ─────────────────────────────────

#[test]
fn test_by_document_reduces_unit_count() {
    // A corpus of many short single-sentence "chunks" representing one topic.
    // ByDocument keeps one unit per doc, but if we imagine these as fragments of
    // 2 logical sources, grouping collapses fragmentation.
    let raw_chunks: Vec<Document> = (0..12)
        .map(|i| {
            doc(
                &format!("c{i}"),
                &format!("fragment number {i} about rust safety"),
            )
        })
        .collect();
    let raw_chunk_count = raw_chunks.len();

    // Group all fragments into ONE long unit via a generous fixed window.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(10_000);
    let grouper = LongUnitGrouper::new(cfg);
    let units = grouper.group(&raw_chunks);
    assert!(
        units.len() < raw_chunk_count,
        "grouping must reduce unit count ({} units) below raw chunk count ({})",
        units.len(),
        raw_chunk_count
    );
    assert_eq!(
        units.len(),
        1,
        "a generous window collapses all chunks into one unit"
    );
}

#[test]
fn test_fixed_window_unit_count_below_chunk_count() {
    // 20 chunks of ~6 words each = ~120 words; budget 40 → 3 units < 20 chunks.
    let raw_chunks: Vec<Document> = (0..20)
        .map(|i| doc(&format!("c{i}"), "alpha beta gamma delta epsilon zeta"))
        .collect();
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(40);
    let grouper = LongUnitGrouper::new(cfg);
    let units = grouper.group(&raw_chunks);
    assert!(
        units.len() < raw_chunks.len(),
        "grouped units ({}) must be fewer than chunks ({})",
        units.len(),
        raw_chunks.len()
    );
}

// ── BySemanticAdjacency grouping ──────────────────────────────────────────────

#[test]
fn test_semantic_adjacency_merges_similar() {
    // Three near-identical documents should merge into a single unit.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.2);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc("a", "rust memory safety borrow checker ownership"),
        doc("b", "rust memory safety borrow checker lifetimes"),
        doc("c", "rust memory safety borrow checker references"),
    ];
    let units = grouper.group(&docs);
    assert_eq!(
        units.len(),
        1,
        "highly similar consecutive docs should merge into one unit"
    );
    assert_eq!(
        units[0].source_ids.len(),
        3,
        "merged unit should track all 3 sources"
    );
}

#[test]
fn test_semantic_adjacency_splits_dissimilar() {
    // Two clearly different topics should NOT merge with a high threshold.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.9);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc("a", "rust memory safety borrow checker ownership lifetimes"),
        doc("b", "cooking pasta tomato basil olive garlic kitchen"),
    ];
    let units = grouper.group(&docs);
    assert_eq!(
        units.len(),
        2,
        "dissimilar docs should stay in separate units"
    );
}

#[test]
fn test_semantic_adjacency_respects_budget() {
    // Similar docs, but the budget forces a split before all merge.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.0)
        .with_max_unit_tokens(8);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc_words("a", "w", 5),
        doc_words("b", "w", 5),
        doc_words("c", "w", 5),
    ];
    let units = grouper.group(&docs);
    assert!(units.len() >= 2, "budget should force at least one split");
    for u in &units {
        assert!(u.token_count <= 8, "no unit may exceed the budget");
    }
}

#[test]
fn test_semantic_adjacency_source_ids_membership() {
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.2);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc("a", "shared topic words alpha beta gamma"),
        doc("b", "shared topic words alpha beta delta"),
    ];
    let units = grouper.group(&docs);
    let all_sources: Vec<&DocumentId> = units.iter().flat_map(|u| u.source_ids.iter()).collect();
    assert!(
        all_sources.contains(&&DocumentId::from_string("a")),
        "source a tracked"
    );
    assert!(
        all_sources.contains(&&DocumentId::from_string("b")),
        "source b tracked"
    );
}

#[test]
fn test_semantic_adjacency_empty_corpus() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::BySemanticAdjacency);
    let grouper = LongUnitGrouper::new(cfg);
    assert!(grouper.group(&[]).is_empty(), "no docs yields no units");
}

#[test]
fn test_semantic_adjacency_total_sources_preserved() {
    // Every input document must appear in exactly one unit's source list.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.5);
    let grouper = LongUnitGrouper::new(cfg);
    let docs = vec![
        doc("a", "rust safety ownership"),
        doc("b", "rust safety ownership"),
        doc("c", "totally different cooking recipe"),
        doc("d", "totally different cooking recipe"),
    ];
    let units = grouper.group(&docs);
    let total: usize = units.iter().map(|u| u.source_ids.len()).sum();
    assert_eq!(
        total, 4,
        "all four source documents must be accounted for exactly once"
    );
}

// ── Retriever build / counts ──────────────────────────────────────────────────

#[test]
fn test_retriever_new_is_empty() {
    let r = LongRagRetriever::new(LongRagConfig::default());
    assert!(r.is_empty(), "fresh retriever should be empty");
    assert_eq!(r.unit_count(), 0, "fresh retriever holds zero units");
}

#[test]
fn test_retriever_build_populates() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    let docs = vec![
        doc("a", "rust memory safety"),
        doc("b", "borrow checker rules"),
    ];
    r.build(&docs).unwrap();
    assert!(
        !r.is_empty(),
        "after build the retriever should not be empty"
    );
    assert_eq!(
        r.unit_count(),
        2,
        "ByDocument build yields one unit per doc"
    );
}

#[test]
fn test_retriever_build_empty_corpus_errors() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    let err = r.build(&[]).unwrap_err();
    assert!(
        matches!(err, LongRagError::EmptyCorpus),
        "empty build should error"
    );
}

#[test]
fn test_retriever_build_replaces_previous() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    r.build(&[doc("a", "first corpus")]).unwrap();
    assert_eq!(r.unit_count(), 1, "first build → 1 unit");
    r.build(&[doc("x", "one"), doc("y", "two"), doc("z", "three")])
        .unwrap();
    assert_eq!(r.unit_count(), 3, "second build should replace the first");
}

// ── Search ────────────────────────────────────────────────────────────────────

#[test]
fn test_search_returns_long_units() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    r.build(&[
        doc(
            "a",
            "rust guarantees memory safety without garbage collection",
        ),
        doc("b", "python is a dynamically typed scripting language"),
    ])
    .unwrap();
    let hits = r.search("memory safety", 5).unwrap();
    assert!(!hits.is_empty(), "search should return hits");
    assert_eq!(
        hits[0].unit.source_ids,
        vec![DocumentId::from_string("a")],
        "the rust unit should rank first for a memory-safety query"
    );
}

#[test]
fn test_search_top_k_limits_results() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    let docs: Vec<Document> = (0..6)
        .map(|i| doc(&format!("d{i}"), &format!("topic {i} rust memory safety")))
        .collect();
    r.build(&docs).unwrap();
    let hits = r.search("rust safety", 3).unwrap();
    assert_eq!(hits.len(), 3, "top_k should cap the number of hits");
}

#[test]
fn test_search_scores_descending() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    let docs: Vec<Document> = (0..5)
        .map(|i| {
            doc(
                &format!("d{i}"),
                &format!("doc {i} alpha beta gamma rust safety memory"),
            )
        })
        .collect();
    r.build(&docs).unwrap();
    let hits = r.search("rust safety memory", 5).unwrap();
    for w in hits.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "hits must be sorted by descending score"
        );
    }
}

#[test]
fn test_search_empty_query_errors() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    r.build(&[doc("a", "rust memory safety")]).unwrap();
    let err = r.search("   ", 5).unwrap_err();
    assert!(
        matches!(err, LongRagError::EmptyQuery),
        "blank query should error"
    );
}

#[test]
fn test_search_empty_corpus_errors() {
    let r = LongRagRetriever::new(LongRagConfig::default());
    let err = r.search("rust", 5).unwrap_err();
    assert!(
        matches!(err, LongRagError::EmptyCorpus),
        "search on empty corpus should error"
    );
}

#[test]
fn test_search_top_k_zero_returns_empty() {
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    r.build(&[doc("a", "rust memory safety")]).unwrap();
    let hits = r.search("rust", 0).unwrap();
    assert!(hits.is_empty(), "top_k of 0 returns no hits");
}

#[test]
fn test_search_over_fixed_window_units() {
    // Retrieval should work over fixed-window units too, not just per-document.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::FixedTokenWindow)
        .with_max_unit_tokens(6);
    let mut r = LongRagRetriever::new(cfg);
    let docs = vec![
        doc("a", "rust memory safety borrow checker ownership lifetimes"),
        doc("b", "cooking pasta tomato basil olive oil garlic"),
    ];
    r.build(&docs).unwrap();
    let hits = r.search("memory safety borrow", 3).unwrap();
    assert!(!hits.is_empty(), "search over windows should return hits");
    assert!(
        hits[0].unit.text.contains("memory") || hits[0].unit.text.contains("borrow"),
        "top window should be the rust one"
    );
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_grouping_deterministic() {
    let cfg = LongRagConfig::new().with_group_by(GroupingStrategy::BySemanticAdjacency);
    let docs = vec![
        doc("a", "rust memory safety ownership"),
        doc("b", "rust memory safety borrow"),
        doc("c", "unrelated cooking pasta recipe"),
    ];
    let g1 = LongUnitGrouper::new(cfg.clone()).group(&docs);
    let g2 = LongUnitGrouper::new(cfg).group(&docs);
    assert_eq!(
        g1.len(),
        g2.len(),
        "grouping should be deterministic in unit count"
    );
    for (u1, u2) in g1.iter().zip(g2.iter()) {
        assert_eq!(u1.text, u2.text, "unit text should be deterministic");
        assert_eq!(
            u1.source_ids, u2.source_ids,
            "source ids should be deterministic"
        );
        assert_eq!(
            u1.embedding, u2.embedding,
            "embeddings should be deterministic"
        );
    }
}

#[test]
fn test_search_deterministic() {
    let docs = vec![
        doc("a", "rust memory safety"),
        doc("b", "python scripting language"),
        doc("c", "rust borrow checker safety"),
    ];
    let mut r1 = LongRagRetriever::new(LongRagConfig::default());
    let mut r2 = LongRagRetriever::new(LongRagConfig::default());
    r1.build(&docs).unwrap();
    r2.build(&docs).unwrap();
    let h1 = r1.search("rust safety", 3).unwrap();
    let h2 = r2.search("rust safety", 3).unwrap();
    assert_eq!(h1.len(), h2.len(), "result counts should match");
    for (a, b) in h1.iter().zip(h2.iter()) {
        assert_eq!(a.unit.id, b.unit.id, "hit order should be deterministic");
        assert_eq!(a.score, b.score, "scores should be deterministic");
    }
}

#[test]
fn test_units_fewer_than_documents_when_merged() {
    // The defining LongRAG property: with merging enabled, the number of long
    // units is strictly fewer than the number of source documents.
    let cfg = LongRagConfig::new()
        .with_group_by(GroupingStrategy::BySemanticAdjacency)
        .with_similarity_threshold(0.1)
        .with_max_unit_tokens(10_000);
    let grouper = LongUnitGrouper::new(cfg);
    let docs: Vec<Document> = (0..8)
        .map(|i| {
            doc(
                &format!("d{i}"),
                "rust memory safety ownership borrow checker",
            )
        })
        .collect();
    let units = grouper.group(&docs);
    assert!(
        units.len() < docs.len(),
        "merged long units ({}) should be fewer than docs ({})",
        units.len(),
        docs.len()
    );
}

#[test]
fn test_search_tie_break_by_id() {
    // Two identical-content docs produce equal scores; ties break by unit id so
    // the lower id comes first.
    let mut r = LongRagRetriever::new(LongRagConfig::default());
    r.build(&[
        doc("a", "identical content here"),
        doc("b", "identical content here"),
    ])
    .unwrap();
    let hits = r.search("identical content", 2).unwrap();
    assert_eq!(hits.len(), 2, "both units should be returned");
    assert_eq!(
        hits[0].score, hits[1].score,
        "equal content yields equal scores"
    );
    assert!(
        hits[0].unit.id < hits[1].unit.id,
        "ties should break by ascending id"
    );
}
