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
//! Tests for the `auto_merging` module.

use crate::types::Document;

use super::hierarchy::{ChunkHierarchy, ChunkNode};
use super::retriever::AutoMergingRetriever;
use super::types::{AutoMergeConfig, AutoMergeError, MergedHit};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Build a retriever over `content` with explicit window/threshold settings.
fn build_retriever(
    content: &str,
    parent_words: usize,
    child_words: usize,
    merge_threshold: f32,
) -> AutoMergingRetriever {
    let cfg = AutoMergeConfig::new()
        .with_parent_words(parent_words)
        .with_child_words(child_words)
        .with_merge_threshold(merge_threshold);
    let mut r = AutoMergingRetriever::new(cfg);
    r.build(&Document::new(content))
        .expect("build should succeed");
    r
}

/// Eight distinct, collision-free (at `dim = 128`) leaf words across two parents.
const EIGHT_WORDS: &str = "alpha beta gamma delta epsilon zeta theta iota";

// ── AutoMergeConfig tests ─────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = AutoMergeConfig::default();
    assert_eq!(cfg.merge_threshold, 0.5);
    assert_eq!(cfg.parent_words, 80);
    assert_eq!(cfg.child_words, 20);
    assert_eq!(cfg.dim, 128);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(AutoMergeConfig::new(), AutoMergeConfig::default());
}

#[test]
fn test_config_with_merge_threshold() {
    let cfg = AutoMergeConfig::new().with_merge_threshold(0.75);
    assert_eq!(cfg.merge_threshold, 0.75);
}

#[test]
fn test_config_with_parent_words() {
    let cfg = AutoMergeConfig::new().with_parent_words(120);
    assert_eq!(cfg.parent_words, 120);
}

#[test]
fn test_config_with_child_words() {
    let cfg = AutoMergeConfig::new().with_child_words(30);
    assert_eq!(cfg.child_words, 30);
}

#[test]
fn test_config_with_dim() {
    let cfg = AutoMergeConfig::new().with_dim(256);
    assert_eq!(cfg.dim, 256);
}

#[test]
fn test_config_builder_chain() {
    let cfg = AutoMergeConfig::new()
        .with_merge_threshold(0.6)
        .with_parent_words(40)
        .with_child_words(10)
        .with_dim(64);
    assert_eq!(cfg.merge_threshold, 0.6);
    assert_eq!(cfg.parent_words, 40);
    assert_eq!(cfg.child_words, 10);
    assert_eq!(cfg.dim, 64);
}

#[test]
fn test_config_is_copy() {
    let cfg = AutoMergeConfig::default();
    let copy = cfg;
    assert_eq!(cfg, copy);
}

// ── ChunkHierarchy basic tests ────────────────────────────────────────────────

#[test]
fn test_hierarchy_new_is_empty() {
    let h = ChunkHierarchy::new();
    assert!(h.is_empty());
    assert_eq!(h.len(), 0);
    assert!(h.leaves().is_empty());
}

#[test]
fn test_hierarchy_default_is_empty() {
    let h = ChunkHierarchy::default();
    assert_eq!(h.len(), 0);
}

#[test]
fn test_hierarchy_node_missing_returns_none() {
    let h = ChunkHierarchy::new();
    assert!(h.node(0).is_none());
    assert!(h.node(99).is_none());
}

#[test]
fn test_hierarchy_children_of_missing_is_empty() {
    let h = ChunkHierarchy::new();
    assert!(h.children_of(0).is_empty());
}

#[test]
fn test_hierarchy_parent_of_missing_is_none() {
    let h = ChunkHierarchy::new();
    assert!(h.parent_of(0).is_none());
}

#[test]
fn test_chunk_node_fields() {
    let node = ChunkNode {
        id: 3,
        text: "hello".to_string(),
        parent: Some(1),
        children: vec![4, 5],
        level: 1,
    };
    assert_eq!(node.id, 3);
    assert_eq!(node.text, "hello");
    assert_eq!(node.parent, Some(1));
    assert_eq!(node.children, vec![4, 5]);
    assert_eq!(node.level, 1);
}

// ── build: hierarchy structure tests ──────────────────────────────────────────

#[test]
fn test_build_creates_parent_and_children() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    // 2 parents + 8 leaves = 10 nodes.
    assert_eq!(h.len(), 10);
}

#[test]
fn test_build_leaves_count() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    assert_eq!(r.hierarchy().leaves().len(), 8);
}

#[test]
fn test_build_every_leaf_has_a_parent() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    for leaf_id in h.leaves() {
        assert!(
            h.parent_of(leaf_id).is_some(),
            "leaf {leaf_id} must reference a parent"
        );
    }
}

#[test]
fn test_build_leaves_are_level_zero() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    for leaf_id in h.leaves() {
        assert_eq!(h.node(leaf_id).unwrap().level, 0);
    }
}

#[test]
fn test_build_parents_are_level_one() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    // Parents are the non-leaf nodes.
    for node in &h.nodes {
        if !node.children.is_empty() {
            assert_eq!(node.level, 1);
            assert!(node.parent.is_none());
        }
    }
}

#[test]
fn test_build_children_of_parent() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    // Parent 0 is the first node; it should have 4 children.
    assert_eq!(h.children_of(0).len(), 4);
}

#[test]
fn test_build_parent_of_round_trips_children() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    for &child in h.children_of(0) {
        assert_eq!(h.parent_of(child), Some(0));
    }
}

#[test]
fn test_build_leaf_text_is_single_word() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    let first_child = h.children_of(0)[0];
    assert_eq!(h.node(first_child).unwrap().text, "alpha");
}

#[test]
fn test_build_parent_text_is_full_window() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    assert_eq!(h.node(0).unwrap().text, "alpha beta gamma delta");
}

#[test]
fn test_build_two_parents() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    let parents: Vec<&ChunkNode> = h.nodes.iter().filter(|n| !n.children.is_empty()).collect();
    assert_eq!(parents.len(), 2);
}

#[test]
fn test_build_second_parent_text() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    // Parent 1 sits at index 5 (parent0=0, leaves 1..=4, parent1=5).
    assert_eq!(h.node(5).unwrap().text, "epsilon zeta theta iota");
}

#[test]
fn test_build_default_window_grouping() {
    // 80-word parent windows, 20-word children: 8 words → 1 parent, 1 child.
    let r = build_retriever(EIGHT_WORDS, 80, 20, 0.5);
    let h = r.hierarchy();
    let parents: Vec<&ChunkNode> = h.nodes.iter().filter(|n| !n.children.is_empty()).collect();
    assert_eq!(parents.len(), 1);
    assert_eq!(parents[0].children.len(), 1);
}

#[test]
fn test_build_partial_trailing_window() {
    // 5 words, parent_words=4 → parents of 4 and 1 words.
    let r = build_retriever("alpha beta gamma delta epsilon", 4, 2, 0.5);
    let h = r.hierarchy();
    let parents: Vec<&ChunkNode> = h.nodes.iter().filter(|n| !n.children.is_empty()).collect();
    assert_eq!(parents.len(), 2);
    assert_eq!(parents[1].text, "epsilon");
}

#[test]
fn test_build_collapses_whitespace() {
    let r = build_retriever("alpha    beta\n\tgamma   delta", 4, 1, 0.5);
    let h = r.hierarchy();
    assert_eq!(h.node(0).unwrap().text, "alpha beta gamma delta");
    assert_eq!(h.leaves().len(), 4);
}

#[test]
fn test_build_rebuild_replaces_state() {
    let cfg = AutoMergeConfig::new()
        .with_parent_words(4)
        .with_child_words(1);
    let mut r = AutoMergingRetriever::new(cfg);
    r.build(&Document::new("alpha beta gamma delta")).unwrap();
    assert_eq!(r.hierarchy().leaves().len(), 4);
    r.build(&Document::new("alpha beta")).unwrap();
    assert_eq!(r.hierarchy().leaves().len(), 2);
}

#[test]
fn test_search_full_parent_coverage_merges_all_children() {
    // parent_words=4, child_words=1 → parent 0 has 4 leaf children.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    // Query all four words of parent 0 with a large top_k → full merge → merged_from == 4.
    let hits = r.search("alpha beta gamma delta", 4).unwrap();
    let merged = hits
        .iter()
        .find(|h| h.node_id == 0)
        .expect("parent 0 merged");
    assert_eq!(merged.merged_from, 4);
}

// ── build: error tests ────────────────────────────────────────────────────────

#[test]
fn test_build_empty_document_errors() {
    let mut r = AutoMergingRetriever::new(AutoMergeConfig::default());
    let err = r.build(&Document::new("")).expect_err("empty doc");
    assert!(matches!(err, AutoMergeError::EmptyDocument));
}

#[test]
fn test_build_whitespace_only_document_errors() {
    let mut r = AutoMergingRetriever::new(AutoMergeConfig::default());
    let err = r.build(&Document::new("   \n\t  ")).expect_err("blank doc");
    assert!(matches!(err, AutoMergeError::EmptyDocument));
}

// ── search: not-built / empty-query errors ────────────────────────────────────

#[test]
fn test_search_not_built_errors() {
    let r = AutoMergingRetriever::new(AutoMergeConfig::default());
    let err = r.search("query", 5).expect_err("not built");
    assert!(matches!(err, AutoMergeError::NotBuilt));
}

#[test]
fn test_search_empty_query_errors() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let err = r.search("", 5).expect_err("empty query");
    assert!(matches!(err, AutoMergeError::EmptyQuery));
}

#[test]
fn test_search_whitespace_query_errors() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let err = r.search("   ", 5).expect_err("blank query");
    assert!(matches!(err, AutoMergeError::EmptyQuery));
}

#[test]
fn test_search_top_k_zero_returns_empty() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha", 0).unwrap();
    assert!(hits.is_empty());
}

// ── search: leaf retrieval (no merge) ─────────────────────────────────────────

#[test]
fn test_search_single_leaf_no_merge() {
    // top_k = 1 → only one of parent0's four children retrieved → 1/4 < 0.5.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha", 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].level, 0, "should remain a leaf");
    assert_eq!(hits[0].text, "alpha");
    assert_eq!(hits[0].merged_from, 1);
}

#[test]
fn test_search_leaf_score_is_one_for_exact_word() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha", 1).unwrap();
    assert!((hits[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn test_search_below_threshold_emits_leaves() {
    // threshold 0.75, retrieve 2 of 4 children → 0.5 < 0.75 → no merge.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.75);
    let hits = r.search("alpha beta", 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|h| h.level == 0));
    assert!(hits.iter().all(|h| h.merged_from == 1));
}

#[test]
fn test_search_below_threshold_leaf_texts() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.75);
    let hits = r.search("alpha beta", 2).unwrap();
    let mut texts: Vec<&str> = hits.iter().map(|h| h.text.as_str()).collect();
    texts.sort_unstable();
    assert_eq!(texts, vec!["alpha", "beta"]);
}

// ── search: merge triggers ────────────────────────────────────────────────────

#[test]
fn test_search_merge_triggers_at_threshold() {
    // threshold 0.5, retrieve 2 of 4 children of parent0 → 0.5 ≥ 0.5 → merge.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha beta", 2).unwrap();
    assert_eq!(hits.len(), 1, "two leaves collapse into one parent");
    assert_eq!(hits[0].node_id, 0, "parent 0 is emitted");
    assert_eq!(hits[0].level, 1, "emitted node is a parent");
}

#[test]
fn test_search_merge_merged_from_count() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha beta", 2).unwrap();
    assert_eq!(hits[0].merged_from, 2, "two children collapsed");
}

#[test]
fn test_search_merged_text_equals_parent_text() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha beta", 2).unwrap();
    let parent_text = r.hierarchy().node(0).unwrap().text.clone();
    assert_eq!(hits[0].text, parent_text);
    assert_eq!(hits[0].text, "alpha beta gamma delta");
}

#[test]
fn test_search_merge_full_parent() {
    // parent_words=2 → each parent has exactly 2 children; retrieve both → 2/2 merge.
    let r = build_retriever("alpha beta gamma delta", 2, 1, 0.5);
    let hits = r.search("alpha beta", 2).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].merged_from, 2);
    assert_eq!(hits[0].text, "alpha beta");
}

#[test]
fn test_search_merge_threshold_one_requires_all_children() {
    // threshold 1.0: parent0 has 4 children; retrieving 2 must NOT merge.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    let hits = r.search("alpha beta", 2).unwrap();
    assert!(
        hits.iter().all(|h| h.level == 0),
        "no merge below full coverage"
    );
    assert_eq!(hits.len(), 2);
}

#[test]
fn test_search_merge_threshold_one_merges_when_all_present() {
    // threshold 1.0 with a 2-child parent: retrieve both → merge.
    let r = build_retriever("alpha beta gamma delta", 2, 1, 1.0);
    let hits = r.search("alpha beta", 2).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].merged_from, 2);
}

#[test]
fn test_search_merge_only_affected_parent() {
    // Retrieve 2 children of parent0 and 1 child of parent1 (3 leaves).
    // parent0: 2/4 = 0.5 → merge; parent1: 1/4 = 0.25 → no merge.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha beta epsilon", 3).unwrap();
    // One merged parent (node 0) + one surviving leaf (epsilon).
    assert_eq!(hits.len(), 2);
    let parent_hit = hits.iter().find(|h| h.level == 1).expect("merged parent");
    assert_eq!(parent_hit.node_id, 0);
    assert_eq!(parent_hit.merged_from, 2);
    let leaf_hit = hits.iter().find(|h| h.level == 0).expect("surviving leaf");
    assert_eq!(leaf_hit.text, "epsilon");
    assert_eq!(leaf_hit.merged_from, 1);
}

#[test]
fn test_search_merge_two_parents_independently() {
    // parent_words=2 → 2 children per parent. Retrieve all 4 of the first two
    // parents' leaves → each parent merges independently.
    let r = build_retriever("alpha beta gamma delta", 2, 1, 0.5);
    let hits = r.search("alpha beta gamma delta", 4).unwrap();
    assert_eq!(hits.len(), 2, "two independent parent merges");
    assert!(hits.iter().all(|h| h.level == 1));
    assert!(hits.iter().all(|h| h.merged_from == 2));
}

#[test]
fn test_search_merged_score_is_best_child_score() {
    // Query "alpha beta": leaf alpha and beta each score 1/sqrt(2); merged parent
    // takes the best (equal here) child score.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha beta", 2).unwrap();
    let expected = 1.0f32 / 2.0f32.sqrt();
    assert!((hits[0].score - expected).abs() < 1e-5);
}

// ── search: top_k ─────────────────────────────────────────────────────────────

#[test]
fn test_search_respects_top_k_on_leaves() {
    // High threshold so nothing merges; retrieve top 3 leaves only.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    let hits = r.search("alpha beta gamma epsilon zeta theta", 3).unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn test_search_top_k_larger_than_corpus() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    let hits = r.search("alpha", 100).unwrap();
    // At most 8 leaves exist; nothing merges at threshold 1.0.
    assert!(hits.len() <= 8);
    assert!(!hits.is_empty());
}

#[test]
fn test_search_hits_sorted_by_score_desc() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    let hits = r.search("alpha beta gamma", 8).unwrap();
    for pair in hits.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "hits must be sorted by descending score"
        );
    }
}

// ── search: ranking / content ─────────────────────────────────────────────────

#[test]
fn test_search_irrelevant_query_zero_scores() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    // "kappa" does not appear among the leaves → all scores 0.
    let hits = r.search("kappa", 3).unwrap();
    assert!(hits.iter().all(|h| h.score.abs() < 1e-6));
}

#[test]
fn test_search_relevant_leaf_ranks_first() {
    // top_k = 1 retrieves only the best leaf, so no sibling coverage → no merge.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 1.0);
    let hits = r.search("theta", 1).unwrap();
    assert_eq!(hits[0].text, "theta");
    assert!((hits[0].score - 1.0).abs() < 1e-5);
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_search_deterministic_repeated() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let a = r.search("alpha beta epsilon", 5).unwrap();
    let b = r.search("alpha beta epsilon", 5).unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_build_deterministic_across_instances() {
    let r1 = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let r2 = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    assert_eq!(r1.hierarchy().nodes, r2.hierarchy().nodes);
}

#[test]
fn test_search_deterministic_across_instances() {
    let r1 = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let r2 = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    assert_eq!(
        r1.search("alpha beta", 3).unwrap(),
        r2.search("alpha beta", 3).unwrap()
    );
}

#[test]
fn test_merged_hit_equality() {
    let a = MergedHit {
        text: "x".to_string(),
        node_id: 1,
        level: 0,
        score: 0.5,
        merged_from: 1,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

// ── multi-window realism ──────────────────────────────────────────────────────

#[test]
fn test_search_partial_coverage_no_merge_then_leaf_text() {
    // parent0 has 4 children; retrieve exactly 1 → emit that leaf, not the parent.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("gamma", 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, "gamma");
    assert_eq!(hits[0].level, 0);
}

#[test]
fn test_search_does_not_merge_across_parents() {
    // Retrieve one child from each parent → neither parent meets the threshold.
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let hits = r.search("alpha epsilon", 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(hits.iter().all(|h| h.level == 0));
}

#[test]
fn test_hierarchy_borrow_after_build_is_consistent() {
    let r = build_retriever(EIGHT_WORDS, 4, 1, 0.5);
    let h = r.hierarchy();
    // Sum of leaf counts equals total leaves.
    let total_children: usize = h
        .nodes
        .iter()
        .filter(|n| !n.children.is_empty())
        .map(|n| n.children.len())
        .sum();
    assert_eq!(total_children, h.leaves().len());
}
