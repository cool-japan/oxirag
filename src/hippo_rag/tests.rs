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

//! Tests for the `hippo_rag` module.

use crate::hippo_rag::graph::{extract_entities, personalized_pagerank};
use crate::hippo_rag::index::HippoRagIndex;
use crate::hippo_rag::types::{HippoConfig, HippoRagError};
use crate::types::Document;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a document with a fixed id so insertion order and identity are stable.
fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

/// Sum of a PPR rank vector.
fn mass(v: &[f32]) -> f32 {
    v.iter().sum()
}

/// Build an index over the given corpus, panicking on failure.
fn built(corpus: &[Document]) -> HippoRagIndex {
    let mut index = HippoRagIndex::new(HippoConfig::default());
    index.build(corpus).expect("build should succeed");
    index
}

/// Resolve an entity surface form to its node id, panicking when absent.
fn seed_of(index: &HippoRagIndex, surface: &str) -> usize {
    index
        .entity_id(surface)
        .unwrap_or_else(|| panic!("entity {surface} not found in index"))
}

// ── HippoConfig: defaults ─────────────────────────────────────────────────────

#[test]
fn config_default_damping() {
    assert_eq!(HippoConfig::default().damping, 0.85);
}

#[test]
fn config_default_ppr_iterations() {
    assert_eq!(HippoConfig::default().ppr_iterations, 50);
}

#[test]
fn config_default_dim() {
    assert_eq!(HippoConfig::default().dim, 128);
}

// ── HippoConfig: builders ─────────────────────────────────────────────────────

#[test]
fn config_with_damping() {
    let c = HippoConfig::new().with_damping(0.5);
    assert_eq!(c.damping, 0.5);
}

#[test]
fn config_with_ppr_iterations() {
    let c = HippoConfig::new().with_ppr_iterations(100);
    assert_eq!(c.ppr_iterations, 100);
}

#[test]
fn config_with_dim() {
    let c = HippoConfig::new().with_dim(256);
    assert_eq!(c.dim, 256);
}

#[test]
fn config_builders_chain() {
    let c = HippoConfig::new()
        .with_damping(0.7)
        .with_ppr_iterations(33)
        .with_dim(64);
    assert_eq!(c.damping, 0.7);
    assert_eq!(c.ppr_iterations, 33);
    assert_eq!(c.dim, 64);
}

// ── entity extraction ─────────────────────────────────────────────────────────

#[test]
fn extract_picks_capitalised_tokens() {
    let ents = extract_entities("Alice met Bob in Paris.");
    assert!(ents.contains(&"alice".to_string()));
    assert!(ents.contains(&"bob".to_string()));
    assert!(ents.contains(&"paris".to_string()));
}

#[test]
fn extract_skips_lowercase_tokens() {
    let ents = extract_entities("the quick brown fox");
    assert!(ents.is_empty());
}

#[test]
fn extract_skips_single_char_tokens() {
    let ents = extract_entities("A B Alice");
    // Single-character capitals are excluded; only the multi-char one survives.
    assert_eq!(ents, vec!["alice".to_string()]);
}

#[test]
fn extract_dedups_repeated_entities() {
    let ents = extract_entities("Alice and Alice and Alice");
    assert_eq!(ents, vec!["alice".to_string()]);
}

#[test]
fn extract_preserves_first_seen_order() {
    let ents = extract_entities("Carol Bob Alice");
    assert_eq!(
        ents,
        vec!["carol".to_string(), "bob".to_string(), "alice".to_string()]
    );
}

#[test]
fn extract_handles_punctuation_boundaries() {
    let ents = extract_entities("Alice,Bob;Carol");
    assert_eq!(ents.len(), 3);
}

#[test]
fn extract_empty_text_is_empty() {
    assert!(extract_entities("").is_empty());
}

// ── build: structural ─────────────────────────────────────────────────────────

#[test]
fn build_sets_len() {
    let index = built(&[doc("a", "Alice met Bob."), doc("b", "Carol saw Dave.")]);
    assert_eq!(index.len(), 2);
}

#[test]
fn new_index_is_empty() {
    let index = HippoRagIndex::new(HippoConfig::default());
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.entity_count(), 0);
}

#[test]
fn build_extracts_distinct_entities() {
    let index = built(&[doc("a", "Alice met Bob."), doc("b", "Bob met Carol.")]);
    // Alice, Bob, Carol → 3 distinct nodes (Bob shared).
    assert_eq!(index.entity_count(), 3);
}

#[test]
fn build_entity_count_dedups_across_passages() {
    let index = built(&[
        doc("a", "Alice Alice Alice."),
        doc("b", "Alice again Alice."),
    ]);
    assert_eq!(index.entity_count(), 1);
}

#[test]
fn rebuild_replaces_previous_corpus() {
    let mut index = HippoRagIndex::new(HippoConfig::default());
    index
        .build(&[doc("a", "Alice."), doc("b", "Bob."), doc("c", "Carol.")])
        .unwrap();
    assert_eq!(index.len(), 3);
    index.build(&[doc("x", "Xavier.")]).unwrap();
    assert_eq!(index.len(), 1);
    assert_eq!(index.entity_count(), 1);
}

// ── build: co-occurrence links ────────────────────────────────────────────────

#[test]
fn build_links_co_occurring_entities() {
    // Alice and Bob co-occur → seeding Alice should give Bob non-zero mass.
    let index = built(&[doc("a", "Alice met Bob.")]);
    let alice = seed_of(&index, "Alice");
    let ranks = index.personalized_pagerank(&[alice]);
    let bob = seed_of(&index, "Bob");
    assert!(ranks[bob] > 0.0, "Bob should receive mass from Alice");
}

#[test]
fn build_no_link_without_co_occurrence() {
    // Alice and Carol never share a passage and there is no bridge → Carol
    // gets no mass when seeding Alice.
    let index = built(&[doc("a", "Alice met Bob."), doc("b", "Carol saw Dave.")]);
    let alice = seed_of(&index, "Alice");
    let ranks = index.personalized_pagerank(&[alice]);
    let carol = seed_of(&index, "Carol");
    assert_eq!(ranks[carol], 0.0);
}

#[test]
fn build_co_occurrence_weight_grows_with_repetition() {
    // Build two indices: in the second, Alice-Bob co-occur in two passages.
    // The stronger edge should route more mass from Alice toward Bob.
    let weak = built(&[doc("a", "Alice Bob."), doc("b", "Alice Carol.")]);
    let strong = built(&[
        doc("a", "Alice Bob."),
        doc("b", "Alice Bob."),
        doc("c", "Alice Carol."),
    ]);
    let bob_weak = {
        let a = seed_of(&weak, "Alice");
        let b = seed_of(&weak, "Bob");
        weak.personalized_pagerank(&[a])[b]
    };
    let bob_strong = {
        let a = seed_of(&strong, "Alice");
        let b = seed_of(&strong, "Bob");
        strong.personalized_pagerank(&[a])[b]
    };
    assert!(
        bob_strong > bob_weak,
        "stronger co-occurrence should give Bob more PPR mass ({bob_strong} vs {bob_weak})"
    );
}

#[test]
fn build_single_entity_passage_has_no_edges() {
    // One lone entity → no co-occurrence edges; PPR puts all mass on the seed.
    let index = built(&[doc("a", "Alice.")]);
    let alice = seed_of(&index, "Alice");
    let ranks = index.personalized_pagerank(&[alice]);
    assert!((ranks[alice] - 1.0).abs() < 1e-4);
}

// ── personalized_pagerank: pure-function level ────────────────────────────────

#[test]
fn ppr_empty_graph_is_empty() {
    let ranks = personalized_pagerank(&[], &[], 0.85, 50);
    assert!(ranks.is_empty());
}

#[test]
fn ppr_no_seeds_is_zero() {
    // Two-node graph, no seed → zero vector.
    let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
    let ranks = personalized_pagerank(&adj, &[], 0.85, 50);
    assert_eq!(ranks, vec![0.0, 0.0]);
}

#[test]
fn ppr_mass_sums_to_one_with_seed() {
    let adj = vec![vec![(1, 1.0)], vec![(0, 1.0), (2, 1.0)], vec![(1, 1.0)]];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 100);
    assert!((mass(&ranks) - 1.0).abs() < 1e-4, "mass = {}", mass(&ranks));
}

#[test]
fn ppr_mass_decays_with_distance() {
    // Path graph 0-1-2-3 seeded at 0. The seed (a leaf) outranks the farthest
    // node, and mass strictly decays with graph distance once past the hub:
    // 1 > 2 > 3. (At high damping the degree-2 hub node 1 can edge out the
    // degree-1 leaf seed, which is a genuine property of personalized PageRank.)
    let adj = vec![
        vec![(1, 1.0)],
        vec![(0, 1.0), (2, 1.0)],
        vec![(1, 1.0), (3, 1.0)],
        vec![(2, 1.0)],
    ];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 200);
    assert!(ranks[0] > ranks[3], "seed should outrank the farthest node");
    assert!(ranks[1] > ranks[2]);
    assert!(ranks[2] > ranks[3]);
}

#[test]
fn ppr_star_seed_dominates() {
    // Star graph with node 0 at the centre; seeding the hub makes it the clear
    // maximum, and the symmetric leaves all share equal mass.
    let adj = vec![
        vec![(1, 1.0), (2, 1.0), (3, 1.0)],
        vec![(0, 1.0)],
        vec![(0, 1.0)],
        vec![(0, 1.0)],
    ];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 200);
    assert!(ranks[0] > ranks[1]);
    assert!((ranks[1] - ranks[2]).abs() < 1e-5);
    assert!((ranks[2] - ranks[3]).abs() < 1e-5);
}

#[test]
fn ppr_dangling_node_preserves_mass() {
    // Node 2 has no outgoing edges (dangling); mass must not leak.
    let adj = vec![vec![(1, 1.0)], vec![(0, 1.0), (2, 1.0)], vec![]];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 100);
    assert!((mass(&ranks) - 1.0).abs() < 1e-4, "mass = {}", mass(&ranks));
}

#[test]
fn ppr_isolated_seed_keeps_all_mass() {
    // Seed node 0 is isolated; node 1 is unreachable.
    let adj = vec![vec![], vec![]];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 50);
    assert!((ranks[0] - 1.0).abs() < 1e-4);
    assert_eq!(ranks[1], 0.0);
}

#[test]
fn ppr_two_seeds_split_mass() {
    // Disconnected pair; two seeds each hold ~half the mass.
    let adj = vec![vec![], vec![]];
    let ranks = personalized_pagerank(&adj, &[0, 1], 0.85, 50);
    assert!((ranks[0] - 0.5).abs() < 1e-4);
    assert!((ranks[1] - 0.5).abs() < 1e-4);
}

#[test]
fn ppr_damping_clamped_high() {
    // damping > 1 is clamped; mass still sums to ~1.
    let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
    let ranks = personalized_pagerank(&adj, &[0], 5.0, 50);
    assert!((mass(&ranks) - 1.0).abs() < 1e-3, "mass = {}", mass(&ranks));
}

#[test]
fn ppr_zero_damping_stays_on_seed() {
    // With damping 0 the walk never follows edges: all mass stays on the seed.
    let adj = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
    let ranks = personalized_pagerank(&adj, &[0], 0.0, 50);
    assert!((ranks[0] - 1.0).abs() < 1e-6);
    assert_eq!(ranks[1], 0.0);
}

#[test]
fn ppr_weighted_edge_favours_heavier_neighbour() {
    // Node 0 links to 1 (weight 3) and 2 (weight 1): 1 should outrank 2.
    let adj = vec![vec![(1, 3.0), (2, 1.0)], vec![(0, 3.0)], vec![(0, 1.0)]];
    let ranks = personalized_pagerank(&adj, &[0], 0.85, 100);
    assert!(ranks[1] > ranks[2]);
}

// ── search: multi-hop retrieval ───────────────────────────────────────────────

#[test]
fn search_returns_seeded_passage() {
    let index = built(&[doc("a", "Alice met Bob."), doc("b", "Carol saw Dave.")]);
    let hits = index.search("Alice", 5).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "a");
}

#[test]
fn search_retrieves_multi_hop_passage() {
    // Query mentions Alice only. Passage "c" mentions Carol, never Alice, but is
    // reachable Alice → Bob → Carol through shared entities. It must surface with
    // non-zero score: this is the defining HippoRAG single-step multi-hop case.
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol presented findings."),
    ]);
    let hits = index.search("Alice", 5).unwrap();
    let c = hits
        .iter()
        .find(|h| h.document.id.as_str() == "c")
        .expect("multi-hop passage c should be retrieved");
    assert!(
        c.score > 0.0,
        "passage c should have non-zero PPR-derived score"
    );
}

#[test]
fn search_direct_passage_outranks_multi_hop() {
    // The directly-seeded passage should outrank the two-hop-away one.
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol presented findings."),
    ]);
    let hits = index.search("Alice", 5).unwrap();
    let score_a = hits
        .iter()
        .find(|h| h.document.id.as_str() == "a")
        .unwrap()
        .score;
    let score_c = hits
        .iter()
        .find(|h| h.document.id.as_str() == "c")
        .unwrap()
        .score;
    assert!(score_a > score_c);
}

#[test]
fn search_unrelated_passage_scores_zero() {
    let index = built(&[doc("a", "Alice met Bob."), doc("b", "Xavier saw Yolanda.")]);
    let hits = index.search("Alice", 5).unwrap();
    let b = hits.iter().find(|h| h.document.id.as_str() == "b").unwrap();
    assert_eq!(b.score, 0.0);
}

#[test]
fn search_respects_top_k() {
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol met Dave."),
        doc("d", "Dave met Eve."),
    ]);
    let hits = index.search("Alice", 2).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_top_k_zero_returns_empty() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    let hits = index.search("Alice", 0).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_results_sorted_descending() {
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol met Dave."),
    ]);
    let hits = index.search("Alice", 5).unwrap();
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score);
    }
}

#[test]
fn search_multiple_query_entities() {
    // Seeding both endpoints of a chain pulls the middle passage up.
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol met Dave."),
    ]);
    let hits = index.search("Alice Dave", 5).unwrap();
    assert!(!hits.is_empty());
    // The middle passage b (Bob-Carol) should be reachable from both ends.
    assert!(hits.iter().any(|h| h.document.id.as_str() == "b"));
}

#[test]
fn search_ignores_unknown_query_entities() {
    // "Zelda" is unknown but "Alice" is a valid seed, so search still works.
    let index = built(&[doc("a", "Alice met Bob.")]);
    let hits = index.search("Zelda and Alice", 5).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "a");
}

// ── search: errors ────────────────────────────────────────────────────────────

#[test]
fn build_empty_corpus_errors() {
    let mut index = HippoRagIndex::new(HippoConfig::default());
    let err = index.build(&[]).unwrap_err();
    assert!(matches!(err, HippoRagError::EmptyCorpus));
}

#[test]
fn search_before_build_errors() {
    let index = HippoRagIndex::new(HippoConfig::default());
    let err = index.search("Alice", 5).unwrap_err();
    assert!(matches!(err, HippoRagError::NotBuilt));
}

#[test]
fn search_empty_query_errors() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    let err = index.search("", 5).unwrap_err();
    assert!(matches!(err, HippoRagError::EmptyQuery));
}

#[test]
fn search_no_entities_in_query_errors() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    // Lowercase-only query has no extractable entities.
    let err = index.search("the quick brown fox", 5).unwrap_err();
    assert!(matches!(err, HippoRagError::NoEntities));
}

#[test]
fn search_unknown_entities_only_errors() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    // "Zelda" is capitalised but absent from the corpus → no usable seed.
    let err = index.search("Zelda", 5).unwrap_err();
    assert!(matches!(err, HippoRagError::NoEntities));
}

#[test]
fn error_messages_render() {
    assert_eq!(HippoRagError::EmptyCorpus.to_string(), "corpus is empty");
    assert_eq!(
        HippoRagError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        HippoRagError::NoEntities.to_string(),
        "no entities found in query"
    );
    assert_eq!(HippoRagError::NotBuilt.to_string(), "index not built");
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn ppr_is_deterministic() {
    let index = built(&[
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol met Dave."),
    ]);
    let alice = seed_of(&index, "Alice");
    let first = index.personalized_pagerank(&[alice]);
    let second = index.personalized_pagerank(&[alice]);
    assert_eq!(first, second);
}

#[test]
fn search_is_deterministic() {
    let corpus = vec![
        doc("a", "Alice met Bob."),
        doc("b", "Bob met Carol."),
        doc("c", "Carol met Dave."),
    ];
    let first = built(&corpus).search("Alice", 5).unwrap();
    let second = built(&corpus).search("Alice", 5).unwrap();
    let ids_first: Vec<&str> = first.iter().map(|h| h.document.id.as_str()).collect();
    let ids_second: Vec<&str> = second.iter().map(|h| h.document.id.as_str()).collect();
    assert_eq!(ids_first, ids_second);
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.score, b.score);
    }
}

#[test]
fn search_tie_break_is_stable() {
    // Two passages with identical single entity get identical scores; insertion
    // order must break the tie so the first-built passage ranks first.
    let index = built(&[doc("a", "Alice."), doc("b", "Alice.")]);
    let hits = index.search("Alice", 5).unwrap();
    assert_eq!(hits[0].document.id.as_str(), "a");
    assert_eq!(hits[1].document.id.as_str(), "b");
    assert_eq!(hits[0].score, hits[1].score);
}

// ── entity_id lookup ──────────────────────────────────────────────────────────

#[test]
fn entity_id_resolves_known_entity() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    assert!(index.entity_id("Alice").is_some());
    assert!(index.entity_id("Bob").is_some());
}

#[test]
fn entity_id_is_case_insensitive() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    assert_eq!(index.entity_id("Alice"), index.entity_id("ALICE"));
    assert_eq!(index.entity_id("Alice"), index.entity_id("alice"));
}

#[test]
fn entity_id_unknown_is_none() {
    let index = built(&[doc("a", "Alice met Bob.")]);
    assert!(index.entity_id("Zelda").is_none());
}
