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
//! Tests for the `speculative_drafting` module.

use super::cluster::{embed, kmeans_lite};
use super::drafter::SpeculativeDrafter;
use super::types::{
    DraftVerifier, Drafter, MockDraftVerifier, MockDrafter, SpecDraftConfig, SpecDraftError,
    token_jaccard,
};
use crate::types::Document;

// ── Fixtures ───────────────────────────────────────────────────────────────────

fn doc(content: &str) -> Document {
    Document::new(content)
}

fn doc_titled(title: &str, content: &str) -> Document {
    Document::new(content).with_title(title)
}

/// Two clearly separable topics: photosynthesis vs. respiration.
fn two_topic_corpus() -> Vec<Document> {
    vec![
        doc("photosynthesis converts sunlight into chemical energy in green plants"),
        doc("chlorophyll inside the chloroplast captures sunlight for photosynthesis"),
        doc("plants use sunlight chlorophyll photosynthesis to build sugar energy"),
        doc("mitochondria produce atp through cellular respiration in animal cells"),
        doc("cellular respiration breaks down glucose to release atp inside mitochondria"),
        doc("animal cells rely on mitochondria respiration atp glucose for energy"),
    ]
}

// ── SpecDraftConfig: defaults & builders ─────────────────────────────────────────

#[test]
fn config_default_values() {
    let cfg = SpecDraftConfig::default();
    assert_eq!(cfg.num_clusters, 3);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.verify_weight, 0.6);
    assert_eq!(cfg.consistency_weight, 0.4);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(SpecDraftConfig::new(), SpecDraftConfig::default());
}

#[test]
fn config_builder_num_clusters() {
    let cfg = SpecDraftConfig::new().with_num_clusters(7);
    assert_eq!(cfg.num_clusters, 7);
    // Other fields untouched.
    assert_eq!(cfg.dim, 128);
}

#[test]
fn config_builder_dim() {
    let cfg = SpecDraftConfig::new().with_dim(64);
    assert_eq!(cfg.dim, 64);
}

#[test]
fn config_builder_verify_weight() {
    let cfg = SpecDraftConfig::new().with_verify_weight(0.9);
    assert_eq!(cfg.verify_weight, 0.9);
}

#[test]
fn config_builder_consistency_weight() {
    let cfg = SpecDraftConfig::new().with_consistency_weight(0.25);
    assert_eq!(cfg.consistency_weight, 0.25);
}

#[test]
fn config_builders_chain() {
    let cfg = SpecDraftConfig::new()
        .with_num_clusters(4)
        .with_dim(32)
        .with_verify_weight(0.7)
        .with_consistency_weight(0.3);
    assert_eq!(cfg.num_clusters, 4);
    assert_eq!(cfg.dim, 32);
    assert_eq!(cfg.verify_weight, 0.7);
    assert_eq!(cfg.consistency_weight, 0.3);
}

#[test]
fn config_clone_eq() {
    let cfg = SpecDraftConfig::new().with_dim(16);
    assert_eq!(cfg.clone(), cfg);
}

// ── embed: deterministic FNV pseudo-embeddings ───────────────────────────────────

#[test]
fn embed_is_deterministic() {
    let a = embed("photosynthesis converts sunlight energy", 64);
    let b = embed("photosynthesis converts sunlight energy", 64);
    assert_eq!(a, b);
}

#[test]
fn embed_respects_dim() {
    assert_eq!(embed("some text here", 32).len(), 32);
    assert_eq!(embed("some text here", 128).len(), 128);
}

#[test]
fn embed_zero_dim_is_empty() {
    assert!(embed("anything", 0).is_empty());
}

#[test]
fn embed_l2_normalised() {
    let v = embed("alpha beta gamma delta", 64);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn embed_ignores_short_tokens() {
    // Tokens shorter than 2 chars are dropped; "a" contributes nothing.
    let with_short = embed("a sunlight energy", 64);
    let without = embed("sunlight energy", 64);
    assert_eq!(with_short, without);
}

#[test]
fn embed_is_case_insensitive() {
    assert_eq!(embed("Sunlight Energy", 64), embed("sunlight energy", 64));
}

#[test]
fn embed_empty_text_is_zero() {
    let v = embed("", 64);
    assert!(v.iter().all(|x| *x == 0.0));
}

// ── kmeans_lite: partition properties ────────────────────────────────────────────

fn embed_all(docs: &[Document], dim: usize) -> Vec<Vec<f32>> {
    docs.iter().map(|d| embed(&d.content, dim)).collect()
}

#[test]
fn kmeans_empty_input() {
    assert!(kmeans_lite(&[], 3).is_empty());
}

#[test]
fn kmeans_partitions_all_indices() {
    let embs = embed_all(&two_topic_corpus(), 128);
    let groups = kmeans_lite(&embs, 3);
    let mut seen: Vec<usize> = groups.iter().flatten().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..embs.len()).collect::<Vec<_>>());
}

#[test]
fn kmeans_groups_are_disjoint() {
    let embs = embed_all(&two_topic_corpus(), 128);
    let groups = kmeans_lite(&embs, 3);
    let total: usize = groups.iter().map(Vec::len).sum();
    let unique: std::collections::HashSet<usize> = groups.iter().flatten().copied().collect();
    assert_eq!(total, unique.len());
}

#[test]
fn kmeans_at_most_k_groups() {
    let embs = embed_all(&two_topic_corpus(), 128);
    assert!(kmeans_lite(&embs, 3).len() <= 3);
    assert!(kmeans_lite(&embs, 2).len() <= 2);
    assert!(kmeans_lite(&embs, 1).len() <= 1);
}

#[test]
fn kmeans_no_empty_groups() {
    let embs = embed_all(&two_topic_corpus(), 128);
    let groups = kmeans_lite(&embs, 4);
    assert!(groups.iter().all(|g| !g.is_empty()));
}

#[test]
fn kmeans_k_capped_to_n() {
    let embs = embed_all(&[doc("only one document here")], 128);
    let groups = kmeans_lite(&embs, 5);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec![0]);
}

#[test]
fn kmeans_is_deterministic() {
    let embs = embed_all(&two_topic_corpus(), 128);
    assert_eq!(kmeans_lite(&embs, 3), kmeans_lite(&embs, 3));
}

#[test]
fn kmeans_sorted_by_first_index() {
    let embs = embed_all(&two_topic_corpus(), 128);
    let groups = kmeans_lite(&embs, 3);
    let firsts: Vec<usize> = groups.iter().map(|g| g[0]).collect();
    let mut sorted = firsts.clone();
    sorted.sort_unstable();
    assert_eq!(firsts, sorted);
}

#[test]
fn kmeans_single_cluster_holds_all() {
    let embs = embed_all(&two_topic_corpus(), 128);
    let groups = kmeans_lite(&embs, 1);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), embs.len());
}

// ── cluster_docs (via the drafter) ───────────────────────────────────────────────

#[test]
fn cluster_docs_empty_corpus() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::default());
    assert!(drafter.cluster_docs(&[]).is_empty());
}

#[test]
fn cluster_docs_partitions_all() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let groups = drafter.cluster_docs(&corpus);
    let mut seen: Vec<usize> = groups.iter().flatten().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..corpus.len()).collect::<Vec<_>>());
}

#[test]
fn cluster_docs_respects_num_clusters() {
    let corpus = two_topic_corpus();
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    assert!(drafter.cluster_docs(&corpus).len() <= 2);
}

#[test]
fn cluster_docs_non_overlapping() {
    let corpus = two_topic_corpus();
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let groups = drafter.cluster_docs(&corpus);
    let total: usize = groups.iter().map(Vec::len).sum();
    let unique: std::collections::HashSet<usize> = groups.iter().flatten().copied().collect();
    assert_eq!(total, unique.len());
    assert_eq!(total, corpus.len());
}

#[test]
fn cluster_docs_uses_titles() {
    // Title text should feed into the embedding without panicking.
    let corpus = vec![
        doc_titled("Energy", "plants convert light"),
        doc_titled("Respiration", "cells release atp"),
    ];
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    let groups = drafter.cluster_docs(&corpus);
    assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), 2);
}

#[test]
fn cluster_docs_deterministic() {
    let corpus = two_topic_corpus();
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    assert_eq!(drafter.cluster_docs(&corpus), drafter.cluster_docs(&corpus));
}

// ── MockDrafter ──────────────────────────────────────────────────────────────────

#[test]
fn mock_drafter_echoes_content() {
    let d = MockDrafter::new();
    let docs = vec![doc("photosynthesis needs sunlight.")];
    let refs: Vec<&Document> = docs.iter().collect();
    let out = d.draft("how do plants grow", &refs);
    assert!(out.contains("photosynthesis"));
    assert!(out.contains("how do plants grow"));
}

#[test]
fn mock_drafter_includes_title() {
    let d = MockDrafter::new();
    let docs = vec![doc_titled("Energy Cycle", "plants use light.")];
    let refs: Vec<&Document> = docs.iter().collect();
    let out = d.draft("q", &refs);
    assert!(out.contains("Energy Cycle"));
}

#[test]
fn mock_drafter_takes_leading_sentence() {
    let d = MockDrafter::new();
    let docs = vec![doc("first sentence here. second sentence ignored.")];
    let refs: Vec<&Document> = docs.iter().collect();
    let out = d.draft("q", &refs);
    assert!(out.contains("first sentence here"));
    assert!(!out.contains("second sentence"));
}

#[test]
fn mock_drafter_deterministic() {
    let d = MockDrafter::new();
    let docs = two_topic_corpus();
    let refs: Vec<&Document> = docs.iter().collect();
    assert_eq!(d.draft("query", &refs), d.draft("query", &refs));
}

// ── MockDraftVerifier ────────────────────────────────────────────────────────────

#[test]
fn verifier_full_support_when_grounded() {
    let v = MockDraftVerifier::new();
    let docs = vec![doc("sunlight chlorophyll energy sugar")];
    let refs: Vec<&Document> = docs.iter().collect();
    // Every non-query draft token appears in the docs.
    let score = v.verify("question", "sunlight chlorophyll energy", &refs);
    assert_eq!(score, 1.0);
}

#[test]
fn verifier_zero_support_when_ungrounded() {
    let v = MockDraftVerifier::new();
    let docs = vec![doc("sunlight chlorophyll energy")];
    let refs: Vec<&Document> = docs.iter().collect();
    let score = v.verify("question", "quantum entanglement spacetime", &refs);
    assert_eq!(score, 0.0);
}

#[test]
fn verifier_partial_support() {
    let v = MockDraftVerifier::new();
    let docs = vec![doc("sunlight energy")];
    let refs: Vec<&Document> = docs.iter().collect();
    // "sunlight" + "energy" grounded; "rocket" not → 2/3.
    let score = v.verify("question", "sunlight energy rocket", &refs);
    assert!((score - (2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn verifier_ignores_query_tokens() {
    let v = MockDraftVerifier::new();
    let docs = vec![doc("nothing relevant here")];
    let refs: Vec<&Document> = docs.iter().collect();
    // The draft is only query echo → no evidence tokens → 0.0.
    let score = v.verify("rocket science", "rocket science", &refs);
    assert_eq!(score, 0.0);
}

#[test]
fn verifier_in_unit_range() {
    let v = MockDraftVerifier::new();
    let docs = two_topic_corpus();
    let refs: Vec<&Document> = docs.iter().collect();
    let score = v.verify("energy", "mitochondria atp glucose rocket banana", &refs);
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn verifier_matches_title_tokens() {
    let v = MockDraftVerifier::new();
    let docs = vec![doc_titled("Mitochondria", "the cell content")];
    let refs: Vec<&Document> = docs.iter().collect();
    // "mitochondria" appears only in the title.
    let score = v.verify("q", "mitochondria", &refs);
    assert_eq!(score, 1.0);
}

#[test]
fn verifier_deterministic() {
    let v = MockDraftVerifier::new();
    let docs = two_topic_corpus();
    let refs: Vec<&Document> = docs.iter().collect();
    let s1 = v.verify("energy", "atp glucose", &refs);
    let s2 = v.verify("energy", "atp glucose", &refs);
    assert_eq!(s1, s2);
}

// ── token_jaccard helper ─────────────────────────────────────────────────────────

#[test]
fn jaccard_identical_is_one() {
    assert_eq!(
        token_jaccard("sunlight energy plants", "energy plants sunlight"),
        1.0
    );
}

#[test]
fn jaccard_disjoint_is_zero() {
    assert_eq!(token_jaccard("alpha beta", "gamma delta"), 0.0);
}

#[test]
fn jaccard_partial_overlap() {
    // {a,b,c} vs {b,c,d} → intersection 2, union 4 → 0.5.
    assert_eq!(token_jaccard("aa bb cc", "bb cc dd"), 0.5);
}

#[test]
fn jaccard_empty_is_zero() {
    assert_eq!(token_jaccard("", ""), 0.0);
}

// ── run: structural guarantees ───────────────────────────────────────────────────

#[test]
fn run_empty_query_errors() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::default());
    let docs = two_topic_corpus();
    let err = drafter
        .run("   ", &docs, &MockDrafter::new(), &MockDraftVerifier::new())
        .unwrap_err();
    assert!(matches!(err, SpecDraftError::EmptyQuery));
}

#[test]
fn run_empty_corpus_errors() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::default());
    let err = drafter
        .run(
            "a real query",
            &[],
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap_err();
    assert!(matches!(err, SpecDraftError::EmptyCorpus));
}

#[test]
fn run_one_draft_per_nonempty_cluster() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    let corpus = two_topic_corpus();
    let clusters = drafter.cluster_docs(&corpus);
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out.drafts.len(), clusters.len());
}

#[test]
fn run_cluster_ids_cover_all_drafts() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    let mut ids: Vec<usize> = out.drafts.iter().map(|d| d.cluster_id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..out.drafts.len()).collect::<Vec<_>>());
}

#[test]
fn run_support_score_wired_from_verifier() {
    // A verifier that always returns 0.42; every draft must echo it.
    struct FixedVerifier;
    impl DraftVerifier for FixedVerifier {
        fn verify(&self, _q: &str, _d: &str, _docs: &[&Document]) -> f32 {
            0.42
        }
    }
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let out = drafter
        .run("energy", &corpus, &MockDrafter::new(), &FixedVerifier)
        .unwrap();
    assert!(
        out.drafts
            .iter()
            .all(|d| (d.support_score - 0.42).abs() < 1e-5)
    );
}

#[test]
fn run_clamps_out_of_range_verifier() {
    // Verifier returns values outside [0,1]; support must be clamped.
    struct WildVerifier;
    impl DraftVerifier for WildVerifier {
        fn verify(&self, _q: &str, _d: &str, _docs: &[&Document]) -> f32 {
            5.0
        }
    }
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    let corpus = two_topic_corpus();
    let out = drafter
        .run("energy", &corpus, &MockDrafter::new(), &WildVerifier)
        .unwrap();
    assert!(out.drafts.iter().all(|d| d.support_score == 1.0));
}

// ── run: self-consistency behaviour ──────────────────────────────────────────────

/// Drafter that emits a fixed string per cluster id, ignoring docs.
struct ScriptedDrafter {
    scripts: Vec<String>,
}

impl Drafter for ScriptedDrafter {
    fn draft(&self, _query: &str, docs: &[&Document]) -> String {
        // Route by the smallest doc index present so clusters map to scripts
        // deterministically regardless of clustering order.
        let key = docs
            .iter()
            .filter_map(|d| d.content.split_whitespace().next())
            .filter_map(|w| w.parse::<usize>().ok())
            .min()
            .unwrap_or(0);
        self.scripts
            .get(key)
            .cloned()
            .unwrap_or_else(|| "default".to_string())
    }
}

#[test]
fn run_self_consistency_rewards_agreement() {
    // Build 4 singleton clusters: 3 near-identical drafts + 1 outlier.
    // Each doc starts with its index so ScriptedDrafter can route.
    let corpus = vec![
        doc("0 alpha"),
        doc("1 beta"),
        doc("2 gamma"),
        doc("3 delta"),
    ];
    let scripts = vec![
        "shared answer about photosynthesis energy".to_string(),
        "shared answer about photosynthesis energy".to_string(),
        "shared answer about photosynthesis energy".to_string(),
        "totally unrelated quantum spacetime physics".to_string(),
    ];
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(4));
    let out = drafter
        .run(
            "question",
            &corpus,
            &ScriptedDrafter { scripts },
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out.drafts.len(), 4);

    // Locate an agreeing draft and the outlier by content.
    let agree = out
        .drafts
        .iter()
        .find(|d| d.content.contains("photosynthesis"))
        .expect("agreeing draft present");
    let outlier = out
        .drafts
        .iter()
        .find(|d| d.content.contains("quantum"))
        .expect("outlier present");
    assert!(agree.self_consistency > outlier.self_consistency);
}

#[test]
fn run_three_similar_beat_one_outlier_on_consistency() {
    let corpus = vec![
        doc("0 alpha"),
        doc("1 beta"),
        doc("2 gamma"),
        doc("3 delta"),
    ];
    let scripts = vec![
        "energy from sunlight in plants".to_string(),
        "energy from sunlight in plants".to_string(),
        "energy from sunlight in plants".to_string(),
        "rockets fly to distant planets".to_string(),
    ];
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(4));
    let out = drafter
        .run(
            "q",
            &corpus,
            &ScriptedDrafter { scripts },
            &MockDraftVerifier::new(),
        )
        .unwrap();
    // The three identical drafts should each have positive consistency.
    let agreeing: Vec<f32> = out
        .drafts
        .iter()
        .filter(|d| d.content.contains("sunlight"))
        .map(|d| d.self_consistency)
        .collect();
    assert_eq!(agreeing.len(), 3);
    assert!(agreeing.iter().all(|c| *c > 0.0));
}

#[test]
fn run_single_cluster_zero_self_consistency() {
    // One cluster ⇒ a lone draft with no peers ⇒ self_consistency == 0.
    let corpus = two_topic_corpus();
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(1));
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out.drafts.len(), 1);
    assert_eq!(out.drafts[0].self_consistency, 0.0);
}

#[test]
fn run_identical_drafts_full_consistency() {
    // All three drafts identical ⇒ self_consistency == 1.0 for each.
    struct ConstDrafter;
    impl Drafter for ConstDrafter {
        fn draft(&self, _q: &str, _docs: &[&Document]) -> String {
            "exactly the same words every time".to_string()
        }
    }
    let corpus = vec![
        doc("alpha topic one"),
        doc("beta topic two"),
        doc("gamma topic three"),
    ];
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let out = drafter
        .run("q", &corpus, &ConstDrafter, &MockDraftVerifier::new())
        .unwrap();
    assert_eq!(out.drafts.len(), 3);
    assert!(
        out.drafts
            .iter()
            .all(|d| (d.self_consistency - 1.0).abs() < 1e-5)
    );
}

// ── run: total score blending ────────────────────────────────────────────────────

#[test]
fn run_total_is_weighted_blend() {
    let cfg = SpecDraftConfig::new()
        .with_num_clusters(3)
        .with_verify_weight(0.6)
        .with_consistency_weight(0.4);
    let drafter = SpeculativeDrafter::new(cfg.clone());
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    for d in &out.drafts {
        let expected =
            cfg.verify_weight * d.support_score + cfg.consistency_weight * d.self_consistency;
        assert!((d.total_score - expected).abs() < 1e-5);
    }
}

#[test]
fn run_total_pure_support_when_consistency_weight_zero() {
    let cfg = SpecDraftConfig::new()
        .with_num_clusters(3)
        .with_verify_weight(1.0)
        .with_consistency_weight(0.0);
    let drafter = SpeculativeDrafter::new(cfg);
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    for d in &out.drafts {
        assert!((d.total_score - d.support_score).abs() < 1e-5);
    }
}

#[test]
fn run_total_pure_consistency_when_verify_weight_zero() {
    let cfg = SpecDraftConfig::new()
        .with_num_clusters(3)
        .with_verify_weight(0.0)
        .with_consistency_weight(1.0);
    let drafter = SpeculativeDrafter::new(cfg);
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    for d in &out.drafts {
        assert!((d.total_score - d.self_consistency).abs() < 1e-5);
    }
}

// ── run: best selection & confidence ─────────────────────────────────────────────

#[test]
fn run_best_is_highest_total() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    let max_total = out
        .drafts
        .iter()
        .map(|d| d.total_score)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!((out.confidence - max_total).abs() < 1e-5);
    // best content equals the top-ranked draft's content.
    assert_eq!(out.best, out.drafts[0].content);
}

#[test]
fn run_drafts_sorted_descending_total() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    for w in out.drafts.windows(2) {
        assert!(w[0].total_score >= w[1].total_score - 1e-6);
    }
}

#[test]
fn run_confidence_equals_best_total() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out.confidence, out.drafts[0].total_score);
}

#[test]
fn run_confidence_in_unit_range() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::default());
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert!(out.confidence >= 0.0 && out.confidence <= 1.0);
}

#[test]
fn run_best_picks_strongest_blend() {
    // The grounded cluster gets a high-support draft; the outlier gets neither.
    // Best must be the strong one.
    struct RoutedDrafter;
    impl Drafter for RoutedDrafter {
        fn draft(&self, _q: &str, docs: &[&Document]) -> String {
            let key = docs
                .iter()
                .filter_map(|d| d.content.split_whitespace().next())
                .filter_map(|w| w.parse::<usize>().ok())
                .min()
                .unwrap_or(0);
            if key == 3 {
                "qqq zzz".to_string()
            } else {
                "sunlight energy chlorophyll plants".to_string()
            }
        }
    }
    let corpus = vec![
        doc("0 sunlight energy chlorophyll plants"),
        doc("1 sunlight energy chlorophyll plants"),
        doc("2 sunlight energy chlorophyll plants"),
        doc("3 banana"),
    ];
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(4));
    let out = drafter
        .run("topic", &corpus, &RoutedDrafter, &MockDraftVerifier::new())
        .unwrap();
    assert!(out.best.contains("sunlight"));
}

// ── run: determinism end-to-end ──────────────────────────────────────────────────

#[test]
fn run_is_deterministic() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(3));
    let corpus = two_topic_corpus();
    let out1 = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    let out2 = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out1.best, out2.best);
    assert_eq!(out1.confidence, out2.confidence);
    assert_eq!(out1.drafts.len(), out2.drafts.len());
    for (a, b) in out1.drafts.iter().zip(out2.drafts.iter()) {
        assert_eq!(a.content, b.content);
        assert_eq!(a.cluster_id, b.cluster_id);
        assert_eq!(a.support_score, b.support_score);
        assert_eq!(a.self_consistency, b.self_consistency);
        assert_eq!(a.total_score, b.total_score);
    }
}

#[test]
fn run_single_document_corpus() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::default());
    let corpus = vec![doc("photosynthesis converts sunlight into energy")];
    let out = drafter
        .run(
            "plants",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert_eq!(out.drafts.len(), 1);
    assert!(!out.best.is_empty());
}

#[test]
fn run_output_best_non_empty() {
    let drafter = SpeculativeDrafter::new(SpecDraftConfig::new().with_num_clusters(2));
    let corpus = two_topic_corpus();
    let out = drafter
        .run(
            "energy",
            &corpus,
            &MockDrafter::new(),
            &MockDraftVerifier::new(),
        )
        .unwrap();
    assert!(!out.best.is_empty());
}
