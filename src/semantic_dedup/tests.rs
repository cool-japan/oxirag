//! Tests for the `semantic_dedup` module.

use crate::types::{Document, DocumentId, SearchResult};

use super::deduplicator::SemanticDeduplicator;
use super::types::{DedupMethod, KeepPolicy, SemanticDedupConfig, SemanticDedupError};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn simhash_dedup() -> SemanticDeduplicator {
    SemanticDeduplicator::new(SemanticDedupConfig::default())
}

fn minhash_dedup() -> SemanticDeduplicator {
    SemanticDeduplicator::new(SemanticDedupConfig::default().with_method(DedupMethod::MinHash))
}

// Two passages that differ by a single token inside a long, shared body.
const LONG_A: &str =
    "The quick brown fox jumps over the lazy dog while the sun rises gently over hills";
const LONG_B: &str =
    "The quick brown fox jumps over the lazy cat while the sun rises gently over hills";
// A wholly unrelated passage.
const UNRELATED: &str = "Quantum entanglement links particles across vast distances defying classical intuition entirely";

// Long natural-language near-duplicates differing only in the final word, with
// enough shared shingles for the MinHash estimate to clear the 0.8 threshold.
const MH_A: &str = "the annual scientific conference brought together leading researchers from around the world to discuss recent breakthroughs in artificial intelligence machine learning robotics and computational biology this spring";
const MH_B: &str = "the annual scientific conference brought together leading researchers from around the world to discuss recent breakthroughs in artificial intelligence machine learning robotics and computational biology this autumn";
// Unrelated long passage with no shared shingles against MH_A / MH_B.
const MH_C: &str = "economic policy debates often center on questions of taxation government spending monetary supply inflation employment trade tariffs and the appropriate role of central banking institutions worldwide today";

// ── DedupMethod / KeepPolicy defaults ─────────────────────────────────────────

#[test]
fn test_dedup_method_default() {
    assert_eq!(DedupMethod::default(), DedupMethod::SimHash);
}

#[test]
fn test_keep_policy_default() {
    assert_eq!(KeepPolicy::default(), KeepPolicy::First);
}

// ── SemanticDedupConfig defaults ──────────────────────────────────────────────

#[test]
fn test_config_default_method() {
    assert_eq!(SemanticDedupConfig::default().method, DedupMethod::SimHash);
}

#[test]
fn test_config_default_shingle_size() {
    assert_eq!(SemanticDedupConfig::default().shingle_size, 2);
}

#[test]
fn test_config_default_num_perm() {
    assert_eq!(SemanticDedupConfig::default().num_perm, 64);
}

#[test]
fn test_config_default_simhash_max_hamming() {
    assert_eq!(SemanticDedupConfig::default().simhash_max_hamming, 3);
}

#[test]
fn test_config_default_minhash_min_jaccard() {
    assert!((SemanticDedupConfig::default().minhash_min_jaccard - 0.8).abs() < 1e-6);
}

#[test]
fn test_config_default_keep() {
    assert_eq!(SemanticDedupConfig::default().keep, KeepPolicy::First);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(SemanticDedupConfig::new(), SemanticDedupConfig::default());
}

// ── SemanticDedupConfig builders ──────────────────────────────────────────────

#[test]
fn test_builder_method() {
    let cfg = SemanticDedupConfig::new().with_method(DedupMethod::MinHash);
    assert_eq!(cfg.method, DedupMethod::MinHash);
}

#[test]
fn test_builder_shingle_size() {
    let cfg = SemanticDedupConfig::new().with_shingle_size(3);
    assert_eq!(cfg.shingle_size, 3);
}

#[test]
fn test_builder_shingle_size_clamped() {
    let cfg = SemanticDedupConfig::new().with_shingle_size(0);
    assert_eq!(cfg.shingle_size, 1);
}

#[test]
fn test_builder_num_perm() {
    let cfg = SemanticDedupConfig::new().with_num_perm(128);
    assert_eq!(cfg.num_perm, 128);
}

#[test]
fn test_builder_num_perm_clamped() {
    let cfg = SemanticDedupConfig::new().with_num_perm(0);
    assert_eq!(cfg.num_perm, 1);
}

#[test]
fn test_builder_simhash_max_hamming() {
    let cfg = SemanticDedupConfig::new().with_simhash_max_hamming(7);
    assert_eq!(cfg.simhash_max_hamming, 7);
}

#[test]
fn test_builder_minhash_min_jaccard() {
    let cfg = SemanticDedupConfig::new().with_minhash_min_jaccard(0.5);
    assert!((cfg.minhash_min_jaccard - 0.5).abs() < 1e-6);
}

#[test]
fn test_builder_keep() {
    let cfg = SemanticDedupConfig::new().with_keep(KeepPolicy::Longest);
    assert_eq!(cfg.keep, KeepPolicy::Longest);
}

#[test]
fn test_builder_chain() {
    let cfg = SemanticDedupConfig::new()
        .with_method(DedupMethod::MinHash)
        .with_num_perm(32)
        .with_keep(KeepPolicy::HighestScore);
    assert_eq!(cfg.keep, KeepPolicy::HighestScore);
}

// ── Deduplicator default ──────────────────────────────────────────────────────

#[test]
fn test_deduplicator_default_method() {
    let dedup = SemanticDeduplicator::default();
    assert_eq!(dedup.config.method, DedupMethod::SimHash);
}

// ── SimHash basics ────────────────────────────────────────────────────────────

#[test]
fn test_simhash_identical_equal() {
    let dedup = simhash_dedup();
    assert_eq!(dedup.simhash(LONG_A), dedup.simhash(LONG_A));
}

#[test]
fn test_simhash_identical_hamming_zero() {
    let dedup = simhash_dedup();
    let h = SemanticDeduplicator::hamming(dedup.simhash(LONG_A), dedup.simhash(LONG_A));
    assert_eq!(h, 0);
}

#[test]
fn test_simhash_near_duplicate_small_hamming() {
    let dedup = simhash_dedup();
    let h = SemanticDeduplicator::hamming(dedup.simhash(LONG_A), dedup.simhash(LONG_B));
    assert!(h <= 8);
}

#[test]
fn test_simhash_very_different_large_hamming() {
    let dedup = simhash_dedup();
    let h = SemanticDeduplicator::hamming(dedup.simhash(LONG_A), dedup.simhash(UNRELATED));
    assert!(h > 8);
}

#[test]
fn test_simhash_empty_is_zero() {
    let dedup = simhash_dedup();
    assert_eq!(dedup.simhash("   !!!  "), 0);
}

#[test]
fn test_hamming_known_value() {
    // 0b0000 vs 0b1011 differ in three bits.
    assert_eq!(SemanticDeduplicator::hamming(0b0000, 0b1011), 3);
}

#[test]
fn test_hamming_symmetric() {
    let dedup = simhash_dedup();
    let a = dedup.simhash(LONG_A);
    let b = dedup.simhash(UNRELATED);
    assert_eq!(
        SemanticDeduplicator::hamming(a, b),
        SemanticDeduplicator::hamming(b, a)
    );
}

#[test]
fn test_simhash_deterministic_across_instances() {
    let one = simhash_dedup().simhash(LONG_A);
    let two = simhash_dedup().simhash(LONG_A);
    assert_eq!(one, two);
}

// ── MinHash basics ────────────────────────────────────────────────────────────

#[test]
fn test_minhash_signature_length() {
    let dedup = minhash_dedup();
    assert_eq!(dedup.minhash_signature(LONG_A).len(), 64);
}

#[test]
fn test_minhash_signature_length_custom_perm() {
    let dedup = SemanticDeduplicator::new(SemanticDedupConfig::default().with_num_perm(16));
    assert_eq!(dedup.minhash_signature(LONG_A).len(), 16);
}

#[test]
fn test_minhash_identical_jaccard_one() {
    let dedup = minhash_dedup();
    let sig = dedup.minhash_signature(LONG_A);
    assert!((dedup.jaccard(&sig, &sig) - 1.0).abs() < 1e-6);
}

#[test]
fn test_minhash_disjoint_jaccard_zero() {
    let dedup = minhash_dedup();
    let a = dedup.minhash_signature("alpha beta gamma delta epsilon zeta");
    let b = dedup.minhash_signature("monday tuesday wednesday thursday friday saturday");
    assert!(dedup.jaccard(&a, &b) < 1e-6);
}

#[test]
fn test_minhash_overlap_between_zero_and_one() {
    let dedup = minhash_dedup();
    let a = dedup.minhash_signature(LONG_A);
    let b = dedup.minhash_signature(LONG_B);
    let j = dedup.jaccard(&a, &b);
    assert!(j > 0.3 && j < 1.0);
}

#[test]
fn test_minhash_signature_deterministic() {
    let dedup = minhash_dedup();
    assert_eq!(
        dedup.minhash_signature(LONG_A),
        dedup.minhash_signature(LONG_A)
    );
}

#[test]
fn test_jaccard_empty_signatures_zero() {
    let dedup = minhash_dedup();
    assert!(dedup.jaccard(&[], &[]).abs() < 1e-6);
}

#[test]
fn test_minhash_high_overlap_above_threshold() {
    let dedup = minhash_dedup();
    // Differ only in the final token; shingle overlap stays very high.
    let a = dedup.minhash_signature(MH_A);
    let b = dedup.minhash_signature(MH_B);
    assert!(dedup.jaccard(&a, &b) >= 0.8);
}

// ── is_duplicate ──────────────────────────────────────────────────────────────

#[test]
fn test_is_duplicate_simhash_near_dup_true() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    assert!(dedup.is_duplicate(LONG_A, LONG_B));
}

#[test]
fn test_is_duplicate_simhash_unrelated_false() {
    let dedup = simhash_dedup();
    assert!(!dedup.is_duplicate(LONG_A, UNRELATED));
}

#[test]
fn test_is_duplicate_simhash_identical_true() {
    let dedup = simhash_dedup();
    assert!(dedup.is_duplicate(LONG_A, LONG_A));
}

#[test]
fn test_is_duplicate_minhash_near_dup_true() {
    let dedup = minhash_dedup();
    assert!(dedup.is_duplicate(MH_A, MH_B));
}

#[test]
fn test_is_duplicate_minhash_unrelated_false() {
    let dedup = minhash_dedup();
    assert!(!dedup.is_duplicate(
        "alpha beta gamma delta epsilon",
        "monday tuesday wednesday thursday friday"
    ));
}

// ── cluster ───────────────────────────────────────────────────────────────────

#[test]
fn test_cluster_empty() {
    let dedup = simhash_dedup();
    assert!(dedup.cluster(&[]).is_empty());
}

#[test]
fn test_cluster_singletons_each_alone() {
    let dedup = simhash_dedup();
    let texts = [
        "alpha beta gamma delta epsilon zeta",
        "monday tuesday wednesday thursday friday saturday",
        "quantum entanglement links particles across distances defying intuition",
    ];
    let clusters = dedup.cluster(&texts);
    assert_eq!(clusters.len(), 3);
}

#[test]
fn test_cluster_groups_duplicates() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    let texts = [LONG_A, UNRELATED, LONG_B];
    let clusters = dedup.cluster(&texts);
    // LONG_A (0) and LONG_B (2) cluster; UNRELATED (1) alone => 2 clusters.
    assert_eq!(clusters.len(), 2);
}

#[test]
fn test_cluster_duplicate_pair_indices() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    let texts = [LONG_A, UNRELATED, LONG_B];
    let clusters = dedup.cluster(&texts);
    let dup_cluster = clusters.iter().find(|c| c.len() == 2).unwrap();
    assert_eq!(dup_cluster, &vec![0, 2]);
}

#[test]
fn test_cluster_identical_triplet_single_cluster() {
    let dedup = simhash_dedup();
    let texts = [LONG_A, LONG_A, LONG_A];
    let clusters = dedup.cluster(&texts);
    assert_eq!(clusters.len(), 1);
}

#[test]
fn test_cluster_covers_all_indices() {
    let dedup = simhash_dedup();
    let texts = [LONG_A, LONG_B, UNRELATED];
    let clusters = dedup.cluster(&texts);
    let total: usize = clusters.iter().map(Vec::len).sum();
    assert_eq!(total, 3);
}

#[test]
fn test_cluster_minhash_groups_duplicates() {
    let dedup = minhash_dedup();
    let texts = [MH_A, MH_B, MH_C];
    let clusters = dedup.cluster(&texts);
    assert_eq!(clusters.len(), 2);
}

// ── deduplicate ───────────────────────────────────────────────────────────────

#[test]
fn test_deduplicate_empty() {
    let dedup = simhash_dedup();
    assert!(dedup.deduplicate(&[]).is_empty());
}

#[test]
fn test_deduplicate_single_kept() {
    let dedup = simhash_dedup();
    let results = [make_result("a", LONG_A, 0.9)];
    assert_eq!(dedup.deduplicate(&results).len(), 1);
}

#[test]
fn test_deduplicate_removes_duplicate() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    let results = [
        make_result("a", LONG_A, 0.9),
        make_result("b", LONG_B, 0.8),
        make_result("c", UNRELATED, 0.7),
    ];
    assert_eq!(dedup.deduplicate(&results).len(), 2);
}

#[test]
fn test_deduplicate_no_duplicates_keeps_all() {
    let dedup = simhash_dedup();
    let results = [
        make_result("a", "alpha beta gamma delta epsilon zeta eta", 0.9),
        make_result(
            "b",
            "monday tuesday wednesday thursday friday saturday sunday",
            0.8,
        ),
        make_result("c", UNRELATED, 0.7),
    ];
    assert_eq!(dedup.deduplicate(&results).len(), 3);
}

#[test]
fn test_deduplicate_one_per_cluster() {
    let dedup = simhash_dedup();
    let results = [
        make_result("a", LONG_A, 0.9),
        make_result("b", LONG_A, 0.8),
        make_result("c", LONG_A, 0.7),
    ];
    assert_eq!(dedup.deduplicate(&results).len(), 1);
}

#[test]
fn test_deduplicate_reranks_from_zero() {
    let dedup = simhash_dedup();
    let results = [
        make_result("a", "alpha beta gamma delta epsilon zeta eta", 0.9),
        make_result("b", UNRELATED, 0.8),
    ];
    let out = dedup.deduplicate(&results);
    assert_eq!(out[1].rank, 1);
}

#[test]
fn test_deduplicate_first_rank_zero() {
    let dedup = simhash_dedup();
    let results = [
        make_result("a", "alpha beta gamma delta epsilon zeta eta", 0.9),
        make_result("b", UNRELATED, 0.8),
    ];
    let out = dedup.deduplicate(&results);
    assert_eq!(out[0].rank, 0);
}

// ── KeepPolicy selection ──────────────────────────────────────────────────────

#[test]
fn test_keep_first_picks_earliest() {
    let dedup = SemanticDeduplicator::new(
        SemanticDedupConfig::default()
            .with_simhash_max_hamming(8)
            .with_keep(KeepPolicy::First),
    );
    let results = [
        make_result("first", LONG_A, 0.5),
        make_result("second", LONG_B, 0.9),
    ];
    let out = dedup.deduplicate(&results);
    assert_eq!(out[0].document.id.as_str(), "first");
}

#[test]
fn test_keep_highest_score_picks_best() {
    let dedup = SemanticDeduplicator::new(
        SemanticDedupConfig::default()
            .with_simhash_max_hamming(8)
            .with_keep(KeepPolicy::HighestScore),
    );
    let results = [
        make_result("low", LONG_A, 0.3),
        make_result("high", LONG_B, 0.95),
    ];
    let out = dedup.deduplicate(&results);
    assert_eq!(out[0].document.id.as_str(), "high");
}

#[test]
fn test_keep_longest_picks_longest() {
    let dedup = SemanticDeduplicator::new(
        SemanticDedupConfig::default()
            .with_simhash_max_hamming(8)
            .with_keep(KeepPolicy::Longest),
    );
    let short =
        "renewable energy sources include solar wind hydro and geothermal power generation methods";
    let long = "renewable energy sources include solar wind hydro and geothermal power generation methods used widely";
    let results = [
        make_result("short", short, 0.9),
        make_result("long", long, 0.1),
    ];
    let out = dedup.deduplicate(&results);
    assert_eq!(out[0].document.id.as_str(), "long");
}

#[test]
fn test_keep_highest_score_keeps_single_survivor() {
    let dedup = SemanticDeduplicator::new(
        SemanticDedupConfig::default()
            .with_simhash_max_hamming(8)
            .with_keep(KeepPolicy::HighestScore),
    );
    let results = [
        make_result("low", LONG_A, 0.3),
        make_result("high", LONG_B, 0.95),
    ];
    assert_eq!(dedup.deduplicate(&results).len(), 1);
}

// ── Order preservation ────────────────────────────────────────────────────────

#[test]
fn test_deduplicate_preserves_order() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    // Items: unrelated(0), LONG_A(1), LONG_B(2 dup of 1). Keep First => 0 then 1.
    let results = [
        make_result("u", UNRELATED, 0.5),
        make_result("a", LONG_A, 0.5),
        make_result("b", LONG_B, 0.5),
    ];
    let out = dedup.deduplicate(&results);
    let ids: Vec<&str> = out.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids, vec!["u", "a"]);
}

// ── deduplicate_checked ───────────────────────────────────────────────────────

#[test]
fn test_deduplicate_checked_empty_errors() {
    let dedup = simhash_dedup();
    assert_eq!(
        dedup.deduplicate_checked(&[]).unwrap_err(),
        SemanticDedupError::EmptyInput
    );
}

#[test]
fn test_deduplicate_checked_non_empty_ok() {
    let dedup = simhash_dedup();
    let results = [make_result("a", LONG_A, 0.9)];
    assert_eq!(dedup.deduplicate_checked(&results).unwrap().len(), 1);
}

#[test]
fn test_error_display() {
    assert_eq!(SemanticDedupError::EmptyInput.to_string(), "empty input");
}

// ── Determinism across the full pipeline ──────────────────────────────────────

#[test]
fn test_deduplicate_deterministic() {
    let dedup =
        SemanticDeduplicator::new(SemanticDedupConfig::default().with_simhash_max_hamming(8));
    let results = [
        make_result("a", LONG_A, 0.9),
        make_result("b", LONG_B, 0.8),
        make_result("c", UNRELATED, 0.7),
    ];
    let first: Vec<String> = dedup
        .deduplicate(&results)
        .iter()
        .map(|r| r.document.id.0.clone())
        .collect();
    let second: Vec<String> = dedup
        .deduplicate(&results)
        .iter()
        .map(|r| r.document.id.0.clone())
        .collect();
    assert_eq!(first, second);
}

#[test]
fn test_cluster_minhash_deterministic() {
    let dedup = minhash_dedup();
    let texts = [LONG_A, LONG_B, UNRELATED];
    assert_eq!(dedup.cluster(&texts), dedup.cluster(&texts));
}
