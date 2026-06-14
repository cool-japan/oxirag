//! Tests for the `noise_filter` module.
#![allow(clippy::float_cmp, clippy::similar_names)]

use crate::types::{Document, DocumentId, SearchResult};

use super::filter::NoiseFilter;
use super::types::{NoiseConfig, NoiseFilterError, NoiseReport, PassageAssessment};

// ── Test helpers ────────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank,
    }
}

/// A coherent on-topic set about Rust memory safety.
fn rust_set() -> Vec<SearchResult> {
    vec![
        make_result(
            "d1",
            "Rust enforces memory safety through its ownership and borrowing system.",
            0.9,
            0,
        ),
        make_result(
            "d2",
            "The Rust borrow checker prevents data races and dangling pointers at compile time.",
            0.85,
            1,
        ),
        make_result(
            "d3",
            "Ownership in Rust guarantees memory safety without a garbage collector.",
            0.8,
            2,
        ),
    ]
}

// ── NoiseConfig tests ───────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = NoiseConfig::default();
    assert_eq!(cfg.relevance_threshold, 0.15);
    assert_eq!(cfg.consensus_weight, 0.3);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.keep_min, 1);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(NoiseConfig::new(), NoiseConfig::default());
}

#[test]
fn test_config_individual_builders() {
    assert_eq!(
        NoiseConfig::new()
            .with_relevance_threshold(0.4)
            .relevance_threshold,
        0.4
    );
    assert_eq!(
        NoiseConfig::new()
            .with_consensus_weight(0.6)
            .consensus_weight,
        0.6
    );
    assert_eq!(NoiseConfig::new().with_dim(64).dim, 64);
    assert_eq!(NoiseConfig::new().with_keep_min(3).keep_min, 3);
}

#[test]
fn test_config_builder_chained() {
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(0.2)
        .with_consensus_weight(0.5)
        .with_dim(256)
        .with_keep_min(2);
    assert_eq!(cfg.relevance_threshold, 0.2);
    assert_eq!(cfg.consensus_weight, 0.5);
    assert_eq!(cfg.dim, 256);
    assert_eq!(cfg.keep_min, 2);
}

#[test]
fn test_filter_default_uses_default_config() {
    let filter = NoiseFilter::default();
    assert_eq!(*filter.config(), NoiseConfig::default());
}

#[test]
fn test_filter_config_accessor() {
    let cfg = NoiseConfig::new().with_dim(64);
    let filter = NoiseFilter::new(cfg.clone());
    assert_eq!(*filter.config(), cfg);
}

// ── assess: shape & ordering ────────────────────────────────────────────────────

#[test]
fn test_assess_one_per_passage() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let assessments = filter.assess("rust memory safety", &results);
    assert_eq!(assessments.len(), results.len());
}

#[test]
fn test_assess_indices_are_in_order() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    for (i, a) in filter.assess("rust", &rust_set()).iter().enumerate() {
        assert_eq!(a.index, i);
        assert!(a.index < 3);
    }
}

#[test]
fn test_assess_empty_results_empty_vec() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let assessments = filter.assess("rust", &[]);
    assert!(assessments.is_empty());
}

#[test]
fn test_assess_scores_in_unit_range() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let assessments = filter.assess("rust memory safety", &rust_set());
    for a in &assessments {
        assert!((0.0..=1.0).contains(&a.relevance));
        assert!((0.0..=1.0).contains(&a.consensus));
        assert!((0.0..=1.0).contains(&a.combined));
    }
}

// ── assess: relevance semantics ─────────────────────────────────────────────────

#[test]
fn test_relevant_passage_has_high_relevance_and_not_noise() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![make_result(
        "d1",
        "Rust memory safety comes from ownership and the borrow checker.",
        0.9,
        0,
    )];
    let assessments = filter.assess("rust memory safety ownership borrow checker", &results);
    assert!(
        assessments[0].relevance > 0.5,
        "expected high relevance, got {}",
        assessments[0].relevance
    );
    assert!(
        !assessments[0].is_noise,
        "clearly relevant passage must not be noise"
    );
}

#[test]
fn test_exact_match_passage_high_relevance() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    // Passage equals the query verbatim → relevance should be ~1.0.
    let text = "ownership borrowing lifetimes memory safety";
    let results = vec![make_result("d1", text, 0.9, 0)];
    let assessments = filter.assess(text, &results);
    assert!(
        assessments[0].relevance > 0.9,
        "verbatim match should have relevance ~1.0, got {}",
        assessments[0].relevance
    );
}

#[test]
fn test_offtopic_single_keyword_passage_is_noise() {
    // The query is about Rust memory safety; the off-topic passage shares no real
    // topical content with it. Lexical distractors are caught on the *relevance*
    // axis, so a relevance-dominant configuration flags it while the coherent
    // Rust set survives.
    let cfg = NoiseConfig::new()
        .with_consensus_weight(0.0)
        .with_relevance_threshold(0.2);
    let filter = NoiseFilter::new(cfg);
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "Photosynthesis converts sunlight into chemical energy stored as glucose inside plant cells.",
        0.6,
        3,
    ));
    let assessments = filter.assess("rust ownership borrow checker memory safety", &results);
    let noise = &assessments[3];
    assert!(
        noise.is_noise,
        "off-topic passage flagged: combined={}, relevance={}, consensus={}",
        noise.combined, noise.relevance, noise.consensus
    );
    // The coherent passages are not flagged under the same configuration.
    assert!(
        !assessments[0].is_noise && !assessments[1].is_noise && !assessments[2].is_noise,
        "coherent Rust passages must survive"
    );
}

#[test]
fn test_offtopic_passage_lower_combined_than_relevant() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![
        make_result(
            "rel",
            "Rust ownership and borrowing deliver memory safety.",
            0.9,
            0,
        ),
        make_result(
            "off",
            "Volcanic eruptions release ash and lava across the landscape.",
            0.5,
            1,
        ),
    ];
    let assessments = filter.assess("rust ownership borrowing memory safety", &results);
    assert!(assessments[0].combined > assessments[1].combined);
}

// ── assess: consensus semantics ─────────────────────────────────────────────────

#[test]
fn test_consensus_outlier_flagged() {
    // A topic-drift outlier: it shares query keywords but diverges from the set
    // consensus. With a high consensus weight, low consensus pulls it under the
    // threshold.
    let cfg = NoiseConfig::new()
        .with_consensus_weight(0.85)
        .with_relevance_threshold(0.45);
    let filter = NoiseFilter::new(cfg);
    let mut results = rust_set();
    // Outlier mentions "rust" (oxidation/iron) — drifts from the set consensus.
    results.push(make_result(
        "outlier",
        "Rust forms when iron reacts with oxygen and moisture producing reddish corrosion flakes.",
        0.7,
        3,
    ));
    let assessments = filter.assess("rust memory safety ownership", &results);
    // The three coherent passages share a higher consensus than the outlier.
    let consensus_set: f32 =
        (assessments[0].consensus + assessments[1].consensus + assessments[2].consensus) / 3.0;
    assert!(
        assessments[3].consensus < consensus_set,
        "outlier consensus {} should be below set mean {}",
        assessments[3].consensus,
        consensus_set
    );
}

#[test]
fn test_coherent_set_has_positive_consensus() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let assessments = filter.assess("rust", &results);
    for a in &assessments {
        assert!(
            a.consensus > 0.0,
            "coherent passage should have positive consensus"
        );
    }
}

#[test]
fn test_single_passage_consensus_is_self_similarity() {
    // With one passage, the centroid equals that passage (after normalisation),
    // so consensus should be ~1.0.
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![make_result("d1", "rust ownership borrowing safety", 0.9, 0)];
    let assessments = filter.assess("rust", &results);
    assert!(
        assessments[0].consensus > 0.99,
        "single-passage consensus should be ~1.0, got {}",
        assessments[0].consensus
    );
}

// ── combined formula respects consensus_weight ──────────────────────────────────

#[test]
fn test_combined_formula_zero_weight_equals_relevance() {
    let filter = NoiseFilter::new(NoiseConfig::new().with_consensus_weight(0.0));
    for a in &filter.assess("rust memory safety", &rust_set()) {
        assert!((a.combined - a.relevance).abs() < 1e-5);
    }
}

#[test]
fn test_combined_formula_full_weight_equals_consensus() {
    let filter = NoiseFilter::new(NoiseConfig::new().with_consensus_weight(1.0));
    for a in &filter.assess("rust memory safety", &rust_set()) {
        assert!((a.combined - a.consensus).abs() < 1e-5);
    }
}

#[test]
fn test_combined_formula_blend_matches_manual() {
    let cw = 0.3f32;
    let filter = NoiseFilter::new(NoiseConfig::new().with_consensus_weight(cw));
    for a in &filter.assess("rust memory safety", &rust_set()) {
        let expected = ((1.0 - cw) * a.relevance + cw * a.consensus).clamp(0.0, 1.0);
        assert!((a.combined - expected).abs() < 1e-5);
    }
}

#[test]
fn test_higher_consensus_weight_raises_combined_for_consensus_passage() {
    // For a passage whose consensus exceeds its relevance, raising the consensus
    // weight should not lower its combined score.
    let results = rust_set();
    let a_low =
        NoiseFilter::new(NoiseConfig::new().with_consensus_weight(0.1)).assess("memory", &results);
    let a_high =
        NoiseFilter::new(NoiseConfig::new().with_consensus_weight(0.9)).assess("memory", &results);
    for i in 0..results.len() {
        if a_low[i].consensus > a_low[i].relevance {
            assert!(a_high[i].combined >= a_low[i].combined - 1e-6);
        }
    }
}

// ── filter: removal & re-ranking ────────────────────────────────────────────────

#[test]
fn test_filter_removes_flagged_passages() {
    let cfg = NoiseConfig::new()
        .with_consensus_weight(0.0)
        .with_relevance_threshold(0.2);
    let filter = NoiseFilter::new(cfg);
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "Jellyfish drift through ocean currents using gelatinous bells and stinging tentacles.",
        0.6,
        3,
    ));
    let kept = filter.filter("rust ownership borrow checker memory safety", &results);
    assert!(
        kept.len() < results.len(),
        "noise passage should be dropped"
    );
    assert!(
        kept.iter().all(|r| r.document.id.as_str() != "noise"),
        "the off-topic passage must not survive"
    );
}

#[test]
fn test_filter_keeps_all_when_none_flagged() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let kept = filter.filter("rust memory safety ownership", &results);
    assert_eq!(
        kept.len(),
        results.len(),
        "coherent set should be fully kept"
    );
}

#[test]
fn test_filter_reranks_survivors_from_zero() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "The Eiffel Tower is a wrought-iron lattice structure in Paris France.",
        0.6,
        3,
    ));
    let kept = filter.filter("rust ownership borrow checker memory safety", &results);
    for (i, r) in kept.iter().enumerate() {
        assert_eq!(r.rank, i, "ranks must be contiguous from 0");
    }
}

#[test]
fn test_filter_preserves_input_order_of_survivors() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let kept = filter.filter("rust memory safety ownership", &results);
    // All survive (coherent set) → ids stay in input order.
    let ids: Vec<&str> = kept.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids, vec!["d1", "d2", "d3"]);
}

#[test]
fn test_filter_preserves_score_field() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let kept = filter.filter("rust memory safety ownership", &results);
    // Only the rank is rewritten; the original similarity score is untouched.
    assert_eq!(kept[0].score, 0.9);
    assert_eq!(kept[1].score, 0.85);
    assert_eq!(kept[2].score, 0.8);
}

#[test]
fn test_filter_empty_results_empty_output() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let kept = filter.filter("rust", &[]);
    assert!(kept.is_empty());
}

#[test]
fn test_filter_high_threshold_flags_everything_but_keeps_floor() {
    // With an unreachable threshold every passage is flagged, yet keep_min must
    // still be honoured.
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(1);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let kept = filter.filter("rust", &results);
    assert_eq!(kept.len(), 1, "all-noise input must still keep keep_min=1");
}

// ── keep_min floor ──────────────────────────────────────────────────────────────

#[test]
fn test_keep_min_floor_all_noise_keeps_best() {
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(2);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let kept = filter.filter("rust", &results);
    assert_eq!(kept.len(), 2, "keep_min=2 must retain two passages");
}

#[test]
fn test_keep_min_keeps_highest_combined() {
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(1);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let assessments = filter.assess("rust memory safety", &results);
    // Identify the index with the highest combined score (tie-break by index).
    let best = (0..results.len())
        .max_by(|&a, &b| {
            assessments[a]
                .combined
                .partial_cmp(&assessments[b].combined)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.cmp(&a))
        })
        .unwrap();
    let best_id = results[best].document.id.as_str().to_string();
    let kept = filter.filter("rust memory safety", &results);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].document.id.as_str(), best_id);
}

#[test]
fn test_keep_min_clamped_to_result_count() {
    // keep_min larger than the number of results must not panic; it keeps all.
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(10);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let kept = filter.filter("rust", &results);
    assert_eq!(kept.len(), results.len());
}

#[test]
fn test_keep_min_zero_can_drop_everything() {
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(0);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let kept = filter.filter("rust", &results);
    assert!(kept.is_empty(), "keep_min=0 with all-noise → empty output");
}

#[test]
fn test_keep_min_does_not_force_extra_when_enough_survive() {
    // A coherent set already passes the threshold; keep_min=1 should not trim it.
    let cfg = NoiseConfig::new().with_keep_min(1);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let kept = filter.filter("rust memory safety", &results);
    assert_eq!(kept.len(), results.len());
}

// ── report counts ───────────────────────────────────────────────────────────────

#[test]
fn test_report_kept_plus_removed_equals_total() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "Saturn's rings are composed mostly of ice particles and rocky debris.",
        0.6,
        3,
    ));
    let (kept, report) = filter.filter_with_report("rust ownership borrow checker", &results);
    assert_eq!(report.kept + report.removed, results.len());
    assert_eq!(report.kept, kept.len());
}

#[test]
fn test_report_assessments_one_per_passage() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let (_, report) = filter.filter_with_report("rust", &results);
    assert_eq!(report.assessments.len(), results.len());
}

#[test]
fn test_report_empty_input() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let (kept, report) = filter.filter_with_report("rust", &[]);
    assert!(kept.is_empty());
    assert_eq!(report.kept, 0);
    assert_eq!(report.removed, 0);
    assert!(report.assessments.is_empty());
}

#[test]
fn test_report_coherent_set_zero_removed() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let (_, report) = filter.filter_with_report("rust memory safety", &results);
    assert_eq!(report.removed, 0);
    assert_eq!(report.kept, results.len());
}

#[test]
fn test_report_removed_counts_flagged() {
    let cfg = NoiseConfig::new()
        .with_relevance_threshold(2.0)
        .with_keep_min(1);
    let filter = NoiseFilter::new(cfg);
    let results = rust_set();
    let (_, report) = filter.filter_with_report("rust", &results);
    // All flagged, but keep_min=1 retained → removed = total - 1.
    assert_eq!(report.kept, 1);
    assert_eq!(report.removed, results.len() - 1);
}

#[test]
fn test_filter_and_report_agree() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "Honeybees communicate the location of flowers through a waggle dance.",
        0.6,
        3,
    ));
    let kept_plain = filter.filter("rust ownership borrow checker", &results);
    let (kept_report, _) = filter.filter_with_report("rust ownership borrow checker", &results);
    let ids_plain: Vec<&str> = kept_plain.iter().map(|r| r.document.id.as_str()).collect();
    let ids_report: Vec<&str> = kept_report.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids_plain, ids_report);
}

// ── filter_checked ──────────────────────────────────────────────────────────────

#[test]
fn test_filter_checked_empty_query_errors() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let err = filter
        .filter_checked("", &results)
        .expect_err("empty query must error");
    assert_eq!(err, NoiseFilterError::EmptyQuery);
}

#[test]
fn test_filter_checked_whitespace_query_errors() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let err = filter
        .filter_checked("   \t\n", &results)
        .expect_err("whitespace query must error");
    assert_eq!(err, NoiseFilterError::EmptyQuery);
}

#[test]
fn test_filter_checked_ok_matches_filter() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let checked = filter
        .filter_checked("rust memory safety", &results)
        .expect("non-empty query ok");
    let plain = filter.filter("rust memory safety", &results);
    let ids_checked: Vec<&str> = checked.iter().map(|r| r.document.id.as_str()).collect();
    let ids_plain: Vec<&str> = plain.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids_checked, ids_plain);
}

#[test]
fn test_filter_checked_empty_results_ok() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let kept = filter
        .filter_checked("rust", &[])
        .expect("non-empty query ok");
    assert!(kept.is_empty());
}

#[test]
fn test_error_display_message() {
    assert_eq!(
        NoiseFilterError::EmptyQuery.to_string(),
        "query must not be empty"
    );
}

// ── determinism ─────────────────────────────────────────────────────────────────

#[test]
fn test_assess_deterministic() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let a1 = filter.assess("rust memory safety", &results);
    let a2 = filter.assess("rust memory safety", &results);
    assert_eq!(a1, a2);
}

#[test]
fn test_filter_deterministic() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let mut results = rust_set();
    results.push(make_result(
        "noise",
        "Glaciers slowly carve valleys as they advance and retreat over millennia.",
        0.6,
        3,
    ));
    let k1 = filter.filter("rust ownership borrow checker", &results);
    let k2 = filter.filter("rust ownership borrow checker", &results);
    let ids1: Vec<&str> = k1.iter().map(|r| r.document.id.as_str()).collect();
    let ids2: Vec<&str> = k2.iter().map(|r| r.document.id.as_str()).collect();
    assert_eq!(ids1, ids2);
}

#[test]
fn test_report_deterministic() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = rust_set();
    let (_, r1) = filter.filter_with_report("rust", &results);
    let (_, r2) = filter.filter_with_report("rust", &results);
    assert_eq!(r1, r2);
}

// ── edge cases ──────────────────────────────────────────────────────────────────

#[test]
fn test_single_passage_kept() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![make_result(
        "d1",
        "rust ownership memory safety borrowing",
        0.9,
        0,
    )];
    let kept = filter.filter("rust memory safety", &results);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].rank, 0);
}

#[test]
fn test_empty_passage_content_does_not_panic() {
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![
        make_result("d1", "rust memory safety ownership", 0.9, 0),
        make_result("empty", "", 0.5, 1),
    ];
    let assessments = filter.assess("rust memory safety", &results);
    assert_eq!(assessments.len(), 2);
    // The empty passage has no tokens → zero relevance & consensus.
    assert_eq!(assessments[1].relevance, 0.0);
    assert_eq!(assessments[1].consensus, 0.0);
    assert!(assessments[1].is_noise);
}

#[test]
fn test_dim_zero_does_not_panic() {
    // Degenerate config: zero-dim embeddings collapse the cosine terms to 0, so
    // consensus is 0 for every passage and relevance reduces to the lexical
    // overlap term alone. The pass must not panic and must still produce one
    // assessment per passage.
    let filter = NoiseFilter::new(NoiseConfig::new().with_dim(0).with_keep_min(1));
    let results = rust_set();
    let assessments = filter.assess("rust", &results);
    assert_eq!(assessments.len(), results.len());
    for a in &assessments {
        assert_eq!(
            a.consensus, 0.0,
            "zero-dim embeddings have no consensus signal"
        );
    }
    let kept = filter.filter("rust", &results);
    assert!(
        !kept.is_empty(),
        "keep_min floor guarantees at least one survivor"
    );
}

#[test]
fn test_dim_zero_no_shared_tokens_keeps_floor() {
    // With zero-dim embeddings AND a query sharing no tokens with any passage,
    // both relevance and consensus are 0 for everything, so every passage is
    // flagged — yet the keep_min floor still retains one.
    let filter = NoiseFilter::new(NoiseConfig::new().with_dim(0).with_keep_min(1));
    let results = rust_set();
    let assessments = filter.assess("zzzz qqqq", &results);
    assert!(
        assessments.iter().all(|a| a.is_noise),
        "all flagged with no overlap"
    );
    let kept = filter.filter("zzzz qqqq", &results);
    assert_eq!(kept.len(), 1, "keep_min=1 honoured under all-noise");
}

#[test]
fn test_threshold_boundary_keeps_passage_at_threshold() {
    // A passage scoring exactly the threshold is kept (strict `<` comparison).
    let filter = NoiseFilter::new(NoiseConfig::new());
    let results = vec![make_result(
        "d1",
        "rust ownership memory safety borrowing",
        0.9,
        0,
    )];
    let assessments = filter.assess("rust memory safety", &results);
    let combined = assessments[0].combined;
    // Build a filter whose threshold equals this passage's combined score.
    let exact = NoiseFilter::new(NoiseConfig::new().with_relevance_threshold(combined));
    let a2 = exact.assess("rust memory safety", &results);
    assert!(
        !a2[0].is_noise,
        "passage at exactly the threshold must be kept"
    );
}

#[test]
fn test_public_types_constructible() {
    // Exercise the public struct fields so the API surface stays usable.
    let assessment = PassageAssessment {
        index: 0,
        relevance: 0.5,
        consensus: 0.4,
        combined: 0.47,
        is_noise: false,
    };
    let report = NoiseReport {
        kept: 1,
        removed: 0,
        assessments: vec![assessment.clone()],
    };
    assert_eq!(report.assessments[0], assessment);
    assert_eq!(report.kept, 1);
}
