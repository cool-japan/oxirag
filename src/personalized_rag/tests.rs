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

use super::reranker::PersonalizedReranker;
use super::types::{PersonalizedConfig, PersonalizedError, UserProfile};
use crate::types::{Document, DocumentId, SearchResult};

// ── Fixtures ────────────────────────────────────────────────────────────────

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

fn result(id: &str, content: &str, score: f32, rank: usize) -> SearchResult {
    SearchResult::new(doc(id, content), score, rank)
}

fn default_reranker() -> PersonalizedReranker {
    PersonalizedReranker::new(PersonalizedConfig::default())
}

// ── PersonalizedConfig ──────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    let config = PersonalizedConfig::default();
    assert!((config.personalization_weight - 0.3).abs() < f32::EPSILON);
    assert_eq!(config.dim, 128);
}

#[test]
fn test_config_new_matches_default() {
    assert_eq!(PersonalizedConfig::new(), PersonalizedConfig::default());
}

#[test]
fn test_config_with_personalization_weight() {
    let config = PersonalizedConfig::new().with_personalization_weight(0.7);
    assert!((config.personalization_weight - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_config_with_personalization_weight_clamps_high() {
    let config = PersonalizedConfig::new().with_personalization_weight(2.5);
    assert!((config.personalization_weight - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_with_personalization_weight_clamps_low() {
    let config = PersonalizedConfig::new().with_personalization_weight(-1.0);
    assert!((config.personalization_weight - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_with_dim() {
    let config = PersonalizedConfig::new().with_dim(256);
    assert_eq!(config.dim, 256);
}

#[test]
fn test_config_builder_chaining() {
    let config = PersonalizedConfig::new()
        .with_personalization_weight(0.5)
        .with_dim(64);
    assert!((config.personalization_weight - 0.5).abs() < f32::EPSILON);
    assert_eq!(config.dim, 64);
}

// ── UserProfile ─────────────────────────────────────────────────────────────

#[test]
fn test_profile_new_is_empty() {
    let profile = UserProfile::new();
    assert!(profile.is_empty());
    assert!(profile.topics().is_empty());
    assert!(profile.history().is_empty());
}

#[test]
fn test_profile_default_is_empty() {
    assert!(UserProfile::default().is_empty());
}

#[test]
fn test_profile_with_interest_populates() {
    let profile = UserProfile::new().with_interest("rust", 0.8);
    assert!(!profile.is_empty());
    assert!((profile.interest("rust") - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_profile_interest_case_insensitive() {
    let profile = UserProfile::new().with_interest("Rust", 0.9);
    assert!((profile.interest("rust") - 0.9).abs() < f32::EPSILON);
    assert!((profile.interest("RUST") - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_profile_interest_missing_topic_is_zero() {
    let profile = UserProfile::new().with_interest("rust", 0.5);
    assert!((profile.interest("python") - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_profile_interest_unknown_on_empty_is_zero() {
    let profile = UserProfile::new();
    assert!((profile.interest("anything") - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_profile_with_interest_overwrites() {
    let profile = UserProfile::new()
        .with_interest("rust", 0.3)
        .with_interest("rust", 0.9);
    assert!((profile.interest("rust") - 0.9).abs() < f32::EPSILON);
    assert_eq!(profile.topics().len(), 1);
}

#[test]
fn test_profile_with_interest_negative_clamped_to_zero() {
    let profile = UserProfile::new().with_interest("rust", -2.0);
    assert!((profile.interest("rust") - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_profile_multiple_interests() {
    let profile = UserProfile::new()
        .with_interest("rust", 1.0)
        .with_interest("safety", 0.5)
        .with_interest("performance", 0.7);
    assert_eq!(profile.topics().len(), 3);
    assert!((profile.interest("safety") - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_profile_add_history() {
    let mut profile = UserProfile::new();
    profile.add_history("rust memory safety guide");
    assert!(!profile.is_empty());
    assert_eq!(profile.history().len(), 1);
}

#[test]
fn test_profile_add_history_blank_ignored() {
    let mut profile = UserProfile::new();
    profile.add_history("   ");
    profile.add_history("");
    assert!(profile.is_empty());
    assert!(profile.history().is_empty());
}

#[test]
fn test_profile_add_history_multiple() {
    let mut profile = UserProfile::new();
    profile.add_history("first interaction text");
    profile.add_history("second interaction text");
    assert_eq!(profile.history().len(), 2);
}

#[test]
fn test_profile_history_only_not_empty() {
    let mut profile = UserProfile::new();
    profile.add_history("clicked document about gardening");
    assert!(!profile.is_empty());
    assert!(profile.topics().is_empty());
}

// ── profile_affinity: topic interest ────────────────────────────────────────

#[test]
fn test_affinity_empty_profile_is_zero() {
    let reranker = default_reranker();
    let profile = UserProfile::new();
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust is fast and safe"));
    assert!((affinity - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_affinity_topic_match_high() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let matching = reranker.profile_affinity(&profile, &doc("d1", "rust delivers memory safety"));
    let non_matching = reranker.profile_affinity(&profile, &doc("d2", "python is dynamic"));
    assert!(matching > non_matching);
    assert!(matching > 0.0);
}

#[test]
fn test_affinity_topic_full_match_is_one() {
    // Single interest topic that appears in the document => topic_score == 1.0,
    // and with no history the affinity equals the topic score.
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust programming language"));
    assert!((affinity - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_affinity_topic_no_match_is_zero() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "cooking pasta recipes"));
    assert!((affinity - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_affinity_topic_partial_match_between_zero_and_one() {
    // Two equally weighted topics, only one present => topic score 0.5.
    let reranker = default_reranker();
    let profile = UserProfile::new()
        .with_interest("rust", 1.0)
        .with_interest("python", 1.0);
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust is great"));
    assert!(affinity > 0.0 && affinity < 1.0);
    assert!((affinity - 0.5).abs() < 1e-4);
}

#[test]
fn test_affinity_weighted_topics() {
    // Heavier weight on the present topic should raise affinity beyond the
    // unweighted midpoint.
    let reranker = default_reranker();
    let profile = UserProfile::new()
        .with_interest("rust", 3.0)
        .with_interest("python", 1.0);
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust is great"));
    assert!(affinity > 0.5);
    assert!((affinity - 0.75).abs() < 1e-4);
}

#[test]
fn test_affinity_matches_title_tokens() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let document = Document::new("a body without the keyword")
        .with_id(DocumentId::from_string("d1"))
        .with_title("Rust handbook");
    let affinity = reranker.profile_affinity(&profile, &document);
    assert!(affinity > 0.0);
}

#[test]
fn test_affinity_in_unit_range() {
    let reranker = default_reranker();
    let mut profile = UserProfile::new().with_interest("rust", 5.0);
    profile.add_history("rust rust rust everywhere");
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust rust rust"));
    assert!(affinity >= 0.0 && affinity <= 1.0);
}

// ── profile_affinity: history centroid ──────────────────────────────────────

#[test]
fn test_affinity_history_centroid_high_for_similar() {
    let reranker = default_reranker();
    let mut profile = UserProfile::new();
    profile.add_history("rust memory safety ownership borrow checker");
    let similar =
        reranker.profile_affinity(&profile, &doc("d1", "rust ownership and borrow checker"));
    let dissimilar =
        reranker.profile_affinity(&profile, &doc("d2", "gardening tomato soil watering"));
    assert!(similar > dissimilar);
    assert!(similar > 0.0);
}

#[test]
fn test_affinity_history_only_no_overlap_is_zero() {
    let reranker = default_reranker();
    let mut profile = UserProfile::new();
    profile.add_history("astronomy telescope galaxy nebula");
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "kitchen recipe cooking flour"));
    assert!((affinity - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_affinity_history_identical_text_strong() {
    let reranker = default_reranker();
    let mut profile = UserProfile::new();
    profile.add_history("quantum computing qubits entanglement");
    let affinity = reranker.profile_affinity(
        &profile,
        &doc("d1", "quantum computing qubits entanglement"),
    );
    assert!(affinity > 0.9);
}

#[test]
fn test_affinity_combines_topic_and_history() {
    // With both signals present, affinity is the mean of the two; verify it
    // stays within the bracket of the individual signals.
    let reranker = default_reranker();
    let mut profile = UserProfile::new().with_interest("rust", 1.0);
    profile.add_history("rust ownership and lifetimes");
    let affinity =
        reranker.profile_affinity(&profile, &doc("d1", "rust ownership lifetimes safety"));
    assert!(affinity > 0.0 && affinity <= 1.0);
}

#[test]
fn test_affinity_history_ignored_when_dim_zero() {
    // dim == 0 disables the embedding half; a history-only profile then yields
    // zero affinity.
    let reranker = PersonalizedReranker::new(PersonalizedConfig::new().with_dim(0));
    let mut profile = UserProfile::new();
    profile.add_history("rust ownership lifetimes");
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust ownership lifetimes"));
    assert!((affinity - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_affinity_topic_survives_dim_zero() {
    // With dim == 0 the topic-interest signal still works.
    let reranker = PersonalizedReranker::new(PersonalizedConfig::new().with_dim(0));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let affinity = reranker.profile_affinity(&profile, &doc("d1", "rust language"));
    assert!((affinity - 1.0).abs() < f32::EPSILON);
}

// ── rerank: boosting behaviour ──────────────────────────────────────────────

#[test]
fn test_rerank_boosts_profile_aligned_doc() {
    // d_off has higher base relevance but is off-profile; d_on is on-profile.
    // With a high personalization weight, d_on should overtake d_off.
    let reranker = PersonalizedReranker::new(
        PersonalizedConfig::new()
            .with_personalization_weight(0.9)
            .with_dim(128),
    );
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d_off", "python data science notebooks", 0.95, 0),
        result("d_on", "rust systems programming", 0.60, 1),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "d_on");
    assert_eq!(ranked[1].document.id.as_str(), "d_off");
}

#[test]
fn test_rerank_low_weight_keeps_strong_relevance() {
    // With a small personalization weight, the high-relevance off-profile doc
    // stays on top despite the affinity boost on the other doc.
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.1));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d_off", "python data science", 0.95, 0),
        result("d_on", "rust systems", 0.60, 1),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "d_off");
}

#[test]
fn test_rerank_weight_zero_is_identity_order() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.0));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d1", "python notebooks", 0.9, 0),
        result("d2", "rust systems", 0.7, 1),
        result("d3", "general topic", 0.5, 2),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "d1");
    assert_eq!(ranked[1].document.id.as_str(), "d2");
    assert_eq!(ranked[2].document.id.as_str(), "d3");
}

#[test]
fn test_rerank_weight_zero_preserves_scores() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.0));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d1", "rust systems", 0.7, 0),
        result("d2", "python notebooks", 0.9, 1),
    ];
    let ranked = reranker.rerank(&profile, &results);
    // Scores are unchanged from the originals (only re-sorted/renumbered).
    let d1 = ranked
        .iter()
        .find(|r| r.document.id.as_str() == "d1")
        .unwrap();
    let d2 = ranked
        .iter()
        .find(|r| r.document.id.as_str() == "d2")
        .unwrap();
    assert!((d1.score - 0.7).abs() < f32::EPSILON);
    assert!((d2.score - 0.9).abs() < f32::EPSILON);
}

#[test]
fn test_rerank_empty_profile_is_identity_order() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.9));
    let profile = UserProfile::new();
    let results = vec![
        result("d1", "alpha content", 0.8, 0),
        result("d2", "beta content", 0.6, 1),
        result("d3", "gamma content", 0.4, 2),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "d1");
    assert_eq!(ranked[1].document.id.as_str(), "d2");
    assert_eq!(ranked[2].document.id.as_str(), "d3");
}

#[test]
fn test_rerank_empty_profile_preserves_scores() {
    // Empty profile => affinity 0 for every doc; even with weight 1.0 the
    // blended score equals the (zeroed) relevance contribution, but ordering by
    // original relevance is preserved.
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.5));
    let profile = UserProfile::new();
    let results = vec![result("d1", "alpha", 0.8, 0), result("d2", "beta", 0.4, 1)];
    let ranked = reranker.rerank(&profile, &results);
    let d1 = ranked
        .iter()
        .find(|r| r.document.id.as_str() == "d1")
        .unwrap();
    // affinity == 0 => score == (1 - 0.5) * 0.8 == 0.4
    assert!((d1.score - 0.4).abs() < 1e-4);
}

#[test]
fn test_rerank_renumbers_ranks() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d1", "python notebooks", 0.9, 7),
        result("d2", "rust systems", 0.8, 3),
        result("d3", "java enterprise", 0.7, 11),
    ];
    let ranked = reranker.rerank(&profile, &results);
    for (i, r) in ranked.iter().enumerate() {
        assert_eq!(r.rank, i);
    }
}

#[test]
fn test_rerank_empty_input_yields_empty() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let ranked = reranker.rerank(&profile, &[]);
    assert!(ranked.is_empty());
}

#[test]
fn test_rerank_does_not_mutate_input() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.9));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d1", "python notebooks", 0.9, 0),
        result("d2", "rust systems", 0.6, 1),
    ];
    let _ = reranker.rerank(&profile, &results);
    // Original slice is untouched.
    assert!((results[0].score - 0.9).abs() < f32::EPSILON);
    assert_eq!(results[0].rank, 0);
    assert_eq!(results[1].document.id.as_str(), "d2");
}

#[test]
fn test_rerank_history_drives_order() {
    // No explicit interests; history alone should lift the matching doc.
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.9));
    let mut profile = UserProfile::new();
    profile.add_history("rust ownership borrow checker lifetimes");
    let results = vec![
        result("d_off", "tropical island vacations", 0.92, 0),
        result("d_on", "rust ownership and lifetimes", 0.55, 1),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "d_on");
}

#[test]
fn test_rerank_blended_scores_in_unit_range() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.6));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![result("d1", "rust", 1.0, 0), result("d2", "python", 0.0, 1)];
    let ranked = reranker.rerank(&profile, &results);
    for r in &ranked {
        assert!(r.score >= 0.0 && r.score <= 1.0);
    }
}

// ── Tie-breaking ────────────────────────────────────────────────────────────

#[test]
fn test_rerank_tie_break_by_id_ascending() {
    // Two docs with identical relevance and identical (zero) affinity tie on
    // score; the deterministic tie-break orders them by ascending id.
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.5));
    let profile = UserProfile::new().with_interest("astronomy", 1.0);
    let results = vec![
        result("zzz", "neutral text here", 0.5, 0),
        result("aaa", "neutral text here", 0.5, 1),
        result("mmm", "neutral text here", 0.5, 2),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked[0].document.id.as_str(), "aaa");
    assert_eq!(ranked[1].document.id.as_str(), "mmm");
    assert_eq!(ranked[2].document.id.as_str(), "zzz");
}

// ── rerank_checked / errors ─────────────────────────────────────────────────

#[test]
fn test_rerank_checked_empty_errors() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let err = reranker.rerank_checked(&profile, &[]).unwrap_err();
    assert_eq!(err, PersonalizedError::EmptyResults);
}

#[test]
fn test_rerank_checked_non_empty_ok() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![result("d1", "rust systems", 0.8, 0)];
    let ranked = reranker.rerank_checked(&profile, &results).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].rank, 0);
}

#[test]
fn test_rerank_checked_matches_rerank() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.7));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("d1", "python data", 0.9, 0),
        result("d2", "rust systems", 0.6, 1),
    ];
    let checked = reranker.rerank_checked(&profile, &results).unwrap();
    let plain = reranker.rerank(&profile, &results);
    assert_eq!(checked.len(), plain.len());
    for (a, b) in checked.iter().zip(plain.iter()) {
        assert_eq!(a.document.id.as_str(), b.document.id.as_str());
        assert!((a.score - b.score).abs() < f32::EPSILON);
    }
}

#[test]
fn test_error_display_message() {
    assert_eq!(
        PersonalizedError::EmptyResults.to_string(),
        "results must not be empty"
    );
}

// ── Reranker accessors & determinism ────────────────────────────────────────

#[test]
fn test_reranker_config_accessor() {
    let reranker = PersonalizedReranker::new(
        PersonalizedConfig::new()
            .with_personalization_weight(0.42)
            .with_dim(96),
    );
    assert!((reranker.config().personalization_weight - 0.42).abs() < f32::EPSILON);
    assert_eq!(reranker.config().dim, 96);
}

#[test]
fn test_affinity_is_deterministic() {
    let reranker = default_reranker();
    let mut profile = UserProfile::new().with_interest("rust", 1.0);
    profile.add_history("rust ownership and lifetimes");
    let document = doc("d1", "rust ownership lifetimes safety");
    let a = reranker.profile_affinity(&profile, &document);
    let b = reranker.profile_affinity(&profile, &document);
    assert!((a - b).abs() < f32::EPSILON);
}

#[test]
fn test_rerank_is_deterministic() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.6));
    let mut profile = UserProfile::new().with_interest("rust", 1.0);
    profile.add_history("rust ownership lifetimes");
    let results = vec![
        result("d1", "python notebooks", 0.9, 0),
        result("d2", "rust systems programming", 0.7, 1),
        result("d3", "rust ownership lifetimes safety", 0.6, 2),
    ];
    let first = reranker.rerank(&profile, &results);
    let second = reranker.rerank(&profile, &results);
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.document.id.as_str(), b.document.id.as_str());
        assert_eq!(a.rank, b.rank);
        assert!((a.score - b.score).abs() < f32::EPSILON);
    }
}

#[test]
fn test_rerank_single_result_ok() {
    let reranker = default_reranker();
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![result("only", "rust content", 0.5, 9)];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].rank, 0);
    assert_eq!(ranked[0].document.id.as_str(), "only");
}

#[test]
fn test_rerank_preserves_all_documents() {
    let reranker =
        PersonalizedReranker::new(PersonalizedConfig::new().with_personalization_weight(0.8));
    let profile = UserProfile::new().with_interest("rust", 1.0);
    let results = vec![
        result("a", "rust", 0.9, 0),
        result("b", "python", 0.8, 1),
        result("c", "java", 0.7, 2),
        result("d", "rust again", 0.6, 3),
    ];
    let ranked = reranker.rerank(&profile, &results);
    assert_eq!(ranked.len(), 4);
    let mut ids: Vec<&str> = ranked.iter().map(|r| r.document.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["a", "b", "c", "d"]);
}
