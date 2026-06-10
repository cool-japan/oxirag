//! Tests for the `answer_aggregator` module.

use super::aggregator::{AnswerAggregator, deduplicate, extract_sentences, jaccard_sentences};
use super::types::{
    AggregatedAnswer, AggregationConfig, AggregationError, AggregationStrategy, CandidateAnswer,
};

// ── Strategy tests ────────────────────────────────────────────────────────────

#[test]
fn test_strategy_as_str() {
    assert_eq!(AggregationStrategy::MajorityVote.as_str(), "majority_vote");
    assert_eq!(
        AggregationStrategy::WeightedFusion.as_str(),
        "weighted_fusion"
    );
    assert_eq!(AggregationStrategy::Extractive.as_str(), "extractive");
}

#[test]
fn test_strategy_default() {
    assert_eq!(
        AggregationStrategy::default(),
        AggregationStrategy::MajorityVote
    );
}

// ── CandidateAnswer tests ─────────────────────────────────────────────────────

#[test]
fn test_candidate_is_valid() {
    let c = CandidateAnswer::new("Rust is safe and fast.", 0.9);
    assert!(c.is_valid());
}

#[test]
fn test_candidate_invalid_empty() {
    let c = CandidateAnswer::new("", 0.5);
    assert!(!c.is_valid());
    let c2 = CandidateAnswer::new("   ", 0.5);
    assert!(!c2.is_valid());
}

#[test]
fn test_candidate_invalid_confidence_out_of_range() {
    let too_high = CandidateAnswer::new("Some answer.", 1.5);
    assert!(!too_high.is_valid());

    let negative = CandidateAnswer::new("Some answer.", -0.1);
    assert!(!negative.is_valid());
}

// ── AggregatedAnswer tests ────────────────────────────────────────────────────

#[test]
fn test_aggregated_answer_is_empty() {
    let empty = AggregatedAnswer {
        text: String::new(),
        consensus_score: 0.0,
        contributing_candidates: vec![],
    };
    assert!(empty.is_empty());

    let non_empty = AggregatedAnswer {
        text: "Some answer.".to_string(),
        consensus_score: 0.5,
        contributing_candidates: vec!["a".to_string()],
    };
    assert!(!non_empty.is_empty());
}

// ── Config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = AggregationConfig::default();
    assert_eq!(cfg.min_candidates, 2);
    assert!((cfg.dedup_threshold - 0.6).abs() < 1e-5);
}

#[test]
fn test_config_builders() {
    let cfg = AggregationConfig::default()
        .with_min_candidates(3)
        .with_dedup_threshold(0.8);
    assert_eq!(cfg.min_candidates, 3);
    assert!((cfg.dedup_threshold - 0.8).abs() < 1e-5);
}

// ── Aggregator error tests ────────────────────────────────────────────────────

#[test]
fn test_aggregator_insufficient_candidates() {
    let cfg = AggregationConfig::default().with_min_candidates(3);
    let agg = AnswerAggregator::new(cfg);

    let candidates = vec![
        CandidateAnswer::new("First answer.", 0.8),
        CandidateAnswer::new("Second answer.", 0.7),
    ];
    let err = agg
        .aggregate(&candidates, &AggregationStrategy::MajorityVote)
        .unwrap_err();
    assert!(matches!(err, AggregationError::InsufficientCandidates(3)));
}

#[test]
fn test_aggregator_all_empty_error() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("", 0.5),
        CandidateAnswer::new("  ", 0.5),
    ];
    let err = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap_err();
    assert!(matches!(err, AggregationError::AllCandidatesEmpty));
}

// ── Aggregator strategy tests ─────────────────────────────────────────────────

#[test]
fn test_aggregator_majority_vote() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("Rust is a systems language. It focuses on safety.", 0.9),
        CandidateAnswer::new("Rust is a systems language. It has great performance.", 0.8),
        CandidateAnswer::new(
            "Rust is a systems language. Memory management is explicit.",
            0.7,
        ),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::MajorityVote)
        .unwrap();
    assert!(!result.text.is_empty());
    // Consensus score should be average of confidences
    assert!(result.consensus_score > 0.0 && result.consensus_score <= 1.0);
}

#[test]
fn test_aggregator_weighted_fusion() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new(
            "Rust provides memory safety without garbage collection.",
            0.95,
        ),
        CandidateAnswer::new("Rust ownership model prevents use-after-free errors.", 0.7),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::WeightedFusion)
        .unwrap();
    assert!(!result.text.is_empty());
}

#[test]
fn test_aggregator_extractive() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("Rust ensures memory safety. Ownership prevents leaks.", 0.8),
        CandidateAnswer::new(
            "Rust has zero-cost abstractions. Performance is excellent.",
            0.7,
        ),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap();
    assert!(!result.text.is_empty());
}

#[test]
fn test_aggregator_dedup_threshold() {
    // With very low dedup threshold, nearly all sentences are duplicates of each other
    let cfg = AggregationConfig::default().with_dedup_threshold(0.01);
    let agg = AnswerAggregator::new(cfg);

    let candidates = vec![
        CandidateAnswer::new("Rust programming language.", 0.9),
        CandidateAnswer::new("Rust programming language features.", 0.8),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap();
    // With low threshold, duplicates aggressively deduplicated → fewer sentences
    assert!(!result.text.is_empty());
}

#[test]
fn test_aggregator_consensus_score_range() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("First answer about rust.", 0.6),
        CandidateAnswer::new("Second answer about rust programming.", 0.4),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::MajorityVote)
        .unwrap();
    assert!(
        result.consensus_score >= 0.0 && result.consensus_score <= 1.0,
        "consensus_score out of range: {}",
        result.consensus_score
    );
    // Should be mean of 0.6 and 0.4 = 0.5
    assert!((result.consensus_score - 0.5).abs() < 1e-4);
}

#[test]
fn test_aggregator_contributing_candidates_populated() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("First candidate answer.", 0.8),
        CandidateAnswer::new("Second candidate answer.", 0.7),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap();
    assert_eq!(result.contributing_candidates.len(), 2);
}

#[test]
fn test_aggregator_single_sentence_candidates() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("Rust is safe", 0.9),
        CandidateAnswer::new("Rust is performant", 0.8),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap();
    assert!(!result.text.is_empty());
}

#[test]
fn test_aggregator_identical_candidates() {
    let agg = AnswerAggregator::default();
    let candidates = vec![
        CandidateAnswer::new("Rust is a safe systems programming language.", 0.9),
        CandidateAnswer::new("Rust is a safe systems programming language.", 0.9),
    ];
    let result = agg
        .aggregate(&candidates, &AggregationStrategy::Extractive)
        .unwrap();
    // Deduplication should collapse to one unique sentence
    assert!(!result.text.is_empty());
}

// ── Helper unit tests ─────────────────────────────────────────────────────────

#[test]
fn test_jaccard_helper_identical() {
    let score = jaccard_sentences("rust programming", "rust programming");
    assert!((score - 1.0).abs() < 1e-5);
}

#[test]
fn test_jaccard_helper_no_overlap() {
    let score = jaccard_sentences("apple banana", "cat dog fish");
    assert!(score < 1e-5);
}

#[test]
fn test_deduplicate_removes_near_duplicates() {
    let sentences = vec![
        "Rust is a programming language.".to_string(),
        "Rust is a programming language!".to_string(), // near-duplicate
        "Python is a scripting language.".to_string(),
    ];
    let deduped = deduplicate(&sentences, 0.5);
    // The near-duplicate should be removed
    assert!(deduped.len() <= 2, "Expected ≤2, got {}", deduped.len());
}

#[test]
fn test_extract_sentences_basic() {
    let text = "Rust is safe. It is fast. Memory is managed.";
    let parts = extract_sentences(text);
    assert!(
        parts.len() >= 2,
        "Expected multiple sentences, got {}",
        parts.len()
    );
    for p in &parts {
        assert!(!p.is_empty());
    }
}

#[test]
fn test_error_display() {
    let e1 = AggregationError::InsufficientCandidates(3);
    assert!(e1.to_string().contains('3'));

    let e2 = AggregationError::AllCandidatesEmpty;
    assert!(e2.to_string().to_lowercase().contains("empty"));
}
