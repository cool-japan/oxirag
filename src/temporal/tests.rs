//! Tests for the `temporal` module.

use crate::types::{Document, DocumentId, SearchResult};

use super::reranker::TemporalReranker;
use super::types::{DecayFunction, TemporalConfig, TemporalError};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn make_result_with_ts(id: &str, score: f32, ts: &str) -> SearchResult {
    SearchResult {
        document: Document::new("content")
            .with_id(DocumentId::from_string(id))
            .with_metadata("created_at", ts),
        score,
        rank: 0,
    }
}

/// Approximate Unix seconds for 2024-01-01 00:00:00 UTC.
const TS_2024_01_01: i64 = 1_704_067_200;
/// Approximately 90 days after `TS_2024_01_01`.
const TS_NOW_90D: i64 = TS_2024_01_01 + 90 * 86_400;

// ── DecayFunction tests ───────────────────────────────────────────────────────

#[test]
fn test_exponential_zero_age() {
    let f = DecayFunction::Exponential {
        half_life_days: 30.0,
    };
    let d = f.apply(0.0);
    assert!((d - 1.0).abs() < 1e-6, "zero age → decay 1.0");
}

#[test]
fn test_exponential_at_half_life() {
    let f = DecayFunction::Exponential {
        half_life_days: 30.0,
    };
    let d = f.apply(30.0);
    assert!((d - 0.5).abs() < 1e-4, "one half-life → decay ~0.5: {d}");
}

#[test]
fn test_exponential_future_clamped() {
    let f = DecayFunction::Exponential {
        half_life_days: 30.0,
    };
    let d = f.apply(-10.0); // future timestamp
    assert!(
        (d - 1.0).abs() < 1e-6,
        "negative age → clamped to 0 → decay 1.0"
    );
}

#[test]
fn test_linear_zero_age() {
    let f = DecayFunction::Linear { max_age_days: 60.0 };
    assert!((f.apply(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_linear_at_max_age() {
    let f = DecayFunction::Linear { max_age_days: 60.0 };
    assert!((f.apply(60.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_linear_clamped_to_zero() {
    let f = DecayFunction::Linear { max_age_days: 60.0 };
    assert!((f.apply(120.0) - 0.0).abs() < 1e-10, "beyond max → 0");
}

#[test]
fn test_gaussian_zero_age() {
    let f = DecayFunction::Gaussian { sigma_days: 30.0 };
    assert!((f.apply(0.0) - 1.0).abs() < 1e-6);
}

#[test]
fn test_gaussian_one_sigma() {
    let f = DecayFunction::Gaussian { sigma_days: 30.0 };
    let d = f.apply(30.0);
    // e^(-0.5) ≈ 0.6065
    assert!((d - 0.6065).abs() < 0.01, "one sigma → ~0.6065: {d}");
}

#[test]
fn test_none_decay() {
    let f = DecayFunction::None;
    assert!((f.apply(0.0) - 1.0).abs() < 1e-10);
    assert!((f.apply(999.0) - 1.0).abs() < 1e-10);
}

#[test]
fn test_decay_function_default() {
    let f = DecayFunction::default();
    assert!(matches!(
        f,
        DecayFunction::Exponential { half_life_days: _ }
    ));
}

#[test]
fn test_decay_function_labels() {
    assert_eq!(
        DecayFunction::Exponential {
            half_life_days: 30.0
        }
        .label(),
        "exponential"
    );
    assert_eq!(
        DecayFunction::Linear { max_age_days: 60.0 }.label(),
        "linear"
    );
    assert_eq!(
        DecayFunction::Gaussian { sigma_days: 30.0 }.label(),
        "gaussian"
    );
    assert_eq!(DecayFunction::None.label(), "none");
}

// ── TemporalConfig tests ──────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = TemporalConfig::default();
    assert!((cfg.weight - 0.5).abs() < 1e-5);
    assert!(cfg.use_updated);
    assert!(matches!(cfg.decay, DecayFunction::Exponential { .. }));
}

#[test]
fn test_config_builders() {
    let cfg = TemporalConfig::default()
        .with_decay(DecayFunction::None)
        .with_weight(0.8)
        .with_use_updated(false);
    assert!((cfg.weight - 0.8).abs() < 1e-5);
    assert!(!cfg.use_updated);
    assert_eq!(cfg.decay, DecayFunction::None);
}

// ── TemporalReranker tests ────────────────────────────────────────────────────

#[test]
fn test_reranker_empty_results_error() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default();
    let err = r.rerank(&[], 0, &cfg).expect_err("should fail");
    assert!(matches!(err, TemporalError::EmptyResults));
}

#[test]
fn test_reranker_none_decay_preserves_order() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default().with_decay(DecayFunction::None);
    let results = vec![
        make_result("d1", "content", 0.9),
        make_result("d2", "content", 0.7),
        make_result("d3", "content", 0.5),
    ];
    let (reranked, _) = r.rerank(&results, TS_NOW_90D, &cfg).expect("ok");
    // With None decay, order should be preserved by original score
    assert!(reranked[0].score >= reranked[1].score);
    assert!(reranked[1].score >= reranked[2].score);
}

#[test]
fn test_reranker_old_document_demoted() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default()
        .with_decay(DecayFunction::Exponential {
            half_life_days: 30.0,
        })
        .with_weight(1.0); // pure decay

    // d1: brand new (no timestamp → age 0 → decay 1.0)
    // d2: 90 days old → decay ≈ 0.125
    let d1 = make_result("d1", "new doc", 0.8);
    let d2 = make_result_with_ts("d2", 0.8, "2024-01-01");
    let (reranked, scores) = r.rerank(&[d1, d2], TS_NOW_90D, &cfg).expect("ok");

    assert_eq!(
        reranked[0].document.id.as_str(),
        "d1",
        "newer doc should rank first"
    );
    assert!(scores[0].decayed >= scores[1].decayed);
}

#[test]
fn test_reranker_future_timestamp_treated_as_new() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default();
    // Timestamp in the future → age = 0 → decay = 1.0
    let d1 = make_result_with_ts("d1", 0.8, "2099-12-31");
    let (reranked, scores) = r.rerank(&[d1], 0, &cfg).expect("ok"); // now = epoch 0, doc is future
    // age = max(0, 0 - future) = 0 → decay 1.0
    assert!(scores[0].age_days == 0.0, "future timestamp → age 0");
    assert!(reranked[0].score <= 1.0);
}

#[test]
fn test_reranker_iso_date_parsed() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default().with_weight(1.0);
    let d = make_result_with_ts("d1", 1.0, "2024-01-01");
    let (_, scores) = r.rerank(&[d], TS_NOW_90D, &cfg).expect("ok");
    assert!(
        scores[0].age_days > 0.0,
        "date should parse to non-zero age"
    );
}

#[test]
fn test_reranker_no_timestamp_age_zero() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default();
    let d = make_result("d1", "no timestamp", 0.8);
    let (_, scores) = r.rerank(&[d], TS_NOW_90D, &cfg).expect("ok");
    assert!(
        (scores[0].age_days - 0.0).abs() < 1e-10,
        "missing timestamp → age 0"
    );
}

#[test]
fn test_reranker_multiple_results_sorted() {
    let r = TemporalReranker::new();
    let cfg = TemporalConfig::default().with_decay(DecayFunction::None);
    let results = vec![
        make_result("d3", "c", 0.3),
        make_result("d1", "a", 0.9),
        make_result("d2", "b", 0.6),
    ];
    let (reranked, _) = r.rerank(&results, 0, &cfg).expect("ok");
    assert_eq!(reranked[0].document.id.as_str(), "d1");
    assert_eq!(reranked[1].document.id.as_str(), "d2");
    assert_eq!(reranked[2].document.id.as_str(), "d3");
}
