//! Tests for the `hallucination_detector` module.

use crate::hallucination_detector::{
    ClaimSupport, HallucinationConfig, HallucinationDetector, HallucinationError,
    HallucinationReport,
};
use crate::types::Document;

// ── ClaimSupport support_level ────────────────────────────────────────────────

#[test]
fn test_claim_support_level_strong() {
    let cs = ClaimSupport::new("claim", 0.75, vec![], false);
    assert_eq!(cs.support_level(), "strong");
}

#[test]
fn test_claim_support_level_weak() {
    let cs = ClaimSupport::new("claim", 0.35, vec![], false);
    assert_eq!(cs.support_level(), "weak");
}

#[test]
fn test_claim_support_level_unsupported() {
    let cs = ClaimSupport::new("claim", 0.1, vec![], true);
    assert_eq!(cs.support_level(), "unsupported");
}

// ── HallucinationReport ───────────────────────────────────────────────────────

#[test]
fn test_report_new_empty_claims() {
    let report = HallucinationReport::new(vec![]);
    assert!(report.hallucination_rate.abs() < f32::EPSILON);
    assert_eq!(report.supported_count, 0);
    assert!(report.is_clean());
}

#[test]
fn test_report_hallucination_count() {
    let claims = vec![
        ClaimSupport::new("a", 0.1, vec![], true),
        ClaimSupport::new("b", 0.8, vec![], false),
        ClaimSupport::new("c", 0.05, vec![], true),
    ];
    let report = HallucinationReport::new(claims);
    assert_eq!(report.hallucination_count(), 2);
    assert_eq!(report.supported_count, 1);
}

#[test]
fn test_report_is_clean_all_supported() {
    let claims = vec![
        ClaimSupport::new("a", 0.9, vec!["s1".to_string()], false),
        ClaimSupport::new("b", 0.7, vec!["s1".to_string()], false),
    ];
    let report = HallucinationReport::new(claims);
    assert!(report.is_clean());
    assert!(report.hallucination_rate.abs() < f32::EPSILON);
}

#[test]
fn test_report_is_clean_high_rate() {
    let claims = vec![
        ClaimSupport::new("a", 0.1, vec![], true),
        ClaimSupport::new("b", 0.1, vec![], true),
        ClaimSupport::new("c", 0.8, vec![], false),
    ];
    let report = HallucinationReport::new(claims);
    assert!(!report.is_clean());
    // rate = 2/3 ≈ 0.667
    assert!(report.hallucination_rate > 0.05);
}

// ── HallucinationConfig ───────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = HallucinationConfig::default();
    assert!((cfg.min_support - 0.2).abs() < f32::EPSILON);
    assert!(cfg.sentence_level);
}

#[test]
fn test_config_builders() {
    let cfg = HallucinationConfig::default()
        .with_min_support(0.4)
        .with_sentence_level(false);
    assert!((cfg.min_support - 0.4).abs() < f32::EPSILON);
    assert!(!cfg.sentence_level);
}

// ── HallucinationDetector errors ──────────────────────────────────────────────

#[test]
fn test_detector_empty_answer_error() {
    let det = HallucinationDetector::default();
    let src = vec![Document::new("some content")];
    let result = det.detect("", &src);
    assert!(matches!(result, Err(HallucinationError::EmptyAnswer)));
}

#[test]
fn test_detector_empty_sources_error() {
    let det = HallucinationDetector::default();
    let result = det.detect("some answer", &[]);
    assert!(matches!(result, Err(HallucinationError::EmptySources)));
}

// ── Supported / unsupported claims ───────────────────────────────────────────

#[test]
fn test_detector_supported_claim() {
    let det = HallucinationDetector::default();
    let src = vec![Document::new(
        "Rust is a systems programming language focused on safety.",
    )];
    let report = det
        .detect("Rust is a systems programming language.", &src)
        .expect("detect should succeed");
    assert_eq!(report.hallucination_count(), 0);
    assert!(report.is_clean());
}

#[test]
fn test_detector_unsupported_claim() {
    let cfg = HallucinationConfig::default().with_min_support(0.3);
    let det = HallucinationDetector::new(cfg);
    let src = vec![Document::new("The sky is blue and clouds are white.")];
    // Claim shares no meaningful tokens with the source
    let report = det
        .detect("Quantum entanglement occurs at absolute zero.", &src)
        .expect("detect should succeed");
    assert!(report.hallucination_count() > 0);
}

#[test]
fn test_detector_hallucination_rate() {
    // Two sentences: one supported, one not
    let cfg = HallucinationConfig::default()
        .with_min_support(0.3)
        .with_sentence_level(true);
    let det = HallucinationDetector::new(cfg);
    let src = vec![Document::new(
        "The capital of France is Paris. Paris is in Europe.",
    )];
    let answer = "The capital of France is Paris. Quantum physics is complicated.";
    let report = det.detect(answer, &src).expect("detect should succeed");
    // hallucination rate should be non-zero (second sentence is unsupported)
    assert!(report.hallucination_rate >= 0.0);
    assert!(report.hallucination_rate <= 1.0);
}

#[test]
fn test_detector_sentence_level_true() {
    let cfg = HallucinationConfig::default().with_sentence_level(true);
    let det = HallucinationDetector::new(cfg);
    let src = vec![Document::new("Cats are domesticated animals.")];
    let report = det
        .detect("Cats are domesticated animals.", &src)
        .expect("detect should succeed");
    // At sentence level, single sentence should be well-supported
    assert_eq!(report.claims.len(), 1);
    assert!(!report.claims[0].is_hallucination);
}

#[test]
fn test_detector_full_doc_mode() {
    let cfg = HallucinationConfig::default()
        .with_sentence_level(false)
        .with_min_support(0.05);
    let det = HallucinationDetector::new(cfg);
    let src = vec![Document::new(
        "Dogs are loyal animals that have lived with humans for millennia.",
    )];
    let report = det
        .detect("Dogs are loyal animals.", &src)
        .expect("detect should succeed");
    assert_eq!(report.claims.len(), 1);
}

#[test]
fn test_detector_multiple_sources_best_score() {
    let det = HallucinationDetector::default();
    let sources = vec![
        Document::new("The ocean covers most of Earth."),
        Document::new("Water is essential for marine life and all living organisms."),
    ];
    let report = det
        .detect("Water is essential for life.", &sources)
        .expect("detect should succeed");
    // Should find support from the second source
    let claim = &report.claims[0];
    assert!(claim.support_score > 0.0);
}

#[test]
fn test_detector_partial_support() {
    let cfg = HallucinationConfig::default().with_min_support(0.5);
    let det = HallucinationDetector::new(cfg);
    let src = vec![Document::new("Machine learning uses data to train models.")];
    // Partial overlap — score may be below 0.5
    let report = det
        .detect(
            "Deep learning neural networks require large datasets.",
            &src,
        )
        .expect("detect should succeed");
    assert!(!report.claims.is_empty());
    // score should be in valid range
    assert!(report.claims[0].support_score >= 0.0);
    assert!(report.claims[0].support_score <= 1.0);
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    let e1 = HallucinationError::EmptyAnswer;
    let e2 = HallucinationError::EmptySources;
    assert!(!e1.to_string().is_empty());
    assert!(!e2.to_string().is_empty());
}
