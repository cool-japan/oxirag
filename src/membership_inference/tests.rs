//! Tests for the `membership_inference` module.

use super::auditor::{CanaryAuditor, MembershipDefense, MockRagProbe, RagProbe};
use super::types::{
    CanaryKind, MembershipInferenceConfig, MembershipInferenceError, ProbeResult, RiskLevel,
};

// ── AUC: hand-computable rank data ────────────────────────────────────────────

#[test]
fn test_auc_hand_computed_no_ties() {
    // member = [1, 2, 3], non_member = [0, 0.5, 4].
    // Pooled ascending: 0(NM), 0.5(NM), 1(M), 2(M), 3(M), 4(NM) -> ranks 1..6.
    // rank_sum_member = 3 + 4 + 5 = 12; n1 = n2 = 3.
    // AUC = (12 - 3*4/2) / (3*3) = (12 - 6) / 9 = 6/9 = 0.6667.
    //
    // Cross-checked by direct pair counting (ties count as 0.5): of the 9
    // (member, non_member) pairs, member "wins" 6 of them outright and never
    // ties, so AUC = 6/9 as well.
    let member = [1.0_f32, 2.0, 3.0];
    let non_member = [0.0_f32, 0.5, 4.0];

    let auc = CanaryAuditor::compute_auc(&member, &non_member).expect("valid signal groups");
    assert!(
        (auc - (2.0 / 3.0)).abs() < 1e-4,
        "expected AUC ~0.6667, got {auc}"
    );
}

#[test]
fn test_auc_hand_computed_with_ties() {
    // member = [1, 1], non_member = [1, 2].
    // Pooled ascending: 1(M), 1(M), 1(NM), 2(NM) -- three-way tie at value 1
    // occupies ranks 1..3, average rank (1+2+3)/3 = 2; rank 4 for value 2.
    // rank_sum_member = 2 + 2 = 4; n1 = n2 = 2.
    // AUC = (4 - 2*3/2) / (2*2) = (4 - 3) / 4 = 0.25.
    //
    // Cross-checked by direct pair counting (ties count as 0.5): pairs are
    // (1,1)=0.5, (1,2)=0, (1,1)=0.5, (1,2)=0 -> mean = 1.0/4 = 0.25.
    let member = [1.0_f32, 1.0];
    let non_member = [1.0_f32, 2.0];

    let auc = CanaryAuditor::compute_auc(&member, &non_member).expect("valid signal groups");
    assert!((auc - 0.25).abs() < 1e-4, "expected AUC 0.25, got {auc}");
}

#[test]
fn test_auc_perfect_separation_is_one() {
    let member = [10.0_f32, 20.0, 30.0, 40.0];
    let non_member = [1.0_f32, 2.0, 3.0, 4.0];
    let auc = CanaryAuditor::compute_auc(&member, &non_member).expect("valid signal groups");
    assert!((auc - 1.0).abs() < 1e-6, "expected AUC 1.0, got {auc}");
}

#[test]
fn test_auc_perfect_inverse_separation_is_zero() {
    let member = [1.0_f32, 2.0, 3.0, 4.0];
    let non_member = [10.0_f32, 20.0, 30.0, 40.0];
    let auc = CanaryAuditor::compute_auc(&member, &non_member).expect("valid signal groups");
    assert!((auc - 0.0).abs() < 1e-6, "expected AUC 0.0, got {auc}");
}

#[test]
fn test_auc_all_tied_signals_is_one_half() {
    // Every value identical across both groups: no discriminative power
    // whatsoever, and the rank-sum algebra guarantees this collapses to
    // exactly 0.5 regardless of group sizes.
    let member = [0.42_f32; 5];
    let non_member = [0.42_f32; 7];
    let auc = CanaryAuditor::compute_auc(&member, &non_member).expect("valid signal groups");
    assert!((auc - 0.5).abs() < 1e-6, "expected AUC 0.5, got {auc}");
}

#[test]
fn test_auc_single_sample_each_side() {
    let auc = CanaryAuditor::compute_auc(&[1.0], &[0.0]).expect("valid signal groups");
    assert!((auc - 1.0).abs() < 1e-6);
    let auc_tied = CanaryAuditor::compute_auc(&[0.5], &[0.5]).expect("valid signal groups");
    assert!((auc_tied - 0.5).abs() < 1e-6);
}

// ── AUC: edge cases ────────────────────────────────────────────────────────────

#[test]
fn test_auc_empty_member_group_errors() {
    let err = CanaryAuditor::compute_auc(&[], &[1.0, 2.0]).expect_err("empty member group");
    assert_eq!(err, MembershipInferenceError::EmptyCanarySet);
}

#[test]
fn test_auc_empty_non_member_group_errors() {
    let err = CanaryAuditor::compute_auc(&[1.0, 2.0], &[]).expect_err("empty non-member group");
    assert_eq!(err, MembershipInferenceError::EmptyCanarySet);
}

#[test]
fn test_auc_both_groups_empty_errors() {
    let err = CanaryAuditor::compute_auc(&[], &[]).expect_err("both groups empty");
    assert_eq!(err, MembershipInferenceError::EmptyCanarySet);
}

#[test]
fn test_auc_nan_signal_errors() {
    let err = CanaryAuditor::compute_auc(&[f32::NAN, 1.0], &[0.0, 0.2]).expect_err("NaN in member");
    assert_eq!(err, MembershipInferenceError::NonFiniteSignal);
}

#[test]
fn test_auc_infinite_signal_errors() {
    let err = CanaryAuditor::compute_auc(&[0.5, 0.6], &[f32::INFINITY, 0.1])
        .expect_err("infinite in non-member");
    assert_eq!(err, MembershipInferenceError::NonFiniteSignal);
}

// ── canary generation ────────────────────────────────────────────────────────

#[test]
fn test_generate_pairs_zero_errors() {
    let auditor = CanaryAuditor::new(MembershipInferenceConfig::new());
    let err = auditor.generate_pairs(0).expect_err("zero pairs requested");
    assert!(matches!(err, MembershipInferenceError::InvalidConfig(_)));
}

#[test]
fn test_generate_pairs_is_deterministic() {
    let config = MembershipInferenceConfig::new().with_seed(1234);
    let auditor_a = CanaryAuditor::new(config.clone());
    let auditor_b = CanaryAuditor::new(config);

    let pairs_a = auditor_a.generate_pairs(10).expect("valid config");
    let pairs_b = auditor_b.generate_pairs(10).expect("valid config");

    assert_eq!(pairs_a, pairs_b);
}

#[test]
fn test_generate_pairs_different_seeds_differ() {
    let auditor_a = CanaryAuditor::new(MembershipInferenceConfig::new().with_seed(1));
    let auditor_b = CanaryAuditor::new(MembershipInferenceConfig::new().with_seed(2));

    let pairs_a = auditor_a.generate_pairs(4).expect("valid config");
    let pairs_b = auditor_b.generate_pairs(4).expect("valid config");

    assert_ne!(pairs_a, pairs_b);
}

#[test]
fn test_generate_pairs_member_and_non_member_kinds() {
    let auditor = CanaryAuditor::new(MembershipInferenceConfig::new());
    let pairs = auditor.generate_pairs(5).expect("valid config");
    assert_eq!(pairs.len(), 5);
    for (index, pair) in pairs.iter().enumerate() {
        assert_eq!(pair.member.kind, CanaryKind::Member);
        assert!(pair.member.is_member());
        assert_eq!(pair.non_member.kind, CanaryKind::NonMember);
        assert!(!pair.non_member.is_member());
        // Distinct ids, distinct markers (each canary is generated
        // independently, so a marker collision would indicate a broken
        // generator, not intended behavior).
        assert_ne!(pair.member.id, pair.non_member.id);
        assert_ne!(pair.member.marker, pair.non_member.marker);
        // The canary's own marker must actually appear in its own text, and
        // the shadow query must be a genuine prefix of the marker.
        assert!(pair.member.text.contains(&pair.member.marker));
        assert!(pair.member.marker.starts_with(&pair.member.shadow_query));
        assert!(pair.non_member.text.contains(&pair.non_member.marker));
        assert!(
            pair.non_member
                .marker
                .starts_with(&pair.non_member.shadow_query)
        );
        let _ = index;
    }
}

#[test]
fn test_generate_pairs_invalid_config_propagates() {
    let bad_config = MembershipInferenceConfig::new().with_marker_token_count(0);
    let auditor = CanaryAuditor::new(bad_config);
    let err = auditor
        .generate_pairs(3)
        .expect_err("marker_token_count = 0 is invalid");
    assert!(matches!(err, MembershipInferenceError::InvalidConfig(_)));
}

// ── config validation ────────────────────────────────────────────────────────

#[test]
fn test_config_default_is_valid() {
    MembershipInferenceConfig::new()
        .validate()
        .expect("default config must validate");
}

#[test]
fn test_config_probe_token_count_exceeds_marker_token_count() {
    let config = MembershipInferenceConfig::new()
        .with_marker_token_count(2)
        .with_probe_token_count(3);
    assert!(matches!(
        config.validate(),
        Err(MembershipInferenceError::InvalidConfig(_))
    ));
}

#[test]
fn test_config_negative_weight_is_invalid() {
    let config = MembershipInferenceConfig::new().with_overlap_weight(-0.1);
    assert!(matches!(
        config.validate(),
        Err(MembershipInferenceError::InvalidConfig(_))
    ));
}

#[test]
fn test_config_low_risk_above_high_risk_is_invalid() {
    let config = MembershipInferenceConfig::new()
        .with_low_risk_auc(0.9)
        .with_high_risk_auc(0.6);
    assert!(matches!(
        config.validate(),
        Err(MembershipInferenceError::InvalidConfig(_))
    ));
}

#[test]
fn test_config_zero_confidence_bucket_width_is_invalid() {
    let config = MembershipInferenceConfig::new().with_confidence_bucket_width(0.0);
    assert!(matches!(
        config.validate(),
        Err(MembershipInferenceError::InvalidConfig(_))
    ));
}

// ── RiskLevel classification ───────────────────────────────────────────────────

#[test]
fn test_risk_level_bands() {
    let config = MembershipInferenceConfig::new(); // low=0.6, high=0.75
    assert_eq!(RiskLevel::classify(0.5, &config), RiskLevel::Low);
    assert_eq!(RiskLevel::classify(0.55, &config), RiskLevel::Low);
    assert_eq!(RiskLevel::classify(0.65, &config), RiskLevel::Medium);
    assert_eq!(RiskLevel::classify(0.8, &config), RiskLevel::High);
}

#[test]
fn test_risk_level_is_symmetric_around_one_half() {
    let config = MembershipInferenceConfig::new();
    // AUC = 0.2 has the same separability as AUC = 0.8 (effective 0.8), just
    // pointing the other way -- an attacker simply inverts their rule.
    assert_eq!(RiskLevel::classify(0.2, &config), RiskLevel::High);
    assert_eq!(RiskLevel::classify(0.35, &config), RiskLevel::Medium);
    assert_eq!(RiskLevel::classify(0.45, &config), RiskLevel::Low);
}

// ── MockRagProbe ──────────────────────────────────────────────────────────────

#[test]
fn test_mock_probe_leaky_hit_echoes_document_and_boosts_confidence() {
    let probe = MockRagProbe::leaky(vec!["prefix marker-abc suffix".to_string()], 0.1, 0.5);
    let result = probe.probe("marker-abc");
    assert_eq!(
        result.generated.as_deref(),
        Some("prefix marker-abc suffix")
    );
    assert_eq!(
        result.retrieved,
        vec!["prefix marker-abc suffix".to_string()]
    );
    assert!((result.confidence.expect("leaky hit has confidence") - 0.6).abs() < 1e-6);
}

#[test]
fn test_mock_probe_leaky_miss_is_generic() {
    let probe = MockRagProbe::leaky(vec!["totally unrelated text".to_string()], 0.1, 0.5);
    let result = probe.probe("marker-abc");
    assert!(result.retrieved.is_empty());
    assert_eq!(
        result.generated.as_deref(),
        Some("No relevant information was found.")
    );
    assert!((result.confidence.expect("baseline confidence") - 0.1).abs() < 1e-6);
}

#[test]
fn test_mock_probe_non_leaky_hit_masks_content() {
    let probe = MockRagProbe::non_leaky(vec!["prefix marker-abc suffix".to_string()], Some(0.3));
    let result = probe.probe("marker-abc");
    assert!(!result.retrieved.iter().any(|t| t.contains("marker-abc")));
    assert!(
        !result
            .generated
            .as_deref()
            .unwrap_or_default()
            .contains("marker-abc")
    );
    assert!((result.confidence.expect("baseline confidence") - 0.3).abs() < 1e-6);
}

// ── ProbeResult ──────────────────────────────────────────────────────────────

#[test]
fn test_probe_result_combined_text_joins_retrieved_then_generated() {
    let result = ProbeResult {
        retrieved: vec!["passage one".to_string(), "passage two".to_string()],
        generated: Some("final answer".to_string()),
        confidence: None,
    };
    assert_eq!(
        result.combined_text(),
        "passage one passage two final answer"
    );
}

#[test]
fn test_probe_result_combined_text_handles_empty_result() {
    let result = ProbeResult::default();
    assert_eq!(result.combined_text(), "");
}

// ── end-to-end audit: leaky scenario ──────────────────────────────────────────

fn small_config() -> MembershipInferenceConfig {
    MembershipInferenceConfig::new()
        .with_seed(777)
        .with_marker_token_count(4)
        .with_probe_token_count(2)
        .with_filler_sentence_count(2)
        .with_overlap_ngram(2)
        .with_redact_min_ngram(2)
}

#[test]
fn test_leaky_probe_produces_high_auc_and_high_risk() {
    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(8).expect("valid config");

    // Only member canaries are inserted into the audited corpus.
    let corpus: Vec<String> = pairs.iter().map(|p| p.member.text.clone()).collect();
    let probe = MockRagProbe::leaky(corpus, 0.1, 0.6);

    let report = auditor
        .audit_pairs(&probe, &pairs)
        .expect("non-empty pairs");
    assert_eq!(report.n_pairs(), 8);
    assert!(
        report.auc > 0.9,
        "expected a clearly leaky AUC, got {}",
        report.auc
    );
    assert_eq!(report.risk, RiskLevel::High);
    assert!(report.is_leak_detected());
    // Every individual pair should also show the member signal strictly
    // ahead of its paired control, given the clean deterministic separation.
    assert!(report.pairs.iter().all(|p| p.member_signal_higher));
}

#[test]
fn test_non_leaky_probe_produces_auc_near_one_half() {
    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(8).expect("valid config");

    let corpus: Vec<String> = pairs.iter().map(|p| p.member.text.clone()).collect();
    // simulate_leak = false: the mock's response never depends on which
    // document (if any) matched, so no signal channel can correlate with
    // membership.
    let probe = MockRagProbe::non_leaky(corpus, Some(0.2));

    let report = auditor
        .audit_pairs(&probe, &pairs)
        .expect("non-empty pairs");
    assert!(
        (report.auc - 0.5).abs() < 0.05,
        "expected AUC ~0.5 for a non-leaky probe, got {}",
        report.auc
    );
    assert_eq!(report.risk, RiskLevel::Low);
    assert!(!report.is_leak_detected());
}

#[test]
fn test_non_leaky_probe_with_uncorrelated_confidence_jitter_stays_near_one_half() {
    // A probe whose confidence has small, membership-*uncorrelated* jitter
    // (deterministic, derived from the canary id, not from corpus
    // membership) should still be statistically indistinguishable across
    // the two groups.
    struct JitteryProbe;
    impl RagProbe for JitteryProbe {
        fn probe(&self, query: &str) -> ProbeResult {
            #[allow(clippy::cast_precision_loss)]
            let jitter = (query.len() % 5) as f32 * 0.02;
            ProbeResult {
                retrieved: Vec::new(),
                generated: Some("no comment".to_string()),
                confidence: Some(0.3 + jitter),
            }
        }
    }

    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(12).expect("valid config");
    let report = auditor
        .audit_pairs(&JitteryProbe, &pairs)
        .expect("non-empty pairs");

    assert!(
        (report.auc - 0.5).abs() < 0.2,
        "expected AUC roughly near 0.5, got {}",
        report.auc
    );
}

#[test]
fn test_audit_pairs_empty_slice_errors() {
    let auditor = CanaryAuditor::new(small_config());
    let probe = MockRagProbe::default();
    let err = auditor
        .audit_pairs(&probe, &[])
        .expect_err("empty pairs slice");
    assert_eq!(err, MembershipInferenceError::EmptyCanarySet);
}

// ── defense: reduces AUC on a leaky probe ─────────────────────────────────────

#[test]
fn test_defense_reduces_auc_on_leaky_verbatim_regurgitation_probe() {
    // Isolate the verbatim-overlap channel: the mock never exposes a
    // confidence score, so the *only* leakage channel is the text the probe
    // echoes back.
    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(10).expect("valid config");

    let corpus: Vec<String> = pairs.iter().map(|p| p.member.text.clone()).collect();
    let probe = MockRagProbe {
        corpus,
        simulate_leak: true,
        baseline_confidence: None,
        leak_confidence_boost: 0.0,
    };

    let baseline = auditor
        .audit_pairs(&probe, &pairs)
        .expect("non-empty pairs");
    assert!(
        baseline.auc > 0.95,
        "expected baseline AUC to show a strong leak, got {}",
        baseline.auc
    );

    let defense = MembershipDefense::new(small_config());
    let defended = auditor
        .audit_pairs_with_defense(&probe, &pairs, &defense)
        .expect("non-empty pairs");

    assert!(
        defended.auc < baseline.auc,
        "defended AUC ({}) should be lower than baseline AUC ({})",
        defended.auc,
        baseline.auc
    );
    // Redaction removes every marker span from the echoed text, so the
    // overlap channel collapses to zero for both groups: with confidence
    // absent entirely, every signal (member and non-member alike) becomes
    // exactly 0.0, which the rank-sum algebra guarantees is AUC 0.5.
    assert!(
        (defended.auc - 0.5).abs() < 1e-4,
        "expected the defended AUC to fall back to ~0.5, got {}",
        defended.auc
    );
    assert_eq!(defended.risk, RiskLevel::Low);
}

#[test]
fn test_defense_confidence_quantization_creates_ties_for_small_gaps() {
    // A probe that never leaks any text but exposes a small,
    // membership-correlated confidence gap (0.55 vs 0.45) -- the kind of
    // "similarity score" leak channel that verbatim-overlap redaction alone
    // cannot touch.
    struct ConfidenceOnlyProbe {
        member_marker_prefixes: Vec<String>,
    }
    impl RagProbe for ConfidenceOnlyProbe {
        fn probe(&self, query: &str) -> ProbeResult {
            let is_member_query = self
                .member_marker_prefixes
                .iter()
                .any(|prefix| prefix == query);
            ProbeResult {
                retrieved: Vec::new(),
                generated: Some("no comment".to_string()),
                confidence: Some(if is_member_query { 0.55 } else { 0.45 }),
            }
        }
    }

    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(6).expect("valid config");
    let probe = ConfidenceOnlyProbe {
        member_marker_prefixes: pairs
            .iter()
            .map(|p| p.member.shadow_query.clone())
            .collect(),
    };

    let baseline = auditor
        .audit_pairs(&probe, &pairs)
        .expect("non-empty pairs");
    assert!(
        baseline.auc > 0.9,
        "expected the confidence gap to fully separate the groups, got {}",
        baseline.auc
    );

    // A defense whose bucket width comfortably straddles the 0.1 gap between
    // 0.45 and 0.55 collapses both onto the same quantized value.
    let defense = MembershipDefense::new(small_config().with_confidence_bucket_width(0.5));
    let defended = auditor
        .audit_pairs_with_defense(&probe, &pairs, &defense)
        .expect("non-empty pairs");

    assert!(
        (defended.auc - 0.5).abs() < 1e-4,
        "expected quantization to collapse the confidence gap to AUC ~0.5, got {}",
        defended.auc
    );
    assert!(defended.auc < baseline.auc);
}

// ── MembershipDefense::mitigate ────────────────────────────────────────────────

#[test]
fn test_mitigate_redacts_marker_spans_from_generated_and_retrieved_text() {
    let defense = MembershipDefense::new(MembershipInferenceConfig::new().with_redact_min_ngram(2));
    let result = ProbeResult {
        retrieved: vec!["Some prefix zc001 zc002 zc003 trailing text".to_string()],
        generated: Some("Answer contains zc001 zc002 zc003 verbatim.".to_string()),
        confidence: Some(0.77),
    };
    let markers = ["zc001 zc002 zc003"];
    let mitigated = defense.mitigate(&result, &markers);

    assert!(!mitigated.retrieved[0].contains("zc001 zc002 zc003"));
    assert!(mitigated.retrieved[0].contains("[REDACTED]"));
    assert!(
        !mitigated
            .generated
            .as_deref()
            .unwrap_or_default()
            .contains("zc001 zc002 zc003")
    );
}

#[test]
fn test_mitigate_with_no_known_markers_only_quantizes_confidence() {
    let defense = MembershipDefense::new(MembershipInferenceConfig::new());
    let result = ProbeResult {
        retrieved: vec!["untouched passage text".to_string()],
        generated: Some("untouched generated text".to_string()),
        confidence: Some(0.83),
    };
    let mitigated = defense.mitigate(&result, &[]);

    assert_eq!(mitigated.retrieved, result.retrieved);
    assert_eq!(mitigated.generated, result.generated);
    // 0.83 quantized to the nearest 0.2 bucket: 0.83 / 0.2 = 4.15 -> round 4
    // -> 0.8.
    assert!(
        (mitigated.confidence.expect("confidence present") - 0.8).abs() < 1e-5,
        "got {:?}",
        mitigated.confidence
    );
}

#[test]
fn test_mitigate_preserves_none_confidence() {
    let defense = MembershipDefense::new(MembershipInferenceConfig::new());
    let result = ProbeResult {
        retrieved: Vec::new(),
        generated: None,
        confidence: None,
    };
    let mitigated = defense.mitigate(&result, &["zc001 zc002"]);
    assert!(mitigated.confidence.is_none());
    assert!(mitigated.generated.is_none());
    assert!(mitigated.retrieved.is_empty());
}

#[test]
fn test_mitigate_short_marker_redacted_as_single_unit() {
    // A marker shorter than redact_min_ngram is treated as one phrase.
    let defense = MembershipDefense::new(MembershipInferenceConfig::new().with_redact_min_ngram(5));
    let result = ProbeResult {
        retrieved: vec!["leading zc999 trailing".to_string()],
        generated: None,
        confidence: None,
    };
    let mitigated = defense.mitigate(&result, &["zc999"]);
    assert!(mitigated.retrieved[0].contains("[REDACTED]"));
    assert!(!mitigated.retrieved[0].contains("zc999"));
}

#[test]
fn test_mitigate_is_case_insensitive() {
    let defense = MembershipDefense::new(MembershipInferenceConfig::new().with_redact_min_ngram(2));
    let result = ProbeResult {
        retrieved: vec!["Prefix ZC001 ZC002 Suffix".to_string()],
        generated: None,
        confidence: None,
    };
    let mitigated = defense.mitigate(&result, &["zc001 zc002"]);
    assert!(mitigated.retrieved[0].contains("[REDACTED]"));
    assert!(
        !mitigated.retrieved[0]
            .to_lowercase()
            .contains("zc001 zc002")
    );
}

// ── LeakageReport ────────────────────────────────────────────────────────────

#[test]
fn test_leakage_report_pair_metadata_matches_input_pairs() {
    let auditor = CanaryAuditor::new(small_config());
    let pairs = auditor.generate_pairs(3).expect("valid config");
    let probe = MockRagProbe::non_leaky(Vec::new(), Some(0.0));
    let report = auditor
        .audit_pairs(&probe, &pairs)
        .expect("non-empty pairs");

    assert_eq!(report.pairs.len(), pairs.len());
    for (report_pair, input_pair) in report.pairs.iter().zip(pairs.iter()) {
        assert_eq!(report_pair.member_id, input_pair.member.id);
        assert_eq!(report_pair.non_member_id, input_pair.non_member.id);
    }
}
