//! Tests for the `output_validation` module.

use super::types::{
    OutputValidationError, RuleKind, RuleSeverity, ValidationConfig, ValidationRule,
};
use super::validator::OutputValidator;

// ── RuleKind tests ────────────────────────────────────────────────────────────

#[test]
fn test_rule_kind_labels() {
    assert_eq!(RuleKind::MinLength(0).label(), "min_length");
    assert_eq!(RuleKind::MaxLength(0).label(), "max_length");
    assert_eq!(RuleKind::RequiresCitation.label(), "requires_citation");
    assert_eq!(
        RuleKind::NoBannedPhrase("x".into()).label(),
        "no_banned_phrase"
    );
    assert_eq!(RuleKind::MustContain("y".into()).label(), "must_contain");
    assert_eq!(RuleKind::JsonParsable.label(), "json_parsable");
    assert_eq!(RuleKind::MaxRepetition(0.5).label(), "max_repetition");
}

#[test]
fn test_rule_severity_default() {
    assert_eq!(RuleSeverity::default(), RuleSeverity::Medium);
}

#[test]
fn test_rule_severity_ordering() {
    assert!(RuleSeverity::Low < RuleSeverity::Medium);
    assert!(RuleSeverity::Medium < RuleSeverity::High);
}

#[test]
fn test_validation_rule_builder() {
    let rule = ValidationRule::new(RuleKind::MinLength(10)).with_severity(RuleSeverity::High);
    assert_eq!(rule.severity, RuleSeverity::High);
}

// ── OutputValidator tests ─────────────────────────────────────────────────────

#[test]
fn test_validator_no_rules_error() {
    let v = OutputValidator::new();
    let cfg = ValidationConfig::default();
    let err = v.validate("answer", &[], &cfg).expect_err("should fail");
    assert!(matches!(err, OutputValidationError::NoRules));
}

#[test]
fn test_min_length_pass() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::MinLength(5))];
    let cfg = ValidationConfig::default();
    let report = v.validate("hello world", &rules, &cfg).expect("ok");
    assert!(report.is_clean(), "should pass min length");
}

#[test]
fn test_min_length_fail() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::MinLength(100)).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v.validate("short", &rules, &cfg).expect("ok");
    assert!(!report.passed);
    assert!(!report.violations.is_empty());
}

#[test]
fn test_max_length_pass() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::MaxLength(1000))];
    let cfg = ValidationConfig::default();
    let report = v.validate("short answer", &rules, &cfg).expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_max_length_fail() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::MaxLength(3)).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v.validate("too long answer", &rules, &cfg).expect("ok");
    assert!(!report.passed);
}

#[test]
fn test_requires_citation_pass_numeric() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::RequiresCitation).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v
        .validate("According to [1] the answer is yes", &rules, &cfg)
        .expect("ok");
    assert!(report.is_clean(), "numeric citation should pass");
}

#[test]
fn test_requires_citation_pass_source() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::RequiresCitation).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v
        .validate("See [source: wiki] for details", &rules, &cfg)
        .expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_requires_citation_fail() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::RequiresCitation).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v
        .validate("plain answer no citation", &rules, &cfg)
        .expect("ok");
    assert!(!report.passed);
}

#[test]
fn test_no_banned_phrase_pass() {
    let v = OutputValidator::new();
    let rules = vec![
        ValidationRule::new(RuleKind::NoBannedPhrase("forbidden".into()))
            .with_severity(RuleSeverity::High),
    ];
    let cfg = ValidationConfig::default();
    let report = v.validate("clean answer here", &rules, &cfg).expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_no_banned_phrase_fail() {
    let v = OutputValidator::new();
    let rules = vec![
        ValidationRule::new(RuleKind::NoBannedPhrase("forbidden".into()))
            .with_severity(RuleSeverity::High),
    ];
    let cfg = ValidationConfig::default();
    let report = v
        .validate("this is forbidden content", &rules, &cfg)
        .expect("ok");
    assert!(!report.passed);
}

#[test]
fn test_must_contain_pass() {
    let v = OutputValidator::new();
    let rules = vec![
        ValidationRule::new(RuleKind::MustContain("rust".into())).with_severity(RuleSeverity::High),
    ];
    let cfg = ValidationConfig::default();
    let report = v.validate("Rust is great", &rules, &cfg).expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_must_contain_fail() {
    let v = OutputValidator::new();
    let rules = vec![
        ValidationRule::new(RuleKind::MustContain("required_term".into()))
            .with_severity(RuleSeverity::High),
    ];
    let cfg = ValidationConfig::default();
    let report = v.validate("answer without it", &rules, &cfg).expect("ok");
    assert!(!report.passed);
}

#[test]
fn test_json_parsable_pass() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::JsonParsable).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v.validate(r#"{"key": "value"}"#, &rules, &cfg).expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_json_parsable_fail() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::JsonParsable).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v.validate("not json at all", &rules, &cfg).expect("ok");
    assert!(!report.passed);
}

#[test]
fn test_max_repetition_pass() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::MaxRepetition(0.8)).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v
        .validate(
            "each word is different unique distinct separate",
            &rules,
            &cfg,
        )
        .expect("ok");
    assert!(report.is_clean());
}

#[test]
fn test_max_repetition_fail() {
    let v = OutputValidator::new();
    let rules =
        vec![ValidationRule::new(RuleKind::MaxRepetition(0.2)).with_severity(RuleSeverity::High)];
    let cfg = ValidationConfig::default();
    let report = v
        .validate("word word word word word different", &rules, &cfg)
        .expect("ok");
    assert!(!report.passed, "high repetition should fail");
}

#[test]
fn test_report_is_clean_when_no_violations() {
    let v = OutputValidator::new();
    let rules = vec![ValidationRule::new(RuleKind::MinLength(1))];
    let cfg = ValidationConfig::default();
    let report = v.validate("hello", &rules, &cfg).expect("ok");
    assert!(report.is_clean());
    assert!(report.max_severity.is_none());
}

#[test]
fn test_fail_on_medium_threshold() {
    let v = OutputValidator::new();
    let rules = vec![
        ValidationRule::new(RuleKind::MustContain("xyz".into()))
            .with_severity(RuleSeverity::Medium),
    ];
    let cfg = ValidationConfig::default().with_fail_on(RuleSeverity::Medium);
    let report = v.validate("no match", &rules, &cfg).expect("ok");
    assert!(
        !report.passed,
        "medium violation should fail with fail_on=Medium"
    );
}
