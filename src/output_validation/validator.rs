//! Output validation engine.

use super::types::{
    OutputValidationError, RuleKind, RuleViolation, ValidationConfig, ValidationReport,
    ValidationRule,
};

// ── repetition helper ─────────────────────────────────────────────────────────

/// Compute the duplicate word ratio `(total - unique) / total`.
fn repetition_ratio(text: &str) -> f32 {
    use std::collections::HashSet;
    let words: Vec<&str> = text.split_whitespace().collect();
    let total = words.len();
    if total == 0 {
        return 0.0;
    }
    let unique: HashSet<&str> = words.iter().copied().collect();
    #[allow(clippy::cast_precision_loss)]
    let ratio = (total - unique.len()) as f32 / total as f32;
    ratio
}

/// Return `true` if `text` contains a citation signal.
fn has_citation(text: &str) -> bool {
    let lower = text.to_lowercase();
    // common citation patterns
    lower.contains("[source")
        || lower.contains("(source")
        || lower.contains("[ref")
        || lower.contains("[citation")
        || {
            // numeric citation [1], [2], etc.
            let bytes = text.as_bytes();
            let n = bytes.len();
            let mut i = 0;
            while i < n {
                if bytes[i] == b'[' {
                    let mut j = i + 1;
                    while j < n && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > i + 1 && j < n && bytes[j] == b']' {
                        return true;
                    }
                }
                i += 1;
            }
            false
        }
}

// ── OutputValidator ───────────────────────────────────────────────────────────

/// Validates a generated answer against a set of rules.
///
/// All checks are synchronous and allocation-light.
#[derive(Debug, Clone, Default)]
pub struct OutputValidator;

impl OutputValidator {
    /// Create a new [`OutputValidator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate `answer` against `rules` with the given `config`.
    ///
    /// # Errors
    ///
    /// Returns [`OutputValidationError::NoRules`] when `rules` is empty.
    pub fn validate(
        &self,
        answer: &str,
        rules: &[ValidationRule],
        config: &ValidationConfig,
    ) -> Result<ValidationReport, OutputValidationError> {
        if rules.is_empty() {
            return Err(OutputValidationError::NoRules);
        }

        let mut violations: Vec<RuleViolation> = Vec::new();

        for rule in rules {
            if let Some(v) = self.check_rule(answer, rule) {
                violations.push(v);
            }
        }

        let max_severity = violations.iter().map(|v| v.severity).max();
        let passed = match max_severity {
            None => true,
            Some(s) => s < config.fail_on,
        };

        Ok(ValidationReport {
            passed,
            violations,
            max_severity,
        })
    }

    #[allow(clippy::unused_self)]
    fn check_rule(&self, answer: &str, rule: &ValidationRule) -> Option<RuleViolation> {
        match &rule.kind {
            RuleKind::MinLength(min) => {
                let len = answer.len();
                (len < *min).then(|| RuleViolation {
                    rule: rule.kind.label().to_string(),
                    severity: rule.severity,
                    message: format!("Answer too short: {len} chars (minimum {min})"),
                })
            }
            RuleKind::MaxLength(max) => {
                let len = answer.len();
                (len > *max).then(|| RuleViolation {
                    rule: rule.kind.label().to_string(),
                    severity: rule.severity,
                    message: format!("Answer too long: {len} chars (maximum {max})"),
                })
            }
            RuleKind::RequiresCitation => (!has_citation(answer)).then(|| RuleViolation {
                rule: rule.kind.label().to_string(),
                severity: rule.severity,
                message: "Answer must include at least one citation".to_string(),
            }),
            RuleKind::NoBannedPhrase(phrase) => answer
                .to_lowercase()
                .contains(phrase.to_lowercase().as_str())
                .then(|| RuleViolation {
                    rule: rule.kind.label().to_string(),
                    severity: rule.severity,
                    message: format!("Banned phrase found: \"{phrase}\""),
                }),
            RuleKind::MustContain(required) => (!answer
                .to_lowercase()
                .contains(required.to_lowercase().as_str()))
            .then(|| RuleViolation {
                rule: rule.kind.label().to_string(),
                severity: rule.severity,
                message: format!("Answer must contain: \"{required}\""),
            }),
            RuleKind::JsonParsable => serde_json::from_str::<serde_json::Value>(answer)
                .is_err()
                .then(|| RuleViolation {
                    rule: rule.kind.label().to_string(),
                    severity: rule.severity,
                    message: "Answer is not valid JSON".to_string(),
                }),
            RuleKind::MaxRepetition(max_ratio) => {
                let ratio = repetition_ratio(answer);
                (ratio > *max_ratio).then(|| RuleViolation {
                    rule: rule.kind.label().to_string(),
                    severity: rule.severity,
                    message: format!("Repetition ratio {ratio:.3} exceeds maximum {max_ratio:.3}"),
                })
            }
        }
    }
}
