//! Types for the `output_validation` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── RuleKind ──────────────────────────────────────────────────────────────────

/// The kind of output validation rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RuleKind {
    /// Answer must be at least this many characters.
    MinLength(usize),
    /// Answer must not exceed this many characters.
    MaxLength(usize),
    /// Answer must contain at least one citation (e.g. `[1]`, `(source:`, `[source`).
    RequiresCitation,
    /// Answer must not contain this banned phrase (case-insensitive).
    NoBannedPhrase(String),
    /// Answer must contain this substring (case-insensitive).
    MustContain(String),
    /// Answer must be valid JSON.
    JsonParsable,
    /// Maximum allowed repetition ratio (0.0–1.0): duplicate word count / total word count.
    MaxRepetition(f32),
}

impl RuleKind {
    /// Short label for this rule kind.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::MinLength(_) => "min_length",
            Self::MaxLength(_) => "max_length",
            Self::RequiresCitation => "requires_citation",
            Self::NoBannedPhrase(_) => "no_banned_phrase",
            Self::MustContain(_) => "must_contain",
            Self::JsonParsable => "json_parsable",
            Self::MaxRepetition(_) => "max_repetition",
        }
    }
}

// ── ValidationRule ────────────────────────────────────────────────────────────

/// A single validation rule with its associated severity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// The rule kind.
    pub kind: RuleKind,
    /// Severity if this rule is violated.
    pub severity: RuleSeverity,
}

impl ValidationRule {
    /// Create a new rule with [`RuleSeverity::Medium`] severity.
    #[must_use]
    pub fn new(kind: RuleKind) -> Self {
        Self {
            kind,
            severity: RuleSeverity::Medium,
        }
    }

    /// Set the severity.
    #[must_use]
    pub fn with_severity(mut self, s: RuleSeverity) -> Self {
        self.severity = s;
        self
    }
}

// ── RuleSeverity ──────────────────────────────────────────────────────────────

/// Severity attached to a rule violation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RuleSeverity {
    /// Informational — does not fail validation.
    Low,
    /// Should be reviewed — does not fail validation by default.
    #[default]
    Medium,
    /// Critical — fails validation when `fail_on` is `Medium` or lower.
    High,
}

// ── RuleViolation ─────────────────────────────────────────────────────────────

/// A single violated rule in the validation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleViolation {
    /// Label of the violated rule kind.
    pub rule: String,
    /// Severity of this violation.
    pub severity: RuleSeverity,
    /// Human-readable explanation.
    pub message: String,
}

// ── ValidationReport ──────────────────────────────────────────────────────────

/// The complete result of output validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the answer passed all rules at the configured threshold.
    pub passed: bool,
    /// All violated rules.
    pub violations: Vec<RuleViolation>,
    /// The highest severity among all violations.
    pub max_severity: Option<RuleSeverity>,
}

impl ValidationReport {
    /// Return `true` when no violations were detected.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}

// ── ValidationConfig ──────────────────────────────────────────────────────────

/// Configuration for the output validator.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Violations at or above this severity cause `passed = false`.
    ///
    /// Defaults to [`RuleSeverity::High`].
    pub fail_on: RuleSeverity,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            fail_on: RuleSeverity::High,
        }
    }
}

impl ValidationConfig {
    /// Set the fail-on threshold.
    #[must_use]
    pub fn with_fail_on(mut self, v: RuleSeverity) -> Self {
        self.fail_on = v;
        self
    }
}

// ── OutputValidationError ─────────────────────────────────────────────────────

/// Errors from the `output_validation` module.
#[derive(Debug, Error)]
pub enum OutputValidationError {
    /// No rules were provided.
    #[error("Validation requires at least one rule")]
    NoRules,
}
