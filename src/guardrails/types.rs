//! Types for the `guardrails` module.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── PiiKind ───────────────────────────────────────────────────────────────────

/// The category of personally-identifiable information detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiKind {
    /// E-mail address pattern.
    Email,
    /// Phone number pattern.
    Phone,
    /// U.S. Social Security Number (XXX-XX-XXXX).
    Ssn,
    /// Credit/debit card number (16 digits, various separators).
    CreditCard,
    /// IPv4 address.
    IpAddress,
}

impl PiiKind {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::Ssn => "ssn",
            Self::CreditCard => "credit_card",
            Self::IpAddress => "ip_address",
        }
    }
}

// ── PiiMatch ──────────────────────────────────────────────────────────────────

/// A single PII match within the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiMatch {
    /// Kind of PII detected.
    pub kind: PiiKind,
    /// Byte span `[start, end)` within the original text.
    pub start: usize,
    /// End of the byte span.
    pub end: usize,
    /// The raw matched text.
    pub matched: String,
}

// ── Severity ──────────────────────────────────────────────────────────────────

/// Severity level of a guardrail violation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Low severity — informational.
    #[default]
    Low,
    /// Medium severity — should be reviewed.
    Medium,
    /// High severity — should be blocked.
    High,
}

// ── Violation ─────────────────────────────────────────────────────────────────

/// A single guardrail violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Short label for this violation type.
    pub kind: String,
    /// Byte span of the offending text.
    pub span: (usize, usize),
    /// Severity level.
    pub severity: Severity,
    /// Human-readable explanation.
    pub message: String,
}

// ── GuardrailReport ───────────────────────────────────────────────────────────

/// The complete result of running guardrails over a text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailReport {
    /// All detected violations.
    pub violations: Vec<Violation>,
    /// Text with PII redacted (if configured).
    pub redacted_text: String,
    /// `true` when at least one `High`-severity violation was detected.
    pub blocked: bool,
    /// Score in `[0.0, 1.0]` — proportion of high-severity violations.
    pub risk_score: f32,
}

impl GuardrailReport {
    /// Return `true` when no violations were found.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}

// ── GuardrailConfig ───────────────────────────────────────────────────────────

/// Configuration for `GuardrailEngine`.
#[derive(Debug, Clone)]
pub struct GuardrailConfig {
    /// Whether to redact PII from the output text.
    ///
    /// Defaults to `true`.
    pub redact_pii: bool,
    /// Whether to block on injection signals.
    ///
    /// Defaults to `true`.
    pub block_injection: bool,
    /// Score above which moderation is considered high severity.
    ///
    /// Defaults to `0.5`.
    pub moderation_threshold: f32,
    /// Topical rails — phrases/terms that are explicitly allowed or denied.
    pub topical_rail: Option<TopicalRail>,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            redact_pii: true,
            block_injection: true,
            moderation_threshold: 0.5,
            topical_rail: None,
        }
    }
}

impl GuardrailConfig {
    /// Set whether to redact PII.
    #[must_use]
    pub fn with_redact_pii(mut self, v: bool) -> Self {
        self.redact_pii = v;
        self
    }

    /// Set whether to block prompt injection.
    #[must_use]
    pub fn with_block_injection(mut self, v: bool) -> Self {
        self.block_injection = v;
        self
    }

    /// Set the moderation score threshold.
    #[must_use]
    pub fn with_moderation_threshold(mut self, v: f32) -> Self {
        self.moderation_threshold = v;
        self
    }

    /// Set topical rails.
    #[must_use]
    pub fn with_topical_rail(mut self, v: TopicalRail) -> Self {
        self.topical_rail = Some(v);
        self
    }
}

// ── TopicalRail ───────────────────────────────────────────────────────────────

/// Allowlist/denylist-based topical guardrail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopicalRail {
    /// Terms that are always allowed (skip moderation check if present).
    pub allow: Vec<String>,
    /// Terms that trigger an immediate `High` violation.
    pub deny: Vec<String>,
}

impl TopicalRail {
    /// Create a new [`TopicalRail`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an allowed term.
    #[must_use]
    pub fn with_allow(mut self, term: impl Into<String>) -> Self {
        self.allow.push(term.into());
        self
    }

    /// Add a denied term.
    #[must_use]
    pub fn with_deny(mut self, term: impl Into<String>) -> Self {
        self.deny.push(term.into());
        self
    }
}

// ── GuardrailError ────────────────────────────────────────────────────────────

/// Errors from the `guardrails` module.
#[derive(Debug, Error)]
pub enum GuardrailError {
    /// Invalid configuration detected at runtime.
    #[error("Invalid guardrail configuration: {0}")]
    InvalidConfig(String),
}
