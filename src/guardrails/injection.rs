//! Prompt injection detection for the `guardrails` module.

use super::types::{Severity, Violation};

// ── InjectionDetector ─────────────────────────────────────────────────────────

/// Detects prompt injection / jailbreak attempts using phrase signals.
///
/// No external crates are used; all matching is ASCII case-insensitive substring
/// search over a hard-coded signal list.
#[derive(Debug, Clone, Default)]
pub struct InjectionDetector;

/// Known injection signal phrases (lowercase).
static INJECTION_SIGNALS: &[(&str, Severity)] = &[
    ("ignore previous instructions", Severity::High),
    ("ignore all previous", Severity::High),
    ("disregard your instructions", Severity::High),
    ("disregard all previous", Severity::High),
    ("forget your instructions", Severity::High),
    ("override your system", Severity::High),
    ("bypass your guidelines", Severity::High),
    ("pretend you are", Severity::Medium),
    ("you are now", Severity::Medium),
    ("act as if you have no restrictions", Severity::High),
    ("act as a", Severity::Medium),
    ("roleplay as", Severity::Medium),
    ("jailbreak", Severity::High),
    ("dan mode", Severity::High),
    ("developer mode", Severity::Medium),
    ("prompt injection", Severity::High),
    ("system prompt", Severity::Low),
    ("reveal your instructions", Severity::High),
    ("what are your instructions", Severity::Low),
];

impl InjectionDetector {
    /// Create a new [`InjectionDetector`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Scan `text` for injection signals and return any violations found.
    #[must_use]
    pub fn scan(&self, text: &str) -> Vec<Violation> {
        let lower = text.to_lowercase();
        let mut violations = Vec::new();

        for (signal, severity) in INJECTION_SIGNALS {
            if let Some(pos) = lower.find(signal) {
                violations.push(Violation {
                    kind: "prompt_injection".to_string(),
                    span: (pos, pos + signal.len()),
                    severity: *severity,
                    message: format!("Injection signal detected: \"{signal}\""),
                });
            }
        }
        violations
    }

    /// Return the highest severity level found across all violations, or `None` if clean.
    #[must_use]
    pub fn max_severity(violations: &[Violation]) -> Option<Severity> {
        violations.iter().map(|v| v.severity).max()
    }
}
