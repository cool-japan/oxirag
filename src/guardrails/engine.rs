//! The main `GuardrailEngine` orchestrator.

use super::injection::InjectionDetector;
use super::moderation::ContentModerator;
use super::pii::PiiDetector;
use super::types::{GuardrailConfig, GuardrailError, GuardrailReport, Severity, Violation};

// ── GuardrailEngine ───────────────────────────────────────────────────────────

/// Runs PII detection, injection detection, content moderation, and topical
/// rails in a single pass over `text`.
///
/// All checks are synchronous; no I/O or async is required.
#[derive(Debug, Clone, Default)]
pub struct GuardrailEngine {
    pii_detector: PiiDetector,
    injection_detector: InjectionDetector,
    content_moderator: ContentModerator,
}

impl GuardrailEngine {
    /// Create a new [`GuardrailEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            pii_detector: PiiDetector::new(),
            injection_detector: InjectionDetector::new(),
            content_moderator: ContentModerator::new(),
        }
    }

    /// Run all configured guardrails over `text` and return a [`GuardrailReport`].
    ///
    /// # Errors
    ///
    /// Returns [`GuardrailError::InvalidConfig`] when `moderation_threshold` is
    /// outside `[0.0, 1.0]`.
    pub fn check(
        &self,
        text: &str,
        config: &GuardrailConfig,
    ) -> Result<GuardrailReport, GuardrailError> {
        if !(0.0..=1.0).contains(&config.moderation_threshold) {
            return Err(GuardrailError::InvalidConfig(
                "moderation_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }

        let mut violations: Vec<Violation> = Vec::new();

        // ── PII ───────────────────────────────────────────────────────────────
        let pii_matches = self.pii_detector.scan(text);
        for m in &pii_matches {
            violations.push(Violation {
                kind: format!("pii_{}", m.kind.label()),
                span: (m.start, m.end),
                severity: Severity::High,
                message: format!("PII detected ({}): \"{}\"", m.kind.label(), &m.matched),
            });
        }

        // ── Redact PII ────────────────────────────────────────────────────────
        let redacted_text = if config.redact_pii {
            self.pii_detector.redact(text)
        } else {
            text.to_string()
        };

        // ── Injection ─────────────────────────────────────────────────────────
        if config.block_injection {
            let injection_violations = self.injection_detector.scan(text);
            violations.extend(injection_violations);
        }

        // ── Content moderation ────────────────────────────────────────────────
        let mod_score = self.content_moderator.score(text);
        if mod_score >= config.moderation_threshold {
            let mod_violations = self.content_moderator.scan(text);
            violations.extend(mod_violations);
        }

        // ── Topical rail ──────────────────────────────────────────────────────
        if let Some(rail) = &config.topical_rail {
            let lower = text.to_lowercase();
            for deny_term in &rail.deny {
                if let Some(pos) = lower.find(deny_term.to_lowercase().as_str()) {
                    violations.push(Violation {
                        kind: "topical_deny".to_string(),
                        span: (pos, pos + deny_term.len()),
                        severity: Severity::High,
                        message: format!("Denied topic: \"{deny_term}\""),
                    });
                }
            }
        }

        // ── Derive summary fields ─────────────────────────────────────────────
        let blocked = violations.iter().any(|v| v.severity == Severity::High);
        let high_count = violations
            .iter()
            .filter(|v| v.severity == Severity::High)
            .count();
        #[allow(clippy::cast_precision_loss)]
        let risk_score = if violations.is_empty() {
            0.0f32
        } else {
            high_count as f32 / violations.len() as f32
        };

        Ok(GuardrailReport {
            violations,
            redacted_text,
            blocked,
            risk_score,
        })
    }
}
