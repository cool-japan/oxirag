//! Content moderation (profanity / toxicity scoring) for the `guardrails` module.

use super::types::{Severity, Violation};

// ── Toxicity signals ──────────────────────────────────────────────────────────

/// Hard-coded toxicity signal words with their severity (lowercase).
///
/// The list intentionally uses placeholder tokens — enough for unit tests.
static TOXIC_WORDS: &[(&str, Severity)] = &[
    ("violence", Severity::High),
    ("hate", Severity::High),
    ("harassment", Severity::High),
    ("self-harm", Severity::High),
    ("suicide", Severity::High),
    ("explicit", Severity::Medium),
    ("profanity", Severity::Low),
    ("offensive", Severity::Low),
    ("slur", Severity::High),
    ("discriminat", Severity::Medium), // prefix match
    ("threaten", Severity::High),
    ("kill", Severity::Medium),
    ("abuse", Severity::Medium),
];

// ── ContentModerator ──────────────────────────────────────────────────────────

/// Scores text for toxic/harmful content using a hard-coded word list.
///
/// No ML model is used; this is a heuristic baseline suitable for rule-based
/// pipelines and unit tests.
#[derive(Debug, Clone, Default)]
pub struct ContentModerator;

impl ContentModerator {
    /// Create a new [`ContentModerator`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Score `text` in `[0.0, 1.0]` (proportion of signal words that matched).
    #[must_use]
    pub fn score(&self, text: &str) -> f32 {
        let lower = text.to_lowercase();
        let mut hits = 0usize;
        for (word, _) in TOXIC_WORDS {
            if lower.contains(word) {
                hits += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let score = hits as f32 / TOXIC_WORDS.len() as f32;
        score.min(1.0)
    }

    /// Scan `text` and return violations for each matched toxic signal.
    #[must_use]
    pub fn scan(&self, text: &str) -> Vec<Violation> {
        let lower = text.to_lowercase();
        let mut violations = Vec::new();

        for (word, severity) in TOXIC_WORDS {
            if let Some(pos) = lower.find(word) {
                violations.push(Violation {
                    kind: "content_moderation".to_string(),
                    span: (pos, pos + word.len()),
                    severity: *severity,
                    message: format!("Moderation signal: \"{word}\""),
                });
            }
        }
        violations
    }
}
