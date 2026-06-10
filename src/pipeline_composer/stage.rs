//! Built-in [`PipelineStage`] implementations.

use super::types::{ComposerError, PipelineStage, StageInput, StageOutput};

// ── PassThroughStage ──────────────────────────────────────────────────────────

/// A no-op stage that passes `input.context` through unchanged.
pub struct PassThroughStage {
    /// Human-readable label for this stage.
    label: String,
}

impl PassThroughStage {
    /// Create a new [`PassThroughStage`] with the given label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl PipelineStage for PassThroughStage {
    fn name(&self) -> &str {
        &self.label
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        Ok(StageOutput::pass(input.context))
    }
}

// ── FormatStage ───────────────────────────────────────────────────────────────

/// Wraps `input.context` with a fixed `prefix` and `suffix`.
pub struct FormatStage {
    /// Text prepended to the context.
    prefix: String,
    /// Text appended to the context.
    suffix: String,
}

impl FormatStage {
    /// Create a new [`FormatStage`].
    #[must_use]
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }
    }
}

#[allow(clippy::unnecessary_literal_bound)]
impl PipelineStage for FormatStage {
    fn name(&self) -> &str {
        "format"
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        Ok(StageOutput::pass(format!(
            "{}{}{}",
            self.prefix, input.context, self.suffix
        )))
    }
}

// ── TruncateStage ─────────────────────────────────────────────────────────────

/// Truncates `input.context` to at most `max_chars` Unicode scalar values.
pub struct TruncateStage {
    /// Maximum number of characters to keep.
    max_chars: usize,
}

impl TruncateStage {
    /// Create a new [`TruncateStage`].
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

#[allow(clippy::unnecessary_literal_bound)]
impl PipelineStage for TruncateStage {
    fn name(&self) -> &str {
        "truncate"
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        let truncated: String = input.context.chars().take(self.max_chars).collect();
        Ok(StageOutput::pass(truncated))
    }
}

// ── KeywordFilterStage ────────────────────────────────────────────────────────

/// Blocks the pipeline when `input.context` contains any blocked keyword.
pub struct KeywordFilterStage {
    /// Keywords that trigger a block.
    blocked_keywords: Vec<String>,
    /// Whether the keyword check is case-insensitive.
    case_insensitive: bool,
}

impl KeywordFilterStage {
    /// Create a new [`KeywordFilterStage`] (case-sensitive by default).
    #[must_use]
    pub fn new(keywords: Vec<String>) -> Self {
        Self {
            blocked_keywords: keywords,
            case_insensitive: false,
        }
    }

    /// Enable case-insensitive keyword matching.
    #[must_use]
    pub fn with_case_insensitive(mut self, v: bool) -> Self {
        self.case_insensitive = v;
        self
    }
}

#[allow(clippy::unnecessary_literal_bound)]
impl PipelineStage for KeywordFilterStage {
    fn name(&self) -> &str {
        "keyword_filter"
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        let haystack = if self.case_insensitive {
            input.context.to_lowercase()
        } else {
            input.context.clone()
        };

        for kw in &self.blocked_keywords {
            let needle = if self.case_insensitive {
                kw.to_lowercase()
            } else {
                kw.clone()
            };
            if haystack.contains(needle.as_str()) {
                return Ok(StageOutput::block(format!(
                    "Blocked keyword detected: {kw}"
                )));
            }
        }

        Ok(StageOutput::pass(input.context))
    }
}

// ── SanitizerStage ────────────────────────────────────────────────────────────

/// Strips leading/trailing whitespace and collapses consecutive whitespace to
/// a single space.
pub struct SanitizerStage {}

impl SanitizerStage {
    /// Create a new [`SanitizerStage`].
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SanitizerStage {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::unnecessary_literal_bound)]
impl PipelineStage for SanitizerStage {
    fn name(&self) -> &str {
        "sanitizer"
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        let sanitized = input
            .context
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        Ok(StageOutput::pass(sanitized))
    }
}

// ── GuardrailStage ────────────────────────────────────────────────────────────

/// Runs the full `GuardrailEngine` over `input.query` and blocks the pipeline
/// if a high-severity violation is detected.
///
/// Only available when the `guardrails` feature is enabled.
#[cfg(feature = "guardrails")]
pub struct GuardrailStage {
    /// Guardrail configuration.
    config: crate::guardrails::GuardrailConfig,
}

#[cfg(feature = "guardrails")]
impl GuardrailStage {
    /// Create a new [`GuardrailStage`] with the given configuration.
    #[must_use]
    pub fn new(config: crate::guardrails::GuardrailConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "guardrails")]
#[allow(clippy::unnecessary_literal_bound)]
impl PipelineStage for GuardrailStage {
    fn name(&self) -> &str {
        "guardrail"
    }

    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        let engine = crate::guardrails::GuardrailEngine::new();
        let report = engine
            .check(&input.query, &self.config)
            .map_err(|e| ComposerError::StageFailed("guardrail".to_string(), e.to_string()))?;

        if report.blocked {
            Ok(StageOutput::block(format!(
                "Guardrail violation: {}",
                report
                    .violations
                    .first()
                    .map_or("policy violated", |v| v.message.as_str())
            )))
        } else {
            Ok(StageOutput::pass(input.context))
        }
    }
}
