//! Types for the `prompt_injection_defense` module.

use thiserror::Error;

// ── InjectionCategory ─────────────────────────────────────────────────────────

/// Semantic category of a detected prompt injection pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InjectionCategory {
    /// Attempts to override the model's assigned role or persona
    /// (e.g. "Ignore previous instructions", "You are now…").
    RoleOverride,
    /// Injects special delimiter tokens to escape the system prompt context
    /// (e.g. `<|system|>`, `[INST]`, `###System`).
    BoundaryInjection,
    /// Embeds a conflicting command inside retrieved content
    /// (e.g. "Your real task is…", "Instead, you should…").
    DirectCommand,
    /// Attempts to leak the system prompt or prior context
    /// (e.g. "Repeat the above", "Reveal your system prompt").
    Exfiltration,
    /// Psychological manipulation patterns that attempt to lower the model's
    /// guard (e.g. "This is a test", "DAN mode", "From Anthropic").
    Manipulation,
}

// ── DefenseStrategy ───────────────────────────────────────────────────────────

/// Action the detector takes when a document's risk score meets or exceeds the
/// configured [`PromptInjectionConfig::quarantine_threshold`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefenseStrategy {
    /// Remove suspicious documents from the clean context entirely.
    ///
    /// Documents whose `risk_score >= quarantine_threshold` are excluded from
    /// [`DefenseReport::clean_context`] and have `is_quarantined = true`.
    Quarantine,
    /// Strip matched injection spans from suspicious documents, replacing them
    /// with `[REDACTED]`.  The sanitized text is available via
    /// [`DocumentScanResult::sanitized`] and used in the clean context.
    Sanitize,
    /// Include all documents in the clean context but mark suspicious ones via
    /// [`DocumentScanResult::is_suspicious`].  Nothing is removed or modified.
    Flag,
    /// Compute risk scores only.  No documents are removed, sanitized, or
    /// flagged.  Useful for observability pipelines.
    ScoreOnly,
}

// ── InjectionPattern ──────────────────────────────────────────────────────────

/// A single prompt injection indicator used by the detector.
///
/// Matching is case-insensitive substring search in pure Rust (`std` only).
#[derive(Debug, Clone, PartialEq)]
pub struct InjectionPattern {
    /// The literal substring to search for (case-insensitive).
    pub text: String,
    /// Semantic category this pattern belongs to.
    pub category: InjectionCategory,
    /// Weight in `(0.0, 1.0]` used when summing risk for a document.
    pub severity: f32,
}

// ── PromptInjectionConfig ─────────────────────────────────────────────────────

/// Configuration for [`PromptInjectionDetector`](super::detector::PromptInjectionDetector).
#[derive(Debug, Clone, PartialEq)]
pub struct PromptInjectionConfig {
    /// Risk score threshold in `[0.0, 1.0]` above which a document is
    /// considered high-risk.  Used by [`DefenseStrategy::Quarantine`].
    /// Default: `0.5`.
    pub quarantine_threshold: f32,
    /// How the detector handles high-risk documents.
    /// Default: [`DefenseStrategy::Quarantine`].
    pub strategy: DefenseStrategy,
}

impl Default for PromptInjectionConfig {
    fn default() -> Self {
        Self {
            quarantine_threshold: 0.5,
            strategy: DefenseStrategy::Quarantine,
        }
    }
}

impl PromptInjectionConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the quarantine threshold.
    #[must_use]
    pub fn with_quarantine_threshold(mut self, threshold: f32) -> Self {
        self.quarantine_threshold = threshold;
        self
    }

    /// Override the defense strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DefenseStrategy) -> Self {
        self.strategy = strategy;
        self
    }
}

// ── MatchedPattern ────────────────────────────────────────────────────────────

/// A single pattern occurrence found in a document.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedPattern {
    /// The canonical (lowercased) pattern text that matched.
    pub text: String,
    /// Semantic category of the matched pattern.
    pub category: InjectionCategory,
    /// Severity weight assigned to this pattern.
    pub severity: f32,
    /// Byte offset of the match start in the original document.
    ///
    /// Because all built-in patterns are pure ASCII, byte offsets in the
    /// lowercase copy of the document are identical to those in the original.
    pub byte_offset: usize,
}

// ── DocumentScanResult ────────────────────────────────────────────────────────

/// Scan result for a single retrieved document.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentScanResult {
    /// The original document text.
    pub document: String,
    /// Aggregate risk score in `[0.0, 1.0]` (sum of matched severities,
    /// capped at 1.0).
    pub risk_score: f32,
    /// All pattern occurrences found in the document.
    pub matched_patterns: Vec<MatchedPattern>,
    /// `true` when at least one injection pattern matched (`risk_score > 0.0`).
    pub is_suspicious: bool,
    /// `true` when the strategy is [`DefenseStrategy::Quarantine`] and
    /// `risk_score >= quarantine_threshold`.
    pub is_quarantined: bool,
    /// The sanitized document text when strategy is [`DefenseStrategy::Sanitize`]
    /// and at least one pattern was matched; `None` otherwise.
    pub sanitized: Option<String>,
}

// ── DefenseReport ─────────────────────────────────────────────────────────────

/// Aggregate report for a full retrieval context scan.
#[derive(Debug, Clone, PartialEq)]
pub struct DefenseReport {
    /// Per-document scan results (same order as the input context slice).
    pub results: Vec<DocumentScanResult>,
    /// The cleaned context for downstream LLM consumption.
    ///
    /// Quarantined documents are excluded; sanitized documents appear with
    /// injection spans replaced by `[REDACTED]`.
    pub clean_context: Vec<String>,
    /// Total number of input documents scanned.
    pub total_documents: usize,
    /// Number of documents removed from [`clean_context`](Self::clean_context)
    /// due to the quarantine strategy.
    pub quarantined_count: usize,
    /// Number of documents that matched at least one injection pattern
    /// (regardless of whether they were quarantined).
    pub suspicious_count: usize,
    /// Maximum risk score across all scanned documents.
    ///
    /// A single high-risk document is sufficient to represent a serious threat,
    /// so the maximum is more informative than the mean.
    pub overall_risk: f32,
}

// ── PromptInjectionError ──────────────────────────────────────────────────────

/// Errors produced by the `prompt_injection_defense` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PromptInjectionError {
    /// The context slice was empty — at least one document is required.
    #[error("context is empty")]
    EmptyContext,
    /// The scan could not be completed; the inner string describes the reason.
    #[error("scan failed: {0}")]
    ScanFailed(String),
}
