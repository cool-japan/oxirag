//! Core types for the attribution module.
//!
//! Defines the fundamental data structures used across the citation and source
//! attribution pipeline: identifiers, citations, annotated spans, configuration,
//! and error variants.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::citation::CitationStyle;

// ── CitationId ────────────────────────────────────────────────────────────────

/// An opaque identifier for a single [`Citation`].
///
/// Typically a numeric string (`"1"`, `"2"`, …) after deduplication, but
/// may be any non-empty string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CitationId(pub String);

impl CitationId {
    /// Create a new [`CitationId`] from any string-like value.
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the inner string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CitationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for CitationId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CitationId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ── Citation ──────────────────────────────────────────────────────────────────

/// A single source citation linking an answer span to a retrieved document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// Unique identifier for this citation (numeric after deduplication).
    pub id: CitationId,
    /// The [`crate::types::DocumentId`] of the source document.
    pub source_id: crate::types::DocumentId,
    /// Optional human-readable title of the source document.
    pub source_title: Option<String>,
    /// The passage from the source document most relevant to the answer span.
    pub supporting_text: String,
}

impl Citation {
    /// Build a new [`Citation`] with no title.
    #[must_use]
    pub fn new(
        id: impl Into<CitationId>,
        source_id: impl Into<crate::types::DocumentId>,
        supporting_text: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source_id: source_id.into(),
            source_title: None,
            supporting_text: supporting_text.into(),
        }
    }

    /// Attach a human-readable title to this citation.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.source_title = Some(title.into());
        self
    }
}

// ── CitedSpan ─────────────────────────────────────────────────────────────────

/// A sentence from the generated answer together with its attached citations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CitedSpan {
    /// The sentence text (trimmed, as produced by the sentence splitter).
    pub sentence: String,
    /// Zero-based position of this sentence in the original answer.
    pub sentence_index: usize,
    /// All citations attached to this span (may be empty).
    pub citations: Vec<Citation>,
    /// Best alignment score among all attached citations; `0.0` when empty.
    pub grounding_score: f32,
}

impl CitedSpan {
    /// Returns `true` when `grounding_score >= threshold`.
    #[must_use]
    pub fn is_grounded(&self, threshold: f32) -> bool {
        self.grounding_score >= threshold
    }
}

// ── AttributionConfig ─────────────────────────────────────────────────────────

/// Configuration knobs for the attribution pipeline.
#[derive(Debug, Clone)]
pub struct AttributionConfig {
    /// Minimum alignment score required to attach a citation to a sentence.
    ///
    /// Defaults to `0.3`.
    pub alignment_threshold: f32,

    /// Minimum grounding score for a span to count as "grounded" in
    /// faithfulness computation.
    ///
    /// Defaults to `0.5`.
    pub grounding_threshold: f32,

    /// Maximum number of citations attached to any single sentence.
    ///
    /// Defaults to `3`.
    pub max_citations_per_sentence: usize,

    /// Style used when rendering inline citation markers.
    ///
    /// Defaults to [`CitationStyle::Numeric`].
    pub citation_style: CitationStyle,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            alignment_threshold: 0.3,
            grounding_threshold: 0.5,
            max_citations_per_sentence: 3,
            citation_style: CitationStyle::Numeric,
        }
    }
}

impl AttributionConfig {
    /// Override the alignment threshold.
    #[must_use]
    pub fn with_alignment_threshold(mut self, t: f32) -> Self {
        self.alignment_threshold = t;
        self
    }

    /// Override the grounding threshold.
    #[must_use]
    pub fn with_grounding_threshold(mut self, t: f32) -> Self {
        self.grounding_threshold = t;
        self
    }

    /// Override the maximum number of citations per sentence.
    #[must_use]
    pub fn with_max_citations_per_sentence(mut self, n: usize) -> Self {
        self.max_citations_per_sentence = n;
        self
    }

    /// Override the citation style.
    #[must_use]
    pub fn with_citation_style(mut self, style: CitationStyle) -> Self {
        self.citation_style = style;
        self
    }
}

// ── AttributedAnswer ──────────────────────────────────────────────────────────

/// The complete output of the attribution pipeline for a single answer.
#[derive(Debug, Clone)]
pub struct AttributedAnswer {
    /// The original answer string, unchanged.
    pub original_answer: String,

    /// The answer with inline citation markers appended after grounded sentences.
    pub annotated_answer: String,

    /// Deduplicated citations ordered by first-seen source ID.
    pub citations: Vec<Citation>,

    /// Per-sentence annotation results.
    pub spans: Vec<CitedSpan>,

    /// Fraction of grounded spans over total spans; `0.0` when there are no spans.
    pub overall_faithfulness: f32,
}

// ── AttributionError ──────────────────────────────────────────────────────────

/// Errors that can be returned by the attribution pipeline.
#[derive(Debug, Error)]
pub enum AttributionError {
    /// The answer string was empty after trimming.
    #[error("Answer must not be empty")]
    EmptyAnswer,

    /// No source documents were provided.
    #[error("At least one source document must be provided")]
    NoSources,

    /// An internal alignment failure occurred.
    #[error("Alignment failed: {0}")]
    AlignmentFailed(String),
}
