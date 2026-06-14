//! Types and traits for the `skeleton_of_thought` module.
//!
//! Skeleton-of-Thought (Ning et al. 2023, "Skeleton-of-Thought: Large Language
//! Models Can Do Parallel Decoding") decouples answer generation into two
//! stages:
//!
//! 1. **Skeleton** — the model first emits a short list of concise *point
//!    headers* that outline the answer (the "skeleton").
//! 2. **Expansion** — each skeleton point is then expanded *independently* (and,
//!    conceptually, in parallel) into a full sentence or short paragraph.
//!
//! The expansions are finally joined back together to form the answer. Because
//! the per-point expansions do not depend on one another, they can be decoded
//! concurrently, reducing end-to-end latency while also giving the answer a
//! cleaner, list-structured shape.
//!
//! The language model is supplied by the caller through two **pure sync** traits:
//! [`SkeletonGenerator`] (stage 1) and [`PointExpander`] (stage 2). Deterministic
//! [`MockSkeletonGenerator`] and [`MockPointExpander`] implementations are
//! provided for testing.

use thiserror::Error;

// ── SkeletonGenerator ───────────────────────────────────────────────────────────

/// Stage 1: produce the *skeleton* — concise point headers outlining the answer.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete generator (e.g. wrapping an LLM); [`MockSkeletonGenerator`] is
/// provided for tests.
pub trait SkeletonGenerator {
    /// Produce concise point headers (the skeleton) for the query.
    fn skeleton(&self, query: &str) -> Vec<String>;
}

// ── PointExpander ───────────────────────────────────────────────────────────────

/// Stage 2: expand a single skeleton point into full content.
///
/// Each point is expanded independently of the others, given the original
/// `query` for context. Implementations are **pure sync**; [`MockPointExpander`]
/// is provided for tests.
pub trait PointExpander {
    /// Expand one skeleton point into full content.
    fn expand(&self, query: &str, point: &str) -> String;
}

// ── MockSkeletonGenerator ───────────────────────────────────────────────────────

/// Deterministic [`SkeletonGenerator`] for tests.
///
/// Returns a clone of its pre-scripted [`MockSkeletonGenerator::points`]
/// regardless of the query, so a run is fully reproducible.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockSkeletonGenerator {
    /// Pre-scripted point headers returned by [`SkeletonGenerator::skeleton`].
    pub points: Vec<String>,
}

impl MockSkeletonGenerator {
    /// Create a mock generator from a fixed list of point headers.
    #[must_use]
    pub fn new(points: Vec<String>) -> Self {
        Self { points }
    }
}

impl SkeletonGenerator for MockSkeletonGenerator {
    fn skeleton(&self, _query: &str) -> Vec<String> {
        self.points.clone()
    }
}

// ── MockPointExpander ───────────────────────────────────────────────────────────

/// Deterministic [`PointExpander`] for tests.
///
/// Expands a point by returning the value of the first `(header, expansion)`
/// pair whose `header` equals the point exactly. When no pair matches, it echoes
/// the point back as `"<header>: details"`, so the expansion always references
/// the header it came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockPointExpander {
    /// Ordered `(header, expansion)` mappings.
    pub expansions: Vec<(String, String)>,
}

impl MockPointExpander {
    /// Create a mock expander from `(header, expansion)` mappings.
    #[must_use]
    pub fn new(expansions: Vec<(String, String)>) -> Self {
        Self { expansions }
    }

    /// Create an empty mock expander that echoes every point as
    /// `"<header>: details"`.
    #[must_use]
    pub fn echo() -> Self {
        Self {
            expansions: Vec::new(),
        }
    }
}

impl PointExpander for MockPointExpander {
    fn expand(&self, _query: &str, point: &str) -> String {
        for (header, expansion) in &self.expansions {
            if header == point {
                return expansion.clone();
            }
        }
        format!("{point}: details")
    }
}

// ── SkeletonPoint ───────────────────────────────────────────────────────────────

/// One resolved skeleton point: a header paired with its expanded content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkeletonPoint {
    /// Zero-based position of this point within the skeleton.
    pub index: usize,
    /// The concise point header from the skeleton (stage 1).
    pub header: String,
    /// The expanded content for [`SkeletonPoint::header`] (stage 2).
    pub content: String,
}

impl SkeletonPoint {
    /// Create a new skeleton point from its index, header, and content.
    #[must_use]
    pub fn new(index: usize, header: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            index,
            header: header.into(),
            content: content.into(),
        }
    }
}

// ── SkeletonConfig ──────────────────────────────────────────────────────────────

/// Configuration for
/// [`SkeletonOfThoughtEngine`](crate::skeleton_of_thought::engine::SkeletonOfThoughtEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkeletonConfig {
    /// Maximum number of skeleton points to expand.
    ///
    /// Defaults to `10`. The skeleton returned by the generator is truncated to
    /// at most this many points before expansion.
    pub max_points: usize,
    /// Separator inserted between expanded contents when joining the answer.
    ///
    /// Defaults to `"\n"`.
    pub join_separator: String,
}

impl Default for SkeletonConfig {
    fn default() -> Self {
        Self {
            max_points: 10,
            join_separator: "\n".to_string(),
        }
    }
}

impl SkeletonConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of skeleton points to expand.
    #[must_use]
    pub fn with_max_points(mut self, max_points: usize) -> Self {
        self.max_points = max_points;
        self
    }

    /// Set the separator inserted between expanded contents.
    #[must_use]
    pub fn with_join_separator(mut self, join_separator: impl Into<String>) -> Self {
        self.join_separator = join_separator.into();
        self
    }
}

// ── SkeletonOutput ──────────────────────────────────────────────────────────────

/// The full record of a Skeleton-of-Thought run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkeletonOutput {
    /// Resolved skeleton points, in skeleton order.
    pub points: Vec<SkeletonPoint>,
    /// The final answer: every point's content joined by the configured
    /// separator.
    pub answer: String,
}

// ── SotError ────────────────────────────────────────────────────────────────────

/// Errors from the `skeleton_of_thought` module.
#[derive(Debug, Error)]
pub enum SotError {
    /// The query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The generator produced no usable skeleton points.
    #[error("skeleton is empty")]
    EmptySkeleton,
}
