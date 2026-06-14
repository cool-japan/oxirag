//! Types, traits, and mocks for the `graph_of_thought` module.
//!
//! Graph-of-Thoughts (Besta et al. 2023, "Graph of Thoughts: Solving Elaborate
//! Problems with Large Language Models") generalizes Tree-of-Thoughts: individual
//! thoughts are vertices of a directed acyclic graph and the reasoning process is
//! expressed as a sequence of graph *operations* over them. This module supports
//! four operations — [`GotOperation::Generate`], [`GotOperation::Aggregate`],
//! [`GotOperation::Refine`], and [`GotOperation::KeepBest`].
//!
//! The language model is modeled by three **pure-sync** caller-supplied traits —
//! [`ThoughtGenerator`], [`ThoughtScorer`], and [`ThoughtAggregator`] — with
//! deterministic [`MockThoughtGenerator`], [`MockThoughtScorer`], and
//! [`MockThoughtAggregator`] implementations provided for testing. There is no
//! async and no I/O.

use thiserror::Error;

// ── ThoughtGenerator ────────────────────────────────────────────────────────────

/// Source of candidate thoughts for a query and accumulated context.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete generator (e.g. wrapping an LLM); [`MockThoughtGenerator`] is
/// provided for tests.
pub trait ThoughtGenerator {
    /// Generate `k` candidate thoughts from a `query` plus `context`.
    ///
    /// `context` holds the contents of the frontier thoughts that the new
    /// thoughts branch from (it is empty for the very first generation). An
    /// implementation may return fewer than `k` items; the engine treats every
    /// returned string as a distinct candidate thought.
    fn generate(&self, query: &str, context: &[&str], k: usize) -> Vec<String>;
}

// ── ThoughtScorer ───────────────────────────────────────────────────────────────

/// Scores the quality of a single thought in `[0, 1]`.
///
/// Implementations are **pure sync**. Higher scores denote better thoughts;
/// the engine uses scores to rank the frontier and to select the best thought.
/// [`MockThoughtScorer`] is provided for tests.
pub trait ThoughtScorer {
    /// Score a `thought`'s quality with respect to `query`, in `[0, 1]`.
    fn score(&self, query: &str, thought: &str) -> f32;
}

// ── ThoughtAggregator ───────────────────────────────────────────────────────────

/// Merges several thoughts into one synthesized thought.
///
/// Implementations are **pure sync**. Used by [`GotOperation::Aggregate`] to
/// fuse the current frontier into a single new thought. [`MockThoughtAggregator`]
/// is provided for tests.
pub trait ThoughtAggregator {
    /// Merge `thoughts` into a single synthesized thought for `query`.
    fn aggregate(&self, query: &str, thoughts: &[&str]) -> String;
}

// ── MockThoughtGenerator ────────────────────────────────────────────────────────

/// Deterministic [`ThoughtGenerator`] for tests.
///
/// Produces `k` thoughts of the form `"{prefix} {n}"` where `prefix` is the
/// configured label and `n` is a 1-based counter. Because the output depends
/// only on `prefix` and `k`, repeated calls with the same arguments are
/// identical, making engine runs fully reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockThoughtGenerator {
    /// Label prepended to each generated thought.
    pub prefix: String,
}

impl MockThoughtGenerator {
    /// Create a mock generator that labels thoughts with `prefix`.
    #[must_use]
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }
}

impl Default for MockThoughtGenerator {
    fn default() -> Self {
        Self::new("thought")
    }
}

impl ThoughtGenerator for MockThoughtGenerator {
    fn generate(&self, _query: &str, _context: &[&str], k: usize) -> Vec<String> {
        (1..=k).map(|n| format!("{} {n}", self.prefix)).collect()
    }
}

// ── MockThoughtScorer ───────────────────────────────────────────────────────────

/// Deterministic [`ThoughtScorer`] for tests.
///
/// Scores a thought by `min(1, len_weight * trailing_number)`, where
/// `trailing_number` is the integer parsed from the thought's final
/// whitespace-separated token (or `0` when absent). This makes thoughts with a
/// larger trailing index score higher, so the highest-numbered mock thought is
/// deterministically "best".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MockThoughtScorer {
    /// Multiplier applied to a thought's trailing number before clamping.
    pub len_weight: f32,
}

impl MockThoughtScorer {
    /// Create a mock scorer with the given per-unit weight.
    #[must_use]
    pub fn new(len_weight: f32) -> Self {
        Self { len_weight }
    }
}

impl Default for MockThoughtScorer {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl ThoughtScorer for MockThoughtScorer {
    #[allow(clippy::cast_precision_loss)]
    fn score(&self, _query: &str, thought: &str) -> f32 {
        let trailing = trailing_number(thought) as f32;
        (self.len_weight * trailing).clamp(0.0, 1.0)
    }
}

// ── MockThoughtAggregator ───────────────────────────────────────────────────────

/// Deterministic [`ThoughtAggregator`] for tests.
///
/// Merges thoughts by joining them with the configured `separator`, prefixed by
/// `"merged: "`. The join order is the order in which thoughts are supplied, so
/// the result is fully deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockThoughtAggregator {
    /// String inserted between merged thoughts.
    pub separator: String,
}

impl MockThoughtAggregator {
    /// Create a mock aggregator joining thoughts with `separator`.
    #[must_use]
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
        }
    }
}

impl Default for MockThoughtAggregator {
    fn default() -> Self {
        Self::new(" + ")
    }
}

impl ThoughtAggregator for MockThoughtAggregator {
    fn aggregate(&self, _query: &str, thoughts: &[&str]) -> String {
        format!("merged: {}", thoughts.join(&self.separator))
    }
}

// ── GotOperation ────────────────────────────────────────────────────────────────

/// A single graph operation applied to the current frontier of thoughts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GotOperation {
    /// Branch `k` new thoughts from each frontier thought (or from scratch when
    /// the frontier is empty), score them, and make them the new frontier.
    Generate {
        /// Number of candidate thoughts to branch from the frontier.
        k: usize,
    },
    /// Merge the current frontier into a single synthesized thought, which
    /// becomes the new frontier.
    Aggregate,
    /// Regenerate and rescore the best frontier thought via a self-edge,
    /// replacing it in the frontier when the refinement scores at least as high.
    Refine,
    /// Prune the frontier to its top-`n` thoughts by score.
    KeepBest {
        /// Number of thoughts to retain.
        n: usize,
    },
}

impl GotOperation {
    /// Short human-readable label for the operation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Generate { .. } => "generate",
            Self::Aggregate => "aggregate",
            Self::Refine => "refine",
            Self::KeepBest { .. } => "keep_best",
        }
    }
}

// ── GotConfig ───────────────────────────────────────────────────────────────────

/// Configuration for
/// [`GraphOfThoughtEngine`](crate::graph_of_thought::engine::GraphOfThoughtEngine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GotConfig {
    /// Hard cap on the total number of nodes the graph may contain.
    ///
    /// Once the graph holds this many nodes, [`GotOperation::Generate`] stops
    /// adding further branches. Defaults to `32`.
    pub max_nodes: usize,
    /// Default branching factor used by [`GotOperation::Generate`] when an
    /// operation does not itself fix `k`. Defaults to `3`.
    pub branch_factor: usize,
}

impl Default for GotConfig {
    fn default() -> Self {
        Self {
            max_nodes: 32,
            branch_factor: 3,
        }
    }
}

impl GotConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of nodes the graph may hold.
    #[must_use]
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    /// Set the default branching factor for [`GotOperation::Generate`].
    #[must_use]
    pub fn with_branch_factor(mut self, branch_factor: usize) -> Self {
        self.branch_factor = branch_factor;
        self
    }
}

// ── GotOutput ───────────────────────────────────────────────────────────────────

/// Result of running a Graph-of-Thoughts plan.
#[derive(Debug, Clone, PartialEq)]
pub struct GotOutput {
    /// The full thought graph constructed during the run.
    pub graph: super::graph::ThoughtGraph,
    /// Content of the highest-scoring thought in the graph.
    pub best_thought: String,
    /// The final answer (equal to [`GotOutput::best_thought`]).
    pub final_answer: String,
}

// ── GotError ────────────────────────────────────────────────────────────────────

/// Errors from the `graph_of_thought` module.
#[derive(Debug, Error)]
pub enum GotError {
    /// The query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
    /// No thoughts were generated, so no answer could be produced.
    #[error("no thoughts generated")]
    NoThoughts,
}

// ── Helpers ─────────────────────────────────────────────────────────────────────

/// Parse the integer in `text`'s final whitespace-separated token, or `0`.
///
/// Used by [`MockThoughtScorer`] to derive a deterministic score from a thought
/// such as `"thought 3"`.
pub(crate) fn trailing_number(text: &str) -> u64 {
    text.split_whitespace()
        .next_back()
        .and_then(|token| token.parse::<u64>().ok())
        .unwrap_or(0)
}
