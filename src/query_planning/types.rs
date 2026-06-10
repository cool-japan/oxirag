//! Types for the `query_planning` module.

use thiserror::Error;

// ── PlanStepKind ──────────────────────────────────────────────────────────────

/// The kind of operation a plan step performs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanStepKind {
    /// Retrieve documents relevant to the step query.
    Retrieve,
    /// Filter a previous step's results by the step query terms.
    Filter,
    /// Aggregate answers from dependency steps.
    Aggregate,
    /// Compare answers from exactly two dependency steps.
    Compare,
    /// Verify the conclusion drawn in a dependency step.
    Verify,
}

impl PlanStepKind {
    /// Return the canonical string label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Retrieve => "retrieve",
            Self::Filter => "filter",
            Self::Aggregate => "aggregate",
            Self::Compare => "compare",
            Self::Verify => "verify",
        }
    }
}

// ── PlanStep ──────────────────────────────────────────────────────────────────

/// A single node in the query execution DAG.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Unique identifier within the plan (zero-based).
    pub id: usize,
    /// The operation this step performs.
    pub kind: PlanStepKind,
    /// The query text used by this step.
    pub query: String,
    /// IDs of steps that must complete before this one.
    pub depends_on: Vec<usize>,
}

impl PlanStep {
    /// Create a new [`PlanStep`] with no dependencies.
    #[must_use]
    pub fn new(id: usize, kind: PlanStepKind, query: impl Into<String>) -> Self {
        Self {
            id,
            kind,
            query: query.into(),
            depends_on: Vec::new(),
        }
    }

    /// Attach dependency step IDs.
    #[must_use]
    pub fn with_depends_on(mut self, ids: Vec<usize>) -> Self {
        self.depends_on = ids;
        self
    }

    /// Return `true` when this step has no dependencies (root node).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.depends_on.is_empty()
    }
}

// ── SynthesisStrategy ─────────────────────────────────────────────────────────

/// Strategy used to combine the answers from all plan steps into a final answer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SynthesisStrategy {
    /// Return the single retrieved answer directly.
    #[default]
    Simple,
    /// Juxtapose two answers side-by-side for comparison.
    Comparative,
    /// Accumulate and merge multiple answers.
    Aggregative,
}

impl SynthesisStrategy {
    /// Return the canonical string label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Simple => "simple",
            Self::Comparative => "comparative",
            Self::Aggregative => "aggregative",
        }
    }
}

// ── QueryPlan ─────────────────────────────────────────────────────────────────

/// A DAG of [`PlanStep`]s that together answer the original query.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Ordered list of steps (topologically consistent after planning).
    pub steps: Vec<PlanStep>,
    /// Strategy for combining the step results.
    pub synthesis_strategy: SynthesisStrategy,
}

impl QueryPlan {
    /// Create a new [`QueryPlan`].
    #[must_use]
    pub fn new(steps: Vec<PlanStep>, synthesis_strategy: SynthesisStrategy) -> Self {
        Self {
            steps,
            synthesis_strategy,
        }
    }

    /// Return `true` when the plan has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return references to all root steps (no dependencies).
    #[must_use]
    pub fn root_steps(&self) -> Vec<&PlanStep> {
        self.steps.iter().filter(|s| s.is_root()).collect()
    }

    /// Return the total number of steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the maximum parallelism: the count of independent root steps.
    #[must_use]
    pub fn max_parallelism(&self) -> usize {
        self.root_steps().len()
    }
}

// ── PlanResult ────────────────────────────────────────────────────────────────

/// The output produced by executing a single [`PlanStep`].
#[derive(Debug, Clone)]
pub struct PlanResult {
    /// The ID of the step that produced this result.
    pub step_id: usize,
    /// The synthesised answer text.
    pub answer: String,
    /// Source document IDs or snippets used to produce the answer.
    pub sources: Vec<String>,
}

impl PlanResult {
    /// Create a new [`PlanResult`] with no sources.
    #[must_use]
    pub fn new(step_id: usize, answer: impl Into<String>) -> Self {
        Self {
            step_id,
            answer: answer.into(),
            sources: Vec::new(),
        }
    }

    /// Attach source references.
    #[must_use]
    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = sources;
        self
    }
}

// ── QueryPlanningError ────────────────────────────────────────────────────────

/// Errors from the `query_planning` module.
#[derive(Debug, Error)]
pub enum QueryPlanningError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// A cyclic dependency was detected in the plan DAG.
    #[error("Cyclic dependency detected in plan")]
    CyclicDependency,
    /// A step failed during execution.
    #[error("Step execution failed: {0}")]
    StepFailed(String),
}
