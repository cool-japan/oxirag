//! Core data types for the `rewoo` module.
//!
//! Holds the plan-shape types ([`PlaceholderVar`], [`RewooAction`],
//! [`RewooStep`], [`RewooPlan`]), the resolved-evidence container
//! ([`RewooEvidence`]), the end-to-end result ([`RewooOutcome`]), the shared
//! configuration ([`RewooConfig`]), and the error enum ([`RewooError`]).
//! Pluggable traits and their `Mock*` implementations, plus the
//! planner/worker/solver orchestration logic, live in
//! [`super::engine`].

use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

// ── PlaceholderVar ────────────────────────────────────────────────────────────

/// A named evidence placeholder produced by exactly one plan step and
/// consumed by zero or more later steps, rendered as `#E<n>` (e.g. `#E1`,
/// `#E2`, ..., `#E10`).
///
/// `PlaceholderVar` is the unit `ReWOO`'s plan-time/execution-time decoupling
/// hinges on: the planner emits `#E<n>` tokens standing in for evidence that
/// does not exist yet, and [`super::engine::RewooWorker`] resolves them one at
/// a time, in plan order, without ever re-invoking the planner. Ordering
/// ([`Ord`]) is by the wrapped numeric index, *not* lexicographic string
/// order, so `#E2 < #E10` holds as expected (a plain string comparison would
/// instead put `"#E10"` before `"#E2"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlaceholderVar(usize);

impl PlaceholderVar {
    /// Creates the placeholder variable `#E<n>` for index `n`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self(n)
    }

    /// Returns the numeric index of this placeholder (the `n` in `#E<n>`).
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }

    /// Parses a placeholder token of the exact form `#E<digits>` (e.g. `#E1`,
    /// `#E10`).
    ///
    /// Returns `None` unless `token` is *precisely* one such token: a bare
    /// `#E` prefix followed by one or more ASCII digits and nothing else.
    /// Lowercase `#e1`, a missing digit run (`#E`), or trailing garbage
    /// (`#E1x`) all fail to parse.
    #[must_use]
    pub fn parse_token(token: &str) -> Option<Self> {
        let digits = token.strip_prefix("#E")?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        digits.parse::<usize>().ok().map(Self)
    }
}

impl fmt::Display for PlaceholderVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#E{}", self.0)
    }
}

// ── RewooAction ───────────────────────────────────────────────────────────────

/// An action a [`RewooStep`] executes to resolve its evidence placeholder.
///
/// Both variants carry raw, *unsubstituted* text as produced by the planner:
/// the text may reference earlier placeholders (e.g.
/// `"search for #E1's birthplace"`), which [`super::engine::RewooWorker`]
/// substitutes before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewooAction {
    /// Retrieve evidence for a query via a [`super::engine::RewooRetriever`].
    Search(String),
    /// Evaluate an arithmetic expression (see [`super::engine`] for the
    /// supported grammar: `+ - * /`, parentheses, unary sign, decimals).
    Compute(String),
}

impl RewooAction {
    /// Returns the raw, unsubstituted text carried by this action (the search
    /// query or compute expression, before any placeholder substitution).
    #[must_use]
    pub fn raw_text(&self) -> &str {
        match self {
            Self::Search(text) | Self::Compute(text) => text,
        }
    }
}

// ── RewooStep ─────────────────────────────────────────────────────────────────

/// A single step of a [`RewooPlan`]: a rationale, an action to resolve it,
/// and the placeholder variable the resolved evidence is bound to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewooStep {
    /// Natural-language description of why this step is needed — the
    /// planner's rationale for this piece of the overall plan.
    pub reasoning: String,
    /// The action to execute to resolve `evidence_var`.
    pub action: RewooAction,
    /// The placeholder variable this step's resolved evidence is bound to.
    pub evidence_var: PlaceholderVar,
}

impl RewooStep {
    /// Creates a new plan step.
    #[must_use]
    pub fn new(
        reasoning: impl Into<String>,
        action: RewooAction,
        evidence_var: PlaceholderVar,
    ) -> Self {
        Self {
            reasoning: reasoning.into(),
            action,
            evidence_var,
        }
    }
}

// ── RewooPlan ─────────────────────────────────────────────────────────────────

/// A complete plan: an ordered sequence of [`RewooStep`]s produced in a
/// single upfront pass by a [`super::engine::RewooPlanSource`], before any
/// evidence has been gathered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewooPlan {
    /// The original query this plan was produced for.
    pub query: String,
    /// The ordered plan steps. Execution order is array order — a step may
    /// only reference placeholders defined by steps at a strictly smaller
    /// index.
    pub steps: Vec<RewooStep>,
}

impl RewooPlan {
    /// Creates a new plan for `query` from an ordered list of `steps`.
    #[must_use]
    pub fn new(query: impl Into<String>, steps: Vec<RewooStep>) -> Self {
        Self {
            query: query.into(),
            steps,
        }
    }

    /// Returns the number of steps in this plan.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if this plan has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Appends `step`, returning `self` for chaining.
    #[must_use]
    pub fn with_step(mut self, step: RewooStep) -> Self {
        self.steps.push(step);
        self
    }
}

// ── RewooEvidence ─────────────────────────────────────────────────────────────

/// The resolved `placeholder -> evidence text` mapping accumulated by
/// [`super::engine::RewooWorker`] as it walks a [`RewooPlan`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RewooEvidence {
    resolved: BTreeMap<PlaceholderVar, String>,
}

impl RewooEvidence {
    /// Creates an empty evidence map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the resolved evidence text for `var`, overwriting any prior
    /// value bound to it.
    pub fn insert(&mut self, var: PlaceholderVar, text: impl Into<String>) {
        self.resolved.insert(var, text.into());
    }

    /// Returns the resolved evidence text for `var`, if any.
    #[must_use]
    pub fn get(&self, var: PlaceholderVar) -> Option<&str> {
        self.resolved.get(&var).map(String::as_str)
    }

    /// Returns the number of resolved placeholders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.resolved.len()
    }

    /// Returns `true` if no placeholders have been resolved yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.resolved.is_empty()
    }

    /// Iterates the resolved `(placeholder, evidence text)` pairs in
    /// ascending placeholder order.
    pub fn iter(&self) -> impl Iterator<Item = (PlaceholderVar, &str)> {
        self.resolved
            .iter()
            .map(|(var, text)| (*var, text.as_str()))
    }
}

// ── RewooOutcome ──────────────────────────────────────────────────────────────

/// The full result of running the `ReWOO` plan → resolve → solve pipeline for
/// a query (see [`super::engine::RewooPipeline`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewooOutcome {
    /// The original, trimmed query.
    pub query: String,
    /// The plan produced by the planner (generated once, upfront).
    pub plan: RewooPlan,
    /// The evidence resolved by the worker for every step of `plan`.
    pub evidence: RewooEvidence,
    /// The final answer synthesized by the solver.
    pub answer: String,
}

// ── RewooConfig ───────────────────────────────────────────────────────────────

/// Shared configuration for [`super::engine::RewooPlanner`],
/// [`super::engine::RewooWorker`], [`super::engine::RewooSolver`], and
/// [`super::engine::RewooPipeline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewooConfig {
    /// Maximum number of steps a plan may contain before it is rejected.
    /// Guards against runaway or malformed plans. Default: `16`.
    pub max_steps: usize,
    /// Whether [`super::engine::RewooPlanner::plan`] eagerly validates, once
    /// and up front, that every placeholder referenced by a step's action
    /// refers to a strictly earlier step. When `false`, a malformed forward
    /// reference is only caught later, at worker execution time. Default:
    /// `true`.
    pub validate_plan_structure: bool,
}

impl Default for RewooConfig {
    fn default() -> Self {
        Self {
            max_steps: 16,
            validate_plan_structure: true,
        }
    }
}

impl RewooConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of steps a plan may contain.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets whether the planner eagerly validates plan structure.
    #[must_use]
    pub fn with_validate_plan_structure(mut self, validate_plan_structure: bool) -> Self {
        self.validate_plan_structure = validate_plan_structure;
        self
    }
}

// ── RewooError ────────────────────────────────────────────────────────────────

/// Errors produced by the `rewoo` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RewooError {
    /// The supplied query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,

    /// The plan has more steps than [`RewooConfig::max_steps`] allows.
    #[error("plan has {len} steps, exceeding the configured maximum of {max}")]
    PlanTooLong {
        /// The number of steps the plan actually has.
        len: usize,
        /// The configured maximum.
        max: usize,
    },

    /// Two steps of the same plan define the same evidence placeholder.
    #[error("placeholder {0} is defined by more than one step")]
    DuplicatePlaceholder(PlaceholderVar),

    /// A step's action or reasoning text references a placeholder that has
    /// not been resolved yet — either because it refers to a step at the same
    /// or a later index, or because no step defines it at all.
    #[error("placeholder {0} is referenced before it has been resolved")]
    UnresolvedPlaceholder(PlaceholderVar),

    /// Eager structural validation ([`RewooConfig::validate_plan_structure`])
    /// found a step whose action references a placeholder not defined by a
    /// strictly earlier step.
    #[error("step {step} references placeholder {var}, which is not defined by an earlier step")]
    ForwardReference {
        /// Zero-based index of the offending step within the plan.
        step: usize,
        /// The placeholder referenced too early.
        var: PlaceholderVar,
    },

    /// A [`RewooAction::Compute`] expression could not be evaluated.
    #[error("failed to evaluate compute expression {expression:?}: {reason}")]
    ComputeFailed {
        /// The (placeholder-substituted) expression that failed to evaluate.
        expression: String,
        /// A human-readable reason for the failure.
        reason: String,
    },
}
