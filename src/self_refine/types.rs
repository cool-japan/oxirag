//! Types and traits for the `self_refine` module.
//!
//! Self-Refine (Madaan et al. 2023) improves a model's output through
//! **iterative self-feedback**: the *same* model first produces an initial
//! output, then critiques that output, then refines it using its own critique.
//! Feedback → refine repeats until the feedback signals the output is good
//! enough, a score threshold is met, or an iteration cap is reached.
//!
//! ```text
//! initial   →  output_0
//! feedback  →  critique_0, score_0           (stop? threshold?)
//! refine    →  output_1
//! feedback  →  critique_1, score_1           (stop? threshold?)
//! ...
//! ```
//!
//! Unlike `reflexion` — which retrieves documents, scores attempts with a
//! *separate* evaluator, and accumulates an episodic memory across retries —
//! Self-Refine uses **no retrieval and no memory**: a single model performs all
//! three roles (generate, critique, refine) on its own previous output.
//!
//! The model is supplied by the caller through the [`Refiner`] trait. A
//! deterministic [`MockRefiner`] is provided for testing.

use thiserror::Error;

// ── Feedback ────────────────────────────────────────────────────────────────

/// The model's critique of one output, produced by [`Refiner::feedback`].
#[derive(Debug, Clone, PartialEq)]
pub struct Feedback {
    /// The natural-language critique describing how to improve the output.
    pub critique: String,
    /// Quality score for the output, conventionally in `[0.0, 1.0]`.
    ///
    /// The engine stops refining once this reaches the configured
    /// `score_threshold`.
    pub score: f32,
    /// Whether the model considers the output good enough to stop refining.
    ///
    /// When `true`, the engine stops immediately regardless of `score`.
    pub stop: bool,
}

impl Feedback {
    /// Create a new feedback from a critique, score, and stop flag.
    #[must_use]
    pub fn new(critique: impl Into<String>, score: f32, stop: bool) -> Self {
        Self {
            critique: critique.into(),
            score,
            stop,
        }
    }
}

// ── RefineStep ──────────────────────────────────────────────────────────────

/// One iteration of the Self-Refine loop: an output and the feedback on it.
///
/// The [`RefineStep::output`] is the output that was critiqued, and
/// [`RefineStep::feedback`] is the critique the model produced for it. The
/// refinement derived from this feedback (if any) appears as the
/// [`RefineStep::output`] of the *next* step.
#[derive(Debug, Clone, PartialEq)]
pub struct RefineStep {
    /// The output that was critiqued in this iteration.
    pub output: String,
    /// The feedback the model produced for [`RefineStep::output`].
    pub feedback: Feedback,
}

// ── SelfRefineConfig ──────────────────────────────────────────────────────────

/// Configuration for
/// [`SelfRefineEngine`](crate::self_refine::engine::SelfRefineEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct SelfRefineConfig {
    /// Maximum number of feedback→refine iterations.
    ///
    /// Defaults to `4`. The loop produces at most this many [`RefineStep`]s,
    /// even if the model never signals completion.
    pub max_iterations: usize,
    /// Score at or above which refinement stops early.
    ///
    /// Defaults to `0.9`. When a feedback's
    /// [`score`](Feedback::score) reaches this value, the loop stops without
    /// refining further.
    pub score_threshold: f32,
}

impl Default for SelfRefineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            score_threshold: 0.9,
        }
    }
}

impl SelfRefineConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of feedback→refine iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the score threshold at or above which refinement stops early.
    #[must_use]
    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = score_threshold;
        self
    }
}

// ── SelfRefineOutput ──────────────────────────────────────────────────────────

/// The full record of a Self-Refine run.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfRefineOutput {
    /// The output of the final iteration — the best refinement produced.
    pub final_output: String,
    /// Every iteration's output paired with the feedback on it, in order.
    pub steps: Vec<RefineStep>,
    /// Number of feedback→refine iterations performed (equal to `steps.len()`).
    pub iterations: usize,
}

// ── Refiner ───────────────────────────────────────────────────────────────────

/// A model that generates, critiques, and refines its own output.
///
/// All three methods are driven by the **same** underlying model — the defining
/// characteristic of Self-Refine. Implementations are **pure sync** — no I/O, no
/// async. The caller supplies a concrete refiner (e.g. wrapping an LLM);
/// [`MockRefiner`] is provided for tests.
pub trait Refiner {
    /// Produce the initial output for the task.
    fn initial(&self, task: &str) -> String;

    /// Critique the current output.
    fn feedback(&self, task: &str, output: &str) -> Feedback;

    /// Improve the output using the feedback.
    fn refine(&self, task: &str, output: &str, feedback: &Feedback) -> String;
}

// ── MockRefiner ───────────────────────────────────────────────────────────────

/// Deterministic [`Refiner`] for tests.
///
/// Scripts the loop with a fixed initial output and a per-iteration sequence of
/// `(Feedback, refinement)` pairs:
///
/// * [`Refiner::initial`] always returns the configured `initial` output.
/// * [`Refiner::feedback`] returns the [`Feedback`] from `steps[iteration]`,
///   where `iteration` is inferred from how many feedbacks have been requested.
///   Because the engine drives the loop in lock-step, the iteration is derived
///   from the `output` argument: the initial output maps to index `0`, and each
///   scripted refinement maps to the next index. When the script is exhausted,
///   a terminal `Feedback { score: 1.0, stop: true }` is returned so callers
///   that out-run the script still terminate.
/// * [`Refiner::refine`] returns the refinement string from `steps[iteration]`.
///
/// This makes the mock fully deterministic and independent of any hidden
/// mutable state.
#[derive(Debug, Clone, PartialEq)]
pub struct MockRefiner {
    /// The initial output returned by [`Refiner::initial`].
    pub initial: String,
    /// Per-iteration `(feedback, refinement)` script, consumed by index.
    pub steps: Vec<(Feedback, String)>,
}

impl MockRefiner {
    /// Create a mock refiner from an initial output and a scripted sequence of
    /// `(feedback, refinement)` pairs.
    #[must_use]
    pub fn new(initial: impl Into<String>, steps: Vec<(Feedback, String)>) -> Self {
        Self {
            initial: initial.into(),
            steps,
        }
    }

    /// Create a mock refiner whose initial output is already good enough.
    ///
    /// [`Refiner::feedback`] immediately returns a stopping [`Feedback`] with the
    /// given `score`, so the engine performs a single iteration and never
    /// refines.
    #[must_use]
    pub fn good_enough(initial: impl Into<String>, score: f32) -> Self {
        Self {
            initial: initial.into(),
            steps: vec![(Feedback::new("looks good", score, true), String::new())],
        }
    }

    /// Resolve the scripted iteration index for `output`.
    ///
    /// The initial output maps to `0`; the refinement produced by step `i` maps
    /// to `i + 1`. Anything unrecognized falls back to the number of scripted
    /// steps (past the end), which yields the terminal feedback.
    fn index_for(&self, output: &str) -> usize {
        if output == self.initial {
            return 0;
        }
        for (idx, (_, refinement)) in self.steps.iter().enumerate() {
            if output == refinement.as_str() {
                return idx + 1;
            }
        }
        self.steps.len()
    }
}

impl Refiner for MockRefiner {
    fn initial(&self, _task: &str) -> String {
        self.initial.clone()
    }

    fn feedback(&self, _task: &str, output: &str) -> Feedback {
        let idx = self.index_for(output);
        self.steps.get(idx).map_or_else(
            || Feedback::new("no further feedback", 1.0, true),
            |(fb, _)| fb.clone(),
        )
    }

    fn refine(&self, _task: &str, output: &str, _feedback: &Feedback) -> String {
        let idx = self.index_for(output);
        self.steps
            .get(idx)
            .map_or_else(|| output.to_string(), |(_, refinement)| refinement.clone())
    }
}

// ── SelfRefineError ────────────────────────────────────────────────────────────

/// Errors from the `self_refine` module.
#[derive(Debug, Error)]
pub enum SelfRefineError {
    /// The task description was empty after trimming.
    #[error("task must not be empty")]
    EmptyTask,
}
