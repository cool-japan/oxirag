//! Types and traits for the `analogical_prompting` module.
//!
//! Analogical Prompting (Yasunaga et al. 2023) improves a model's reasoning by
//! having it **self-generate** relevant exemplars — similar problems it has
//! already solved — and optionally a high-level tutorial of the knowledge needed,
//! *before* it attempts the target problem. The model then solves the problem
//! conditioned on the exemplars and knowledge it produced for itself. The
//! prompting scaffold mirrors the paper:
//!
//! ```text
//! # Recall relevant exemplars:
//! Problem: <p1>
//! Solution: <s1>
//! Problem: <p2>
//! Solution: <s2>
//!
//! # Relevant knowledge:
//! <tutorial>
//!
//! # Solve the initial problem:
//! <problem>  ->  <answer>
//! ```
//!
//! Unlike `prompt_optimization`, which **selects** demonstrations from a fixed
//! pool of human-authored examples, Analogical Prompting asks the model to
//! *invent* its own exemplars on demand — no pool is required.
//!
//! The model is supplied by the caller through the [`AnalogicalModel`] trait. A
//! deterministic [`MockAnalogicalModel`] is provided for testing.

use thiserror::Error;

// ── Exemplar ──────────────────────────────────────────────────────────────────

/// A single self-generated exemplar: a related problem and its worked solution.
///
/// Exemplars are produced by [`AnalogicalModel::generate_exemplars`] and then fed
/// back into [`AnalogicalModel::solve`] as in-context demonstrations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exemplar {
    /// The related problem the model recalled.
    pub problem: String,
    /// The worked solution to [`Exemplar::problem`].
    pub solution: String,
}

impl Exemplar {
    /// Create a new exemplar from a problem and its solution.
    #[must_use]
    pub fn new(problem: impl Into<String>, solution: impl Into<String>) -> Self {
        Self {
            problem: problem.into(),
            solution: solution.into(),
        }
    }
}

// ── AnalogicalConfig ──────────────────────────────────────────────────────────

/// Configuration for
/// [`AnalogicalEngine`](crate::analogical_prompting::engine::AnalogicalEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalogicalConfig {
    /// Maximum number of exemplars to self-generate before solving.
    ///
    /// Defaults to `3`. The engine requests at most this many exemplars from the
    /// model and never forwards more than this many into the solve step.
    pub num_exemplars: usize,
    /// Whether to self-generate high-level knowledge in addition to exemplars.
    ///
    /// Defaults to `true`. When `false`, the engine skips
    /// [`AnalogicalModel::generate_knowledge`] entirely and passes an empty
    /// knowledge string to [`AnalogicalModel::solve`].
    pub use_knowledge: bool,
}

impl Default for AnalogicalConfig {
    fn default() -> Self {
        Self {
            num_exemplars: 3,
            use_knowledge: true,
        }
    }
}

impl AnalogicalConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of self-generated exemplars.
    #[must_use]
    pub fn with_num_exemplars(mut self, num_exemplars: usize) -> Self {
        self.num_exemplars = num_exemplars;
        self
    }

    /// Set whether high-level knowledge is self-generated.
    #[must_use]
    pub fn with_use_knowledge(mut self, use_knowledge: bool) -> Self {
        self.use_knowledge = use_knowledge;
        self
    }
}

// ── AnalogicalOutput ──────────────────────────────────────────────────────────

/// The full record of an Analogical Prompting run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalogicalOutput {
    /// The exemplars the model self-generated, in the order produced.
    pub exemplars: Vec<Exemplar>,
    /// The high-level knowledge tutorial, or empty when knowledge was disabled.
    pub knowledge: String,
    /// The final answer produced for the target problem.
    pub answer: String,
}

// ── AnalogicalModel ───────────────────────────────────────────────────────────

/// A language model that self-generates exemplars and knowledge, then solves.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete model (e.g. wrapping an LLM); [`MockAnalogicalModel`] is provided for
/// tests.
pub trait AnalogicalModel {
    /// Self-generate up to `n` relevant exemplars for `problem`.
    ///
    /// Implementations should return at most `n` exemplars; the engine also
    /// truncates the result to the configured limit as a safeguard.
    fn generate_exemplars(&self, problem: &str, n: usize) -> Vec<Exemplar>;

    /// Generate high-level knowledge / a tutorial for `problem`.
    fn generate_knowledge(&self, problem: &str) -> String;

    /// Solve `problem`, conditioned on the generated `exemplars` and `knowledge`.
    ///
    /// When knowledge generation is disabled, `knowledge` is the empty string.
    fn solve(&self, problem: &str, exemplars: &[Exemplar], knowledge: &str) -> String;
}

// ── MockAnalogicalModel ───────────────────────────────────────────────────────

/// Deterministic [`AnalogicalModel`] for tests.
///
/// [`AnalogicalModel::generate_exemplars`] returns the first `n` of its scripted
/// `exemplars`. [`AnalogicalModel::generate_knowledge`] returns the scripted
/// `knowledge`. [`AnalogicalModel::solve`] returns the scripted `answer`,
/// ignoring its arguments. To assert on what the engine forwarded into `solve`,
/// use a recording wrapper in the test instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockAnalogicalModel {
    /// Pre-scripted exemplars, truncated to the requested count.
    pub exemplars: Vec<Exemplar>,
    /// Pre-scripted high-level knowledge tutorial.
    pub knowledge: String,
    /// Pre-scripted final answer returned by [`AnalogicalModel::solve`].
    pub answer: String,
}

impl MockAnalogicalModel {
    /// Create a mock model from scripted exemplars, knowledge, and an answer.
    #[must_use]
    pub fn new(
        exemplars: Vec<Exemplar>,
        knowledge: impl Into<String>,
        answer: impl Into<String>,
    ) -> Self {
        Self {
            exemplars,
            knowledge: knowledge.into(),
            answer: answer.into(),
        }
    }
}

impl AnalogicalModel for MockAnalogicalModel {
    fn generate_exemplars(&self, _problem: &str, n: usize) -> Vec<Exemplar> {
        self.exemplars.iter().take(n).cloned().collect()
    }

    fn generate_knowledge(&self, _problem: &str) -> String {
        self.knowledge.clone()
    }

    fn solve(&self, _problem: &str, _exemplars: &[Exemplar], _knowledge: &str) -> String {
        self.answer.clone()
    }
}

// ── AnalogicalError ───────────────────────────────────────────────────────────

/// Errors from the `analogical_prompting` module.
#[derive(Debug, Error)]
pub enum AnalogicalError {
    /// The target problem was empty after trimming.
    #[error("problem must not be empty")]
    EmptyProblem,
}
