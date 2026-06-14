//! Types and traits for the `self_discover` module.
//!
//! Self-Discover (Zhou et al. 2024) is a meta-reasoning framework that proceeds
//! in three sequential stages before answering a query:
//!
//! 1. **SELECT** — the model picks the most relevant *atomic reasoning modules*
//!    from a built-in bank (e.g., `"critical_thinking"`, `"decomposition"`,
//!    `"analogy"`).
//! 2. **ADAPT** — each selected module is rephrased to suit the specific task.
//! 3. **IMPLEMENT** — the adapted modules are assembled into a structured
//!    reasoning plan and executed to produce the final answer.
//!
//! All three stages are simulated deterministically in pure Rust via the
//! [`SelfDiscoverModel`] trait. A drop-in [`MockSelfDiscoverModel`] is provided
//! for tests and benchmarks.

use std::collections::HashSet;

use thiserror::Error;

// ── Module-name constants ─────────────────────────────────────────────────────

/// Built-in module name: critical thinking.
pub const MODULE_CRITICAL_THINKING: &str = "critical_thinking";

/// Built-in module name: deductive reasoning.
pub const MODULE_DEDUCTIVE_REASONING: &str = "deductive_reasoning";

/// Built-in module name: analogy.
pub const MODULE_ANALOGY: &str = "analogy";

/// Built-in module name: causal reasoning.
pub const MODULE_CAUSAL_REASONING: &str = "causal_reasoning";

/// Built-in module name: step-by-step.
pub const MODULE_STEP_BY_STEP: &str = "step_by_step";

/// Built-in module name: comparative analysis.
pub const MODULE_COMPARATIVE_ANALYSIS: &str = "comparative_analysis";

/// Built-in module name: hypothesis testing.
pub const MODULE_HYPOTHESIS_TESTING: &str = "hypothesis_testing";

/// Built-in module name: decomposition.
pub const MODULE_DECOMPOSITION: &str = "decomposition";

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokenize `text` into lower-cased alphanumeric tokens of length ≥ 2.
///
/// The split boundary is any non-alphanumeric character. Tokens shorter than
/// two characters are discarded.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
}

// ── ReasoningModule ───────────────────────────────────────────────────────────

/// An atomic reasoning capability in the Self-Discover module bank.
///
/// Each module has a short `name` identifier (e.g., `"critical_thinking"`) and
/// a natural-language `description` that explains its reasoning approach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningModule {
    /// Short identifier for this module (e.g., `"critical_thinking"`).
    pub name: String,
    /// Natural-language description of the reasoning approach.
    pub description: String,
}

impl ReasoningModule {
    /// Create a new reasoning module from a name and description.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

// ── ReasoningStructure ────────────────────────────────────────────────────────

/// An adapted reasoning plan produced by the ADAPT + IMPLEMENT stages.
///
/// `modules` contains the reasoning modules active in this plan, and `plan`
/// lists the ordered steps the model will execute when answering the query.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningStructure {
    /// The reasoning modules active in this plan.
    pub modules: Vec<ReasoningModule>,
    /// Ordered plan steps, one per selected module (or more, as the model
    /// chooses).
    pub plan: Vec<String>,
}

// ── SelfDiscoverConfig ────────────────────────────────────────────────────────

/// Configuration for
/// [`SelfDiscoverEngine`](crate::self_discover::engine::SelfDiscoverEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfDiscoverConfig {
    /// Maximum number of reasoning modules selected in the SELECT stage.
    ///
    /// Defaults to `3`.
    pub max_modules: usize,

    /// Maximum number of retrieved documents forwarded to ADAPT + IMPLEMENT.
    ///
    /// Defaults to `5`.
    pub top_k_docs: usize,
}

impl Default for SelfDiscoverConfig {
    fn default() -> Self {
        Self {
            max_modules: 3,
            top_k_docs: 5,
        }
    }
}

impl SelfDiscoverConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of reasoning modules to SELECT.
    #[must_use]
    pub fn with_max_modules(mut self, max_modules: usize) -> Self {
        self.max_modules = max_modules;
        self
    }

    /// Set the maximum number of retrieved documents passed to the IMPLEMENT
    /// stage.
    #[must_use]
    pub fn with_top_k_docs(mut self, top_k_docs: usize) -> Self {
        self.top_k_docs = top_k_docs;
        self
    }
}

// ── SelfDiscoverResult ────────────────────────────────────────────────────────

/// The complete output of one Self-Discover run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfDiscoverResult {
    /// The original query passed to the engine.
    pub query: String,
    /// The modules chosen in the SELECT stage (at most `max_modules`).
    pub selected_modules: Vec<ReasoningModule>,
    /// The structured reasoning plan produced by ADAPT + IMPLEMENT.
    pub reasoning_structure: ReasoningStructure,
    /// The final answer produced by the IMPLEMENT stage.
    pub answer: String,
}

// ── SelfDiscoverError ─────────────────────────────────────────────────────────

/// Errors that can arise during a Self-Discover run.
#[derive(Debug, Error)]
pub enum SelfDiscoverError {
    /// The SELECT stage failed.
    #[error("selection failed: {0}")]
    SelectionFailed(String),

    /// The ADAPT stage failed.
    #[error("adaptation failed: {0}")]
    AdaptationFailed(String),

    /// The IMPLEMENT stage failed.
    #[error("implementation failed: {0}")]
    ImplementationFailed(String),
}

// ── SelfDiscoverModel trait ───────────────────────────────────────────────────

/// The model interface driving Self-Discover's three stages.
///
/// Implementations are **pure-sync** — no I/O, no async. The engine calls
/// [`SelfDiscoverModel::select`] first to rank the module bank, then calls
/// [`SelfDiscoverModel::adapt_and_implement`] to produce the reasoning
/// structure and final answer.
///
/// The built-in [`MockSelfDiscoverModel`] provides a deterministic
/// implementation suitable for tests.
pub trait SelfDiscoverModel {
    /// SELECT stage: rank the module bank for `query`.
    ///
    /// Returns indices into `modules`, sorted best-first. The engine truncates
    /// the result to [`SelfDiscoverConfig::max_modules`].
    ///
    /// # Errors
    ///
    /// Return [`SelfDiscoverError::SelectionFailed`] if selection cannot be
    /// completed.
    fn select(
        &self,
        query: &str,
        modules: &[ReasoningModule],
    ) -> Result<Vec<usize>, SelfDiscoverError>;

    /// ADAPT + IMPLEMENT stage: build a reasoning structure and produce an
    /// answer.
    ///
    /// `modules` contains only the *selected* modules (already truncated by
    /// the engine to `max_modules`). `docs` contains the retrieved documents
    /// (already limited to `top_k_docs`).
    ///
    /// Returns a tuple of the adapted [`ReasoningStructure`] and the final
    /// answer string.
    ///
    /// # Errors
    ///
    /// Return [`SelfDiscoverError::AdaptationFailed`] or
    /// [`SelfDiscoverError::ImplementationFailed`] on failure.
    fn adapt_and_implement(
        &self,
        query: &str,
        modules: &[ReasoningModule],
        docs: &[String],
    ) -> Result<(ReasoningStructure, String), SelfDiscoverError>;
}

// ── MockSelfDiscoverModel ─────────────────────────────────────────────────────

/// Deterministic [`SelfDiscoverModel`] for tests and benchmarks.
///
/// **SELECT**: ranks every module in the bank by the token overlap between the
/// query and the concatenation of `name + " " + description` for each module.
/// Ties are broken by the original bank index (lowest index first). All
/// indices are returned sorted best-first; the engine applies the
/// `max_modules` cap.
///
/// **ADAPT**: produces one plan step per module: `"Apply {name}: {description}"`.
///
/// **IMPLEMENT**: joins `docs` with a single space to form the answer. An
/// empty `docs` slice produces an empty answer string.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockSelfDiscoverModel;

impl SelfDiscoverModel for MockSelfDiscoverModel {
    fn select(
        &self,
        query: &str,
        modules: &[ReasoningModule],
    ) -> Result<Vec<usize>, SelfDiscoverError> {
        let query_tokens: HashSet<String> = tokenize(query).collect();

        let mut scored: Vec<(usize, usize)> = modules
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let module_text = format!("{} {}", m.name, m.description);
                let module_tokens: HashSet<String> = tokenize(&module_text).collect();
                let overlap = query_tokens.intersection(&module_tokens).count();
                (idx, overlap)
            })
            .collect();

        // Higher overlap first; tie-break by original index (lower wins).
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let indices: Vec<usize> = scored.into_iter().map(|(idx, _)| idx).collect();
        Ok(indices)
    }

    fn adapt_and_implement(
        &self,
        _query: &str,
        modules: &[ReasoningModule],
        docs: &[String],
    ) -> Result<(ReasoningStructure, String), SelfDiscoverError> {
        let plan: Vec<String> = modules
            .iter()
            .map(|m| format!("Apply {}: {}", m.name, m.description))
            .collect();

        let structure = ReasoningStructure {
            modules: modules.to_vec(),
            plan,
        };

        let answer = docs.join(" ");

        Ok((structure, answer))
    }
}
