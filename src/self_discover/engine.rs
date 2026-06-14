//! [`SelfDiscoverEngine`] — the Self-Discover pipeline driver.
//!
//! The engine wires together the three stages of Self-Discover
//! (Zhou et al. 2024): SELECT → ADAPT → IMPLEMENT. It owns the configuration
//! and the model, and exposes a single [`SelfDiscoverEngine::run`] entry point.

use crate::self_discover::types::{
    ReasoningModule, SelfDiscoverConfig, SelfDiscoverError, SelfDiscoverModel, SelfDiscoverResult,
};

// ── Built-in module bank ──────────────────────────────────────────────────────

/// Static data for the built-in reasoning module bank: `(name, description)`.
static BUILT_IN_DATA: &[(&str, &str)] = &[
    (
        "critical_thinking",
        "Evaluate arguments carefully, identify assumptions, assess evidence, \
         and challenge conclusions through rigorous logical analysis.",
    ),
    (
        "deductive_reasoning",
        "Derive specific conclusions from general principles using valid \
         logical inference chains.",
    ),
    (
        "analogy",
        "Map structural similarities between the target problem and familiar \
         domains to transfer known solutions.",
    ),
    (
        "causal_reasoning",
        "Identify cause-and-effect relationships, trace chains of causation, \
         and distinguish correlation from causation.",
    ),
    (
        "step_by_step",
        "Break the problem into an ordered sequence of smaller sub-problems, \
         solving each before proceeding to the next.",
    ),
    (
        "comparative_analysis",
        "Systematically compare options, entities, or hypotheses across \
         multiple relevant dimensions.",
    ),
    (
        "hypothesis_testing",
        "Formulate testable hypotheses, design checks, and update beliefs \
         based on observed evidence.",
    ),
    (
        "decomposition",
        "Partition the problem into independent sub-components, solve each \
         independently, and integrate the partial solutions.",
    ),
];

// ── SelfDiscoverEngine ────────────────────────────────────────────────────────

/// Drives the Self-Discover pipeline: SELECT → ADAPT → IMPLEMENT.
///
/// The engine is generic over the model type `M` (which must implement
/// [`SelfDiscoverModel`]). Construct it via [`SelfDiscoverEngine::new`] and
/// call [`SelfDiscoverEngine::run`] for each query.
///
/// # Example
///
/// ```
/// use oxirag::self_discover::{
///     MockSelfDiscoverModel, SelfDiscoverConfig, SelfDiscoverEngine,
/// };
///
/// let engine = SelfDiscoverEngine::new(
///     SelfDiscoverConfig::default(),
///     MockSelfDiscoverModel,
/// );
/// let docs = vec!["Decomposing a task makes it tractable.".to_string()];
/// let result = engine.run("How do we solve large problems?", &docs).unwrap();
/// assert!(!result.selected_modules.is_empty());
/// assert!(!result.reasoning_structure.plan.is_empty());
/// ```
pub struct SelfDiscoverEngine<M: SelfDiscoverModel> {
    /// Configuration for this engine instance.
    pub config: SelfDiscoverConfig,
    /// The model driving SELECT, ADAPT, and IMPLEMENT.
    pub model: M,
}

impl<M: SelfDiscoverModel> SelfDiscoverEngine<M> {
    /// Create a new engine with the given configuration and model.
    #[must_use]
    pub fn new(config: SelfDiscoverConfig, model: M) -> Self {
        Self { config, model }
    }

    /// Return the full built-in reasoning module bank (8 modules).
    ///
    /// The bank contains the following modules:
    /// `critical_thinking`, `deductive_reasoning`, `analogy`,
    /// `causal_reasoning`, `step_by_step`, `comparative_analysis`,
    /// `hypothesis_testing`, `decomposition`.
    #[must_use]
    pub fn built_in_modules() -> Vec<ReasoningModule> {
        BUILT_IN_DATA
            .iter()
            .map(|(name, desc)| ReasoningModule::new(*name, *desc))
            .collect()
    }

    /// Run the Self-Discover pipeline for `query` against `docs`.
    ///
    /// ## Pipeline
    ///
    /// 1. **SELECT** — [`SelfDiscoverModel::select`] ranks the built-in module
    ///    bank. The result is truncated to
    ///    [`SelfDiscoverConfig::max_modules`].
    /// 2. **ADAPT + IMPLEMENT** — [`SelfDiscoverModel::adapt_and_implement`]
    ///    receives the selected modules and the first
    ///    [`SelfDiscoverConfig::top_k_docs`] documents, and returns the
    ///    structured reasoning plan together with the final answer.
    ///
    /// # Errors
    ///
    /// Propagates [`SelfDiscoverError::SelectionFailed`] from the SELECT stage
    /// and [`SelfDiscoverError::AdaptationFailed`] or
    /// [`SelfDiscoverError::ImplementationFailed`] from the ADAPT + IMPLEMENT
    /// stage.
    pub fn run(
        &self,
        query: &str,
        docs: &[String],
    ) -> Result<SelfDiscoverResult, SelfDiscoverError> {
        let all_modules = Self::built_in_modules();

        // ── SELECT ─────────────────────────────────────────────────────────
        let mut indices = self.model.select(query, &all_modules)?;
        indices.truncate(self.config.max_modules);

        let selected_modules: Vec<ReasoningModule> = indices
            .iter()
            .filter_map(|&idx| all_modules.get(idx))
            .cloned()
            .collect();

        // ── ADAPT + IMPLEMENT ──────────────────────────────────────────────
        let effective_docs = if docs.len() > self.config.top_k_docs {
            &docs[..self.config.top_k_docs]
        } else {
            docs
        };

        let (reasoning_structure, answer) =
            self.model
                .adapt_and_implement(query, &selected_modules, effective_docs)?;

        Ok(SelfDiscoverResult {
            query: query.to_string(),
            selected_modules,
            reasoning_structure,
            answer,
        })
    }
}
