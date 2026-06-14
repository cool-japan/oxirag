//! [`AnalogicalEngine`] — the Analogical Prompting self-generation pipeline.

use crate::analogical_prompting::types::{
    AnalogicalConfig, AnalogicalError, AnalogicalModel, AnalogicalOutput, Exemplar,
};

// ── AnalogicalEngine ──────────────────────────────────────────────────────────

/// Drives Analogical Prompting: the model first self-generates relevant
/// exemplars (and optionally high-level knowledge), then solves the target
/// problem conditioned on what it produced.
///
/// The model is provided *per call* through the generic [`AnalogicalEngine::run`]
/// method, mirroring the caller-supplies-executor pattern used across the crate.
#[derive(Debug, Clone, Default)]
pub struct AnalogicalEngine {
    /// Configuration for this engine.
    pub config: AnalogicalConfig,
}

impl AnalogicalEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: AnalogicalConfig) -> Self {
        Self { config }
    }

    /// Run Analogical Prompting for `problem`.
    ///
    /// The engine asks `model` to self-generate up to `num_exemplars` exemplars,
    /// truncating the result to that limit. When `use_knowledge` is set it then
    /// asks `model` for a high-level knowledge tutorial; otherwise the knowledge
    /// is the empty string. Finally it asks `model` to solve the problem,
    /// conditioned on the generated exemplars and knowledge.
    ///
    /// # Errors
    ///
    /// Returns [`AnalogicalError::EmptyProblem`] if `problem` is empty after
    /// trimming.
    pub fn run<M>(&self, problem: &str, model: &M) -> Result<AnalogicalOutput, AnalogicalError>
    where
        M: AnalogicalModel + ?Sized,
    {
        if problem.trim().is_empty() {
            return Err(AnalogicalError::EmptyProblem);
        }

        let mut exemplars: Vec<Exemplar> =
            model.generate_exemplars(problem, self.config.num_exemplars);
        exemplars.truncate(self.config.num_exemplars);

        let knowledge = if self.config.use_knowledge {
            model.generate_knowledge(problem)
        } else {
            String::new()
        };

        let answer = model.solve(problem, &exemplars, &knowledge);

        Ok(AnalogicalOutput {
            exemplars,
            knowledge,
            answer,
        })
    }
}
