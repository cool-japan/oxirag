//! [`SelfRefineEngine`] — the Self-Refine iterative self-feedback loop.

use crate::self_refine::types::{
    RefineStep, Refiner, SelfRefineConfig, SelfRefineError, SelfRefineOutput,
};

// ── SelfRefineEngine ───────────────────────────────────────────────────────────

/// Drives the Self-Refine loop: the same model generates, critiques, and refines
/// its own output until the feedback says to stop, a score threshold is met, or
/// the iteration cap is reached.
///
/// The model is provided *per call* through the generic [`SelfRefineEngine::run`]
/// method, mirroring the caller-supplies-executor pattern used across the crate.
#[derive(Debug, Clone, Default)]
pub struct SelfRefineEngine {
    /// Configuration for this engine.
    pub config: SelfRefineConfig,
}

impl SelfRefineEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SelfRefineConfig) -> Self {
        Self { config }
    }

    /// Run the Self-Refine loop for `task`.
    ///
    /// The model first produces an initial output. Then, for up to
    /// `max_iterations` iterations, the model critiques the current output and
    /// the `(output, feedback)` pair is recorded as a [`RefineStep`]. The loop
    /// stops as soon as the feedback's `stop` flag is set or its `score` reaches
    /// the configured `score_threshold`; otherwise the model refines the output
    /// and the loop continues. The `final_output` is the output of the last
    /// recorded step.
    ///
    /// # Errors
    ///
    /// Returns [`SelfRefineError::EmptyTask`] if `task` is empty after trimming.
    pub fn run<R>(&self, task: &str, refiner: &R) -> Result<SelfRefineOutput, SelfRefineError>
    where
        R: Refiner + ?Sized,
    {
        if task.trim().is_empty() {
            return Err(SelfRefineError::EmptyTask);
        }

        let mut output = refiner.initial(task);
        let mut steps: Vec<RefineStep> = Vec::new();

        for _ in 0..self.config.max_iterations {
            let feedback = refiner.feedback(task, &output);
            let stop = feedback.stop || feedback.score >= self.config.score_threshold;
            steps.push(RefineStep {
                output: output.clone(),
                feedback: feedback.clone(),
            });
            if stop {
                break;
            }
            output = refiner.refine(task, &output, &feedback);
        }

        let final_output = steps.last().map_or(output, |step| step.output.clone());
        let iterations = steps.len();

        Ok(SelfRefineOutput {
            final_output,
            steps,
            iterations,
        })
    }
}
