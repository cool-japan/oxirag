//! The [`ComposedPipeline`] builder and execution engine.

use super::types::{ComposerError, PipelineStage, StageInput, StageOutput};

// ── ComposedPipeline ──────────────────────────────────────────────────────────

/// A linear chain of [`PipelineStage`]s executed in order.
///
/// Each stage receives the `context` produced by the previous stage.  When
/// `stop_on_block` is `true` (the default), a blocking stage short-circuits
/// the pipeline and its blocked output is returned immediately.
pub struct ComposedPipeline {
    /// The ordered stages.
    stages: Vec<Box<dyn PipelineStage>>,
    /// Whether to stop execution when a stage blocks the pipeline.
    stop_on_block: bool,
}

impl ComposedPipeline {
    /// Create a new empty pipeline (no stages, `stop_on_block = true`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            stop_on_block: true,
        }
    }

    /// Set the `stop_on_block` behaviour.
    #[must_use]
    pub fn with_stop_on_block(mut self, v: bool) -> Self {
        self.stop_on_block = v;
        self
    }

    /// Append a stage to the pipeline (builder pattern).
    #[must_use]
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Return the number of stages in the pipeline.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` when the pipeline has no stages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Execute all stages in order, threading the context through each one.
    ///
    /// # Errors
    ///
    /// Returns [`ComposerError::EmptyPipeline`] when no stages are present.
    /// Returns [`ComposerError::StageFailed`] when a stage returns a hard
    /// error.
    ///
    /// A blocking stage does **not** produce an error — the blocked
    /// [`StageOutput`] is returned as `Ok`.
    pub fn run(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
        if self.stages.is_empty() {
            return Err(ComposerError::EmptyPipeline);
        }

        let mut current_input = input;

        for stage in &self.stages {
            let stage_name = stage.name().to_string();
            let output = stage
                .process(current_input.clone())
                .map_err(|e| ComposerError::StageFailed(stage_name.clone(), e.to_string()))?;

            if output.is_blocked() && self.stop_on_block {
                return Ok(output);
            }

            // Forward the output content as the context for the next stage,
            // merging any metadata produced by this stage.
            let mut next_input = StageInput {
                query: current_input.query,
                context: output.content.clone(),
                metadata: current_input.metadata,
            };
            for (k, v) in &output.metadata {
                next_input.metadata.insert(k.clone(), v.clone());
            }
            current_input = next_input;
        }

        // All stages completed — return the final context as a passing output.
        let final_output = StageOutput {
            content: current_input.context,
            blocked: false,
            metadata: current_input.metadata,
        };
        Ok(final_output)
    }
}

impl Default for ComposedPipeline {
    fn default() -> Self {
        Self::new()
    }
}
