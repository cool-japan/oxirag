//! Types for the `chain_of_density` module.

use thiserror::Error;

// ── CodConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the Chain-of-Density summarization engine.
///
/// Controls how many densification passes are performed, the word budget that
/// the summary must respect, and how many fresh salient entities are folded in
/// per pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodConfig {
    /// Number of densification iterations to perform after the initial summary.
    ///
    /// Defaults to `3`.
    pub iterations: usize,
    /// Target word budget the summary must stay close to (a small slack is
    /// permitted before trailing filler is trimmed).
    ///
    /// Defaults to `60`.
    pub target_words: usize,
    /// Number of new salient source entities to incorporate per iteration.
    ///
    /// Defaults to `2`.
    pub entities_per_step: usize,
}

impl Default for CodConfig {
    fn default() -> Self {
        Self {
            iterations: 3,
            target_words: 60,
            entities_per_step: 2,
        }
    }
}

impl CodConfig {
    /// Create a new [`CodConfig`] with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of densification iterations.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the target word budget for the summary.
    #[must_use]
    pub fn with_target_words(mut self, target_words: usize) -> Self {
        self.target_words = target_words;
        self
    }

    /// Set the number of new entities incorporated per iteration.
    #[must_use]
    pub fn with_entities_per_step(mut self, entities_per_step: usize) -> Self {
        self.entities_per_step = entities_per_step;
        self
    }
}

// ── DensityStep ───────────────────────────────────────────────────────────────

/// A single densification step in the Chain-of-Density process.
///
/// Captures the rewritten summary after the step, the source entities that were
/// newly folded in, and density statistics (word and distinct-entity counts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensityStep {
    /// The summary text produced by this step.
    pub summary: String,
    /// Source entities incorporated by this step that were absent beforehand.
    pub added_entities: Vec<String>,
    /// Number of whitespace-delimited words in [`summary`](Self::summary).
    pub word_count: usize,
    /// Number of distinct source entities present in [`summary`](Self::summary).
    pub entity_count: usize,
}

impl DensityStep {
    /// Entity density: distinct source entities per word.
    ///
    /// Returns `0.0` when the summary contains no words.
    #[must_use]
    pub fn density(&self) -> f32 {
        if self.word_count == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let density = self.entity_count as f32 / self.word_count as f32;
            density
        }
    }
}

// ── ChainOfDensityOutput ──────────────────────────────────────────────────────

/// The result of a Chain-of-Density run.
///
/// Holds every recorded [`DensityStep`] in order plus the final, most
/// entity-dense summary (which equals the summary of the last step).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainOfDensityOutput {
    /// The densification steps in execution order.
    pub steps: Vec<DensityStep>,
    /// The final, densest summary (identical to the last step's summary).
    pub final_summary: String,
}

impl ChainOfDensityOutput {
    /// Returns the number of recorded densification steps.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns the distinct-entity count of the final summary, or `0` when no
    /// steps were recorded.
    #[must_use]
    pub fn final_entity_count(&self) -> usize {
        self.steps.last().map_or(0, |s| s.entity_count)
    }
}

// ── CodError ──────────────────────────────────────────────────────────────────

/// Errors that can occur during Chain-of-Density summarization.
#[derive(Debug, Error)]
pub enum CodError {
    /// The source text was empty or contained only whitespace.
    #[error("source must not be empty")]
    EmptySource,
}
