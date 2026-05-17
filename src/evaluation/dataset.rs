//! [`EvaluationDataset`] and related statistics for batch evaluation.

use std::{io, path::Path};

use serde::{Deserialize, Serialize};

use crate::types::PipelineOutput;

use super::types::EvaluationSample;

// ---------------------------------------------------------------------------
// DatasetStats
// ---------------------------------------------------------------------------

/// Summary statistics for an [`EvaluationDataset`].
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of samples in the dataset.
    pub total_samples: usize,
    /// Number of samples that have a `ground_truth` field set.
    pub samples_with_ground_truth: usize,
    /// Average number of context passages per sample.
    pub avg_context_len: f32,
    /// Average number of characters in the generated answers.
    pub avg_answer_len: f32,
}

// ---------------------------------------------------------------------------
// EvaluationDataset
// ---------------------------------------------------------------------------

/// An ordered collection of [`EvaluationSample`]s used for batch evaluation.
///
/// Supports JSON serialisation / deserialisation for persistence, as well as
/// construction from [`PipelineOutput`] vectors.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationDataset {
    samples: Vec<EvaluationSample>,
}

impl EvaluationDataset {
    /// Create an empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a sample to the dataset.
    pub fn add(&mut self, sample: EvaluationSample) {
        self.samples.push(sample);
    }

    /// Build a dataset from a slice of [`PipelineOutput`] references.
    #[must_use]
    pub fn from_pipeline_outputs(outputs: Vec<&PipelineOutput>) -> Self {
        let samples = outputs
            .into_iter()
            .map(EvaluationSample::from_pipeline_output)
            .collect();
        Self { samples }
    }

    /// Return the number of samples in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Return `true` when the dataset contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Iterate over samples in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &EvaluationSample> {
        self.samples.iter()
    }

    /// Serialise the dataset to a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be created or written.
    pub fn save_json(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Deserialise a dataset from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be read or parsed.
    pub fn load_json(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Compute summary statistics over the dataset.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> DatasetStats {
        let total_samples = self.samples.len();
        let samples_with_ground_truth = self
            .samples
            .iter()
            .filter(|s| s.ground_truth.is_some())
            .count();

        let avg_context_len = if total_samples == 0 {
            0.0
        } else {
            self.samples
                .iter()
                .map(|s| s.context.len() as f32)
                .sum::<f32>()
                / total_samples as f32
        };

        let avg_answer_len = if total_samples == 0 {
            0.0
        } else {
            self.samples
                .iter()
                .map(|s| s.answer.len() as f32)
                .sum::<f32>()
                / total_samples as f32
        };

        DatasetStats {
            total_samples,
            samples_with_ground_truth,
            avg_context_len,
            avg_answer_len,
        }
    }
}
