#![cfg(feature = "python")]
//! Fluent builder for constructing a [`PyPipeline`] from Python.
//!
//! Python callers never need to deal with generic type parameters — this
//! module hard-codes the mock/in-memory backend combination that is always
//! available without additional ML dependencies.

use std::sync::Arc;

use pyo3::prelude::*;

use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
use crate::observability::MemoryObserver;
use crate::pipeline::{PipelineBuilder, PipelineConfig};
use crate::python::pipeline::PyPipeline;

// ────────────────────────────────────────────────────────────────────────────
// PyPipelineBuilder
// ────────────────────────────────────────────────────────────────────────────

/// Fluent builder for `Pipeline` instances exposed to Python.
///
/// # Examples (Python)
///
/// ```python
/// pipeline = (
///     oxirag.PipelineBuilder()
///         .with_dimension(384)
///         .with_max_results(10)
///         .build()
/// )
/// ```
#[pyclass(name = "PipelineBuilder")]
pub struct PyPipelineBuilder {
    /// Vector embedding dimension.
    dimension: usize,
    /// Maximum number of Echo-layer search results forwarded to later layers.
    max_results: usize,
    /// Whether to enable the fast-path short-circuit (skip L2+L3 when Echo
    /// confidence is already high).
    enable_fast_path: bool,
    /// Fast-path confidence threshold.
    fast_path_threshold: f32,
}

impl Default for PyPipelineBuilder {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_results: 10,
            enable_fast_path: true,
            fast_path_threshold: 0.95,
        }
    }
}

#[pymethods]
impl PyPipelineBuilder {
    /// Create a new builder with sensible defaults
    /// (dimension=384, `max_results=10`, `fast_path` enabled at 0.95).
    #[new]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the embedding vector dimension.
    ///
    /// Must match the dimension produced by the embedding provider (default 384).
    #[must_use]
    pub fn with_dimension(&self, dim: usize) -> Self {
        Self {
            dimension: dim,
            max_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Override the maximum number of search results forwarded through the pipeline.
    #[must_use]
    pub fn with_max_results(&self, k: usize) -> Self {
        Self {
            dimension: self.dimension,
            max_results: k,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Enable or disable the fast-path optimisation.
    ///
    /// When enabled, queries with a top Echo score above `fast_path_threshold`
    /// skip Layer 2 (Speculator) and Layer 3 (Judge).
    #[must_use]
    pub fn with_fast_path(&self, enable: bool) -> Self {
        Self {
            dimension: self.dimension,
            max_results: self.max_results,
            enable_fast_path: enable,
            fast_path_threshold: self.fast_path_threshold,
        }
    }

    /// Override the fast-path confidence threshold (default 0.95).
    #[must_use]
    pub fn with_fast_path_threshold(&self, threshold: f32) -> Self {
        Self {
            dimension: self.dimension,
            max_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: threshold,
        }
    }

    /// Construct and return a ready-to-use [`PyPipeline`].
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if the internal builder fails (this
    /// should not happen for the default in-memory configuration).
    pub fn build(&self) -> PyResult<PyPipeline> {
        let echo = EchoLayer::new(
            MockEmbeddingProvider::new(self.dimension),
            InMemoryVectorStore::new(self.dimension),
        );
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        let config = PipelineConfig {
            max_search_results: self.max_results,
            enable_fast_path: self.enable_fast_path,
            fast_path_threshold: self.fast_path_threshold,
            ..PipelineConfig::default()
        };

        let observer = Arc::new(MemoryObserver::new());

        let pipeline = PipelineBuilder::new()
            .with_echo(echo)
            .with_speculator(speculator)
            .with_judge(judge)
            .with_config(config)
            .with_observers(vec![
                Arc::clone(&observer) as Arc<dyn crate::observability::SpanObserver>
            ])
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyPipeline {
            inner: Arc::new(tokio::sync::Mutex::new(pipeline)),
            observer,
        })
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!(
            "PipelineBuilder(dimension={}, max_results={}, fast_path={}, threshold={:.2})",
            self.dimension, self.max_results, self.enable_fast_path, self.fast_path_threshold
        )
    }
}
