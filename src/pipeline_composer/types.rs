//! Types for the `pipeline_composer` module.

use std::collections::HashMap;

use thiserror::Error;

// ── StageInput ────────────────────────────────────────────────────────────────

/// Input passed into a [`PipelineStage`].
#[derive(Debug, Clone, Default)]
pub struct StageInput {
    /// The original user query.
    pub query: String,
    /// The accumulated context text flowing through the pipeline.
    pub context: String,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl StageInput {
    /// Create a new [`StageInput`] with `query` and `context`.
    #[must_use]
    pub fn new(query: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            context: context.into(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a metadata entry and return `self` (builder).
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ── StageOutput ───────────────────────────────────────────────────────────────

/// Output produced by a [`PipelineStage`].
#[derive(Debug, Clone, Default)]
pub struct StageOutput {
    /// The produced content (replaces `context` for the next stage).
    pub content: String,
    /// `true` when this stage has blocked the pipeline.
    pub blocked: bool,
    /// Arbitrary key-value metadata forwarded downstream.
    pub metadata: HashMap<String, String>,
}

impl StageOutput {
    /// Create an unblocked output carrying `content`.
    #[must_use]
    pub fn pass(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            blocked: false,
            metadata: HashMap::new(),
        }
    }

    /// Create a blocked output, recording `reason` under `"block_reason"`.
    #[must_use]
    pub fn block(reason: impl Into<String>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("block_reason".to_string(), reason.into());
        Self {
            content: String::new(),
            blocked: true,
            metadata,
        }
    }

    /// Return `true` when this output is blocked.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.blocked
    }

    /// Attach a metadata entry and return `self` (builder).
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// ── PipelineStage ─────────────────────────────────────────────────────────────

/// A single synchronous processing stage in a composed pipeline.
///
/// Implementors should be pure (no I/O) and `Send + Sync` for safe use across
/// thread boundaries.
pub trait PipelineStage: Send + Sync {
    /// A human-readable name for this stage (used in error messages).
    fn name(&self) -> &str;

    /// Process `input` and return a [`StageOutput`].
    ///
    /// # Errors
    ///
    /// Returns [`ComposerError`] if the stage encounters an unrecoverable error.
    fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError>;
}

// ── ComposerError ─────────────────────────────────────────────────────────────

/// Errors from the `pipeline_composer` module.
#[derive(Debug, Error)]
pub enum ComposerError {
    /// A stage blocked the pipeline (soft block, propagated as `Ok`).
    #[error("Stage '{0}' blocked the pipeline")]
    StageBlocked(String),
    /// A stage returned a hard error.
    #[error("Stage '{0}' failed: {1}")]
    StageFailed(String, String),
    /// The pipeline has no stages to execute.
    #[error("Pipeline has no stages")]
    EmptyPipeline,
}
