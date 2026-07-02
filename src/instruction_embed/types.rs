//! Types, configuration, and errors for the `instruction_embed` module.
//!
//! Instruction-conditioned embeddings (INSTRUCTOR, Su et al. 2023; TART,
//! Asai et al. 2023) pair every piece of embedded text with a
//! natural-language *task instruction*. This file defines the building
//! blocks:
//!
//! - [`TaskInstruction`] — a named natural-language instruction.
//! - [`InstructionEmbedding`] — the conditioned output vector.
//! - [`InstructionEmbedConfig`] — dimension, influence, normalisation,
//!   default top-k.
//! - [`InstructionEmbedError`] / [`InstructionEmbedResult`] — the module's
//!   error type and result alias.

use thiserror::Error;

// ── InstructionEmbedError / InstructionEmbedResult ──────────────────────────

/// Errors produced by the `instruction_embed` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InstructionEmbedError {
    /// The text passed to
    /// [`InstructionEmbedder::embed`](crate::instruction_embed::InstructionEmbedder::embed)
    /// was empty or whitespace-only.
    #[error("text must not be empty")]
    EmptyText,
    /// The query passed to
    /// [`InstructionIndex::search`](crate::instruction_embed::InstructionIndex::search)
    /// was empty or whitespace-only.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The corpus passed to
    /// [`InstructionIndex::index`](crate::instruction_embed::InstructionIndex::index)
    /// was empty, or the index otherwise holds no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
    /// A search was attempted before
    /// [`InstructionIndex::index`](crate::instruction_embed::InstructionIndex::index)
    /// had ever run.
    #[error("index has not been built; call `index` first")]
    NotIndexed,
    /// The configured embedding dimension was invalid (zero).
    #[error("invalid embedding dimension: {0}")]
    InvalidDim(usize),
    /// [`InstructionRegistry::try_get`](crate::instruction_embed::InstructionRegistry::try_get)
    /// was called with a name that was never registered.
    #[error("unknown instruction: {0}")]
    UnknownInstruction(String),
}

/// Convenient result alias for the `instruction_embed` module.
pub type InstructionEmbedResult<T> = Result<T, InstructionEmbedError>;

// ── TaskInstruction ──────────────────────────────────────────────────────────

/// A single named natural-language task instruction.
///
/// Pairing the exact same input text with a different [`TaskInstruction`]
/// and re-embedding it with
/// [`InstructionEmbedder::embed`](crate::instruction_embed::InstructionEmbedder::embed)
/// produces a genuinely different vector — that is the whole premise of
/// instruction-conditioned embeddings. `name` is a short, stable identifier
/// (suitable for
/// [`InstructionRegistry`](crate::instruction_embed::InstructionRegistry)
/// lookups); `text` is the natural-language instruction itself, e.g.
/// `"Represent the science document for retrieval:"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInstruction {
    /// Short, stable identifier for this instruction (registry lookup key).
    pub name: String,
    /// The natural-language instruction text that conditions the embedding.
    pub text: String,
}

impl TaskInstruction {
    /// Construct a named task instruction.
    #[must_use]
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            text: text.into(),
        }
    }

    /// Construct an ad hoc, unregistered instruction whose `name` equals its
    /// own `text`.
    ///
    /// Convenient when an instruction is used once inline and never looked
    /// up by name through an [`InstructionRegistry`](crate::instruction_embed::InstructionRegistry).
    #[must_use]
    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            name: text.clone(),
            text,
        }
    }

    /// Return `true` when the instruction text is empty or whitespace-only.
    #[must_use]
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

// ── InstructionEmbedding ─────────────────────────────────────────────────────

/// A single instruction-conditioned embedding produced by
/// [`InstructionEmbedder::embed`](crate::instruction_embed::InstructionEmbedder::embed).
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionEmbedding {
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// The [`TaskInstruction::name`] this embedding was conditioned on.
    pub instruction_name: String,
}

impl InstructionEmbedding {
    /// Dimensionality of the embedding.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.vector.len()
    }

    /// The vector's L2 norm.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    }
}

// ── InstructionEmbedConfig ───────────────────────────────────────────────────

/// Configuration for [`InstructionEmbedder`](crate::instruction_embed::InstructionEmbedder)
/// and [`InstructionIndex`](crate::instruction_embed::InstructionIndex).
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionEmbedConfig {
    /// Dimensionality of every produced embedding. Must be non-zero.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Strength of the instruction conditioning applied on top of the base,
    /// instruction-agnostic text embedding.
    ///
    /// Scales the magnitude of the additive task-shift vector *without
    /// bound*, and the per-dimension gate's maximum deviation from `1.0` up
    /// to a saturation point at `1.0` (see
    /// [`InstructionEmbedder::embed`](crate::instruction_embed::InstructionEmbedder::embed)
    /// for why the gate deliberately stops growing there). `0.0` disables
    /// conditioning entirely: the embedding collapses to the
    /// instruction-agnostic base embedding for *any* instruction. Larger
    /// values make the instruction dominate the resulting vector more, and
    /// values above `1.0` are meaningful — they keep strengthening the
    /// semantic task-shift pull even once the gate has saturated.
    ///
    /// Defaults to `0.65`.
    pub instruction_influence: f32,
    /// Whether to L2-normalise the final, instruction-conditioned embedding.
    ///
    /// Defaults to `true`.
    pub normalize: bool,
    /// Default number of results returned by
    /// [`InstructionIndex::search_default`](crate::instruction_embed::InstructionIndex::search_default).
    ///
    /// Defaults to `10`.
    pub default_top_k: usize,
}

impl Default for InstructionEmbedConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            instruction_influence: 0.65,
            normalize: true,
            default_top_k: 10,
        }
    }
}

impl InstructionEmbedConfig {
    /// Create a configuration with the default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the embedding dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the instruction-influence strength.
    #[must_use]
    pub fn with_instruction_influence(mut self, instruction_influence: f32) -> Self {
        self.instruction_influence = instruction_influence;
        self
    }

    /// Set whether the final embedding is L2-normalised.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the default top-k result count used by
    /// [`InstructionIndex::search_default`](crate::instruction_embed::InstructionIndex::search_default).
    #[must_use]
    pub fn with_default_top_k(mut self, default_top_k: usize) -> Self {
        self.default_top_k = default_top_k;
        self
    }

    /// Validate this configuration's invariants.
    ///
    /// # Errors
    ///
    /// Returns [`InstructionEmbedError::InvalidDim`] when
    /// [`InstructionEmbedConfig::dim`] is `0`.
    pub fn validate(&self) -> InstructionEmbedResult<()> {
        if self.dim == 0 {
            return Err(InstructionEmbedError::InvalidDim(self.dim));
        }
        Ok(())
    }
}
