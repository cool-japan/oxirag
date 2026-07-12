//! Core data types, configuration, and error enum for the `chain_of_agents`
//! module.
//!
//! The pluggable worker/manager traits and their deterministic default
//! implementations live in [`super::worker`]; the bounded communication-unit
//! merge policy lives in [`super::unit`]; the sequential orchestration
//! engine lives in [`super::engine`]. This file holds only the plain data
//! that flows between them.

use thiserror::Error;

use super::unit::CoaCommunicationUnit;

// ── CoaError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `chain_of_agents` module.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoaError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,

    /// The document to chunk and process was empty or contained only
    /// whitespace, or splitting it produced zero chunks (equivalently: the
    /// `chunks` slice passed to
    /// [`CoaEngine::run_chunks`](crate::chain_of_agents::CoaEngine::run_chunks)
    /// was empty).
    #[error("document must not be empty")]
    EmptyDocument,

    /// A [`CoaConfig`] value failed validation — see [`CoaConfig::validate`].
    #[error("invalid chain-of-agents configuration: {reason}")]
    InvalidConfig {
        /// A human-readable description of what was invalid.
        reason: String,
    },

    /// A [`CoaWorker`](crate::chain_of_agents::CoaWorker) implementation
    /// failed to process its chunk.
    ///
    /// Provided for custom (non-default) implementations backed by a real
    /// model or service, so a failure (timeout, refusal, malformed
    /// response, ...) can be reported without extending this enum.
    #[error("worker {worker_index} failed: {reason}")]
    WorkerFailed {
        /// The zero-based index of the worker (and chunk) that failed.
        worker_index: usize,
        /// A human-readable reason for the failure.
        reason: String,
    },

    /// A [`CoaManager`](crate::chain_of_agents::CoaManager) implementation
    /// failed to synthesize a final answer.
    #[error("manager failed: {reason}")]
    ManagerFailed {
        /// A human-readable reason for the failure.
        reason: String,
    },
}

// ── CoaConfig ────────────────────────────────────────────────────────────────

/// The smallest `chunk_size` [`CoaConfig::validate`] accepts.
///
/// Chunks below this are almost certainly a configuration mistake (too small
/// to hold a single sentence of most languages) rather than a deliberate
/// choice, so they are rejected up front rather than silently producing a
/// chain of near-empty, low-information workers.
pub const MIN_CHUNK_SIZE: usize = 16;

/// Configuration for [`CoaEngine`](crate::chain_of_agents::CoaEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoaConfig {
    /// Target maximum number of characters per chunk. The local splitter
    /// (see the [module-level documentation](crate::chain_of_agents)) packs
    /// whole sentences up to this budget; a single sentence longer than
    /// `chunk_size` becomes its own oversized chunk rather than being cut
    /// mid-word. Must be at least [`MIN_CHUNK_SIZE`]. Defaults to `800`.
    pub chunk_size: usize,
    /// Approximate number of trailing characters from one chunk that are
    /// repeated as the seed of the next chunk, so a sentence right at a
    /// chunk boundary is visible to the worker on both sides of the cut.
    /// Must be strictly smaller than `chunk_size`. Defaults to `100`.
    pub chunk_overlap: usize,
    /// Maximum number of evidence entries a [`CoaCommunicationUnit`] retains
    /// at once — see
    /// [`CoaCommunicationUnit::merge_evidence`] for the eviction rule this
    /// bounds. This is what keeps the communication unit from growing
    /// without limit as the chain lengthens, which is the entire point of
    /// chunking in the first place. Must be at least `1`. Defaults to `6`.
    pub evidence_budget: usize,
    /// Maximum number of open (unresolved) questions a
    /// [`CoaCommunicationUnit`] retains at once — see
    /// [`CoaCommunicationUnit::merge_open_questions`]. Defaults to `4`.
    pub open_question_budget: usize,
    /// Hard cap on chain length: at most this many workers ever run,
    /// regardless of how many chunks the document splits into. When a
    /// document produces more chunks than this,
    /// [`CoaEngine::run`](crate::chain_of_agents::CoaEngine::run) processes
    /// exactly the *first* `max_workers` chunks, in order, and reports the
    /// truncation via
    /// [`CoaTrace::truncated`](crate::chain_of_agents::CoaTrace::truncated)
    /// and
    /// [`CoaTrace::chunk_count`](crate::chain_of_agents::CoaTrace::chunk_count) —
    /// the remaining chunks are dropped, but never silently. Defaults to
    /// `64`.
    pub max_workers: usize,
}

impl Default for CoaConfig {
    fn default() -> Self {
        Self {
            chunk_size: 800,
            chunk_overlap: 100,
            evidence_budget: 6,
            open_question_budget: 4,
            max_workers: 64,
        }
    }
}

impl CoaConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target chunk size, in characters.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the chunk overlap, in characters.
    #[must_use]
    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    /// Set the communication unit's evidence budget.
    #[must_use]
    pub fn with_evidence_budget(mut self, evidence_budget: usize) -> Self {
        self.evidence_budget = evidence_budget;
        self
    }

    /// Set the communication unit's open-question budget.
    #[must_use]
    pub fn with_open_question_budget(mut self, open_question_budget: usize) -> Self {
        self.open_question_budget = open_question_budget;
        self
    }

    /// Set the hard cap on chain length.
    #[must_use]
    pub fn with_max_workers(mut self, max_workers: usize) -> Self {
        self.max_workers = max_workers;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CoaError::InvalidConfig`] when `chunk_size` is below
    /// [`MIN_CHUNK_SIZE`], `chunk_overlap` is not strictly smaller than
    /// `chunk_size`, `evidence_budget` is `0`, or `max_workers` is `0`.
    pub fn validate(&self) -> Result<(), CoaError> {
        if self.chunk_size < MIN_CHUNK_SIZE {
            return Err(CoaError::InvalidConfig {
                reason: format!(
                    "chunk_size must be at least {MIN_CHUNK_SIZE}, got {}",
                    self.chunk_size
                ),
            });
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(CoaError::InvalidConfig {
                reason: format!(
                    "chunk_overlap ({}) must be smaller than chunk_size ({})",
                    self.chunk_overlap, self.chunk_size
                ),
            });
        }
        if self.evidence_budget == 0 {
            return Err(CoaError::InvalidConfig {
                reason: "evidence_budget must be at least 1".to_string(),
            });
        }
        if self.max_workers == 0 {
            return Err(CoaError::InvalidConfig {
                reason: "max_workers must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

// ── CoaEvidence ──────────────────────────────────────────────────────────────

/// One snippet of evidence carried inside a [`CoaCommunicationUnit`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoaEvidence {
    /// The evidence text itself (typically one extracted sentence).
    pub text: String,
    /// The zero-based index, into the chain's chunk sequence, of the chunk
    /// this evidence was extracted from.
    pub source_chunk_index: usize,
    /// This evidence's relevance to the query, in `[0.0, 1.0]`. Higher is
    /// more relevant. Drives the eviction order described at
    /// [`CoaCommunicationUnit::merge_evidence`].
    pub relevance_score: f32,
}

impl CoaEvidence {
    /// Create a new evidence entry. `relevance_score` is clamped into
    /// `[0.0, 1.0]`.
    #[must_use]
    pub fn new(text: impl Into<String>, source_chunk_index: usize, relevance_score: f32) -> Self {
        Self {
            text: text.into(),
            source_chunk_index,
            relevance_score: relevance_score.clamp(0.0, 1.0),
        }
    }
}

// ── CoaWorkerStep ────────────────────────────────────────────────────────────

/// One worker's contribution to a [`CoaTrace`]: the chunk it read, the
/// communication unit it received, and the communication unit it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct CoaWorkerStep {
    /// Zero-based index of this worker in the chain.
    pub worker_index: usize,
    /// Zero-based index of the chunk this worker processed. Always equal to
    /// `worker_index` (workers and chunks are processed strictly in order,
    /// one worker per chunk) but kept as its own self-describing field.
    pub chunk_index: usize,
    /// The chunk text this worker read.
    pub chunk: String,
    /// The communication unit this worker received (worker `0` always
    /// receives an empty unit — see [`CoaCommunicationUnit::new`]).
    pub incoming_unit: CoaCommunicationUnit,
    /// The communication unit this worker produced, after the engine's
    /// bookkeeping (chunk counter, budget enforcement) has been applied.
    pub outgoing_unit: CoaCommunicationUnit,
}

// ── CoaTrace ─────────────────────────────────────────────────────────────────

/// The complete output of a
/// [`CoaEngine::run`](crate::chain_of_agents::CoaEngine::run) (or
/// [`CoaEngine::run_chunks`](crate::chain_of_agents::CoaEngine::run_chunks))
/// call.
#[derive(Debug, Clone, PartialEq)]
pub struct CoaTrace {
    /// The original query.
    pub query: String,
    /// Every worker's step, in chain order (`steps[i].worker_index == i`).
    pub steps: Vec<CoaWorkerStep>,
    /// The final communication unit, i.e. `steps.last().outgoing_unit`.
    /// Kept as an owned copy for direct access without indexing into
    /// `steps`. This — and *only* this — is what
    /// [`CoaManager::synthesize`](crate::chain_of_agents::CoaManager::synthesize)
    /// saw when it produced `answer`.
    pub final_unit: CoaCommunicationUnit,
    /// The manager's synthesized answer, derived from `final_unit` alone.
    pub answer: String,
    /// Total number of chunks the input document was split into (or the
    /// length of the `chunks` slice passed to
    /// [`CoaEngine::run_chunks`](crate::chain_of_agents::CoaEngine::run_chunks)),
    /// *before* any [`CoaConfig::max_workers`] truncation.
    pub chunk_count: usize,
    /// Number of workers actually run: `min(chunk_count, max_workers)`.
    pub workers_run: usize,
    /// `true` when `chunk_count > max_workers`, i.e. one or more trailing
    /// chunks were dropped without being processed. See
    /// [`CoaConfig::max_workers`] for the exact rule (first `max_workers`
    /// chunks, in order — never a silent, unreported truncation).
    pub truncated: bool,
}
