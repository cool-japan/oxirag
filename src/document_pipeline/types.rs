//! Core types for the document processing pipeline.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::types::SearchResult;

// ── ChunkStrategyKind ─────────────────────────────────────────────────────────

/// Selects the chunking strategy used by [`IndexingPipeline`].
///
/// [`IndexingPipeline`]: super::indexing::IndexingPipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChunkStrategyKind {
    /// Fixed-size character windows with overlap.
    #[default]
    FixedSize,
    /// Split on sentence boundaries.
    Sentence,
    /// Recursive, hierarchy-aware splitting.
    Recursive,
    /// Markdown header-aware splitting.
    Markdown,
}

// ── IndexingConfig ────────────────────────────────────────────────────────────

/// Configuration for the [`IndexingPipeline`].
///
/// [`IndexingPipeline`]: super::indexing::IndexingPipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Low-level chunk configuration (size, overlap, `min_size`).
    pub chunk_config: crate::chunking::ChunkConfig,
    /// Which chunking strategy to apply.
    pub chunk_strategy: ChunkStrategyKind,
    /// Skip chunks whose content is identical to one already indexed in this
    /// pipeline session (hash-based deduplication).
    pub auto_dedup: bool,
    /// Record a [`ChunkProvenance`] for every indexed chunk so callers can
    /// trace a search result back to its source document.
    pub store_provenance: bool,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            chunk_config: crate::chunking::ChunkConfig::default(),
            chunk_strategy: ChunkStrategyKind::FixedSize,
            auto_dedup: true,
            store_provenance: true,
        }
    }
}

impl IndexingConfig {
    /// Create a new `IndexingConfig` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunk configuration.
    #[must_use]
    pub fn with_chunk_config(mut self, config: crate::chunking::ChunkConfig) -> Self {
        self.chunk_config = config;
        self
    }

    /// Set the chunk strategy.
    #[must_use]
    pub fn with_chunk_strategy(mut self, strategy: ChunkStrategyKind) -> Self {
        self.chunk_strategy = strategy;
        self
    }

    /// Enable or disable automatic deduplication.
    #[must_use]
    pub fn with_auto_dedup(mut self, dedup: bool) -> Self {
        self.auto_dedup = dedup;
        self
    }

    /// Enable or disable provenance tracking.
    #[must_use]
    pub fn with_store_provenance(mut self, provenance: bool) -> Self {
        self.store_provenance = provenance;
        self
    }
}

// ── ChunkProvenance ───────────────────────────────────────────────────────────

/// Records how a single indexed chunk relates to its source document.
#[derive(Debug, Clone)]
pub struct ChunkProvenance {
    /// The UUID of the chunk document (assigned by the Echo layer on index).
    pub chunk_id: Uuid,
    /// The UUID of the original (unchunked) source document.
    pub source_doc_id: Uuid,
    /// Zero-based position of this chunk within the source document.
    pub chunk_index: usize,
    /// Inclusive start character offset in the source document's content.
    pub char_start: usize,
    /// Exclusive end character offset in the source document's content.
    pub char_end: usize,
}

impl ChunkProvenance {
    /// Create a new provenance record.
    #[must_use]
    pub fn new(
        chunk_id: Uuid,
        source_doc_id: Uuid,
        chunk_index: usize,
        char_start: usize,
        char_end: usize,
    ) -> Self {
        Self {
            chunk_id,
            source_doc_id,
            chunk_index,
            char_start,
            char_end,
        }
    }
}

// ── IndexingResult ────────────────────────────────────────────────────────────

/// Output of indexing a single source document through the pipeline.
#[derive(Debug, Clone)]
pub struct IndexingResult {
    /// UUID of the original source document.
    pub source_doc_id: Uuid,
    /// UUIDs of all chunk documents that were successfully indexed.
    pub chunk_ids: Vec<Uuid>,
    /// Provenance records (empty when `store_provenance = false`).
    pub provenance: Vec<ChunkProvenance>,
}

// ── DocumentAwareResult ───────────────────────────────────────────────────────

/// A search result enriched with provenance information.
#[derive(Debug, Clone)]
pub struct DocumentAwareResult {
    /// The base search result from the Echo layer.
    pub search_result: SearchResult,
    /// UUID of the source document this chunk came from (`None` when provenance
    /// tracking was disabled or the chunk was indexed without provenance).
    pub source_doc_id: Option<Uuid>,
    /// Zero-based index of this chunk within the source document.
    pub chunk_index: Option<usize>,
}

// ── PipelineStats ─────────────────────────────────────────────────────────────

/// Cumulative statistics for a pipeline instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Number of source documents processed by `index_document`.
    pub documents_indexed: usize,
    /// Total chunks produced across all indexed documents.
    pub chunks_created: usize,
    /// Chunks skipped due to deduplication.
    pub chunks_deduped: usize,
    /// Number of search calls.
    pub total_searches: usize,
    /// Number of searches that hit the semantic cache.
    pub cache_hits: usize,
}

// ── DocumentPipelineError ─────────────────────────────────────────────────────

/// Errors produced by the document pipeline.
#[derive(Debug, Error)]
pub enum DocumentPipelineError {
    /// Chunking the document failed.
    #[error("chunking failed: {0}")]
    ChunkingFailed(String),

    /// Indexing a chunk into the Echo layer failed.
    #[error("indexing failed: {0}")]
    IndexingFailed(String),

    /// A search operation failed.
    #[error("search failed: {0}")]
    SearchFailed(String),

    /// An interaction with the semantic cache failed.
    #[error("cache error: {0}")]
    CacheError(String),

    /// Provenance was requested for a chunk ID that has no recorded provenance.
    #[error("provenance not found for chunk {0}")]
    ProvenanceNotFound(Uuid),
}
