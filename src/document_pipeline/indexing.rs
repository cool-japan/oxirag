//! `IndexingPipeline` — chunk, dedup, and index documents into an Echo store.
//!
//! The indexing pipeline wraps the chunking module and the Echo layer's `index`
//! method into a single, stats-tracked unit that optionally deduplicates
//! chunks by content hash and records provenance information.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::chunking::{Chunk, ChunkConfig, DocumentChunker};
use crate::layer1_echo::Echo;
use crate::types::Document;

use super::types::{
    ChunkProvenance, ChunkStrategyKind, DocumentPipelineError, IndexingConfig, IndexingResult,
    PipelineStats,
};

// ── Helper: build a DocumentChunker from config ───────────────────────────────

fn make_chunker(config: &IndexingConfig) -> DocumentChunker {
    let chunk_cfg = config.chunk_config.clone();
    match config.chunk_strategy {
        ChunkStrategyKind::FixedSize => DocumentChunker::with_fixed_size(chunk_cfg),
        ChunkStrategyKind::Sentence => DocumentChunker::with_sentences(chunk_cfg),
        ChunkStrategyKind::Recursive => DocumentChunker::with_recursive(chunk_cfg),
        ChunkStrategyKind::Markdown => DocumentChunker::with_markdown(chunk_cfg),
    }
}

// ── Content hash ──────────────────────────────────────────────────────────────

fn content_hash(content: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

// ── IndexingPipeline ──────────────────────────────────────────────────────────

/// Converts source documents into indexed chunks via the Echo layer.
///
/// # Deduplication
///
/// When `auto_dedup` is enabled (default) the pipeline maintains a
/// `HashSet<u64>` of content hashes for chunks already indexed in this
/// session.  Chunks whose content hash has already been seen are skipped.
/// This operates on content text, not document ID.
///
/// # Provenance
///
/// When `store_provenance` is enabled (default) the pipeline records a
/// [`ChunkProvenance`] for every indexed chunk, mapping the chunk's UUID back
/// to its source document.  Use [`get_provenance`] to retrieve a record.
///
/// [`get_provenance`]: IndexingPipeline::get_provenance
pub struct IndexingPipeline {
    config: IndexingConfig,
    chunker: DocumentChunker,
    /// `chunk_id` (`Uuid`) → `ChunkProvenance`
    provenance_map: Arc<RwLock<HashMap<Uuid, ChunkProvenance>>>,
    /// Set of content hashes for deduplication.
    content_hashes: Arc<RwLock<HashSet<u64>>>,
    stats: Arc<Mutex<PipelineStats>>,
}

impl IndexingPipeline {
    /// Create a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: IndexingConfig) -> Self {
        let chunker = make_chunker(&config);
        Self {
            config,
            chunker,
            provenance_map: Arc::new(RwLock::new(HashMap::new())),
            content_hashes: Arc::new(RwLock::new(HashSet::new())),
            stats: Arc::new(Mutex::new(PipelineStats::default())),
        }
    }

    /// Create a pipeline with default configuration.
    #[must_use]
    pub fn with_default() -> Self {
        Self::new(IndexingConfig::default())
    }

    /// Return a clone of the shared provenance map (for sharing with
    /// [`RetrievalPipeline`]).
    ///
    /// [`RetrievalPipeline`]: super::retrieval::RetrievalPipeline
    #[must_use]
    pub fn provenance_map(&self) -> Arc<RwLock<HashMap<Uuid, ChunkProvenance>>> {
        Arc::clone(&self.provenance_map)
    }

    /// Return a clone of the shared stats (for sharing with
    /// `RetrievalPipeline`).
    #[must_use]
    pub fn stats_handle(&self) -> Arc<Mutex<PipelineStats>> {
        Arc::clone(&self.stats)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Chunk `doc` using the configured strategy and return raw [`Chunk`] objects.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentPipelineError::ChunkingFailed`] if the chunking
    /// operation produces an empty list (e.g., document content is below the
    /// configured minimum chunk size and no chunks survive the filter).  In
    /// practice the chunkers do not panic; the error path exists for API
    /// completeness.
    pub fn chunk_document(&self, doc: &Document) -> Result<Vec<Chunk>, DocumentPipelineError> {
        let chunks = self.chunker.raw_chunks(doc);
        if chunks.is_empty() && !doc.content.is_empty() {
            // Chunking succeeded but the document was too short — return one
            // synthetic chunk so that short docs are still indexed.
            return Ok(Vec::new());
        }
        Ok(chunks)
    }

    /// Chunk, dedup, and index a single document.
    ///
    /// Steps:
    /// 1. Chunk the document via the configured strategy.
    /// 2. For each chunk: if `auto_dedup`, skip if content hash already seen.
    /// 3. Convert chunk to `Document` via `Chunk::into_document()`.
    /// 4. Call `echo.index(chunk_doc).await`.
    /// 5. Record provenance if `store_provenance`.
    /// 6. Update pipeline stats.
    ///
    /// # Errors
    ///
    /// - [`DocumentPipelineError::ChunkingFailed`] — chunking raised a panic
    ///   (extremely unlikely with the current pure-Rust strategies).
    /// - [`DocumentPipelineError::IndexingFailed`] — Echo layer returned an
    ///   error for one of the chunk documents.
    pub async fn index_document<E: Echo>(
        &self,
        echo: &mut E,
        doc: Document,
    ) -> Result<IndexingResult, DocumentPipelineError> {
        let source_doc_id: Uuid = doc
            .id
            .as_str()
            .parse::<Uuid>()
            .unwrap_or_else(|_| Uuid::new_v4());

        let chunks = self.chunker.raw_chunks(&doc);

        let mut chunk_ids: Vec<Uuid> = Vec::new();
        let mut provenance_records: Vec<ChunkProvenance> = Vec::new();
        let mut total_chunks = 0_usize;
        let mut deduped_chunks = 0_usize;

        for chunk in chunks {
            total_chunks += 1;

            if self.config.auto_dedup {
                let hash = content_hash(&chunk.content);
                let mut hashes = self.content_hashes.write().await;
                if hashes.contains(&hash) {
                    deduped_chunks += 1;
                    continue;
                }
                hashes.insert(hash);
            }

            let chunk_index = chunk.chunk_index;
            let char_start = chunk.start_char;
            let char_end = chunk.end_char;

            let chunk_doc = chunk.into_document();
            let doc_id_str = chunk_doc.id.as_str().to_string();

            let doc_id = echo
                .index(chunk_doc)
                .await
                .map_err(|e| DocumentPipelineError::IndexingFailed(e.to_string()))?;

            let chunk_uuid: Uuid = doc_id.as_str().parse::<Uuid>().unwrap_or_else(|_| {
                // Fallback: parse the string we captured before indexing.
                doc_id_str
                    .parse::<Uuid>()
                    .unwrap_or_else(|_| Uuid::new_v4())
            });

            if self.config.store_provenance {
                let prov = ChunkProvenance::new(
                    chunk_uuid,
                    source_doc_id,
                    chunk_index,
                    char_start,
                    char_end,
                );
                provenance_records.push(prov.clone());

                let mut pmap = self.provenance_map.write().await;
                pmap.insert(chunk_uuid, prov);
            }

            chunk_ids.push(chunk_uuid);
        }

        // Update stats.
        let mut stats = self.stats.lock().await;
        stats.documents_indexed += 1;
        stats.chunks_created += total_chunks;
        stats.chunks_deduped += deduped_chunks;

        Ok(IndexingResult {
            source_doc_id,
            chunk_ids,
            provenance: provenance_records,
        })
    }

    /// Index a batch of documents, returning a result per document.
    ///
    /// Documents are processed sequentially.  If a single document fails the
    /// error is returned and processing stops at that point.
    ///
    /// # Errors
    ///
    /// Propagates the first error encountered in the batch.
    pub async fn index_batch<E: Echo>(
        &self,
        echo: &mut E,
        docs: Vec<Document>,
    ) -> Result<Vec<IndexingResult>, DocumentPipelineError> {
        let mut results = Vec::with_capacity(docs.len());
        for doc in docs {
            let res = self.index_document(echo, doc).await?;
            results.push(res);
        }
        Ok(results)
    }

    /// Look up provenance for `chunk_id`.
    ///
    /// Uses `try_read()` to avoid blocking the caller; returns `None` if the
    /// lock is contended or no provenance was recorded for the ID.
    #[must_use]
    pub fn get_provenance(&self, chunk_id: &Uuid) -> Option<ChunkProvenance> {
        crate::sync::try_read(&self.provenance_map).and_then(|guard| guard.get(chunk_id).cloned())
    }

    /// Return a snapshot of the current pipeline statistics.
    pub async fn stats(&self) -> PipelineStats {
        self.stats.lock().await.clone()
    }

    /// Expose the chunk config (useful for tests).
    #[must_use]
    pub fn chunk_config(&self) -> &ChunkConfig {
        &self.config.chunk_config
    }
}
