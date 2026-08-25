//! `RetrievalPipeline` and `DocumentPipelineBuilder` — query the Echo layer
//! with optional MMR reranking, semantic caching, and provenance enrichment.

use std::collections::HashMap;
use std::sync::Arc;

use crate::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::advanced_retrieval::{MmrConfig, MmrReranker};
use crate::layer1_echo::Echo;
use crate::semantic_cache::{InMemorySemanticCache, SemanticCache, SemanticCacheConfig};
use crate::types::{Draft, PipelineOutput, Query, SearchResult};

use super::indexing::IndexingPipeline;
use super::types::{
    ChunkProvenance, DocumentAwareResult, DocumentPipelineError, IndexingConfig, PipelineStats,
};

// ── RetrievalPipeline ─────────────────────────────────────────────────────────

/// High-level retrieval component that wraps Echo search with:
///
/// - **Semantic cache** — returns a cached `PipelineOutput` when the query
///   embedding is sufficiently similar to a previously seen query.
/// - **MMR reranking** — applies Maximal Marginal Relevance to increase result
///   diversity.
/// - **Provenance enrichment** — annotates results with source-document UUIDs
///   and chunk positions.
pub struct RetrievalPipeline {
    mmr_lambda: f32,
    enable_mmr: bool,
    semantic_cache: Option<Arc<Mutex<InMemorySemanticCache>>>,
    provenance_map: Arc<RwLock<HashMap<Uuid, ChunkProvenance>>>,
    stats: Arc<Mutex<PipelineStats>>,
}

impl RetrievalPipeline {
    /// Create a retrieval pipeline with the given MMR settings.
    #[must_use]
    pub fn new(mmr_lambda: f32, enable_mmr: bool) -> Self {
        Self {
            mmr_lambda,
            enable_mmr,
            semantic_cache: None,
            provenance_map: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PipelineStats::default())),
        }
    }

    /// Attach a semantic cache (builder pattern).
    #[must_use]
    pub fn with_semantic_cache(mut self, cache: Arc<Mutex<InMemorySemanticCache>>) -> Self {
        self.semantic_cache = Some(cache);
        self
    }

    /// Share a provenance map with an [`IndexingPipeline`] (builder pattern).
    #[must_use]
    pub fn with_provenance(
        mut self,
        provenance_map: Arc<RwLock<HashMap<Uuid, ChunkProvenance>>>,
    ) -> Self {
        self.provenance_map = provenance_map;
        self
    }

    /// Share a stats object with an [`IndexingPipeline`] (builder pattern).
    #[must_use]
    pub fn with_stats(mut self, stats: Arc<Mutex<PipelineStats>>) -> Self {
        self.stats = stats;
        self
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Search the Echo layer for documents relevant to `query_text`.
    ///
    /// The optional `query_embedding` is used for cache lookup/store and for
    /// MMR doc-embedding lookup. When `None` the semantic cache is bypassed.
    ///
    /// # Steps
    ///
    /// 1. Check semantic cache (when embedding provided).
    /// 2. On cache miss: call `echo.search(query_text, top_k, None)`.
    /// 3. If `enable_mmr`: rerank with [`MmrReranker`].
    /// 4. Enrich every result with provenance.
    /// 5. Store results in cache (when embedding provided and cache present).
    /// 6. Update stats.
    ///
    /// # Errors
    ///
    /// - [`DocumentPipelineError::SearchFailed`] — Echo layer returned an error.
    /// - [`DocumentPipelineError::CacheError`] — cache lookup raised a panic
    ///   (should never happen with `InMemorySemanticCache`).
    pub async fn search<E: Echo>(
        &self,
        echo: &mut E,
        query_text: &str,
        top_k: usize,
        query_embedding: Option<Vec<f32>>,
    ) -> Result<Vec<DocumentAwareResult>, DocumentPipelineError> {
        // ── 1. Semantic cache lookup ──────────────────────────────────────────
        if let (Some(cache_arc), Some(emb)) = (&self.semantic_cache, &query_embedding) {
            let cache_guard = cache_arc.lock().await;
            if let Some(cached) = cache_guard.lookup(emb).await {
                drop(cache_guard);
                // Cache hit — enrich the cached search results with provenance.
                let mut stats = self.stats.lock().await;
                stats.total_searches += 1;
                stats.cache_hits += 1;
                drop(stats);

                let results = self.enrich_with_provenance(cached.search_results).await;
                return Ok(results);
            }
        }

        // ── 2. Echo search ────────────────────────────────────────────────────
        let raw_results = echo
            .search(query_text, top_k, None)
            .await
            .map_err(|e| DocumentPipelineError::SearchFailed(e.to_string()))?;

        // ── 3. MMR reranking ──────────────────────────────────────────────────
        let reranked: Vec<SearchResult> = if self.enable_mmr && !raw_results.is_empty() {
            let mmr_cfg = MmrConfig {
                lambda: self.mmr_lambda,
                top_k,
            };
            let reranker = MmrReranker::new(mmr_cfg);
            // Build a placeholder embedding map: all doc embeddings are zero
            // (MMR will fall back to the original score when embeddings are absent).
            // In production you would pass real doc embeddings; here we use the
            // query embedding or empty so that the MMR ordering at least shuffles
            // by relevance score.
            let doc_embeddings: HashMap<String, Vec<f32>> = HashMap::new();
            let q_emb = query_embedding.clone().unwrap_or_default();
            reranker.rerank(&raw_results, &q_emb, &doc_embeddings)
        } else {
            raw_results
        };

        // ── 4. Provenance enrichment ──────────────────────────────────────────
        let enriched = self.enrich_with_provenance(reranked.clone()).await;

        // ── 5. Store in cache ─────────────────────────────────────────────────
        if let (Some(cache_arc), Some(emb)) = (&self.semantic_cache, query_embedding) {
            let mut cache_guard = cache_arc.lock().await;
            // Build a minimal PipelineOutput to store in the cache.
            let query_obj = Query::new(query_text).with_top_k(top_k);
            let draft = Draft::new(query_text, query_text);
            let mut output = PipelineOutput::new(query_obj, draft);
            output.search_results = reranked;
            cache_guard.store(emb, query_text, output).await;
        }

        // ── 6. Stats ──────────────────────────────────────────────────────────
        let mut stats = self.stats.lock().await;
        stats.total_searches += 1;

        Ok(enriched)
    }

    /// Return a snapshot of the current pipeline statistics.
    pub async fn stats(&self) -> PipelineStats {
        self.stats.lock().await.clone()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Enrich a list of `SearchResult`s with provenance data.
    async fn enrich_with_provenance(&self, results: Vec<SearchResult>) -> Vec<DocumentAwareResult> {
        let pmap = self.provenance_map.read().await;

        results
            .into_iter()
            .map(|sr| {
                // Try to parse the document ID as a UUID and look up provenance.
                let prov = sr
                    .document
                    .id
                    .as_str()
                    .parse::<Uuid>()
                    .ok()
                    .and_then(|uuid| pmap.get(&uuid).cloned());

                let source_doc_id = prov.as_ref().map(|p| p.source_doc_id);
                let chunk_index = prov.as_ref().map(|p| p.chunk_index);

                DocumentAwareResult {
                    search_result: sr,
                    source_doc_id,
                    chunk_index,
                }
            })
            .collect()
    }
}

// ── DocumentPipelineBuilder ───────────────────────────────────────────────────

/// Fluent builder for constructing an `(IndexingPipeline, RetrievalPipeline)`
/// pair that share provenance and stats.
pub struct DocumentPipelineBuilder {
    indexing_config: IndexingConfig,
    mmr_lambda: f32,
    enable_mmr: bool,
    semantic_cache: Option<Arc<Mutex<InMemorySemanticCache>>>,
}

impl Default for DocumentPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentPipelineBuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            indexing_config: IndexingConfig::default(),
            mmr_lambda: 0.5,
            enable_mmr: true,
            semantic_cache: None,
        }
    }

    /// Set the indexing configuration.
    #[must_use]
    pub fn with_indexing_config(mut self, config: IndexingConfig) -> Self {
        self.indexing_config = config;
        self
    }

    /// Set the MMR lambda parameter.
    #[must_use]
    pub fn with_mmr(mut self, lambda: f32) -> Self {
        self.mmr_lambda = lambda;
        self
    }

    /// Attach an `InMemorySemanticCache` with the given threshold and capacity.
    #[must_use]
    pub fn with_semantic_cache(mut self, threshold: f32, capacity: usize) -> Self {
        let cfg = SemanticCacheConfig::default()
            .with_threshold(threshold)
            .with_max_entries(capacity);
        let cache = InMemorySemanticCache::new(cfg);
        self.semantic_cache = Some(Arc::new(Mutex::new(cache)));
        self
    }

    /// Build both pipeline components, sharing provenance and stats.
    #[must_use]
    pub fn build(self) -> (IndexingPipeline, RetrievalPipeline) {
        let indexing = IndexingPipeline::new(self.indexing_config);

        let pmap = indexing.provenance_map();
        let stats = indexing.stats_handle();

        let mut retrieval = RetrievalPipeline::new(self.mmr_lambda, self.enable_mmr)
            .with_provenance(pmap)
            .with_stats(stats);

        if let Some(cache) = self.semantic_cache {
            retrieval = retrieval.with_semantic_cache(cache);
        }

        (indexing, retrieval)
    }
}
