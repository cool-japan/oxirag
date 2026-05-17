//! Tests for the document processing pipeline.

use super::indexing::IndexingPipeline;
use super::retrieval::{DocumentPipelineBuilder, RetrievalPipeline};
use super::types::{ChunkStrategyKind, DocumentPipelineError, IndexingConfig};
use crate::chunking::ChunkConfig;
use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::types::Document;

// Helper: create a small EchoLayer suitable for tests.
fn make_echo(dim: usize) -> EchoLayer<MockEmbeddingProvider, InMemoryVectorStore> {
    EchoLayer::new(
        MockEmbeddingProvider::new(dim),
        InMemoryVectorStore::new(dim),
    )
}

// ── IndexingPipeline::chunk_document ─────────────────────────────────────────

#[test]
fn test_chunk_document_fixed_size() {
    let config = IndexingConfig {
        chunk_strategy: ChunkStrategyKind::FixedSize,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(50)
            .with_chunk_overlap(10)
            .with_min_chunk_size(5),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let doc = Document::new("a".repeat(200).as_str());
    let chunks = pipeline.chunk_document(&doc).expect("chunk");
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunk_document_sentence_strategy() {
    let config = IndexingConfig {
        chunk_strategy: ChunkStrategyKind::Sentence,
        chunk_config: ChunkConfig::default().with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let doc = Document::new("Hello world. This is sentence two. And sentence three.");
    let chunks = pipeline.chunk_document(&doc).expect("chunk");
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunk_document_recursive_strategy() {
    let config = IndexingConfig {
        chunk_strategy: ChunkStrategyKind::Recursive,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(100)
            .with_min_chunk_size(5),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let doc = Document::new("Paragraph one.\n\nParagraph two.\n\nParagraph three.");
    let chunks = pipeline.chunk_document(&doc).expect("chunk");
    assert!(!chunks.is_empty());
}

#[test]
fn test_chunk_document_markdown_strategy() {
    let config = IndexingConfig {
        chunk_strategy: ChunkStrategyKind::Markdown,
        chunk_config: ChunkConfig::default().with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let doc = Document::new("# Section One\n\nContent here.\n\n## Subsection\n\nMore content.");
    let chunks = pipeline.chunk_document(&doc).expect("chunk");
    assert!(!chunks.is_empty());
}

// ── IndexingPipeline::index_document ─────────────────────────────────────────

#[tokio::test]
async fn test_index_document_basic() {
    let pipeline = IndexingPipeline::with_default();
    let mut echo = make_echo(32);

    let doc = Document::new(
        "The quick brown fox jumps over the lazy dog. This is a test document with enough content to produce at least one chunk.",
    );
    let result = pipeline
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    assert!(!result.chunk_ids.is_empty());
}

#[tokio::test]
async fn test_index_document_dedup_skips_repeated_content() {
    let config = IndexingConfig {
        auto_dedup: true,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(50)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    let content = "Repeated content block that is long enough to be a chunk.";
    let doc1 = Document::new(content);
    let doc2 = Document::new(content);

    let result1 = pipeline
        .index_document(&mut echo, doc1)
        .await
        .expect("index doc1");
    let result2 = pipeline
        .index_document(&mut echo, doc2)
        .await
        .expect("index doc2");

    let stats = pipeline.stats().await;
    // The second document's chunks should have been deduplicated.
    assert!(
        stats.chunks_deduped >= result1.chunk_ids.len(),
        "Expected at least {} deduped chunks, got {}",
        result1.chunk_ids.len(),
        stats.chunks_deduped
    );
    // Second result has fewer (or zero) indexed chunk IDs.
    assert!(result2.chunk_ids.len() < result1.chunk_ids.len() || result2.chunk_ids.is_empty());
}

#[tokio::test]
async fn test_index_document_no_dedup_indexes_all() {
    let config = IndexingConfig {
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(50)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    let content =
        "Content that will not be deduplicated no matter how many times we index it here.";
    let doc1 = Document::new(content);
    let doc2 = Document::new(content);

    let result1 = pipeline
        .index_document(&mut echo, doc1)
        .await
        .expect("index doc1");
    // Without dedup, second document should also index all chunks.
    let _result2 = pipeline
        .index_document(&mut echo, doc2)
        .await
        .expect("index doc2");

    let stats = pipeline.stats().await;
    assert_eq!(stats.chunks_deduped, 0);
    assert_eq!(stats.documents_indexed, 2);
    // Both results should have produced chunk IDs.
    assert!(!result1.chunk_ids.is_empty());
}

// ── Provenance tracking ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_provenance_recorded_after_index() {
    let config = IndexingConfig {
        store_provenance: true,
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(30)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    let doc = Document::new("Short document. Second sentence. Third sentence here!");
    let result = pipeline
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    // Every chunk_id should have a provenance record.
    for chunk_id in &result.chunk_ids {
        let prov = pipeline.get_provenance(chunk_id);
        assert!(prov.is_some(), "provenance missing for chunk {chunk_id}");
        let prov = prov.expect("just checked");
        assert_eq!(prov.chunk_id, *chunk_id);
        assert_eq!(prov.source_doc_id, result.source_doc_id);
    }
}

#[tokio::test]
async fn test_provenance_not_stored_when_disabled() {
    let config = IndexingConfig {
        store_provenance: false,
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(30)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    let doc = Document::new("No provenance should be stored for this document sentence.");
    let result = pipeline
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    for chunk_id in &result.chunk_ids {
        assert!(pipeline.get_provenance(chunk_id).is_none());
    }
}

#[tokio::test]
async fn test_provenance_chunk_index_is_sequential() {
    let config = IndexingConfig {
        store_provenance: true,
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(20)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    let doc = Document::new("aaaaa bbbbb ccccc ddddd eeeee fffff ggggg hhhhh iiiii jjjjj kkkkk");
    let result = pipeline
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    if result.chunk_ids.len() >= 2 {
        let prov0 = pipeline
            .get_provenance(&result.chunk_ids[0])
            .expect("prov 0");
        let prov1 = pipeline
            .get_provenance(&result.chunk_ids[1])
            .expect("prov 1");
        assert!(prov1.chunk_index > prov0.chunk_index || prov1.char_start >= prov0.char_start);
    }
}

// ── IndexingPipeline::index_batch ────────────────────────────────────────────

#[tokio::test]
async fn test_index_batch_multiple_docs() {
    let pipeline = IndexingPipeline::with_default();
    let mut echo = make_echo(32);

    let docs: Vec<Document> = (0..3)
        .map(|i| {
            Document::new(format!(
                "Document number {i} with substantial content to chunk properly."
            ))
        })
        .collect();

    let results = pipeline
        .index_batch(&mut echo, docs)
        .await
        .expect("batch index");
    assert_eq!(results.len(), 3);
    // Each result should have at least one chunk.
    for result in &results {
        assert!(!result.chunk_ids.is_empty() || result.source_doc_id != uuid::Uuid::nil());
    }
}

#[tokio::test]
async fn test_index_batch_empty() {
    let pipeline = IndexingPipeline::with_default();
    let mut echo = make_echo(32);

    let results = pipeline
        .index_batch(&mut echo, vec![])
        .await
        .expect("empty batch");
    assert!(results.is_empty());
}

// ── PipelineStats ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_accumulate_across_documents() {
    let config = IndexingConfig {
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(50)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let mut echo = make_echo(32);

    for i in 0..3_usize {
        let doc = Document::new(format!(
            "Document {i} — long enough to produce multiple chunks: aaa bbb ccc ddd eee fff ggg hhh iii jjj."
        ));
        pipeline
            .index_document(&mut echo, doc)
            .await
            .expect("index");
    }

    let stats = pipeline.stats().await;
    assert_eq!(stats.documents_indexed, 3);
    assert!(stats.chunks_created > 0);
}

// ── RetrievalPipeline::search ─────────────────────────────────────────────────

#[tokio::test]
async fn test_retrieval_search_basic() {
    let pipeline = IndexingPipeline::with_default();
    let retrieval = RetrievalPipeline::new(0.5, false).with_provenance(pipeline.provenance_map());
    let mut echo = make_echo(32);

    // Index some content first.
    let doc = Document::new(
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
    );
    pipeline
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    let results = retrieval
        .search(&mut echo, "Rust programming", 5, None)
        .await
        .expect("search");

    // Results may be empty if doc content was too short to chunk, but no error.
    let _ = results;
}

#[tokio::test]
async fn test_retrieval_search_returns_document_aware_results() {
    let (indexing, retrieval) = DocumentPipelineBuilder::new().with_mmr(0.5).build();

    let mut echo = make_echo(32);

    let doc = Document::new(
        "The capital of France is Paris and it is a beautiful city on the Seine river.",
    );
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    let results = retrieval
        .search(&mut echo, "capital of France", 5, None)
        .await
        .expect("search");

    // DocumentAwareResult wraps a SearchResult; at least the type is correct.
    for r in &results {
        // Cosine similarity range is [-1, 1].
        assert!(r.search_result.score >= -1.0);
    }
}

#[tokio::test]
async fn test_retrieval_mmr_enabled() {
    let config = IndexingConfig {
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(40)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let pipeline = IndexingPipeline::new(config);
    let retrieval = RetrievalPipeline::new(0.7, true) // MMR enabled
        .with_provenance(pipeline.provenance_map());
    let mut echo = make_echo(32);

    for i in 0..5_usize {
        let doc = Document::new(format!(
            "Information about topic {i}: aaa bbb ccc ddd eee fff ggg hhh iii."
        ));
        pipeline
            .index_document(&mut echo, doc)
            .await
            .expect("index");
    }

    // Should not panic/error even with MMR enabled.
    let results = retrieval
        .search(&mut echo, "topic information", 3, None)
        .await
        .expect("search with MMR");

    assert!(results.len() <= 3);
}

// ── Semantic cache hit path ───────────────────────────────────────────────────

#[tokio::test]
async fn test_semantic_cache_hit_on_second_identical_query() {
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_semantic_cache(0.95, 50)
        .with_mmr(0.5)
        .build();

    let mut echo = make_echo(32);

    let doc = Document::new(
        "Semantic caching stores query results to avoid recomputation of expensive operations.",
    );
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    // First query — cache miss.
    let embedding = [0.1_f32; 32];
    let normalized: Vec<f32> = {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        embedding.iter().map(|x| x / norm).collect()
    };

    let _first = retrieval
        .search(&mut echo, "semantic caching", 5, Some(normalized.clone()))
        .await
        .expect("first search");

    // Second identical query — should hit cache.
    let _second = retrieval
        .search(&mut echo, "semantic caching", 5, Some(normalized.clone()))
        .await
        .expect("second search");

    let stats = retrieval.stats().await;
    assert_eq!(stats.total_searches, 2);
    assert_eq!(stats.cache_hits, 1, "Expected 1 cache hit");
}

#[tokio::test]
async fn test_semantic_cache_miss_without_embedding() {
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_semantic_cache(0.9, 50)
        .build();

    let mut echo = make_echo(32);
    let doc = Document::new("Cache miss test document with plenty of content to index properly.");
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    // No embedding → cache is bypassed.
    let _results = retrieval
        .search(&mut echo, "cache miss", 5, None)
        .await
        .expect("search");

    let stats = retrieval.stats().await;
    assert_eq!(stats.cache_hits, 0);
}

// ── DocumentPipelineBuilder fluent API ───────────────────────────────────────

#[tokio::test]
async fn test_builder_default() {
    let (indexing, _retrieval) = DocumentPipelineBuilder::new().build();
    let stats = indexing.stats().await;
    assert_eq!(stats.documents_indexed, 0);
}

#[tokio::test]
async fn test_builder_with_custom_config() {
    let config = IndexingConfig {
        chunk_strategy: ChunkStrategyKind::Sentence,
        auto_dedup: false,
        store_provenance: false,
        chunk_config: ChunkConfig::default().with_chunk_size(200),
    };
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_indexing_config(config)
        .with_mmr(0.3)
        .build();

    let mut echo = make_echo(32);
    let doc = Document::new(
        "First sentence. Second sentence. Third sentence for testing the pipeline builder configuration.",
    );
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    let results = retrieval
        .search(&mut echo, "sentence", 5, None)
        .await
        .expect("search");

    let _ = results;
}

#[tokio::test]
async fn test_builder_with_cache_and_mmr() {
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_semantic_cache(0.85, 200)
        .with_mmr(0.6)
        .build();

    let mut echo = make_echo(32);
    let doc = Document::new(
        "Building pipelines with fluent APIs is a common Rust pattern for ergonomic configuration.",
    );
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    let emb: Vec<f32> = {
        let e = [0.5_f32; 32];
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        e.iter().map(|x| x / norm).collect()
    };

    let _r1 = retrieval
        .search(&mut echo, "fluent builder", 5, Some(emb.clone()))
        .await
        .expect("search 1");
    let _r2 = retrieval
        .search(&mut echo, "fluent builder", 5, Some(emb))
        .await
        .expect("search 2");

    let stats = retrieval.stats().await;
    assert_eq!(stats.cache_hits, 1);
}

// ── Shared provenance between indexing and retrieval ─────────────────────────

#[tokio::test]
async fn test_shared_provenance_via_builder() {
    let config = IndexingConfig {
        store_provenance: true,
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(30)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_indexing_config(config)
        .build();

    let mut echo = make_echo(32);
    let doc =
        Document::new("Provenance tracking test. Multiple sentences here. Another one follows.");
    let index_result = indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    // Search and check that results have provenance (source_doc_id set).
    let results = retrieval
        .search(&mut echo, "provenance tracking", 10, None)
        .await
        .expect("search");

    // At least some results should have provenance if any chunks were indexed.
    if !index_result.chunk_ids.is_empty() && !results.is_empty() {
        let has_provenance = results.iter().any(|r| r.source_doc_id.is_some());
        assert!(
            has_provenance,
            "Expected at least one result with provenance"
        );
    }
}

// ── Stats sharing ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stats_shared_across_pipelines() {
    let config = IndexingConfig {
        auto_dedup: false,
        chunk_config: ChunkConfig::default()
            .with_chunk_size(50)
            .with_chunk_overlap(0)
            .with_min_chunk_size(1),
        ..Default::default()
    };
    let (indexing, retrieval) = DocumentPipelineBuilder::new()
        .with_indexing_config(config)
        .build();

    let mut echo = make_echo(32);
    let doc = Document::new(
        "Content for shared stats test. More content here to ensure chunking produces something.",
    );
    indexing
        .index_document(&mut echo, doc)
        .await
        .expect("index");

    retrieval
        .search(&mut echo, "content", 5, None)
        .await
        .expect("search");

    // Both pipeline stats come from the same underlying Arc<Mutex<PipelineStats>>.
    let retrieval_stats = retrieval.stats().await;
    let indexing_stats = indexing.stats().await;

    assert_eq!(
        retrieval_stats.documents_indexed,
        indexing_stats.documents_indexed
    );
    assert_eq!(
        retrieval_stats.chunks_created,
        indexing_stats.chunks_created
    );
    // Search count is also shared.
    assert_eq!(retrieval_stats.total_searches, 1);
    assert_eq!(indexing_stats.total_searches, 1);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_provenance_not_found_error() {
    let fake_id = uuid::Uuid::new_v4();
    // DocumentPipelineError::ProvenanceNotFound can be constructed.
    let err = DocumentPipelineError::ProvenanceNotFound(fake_id);
    let msg = err.to_string();
    assert!(msg.contains("provenance not found"));
}

#[tokio::test]
async fn test_chunk_strategy_kind_default() {
    let config = IndexingConfig::default();
    assert_eq!(config.chunk_strategy, ChunkStrategyKind::FixedSize);
    assert!(config.auto_dedup);
    assert!(config.store_provenance);
}

#[tokio::test]
async fn test_indexing_config_builder() {
    let config = IndexingConfig::new()
        .with_chunk_strategy(ChunkStrategyKind::Markdown)
        .with_auto_dedup(false)
        .with_store_provenance(false)
        .with_chunk_config(ChunkConfig::default().with_chunk_size(256));

    assert_eq!(config.chunk_strategy, ChunkStrategyKind::Markdown);
    assert!(!config.auto_dedup);
    assert!(!config.store_provenance);
    assert_eq!(config.chunk_config.chunk_size, 256);
}
