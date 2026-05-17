//! Tests for the streaming pipeline module.

#[cfg(all(test, feature = "native"))]
mod inner {
    use crate::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
    use crate::layer2_speculator::RuleBasedSpeculator;
    use crate::layer3_judge::{AdvancedClaimExtractor, JudgeConfig, JudgeImpl, MockSmtVerifier};
    use crate::pipeline::{Pipeline, PipelineConfig};
    use crate::streaming::progress::ProgressReporter;
    use crate::streaming::types::{ChunkMetadata, ChunkType, PipelineChunk};
    use crate::streaming::wrapper::{
        StreamingPipeline, StreamingPipelineResult, StreamingPipelineWrapper,
    };
    use crate::types::{Document, Query, SpeculationDecision};
    use tokio::sync::mpsc;

    use crate::pipeline::RagPipeline;

    type TestPipeline = Pipeline<
        EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
        RuleBasedSpeculator,
        JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
    >;

    fn create_test_pipeline() -> TestPipeline {
        let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));
        let speculator = RuleBasedSpeculator::default();
        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );
        Pipeline::new(
            echo,
            speculator,
            judge,
            PipelineConfig {
                enable_fast_path: false,
                ..Default::default()
            },
        )
    }

    #[tokio::test]
    async fn test_pipeline_chunk_creation() {
        let chunk = PipelineChunk::new(0, ChunkType::SearchStarted, "test content")
            .with_timestamp(100)
            .with_layer("Echo")
            .with_confidence(0.9);

        assert_eq!(chunk.chunk_id, 0);
        assert_eq!(chunk.content, "test content");
        assert_eq!(chunk.metadata.timestamp_ms, 100);
        assert_eq!(chunk.metadata.layer, Some("Echo".to_string()));
        assert_eq!(chunk.metadata.confidence, Some(0.9));
    }

    #[tokio::test]
    async fn test_chunk_metadata_builder() {
        let metadata = ChunkMetadata::new(50)
            .with_layer("Speculator")
            .with_confidence(0.85)
            .with_duration(200);

        assert_eq!(metadata.timestamp_ms, 50);
        assert_eq!(metadata.layer, Some("Speculator".to_string()));
        assert_eq!(metadata.confidence, Some(0.85));
        assert_eq!(metadata.duration_ms, Some(200));
    }

    #[tokio::test]
    async fn test_chunk_type_equality() {
        assert_eq!(ChunkType::SearchStarted, ChunkType::SearchStarted);
        assert_eq!(
            ChunkType::SearchResult {
                rank: 0,
                score: 0.9
            },
            ChunkType::SearchResult {
                rank: 0,
                score: 0.9
            }
        );
        assert_ne!(ChunkType::SearchStarted, ChunkType::DraftGenerated);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_creation() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline).with_buffer_size(64);

        assert_eq!(wrapper.buffer_size(), 64);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_buffer_minimum() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline).with_buffer_size(0);

        assert_eq!(wrapper.buffer_size(), 1);
    }

    #[tokio::test]
    async fn test_streaming_empty_index() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("What is the meaning of life?");

        let mut result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");

        let mut chunks = Vec::new();
        while let Some(chunk) = result.next().await {
            chunks.push(chunk);
        }

        // Should have at least search started, search completed, draft generated, and final answer
        assert!(chunks.len() >= 4);
        assert!(matches!(chunks[0].chunk_type, ChunkType::SearchStarted));
    }

    #[tokio::test]
    async fn test_streaming_with_documents() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("The capital of France is Paris."))
            .await
            .expect("test operation should succeed");

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("What is the capital of France?");

        let mut result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");

        let mut has_search_result = false;
        let mut has_final_answer = false;

        while let Some(chunk) = result.next().await {
            if matches!(chunk.chunk_type, ChunkType::SearchResult { .. }) {
                has_search_result = true;
            }
            if matches!(chunk.chunk_type, ChunkType::FinalAnswer) {
                has_final_answer = true;
            }
        }

        assert!(has_search_result);
        assert!(has_final_answer);
    }

    #[tokio::test]
    async fn test_streaming_chunk_ordering() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Test document content."))
            .await
            .expect("test operation should succeed");

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let mut result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");

        let mut chunks = Vec::new();
        while let Some(chunk) = result.next().await {
            chunks.push(chunk);
        }

        // Verify chunk IDs are in order
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i);
        }
    }

    #[tokio::test]
    async fn test_streaming_into_stream() {
        use futures::StreamExt;

        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Stream test document."))
            .await
            .expect("test operation should succeed");

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("stream");

        let result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");
        let mut stream = result.into_stream();

        let mut count = 0;
        while let Some(chunk) = stream.next().await {
            count += 1;
            assert!(chunk.chunk_id < 100); // Sanity check
        }

        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_streaming_batch() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("Alpha document."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("Beta document."))
            .await
            .expect("test operation should succeed");

        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let queries = vec![Query::new("alpha"), Query::new("beta")];

        let results = wrapper.process_batch_streaming(queries).await;

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_streaming_collected_chunks() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let mut result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");

        // Consume some chunks
        let _ = result.next().await;
        let _ = result.next().await;

        assert!(result.collected_chunks().len() >= 2);
        assert!(result.chunk_count() >= 2);
    }

    #[tokio::test]
    async fn test_streaming_for_each() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);
        let query = Query::new("test");

        let result = wrapper
            .process_streaming(query)
            .await
            .expect("test operation should succeed");

        let mut count = 0;
        result
            .for_each(|_chunk| {
                count += 1;
            })
            .await;

        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_progress_reporter_basic() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_search_started("test query").await;
        reporter.report_search_result(0, 0.9, "Test document").await;
        reporter.report_search_completed(1).await;

        let chunk1 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(chunk1.chunk_type, ChunkType::SearchStarted));

        let chunk2 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::SearchResult { rank: 0, .. }
        ));

        let chunk3 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::SearchCompleted { total: 1 }
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_speculation() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_speculation_started().await;
        reporter.report_speculation("verification", 0.8).await;
        reporter
            .report_speculation_decision(SpeculationDecision::Accept, 0.9)
            .await;

        let chunk1 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(chunk1.chunk_type, ChunkType::SpeculationStarted));

        let chunk2 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::SpeculationProgress { .. }
        ));

        let chunk3 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::SpeculationDecision(SpeculationDecision::Accept)
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_verification() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_verification_started().await;
        reporter.report_claim(0, "Test claim").await;
        reporter
            .report_claim_verified(0, "Verified", "Explanation")
            .await;
        reporter
            .report_verification_completed("Summary", 0.85, 100)
            .await;

        let chunk1 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(chunk1.chunk_type, ChunkType::VerificationStarted));

        let chunk2 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk2.chunk_type,
            ChunkType::ClaimExtracted { claim_id: 0 }
        ));

        let chunk3 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk3.chunk_type,
            ChunkType::ClaimVerified { claim_id: 0, .. }
        ));

        let chunk4 = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(
            chunk4.chunk_type,
            ChunkType::VerificationCompleted
        ));
    }

    #[tokio::test]
    async fn test_progress_reporter_error() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_error("Test error").await;

        let chunk = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(chunk.chunk_type, ChunkType::Error(_)));
    }

    #[tokio::test]
    async fn test_progress_reporter_final_answer() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        reporter.report_final("The answer is 42", 0.95).await;

        let chunk = rx.recv().await.expect("test operation should succeed");
        assert!(matches!(chunk.chunk_type, ChunkType::FinalAnswer));
        assert_eq!(chunk.content, "The answer is 42");
        assert_eq!(chunk.metadata.confidence, Some(0.95));
    }

    #[tokio::test]
    async fn test_progress_reporter_chunk_counter() {
        let (tx, _rx) = mpsc::channel(32);
        let mut reporter = ProgressReporter::new(tx);

        assert_eq!(reporter.chunk_count(), 0);

        reporter.report(ChunkType::SearchStarted, "test").await;
        assert_eq!(reporter.chunk_count(), 1);

        reporter.report(ChunkType::DraftGenerated, "test").await;
        assert_eq!(reporter.chunk_count(), 2);
    }

    #[tokio::test]
    async fn test_truncate_content() {
        use crate::streaming::types::truncate_content;
        assert_eq!(truncate_content("short", 10), "short");
        assert_eq!(
            truncate_content("a longer string that needs truncation", 10),
            "a longe..."
        );
        assert_eq!(truncate_content("exactly10c", 10), "exactly10c");
    }

    #[tokio::test]
    async fn test_streaming_result_from_output() {
        let query = Query::new("test");
        let draft = crate::types::Draft::new("Test answer", "test");
        let output = crate::types::PipelineOutput::new(query, draft);

        let result = StreamingPipelineResult::from_output(output);
        assert!(result.has_final_output());
    }

    #[tokio::test]
    async fn test_chunk_with_metadata() {
        let metadata = ChunkMetadata::new(100)
            .with_layer("Test")
            .with_confidence(0.5)
            .with_duration(50);

        let chunk = PipelineChunk::new(0, ChunkType::SearchStarted, "test").with_metadata(metadata);

        assert_eq!(chunk.metadata.timestamp_ms, 100);
        assert_eq!(chunk.metadata.layer, Some("Test".to_string()));
        assert_eq!(chunk.metadata.confidence, Some(0.5));
        assert_eq!(chunk.metadata.duration_ms, Some(50));
    }

    #[tokio::test]
    async fn test_concurrent_streaming_queries() {
        let mut pipeline = create_test_pipeline();
        pipeline
            .index(Document::new("First document content."))
            .await
            .expect("test operation should succeed");
        pipeline
            .index(Document::new("Second document content."))
            .await
            .expect("test operation should succeed");

        let wrapper = std::sync::Arc::new(StreamingPipelineWrapper::new(pipeline));

        let handles: Vec<_> = (0..3)
            .map(|i| {
                let wrapper = wrapper.clone();
                tokio::spawn(async move {
                    let query = Query::new(format!("query {i}"));
                    let mut result = wrapper
                        .process_streaming(query)
                        .await
                        .expect("test operation should succeed");
                    let mut count = 0;
                    while let Some(_chunk) = result.next().await {
                        count += 1;
                    }
                    count
                })
            })
            .collect();

        for handle in handles {
            let count = handle.await.expect("test operation should succeed");
            assert!(count > 0);
        }
    }

    #[tokio::test]
    async fn test_streaming_wrapper_inner_access() {
        let pipeline = create_test_pipeline();
        let wrapper = StreamingPipelineWrapper::new(pipeline);

        // Test inner() returns a reference
        let _ = wrapper.inner();

        // Test config access through inner
        let config = wrapper.inner().config();
        assert!(!config.enable_fast_path);
    }

    #[tokio::test]
    async fn test_streaming_wrapper_inner_mut() {
        let pipeline = create_test_pipeline();
        let mut wrapper = StreamingPipelineWrapper::new(pipeline);

        // Test inner_mut() returns a mutable reference
        let _inner = wrapper.inner_mut();
    }
}
