//! `OxiRAG` - A three-layer RAG engine with SMT-based logic verification.
//!
//! `OxiRAG` provides a robust Retrieval-Augmented Generation (RAG) pipeline with:
//!
//! - **Layer 1 (Echo)**: Semantic search using vector embeddings
//! - **Layer 2 (Speculator)**: Draft verification using small language models
//! - **Layer 3 (Judge)**: Logic verification using SMT solvers
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use oxirag::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OxiRagError> {
//!     // Create the Echo layer with mock embedding provider
//!     let echo = EchoLayer::new(
//!         MockEmbeddingProvider::new(384),
//!         InMemoryVectorStore::new(384),
//!     );
//!
//!     // Create the Speculator layer
//!     let speculator = RuleBasedSpeculator::default();
//!
//!     // Create the Judge layer
//!     let judge = JudgeImpl::new(
//!         AdvancedClaimExtractor::new(),
//!         MockSmtVerifier::default(),
//!         JudgeConfig::default(),
//!     );
//!
//!     // Build the pipeline
//!     let mut pipeline = PipelineBuilder::new()
//!         .with_echo(echo)
//!         .with_speculator(speculator)
//!         .with_judge(judge)
//!         .build()?;
//!
//!     // Index documents
//!     pipeline.index(Document::new("The capital of France is Paris.")).await?;
//!
//!     // Query the pipeline
//!     let query = Query::new("What is the capital of France?");
//!     let result = pipeline.process(query).await?;
//!
//!     println!("Answer: {}", result.final_answer);
//!     println!("Confidence: {:.2}", result.confidence);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - `echo` (default): Enable Layer 1 with numrs2 for SIMD similarity
//! - `speculator` (default): Enable Layer 2 with Candle for SLM inference
//! - `judge` (default): Enable Layer 3 with `OxiZ` for SMT verification
//! - `cuda`: Enable CUDA acceleration for Candle models
//! - `metal`: Enable Metal acceleration for Candle models
//!
//! # Architecture
//!
//! ```text
//! Query
//!   │
//!   ▼
//! ┌─────────────────┐
//! │  Layer 1: Echo  │  ← Semantic search with embeddings
//! │  (Vector Store) │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────────┐
//! │ Layer 2: Speculator │  ← Draft verification with SLM
//! │  (Draft Checker)    │
//! └─────────┬───────────┘
//!           │
//!           ▼
//! ┌─────────────────┐
//! │  Layer 3: Judge │  ← Logic verification with SMT
//! │  (SMT Solver)   │
//! └────────┬────────┘
//!          │
//!          ▼
//!       Response
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(unexpected_cfgs)]

pub mod circuit_breaker;
pub mod config;
pub mod connection_pool;
#[cfg(feature = "distillation")]
pub mod distillation;
pub mod error;
#[cfg(feature = "hidden-states")]
pub mod hidden_states;
pub mod hybrid_search;
pub mod index_management;
pub mod layer1_echo;
pub mod layer2_speculator;
pub mod layer3_judge;
#[cfg(feature = "graphrag")]
pub mod layer4_graph;
#[cfg(feature = "native")]
pub mod load_testing;
pub mod memory;
pub mod metrics;
pub mod pipeline;
pub mod pipeline_debug;
#[cfg(feature = "prefix-cache")]
pub mod prefix_cache;
#[cfg(feature = "quantization")]
pub mod quantization;
pub mod query_builder;
pub mod query_expansion;
pub mod relevance_feedback;
pub mod reranker;
pub mod retry;
pub mod simd_similarity;
pub mod streaming;
pub mod types;

#[cfg(feature = "wasm")]
pub mod wasm;

/// Convenient re-exports for common usage.
pub mod prelude {
    pub use crate::circuit_breaker::{
        CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOrOperationError,
        CircuitBreakerRegistry, CircuitBreakerStats, CircuitPermit, CircuitState,
        with_circuit_breaker, with_service_circuit_breaker,
    };
    pub use crate::config::{
        EchoConfig, JudgeConfig as JudgeCfg, OxiRagConfig, PipelineConfig as PipelineCfg,
        RetryConfig, SimilarityMetricConfig, SpeculatorConfig as SpeculatorCfg,
    };
    pub use crate::connection_pool::{
        Connection, ConnectionError, ConnectionPool, MockConnection, PoolConfig, PoolError,
        PoolStats, PooledConnection,
    };
    pub use crate::error::{
        EmbeddingError, JudgeError, OxiRagError, PipelineError, SpeculatorError, VectorStoreError,
    };
    pub use crate::layer1_echo::{
        Echo, EchoLayer, EmbeddingProvider, InMemoryVectorStore, IndexedDocument, MetadataFilter,
        MockEmbeddingProvider, SimilarityMetric, VectorStore,
    };
    pub use crate::layer2_speculator::{RuleBasedSpeculator, Speculator, SpeculatorConfig};
    pub use crate::layer3_judge::{
        AdvancedClaimExtractor, ClaimExtractor, Judge, JudgeConfig, JudgeImpl, MockSmtVerifier,
        SmtVerifier,
    };
    pub use crate::memory::{
        MemoryBreakdown, MemoryBudget, MemoryComponent, MemoryError, MemoryGuard, MemoryMonitor,
        MemoryStats,
    };
    pub use crate::metrics::{LayerTiming, MetricsCollector, PipelineMetrics, TimedOperation};
    pub use crate::pipeline::{Pipeline, PipelineBuilder, PipelineConfig, RagPipeline};
    pub use crate::query_builder::{ExtendedQuery, LayerHints, QueryBuilder};
    pub use crate::retry::RetryPolicy;
    pub use crate::simd_similarity::{
        SimdBackend, SimilarityEngine, detect_backend, simd_batch_cosine, simd_cosine_similarity,
        simd_dot_product, simd_euclidean_distance, simd_l2_norm,
    };
    pub use crate::types::{
        ClaimStructure, ClaimVerificationResult, ComparisonOp, Document, DocumentId, Draft,
        LogicalClaim, PipelineOutput, Quantifier, Query, SearchResult, SpeculationDecision,
        SpeculationResult, VerificationResult, VerificationStatus,
    };

    // Index management exports
    pub use crate::index_management::{
        IndexManagement, IndexManager, IndexSnapshot, IndexStats, MergeResult, OptimizeConfig,
        OptimizeResult, SerializedDocument, SerializedIndex, VacuumResult,
    };

    // Streaming pipeline exports
    #[cfg(feature = "native")]
    pub use crate::streaming::ProgressReporter;
    pub use crate::streaming::{
        ChunkMetadata, ChunkType, PipelineChunk, StreamingPipeline, StreamingPipelineResult,
        StreamingPipelineWrapper,
    };

    #[cfg(feature = "speculator")]
    pub use crate::layer1_echo::CandleEmbeddingProvider;
    #[cfg(feature = "speculator")]
    pub use crate::layer2_speculator::CandleSlmSpeculator;
    #[cfg(feature = "judge")]
    pub use crate::layer3_judge::OxizVerifier;

    // Graph layer exports
    #[cfg(feature = "graphrag")]
    pub use crate::config::GraphConfig;
    #[cfg(feature = "graphrag")]
    pub use crate::error::GraphError;
    #[cfg(feature = "graphrag")]
    pub use crate::layer4_graph::{
        Direction, EntityExtractor, EntityId, EntityType, Graph, GraphEntity, GraphLayer,
        GraphLayerBuilder, GraphPath, GraphQuery, GraphRelationship, GraphStore,
        HybridSearchResult, InMemoryGraphStore, MockEntityExtractor, MockRelationshipExtractor,
        PatternEntityExtractor, PatternRelationshipExtractor, RelationshipExtractor,
        RelationshipType, bfs_traverse, find_entities_within_hops, find_shortest_path,
    };
    #[cfg(feature = "graphrag")]
    pub use crate::query_builder::GraphContext;

    // Distillation layer exports
    #[cfg(feature = "distillation")]
    pub use crate::distillation::{
        CandidateDetector, CandidateEvaluation, CollectorStatistics, DistillationCandidate,
        DistillationConfig, DistillationStats, DistillationTracker, InMemoryDistillationTracker,
        NearReadyReason, QAPair, QAPairCollector, QueryFrequencyTracker, QueryPattern,
        TrainingExample,
    };
    #[cfg(feature = "distillation")]
    pub use crate::error::DistillationError;

    // Prefix cache exports
    #[cfg(feature = "prefix-cache")]
    pub use crate::error::PrefixCacheError;
    #[cfg(feature = "prefix-cache")]
    pub use crate::prefix_cache::{
        CacheKey, CacheLookupResult, CacheStats, ContextFingerprint, ContextFingerprintGenerator,
        Fingerprintable, InMemoryPrefixCache, KVCacheEntry, PrefixCacheConfig, PrefixCacheExt,
        PrefixCacheStore, RollingHasher,
    };

    // Hidden states exports
    #[cfg(feature = "hidden-states")]
    pub use crate::error::HiddenStateError;
    #[cfg(feature = "hidden-states")]
    pub use crate::hidden_states::{
        AdaptiveReuseStrategy, CachedHiddenState, DType, Device, HiddenStateCache,
        HiddenStateCacheConfig, HiddenStateCacheStats, HiddenStateConfig, HiddenStateProvider,
        HiddenStateProviderExt, HiddenStateTensor, HybridReuseStrategy, KVCache, LayerExtractor,
        LayerHiddenState, LengthAwareReuseStrategy, MockHiddenStateProvider, ModelHiddenStates,
        ModelKVCache, PrefixReuseStrategy, SemanticReuseStrategy, StatePooling, StateReuseStrategy,
        StateSimilarity, TensorShape,
    };

    // Load testing exports
    #[cfg(feature = "native")]
    pub use crate::load_testing::{
        LoadTest, LoadTestBuilder, LoadTestConfig, LoadTestResult, LoadTestStats,
        MockQueryExecutor, MockQueryGenerator, QueryExecutor, QueryGenerator, RequestResult,
    };

    // Reranker exports
    pub use crate::reranker::{
        CrossEncoderReranker, FusionStrategy, HybridReranker, KeywordReranker,
        MockCrossEncoderReranker, MockReranker, Reranker, RerankerConfig, RerankerPipeline,
        RerankerPipelineBuilder, SemanticReranker,
    };

    // Pipeline debug exports
    pub use crate::pipeline_debug::{
        DebugConfig, GanttTraceFormatter, JsonTraceFormatter, LayerTraceGuard,
        MermaidTraceFormatter, PipelineDebugger, PipelineTrace, SharedPipelineDebugger,
        TextTraceFormatter, TraceEntry, TraceFormatter, TraceId, create_shared_debugger,
    };

    // Relevance feedback exports
    pub use crate::relevance_feedback::{
        FeedbackAdjuster, FeedbackConfig, FeedbackEntry, FeedbackStore, InMemoryFeedbackStore,
        RelevanceFeedback, RelevanceModel, RocchioFeedbackAdjuster, SimpleBoostAdjuster,
    };

    // Query expansion exports
    pub use crate::query_expansion::{
        CompositeExpander, ExpandedQuery, ExpansionConfig, ExpansionMethod, NGramExpander,
        PseudoRelevanceFeedback, QueryExpander, QueryReformulator, StemExpander, SynonymExpander,
    };

    // Hybrid search exports
    pub use crate::hybrid_search::{
        BM25Encoder, BM25Params, FusionStrategy as HybridFusionStrategy, HybridConfig,
        HybridResult, HybridSearcher, InMemorySparseStore, SparseVector, SparseVectorStore,
    };

    // Quantization exports
    #[cfg(feature = "quantization")]
    pub use crate::quantization::{
        BinaryQuantizer, Int4Quantizer, Int8Quantizer, MockQuantizedVectorStore,
        QuantizationConfig, QuantizationType, QuantizedDocument, QuantizedTensor,
        QuantizedVectorStore, Quantizer, compute_quantization_error, compute_snr_db,
        hamming_distance, int4_dot_product, int8_dot_product,
    };
}

pub use error::{OxiRagError, Result};

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[tokio::test]
    async fn test_full_pipeline_integration() {
        // Create all layers with mock implementations
        let echo = EchoLayer::new(MockEmbeddingProvider::new(64), InMemoryVectorStore::new(64));

        let speculator = RuleBasedSpeculator::default();

        let judge = JudgeImpl::new(
            AdvancedClaimExtractor::new(),
            MockSmtVerifier::default(),
            JudgeConfig::default(),
        );

        // Build pipeline
        let mut pipeline = PipelineBuilder::new()
            .with_echo(echo)
            .with_speculator(speculator)
            .with_judge(judge)
            .with_config(PipelineConfig {
                enable_fast_path: false,
                ..Default::default()
            })
            .build()
            .expect("Failed to build pipeline");

        // Index some documents
        let documents = vec![
            Document::new(
                "Rust is a systems programming language focused on safety and performance.",
            ),
            Document::new("The Rust compiler prevents data races at compile time."),
            Document::new("Cargo is Rust's package manager and build system."),
        ];

        pipeline
            .index_batch(documents)
            .await
            .expect("Failed to index documents");

        // Process a query
        let query = Query::new("What is Rust?").with_top_k(3);
        let result = pipeline
            .process(query)
            .await
            .expect("Failed to process query");

        // Verify results
        assert!(
            !result.search_results.is_empty(),
            "Should have search results"
        );
        assert!(
            !result.final_answer.is_empty(),
            "Should have a final answer"
        );
        assert!(result.confidence > 0.0, "Should have positive confidence");
        assert!(
            result.layers_used.len() >= 2,
            "Should use at least Echo and Speculator"
        );
    }

    #[tokio::test]
    async fn test_document_lifecycle() {
        let mut echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));

        // Index
        let doc = Document::new("Test document content").with_title("Test");
        let id = echo.index(doc).await.expect("Failed to index");

        // Retrieve
        let retrieved = echo
            .get(&id)
            .await
            .expect("Failed to get")
            .expect("Document not found");
        assert_eq!(retrieved.title, Some("Test".to_string()));

        // Search
        let results = echo
            .search("test document", 5, None)
            .await
            .expect("Failed to search");
        assert!(!results.is_empty());

        // Delete
        let deleted = echo.delete(&id).await.expect("Failed to delete");
        assert!(deleted);

        // Verify deleted
        let retrieved = echo.get(&id).await.expect("Failed to get");
        assert!(retrieved.is_none());
    }

    #[tokio::test]
    async fn test_query_filtering() {
        let mut echo = EchoLayer::new(MockEmbeddingProvider::new(32), InMemoryVectorStore::new(32));

        echo.index(Document::new("High relevance content"))
            .await
            .unwrap();
        echo.index(Document::new("Medium relevance")).await.unwrap();
        echo.index(Document::new("Low relevance")).await.unwrap();

        // Search with min_score filter
        let results = echo
            .search("high relevance", 10, Some(0.8))
            .await
            .expect("Failed to search");

        // Results should be filtered by score
        for result in &results {
            assert!(result.score >= 0.8);
        }
    }

    #[test]
    fn test_types_serialization() {
        let doc = Document::new("Test content")
            .with_title("Title")
            .with_metadata("key", "value");

        let json = serde_json::to_string(&doc).expect("Failed to serialize");
        let parsed: Document = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(parsed.content, doc.content);
        assert_eq!(parsed.title, doc.title);
    }

    #[test]
    fn test_claim_structure_smtlib() {
        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::Comparison {
                left: "a".to_string(),
                operator: ComparisonOp::GreaterThan,
                right: "b".to_string(),
            },
        );

        let extractor = AdvancedClaimExtractor::new();
        let smt = extractor
            .to_smtlib(&claim)
            .expect("Failed to generate SMT-LIB");

        assert!(smt.contains("assert"));
        assert!(smt.contains('>'));
    }
}
