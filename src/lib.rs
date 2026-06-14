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

#[cfg(feature = "advanced-retrieval")]
pub mod advanced_retrieval;
#[cfg(feature = "chunking")]
pub mod chunking;
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
pub mod observability;
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
#[cfg(feature = "semantic-cache")]
pub mod semantic_cache;
pub mod simd_similarity;
pub mod streaming;
pub mod types;

#[cfg(feature = "rag-eval")]
pub mod evaluation;

#[cfg(feature = "conversational")]
pub mod conversation;

#[cfg(feature = "flare")]
pub mod retrieval_loop;

#[cfg(feature = "collections")]
pub mod collections;

#[cfg(feature = "document-pipeline")]
pub mod document_pipeline;

#[cfg(feature = "prompt-templates")]
pub mod prompt_templates;

#[cfg(feature = "query-routing")]
pub mod query_router;

#[cfg(feature = "corrective-rag")]
pub mod corrective_rag;

#[cfg(feature = "attribution")]
pub mod attribution;

// Theme 1 — Agentic & Reasoning
#[cfg(feature = "self-rag")]
pub mod self_rag;

#[cfg(feature = "agentic")]
pub mod agentic;

#[cfg(feature = "query-decomposition")]
pub mod query_decomposition;

#[cfg(feature = "context-compression")]
pub mod context_compression;

// Theme 2 — Safety & Governance
#[cfg(feature = "guardrails")]
pub mod guardrails;

#[cfg(feature = "structured-extraction")]
pub mod structured_extraction;

#[cfg(feature = "output-validation")]
pub mod output_validation;

// Theme 3 — Knowledge Graph Intelligence
#[cfg(feature = "graph-community")]
pub mod graph_community;

#[cfg(feature = "graph-summarization")]
pub mod graph_summarization;

#[cfg(feature = "raptor")]
pub mod raptor;

// Theme 4 — Retrieval Depth
#[cfg(feature = "parent-document")]
pub mod parent_document;

#[cfg(feature = "temporal-retrieval")]
pub mod temporal;

// v0.11.0 — Theme 1: Multi-hop & Graph Reasoning
#[cfg(feature = "multi-hop")]
pub mod multi_hop;

#[cfg(feature = "fact-triples")]
pub mod fact_triple;

#[cfg(feature = "knowledge-graph-qa")]
pub mod knowledge_graph_qa;

// v0.11.0 — Theme 2: Iterative Generation
#[cfg(feature = "iterative-rag")]
pub mod iterative_rag;

#[cfg(feature = "chain-of-note")]
pub mod chain_of_note;

#[cfg(feature = "answer-aggregation")]
pub mod answer_aggregator;

// v0.11.0 — Theme 3: Trust & Verification
#[cfg(feature = "hallucination-detection")]
pub mod hallucination_detector;

#[cfg(feature = "consistency-checking")]
pub mod consistency_checker;

#[cfg(feature = "trust-scoring")]
pub mod trust_score;

// v0.11.0 — Theme 4: Routing & Composition
#[cfg(feature = "semantic-router")]
pub mod semantic_router;

#[cfg(feature = "query-planning")]
pub mod query_planning;

#[cfg(feature = "pipeline-composer")]
pub mod pipeline_composer;

// v0.12.0 — Theme 1: Reranking & Retrieval Precision
#[cfg(feature = "cross-encoder")]
pub mod cross_encoder;

#[cfg(feature = "contextual-retrieval")]
pub mod contextual_retrieval;

#[cfg(feature = "lost-in-middle")]
pub mod lost_in_middle;

// v0.12.0 — Theme 2: Advanced Reasoning
#[cfg(feature = "reflexion")]
pub mod reflexion;

#[cfg(feature = "tree-of-thought")]
pub mod tree_of_thought;

#[cfg(feature = "chain-of-verification")]
pub mod chain_of_verification;

// v0.12.0 — Theme 3: Memory & State
#[cfg(feature = "long-term-memory")]
pub mod long_term_memory;

#[cfg(feature = "memory-compression")]
pub mod memory_compression;

#[cfg(feature = "entity-memory")]
pub mod entity_memory;

// v0.12.0 — Theme 4: Evaluation & Optimization
#[cfg(feature = "retrieval-eval")]
pub mod retrieval_eval;

#[cfg(feature = "llm-judge")]
pub mod llm_judge;

#[cfg(feature = "prompt-optimization")]
pub mod prompt_optimization;

// v0.13.0 — Theme 1: Index-Time Representation
#[cfg(feature = "late-chunking")]
pub mod late_chunking;

#[cfg(feature = "proposition-retrieval")]
pub mod proposition;

#[cfg(feature = "doc2query")]
pub mod doc2query;

// v0.13.0 — Theme 2: Reranking & Result-Set Selection
#[cfg(feature = "listwise-rerank")]
pub mod listwise_rerank;

#[cfg(feature = "autocut")]
pub mod autocut;

#[cfg(feature = "semantic-dedup")]
pub mod semantic_dedup;

// v0.13.0 — Theme 3: Compositional Reasoning
#[cfg(feature = "self-ask")]
pub mod self_ask;

#[cfg(feature = "self-consistency")]
pub mod self_consistency;

#[cfg(feature = "adaptive-rag")]
pub mod adaptive_rag;

// v0.13.0 — Theme 4: Calibration, Uncertainty & Embedding Geometry
#[cfg(feature = "semantic-entropy")]
pub mod semantic_entropy;

#[cfg(feature = "matryoshka")]
pub mod matryoshka;

#[cfg(feature = "synthetic-eval")]
pub mod synthetic_eval;

// v0.14.0 — Theme 1: Retrieval & Indexing
#[cfg(feature = "sparse-retrieval")]
pub mod sparse_retrieval;

#[cfg(feature = "self-query")]
pub mod self_query;

#[cfg(feature = "summary-index")]
pub mod summary_index;

// v0.14.0 — Theme 2: Ranking & Fusion
#[cfg(feature = "rank-fusion")]
pub mod rank_fusion;

#[cfg(feature = "diversity-rank")]
pub mod diversity_rank;

#[cfg(feature = "source-credibility")]
pub mod source_credibility;

// v0.14.0 — Theme 3: Structured Reasoning
#[cfg(feature = "graph-of-thought")]
pub mod graph_of_thought;

#[cfg(feature = "skeleton-of-thought")]
pub mod skeleton_of_thought;

#[cfg(feature = "program-of-thought")]
pub mod program_of_thought;

// v0.14.0 — Theme 4: Verification & Robustness
#[cfg(feature = "fact-check")]
pub mod fact_check;

#[cfg(feature = "noise-filter")]
pub mod noise_filter;

#[cfg(feature = "answer-calibration")]
pub mod answer_calibration;

// v0.15.0 — Theme 1: Next-Gen Retrieval Architectures
#[cfg(feature = "hipporag")]
pub mod hippo_rag;

#[cfg(feature = "long-rag")]
pub mod long_rag;

#[cfg(feature = "dragin")]
pub mod dragin;

// v0.15.0 — Theme 2: Adaptive Generation Strategies
#[cfg(feature = "astute-rag")]
pub mod astute_rag;

#[cfg(feature = "self-route")]
pub mod self_route;

#[cfg(feature = "speculative-drafting")]
pub mod speculative_drafting;

// v0.15.0 — Theme 3: Knowledge & Context Management
#[cfg(feature = "memorag")]
pub mod memorag;

#[cfg(feature = "context-pruning")]
pub mod context_pruning;

#[cfg(feature = "knowledge-conflict")]
pub mod knowledge_conflict;

// v0.15.0 — Theme 4: Evaluation & Benchmarking
#[cfg(feature = "rgb-eval")]
pub mod rgb_eval;

#[cfg(feature = "nugget-eval")]
pub mod nugget_eval;

#[cfg(feature = "ab-eval")]
pub mod ab_eval;

#[cfg(feature = "rest-server")]
pub mod rest_server;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub mod wasm_worker;

#[cfg(feature = "python")]
pub mod python;

// Gate nodejs out of test builds: napi symbols come from the Node.js runtime
// (loaded as a .node cdylib), not from cargo test executables.
// Node.js integration tests run via `npm test` / `napi test`.
#[cfg(all(feature = "nodejs", not(test)))]
pub mod nodejs;

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
        Echo, EchoLayer, EmbeddingInput, EmbeddingProvider, InMemoryVectorStore, IndexedDocument,
        MetadataFilter, MockEmbeddingProvider, MultiModalEmbeddingProvider, SimilarityMetric,
        VectorStore,
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

    // Multi-modal (CLIP) exports
    #[cfg(all(feature = "multimodal", not(target_arch = "wasm32")))]
    pub use crate::layer1_echo::{CandleClipProvider, ClipPreset};
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

    // Chunking exports
    #[cfg(feature = "chunking")]
    pub use crate::chunking::{
        Chunk, ChunkConfig, ChunkStrategy, DocumentChunker, FixedSizeChunker, MarkdownChunker,
        RecursiveChunker, SentenceChunker,
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

    // Observability exports
    pub use crate::observability::{
        LayerSpanRecord, MemoryObserver, PipelineSpanContext, SpanObserver, SpanReport, SpanStatus,
        record_pipeline_event,
    };

    // WASM IndexedDB exports
    #[cfg(all(target_arch = "wasm32", feature = "wasm-indexeddb"))]
    pub use crate::layer1_echo::IndexedDbVectorStore;

    #[cfg(all(target_arch = "wasm32", feature = "wasm-prefix-indexeddb"))]
    pub use crate::prefix_cache::IndexedDbPrefixCache;

    // Quantization exports
    #[cfg(feature = "quantization")]
    pub use crate::quantization::{
        BinaryQuantizer, Int4Quantizer, Int8Quantizer, MockQuantizedVectorStore,
        QuantizationConfig, QuantizationType, QuantizedDocument, QuantizedTensor,
        QuantizedVectorStore, Quantizer, compute_quantization_error, compute_snr_db,
        hamming_distance, int4_dot_product, int8_dot_product,
    };

    // RAG evaluation framework exports
    #[cfg(feature = "rag-eval")]
    pub use crate::evaluation::{
        AggregateStats, AnswerRelevanceScorer, ContextPrecisionScorer, ContextRecallScorer,
        DatasetStats, EvalError, EvaluationDataset, EvaluationMetric, EvaluationResult,
        EvaluationSample, FaithfulnessScorer, OverallScorer, RagEvaluator,
    };

    // Conversational RAG exports
    #[cfg(feature = "conversational")]
    pub use crate::conversation::{
        ConversationAwareQuery, ConversationError, ConversationHistory, ConversationId,
        ConversationalPipeline, FollowUpDetector, FullHistoryBuffer, HistoryBuffer, HybridBuffer,
        InMemorySessionManager, QueryReformulator as ConversationQueryReformulator,
        ReformulationStrategy, Session, SessionConfig, SessionManager, SlidingWindowBuffer,
        SummaryBuffer, Turn, TurnRole,
    };

    // FLARE adaptive retrieval loop exports
    #[cfg(feature = "flare")]
    pub use crate::retrieval_loop::{
        ConfidenceEstimator, ContextDoc, ContextWindow, FlareConfig, FlareEngine, FlareError,
        FlareGenerator, FlareOutput, FlareRetriever, IterationRecord, MockFlareGenerator,
        MockFlareRetriever, QueryAugmentedRetriever, SentenceConfidence, TemplateGenerator,
        TokenConfidence,
    };

    // Knowledge base collections exports
    #[cfg(feature = "collections")]
    pub use crate::collections::{
        Collection, CollectionConfig, CollectionError, CollectionId, CollectionIndex,
        CollectionMetadata, CollectionSimilarityMetric, CollectionStats, CollectionStore,
        FederatedResult, InMemoryCollectionStore,
    };

    // Integrated document processing pipeline exports
    #[cfg(feature = "document-pipeline")]
    pub use crate::document_pipeline::{
        ChunkProvenance, ChunkStrategyKind, DocumentAwareResult, DocumentPipelineBuilder,
        DocumentPipelineError, IndexingConfig, IndexingPipeline, IndexingResult, PipelineStats,
        RetrievalPipeline,
    };

    // Prompt template registry exports
    #[cfg(feature = "prompt-templates")]
    pub use crate::prompt_templates::{
        PromptRegistry, PromptTemplate, PromptTemplateError, RenderContext, TemplateEngine,
        TemplateId, builtin_templates,
    };

    // Query routing exports
    #[cfg(feature = "query-routing")]
    pub use crate::query_router::{
        HeuristicIntentClassifier, IntentClassifier, IntentScores, MockIntentClassifier,
        QueryIntent, QueryRouter, QueryRouterError, RouterConfig, RoutingDecision, RoutingStrategy,
    };

    // Corrective RAG exports
    #[cfg(feature = "corrective-rag")]
    pub use crate::corrective_rag::{
        CorrectiveAction, CorrectiveRagEngine, CorrectiveRagError, CragConfig, CragOutput,
        GradedDocument, HeuristicRetrievalGrader, KnowledgeRefiner, KnowledgeStrip,
        MockRetrievalGrader, QueryRefiner, RetrievalGrade, RetrievalGrader,
    };

    // Attribution exports
    #[cfg(feature = "attribution")]
    pub use crate::attribution::{
        AlignmentScorer, AttributedAnswer, AttributionConfig, AttributionError, Attributor,
        Citation, CitationFormatter, CitationId, CitationStyle, CitedSpan, FaithfulnessChecker,
        LexicalAligner, SentenceAligner,
    };

    // Self-RAG exports
    #[cfg(feature = "self-rag")]
    pub use crate::self_rag::{
        HeuristicReflector, MockReflector, ReflectionToken, Reflector, SelfRagConfig,
        SelfRagEngine, SelfRagError, SelfRagOutput,
    };

    // Agentic RAG exports
    #[cfg(feature = "agentic")]
    pub use crate::agentic::{
        AgentAction, AgentStep, AgentTrace, AgenticConfig, AgenticError, CalculatorTool,
        LookupTool, MockTool, ReActAgent, Tool, ToolRegistry,
    };

    // Query decomposition exports
    #[cfg(feature = "query-decomposition")]
    pub use crate::query_decomposition::{
        DecomposedQuery, DecompositionConfig, DecompositionStrategy, QueryDecomposer,
        QueryDecompositionEngine, QueryDecompositionError, SubAnswer, SubQuestion,
    };

    // Context compression exports
    #[cfg(feature = "context-compression")]
    pub use crate::context_compression::{
        CompressedContext, CompressionConfig, CompressionError, ContextCompressor,
        ExtractiveCompressor, MockCompressor, RedundancyFilter,
    };

    // Guardrails exports
    #[cfg(feature = "guardrails")]
    pub use crate::guardrails::{
        ContentModerator, GuardrailConfig, GuardrailEngine, GuardrailError, GuardrailReport,
        InjectionDetector, PiiDetector, PiiKind, PiiMatch, Severity, TopicalRail, Violation,
    };

    // Structured extraction exports
    #[cfg(feature = "structured-extraction")]
    pub use crate::structured_extraction::{
        ExtractedRecord, ExtractedValue, ExtractionConfig, ExtractionSchema, FieldSchema,
        FieldType, SchemaExtractor, StructuredExtractionError,
    };

    // Output validation exports
    #[cfg(feature = "output-validation")]
    pub use crate::output_validation::{
        OutputValidationError, OutputValidator, RuleKind, RuleViolation, ValidationConfig,
        ValidationReport, ValidationRule,
    };

    // Graph community exports
    #[cfg(feature = "graph-community")]
    pub use crate::graph_community::{
        Community, CommunityDetector, CommunityGraph, CommunityId, GraphCommunityConfig,
        GraphCommunityError, LouvainDetector,
    };

    // Graph summarization exports
    #[cfg(feature = "graph-summarization")]
    pub use crate::graph_summarization::{
        CommunitySummarizer, CommunitySummary, GlobalSearchEngine, GraphSummarizationConfig,
        GraphSummarizationError, LocalSearchEngine, SummaryReport,
    };

    // RAPTOR exports
    #[cfg(feature = "raptor")]
    pub use crate::raptor::{
        ClusterStrategy, RaptorBuilder, RaptorConfig, RaptorError, RaptorNode, RaptorTree,
    };

    // Parent document retrieval exports
    #[cfg(feature = "parent-document")]
    pub use crate::parent_document::{
        ChunkHierarchy, ExpandedResult, ParentChildIndex, ParentDocumentConfig,
        ParentDocumentError, ParentDocumentRetriever,
    };

    // Temporal retrieval exports
    #[cfg(feature = "temporal-retrieval")]
    pub use crate::temporal::{
        DecayFunction, TemporalConfig, TemporalError, TemporalReranker, TemporalScore,
    };

    // Multi-hop traversal exports
    #[cfg(feature = "multi-hop")]
    pub use crate::multi_hop::{
        HopConfig, HopState, MultiHopError, MultiHopResult, MultiHopRetriever,
    };

    // Fact triple extraction exports
    #[cfg(feature = "fact-triples")]
    pub use crate::fact_triple::{Triple, TripleConfig, TripleError, TripleExtractor, TripleStore};

    // Knowledge-graph QA exports
    #[cfg(feature = "knowledge-graph-qa")]
    pub use crate::knowledge_graph_qa::{KgqaAnswer, KgqaConfig, KgqaEngine, KgqaError};

    // Iterative RAG exports
    #[cfg(feature = "iterative-rag")]
    pub use crate::iterative_rag::{
        IterationStep, IterativeConfig, IterativeOutput, IterativeRagEngine, IterativeRagError,
    };

    // Chain-of-Note exports
    #[cfg(feature = "chain-of-note")]
    pub use crate::chain_of_note::{
        ChainOfNoteEngine, ChainOfNoteError, DocumentNote, NoteChain, NoteConfig,
    };

    // Answer aggregation exports
    #[cfg(feature = "answer-aggregation")]
    pub use crate::answer_aggregator::{
        AggregatedAnswer, AggregationConfig, AggregationError, AggregationStrategy,
        AnswerAggregator, CandidateAnswer,
    };

    // Hallucination detection exports
    #[cfg(feature = "hallucination-detection")]
    pub use crate::hallucination_detector::{
        ClaimSupport, HallucinationConfig, HallucinationDetector, HallucinationError,
        HallucinationReport,
    };

    // Consistency checking exports
    #[cfg(feature = "consistency-checking")]
    pub use crate::consistency_checker::{
        ConflictType, ConsistencyChecker, ConsistencyConfig, ConsistencyError, ConsistencyReport,
        Inconsistency,
    };

    // Trust scoring exports
    #[cfg(feature = "trust-scoring")]
    pub use crate::trust_score::{
        TrustComponents, TrustConfig, TrustError, TrustScore, TrustScorer,
    };

    // Semantic router exports
    #[cfg(feature = "semantic-router")]
    pub use crate::semantic_router::{
        RouterError, RouterExample, RoutingDecision as SemanticRoutingDecision, RoutingTarget,
        SemanticRouter, SemanticRoutingConfig,
    };

    // Query planning exports
    #[cfg(feature = "query-planning")]
    pub use crate::query_planning::{
        PlanExecutor, PlanResult, PlanStep, PlanStepKind, QueryPlan, QueryPlanner,
        QueryPlanningError, SynthesisStrategy,
    };

    // Pipeline composer exports
    #[cfg(feature = "pipeline-composer")]
    pub use crate::pipeline_composer::{
        ComposedPipeline, ComposerError, PipelineStage, StageInput, StageOutput,
    };

    // Cross-encoder reranking exports
    #[cfg(feature = "cross-encoder")]
    pub use crate::cross_encoder::{
        CrossEncoderConfig, CrossEncoderError,
        CrossEncoderReranker as PairwiseCrossEncoderReranker, CrossEncoderScorer, FeatureWeights,
        InteractionFeatures, LexicalCrossEncoder, RerankedResult,
    };

    // Contextual retrieval exports
    #[cfg(feature = "contextual-retrieval")]
    pub use crate::contextual_retrieval::{
        ChunkContext, ContextualChunk, ContextualConfig, ContextualIndexBuilder,
        ContextualRetrievalError, Contextualizer, ExtractiveContextualizer,
    };

    // Lost-in-the-middle reordering exports
    #[cfg(feature = "lost-in-middle")]
    pub use crate::lost_in_middle::{
        LostInMiddleError, LostInMiddleReorderer, ReorderConfig, ReorderReport, ReorderStrategy,
    };

    // Reflexion exports
    #[cfg(feature = "reflexion")]
    pub use crate::reflexion::{
        Attempt, AttemptEvaluator, AttemptScore, EpisodicMemory, HeuristicEvaluator,
        HeuristicSelfReflector, Reflection, ReflexionConfig, ReflexionEngine, ReflexionError,
        ReflexionOutcome, SelfReflector,
    };

    // Tree-of-Thought exports
    #[cfg(feature = "tree-of-thought")]
    pub use crate::tree_of_thought::{
        HeuristicThoughtEvaluator, HeuristicThoughtGenerator, ThoughtEvaluator, ThoughtGenerator,
        ThoughtSearchStrategy, ThoughtState, ThoughtTree, ThoughtTreeNode, ToTConfig, ToTOutput,
        TreeOfThoughtEngine, TreeOfThoughtError,
    };

    // Chain-of-Verification exports
    #[cfg(feature = "chain-of-verification")]
    pub use crate::chain_of_verification::{
        ChainOfVerificationEngine, ChainOfVerificationError, ClaimVerdict, CoVeConfig, CoVeOutput,
        HeuristicQuestionPlanner, QuestionPlanner, VerificationAnswer, VerificationQuestion,
    };

    // Long-term memory exports
    #[cfg(feature = "long-term-memory")]
    pub use crate::long_term_memory::{
        HeuristicImportanceScorer, ImportanceScorer, LongTermMemoryConfig, LongTermMemoryError,
        LongTermMemoryStore, MemoryKind, MemoryQuery, MemoryRecord, MemoryRetriever,
        RetrievedMemory,
    };

    // Memory compression exports
    #[cfg(feature = "memory-compression")]
    pub use crate::memory_compression::{
        CompressedBlock, CompressionStats, ExtractiveTurnCompressor, HierarchicalMemory,
        MemoryCompressionConfig, MemoryCompressionError, MemoryTurn, TurnCompressor,
    };

    // Entity memory exports
    #[cfg(feature = "entity-memory")]
    pub use crate::entity_memory::{
        EntityCategory, EntityKnowledge, EntityMemoryConfig, EntityMemoryError, EntityMemoryStore,
        EntityMentionExtractor, EntityMentionSpan, HeuristicEntityMentionExtractor,
    };

    // Retrieval eval exports
    #[cfg(feature = "retrieval-eval")]
    pub use crate::retrieval_eval::{
        AggregateScores, Qrels, RelevanceJudgment, RetrievalEvalConfig, RetrievalEvalError,
        RetrievalEvaluator, RetrievalScores, average_precision, dcg_at_k, f1_at_k, hit_rate_at_k,
        mrr, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank,
    };

    // LLM-judge exports
    #[cfg(feature = "llm-judge")]
    pub use crate::llm_judge::{
        Criterion, CriterionScores, HeuristicJudge, JudgeContext, JudgeMode, JudgeModel, LlmJudge,
        LlmJudgeConfig, LlmJudgeError, PairwiseVerdict, PointwiseVerdict, Pref, Rubric,
    };

    // Prompt optimization exports
    #[cfg(feature = "prompt-optimization")]
    pub use crate::prompt_optimization::{
        DemoPool, DemoSelectionStrategy, DemoSelector, Demonstration, DevExample, OutputScorer,
        PromptOptimizationConfig, PromptOptimizationError, PromptOptimizer, PromptVariant,
        VariantScore,
    };

    // ── v0.13.0 ────────────────────────────────────────────────────────────────

    // Late chunking exports
    #[cfg(feature = "late-chunking")]
    pub use crate::late_chunking::{
        LateChunk, LateChunkConfig, LateChunkError, LateChunker, LatePooling,
    };

    // Proposition retrieval exports
    #[cfg(feature = "proposition-retrieval")]
    pub use crate::proposition::{
        HeuristicPropositionExtractor, Proposition, PropositionConfig, PropositionError,
        PropositionExtractor, PropositionHit, PropositionIndex,
    };

    // Doc2Query expansion exports (QueryGenerator aliased — load_testing already exports one)
    #[cfg(feature = "doc2query")]
    pub use crate::doc2query::{
        Doc2QueryConfig, Doc2QueryError, Doc2QueryExpander, ExpandedDocument,
        HeuristicQueryGenerator, QueryGenerator as Doc2QueryGenerator,
    };

    // Listwise reranking exports
    #[cfg(feature = "listwise-rerank")]
    pub use crate::listwise_rerank::{
        LexicalListwiseJudge, ListwiseError, ListwiseJudge, ListwiseReranker, ListwiseResult,
        WindowConfig,
    };

    // Autocut exports
    #[cfg(feature = "autocut")]
    pub use crate::autocut::{
        AutoCutConfig, AutoCutError, AutoCutReport, AutoCutStrategy, AutoCutter,
    };

    // Semantic dedup exports
    #[cfg(feature = "semantic-dedup")]
    pub use crate::semantic_dedup::{
        DedupMethod, KeepPolicy, SemanticDedupConfig, SemanticDedupError, SemanticDeduplicator,
    };

    // Self-Ask exports
    #[cfg(feature = "self-ask")]
    pub use crate::self_ask::{
        FollowUp, MockSelfAskModel, MockSubAnswerer, SelfAskConfig, SelfAskEngine, SelfAskError,
        SelfAskModel, SelfAskTrace, SubAnswerer,
    };

    // Self-Consistency exports
    #[cfg(feature = "self-consistency")]
    pub use crate::self_consistency::{
        AnswerCluster, MockReasoningSampler, ReasoningPath, ReasoningSampler,
        SelfConsistencyConfig, SelfConsistencyEngine, SelfConsistencyError, SelfConsistencyOutput,
        VoteWeighting,
    };

    // Adaptive-RAG exports
    #[cfg(feature = "adaptive-rag")]
    pub use crate::adaptive_rag::{
        AdaptiveRagConfig, AdaptiveRagError, AdaptiveRagRouter, ComplexityClassification,
        ComplexityClassifier, ComplexitySignal, QueryComplexity, RetrievalStrategy, RoutingPlan,
    };

    // Semantic entropy exports
    #[cfg(feature = "semantic-entropy")]
    pub use crate::semantic_entropy::{
        MeaningCluster, SemanticEntropyConfig, SemanticEntropyError, SemanticEntropyEstimator,
        SemanticEntropyResult,
    };

    // Matryoshka exports
    #[cfg(feature = "matryoshka")]
    pub use crate::matryoshka::{
        MatryoshkaConfig, MatryoshkaEmbedding, MatryoshkaEncoder, MatryoshkaError, MatryoshkaHit,
        MatryoshkaRetriever,
    };

    // Synthetic eval exports
    #[cfg(feature = "synthetic-eval")]
    pub use crate::synthetic_eval::{
        HeuristicTemplater, QuestionTemplater, QuestionType, SyntheticEvalConfig,
        SyntheticEvalError, SyntheticEvalGenerator, SyntheticQa,
    };

    // ── v0.14.0 ────────────────────────────────────────────────────────────────

    // Sparse retrieval exports (SparseVector aliased — hybrid_search already exports one)
    #[cfg(feature = "sparse-retrieval")]
    pub use crate::sparse_retrieval::{
        SparseConfig, SparseEncoder, SparseHit, SparseIndex, SparseRetrievalError,
        SparseVector as LearnedSparseVector,
    };

    // Self-query retriever exports (FieldType aliased — structured_extraction already exports one)
    #[cfg(feature = "self-query")]
    pub use crate::self_query::{
        FieldSpec, FieldType as FilterFieldType, FilterCondition, FilterOp, FilterSchema,
        ParsedFilter, SelfQueryConfig, SelfQueryError, SelfQueryParser, SelfQueryRetriever,
        StructuredQuery,
    };

    // Summary index exports
    #[cfg(feature = "summary-index")]
    pub use crate::summary_index::{
        DocumentSummary, ExtractiveSummarizer, Summarizer, SummaryConfig, SummaryHit, SummaryIndex,
        SummaryIndexError,
    };

    // Rank fusion exports
    #[cfg(feature = "rank-fusion")]
    pub use crate::rank_fusion::{
        FusionMethod, RankFusion, RankFusionConfig, RankFusionError, ScoreNormalization,
    };

    // Diversity rank exports
    #[cfg(feature = "diversity-rank")]
    pub use crate::diversity_rank::{
        DiversityConfig, DiversityRankError, DiversityRanker, DiversitySelection, SimilarityKind,
    };

    // Source credibility exports
    #[cfg(feature = "source-credibility")]
    pub use crate::source_credibility::{
        AuthoritySignals, CredibilityConfig, CredibilityScore, CredibilityScorer,
        SourceCredibilityError, SourceGraph,
    };

    // Graph-of-Thoughts exports (ThoughtGenerator aliased — tree_of_thought already exports one)
    #[cfg(feature = "graph-of-thought")]
    pub use crate::graph_of_thought::{
        GotConfig, GotError, GotOperation, GotOutput, GraphOfThoughtEngine, MockThoughtAggregator,
        MockThoughtGenerator, MockThoughtScorer, Thought, ThoughtAggregator,
        ThoughtGenerator as GotThoughtGenerator, ThoughtGraph, ThoughtScorer,
    };

    // Skeleton-of-Thought exports
    #[cfg(feature = "skeleton-of-thought")]
    pub use crate::skeleton_of_thought::{
        MockPointExpander, MockSkeletonGenerator, PointExpander, SkeletonConfig, SkeletonGenerator,
        SkeletonOfThoughtEngine, SkeletonOutput, SkeletonPoint, SotError,
    };

    // Program-of-Thoughts exports
    #[cfg(feature = "program-of-thought")]
    pub use crate::program_of_thought::{
        Expr, Interpreter, MockProgramGenerator, Op, PotError, PotOutput, Program,
        ProgramGenerator, ProgramOfThoughtEngine, Statement, parse_program,
    };

    // Fact-check exports
    #[cfg(feature = "fact-check")]
    pub use crate::fact_check::{
        Evidence, FactCheckConfig, FactCheckError, FactCheckResult, FactChecker, Verdict,
    };

    // Noise filter exports
    #[cfg(feature = "noise-filter")]
    pub use crate::noise_filter::{
        NoiseConfig, NoiseFilter, NoiseFilterError, NoiseReport, PassageAssessment,
    };

    // Answer calibration exports
    #[cfg(feature = "answer-calibration")]
    pub use crate::answer_calibration::{
        AnswerCalibrationError, AnswerCalibrator, AnswerCalibratorConfig, CalibrationMetrics,
        ConfidenceSignals, ReliabilityBin, brier_score, compute_metrics,
        expected_calibration_error, maximum_calibration_error, reliability_bins,
    };

    // ── v0.15.0 ────────────────────────────────────────────────────────────────

    // HippoRAG exports
    #[cfg(feature = "hipporag")]
    pub use crate::hippo_rag::{HippoConfig, HippoHit, HippoRagError, HippoRagIndex};

    // LongRAG exports
    #[cfg(feature = "long-rag")]
    pub use crate::long_rag::{
        GroupingStrategy, LongHit, LongRagConfig, LongRagError, LongRagRetriever, LongUnit,
        LongUnitGrouper,
    };

    // DRAGIN exports
    #[cfg(feature = "dragin")]
    pub use crate::dragin::{
        DraginConfig, DraginEngine, DraginError, DraginTrace, MockRetriever,
        MockUncertaintyGenerator, RetrievalTrigger, Retriever, TokenInfo, UncertaintyGenerator,
    };

    // Astute RAG exports
    #[cfg(feature = "astute-rag")]
    pub use crate::astute_rag::{
        AstuteConfig, AstuteConsolidator, AstuteError, ConsolidatedKnowledge, InternalKnowledge,
        KnowledgeConflict, KnowledgeSource, KnowledgeStatement, MockInternalKnowledge,
    };

    // Self-Route exports
    #[cfg(feature = "self-route")]
    pub use crate::self_route::{
        RouteAssessment, RouteDecision, SelfRouteConfig, SelfRouteError, SelfRouter,
    };

    // Speculative Drafting exports
    #[cfg(feature = "speculative-drafting")]
    pub use crate::speculative_drafting::{
        DraftCandidate, DraftVerifier, Drafter, MockDraftVerifier, MockDrafter, SpecDraftConfig,
        SpecDraftError, SpeculativeDrafter, SpeculativeOutput,
    };

    // MemoRAG exports
    #[cfg(feature = "memorag")]
    pub use crate::memorag::{MemoHit, MemoRagConfig, MemoRagEngine, MemoRagError, MemoryGist};

    // Context Pruning exports
    #[cfg(feature = "context-pruning")]
    pub use crate::context_pruning::{
        ContextPruningError, PruneConfig, PrunedContext, TokenPruner,
    };

    // Knowledge Conflict exports
    #[cfg(feature = "knowledge-conflict")]
    pub use crate::knowledge_conflict::{
        ConflictDetector, ConflictKind, ConflictPolicy, ConflictResolution, ConflictResolver,
        KnowledgeConflictConfig, KnowledgeConflictError, PassageConflict,
    };

    // RGB Eval exports
    #[cfg(feature = "rgb-eval")]
    pub use crate::rgb_eval::{
        RgbAbility, RgbConfig, RgbError, RgbEvaluator, RgbScores, RgbTestCase,
    };

    // Nugget Eval exports
    #[cfg(feature = "nugget-eval")]
    pub use crate::nugget_eval::{
        HeuristicNuggetExtractor, Nugget, NuggetConfig, NuggetEvalError, NuggetExtractor,
        NuggetImportance, NuggetScore, NuggetScorer,
    };

    // A/B Eval exports
    #[cfg(feature = "ab-eval")]
    pub use crate::ab_eval::{AbConfig, AbError, AbEvaluator, AbResult, AbWinner};
}

pub use error::{OxiRagError, Result};

#[cfg(feature = "otel")]
pub use observability::otel::OtelSpanObserver;

// Node.js napi-rs re-exports (not available in test builds — see nodejs module gate above)
#[cfg(all(feature = "nodejs", not(test)))]
pub use crate::nodejs::{
    NapiDocument, NapiPipeline, NapiPipelineBuilder, NapiQuery, NapiSearchResult,
};

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
            .expect("test operation should succeed");
        echo.index(Document::new("Medium relevance"))
            .await
            .expect("test operation should succeed");
        echo.index(Document::new("Low relevance"))
            .await
            .expect("test operation should succeed");

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
