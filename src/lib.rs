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

// v0.16.0 — Theme 1: Graph & Generative Retrieval
#[cfg(feature = "drift-search")]
pub mod drift_search;

#[cfg(feature = "entity-linking")]
pub mod entity_linking;

#[cfg(feature = "generative-retrieval")]
pub mod generative_retrieval;

// v0.16.0 — Theme 2: Retrieval Composition
#[cfg(feature = "auto-merging")]
pub mod auto_merging;

#[cfg(feature = "ensemble-retriever")]
pub mod ensemble_retriever;

#[cfg(feature = "gen-read")]
pub mod gen_read;

// v0.16.0 — Theme 3: Time, Language & Personalization
#[cfg(feature = "fresh-retrieval")]
pub mod fresh_retrieval;

#[cfg(feature = "cross-lingual")]
pub mod cross_lingual;

#[cfg(feature = "personalized-rag")]
pub mod personalized_rag;

// v0.16.0 — Theme 4: Grounding & Fine-grained Verification
#[cfg(feature = "quote-grounding")]
pub mod quote_grounding;

#[cfg(feature = "claim-decomposition")]
pub mod claim_decomposition;

#[cfg(feature = "fusion-in-decoder")]
pub mod fusion_in_decoder;

// v0.17.0 — Theme 1: ANN Indexing & Late Interaction
#[cfg(feature = "product-quantization")]
pub mod product_quantization;

#[cfg(feature = "ivf-index")]
pub mod ivf_index;

#[cfg(feature = "plaid")]
pub mod plaid_retrieval;

// v0.17.0 — Theme 2: Generation Refinement
#[cfg(feature = "self-refine")]
pub mod self_refine;

#[cfg(feature = "chain-of-density")]
pub mod chain_of_density;

#[cfg(feature = "analogical")]
pub mod analogical_prompting;

// v0.17.0 — Theme 3: Robustness & Privacy
#[cfg(feature = "poisoning-defense")]
pub mod poisoning_defense;

#[cfg(feature = "anonymization")]
pub mod anonymization;

#[cfg(feature = "abstention")]
pub mod abstention;

// v0.17.0 — Theme 4: Advanced Evaluation
#[cfg(feature = "ragchecker")]
pub mod ragchecker;

#[cfg(feature = "retrieval-diversity")]
pub mod retrieval_diversity;

#[cfg(feature = "ares-eval")]
pub mod ares_eval;

// v0.18.0 — Theme 1: ANN & Vector Indexing
#[cfg(feature = "hnsw")]
pub mod hnsw_index;

#[cfg(feature = "lsh")]
pub mod lsh_index;

#[cfg(feature = "scalar-quantization")]
pub mod scalar_quantization;

// v0.18.0 — Theme 2: Advanced Prompting & Reasoning
#[cfg(feature = "step-back")]
pub mod step_back;

#[cfg(feature = "least-to-most")]
pub mod least_to_most;

#[cfg(feature = "self-discover")]
pub mod self_discover;

// v0.18.0 — Theme 3: Reranking & Retrieval Quality
#[cfg(feature = "pairwise-rerank")]
pub mod pairwise_rerank;

#[cfg(feature = "multi-query")]
pub mod multi_query;

#[cfg(feature = "citation-verification")]
pub mod citation_verification;

// v0.18.0 — Theme 4: Evaluation & Safety
#[cfg(feature = "faithfulness-eval")]
pub mod faithfulness_eval;

#[cfg(feature = "prompt-injection-defense")]
pub mod prompt_injection_defense;

#[cfg(feature = "query-difficulty")]
pub mod query_difficulty;

// v0.19.0 — Theme 1: Scalable Indexing & Quantization
#[cfg(feature = "disk-ann")]
pub mod disk_ann;

#[cfg(feature = "spann")]
pub mod spann;

#[cfg(feature = "rabitq")]
pub mod rabitq;

// v0.19.0 — Theme 2: Query Transformation & Clarification
#[cfg(feature = "rq-rag")]
pub mod rq_rag;

#[cfg(feature = "tree-of-clarifications")]
pub mod tree_of_clarifications;

#[cfg(feature = "query2doc")]
pub mod query2doc;

// v0.19.0 — Theme 3: Retrieval-Augmented Reasoning & Compression
#[cfg(feature = "retrieval-augmented-thoughts")]
pub mod retrieval_augmented_thoughts;

#[cfg(feature = "recomp")]
pub mod recomp;

#[cfg(feature = "filco")]
pub mod filco;

// v0.19.0 — Theme 4: Hallucination Detection & Trustworthy Eval
#[cfg(feature = "selfcheckgpt")]
pub mod selfcheckgpt;

#[cfg(feature = "erag")]
pub mod erag;

#[cfg(feature = "eigenscore")]
pub mod eigenscore;

// v0.20.0 — Theme 1: Learned Vector Compression
#[cfg(feature = "anisotropic-vq")]
pub mod anisotropic_vq;

#[cfg(feature = "itq-hashing")]
pub mod itq_hashing;

#[cfg(feature = "residual-vq")]
pub mod residual_vq;

// v0.20.0 — Theme 2: Decoupled & Persistent Reasoning Architectures
#[cfg(feature = "rewoo")]
pub mod rewoo;

#[cfg(feature = "searchain")]
pub mod searchain;

#[cfg(feature = "buffer-of-thoughts")]
pub mod buffer_of_thoughts;

// v0.20.0 — Theme 3: Prompt & Context Efficiency
#[cfg(feature = "llmlingua")]
pub mod llmlingua;

#[cfg(feature = "memory-paging")]
pub mod memory_paging;

#[cfg(feature = "uprise-retrieval")]
pub mod uprise_retrieval;

// v0.20.0 — Theme 4: Statistical Calibration & Privacy Robustness
#[cfg(feature = "conformal-rag")]
pub mod conformal_rag;

#[cfg(feature = "chainpoll")]
pub mod chainpoll;

#[cfg(feature = "membership-inference")]
pub mod membership_inference;

// v0.21.0 — Theme 1: Graph-Structured Knowledge RAG
#[cfg(feature = "g-retriever")]
pub mod g_retriever;

#[cfg(feature = "think-on-graph")]
pub mod think_on_graph;

#[cfg(feature = "lightrag")]
pub mod lightrag;

// v0.21.0 — Theme 2: Fine-Grained & Late-Interaction Retrieval
#[cfg(feature = "coil-retrieval")]
pub mod coil_retrieval;

#[cfg(feature = "muvera")]
pub mod muvera;

#[cfg(feature = "instruction-embed")]
pub mod instruction_embed;

// v0.21.0 — Theme 3: Tabular & Structured Knowledge
#[cfg(feature = "table-rag")]
pub mod table_rag;

#[cfg(feature = "structrag")]
pub mod structrag;

#[cfg(feature = "chain-of-table")]
pub mod chain_of_table;

// v0.21.0 — Theme 4: Reranking, Adaptive Control & Evaluation
#[cfg(feature = "setwise-rerank")]
pub mod setwise_rerank;

#[cfg(feature = "skr")]
pub mod skr;

#[cfg(feature = "crud-rag")]
pub mod crud_rag;

// v0.22.0 — Theme 1: Classical IR Reimagined
#[cfg(feature = "learning-to-rank")]
pub mod learning_to_rank;

#[cfg(feature = "rp-tree-index")]
pub mod rp_tree_index;

#[cfg(feature = "bm25f-retrieval")]
pub mod bm25f_retrieval;

// v0.22.0 — Theme 2: Multi-Agent & Principled Reasoning
#[cfg(feature = "multi-agent-debate")]
pub mod multi_agent_debate;

#[cfg(feature = "constitutional-critique")]
pub mod constitutional_critique;

#[cfg(feature = "tool-retrieval")]
pub mod tool_retrieval;

// v0.22.0 — Theme 3: Domain-Specialized Retrieval
#[cfg(feature = "code-retrieval")]
pub mod code_retrieval;

#[cfg(feature = "extractive-qa")]
pub mod extractive_qa;

#[cfg(feature = "hard-negative-mining")]
pub mod hard_negative_mining;

// v0.22.0 — Theme 4: Retrieval Governance
#[cfg(feature = "active-learning-retrieval")]
pub mod active_learning_retrieval;

#[cfg(feature = "belief-revision")]
pub mod belief_revision;

#[cfg(feature = "shard-selection")]
pub mod shard_selection;

// v0.23.0 — Theme 1: Efficient Search & Index Maintenance
#[cfg(feature = "filtered-vector-search")]
pub mod filtered_vector_search;

#[cfg(feature = "dynamic-pruning")]
pub mod dynamic_pruning;

#[cfg(feature = "index-maintenance")]
pub mod index_maintenance;

// v0.23.0 — Theme 2: Probabilistic Information Retrieval
#[cfg(feature = "language-model-retrieval")]
pub mod language_model_retrieval;

#[cfg(feature = "click-model")]
pub mod click_model;

#[cfg(feature = "bandit-ranker")]
pub mod bandit_ranker;

// v0.23.0 — Theme 3: Collaborative Generation
#[cfg(feature = "replug")]
pub mod replug;

#[cfg(feature = "mixture-of-agents")]
pub mod mixture_of_agents;

#[cfg(feature = "chain-of-agents")]
pub mod chain_of_agents;

// v0.23.0 — Theme 4: Privacy & Provenance
#[cfg(feature = "kv-cache-compression")]
pub mod kv_cache_compression;

#[cfg(feature = "watermarking")]
pub mod watermarking;

#[cfg(feature = "knowledge-unlearning")]
pub mod knowledge_unlearning;

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

    // ── v0.16.0 ────────────────────────────────────────────────────────────────

    // DRIFT search exports
    #[cfg(feature = "drift-search")]
    pub use crate::drift_search::{
        CommunityReport, DriftAnswer, DriftConfig, DriftError, DriftSearchEngine, DriftStep,
        DriftStepKind,
    };

    // Entity linking exports
    #[cfg(feature = "entity-linking")]
    pub use crate::entity_linking::{
        CanonicalEntity, EntityCatalog, EntityLinkConfig, EntityLinkError, EntityLinker,
        EntityMention, LinkedEntity,
    };

    // Generative retrieval exports
    #[cfg(feature = "generative-retrieval")]
    pub use crate::generative_retrieval::{
        GenHit, GenRetrievalConfig, GenRetrievalError, GenerativeRetriever, SemanticDocId,
    };

    // Auto-merging exports (ChunkHierarchy aliased — parent_document already exports one)
    #[cfg(feature = "auto-merging")]
    pub use crate::auto_merging::{
        AutoMergeConfig, AutoMergeError, AutoMergingRetriever,
        ChunkHierarchy as AutoMergeHierarchy, ChunkNode, MergedHit,
    };

    // Ensemble retriever exports
    #[cfg(feature = "ensemble-retriever")]
    pub use crate::ensemble_retriever::{
        EnsembleConfig, EnsembleError, EnsembleFusion, EnsembleRetriever, LexicalSubRetriever,
        SubRetriever,
    };

    // GenRead exports
    #[cfg(feature = "gen-read")]
    pub use crate::gen_read::{
        ContextGenerator, GenReadConfig, GenReadEngine, GenReadError, GenReadOutput, GeneratedDoc,
        MockContextGenerator,
    };

    // Fresh retrieval exports
    #[cfg(feature = "fresh-retrieval")]
    pub use crate::fresh_retrieval::{
        FreshConfig, FreshError, FreshnessAnalyzer, FreshnessAssessment, TimeSensitivity,
    };

    // Cross-lingual exports
    #[cfg(feature = "cross-lingual")]
    pub use crate::cross_lingual::{
        BilingualLexicon, CrossLingualConfig, CrossLingualError, CrossLingualHit,
        CrossLingualRetriever,
    };

    // Personalized RAG exports
    #[cfg(feature = "personalized-rag")]
    pub use crate::personalized_rag::{
        PersonalizedConfig, PersonalizedError, PersonalizedReranker, UserProfile,
    };

    // Quote grounding exports
    #[cfg(feature = "quote-grounding")]
    pub use crate::quote_grounding::{GroundedQuote, QuoteConfig, QuoteError, QuoteGrounder};

    // Claim decomposition exports (ClaimExtractor aliased — layer3_judge already exports one)
    #[cfg(feature = "claim-decomposition")]
    pub use crate::claim_decomposition::{
        AtomicClaim, AtomicClaimExtractor, ClaimDecompConfig, ClaimDecompError,
        ClaimExtractor as AtomicClaimExtractorTrait, HeuristicAtomicExtractor,
    };

    // Fusion-in-Decoder exports
    #[cfg(feature = "fusion-in-decoder")]
    pub use crate::fusion_in_decoder::{
        FidConfig, FidError, FusedAnswer, FusionInDecoder, PassageEvidence,
    };

    // ── v0.17.0 ────────────────────────────────────────────────────────────────

    // Product quantization exports
    #[cfg(feature = "product-quantization")]
    pub use crate::product_quantization::{
        PqCode, PqConfig, PqError, PqHit, PqIndex, ProductQuantizer,
    };

    // IVF index exports
    #[cfg(feature = "ivf-index")]
    pub use crate::ivf_index::{IvfConfig, IvfError, IvfHit, IvfIndex};

    // PLAID late-interaction exports
    #[cfg(feature = "plaid")]
    pub use crate::plaid_retrieval::{PlaidConfig, PlaidError, PlaidHit, PlaidRetriever};

    // Self-Refine exports
    #[cfg(feature = "self-refine")]
    pub use crate::self_refine::{
        Feedback, MockRefiner, RefineStep, Refiner, SelfRefineConfig, SelfRefineEngine,
        SelfRefineError, SelfRefineOutput,
    };

    // Chain-of-Density exports
    #[cfg(feature = "chain-of-density")]
    pub use crate::chain_of_density::{
        ChainOfDensityEngine, ChainOfDensityOutput, CodConfig, CodError, DensityStep,
    };

    // Analogical prompting exports
    #[cfg(feature = "analogical")]
    pub use crate::analogical_prompting::{
        AnalogicalConfig, AnalogicalEngine, AnalogicalError, AnalogicalModel, AnalogicalOutput,
        Exemplar, MockAnalogicalModel,
    };

    // Poisoning-defense exports
    #[cfg(feature = "poisoning-defense")]
    pub use crate::poisoning_defense::{
        PoisonAssessment, PoisonConfig, PoisonError, PoisoningDetector,
    };

    // Anonymization exports (PiiKind aliased — guardrails already exports one)
    #[cfg(feature = "anonymization")]
    pub use crate::anonymization::{
        AnonConfig, AnonError, AnonMapping, AnonymizedText, Anonymizer, PiiKind as AnonPiiKind,
    };

    // Abstention exports
    #[cfg(feature = "abstention")]
    pub use crate::abstention::{
        AbstentionAssessment, AbstentionConfig, AbstentionDecision, AbstentionError,
        AbstentionPolicy, RiskCoverage,
    };

    // RAGChecker exports
    #[cfg(feature = "ragchecker")]
    pub use crate::ragchecker::{
        RagCheckResult, RagChecker, RagCheckerConfig, RagCheckerError, RagCheckerMetrics,
    };

    // Retrieval diversity exports
    #[cfg(feature = "retrieval-diversity")]
    pub use crate::retrieval_diversity::{
        DiversityMetrics, DiversityScorer, RetrievalDiversityConfig, RetrievalDiversityError,
    };

    // ARES eval exports
    #[cfg(feature = "ares-eval")]
    pub use crate::ares_eval::{AresConfig, AresError, AresEvaluator, PpiInterval};

    // ── v0.18.0 ────────────────────────────────────────────────────────────────
    #[cfg(feature = "citation-verification")]
    pub use crate::citation_verification::{
        CitationCheck, CitationConfig, CitationError, CitationReport, CitationVerifier,
        VerifiedCitation,
    };
    #[cfg(feature = "faithfulness-eval")]
    pub use crate::faithfulness_eval::{
        ClaimEntailment, FaithfulnessConfig, FaithfulnessError, FaithfulnessEvaluator,
        FaithfulnessScore,
    };
    #[cfg(feature = "hnsw")]
    pub use crate::hnsw_index::{HnswConfig, HnswError, HnswHit, HnswIndex};
    #[cfg(feature = "least-to-most")]
    pub use crate::least_to_most::{
        LtmConfig, LtmEngine, LtmError, LtmResult, LtmSolver, MockLtmSolver, SubProblem,
    };
    #[cfg(feature = "lsh")]
    pub use crate::lsh_index::{LshConfig, LshError, LshHit, LshIndex, MinHashIndex};
    #[cfg(feature = "multi-query")]
    pub use crate::multi_query::{
        GeneratedQuery, MockQueryVariantGenerator, MultiQueryConfig, MultiQueryError,
        MultiQueryGenerator, MultiQueryHit, MultiQueryResult, QueryVariantGenerator,
    };
    #[cfg(feature = "pairwise-rerank")]
    pub use crate::pairwise_rerank::{
        MockPairwiseComparer, PairwiseComparer, PairwiseConfig, PairwiseError, PairwiseHit,
        PairwiseReranker, PairwiseScoredPair,
    };
    #[cfg(feature = "scalar-quantization")]
    pub use crate::scalar_quantization::{
        BinaryVector, QuantizedVector, ScalarQuantizer, SqConfig, SqError,
    };
    #[cfg(feature = "self-discover")]
    pub use crate::self_discover::{
        MockSelfDiscoverModel, ReasoningModule, ReasoningStructure, SelfDiscoverConfig,
        SelfDiscoverEngine, SelfDiscoverError, SelfDiscoverModel, SelfDiscoverResult,
    };
    #[cfg(feature = "step-back")]
    pub use crate::step_back::{
        MockStepBackModel, StepBackConfig, StepBackEngine, StepBackError, StepBackModel,
        StepBackResult,
    };
    // InjectionDetector aliased — guardrails already exports one
    #[cfg(feature = "prompt-injection-defense")]
    pub use crate::prompt_injection_defense::{
        DefenseReport, DefenseStrategy, DocumentScanResult, InjectionCategory, InjectionPattern,
        MatchedPattern, PromptInjectionConfig, PromptInjectionDetector, PromptInjectionError,
    };
    #[cfg(feature = "query-difficulty")]
    pub use crate::query_difficulty::{
        DifficultyBand, DifficultyConfig, DifficultyError, DifficultyPredictor, DifficultyScore,
        DifficultySignals,
    };

    // ── v0.19.0 ────────────────────────────────────────────────────────────────
    #[cfg(feature = "disk-ann")]
    pub use crate::disk_ann::{
        DiskAnnConfig, DiskAnnError, DiskAnnHit, DiskAnnIndex, DiskAnnMetric, VamanaGraph,
    };
    #[cfg(feature = "eigenscore")]
    pub use crate::eigenscore::{
        EigenScoreConfig, EigenScoreDetector, EigenScoreError, EigenScoreResult,
        symmetric_eigenvalues,
    };
    #[cfg(feature = "erag")]
    pub use crate::erag::{
        AggregationMethod, DownstreamTask, ERagBatchReport, ERagCase, ERagConfig, ERagError,
        ERagEvaluator, ERagReport, MockDownstreamTask, PerDocScore, RougeLiteUtility,
        UtilityMetric, kendall_tau, spearman_rho,
    };
    #[cfg(feature = "filco")]
    pub use crate::filco::{
        FilcoConfig, FilcoError, FilcoFilter, FilcoReport, FilterMeasure, ScoredSentence,
    };
    #[cfg(feature = "query2doc")]
    pub use crate::query2doc::{
        MockPseudoDocGenerator, PseudoDocGenerator, PseudoDocument, Query2DocConfig,
        Query2DocError, Query2DocExpander, Query2DocVariant,
    };
    #[cfg(feature = "rabitq")]
    pub use crate::rabitq::{
        RaBitQCode, RaBitQConfig, RaBitQError, RaBitQHit, RaBitQIndex, RaBitQMetric, RaBitQuantizer,
    };
    #[cfg(feature = "recomp")]
    pub use crate::recomp::{
        AbstractiveSummaryCompressor, CompressorStrategy, ExtractiveSummaryCompressor,
        RecompConfig, RecompDecision, RecompError, RecompOutcome, RecompPipeline,
    };
    #[cfg(feature = "retrieval-augmented-thoughts")]
    pub use crate::retrieval_augmented_thoughts::{
        MockRatGenerator, MockRatRetriever, RatConfig, RatEngine, RatError, RatGenerator,
        RatRetriever, RatThought, RatTrace,
    };
    // QueryRefiner/MockRefiner aliased — corrective_rag/self_refine already export those
    #[cfg(feature = "rq-rag")]
    pub use crate::rq_rag::{
        AMBIGUITY_SENSE_TABLE, DEFAULT_COLLOQUIAL_MARKERS, MockRefiner as RqRagMockRefiner,
        QueryRefinementEngine, QueryRefiner as RqRagQueryRefiner, RefinementAction, RefinementPlan,
        RqRagConfig, RqRagError,
    };
    #[cfg(feature = "selfcheckgpt")]
    pub use crate::selfcheckgpt::{
        SelfCheckConfig, SelfCheckError, SelfCheckScore, SelfCheckScorer, SelfCheckVariant,
        SentenceCheck,
    };
    #[cfg(feature = "spann")]
    pub use crate::spann::{
        Posting, SpannConfig, SpannError, SpannHit, SpannIndex, SpannMetric, SpannStats,
    };
    #[cfg(feature = "tree-of-clarifications")]
    pub use crate::tree_of_clarifications::{
        ClarificationAnswerer, ClarificationNode, ClarificationRetriever, ClarificationTree,
        Disambiguator, MockClarificationAnswerer, MockClarificationRetriever, MockDisambiguator,
        ToCConfig, ToCEngine, ToCError, aggregate,
    };

    // ── v0.20.0 ────────────────────────────────────────────────────────────────
    #[cfg(feature = "anisotropic-vq")]
    pub use crate::anisotropic_vq::{
        AnisotropicCode, AnisotropicQuantizer, AnisotropicVqConfig, AnisotropicVqError,
        AnisotropicVqHit, AnisotropicVqIndex, AnisotropicVqMetric, DEFAULT_ANISOTROPIC_VQ_SEED,
        MAX_CODEBOOK_SIZE, anisotropic_loss, decompose_residual,
    };
    #[cfg(feature = "buffer-of-thoughts")]
    pub use crate::buffer_of_thoughts::{
        BotConfig, BotEngine, BotError, BotGenerator, BotSolveResult, DEFAULT_TEMPLATE_TEXT,
        DistillAction, EvictionPolicy, MockBotGenerator, ProblemSignature, SolveOutcome,
        ThoughtBuffer, ThoughtTemplate,
    };
    #[cfg(feature = "chainpoll")]
    pub use crate::chainpoll::{
        ChainOfThoughtJudge, ChainPollConfig, ChainPollError, ChainPollFormulationVerdict,
        ChainPollResult, ChainPollScorer, MockChainPollJudge, PollFormulation, PollFraming,
    };
    #[cfg(feature = "conformal-rag")]
    pub use crate::conformal_rag::{
        ConformalCalibrator, ConformalConfig, ConformalError, ConformalMember,
        LexicalOverlapScorer, MondrianConformalCalibrator, NonconformityScorer, PredictionKind,
        PredictionSet,
    };
    #[cfg(feature = "itq-hashing")]
    pub use crate::itq_hashing::{
        DEFAULT_ITQ_SEED, ItqCode, ItqConfig, ItqError, ItqHasher, ItqHit, ItqIndex,
    };
    #[cfg(feature = "llmlingua")]
    pub use crate::llmlingua::{
        BudgetController, CompressionResult, CompressionTarget, LlmLinguaConfig, LlmLinguaError,
        PerplexityCompressor, PerplexityModel, SegmentStats,
    };
    #[cfg(feature = "membership-inference")]
    pub use crate::membership_inference::{
        Canary, CanaryAuditor, CanaryKind, CanaryPair, CanaryPairSignal, LeakageReport,
        MembershipDefense, MembershipInferenceConfig, MembershipInferenceError, MockRagProbe,
        ProbeResult, RagProbe, RiskLevel,
    };
    #[cfg(feature = "memory-paging")]
    pub use crate::memory_paging::{
        ArchivalStore, ContextPager, MainContext, MemoryPage, MemoryPagingConfig,
        MemoryPagingError, PagingAction, PagingOutcome, PagingPolicy,
    };
    #[cfg(feature = "residual-vq")]
    pub use crate::residual_vq::{
        ResidualCode, ResidualQuantizer, ResidualVqConfig, ResidualVqError, ResidualVqHit,
        ResidualVqIndex,
    };
    #[cfg(feature = "rewoo")]
    pub use crate::rewoo::{
        MockRewooGenerator, MockRewooPlanSource, MockRewooRetriever, PlaceholderVar, RewooAction,
        RewooConfig, RewooError, RewooEvidence, RewooGenerator, RewooOutcome, RewooPipeline,
        RewooPlan, RewooPlanSource, RewooPlanner, RewooRetriever, RewooSolver, RewooStep,
        RewooWorker, evaluate_expression, substitute_placeholders,
    };
    // Retriever aliased — dragin already exports that bare trait name; Generator aliased
    // alongside it for naming symmetry (no current collision, but bare "Generator" is
    // generic enough to risk future ones).
    #[cfg(feature = "searchain")]
    pub use crate::searchain::{
        AnswerAssemblyStrategy, ChainGenerator, ChainNode, Generator as SearchainGenerator,
        MockChainGenerator, MockSearchainGenerator, MockSearchainRetriever, NodeVerdict,
        Retriever as SearchainRetriever, SearChainConfig, SearChainEngine, SearChainError,
        SearChainResult, SearChainStatus,
    };
    #[cfg(feature = "uprise-retrieval")]
    pub use crate::uprise_retrieval::{
        PromptExemplar, UpriseConfig, UpriseError, UpriseHit, UpriseIndex, UpriseRetriever,
    };
    // ── v0.21.0 ────────────────────────────────────────────────────────────────
    // No aliasing needed: all 142 public names are module-prefixed and collision-free.
    // chain_of_table's generic free fns (apply/apply_with_config/format_number) are
    // intentionally NOT re-exported here (reachable via crate::chain_of_table::*).
    #[cfg(feature = "chain-of-table")]
    pub use crate::chain_of_table::{
        ChainOfTableConfig, ChainOfTableEngine, ChainOfTableError, ChainOfTableResult, CotAddRule,
        CotAggregate, CotAnswer, CotAnswerValue, CotCell, CotColumn, CotColumnType, CotComparator,
        CotEnabledOperations, CotOperationKind, CotOperationTrace, CotPredicate, CotRow,
        CotTableOperation, CotTableState, cot_cell_cmp,
    };
    #[cfg(feature = "coil-retrieval")]
    pub use crate::coil_retrieval::{
        CoilConfig, CoilDocument, CoilError, CoilInvertedIndex, CoilPosting, CoilResult,
        CoilRetriever, CoilScoreMode, CoilTokenVector,
    };
    #[cfg(feature = "crud-rag")]
    pub use crate::crud_rag::{
        CrudCase, CrudMetric, CrudOperation, CrudRagConfig, CrudRagError, CrudRagHarness,
        CrudRagReport, CrudRagResult, CrudScore,
    };
    #[cfg(feature = "g-retriever")]
    pub use crate::g_retriever::{
        GRetrieverConfig, GRetrieverEngine, GRetrieverEntity, GRetrieverError, GRetrieverRelation,
        GRetrieverResult, GRetrieverRootMode, GRetrieverSubgraph, PcstEdge, PcstForest, PcstNode,
        PcstSolver,
    };
    #[cfg(feature = "instruction-embed")]
    pub use crate::instruction_embed::{
        InstructionEmbedConfig, InstructionEmbedError, InstructionEmbedResult, InstructionEmbedder,
        InstructionEmbedding, InstructionIndex, InstructionRegistry, TaskInstruction,
    };
    #[cfg(feature = "lightrag")]
    pub use crate::lightrag::{
        LightRagChunk, LightRagConfig, LightRagDualKeywords, LightRagEngine, LightRagEntity,
        LightRagEntityKind, LightRagError, LightRagIndex, LightRagIndexStats, LightRagMode,
        LightRagRelation, LightRagResult,
    };
    #[cfg(feature = "muvera")]
    pub use crate::muvera::{
        DEFAULT_MUVERA_SEED, FixedDimEncoding, MAX_K_SIM, MuveraConfig, MuveraDocument,
        MuveraEncoder, MuveraError, MuveraIndex, MuveraResult, MuveraSimilarity,
    };
    #[cfg(feature = "setwise-rerank")]
    pub use crate::setwise_rerank::{
        SetwiseCandidate, SetwiseComparison, SetwiseConfig, SetwiseError, SetwiseRankEntry,
        SetwiseRanking, SetwiseReranker, SetwiseResult, SetwiseSortStrategy,
    };
    #[cfg(feature = "skr")]
    pub use crate::skr::{
        SelfKnowledgePool, SkrConfig, SkrDecision, SkrError, SkrExemplar, SkrGate, SkrNeighbor,
        SkrResult, SkrRetrievalChoice,
    };
    #[cfg(feature = "structrag")]
    pub use crate::structrag::{
        StructRagAlgorithm, StructRagCatalogue, StructRagCatalogueItem, StructRagConfig,
        StructRagEngine, StructRagError, StructRagGraph, StructRagGraphEdge, StructRagGraphNode,
        StructRagKnowledgeStructure, StructRagPassage, StructRagReasoner, StructRagRestructurer,
        StructRagResult, StructRagRouter, StructRagRoutingDecision, StructRagStep,
        StructRagStructureKind, StructRagTable, StructRagTableRow, StructRagTree,
        StructRagTreeNode,
    };
    #[cfg(feature = "table-rag")]
    pub use crate::table_rag::{
        CellProbe, TableColumnSpec, TableColumnType, TableRagConfig, TableRagEngine, TableRagError,
        TableRagIndex, TableRagResult, TableRagSubTable, TableRagTable, TableSchema,
    };
    #[cfg(feature = "think-on-graph")]
    pub use crate::think_on_graph::{
        TogBeamPath, TogConfig, TogDecision, TogEngine, TogEntity, TogError, TogExploration,
        TogKnowledgeGraph, TogRelation, TogResult, TogTriple,
    };
    // ── v0.22.0 ────────────────────────────────────────────────────────────────
    // 1 alias: extractive_qa::QuestionType collides with synthetic_eval::QuestionType
    // (already in the v0.19.0 block above) — aliased to disambiguate; unaliased access
    // remains available via crate::extractive_qa::QuestionType. All other 125 names are
    // module-prefixed and collision-free.
    #[cfg(feature = "active-learning-retrieval")]
    pub use crate::active_learning_retrieval::{
        ActiveLearningConfig, ActiveLearningError, ActiveLearningResult, ActiveLearningSelector,
        PoolItem, SelectionBatch, UncertaintyMeasure, UncertaintySample,
    };
    #[cfg(feature = "belief-revision")]
    pub use crate::belief_revision::{
        BeliefHypothesis, BeliefRevisionConfig, BeliefRevisionEngine, BeliefRevisionError,
        BeliefRevisionResult, BeliefState, EvidenceUpdate, LikelihoodRatio,
    };
    #[cfg(feature = "bm25f-retrieval")]
    pub use crate::bm25f_retrieval::{
        Bm25fConfig, Bm25fDocument, Bm25fError, Bm25fField, Bm25fFieldWeight, Bm25fHit, Bm25fIndex,
        Bm25fResult,
    };
    #[cfg(feature = "code-retrieval")]
    pub use crate::code_retrieval::{
        CodeRetrievalConfig, CodeRetrievalEngine, CodeRetrievalError, CodeRetrievalHit,
        CodeRetrievalIndex, CodeRetrievalLanguageHint, CodeRetrievalResult, CodeRetrievalSymbol,
        CodeRetrievalUnit, CodeRetrievalUnitKind, parse_source,
    };
    #[cfg(feature = "constitutional-critique")]
    pub use crate::constitutional_critique::{
        ConstitutionalConfig, ConstitutionalCritique, ConstitutionalEngine, ConstitutionalError,
        ConstitutionalMatch, ConstitutionalPrinciple, ConstitutionalResult, ConstitutionalReviser,
        ConstitutionalRevision, ConstitutionalTraceEntry, ConstitutionalTrigger,
        MockConstitutionalReviser,
    };
    #[cfg(feature = "extractive-qa")]
    pub use crate::extractive_qa::{
        AnswerSpan, ExtractiveQaConfig, ExtractiveQaEngine, ExtractiveQaError, ExtractiveQaResult,
        QuestionType as ExtractiveQuestionType, SpanCandidate, SpanScore,
    };
    #[cfg(feature = "hard-negative-mining")]
    pub use crate::hard_negative_mining::{
        EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeError,
        HardNegativeMiner, HardNegativePositivePair, HardNegativeQueryOverlap,
        HardNegativeQueryResult, HardNegativeResult, HardNegativeSample, HardNegativeStaleness,
        MiningRound,
    };
    #[cfg(feature = "learning-to-rank")]
    pub use crate::learning_to_rank::{
        LTR_DEFAULT_FEATURE_DIM, LTR_FEATURE_BM25, LTR_FEATURE_EMBEDDING_SIM, LTR_FEATURE_LENGTH,
        LTR_FEATURE_POPULARITY, LTR_FEATURE_RECENCY, LtrConfig, LtrDocument, LtrEngine, LtrError,
        LtrFeatureExtractor, LtrFeatureVector, LtrModel, LtrResult, LtrTrainingPair,
        LtrTrainingSet,
    };
    #[cfg(feature = "multi-agent-debate")]
    pub use crate::multi_agent_debate::{
        DebateArgument, DebateConfig, DebateEngine, DebateError, DebateJudge, DebateJudgeWeights,
        DebateParticipant, DebatePersona, DebatePositionScore, DebateResult, DebateRound,
        DebateVerdict, MockDebateJudge, MockDebatePersona,
    };
    #[cfg(feature = "rp-tree-index")]
    pub use crate::rp_tree_index::{
        RpTreeConfig, RpTreeError, RpTreeForest, RpTreeHit, RpTreeHyperplane, RpTreeIndex,
        RpTreeMetric, RpTreeNode, RpTreeResult,
    };
    #[cfg(feature = "shard-selection")]
    pub use crate::shard_selection::{
        ShardDescriptor, ShardSelectionConfig, ShardSelectionEngine, ShardSelectionError,
        ShardSelectionResult, ShardSelectionScore, ShardSelector, ShardTermStats,
    };
    #[cfg(feature = "tool-retrieval")]
    pub use crate::tool_retrieval::{
        ArgumentGrounder, ArgumentGroundingStatus, GroundedArgument, ToolMatch, ToolParameter,
        ToolParameterType, ToolRetrievalConfig, ToolRetrievalEngine, ToolRetrievalError,
        ToolRetrievalIndex, ToolRetrievalResult, ToolSpecEntry,
    };
    // ── v0.23.0 ────────────────────────────────────────────────────────────────
    // No aliasing needed: all 210 public names are module-prefixed and collision-free.
    #[cfg(feature = "bandit-ranker")]
    pub use crate::bandit_ranker::{
        BanditArm, BanditArmSet, BanditArmStats, BanditConfig, BanditContext, BanditError,
        BanditLinearModel, BanditLoggedEvent, BanditOffPolicyEstimate, BanditOffPolicyEvaluator,
        BanditRankedArm, BanditRanker, BanditRanking, BanditRegretTracker, BanditResult,
        BanditStats, EpsilonGreedyRanker, LinUcbRanker, LinalgError, SplitMix64Rng,
        ThompsonSamplingRanker,
    };
    #[cfg(feature = "chain-of-agents")]
    pub use crate::chain_of_agents::{
        CoaCommunicationUnit, CoaConfig, CoaEngine, CoaError, CoaEvidence, CoaLexicalManager,
        CoaLexicalWorker, CoaManager, CoaTrace, CoaWorker, CoaWorkerStep, MIN_CHUNK_SIZE,
    };
    #[cfg(feature = "click-model")]
    pub use crate::click_model::{
        CascadeClickModel, ClickImpression, ClickLog, ClickLogSimulator, ClickModel,
        ClickModelConfig, ClickModelError, ClickModelFit, ClickModelResult, ClickRankingPolicy,
        ClickRewardModel, ClickSession, ClickSimulationModel, ClickSplitMix64,
        CounterfactualEstimate, CounterfactualEstimator, CounterfactualStrategy, DbnClickModel,
        DebiasedRanker, DebiasedRankerConfig, DebiasedRankerFit, DoublyRobustEstimator,
        IpsEstimator, PbmIdentifiability, PositionBasedModel, PropensityEstimates, SnipsEstimator,
        TableClickPolicy, cascade_examined_depth, naive_click_through_rates, policy_positions,
        position_gain,
    };
    #[cfg(feature = "dynamic-pruning")]
    pub use crate::dynamic_pruning::{
        DynamicPruningError, DynamicPruningIndex, DynamicPruningResult, PruningBlock,
        PruningConfig, PruningHit, PruningPosting, PruningPostingList, PruningSearchResult,
        PruningStats, PruningStrategy, bm25_idf, bm25_term_score,
    };
    #[cfg(feature = "filtered-vector-search")]
    pub use crate::filtered_vector_search::{
        AttrValue, FilterBound, FilterPredicate, FilterStrategy, FilteredAttributeStats,
        FilteredDistanceMetric, FilteredHit, FilteredMetadata, FilteredNumericHistogram,
        FilteredSearchConfig, FilteredSearchError, FilteredSearchStats, FilteredVectorIndex,
        FilteredVectorRecord, SelectivityEstimate, SelectivityEstimator, StrategySelector,
    };
    #[cfg(feature = "index-maintenance")]
    pub use crate::index_maintenance::{
        ConsolidationReport, MaintainableIndex, MaintenanceConfig, MaintenanceError,
        MaintenanceHit, MaintenanceMetric, MaintenanceStats,
    };
    #[cfg(feature = "knowledge-unlearning")]
    pub use crate::knowledge_unlearning::{
        UnlearnableStore, UnlearningArtifactKind, UnlearningArtifactLink,
        UnlearningArtifactRegistry, UnlearningAuditEvidence, UnlearningAuditRound,
        UnlearningAuditor, UnlearningCertificate, UnlearningConfig, UnlearningEngine,
        UnlearningError, UnlearningLeakageVerdict, UnlearningMemoryStore,
        UnlearningNearDuplicateDetector, UnlearningRequest, UnlearningResult, UnlearningScope,
        UnlearningScopeItem, UnlearningScopeReason, UnlearningScopeResolver, UnlearningStatus,
        UnlearningTarget,
    };
    #[cfg(feature = "kv-cache-compression")]
    pub use crate::kv_cache_compression::{
        KvAttentionOutput, KvAttentionStats, KvCacheCompressor, KvCacheTensor, KvCompressionConfig,
        KvCompressionError, KvCompressionReport, KvEvictionPolicy, KvResult, KvScoreNormalization,
        KvSnapPooling, plan_h2o, plan_recency_lru, plan_snap_kv, plan_streaming_llm, pool_scores,
        scaled_dot_product_attention,
    };
    #[cfg(feature = "language-model-retrieval")]
    pub use crate::language_model_retrieval::{
        DfrModel, LM_MIN_PROBABILITY, LmCollectionModel, LmDocumentStats, LmHit, LmRelevanceModel,
        LmRetrievalConfig, LmRetrievalError, LmRetrievalIndex, LmRetrievalResult, LmScoringModel,
        LmSmoothing, LmSmoothingComponents, PL2_MIN_TFN, Rm3Config, Rm3ExpandedQuery, Rm3Term,
        dph_term_score, lm_log_sum_exp, lm_query_posteriors, lm_tokenize, pl2_term_score,
    };
    #[cfg(feature = "mixture-of-agents")]
    pub use crate::mixture_of_agents::{
        MoaAggregator, MoaConfig, MoaContextMode, MoaEngine, MoaError, MoaLayer, MoaLayerStats,
        MoaProposer, MoaResponse, MoaSynthesisAggregator, MoaTrace, MockMoaProposer,
    };
    #[cfg(feature = "replug")]
    pub use crate::replug::{
        ReplugConfig, ReplugContextRule, ReplugDecoding, ReplugDocument, ReplugDocumentGradient,
        ReplugEngine, ReplugEnsembleOutput, ReplugError, ReplugLanguageModel, ReplugLsrSignal,
        ReplugResult, ReplugRng, ReplugStaticLanguageModel, ReplugStats, ReplugStep,
        replug_arg_max, replug_entropy_from_log_probs, replug_kl_divergence,
        replug_log_linear_pool, replug_log_softmax, replug_log_sum_exp, replug_mixture_log_probs,
        replug_promote_logits, replug_softmax, replug_temperature_log_softmax,
        replug_temperature_softmax,
    };
    #[cfg(feature = "watermarking")]
    pub use crate::watermarking::{
        WatermarkConfig, WatermarkDetection, WatermarkDetector, WatermarkError, WatermarkGenerator,
        WatermarkHasher, WatermarkMode, WatermarkResult, WatermarkTokenId,
    };
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
