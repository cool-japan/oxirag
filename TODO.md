# OxiRAG TODO

## Vision Alignment

This project implements four innovative concepts:

1. **Speculative RAG** - Use cache as "drafts" not "answers", verify with SLM
2. **Context-Aware Prefix Caching** - Efficiently manage KV Cache for "premise knowledge"
3. **On-the-fly Distillation** - Automatically generate specialized lightweight models for frequent queries
4. **Hidden States Manipulation** - Direct manipulation of transformer hidden states for verification

---

## High Priority

### Speculative RAG (Core Vision #1) - 99% Complete

**Goal**: Use cache as "drafts" instead of "final answers", verify with SLM

- [x] Vector search for similar documents (Layer 1: Echo)
- [x] Draft generation pipeline (`generate_draft`)
- [x] Accept/Revise/Reject decision flow (Layer 2: Speculator)
- [x] High-speed parallel processing (`tokio::join!`, `process_batch`)
- [x] **Streaming Verification**: Stream verification results with chunks
- [x] **Confidence Calibration**: Platt scaling, isotonic, temperature scaling, histogram binning
- [x] **SLM Interface**: SmallLanguageModel trait with MockSlm implementation
- [x] **Verification Pipeline**: Multi-stage verification with built-in stages
- [x] **Speculative Decoding**: Full speculative decoding with hidden states
- [ ] **Real SLM Integration**: Actual verification with Candle + Phi-2/Phi-3

### Context-Aware Prefix Caching (Core Vision #2) - 95% Complete

**Goal**: Manage "understanding state of loaded documents (KV Cache)" not just "answers"

- [x] Embedding Cache with LRU (Layer 1)
- [x] **KV-Cache Types**: Core types for KV cache management
- [x] **Context Fingerprinting**: Hash-based context identification and reuse
- [x] **Prefix Cache Store**: In-memory prefix cache with eviction
- [x] **Paged Cache**: PagedAttention-inspired paging with CachePage, PageTable
- [x] **Hierarchical Cache**: L1/L2/L3 tier cache with promotion/demotion
- [x] **Cache Invalidation**: TTL, MaxAge, dependency-based invalidation policies
- [x] **Persistent Backend**: File-based persistent cache with HybridPersistentCache
- [ ] **External Backend**: RocksDB/Redis integration

### Hidden States (Core Vision #4) - 90% Complete

**Goal**: Direct manipulation of transformer hidden states for speculative verification

- [x] **Hidden State Types**: HiddenStateTensor, LayerHiddenState, ModelHiddenStates
- [x] **KV Cache**: ModelKVCache for efficient incremental decoding
- [x] **State Provider**: HiddenStateProvider trait with MockHiddenStateProvider
- [x] **State Caching**: HiddenStateCache with LRU eviction and prefix matching
- [x] **Reuse Strategies**: Prefix, Semantic, Hybrid, Adaptive strategies
- [x] **State Similarity**: Cosine, L2, layer-wise comparison utilities
- [x] **Speculative Decoder**: SpeculativeDecoder with draft/target model architecture
- [x] **Hidden State Speculator**: HiddenStateSpeculator for verification via state comparison
- [x] **Divergence Detection**: Identify factual inconsistencies via hidden state divergence
- [ ] **Candle Integration**: Connect with actual Candle model inference

### On-the-fly Distillation (Core Vision #3) - 85% Complete

**Goal**: Automatically generate specialized lightweight models (SLM) for frequent queries

- [x] **Query Frequency Tracking**: Track query pattern frequency
- [x] **Q&A Pair Collection**: Automatic collection of Q&A pairs for distillation
- [x] **Distillation Candidate Detection**: Identify patterns ready for distillation
- [x] **LoRA Training Types**: LoraConfig, TrainingJob, LoraTrainer trait, MockLoraTrainer
- [x] **Model Registry**: ModelMetadata, ModelMetrics, find by pattern, metrics tracking
- [x] **Distillation Trigger**: TriggerCondition (frequency, count, confidence, time, combined)
- [x] **Hot-swap Models**: ModelSelector with strategies (PatternMatch, LowestLatency, etc.)
- [x] **Feature Distillation**: FitNet-style intermediate layer distillation, attention transfer
- [x] **Loss Functions**: KL divergence, MSE, cosine similarity, combined losses
- [x] **Training Metrics**: Accuracy, loss tracking, early stopping criteria
- [x] **Teacher-Student**: Teacher-student architecture with inference support
- [x] **Progressive Distillation**: Multi-stage progressive knowledge transfer
- [ ] **Real LoRA Training**: Integration with Candle for actual fine-tuning

---

## Layer-Specific Tasks

### Layer 1: Echo (Semantic Search)

- [x] In-memory vector store
- [x] Mock embedding provider
- [x] Similarity metrics (cosine, euclidean, dot product)
- [x] Embedding cache with LRU eviction
- [x] Metadata filtering for search
- [x] Document update/upsert operations
- [x] **HNSW Index**: Approximate nearest neighbor search (`ann.rs`)
- [x] **Multi-vector Documents**: ColBERT-style late interaction (`multi_vector.rs`)
- [x] **SIMD Similarity**: Hardware-accelerated similarity computation (`simd_similarity.rs`)
- [ ] Implement persistent vector store (SQLite/RocksDB backend)
- [ ] Integrate real embedding models (all-MiniLM-L6-v2, BGE, etc.)

### Layer 2: Speculator (Draft Verification)

- [x] Rule-based speculator
- [x] Mock SLM speculator
- [x] Streaming verification support
- [x] Confidence calibration system
- [x] SmallLanguageModel trait and MockSlm
- [x] Multi-stage verification pipeline
- [x] **Quantization Types**: INT8, INT4, Binary quantization support (`quantization.rs`)
- [ ] Complete Candle SLM integration with Phi-2/Phi-3

### Layer 3: Judge (SMT Verification)

- [x] Pattern-based claim extractor
- [x] Advanced claim extractor
- [x] Mock SMT verifier
- [x] Temporal, causal, and modal claim structure types
- [x] Claim deduplication and normalization
- [x] Explanation generation for verification results
- [x] **Dependency Parsing**: Improved claim extraction with SVO extraction (`dependency_parser.rs`)
- [x] **Incremental Consistency**: Incremental consistency checking with conflict detection (`incremental.rs`)
- [ ] Real OxiZ SMT solver integration

### Layer 4: Graph (GraphRAG)

- [x] Core types (GraphEntity, GraphRelationship, GraphPath)
- [x] EntityExtractor and RelationshipExtractor traits
- [x] Mock entity/relationship extractors
- [x] Pattern-based entity/relationship extractors
- [x] In-memory graph store
- [x] BFS traversal algorithms
- [x] Shortest path and find entities within N hops
- [x] GraphLayer with builder pattern
- [x] HybridSearchResult for combining vector + graph search

### Pipeline

- [x] Unified pipeline with fast-path optimization
- [x] Pipeline builder pattern
- [x] Retry logic with exponential backoff
- [x] Parallel layer execution
- [x] Metrics and tracing
- [x] Batch query processing
- [x] **Pipeline Debugging**: Visualization and tracing tools (`pipeline_debug.rs`)
- [x] **Circuit Breaker**: Resilience pattern for external service failures (`circuit_breaker.rs`)

---

## Medium Priority

### Performance

- [x] **SIMD Optimization**: Hardware-accelerated similarity computation (`simd_similarity.rs`)
- [x] **Connection Pooling**: Generic connection pool for external services (`connection_pool.rs`)
- [x] **Memory Monitoring**: Memory limits and usage tracking (`memory.rs`)
- [ ] Profile and optimize hot paths

### WASM

- [ ] Test and optimize WASM bundle size
- [ ] Add Web Worker support for background processing
- [ ] Implement IndexedDB backend for persistent storage
- [ ] Add streaming response support
- [ ] Create React/Vue/Svelte component wrappers

### API Improvements

- [x] **Query Builder**: Fluent interface for building queries (`query_builder.rs`)
- [x] **Index Management**: Rebuild, optimize, vacuum, snapshot API (`index_management.rs`)
- [x] Async streaming API for long-running queries (`streaming.rs`)

### Testing

- [x] **Load Testing**: Utilities for concurrent operations testing (`load_testing.rs`)
- [ ] Add property-based testing with proptest
- [ ] Add integration tests with real models
- [ ] Implement fuzzing for claim extraction

---

## Low Priority

### Documentation

- [ ] Add API documentation with examples
- [ ] Create architecture decision records (ADRs)
- [ ] Write layer-specific tutorials
- [ ] Add troubleshooting guide
- [ ] Create performance tuning guide

### Ecosystem

- [ ] Create Python bindings with PyO3
- [ ] Add Node.js bindings
- [ ] Implement REST API server example
- [ ] Create Docker image
- [ ] Add OpenTelemetry integration

### Advanced Features

- [x] **Hybrid Search**: Dense + sparse (BM25) retrieval with fusion (`hybrid_search.rs`)
- [x] **Reranking**: Cross-encoder style reranking pipeline (`reranker.rs`)
- [x] **Query Expansion**: Synonyms, stemming, PRF expansion (`query_expansion.rs`)
- [x] **Relevance Feedback**: User feedback loop with Rocchio algorithm (`relevance_feedback.rs`)
- [ ] Support multi-modal embeddings (text + image)

---

## Completed (v0.1.0)

### Core Infrastructure
- [x] Core type system (Document, Query, SearchResult, Draft, etc.)
- [x] Error handling with thiserror
- [x] Configuration management (JSON serialization)
- [x] WASM bindings structure
- [x] Comprehensive test suite (1,500 tests)
- [x] Clippy compliance (no warnings)
- [x] Rustdoc compliance (no warnings)
- [x] Criterion benchmark suite
- [x] Release preparation (LICENSE, CHANGELOG, publish script)

### Speculative RAG (99%)
- [x] Draft-based pipeline architecture
- [x] Accept/Revise/Reject decision flow
- [x] Parallel layer execution (tokio::join!)
- [x] Batch query processing (process_batch)
- [x] Streaming verification with VerificationChunk
- [x] Confidence calibration (Platt, Isotonic, Temperature, Histogram)
- [x] SmallLanguageModel trait with MockSlm
- [x] Multi-stage VerificationPipeline (Keyword, Semantic, Factual stages)

### Prefix Caching (95%)
- [x] KVCacheEntry and ContextFingerprint types
- [x] PrefixCacheStore trait with async operations
- [x] InMemoryPrefixCache with LRU eviction
- [x] Context fingerprinting with rolling hash
- [x] TTL-based cache expiration
- [x] Prefix matching for partial cache hits
- [x] CachePage and PageTable for paged cache management
- [x] HierarchicalCache with L1/L2/L3 tiers
- [x] InvalidationManager with dependency tracking
- [x] InvalidationPolicy (TTL, MaxAge, MaxStale, DependencyBased)

### Distillation (85%)
- [x] QueryPattern and QAPair types
- [x] DistillationTracker trait
- [x] QueryFrequencyTracker with pattern normalization
- [x] QAPairCollector with deduplication
- [x] CandidateDetector with priority ranking
- [x] Training example export functionality
- [x] LoraConfig, TrainingJob, LoraTrainer trait
- [x] MockLoraTrainer for testing
- [x] ModelRegistry with metrics tracking
- [x] DistillationTrigger with configurable conditions
- [x] ModelSelector with hot-swap strategies
- [x] Feature-based distillation (FitNet, attention transfer)
- [x] Distillation loss functions (KL, MSE, cosine, combined)
- [x] Training metrics and evaluation utilities
- [x] Teacher-student architecture with inference
- [x] Progressive knowledge distillation

### Layer Implementations
- [x] Layer 1: Echo (Vector Search + HNSW + Multi-vector + SIMD)
- [x] Layer 2: Speculator (Rule-based + Mock + Streaming + Calibration + Quantization)
- [x] Layer 3: Judge (Claim extraction + Mock SMT + Dependency parsing + Incremental)
- [x] Layer 4: GraphRAG (Entity/Relationship extraction)

### Hidden States (Core Feature)
- [x] HiddenStateTensor and ModelHiddenStates types
- [x] KVCache and ModelKVCache for attention caching
- [x] HiddenStateProvider trait with MockHiddenStateProvider
- [x] HiddenStateCache with LRU eviction
- [x] StateReuseStrategy trait with Prefix, Semantic, Hybrid strategies
- [x] StateSimilarity utilities (cosine, L2, layer-wise)
- [x] SpeculativeDecoder with hidden state manipulation
- [x] HiddenStateSpeculator for verification
- [x] Divergence detection via hidden state comparison

### New Modules (v0.1.0)
- [x] `query_builder.rs` - Fluent query construction API
- [x] `index_management.rs` - Index rebuild, optimize, vacuum, snapshot
- [x] `simd_similarity.rs` - SIMD-accelerated vector operations
- [x] `quantization.rs` - INT8/INT4/Binary quantization types
- [x] `hybrid_search.rs` - Dense + sparse retrieval fusion
- [x] `reranker.rs` - Multi-stage reranking pipeline
- [x] `query_expansion.rs` - Query expansion and reformulation
- [x] `relevance_feedback.rs` - User feedback with Rocchio algorithm
- [x] `connection_pool.rs` - Generic connection pooling
- [x] `memory.rs` - Memory monitoring and limits
- [x] `load_testing.rs` - Load testing utilities
- [x] `pipeline_debug.rs` - Pipeline visualization and tracing
- [x] `layer1_echo/multi_vector.rs` - ColBERT-style multi-vector support
- [x] `layer3_judge/dependency_parser.rs` - Dependency parsing for claims
- [x] `layer3_judge/incremental.rs` - Incremental consistency checking
- [x] `distillation/feature.rs` - Feature-based distillation (FitNet, attention transfer)
- [x] `distillation/losses.rs` - Distillation loss functions (KL, MSE, cosine)
- [x] `distillation/metrics.rs` - Training metrics and evaluation
- [x] `distillation/teacher_student.rs` - Teacher-student architecture
- [x] `distillation/progressive.rs` - Progressive knowledge distillation

---

## Notes

### Dependencies to Watch
- `candle`: Monitor for WASM support improvements
- `oxiz`: Check for new SMT theories and features
- `tokenizers`: Watch for WASM compatibility

### Breaking Changes Planned
- v0.2: Refactor `SearchResult` to include more metadata
- v0.2: Change `PipelineOutput` to use cow strings for efficiency
- v0.3: Unified async trait without `async-trait` crate (when stable)

### Vision Milestone Targets

| Milestone | Speculative RAG | Prefix Caching | Distillation | Hidden States |
|-----------|-----------------|----------------|--------------|---------------|
| v0.1.0    | **99%**         | **95%**        | **85%**      | **90%**       |
| v0.2.0    | 100%            | 98%            | 92%          | 95%           |
| v0.3.0    | 100%            | 99%            | 96%          | 98%           |
| v1.0.0    | 100%            | 100%           | 100%         | 100%          |

### Codebase Statistics (v0.1.0)
- **Source Files**: 91 Rust files
- **Total Lines**: 60,001 (Rust code), 48,463 pure code
- **Tests**: 1,500
- **Doc Tests**: 23 (21 ignored for async)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
