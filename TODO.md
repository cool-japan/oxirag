# OxiRAG TODO

## Vision Alignment

This project implements four innovative concepts:

1. **Speculative RAG** - Use cache as "drafts" not "answers", verify with SLM
2. **Context-Aware Prefix Caching** - Efficiently manage KV Cache for "premise knowledge"
3. **On-the-fly Distillation** - Automatically generate specialized lightweight models for frequent queries
4. **Hidden States Manipulation** - Direct manipulation of transformer hidden states for verification

---

## High Priority

### Speculative RAG (Core Vision #1) - 100% Complete ✅

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
- [x] **Real SLM Integration**: Complete Candle-based SLM with Phi-2/Phi-3 support (`candle_slm.rs`)

### Context-Aware Prefix Caching (Core Vision #2) - 98% Complete ✅

**Goal**: Manage "understanding state of loaded documents (KV Cache)" not just "answers"

- [x] Embedding Cache with LRU (Layer 1)
- [x] **KV-Cache Types**: Core types for KV cache management
- [x] **Context Fingerprinting**: Hash-based context identification and reuse
- [x] **Prefix Cache Store**: In-memory prefix cache with eviction
- [x] **Paged Cache**: PagedAttention-inspired paging with CachePage, PageTable
- [x] **Hierarchical Cache**: L1/L2/L3 tier cache with promotion/demotion
- [x] **Cache Invalidation**: TTL, MaxAge, dependency-based invalidation policies
- [x] **Persistent Backend**: File-based persistent cache with HybridPersistentCache
- [x] **External Backend**: redb (pure-Rust ACID embedded DB) — `RedbPrefixCache` with TTL eviction, prefix-match, LRU capacity enforcement (`prefix_cache/redb_backend.rs`, feature `prefix-cache-redb`)

### Hidden States (Core Vision #4) - 95% Complete ✅

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
- [x] **Candle Integration**: Real BERT-based `CandleHiddenStateProvider` with HuggingFace Hub model loading, full tokenisation pipeline, real forward-pass hidden state extraction (`hidden_states/candle_provider.rs`, feature `hidden-states + speculator`)
- [x] **Hidden State Pooling**: `HiddenStatePooling` enum (CLS/MeanPool/MaxPool/MaskMean), `apply_hidden_state_pooling` free function, `extract_sentence_embedding` for pooled sentence-level vectors

### On-the-fly Distillation (Core Vision #3) - 100% Complete ✅

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
- [x] **Real LoRA Training**: Complete Candle-based LoRA training system (`candle_lora.rs`)

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
- [x] **Persistent Vector Store**: redb-backed `RedbVectorStore` with full `VectorStore` trait impl, persistence across restarts, similarity search (`layer1_echo/storage/redb.rs`, feature `echo-redb`)
- [x] **Real Embedding Models**: `CandleEmbeddingProvider` (all-MiniLM-L6-v2 default) + presets for BGE-base/large/small-en-v1.5 and all-mpnet-base-v2 (`embedding/candle.rs`)

### Layer 2: Speculator (Draft Verification)

- [x] Rule-based speculator
- [x] Mock SLM speculator
- [x] Streaming verification support
- [x] Confidence calibration system
- [x] SmallLanguageModel trait and MockSlm
- [x] Multi-stage verification pipeline
- [x] **Quantization Types**: INT8, INT4, Binary quantization support (`quantization.rs`)
- [x] Complete Candle SLM integration with Phi-2/Phi-3 (production-ready)

### Layer 3: Judge (SMT Verification)

- [x] Pattern-based claim extractor
- [x] Advanced claim extractor
- [x] Mock SMT verifier
- [x] Temporal, causal, and modal claim structure types
- [x] Claim deduplication and normalization
- [x] Explanation generation for verification results
- [x] **Dependency Parsing**: Improved claim extraction with SVO extraction (`dependency_parser.rs`)
- [x] **Incremental Consistency**: Incremental consistency checking with conflict detection (`incremental.rs`)
- [x] **Real OxiZ SMT solver integration**: Complete with timeout handling, 26 tests, 9 benchmarks (`oxiz_verifier.rs`)

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
- [x] **Persistent Graph Store**: redb-backed `RedbGraphStore` with full `GraphStore` trait impl, atomic entity/relationship writes, BFS traversal, NAME\_IDX/TYPE\_IDX secondary indexes, 12 unit tests (`layer4_graph/redb_store.rs`, feature `graphrag-redb`)

### Pipeline

- [x] Unified pipeline with fast-path optimization
- [x] Pipeline builder pattern
- [x] Retry logic with exponential backoff
- [x] Parallel layer execution
- [x] Metrics and tracing
- [x] Batch query processing
- [x] **Pipeline Debugging**: Visualization and tracing tools (`pipeline_debug.rs`)
- [x] **Circuit Breaker**: Resilience pattern for external service failures (`circuit_breaker.rs`)
- [x] **Observability**: `PipelineSpanContext` with RAII `LayerSpan<'_>` guard, `SpanReport` ASCII/JSON table, `record_pipeline_event` free function — zero external tracing backend required (`observability.rs`)

---

## Medium Priority

### Performance

- [x] **SIMD Optimization**: Hardware-accelerated similarity computation with ARM NEON + AVX/SSE2 (`similarity_simd.rs`)
- [x] **Connection Pooling**: Generic connection pool for external services (`connection_pool.rs`)
- [x] **Memory Monitoring**: Memory limits and usage tracking (`memory.rs`)
- [x] **Profile and optimize hot paths**: 5.6x-9.0x speedup for cosine similarity, 8x for search workloads

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
- [x] **Property-based testing**: 60+ proptest tests for vectors, cache, graphs, claims, normalization
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

## Completed (v0.1.1)

### OxiZ SMT Solver Integration
- [x] Real OxiZ SMT solver implementation (`oxiz_verifier.rs`)
- [x] Timeout handling with configurable SolverConfig
- [x] 26 comprehensive tests covering all claim types
- [x] 9 performance benchmarks for solver operations
- [x] Support for predicate, numeric, temporal, causal, modal claims
- [x] Batch verification and consistency checking
- [x] Zero unwrap() in production code

### SIMD Performance Optimization
- [x] ARM NEON intrinsics for Apple Silicon (`similarity_simd.rs`)
- [x] x86_64 AVX and SSE2 intrinsics for Intel/AMD
- [x] Platform-specific optimizations with compile-time dispatch
- [x] 5.6x-9.0x speedup for cosine similarity
- [x] 8x speedup for 5000 document search workloads
- [x] Safe abstractions around unsafe SIMD code
- [x] Comprehensive performance report at `/tmp/oxirag_performance_report.md`

### Property-Based Testing
- [x] Added proptest 1.10.0 to dev-dependencies
- [x] 60+ property-based tests across modules
- [x] Vector operations: commutativity, range checks, normalization idempotence
- [x] Cache eviction: LRU correctness, size limits, deterministic behavior
- [x] Graph traversal: shortest path properties, BFS correctness, hop limits
- [x] Claim extraction: SMT-LIB generation validation
- [x] Query normalization: idempotence, consistency, whitespace handling

### Bug Fixes
- [x] Fixed `test_search_performance_scales` timing for debug builds
- [x] Eliminated unwrap() from `src/distillation/progressive.rs`
- [x] Improved error handling across codebase

### Test Suite Expansion
- [x] Expanded from 1,500 to 1,451 tests (some consolidated)
- [x] All tests passing with zero warnings
- [x] Zero clippy warnings with `--all-features`

---

## Completed (v0.3.0)

### Persistent Graph Store (`graphrag-redb`)
- [x] `RedbGraphStore` with 6 redb tables (ENTITIES, RELATIONSHIPS, OUTGOING, INCOMING, NAME\_IDX, TYPE\_IDX)
- [x] Full `GraphStore` trait implementation with atomic write transactions
- [x] BFS traversal mirroring `InMemoryGraphStore` exactly
- [x] Exact index hit + full-scan fallback for `find_entities_by_name`
- [x] Persistence across restarts with entity/relationship count restoration
- [x] 12 unit tests including persistence round-trip test
- [x] Feature gate: `graphrag-redb = ["graphrag", "dep:redb"]`

### Observability (`observability.rs`)
- [x] `PipelineSpanContext` with UUID execution id, per-context attributes, elapsed timing
- [x] `LayerSpan<'_>` RAII guard with panic-safe `Drop` (synthesises Error record on unwind)
- [x] `SpanStatus`: Success / Error(String) / Skipped / InProgress
- [x] `LayerSpanRecord` with duration\_ms, item\_count, arbitrary attributes
- [x] `SpanReport` with `format_table()` (ASCII) and `to_json()` (serde\_json)
- [x] `record_pipeline_event` free function for lightweight event logging
- [x] Zero external dependencies (uses only `tracing` + `uuid` already in scope)
- [x] 12 unit tests covering all commit paths and panic-safe Drop
- [x] Re-exported from crate prelude in `lib.rs`

### Hidden State Pooling
- [x] `HiddenStatePooling` enum: `Cls` (default), `MeanPool`, `MaxPool`, `MaskMean`
- [x] `apply_hidden_state_pooling` pure free function (testable without model loading)
- [x] `CandleHiddenStateConfig.pooling` field + `with_pooling` builder
- [x] `extract_sentence_embedding` method on `CandleHiddenStateProvider`
- [x] 6 pooling unit tests verifying exact arithmetic for each strategy

### Expanded Property-Based Testing
- [x] 8 proptest tests in `prefix_cache/redb_backend.rs`: put/get roundtrip, TTL expiry, multi-key isolation, clear semantics, capacity enforcement
- [x] 8 proptest tests in `layer1_echo/storage/redb.rs`: document roundtrip, upsert idempotence, dimension rejection, search result ordering
- [x] 7 tests in `embedding/candle.rs`: preset field validation, URL format, max\_length ranges
- [x] 9 tests in `hidden_states/candle_provider.rs`: config builder, sequence-length variants, pooling variant existence

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
| v0.1.1    | 100%            | 95%            | 100%         | 90%           |
| **v0.2.0**| **100%** ✅     | **98%** ✅     | **100%** ✅  | **95%** ✅    |
| **v0.3.0**| **100%** ✅     | **99%** ✅     | **100%** ✅  | **98%** ✅    |
| v1.0.0    | 100%            | 100%           | 100%         | 100%          |

### Codebase Statistics (v0.3.0)
- **Source Files**: 99 Rust files (+2 from v0.2.0: `layer4_graph/redb_store.rs`, `observability.rs`)
- **Total Lines**: ~68,000 (Rust code)
- **Tests**: 1,569 (+65 from v0.2.0: 12 RedbGraphStore, 12 observability, 8 prefix_cache proptest, 8 echo-redb proptest, 7 embedding preset, 9 hidden-state pooling, 9 more candle_provider)
- **Clippy Warnings**: 0
- **Rustdoc Warnings**: 0
- **New Features (v0.3.0)**: RedbGraphStore (feature `graphrag-redb`), HiddenStatePooling + apply\_hidden\_state\_pooling, extract\_sentence\_embedding, PipelineSpanContext / LayerSpan / SpanReport observability module
- **New Features (v0.2.0)**: RedbPrefixCache (feature `prefix-cache-redb`), RedbVectorStore (feature `echo-redb`), CandleHiddenStateProvider (features `hidden-states+speculator`), BGE/MPNet embedding presets
- **Refactored (v0.2.0)**: `distillation/progressive.rs` (1741 lines) → `distillation/progressive/` (4 files, all under 700 lines)
- **Performance**: 5.6x-9.0x faster similarity computations with SIMD (unchanged)
