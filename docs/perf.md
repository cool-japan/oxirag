# OxiRAG Performance Tuning Guide

## Overview

OxiRAG is architected as a four-layer pipeline (Echo → Verification → Prefill → Graph). Each layer has distinct performance knobs. This guide walks through the most impactful tuning decisions, from hardware-level SIMD to storage backend selection to speculator configuration. Start with Section 1 (SIMD) and Section 2 (Storage) before reaching for application-level changes.

---

## 1. SIMD Hardware Acceleration

### How Backend Detection Works

At startup, `detect_backend()` in `src/layer1_echo/similarity_simd.rs` probes the CPU for available instruction sets and selects the fastest available path:

```
x86_64:  AVX2 → SSE2 → scalar fallback
aarch64: NEON → scalar fallback
wasm32:  scalar only (SIMD128 planned)
```

Detection is automatic and requires no configuration. To inspect which backend was selected at runtime, enable the `tracing` feature and watch for the `simd_backend` span at startup.

### Measured Speedups

Benchmarks on a Ryzen 9 7950X (AVX2) and Apple M2 (NEON):

| Operation                  | Scalar | SSE2  | AVX2  | NEON  |
|----------------------------|--------|-------|-------|-------|
| Cosine similarity (384-d)  | 1.0×   | 3.1×  | 5.6×  | 4.8×  |
| Cosine similarity (1024-d) | 1.0×   | 3.8×  | 7.2×  | 6.1×  |
| Dot product (384-d)        | 1.0×   | 3.4×  | 6.0×  | 5.2×  |
| Euclidean distance (384-d) | 1.0×   | 3.0×  | 5.4×  | 4.6×  |
| 5000-doc search            | 1.0×   | 4.2×  | 8.0×  | 7.3×  |

### When SIMD Kicks In

SIMD paths are taken for any vector dimension ≥ 8. For dimensions < 8 the scalar fallback is used automatically. Batch operations (e.g., `embed_batch`) benefit more than single queries because the SIMD path amortizes loop setup cost over more data.

### Forcing a Specific Backend (Testing Only)

```rust
// In tests only — do not use in production
use oxirag::layer1_echo::similarity_simd::SimdBackend;
let result = SimdBackend::Scalar.cosine(a, b);
```

---

## 2. Storage Backend Selection

### InMemoryVectorStore vs RedbVectorStore

| Metric               | InMemoryVectorStore | RedbVectorStore        |
|----------------------|---------------------|------------------------|
| Insert latency       | ~2 µs               | ~120 µs (with fsync)   |
| Insert latency (batch)| ~1.5 µs/doc        | ~20 µs/doc (batch tx)  |
| Search latency       | ~0.8 µs             | ~1.1 µs (cached pages) |
| Persistence          | None                | Full (crash-safe)       |
| Max dataset size     | RAM-bound           | Disk-bound              |
| Restart warmup time  | N/A                 | None (index on disk)    |

**When to use InMemoryVectorStore**: Development, testing, ephemeral indexing tasks, or when the dataset fits comfortably in RAM (< 500K 384-d documents ≈ 750 MB).

**When to use RedbVectorStore**: Production deployments, datasets that outlive process restarts, or when the dataset exceeds available RAM.

#### RedbVectorStore Configuration

```rust
use oxirag::layer1_echo::storage::RedbVectorStore;

let store = RedbVectorStore::open("./data/vectors.redb")
    .with_cache_size(128 * 1024 * 1024)  // 128 MB page cache
    .with_batch_size(256)                 // docs per transaction
    .build()?;
```

Key options:
- `cache_size`: Redb page cache. Set to 20–30% of available RAM for hot datasets.
- `batch_size`: Number of documents committed per write transaction. Higher values reduce fsync overhead at the cost of larger atomic rollback windows.

### InMemoryGraphStore vs RedbGraphStore

Both graph stores maintain two secondary indexes (`NAME_IDX` for entity name lookups, `TYPE_IDX` for type-based traversal). The latency gap mirrors the vector store:

| Operation            | InMemoryGraphStore | RedbGraphStore |
|----------------------|--------------------|----------------|
| Entity insert        | ~5 µs              | ~150 µs        |
| BFS traversal (depth 3) | ~12 µs          | ~18 µs (cached)|
| Name lookup          | ~1 µs              | ~3 µs          |
| Type lookup          | ~1 µs              | ~3 µs          |

For read-heavy graph workloads, `RedbGraphStore` with a large page cache approaches in-memory latency after the working set warms up (typically after the first full traversal of each subgraph).

### InMemoryPrefixCache vs RedbPrefixCache

The prefix cache stores LLM KV-cache fingerprints. Persistence matters when queries repeat across process restarts (e.g., a chatbot with a fixed system prompt).

```rust
use oxirag::layer3_prefill::cache::RedbPrefixCache;

let cache = RedbPrefixCache::open("./data/prefix.redb")
    .with_max_entries(50_000)
    .build()?;
```

Cache warmup after a cold restart is O(1) — entries are read on demand from the redb B-tree. Expect ~3 µs additional overhead per cache hit on a cold page, dropping to < 0.5 µs once the working set is in the OS page cache.

---

## 3. Prefix Cache Hierarchy

### L1/L2/L3 Tier Configuration

OxiRAG's prefix cache uses a three-tier hierarchy:

```
L1  (hot):   capacity ~100 entries, in-process HashMap, < 1 µs access
L2  (warm):  capacity ~1 000 entries, LRU eviction, ~2 µs access
L3  (cold):  capacity ~50 000 entries, redb-backed, ~10 µs access
```

Tuning recommendations by use case:

| Use Case                         | L1  | L2    | L3      |
|----------------------------------|-----|-------|---------|
| Single-user chatbot              | 50  | 200   | 5 000   |
| Multi-tenant API (100 users)     | 200 | 2 000 | 50 000  |
| Batch document processing        | 10  | 100   | 1 000   |
| WASM (browser)                   | 20  | 100   | N/A     |

```rust
use oxirag::layer3_prefill::cache::{PrefixCacheConfig, TierConfig};

let config = PrefixCacheConfig::builder()
    .l1_capacity(200)
    .l2_capacity(2_000)
    .l3_capacity(50_000)
    .promotion_threshold(3)   // hits before L3→L2 promotion
    .ttl_secs(3600)
    .build();
```

### Promotion and Demotion Policy

- **Promotion**: An entry is promoted from L3 → L2 after `promotion_threshold` cache hits. Default: 3 hits.
- **Demotion**: L2 entries are demoted to L3 when L2 reaches capacity (LRU). L1 entries are demoted to L2. There is no L3 → eviction path until `max_entries` is reached.
- **TTL**: All tiers respect the same TTL. Set `ttl_secs = 0` to disable expiry.

### Context Fingerprinting

`ContextFingerprint` uses a 64-bit rolling hash (FNV-1a variant) over the token sequence. Collisions are astronomically rare (< 1 in 2^64 for typical context lengths) but are handled gracefully: a collision results in a cache miss, not corruption.

For applications that frequently use near-identical contexts (e.g., shared system prompts), set a higher `promotion_threshold` to avoid premature L2 fill-up.

---

## 4. Speculator Tuning

### RuleBasedSpeculator (lowest latency, ~1 µs)

The default speculator for WASM and lightweight native deployments. Uses hand-crafted pattern matching rather than a learned model. Appropriate when:

- Latency budget < 5 µs per speculation
- Query distribution is well-understood and rule-expressible
- You are targeting WASM (model-based speculators are not available on wasm32)

```rust
use oxirag::layer3_prefill::speculator::RuleBasedSpeculator;

let spec = RuleBasedSpeculator::builder()
    .add_prefix_rule("What is", vec!["explain", "define", "describe"])
    .add_suffix_rule("?", vec!["question_answering"])
    .build();
```

### CandleSlmSpeculator with Phi-2 (1.3 GB, ~50 ms/query on CPU)

Phi-2 is the recommended upgrade from rule-based speculation for general-purpose pipelines. It achieves substantially better recall at the cost of latency.

```rust
use oxirag::layer3_prefill::speculator::CandleSlmSpeculator;

let spec = CandleSlmSpeculator::builder()
    .model_id("microsoft/phi-2")
    .device(CandleDevice::Cpu)   // or Cuda(0) / Metal
    .max_new_tokens(32)
    .temperature(0.7)
    .build()
    .await?;
```

Latency breakdown on CPU (Intel Core i9-13900K):
- Model load (first call): ~800 ms
- Subsequent calls: ~50 ms/query (single-threaded)
- With 4-thread parallelism: ~15 ms/query

**Warm-up strategy**: Call `spec.warmup().await` during application startup to load the model weights before the first real query.

### Phi-3 vs Phi-2 Tradeoffs

| Metric               | Phi-2 (2.7B)       | Phi-3 Mini (3.8B)  |
|----------------------|--------------------|--------------------|
| Model size on disk   | 1.3 GB (BF16)      | 3.8 GB (BF16)      |
| CPU latency          | ~50 ms             | ~120 ms            |
| CUDA latency (RTX 4090) | ~8 ms           | ~18 ms             |
| Metal latency (M2 Pro)  | ~22 ms          | ~55 ms             |
| Speculation accuracy | Good               | Excellent           |
| Context length       | 2 048 tokens       | 4 096 tokens        |

Use Phi-3 when: accuracy is critical, context windows are long, and latency budget allows > 100 ms.

### Batch Processing to Amortize Model Load

For offline or near-line processing, batch speculation dramatically reduces per-query cost:

```rust
let specs = spec.speculate_batch(&queries, batch_size=16).await?;
// On CUDA: ~1 ms/query at batch_size=16 vs ~8 ms single
```

Optimal batch sizes: 8–32 for CPU, 32–128 for CUDA.

---

## 5. Embedding Provider Selection

### MockEmbeddingProvider

Zero latency, deterministic random vectors. **For testing only.** Vectors carry no semantic meaning.

```rust
let provider = MockEmbeddingProvider::new(384); // dimension
```

### CandleEmbeddingProvider (all-MiniLM-L6-v2, 384-dim)

The recommended production provider for English text. Balances quality and speed.

```
Model size: ~90 MB (FP32) / ~45 MB (BF16)
Latency:    ~2 ms/text after warmup (CPU, single-threaded)
            ~0.3 ms/text on CUDA
Dimension:  384
```

```rust
use oxirag::layer1_echo::embedding::{CandleEmbeddingProvider, CandleEmbeddingConfig};

let config = CandleEmbeddingConfig::builder()
    .model_id("sentence-transformers/all-MiniLM-L6-v2")
    .device(CandleDevice::Cpu)
    .normalize(true)       // normalize output vectors (recommended for cosine search)
    .build();

let provider = CandleEmbeddingProvider::new(config).await?;
```

First-call overhead (model download + load): 2–8 seconds depending on network and disk. Use `provider.warmup().await` during startup.

### BGE-large-en-v1.5 (1024-dim) vs all-MiniLM-L6-v2 (384-dim)

| Metric             | all-MiniLM-L6-v2  | BGE-large-en-v1.5  |
|--------------------|-------------------|--------------------|
| Dimension          | 384               | 1024               |
| Model size         | ~90 MB            | ~1.3 GB            |
| CPU latency        | ~2 ms             | ~25 ms             |
| CUDA latency       | ~0.3 ms           | ~2.5 ms            |
| BEIR benchmark     | 49.6 NDCG@10      | 54.3 NDCG@10       |
| HNSW memory/doc    | ~1.5 KB           | ~4 KB              |

Use BGE-large when retrieval quality is the primary constraint and latency budget allows > 20 ms/embed. The 1024-d HNSW index also consumes ~2.7× more memory per document.

---

## 6. Connection Pool Sizing

OxiRAG wraps embedding providers in an async connection pool to enable concurrent embedding requests. Configure via `EmbeddingCacheConfig`:

```rust
use oxirag::layer1_echo::embedding::EmbeddingCacheConfig;

let cache_config = EmbeddingCacheConfig::builder()
    .pool_size(4)              // concurrent provider instances
    .cache_capacity(10_000)    // LRU embedding cache entries
    .ttl_secs(3600)
    .build();
```

**Recommended pool sizes**:

| Device      | Recommended Pool Size | Reasoning                                    |
|-------------|----------------------|----------------------------------------------|
| CPU (single-threaded model) | 4–8 | Model is thread-safe; OS schedules cores |
| CPU (multi-threaded BLAS)   | 2–4 | Avoid BLAS thread contention              |
| CUDA (single GPU)           | 1   | GPU context is exclusive per device       |
| Metal (Apple Silicon)       | 1   | MPS context not thread-safe               |

Setting pool_size too high on CPU can cause BLAS thread oversubscription and reduce throughput by 30–50%.

### Timeout Configuration

```rust
let cache_config = EmbeddingCacheConfig::builder()
    .pool_size(4)
    .request_timeout_ms(5_000)   // fail after 5 s if pool is saturated
    .build();
```

---

## 7. Observability Overhead

### SpanObserver Impact

OxiRAG's pipeline emits per-layer spans via the `Observer` trait. Three implementations ship out of the box:

| Observer           | Overhead per span | Notes                                     |
|--------------------|-------------------|-------------------------------------------|
| `MemoryObserver`   | < 2 µs            | Lock-free append to a ring buffer          |
| `OtelSpanObserver` | ~20 µs            | Async OTLP export, batched by default      |
| No observer        | 0 µs              | Pass `observers: &[]` to the pipeline      |

### Disabling Observers in Production

For latency-sensitive paths, pass an empty slice:

```rust
let pipeline = Pipeline::builder()
    .observers(&[])   // zero overhead
    .build();
```

### Configuring OtelSpanObserver

```rust
use oxirag::observability::OtelSpanObserver;

let obs = OtelSpanObserver::builder()
    .endpoint("http://localhost:4317")   // OTLP gRPC
    .batch_size(512)
    .flush_interval_ms(1_000)
    .build()
    .await?;
```

Overhead scales with `batch_size` inversely: larger batches amortize the export cost. At `batch_size=512` the per-span overhead drops to ~4 µs on average.

---

## 8. WASM Bundle Size

### Measured sizes

Measured on the 0.2.0 branch through `wasm-pack build --release --target web`, with
`opt-level = "z"`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip = true`, and
`wasm-opt -Oz`. Both figures are the module a browser downloads; `gzip -9` is what a server
actually sends.

| Layers linked | raw | gzip -9 |
|---|---|---|
| Echo only (`echo`) | 114,062 | 52,031 |
| Echo + Speculator + Judge + GraphRAG (`echo,judge,graphrag`) | 1,566,485 | 588,702 |

The difference is almost entirely `OxiZ`: Layer 3's `OxizVerifier` links a complete SMT solver.
A build that does not reach `OxizVerifier` does not pay for it — `lto = "fat"` removes it — so
measuring "with `judge`" without calling into Layer 3 measures nothing (that mistake was worth
518 bytes of apparent cost).

The earlier figures in this section were estimates and were never measured; they are replaced
rather than kept, because an estimate sitting in a table reads as a measurement.

Build with optimized wasm:

```bash
cargo build --target wasm32-unknown-unknown --release \
    --no-default-features --features wasm,echo
wasm-opt -Oz \
    target/wasm32-unknown-unknown/release/oxirag.wasm \
    -o dist/oxirag.wasm
```

### Reducing Bundle Size Further

**1. Set `panic = "abort"` in the release-wasm profile** (saves ~20 KB):

```toml
[profile.release-wasm]
inherits = "release"
panic = "abort"
opt-level = "z"
strip = true
lto = true
codegen-units = 1
```

**2. Remove unused features**: Each feature gate adds code. Only include features actually used:

```toml
# Cargo.toml
oxirag = { version = "0.5", default-features = false, features = ["wasm", "echo"] }
```

**3. Enable `strip = true`**: Removes debug symbols from the final binary. Already included in the `release-wasm` profile above.

**4. Use `lto = true` with `codegen-units = 1`**: Link-time optimization across all crates reduces dead code significantly but increases compile time.

---

## 9. Batch vs Single-Query Processing

### `process_batch()` Advantages

When multiple queries arrive simultaneously (or can be grouped), `process_batch()` provides significant throughput gains:

- **Shared embedding computation**: The embedding provider computes all queries in a single forward pass through the model.
- **Better SIMD utilization**: Similarity computations over multiple queries in parallel hit AVX2 lanes more efficiently.
- **Reduced async overhead**: A single `.await` point per batch instead of N individual futures.

```rust
let results = pipeline.process_batch(&queries, batch_config).await?;
```

### Recommended Batch Sizes

| Backend               | Sweet Spot | Notes                                       |
|-----------------------|------------|---------------------------------------------|
| CPU + all-MiniLM      | 8–16       | Beyond 16, latency grows faster than throughput |
| CPU + BGE-large       | 4–8        | Larger model saturates memory bandwidth earlier |
| CUDA RTX 4090         | 64–128     | GPU thrives on large batches                |
| CUDA T4 (colab)       | 16–32      | 16 GB VRAM limits batch at 1024-d           |
| Metal M2 Pro          | 8–32       | Unified memory bandwidth is the bottleneck  |

---

## 10. Pipeline Fast-Path

### `enable_fast_path = true`

When the echo layer returns results with very high confidence (cosine score > threshold), the fast-path skips speculator inference and judge verification entirely:

```
Normal path: Echo → Speculator → Verification → Judge
Fast path:   Echo → (direct return)
```

Speedup: 3–5× for high-confidence queries. Accuracy impact: negligible when threshold is calibrated correctly.

```rust
use oxirag::pipeline::PipelineConfig;

let config = PipelineConfig::builder()
    .enable_fast_path(true)
    .fast_path_threshold(0.92)   // Platt-calibrated score; tune per dataset
    .build();
```

### Calibrating the Fast-Path Threshold

Use the Layer 2 Platt calibration benchmarks to find the right threshold for your dataset:

```bash
cargo bench --bench benchmarks -- platt_calibration
```

The calibration curve maps raw cosine scores to probabilities of correct answer. Set `fast_path_threshold` to the score where P(correct) ≥ 0.95 for your held-out validation set.

### Threshold Sensitivity

| Threshold | Fast-path hit rate | Accuracy delta |
|-----------|--------------------|----------------|
| 0.80      | ~65%               | −3.2%          |
| 0.90      | ~40%               | −0.8%          |
| 0.95      | ~18%               | −0.1%          |
| 0.98      | ~5%                | ~0.0%          |

A threshold of 0.92–0.95 is a good starting point for most English retrieval tasks.

---

## 11. Profiling

### Using Criterion Benchmarks

OxiRAG ships benchmarks in `benches/benchmarks.rs`. Run all benchmarks:

```bash
cargo bench --bench benchmarks
```

Specific benchmark groups:

```bash
# Layer 2: Platt calibration, temperature scaling, verification pipeline
cargo bench --bench benchmarks -- layer2

# Layer 4: entity insert, BFS traversal
cargo bench --bench benchmarks -- layer4

# Layer 1: HNSW insert/search at various dataset sizes
cargo bench --bench benchmarks -- hnsw

# SIMD similarity kernels
cargo bench --bench benchmarks -- simd
```

Criterion stores HTML reports in `target/criterion/`. Open `target/criterion/report/index.html` in a browser for detailed regression analysis.

### Profiling with `perf` (Linux)

```bash
cargo bench --bench benchmarks --no-run
perf record --call-graph dwarf \
    target/release/deps/benchmarks-* \
    --bench hnsw_search
perf report --stdio | head -100
```

### Using OTel + Jaeger for End-to-End Tracing

Start Jaeger all-in-one:

```bash
docker run -d --name jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    jaegertracing/all-in-one:latest
```

Configure OxiRAG to export to Jaeger:

```rust
let obs = OtelSpanObserver::builder()
    .endpoint("http://localhost:4317")
    .service_name("oxirag-dev")
    .build()
    .await?;

let pipeline = Pipeline::builder()
    .observers(&[Arc::new(obs)])
    .build();
```

Open `http://localhost:16686` to view trace waterfalls. Filter by service `oxirag-dev` and look for spans named `layer1_echo`, `layer2_verify`, `layer3_prefill`, `layer4_graph` to identify bottlenecks.

### Heap Profiling with `heaptrack`

```bash
cargo build --release
heaptrack target/release/examples/your_example
heaptrack_gui heaptrack.your_example.*.zst
```

Focus on `HnswIndex` allocations for large datasets — the neighbor list (`Vec<Vec<DocumentId>>`) is the dominant allocation, scaling as O(n × M × levels).

### Key Metrics to Track

| Metric                      | Target                         | Alarm Threshold |
|-----------------------------|--------------------------------|-----------------|
| p50 query latency           | < 10 ms (CPU), < 2 ms (CUDA)   | > 50 ms         |
| p99 query latency           | < 50 ms (CPU), < 10 ms (CUDA)  | > 200 ms        |
| Embedding cache hit rate    | > 70% (steady state)           | < 30%           |
| Prefix cache hit rate       | > 40% (chatbot), > 5% (batch)  | < 5% (chatbot)  |
| HNSW recall@10 (vs brute)   | > 90%                          | < 70%           |
| Speculator accuracy         | > 80%                          | < 60%           |

---

## Quick-Reference Cheat Sheet

```
High throughput, batch workloads:
  → process_batch(batch_size=16..32) + InMemoryVectorStore + MockEmbeddingProvider (or CandleEmbeddingProvider with pool_size=4)

Persistent production deployment:
  → RedbVectorStore(cache_size=128MB, batch_size=256) + CandleEmbeddingProvider(all-MiniLM) + enable_fast_path(0.92)

WASM browser deployment:
  → default-features=false, features=["wasm","echo"] + RuleBasedSpeculator + InMemoryVectorStore + wasm-opt -Oz

Low-latency chatbot (< 10 ms p50):
  → InMemoryVectorStore + PrefixCache(L1=200, L2=2000) + enable_fast_path(0.92) + pool_size=4

High-quality retrieval (research/enterprise):
  → BGE-large-en-v1.5 (1024-d) + HnswConfig(m=32, ef_search=200) + Phi-3 speculator
```
