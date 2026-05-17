# Layer 1: Echo — Semantic Search

Layer 1 (Echo) is the foundation of OxiRAG's retrieval pipeline. It converts
documents and queries into dense vector embeddings and performs similarity search
to retrieve the most relevant documents. Every query that enters the pipeline
passes through Echo first.

## Architecture Overview

```
Document text
     │
     ▼
┌─────────────────────┐
│  EmbeddingProvider  │  embed(&str) → Vec<f32>
│  (E: generic)       │
└─────────────────────┘
     │  Vec<f32>
     ▼
┌─────────────────────┐
│    VectorStore      │  insert / search / delete
│    (V: generic)     │
└─────────────────────┘
     │  Vec<SearchResult>
     ▼
  Pipeline continues → Layer 2 (Speculator)
```

`EchoLayer<E, V>` binds these two components. It is generic so that any combination
of embedding provider and vector store can be used. In production you swap the
concrete types; the rest of the pipeline is unchanged.

Relevant source files:

- `src/layer1_echo/mod.rs` — `EchoLayer` struct and `Echo` trait impl
- `src/layer1_echo/traits.rs` — `EmbeddingProvider`, `VectorStore`, `Echo` traits
- `src/layer1_echo/embedding/` — concrete embedding providers
- `src/layer1_echo/storage/` — concrete vector stores
- `src/layer1_echo/ann/` — HNSW approximate nearest neighbor index

## Quick Start with MockEmbeddingProvider

`MockEmbeddingProvider` generates deterministic hash-based embeddings without
downloading any model. It is suitable for testing, CI, and for validating your
indexing pipeline before committing to a specific embedding model.

```rust
use oxirag::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use oxirag::layer1_echo::Echo;
use oxirag::types::Document;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Choose an embedding dimension. Must be consistent across provider and store.
    let dim = 384;

    let mut echo = EchoLayer::new(
        MockEmbeddingProvider::new(dim),
        InMemoryVectorStore::new(dim),
    );

    // Index some documents.
    echo.index(Document::new("Rust is a systems programming language.")).await?;
    echo.index(Document::new("Python is popular for data science.")).await?;
    echo.index(Document::new("LLVM is a compiler infrastructure framework.")).await?;

    println!("Indexed {} documents", echo.count().await);

    // Search returns results ordered by descending similarity score.
    let results = echo.search("low-level programming", 3, None).await?;
    for result in &results {
        println!("[{:.3}] {}", result.score, result.document.content);
    }

    Ok(())
}
```

### Batch Indexing

For large document collections, use `index_batch` to embed all documents in a
single call. This is significantly faster than calling `index` in a loop because
the embedding provider can parallelize the work across the batch.

```rust
let docs = vec![
    Document::new("Document one: introduction to Rust."),
    Document::new("Document two: memory safety without garbage collection."),
    Document::new("Document three: zero-cost abstractions in Rust."),
];

let ids = echo.index_batch(docs).await?;
println!("Indexed {} documents in a single batch", ids.len());
```

### Document Metadata

Documents can carry arbitrary string metadata. Metadata is stored alongside the
document and is preserved in `SearchResult` objects returned from search.

```rust
let doc = Document::new("Rust ownership rules prevent data races.")
    .with_metadata("category", "memory-safety")
    .with_metadata("source", "rustbook")
    .with_metadata("year", "2024");

echo.index(doc).await?;
```

### Update and Upsert

```rust
// Update: re-embeds the document content. Returns Err if the document
// does not exist. Only updates the embedding, not the stored content.
let mut updated = Document::new("Updated content about Rust ownership.");
updated.id = existing_id.clone();
echo.update_document(&updated).await?;

// Upsert: inserts if new (returns true), updates if exists (returns false).
let was_inserted = echo.upsert_document(doc).await?;
```

## Using a Real Embedding Model

For production retrieval quality, use `CandleEmbeddingProvider` backed by a
HuggingFace BERT-family model. This requires the `speculator` feature (Candle
is shared between the embedding provider and the SLM speculator).

```toml
# Cargo.toml
oxirag = { version = "0.6", features = ["speculator"] }
```

```rust
use oxirag::layer1_echo::{
    CandleEmbeddingProvider, CandleEmbeddingConfig, CandleDevice,
    EchoLayer, InMemoryVectorStore,
};
use oxirag::layer1_echo::Echo;

// Default: sentence-transformers/all-MiniLM-L6-v2 (384 dim, 22M params, fast)
let config = CandleEmbeddingConfig::default();
let provider = CandleEmbeddingProvider::new(config)?;

// The provider knows its dimension; pass it to the store.
let dim = provider.dimension();
let store = InMemoryVectorStore::new(dim);

let mut echo = EchoLayer::new(provider, store);
```

### Embedding Model Presets

`CandleEmbeddingConfig` provides factory methods for common presets:

| Preset | Model | Dimension | Notes |
|---|---|---|---|
| `default()` | all-MiniLM-L6-v2 | 384 | Fast, lightweight; good for CPU |
| `bge_base_en_v15()` | BAAI/bge-base-en-v1.5 | 768 | Recommended default for accuracy |
| `bge_large_en_v15()` | BAAI/bge-large-en-v1.5 | 1024 | Best accuracy on MTEB |
| `bge_small_en_v15()` | BAAI/bge-small-en-v1.5 | 384 | Fastest, lowest memory |
| `all_mpnet_base_v2()` | all-mpnet-base-v2 | 768 | Strong semantic similarity |

```rust
// Recommended for most production English workloads:
let config = CandleEmbeddingConfig::bge_base_en_v15();
let provider = CandleEmbeddingProvider::new(config)?;

// When accuracy is the top priority:
let config = CandleEmbeddingConfig::bge_large_en_v15();
let provider = CandleEmbeddingProvider::new(config)?;
```

Models are downloaded from HuggingFace Hub on first use and cached in
`~/.cache/huggingface/` (or the path set by `HF_HOME`). Set `HF_HOME` to
a custom directory if you need to control the cache location:

```sh
export HF_HOME=/data/models
```

### CUDA Acceleration

When compiling for x86_64 Linux or Windows with a CUDA toolkit installed:

```toml
oxirag = { version = "0.6", features = ["cuda"] }
```

```rust
use oxirag::layer1_echo::{CandleDevice, CandleEmbeddingConfig};

// GPU 0
let config = CandleEmbeddingConfig {
    device: CandleDevice::Cuda(0),
    ..CandleEmbeddingConfig::bge_base_en_v15()
};
```

### Embedding Cache

Wrap any `EmbeddingProvider` with `CachedEmbeddingProvider` to avoid redundant
embedding calls for repeated text:

```rust
use oxirag::layer1_echo::{CachedEmbeddingProvider, EmbeddingCacheConfig, MockEmbeddingProvider};

let inner = MockEmbeddingProvider::new(384);
let cache_config = EmbeddingCacheConfig {
    max_entries: 10_000,
    ..EmbeddingCacheConfig::default()
};
let provider = CachedEmbeddingProvider::new(inner, cache_config);

let mut echo = EchoLayer::new(provider, InMemoryVectorStore::new(384));
```

`CacheStats` tracks hits, misses, and evictions:

```rust
let stats = echo.embedding_provider().stats();
println!("Cache hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

## Persistent Storage with RedbVectorStore

For production deployments that survive process restarts, use the redb-backed
persistent vector store. Requires the `echo-redb` feature.

```toml
# Cargo.toml
oxirag = { version = "0.6", features = ["echo-redb"] }
```

```rust
use oxirag::layer1_echo::storage::RedbVectorStore;
use oxirag::layer1_echo::{EchoLayer, MockEmbeddingProvider};

// Open (or create) a redb database file. The dimension must match the
// embedding provider's output dimension.
let store = RedbVectorStore::open("./data/vectors.redb", 384)?;

// Use exactly like InMemoryVectorStore.
let mut echo = EchoLayer::new(MockEmbeddingProvider::new(384), store);
echo.index(Document::new("Persistent document")).await?;

// The data survives process restarts — reopen the same file.
```

`RedbVectorStore` uses ACID transactions: each `insert`, `insert_batch`, `delete`,
and `update` is wrapped in a redb write transaction. Reads use snapshot-consistent
read transactions. The file can be safely opened from multiple processes in
read-only mode.

### When to use redb vs. in-memory

| Scenario | Recommended store |
|---|---|
| Tests, CI, ephemeral workloads | `InMemoryVectorStore` |
| Persistent single-process server | `RedbVectorStore` |
| >10K documents, latency-critical | `AnnVectorStore` |
| Browser/WASM with persistence | `IndexedDbVectorStore` |

## HNSW Approximate Nearest Neighbor Search

For document collections larger than approximately 10,000 documents, brute-force
linear scan becomes the bottleneck. Switch to the HNSW-based `AnnVectorStore`
for sub-linear search time:

```rust
use oxirag::layer1_echo::ann::{AnnVectorStore, AnnConfig};
use oxirag::layer1_echo::{EchoLayer, MockEmbeddingProvider};

let config = AnnConfig {
    m: 16,              // Number of bidirectional links per node. Higher = better
                        // recall at the cost of memory. Default: 16.
    ef_construction: 200, // Build-time beam width. Higher = better index quality,
                          // slower construction. Default: 200.
    ef_search: 50,      // Search-time beam width. Higher = better recall,
                        // slower queries. Default: 50.
    ..AnnConfig::default()
};

let store = AnnVectorStore::new(384, config);
let mut echo = EchoLayer::new(MockEmbeddingProvider::new(384), store);
```

`AnnVectorStore` implements the same `VectorStore` trait as `InMemoryVectorStore`,
so swapping is a one-line change. Search recall (fraction of true nearest neighbors
returned) depends on `ef_search`: at the default of 50, recall exceeds 99% for
most real-world corpora.

Query `AnnStats` after a workload to tune parameters:

```rust
let stats = echo.vector_store().stats();
println!("HNSW layers: {}", stats.num_layers);
println!("Avg neighbors per node: {:.1}", stats.avg_connections);
```

## Multi-Modal Search with CLIP

CLIP (Contrastive Language-Image Pre-Training) encodes text and images into a
shared embedding space, enabling cross-modal search (query with text, retrieve
images or vice versa). Requires the `multimodal` feature (native only — not
available on WASM).

```toml
oxirag = { version = "0.6", features = ["multimodal"] }
```

```rust
use oxirag::layer1_echo::{CandleClipProvider, ClipPreset, EchoLayer, InMemoryVectorStore};
use oxirag::layer1_echo::traits::{EmbeddingInput, MultiModalEmbeddingProvider};

// Load the CLIP ViT-B/32 model (~340 MB checkpoint).
let provider = CandleClipProvider::from_preset(ClipPreset::VitBase32)?;
let dim = provider.dimension(); // 512 for ViT-B/32

// Thanks to the blanket impl (ADR-0002), CandleClipProvider works as an
// EmbeddingProvider for text-only pipelines transparently.
let store = InMemoryVectorStore::new(dim);
let mut echo = EchoLayer::new(provider, store);

// Index image documents by embedding their byte content.
let image_bytes = std::fs::read("photo.jpg")?;
let embedding = echo.embedding_provider()
    .embed_multi(EmbeddingInput::Image(&image_bytes))
    .await?;

// For cross-modal text → image search, search normally with a text query.
let results = echo.search("sunset over mountains", 5, None).await?;
```

## Metadata Filtering

`EchoLayer::search_with_filter` combines semantic similarity with structured
metadata constraints. Results must satisfy both the similarity threshold and the
filter predicate.

```rust
use oxirag::layer1_echo::filter::MetadataFilter;

// Equality filter.
let filter = MetadataFilter::eq("category", "science");
let results = echo
    .search_with_filter("quantum mechanics", 10, None, Some(&filter))
    .await?;

// AND filter: category = "science" AND status = "published".
let filter = MetadataFilter::and(vec![
    MetadataFilter::eq("category", "science"),
    MetadataFilter::eq("status", "published"),
]);
let results = echo
    .search_with_filter("wave-particle duality", 5, Some(0.7), Some(&filter))
    .await?;

// OR filter: category = "physics" OR category = "chemistry".
let filter = MetadataFilter::or(vec![
    MetadataFilter::eq("category", "physics"),
    MetadataFilter::eq("category", "chemistry"),
]);
```

The filter is applied after the vector similarity ranking, meaning the raw
candidate pool from the vector store may be larger than `top_k` to account for
filtered-out results. In the worst case (all candidates filtered), the result
set is empty.

## Multi-Vector Documents (ColBERT-style)

For fine-grained token-level matching, use `InMemoryMultiVectorStore` with a
`TokenEmbeddingProvider`. This is inspired by the ColBERT late-interaction model
where each token gets its own embedding and matching uses the MaxSim operator.

```rust
use oxirag::layer1_echo::{
    InMemoryMultiVectorStore, MockTokenEmbeddingProvider,
    TokenEmbeddingProvider, MultiVectorDocument,
};

let provider = MockTokenEmbeddingProvider::new(128);
let store = InMemoryMultiVectorStore::new(128);

let multi_doc = MultiVectorDocument::from_tokens(
    "doc_1",
    &["Rust", "memory", "safety"],
    &provider,
).await?;

store.insert(multi_doc).await?;

// MaxSim search: each query token matched against all document tokens,
// sum of per-token maximum similarities determines the final score.
let query_embeddings = provider.embed_tokens(&["safety", "programming"]).await?;
let results = store.max_sim_search(&query_embeddings, 5).await?;
```

## Performance Tips

### Batch vs. Sequential Indexing

Always prefer `index_batch` over a loop of `index` calls for initial loading.
The embedding provider groups all texts into a single forward pass through the
model:

```rust
// Slow: one forward pass per document (n passes total)
for doc in docs.iter() {
    echo.index(doc.clone()).await?;
}

// Fast: one forward pass for all n documents
echo.index_batch(docs).await?;
```

### Embedding Dimension Selection

Lower-dimensional embeddings are faster to compute, index, and search:

- 384-dim (MiniLM, BGE-small): best for CPU-bound production, edge devices
- 768-dim (BGE-base, MPNet): recommended balance for accuracy-sensitive workloads
- 1024-dim (BGE-large): maximum MTEB accuracy, requires ~2x memory vs. 768-dim

### SIMD Acceleration

On x86_64, OxiRAG automatically uses AVX2 or SSE2 intrinsics for cosine
similarity. On Apple Silicon, NEON intrinsics are used. These paths are in
`src/layer1_echo/similarity_simd.rs` and are selected at compile time — no
runtime feature detection is needed.

```sh
# Enable AVX2 explicitly for x86_64 Linux:
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Measured speedup: 5.6x–9.0x for cosine similarity vs. scalar Rust on a
5,000-document workload (see `docs/perf.md` for detailed benchmarks).

### Minimum Score Threshold

Pass `min_score` to avoid returning low-confidence results even when `top_k`
slots are available:

```rust
// Only return results with cosine similarity >= 0.75.
let results = echo.search("quantum entanglement", 10, Some(0.75)).await?;
```

### Document Count Scaling Guide

| Document count | Recommended store | Search latency (384-dim, CPU) |
|---|---|---|
| < 1,000 | `InMemoryVectorStore` | < 1 ms |
| 1,000 – 50,000 | `InMemoryVectorStore` or `AnnVectorStore` | 1–20 ms |
| 50,000 – 1,000,000 | `AnnVectorStore` | 5–50 ms |
| > 1,000,000 | `AnnVectorStore` + shard | 20–100 ms per shard |
