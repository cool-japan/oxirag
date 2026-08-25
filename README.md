# OxiRAG

[![Crates.io](https://img.shields.io/crates/v/oxirag.svg)](https://crates.io/crates/oxirag)
[![Documentation](https://docs.rs/oxirag/badge.svg)](https://docs.rs/oxirag)
[![License](https://img.shields.io/crates/l/oxirag.svg)](LICENSE)

A four-layer Retrieval-Augmented Generation (RAG) engine in **Pure Rust** with SMT-based logic verification and knowledge graph support.

**Key Innovations:**
- **Speculative RAG**: Use cache as "drafts" not "answers" - verify with SLM before finalizing
- **Context-Aware Prefix Caching**: Efficiently manage KV Cache for "premise knowledge"
- **On-the-fly Distillation**: Automatically generate specialized lightweight models for frequent queries
- **Hidden States Manipulation**: Direct manipulation of transformer hidden states for verification

## Overview

OxiRAG provides a robust RAG pipeline with four specialized layers:

```
Query
  │
  ├───────────────────────────────┐
  ▼                               ▼
┌─────────────────┐       ┌─────────────────┐
│  Layer 1: Echo  │       │ Layer 4: Graph  │
│  (Vector Store) │       │   (GraphRAG)    │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └───────────┬─────────────┘
                     ▼
         ┌─────────────────────┐
         │ Layer 2: Speculator │  ← Draft verification with SLM
         │  (Draft Checker)    │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Layer 3: Judge │  ← Logic verification with SMT solver
         │  (SMT Verifier) │
         └────────┬────────┘
                  │
                  ▼
             Verified Answer
```

## Features

- **Layer 1 (Echo)**: Semantic search using vector embeddings
  - Cosine, Euclidean, and Dot Product similarity metrics
  - In-memory vector store with configurable capacity
  - Pluggable embedding providers (Candle BERT, mock for testing)

- **Layer 2 (Speculator)**: Draft verification
  - Rule-based speculator for quick verification
  - Candle-based SLM for advanced verification (optional)
  - Accept/Revise/Reject decision pipeline

- **Layer 3 (Judge)**: SMT-based logic verification
  - Claim extraction from natural language
  - SMT-LIB encoding of logical claims
  - OxiZ SMT solver integration for formal verification
  - Temporal, causal, and modal claim structures

- **Layer 4 (Graph)**: Knowledge graph-based retrieval (GraphRAG)
  - Entity and relationship extraction from documents
  - In-memory graph store with traversal algorithms
  - BFS traversal, shortest path, and N-hop queries
  - Hybrid search combining vector and graph results

- **Prefix Caching**: Context-aware KV cache management
  - Efficient caching of processed document contexts
  - Context fingerprinting for cache key generation
  - LRU eviction with TTL support
  - Prefix matching for partial cache hits

- **Distillation**: On-the-fly model distillation support
  - Query pattern frequency tracking
  - Automatic Q&A pair collection
  - Distillation candidate detection
  - Training example export for LoRA fine-tuning
  - Feature-based distillation (FitNet, attention transfer)
  - Teacher-student architecture with progressive learning
  - Distillation loss functions (KL, MSE, cosine)

- **Cross-Platform**: Native and WASM support
  - Full async/await support with Tokio (native)
  - WASM bindings for browser/edge deployment

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirag = "0.1"
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `native` | Enable native runtime with Tokio | Yes |
| `wasm` | Enable WASM bindings | No |
| `echo` | Enable Layer 1 (semantic search) | Yes |
| `speculator` | Enable Layer 2 with Candle SLM | No |
| `judge` | Enable Layer 3 with OxiZ SMT solver | No |
| `graphrag` | Enable Layer 4 (knowledge graph) | No |
| `prefix-cache` | Enable prefix caching for KV cache management | No |
| `distillation` | Enable distillation tracking and Q&A collection | No |
| `full` | Enable all features (native only) | No |
| `cuda` | Enable CUDA acceleration for Candle | No |
| `metal` | Enable Metal acceleration for Candle | No |

## Quick Start

```rust
use oxirag::prelude::*;

#[tokio::main]
async fn main() -> Result<(), OxiRagError> {
    // Create the Echo layer
    let echo = EchoLayer::new(
        MockEmbeddingProvider::new(384),
        InMemoryVectorStore::new(384),
    );

    // Create the Speculator layer
    let speculator = RuleBasedSpeculator::default();

    // Create the Judge layer
    let judge = JudgeImpl::new(
        AdvancedClaimExtractor::new(),
        MockSmtVerifier::default(),
        JudgeConfig::default(),
    );

    // Build the pipeline
    let mut pipeline = PipelineBuilder::new()
        .with_echo(echo)
        .with_speculator(speculator)
        .with_judge(judge)
        .build()?;

    // Index documents
    pipeline.index(Document::new("The capital of France is Paris.")).await?;
    pipeline.index(Document::new("Paris is known for the Eiffel Tower.")).await?;

    // Query the pipeline
    let query = Query::new("What is the capital of France?");
    let result = pipeline.process(query).await?;

    println!("Answer: {}", result.final_answer);
    println!("Confidence: {:.2}", result.confidence);
    println!("Layers used: {:?}", result.layers_used);

    Ok(())
}
```

## Architecture

### Layer 1: Echo

The Echo layer handles semantic search:

```rust
// Configure embedding provider
let provider = MockEmbeddingProvider::new(384);

// Configure vector store
let store = InMemoryVectorStore::new(384)
    .with_metric(SimilarityMetric::Cosine)
    .with_max_capacity(10000);

// Create Echo layer
let echo = EchoLayer::new(provider, store);
```

### Layer 2: Speculator

The Speculator layer verifies draft answers:

```rust
// Rule-based (default)
let speculator = RuleBasedSpeculator::default();

// Or with custom config
let config = SpeculatorConfig {
    accept_threshold: 0.9,
    reject_threshold: 0.3,
    ..Default::default()
};
let speculator = RuleBasedSpeculator::new(config);
```

### Layer 3: Judge

The Judge layer performs formal verification:

```rust
let judge = JudgeImpl::new(
    AdvancedClaimExtractor::new(),
    MockSmtVerifier::default(),
    JudgeConfig {
        max_claims: 10,
        check_consistency: true,
        ..Default::default()
    },
);
```

### Layer 4: Graph (GraphRAG)

The Graph layer enables knowledge graph-based retrieval:

```rust
use oxirag::layer4_graph::*;

// Create extractors
let entity_extractor = PatternEntityExtractor::new();
let relationship_extractor = PatternRelationshipExtractor::new();

// Create graph store
let graph_store = InMemoryGraphStore::new();

// Build the GraphLayer
let mut graph_layer = GraphLayerBuilder::new()
    .with_entity_extractor(entity_extractor)
    .with_relationship_extractor(relationship_extractor)
    .with_graph_store(graph_store)
    .build()?;

// Index documents (extracts entities and relationships)
graph_layer.index_document(&Document::new("Rust was created by Mozilla.")).await?;

// Query the graph
let query = GraphQuery::new(vec!["Rust".to_string()])
    .with_max_hops(2);
let paths = graph_layer.query(&query).await?;

// Find related entities
let related = graph_layer.find_related("Rust", 2).await?;
```

### Prefix Caching

The Prefix Cache module enables efficient caching of processed document contexts:

```rust
use oxirag::prefix_cache::*;

// Create a prefix cache
let config = PrefixCacheConfig::default()
    .with_max_entries(1000)
    .with_default_ttl_secs(3600);
let mut cache = InMemoryPrefixCache::new(config);

// Generate a fingerprint for context
let generator = ContextFingerprintGenerator::new();
let fingerprint = generator.generate("This is a large document context...");

// Create and store a KV cache entry
let entry = KVCacheEntry::new(fingerprint.clone(), vec![0.1, 0.2, 0.3], 512);
cache.put(entry).await?;

// Retrieve cached entry
if let Some(cached) = cache.get(&fingerprint).await {
    println!("Cache hit! Sequence length: {}", cached.sequence_length);
}

// Check cache statistics
let stats = cache.stats();
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
```

### Distillation

The Distillation module tracks query patterns for model fine-tuning:

```rust
use oxirag::distillation::*;

// Create a distillation tracker
let config = DistillationConfig::default()
    .with_min_frequency_threshold(10)
    .with_max_qa_pairs_per_pattern(100);
let mut tracker = InMemoryDistillationTracker::new(config);

// Track queries with their answers
tracker.track_query("What is Rust?", Some("Rust is a systems programming language."), 0.95).await?;
tracker.track_query("What is Rust?", Some("Rust focuses on safety and performance."), 0.92).await?;

// Get candidates ready for distillation
let candidates = tracker.get_candidates().await;
for candidate in candidates {
    if candidate.ready_for_distillation {
        println!("Pattern '{}' has {} Q&A pairs",
            candidate.pattern.normalized_text,
            candidate.qa_pairs.len());
    }
}

// Export training examples for LoRA fine-tuning
let examples = tracker.export_training_examples().await;
```

## WASM Usage

OxiRAG builds for `wasm32-unknown-unknown` and runs in a browser tab or a Web Worker — including
Layer 3, whose SMT verification goes through `OxiZ`. See
[ADR-0006](docs/adr/0006-wasm-portability-substrate.md) for what the target requires and how the
crate keeps meeting it.

Build:

```bash
wasm-pack build --target web --features wasm
```

Two things a consumer has to know before the first build.

**The `judge` feature needs a rustflag.** `getrandom 0.3` reaches the graph through
`ahash <- dashmap <- lasso <- oxiz-core` and refuses to compile for wasm32 without a backend named.
A rustflag cannot be expressed in a manifest, so put this in your own `.cargo/config.toml` — cargo
reads config from the invocation directory upwards, never from a dependency:

```toml
[target.wasm32-unknown-unknown]
rustflags = ['--cfg', 'getrandom_backend="wasm_js"']
```

### Japanese

Layers 2, 3 and 4 handle Japanese as of the current `Unreleased` entry. They did not before, and
the failure was silent rather than loud: a Japanese corpus indexed and retrieved correctly (Layer 1
works on character bigrams) and then produced `claims: 0`, `entities: 0`, `relationships: 0`,
because every heuristic above Layer 1 assumed sentences end in `.` and words are separated by
spaces.

`crate::text` now carries the script-aware primitives — sentence splitting for both punctuation
families, particle-boundary segmentation, topic/copula/negation handling, Han and Katakana noun
candidates — and the extractors use them. What this is NOT is morphological analysis: there is no
dictionary here, and `A は B です` is recognised by its particles, not parsed. For real
segmentation, use [MeCrab](https://github.com/cool-japan/mecrab).

**`MockEmbeddingProvider` does not retrieve, and `WasmRagEngine` is built on it.** It hashes the
*whole text* into one `u64`, so cosine similarity between any two distinct strings is noise —
measured at 384 dimensions, a near-duplicate pair scores 0.0284 while an unrelated pair scores
0.0762. It is deterministic and fine for tests; it is not a retrieval provider. Supply your own:
`EmbeddingProvider` is a public trait and `EchoLayer::new` accepts any implementation.

Use in JavaScript (the bundled engine, mock provider and all):

```javascript
import init, { WasmRagEngine } from './pkg/oxirag.js';

await init();

const engine = new WasmRagEngine(384);
await engine.index("The capital of France is Paris.", "geography");
const output = await engine.query("What is the capital of France?", 5);
```

For a worked example with a real embedding provider, all four layers wired, and the numbers
measured in a browser, see the COOLJAPAN Playground `/rag` demo
(`cool-japan/cooljapan-playground`, `crates/oxirag-wasm`).

### Building for wasm32 without the JavaScript bindings

The `wasm` feature means "expose `src/wasm.rs`", not "be buildable for wasm32" — the wasm
dependencies are unconditional target dependencies. A downstream crate writing its own
`#[wasm_bindgen]` boundary should leave the feature off, and will not inherit a second
`#[wasm_bindgen(start)]` or an export surface it does not use:

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features echo,judge,graphrag
```

## Configuration

### Pipeline Configuration

```rust
let config = PipelineConfig {
    fast_path_threshold: 0.95,      // Skip layers if confidence is high
    skip_verification_threshold: 0.9,
    enable_fast_path: true,
    max_retries: 3,
    parallel_execution: false,
    max_search_results: 5,
};
```

### Loading from JSON

```rust
let config = OxiRagConfig::from_json(r#"{
    "echo": { "dimension": 384 },
    "pipeline": { "enable_fast_path": true }
}"#)?;
```

## Testing

```bash
# Run all tests (1,500 tests)
cargo nextest run --all-features

# Run clippy (no warnings)
cargo clippy --all-features -- -D warnings

# Run rustdoc validation (no warnings)
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Run doc tests
cargo test --doc --all-features
```

## Benchmarks

```bash
cargo bench
```

## Project Statistics

| Metric | Value |
|--------|-------|
| Source Files | 91 Rust files |
| Lines of Code | 48,463 |
| Tests | 1,500 |
| Clippy Warnings | 0 |
| Rustdoc Warnings | 0 |

## Sponsorship

OxiRAG is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiRAG useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

## Roadmap

See [TODO.md](TODO.md) for detailed progress on each layer:

| Feature | Progress |
|---------|----------|
| Speculative RAG | 99% |
| Context-Aware Prefix Caching | 95% |
| On-the-fly Distillation | 85% |
| Hidden States Manipulation | 90% |

## COOLJAPAN Ecosystem

OxiRAG is part of the [COOLJAPAN](https://github.com/cool-japan) Pure Rust ecosystem:

- **OxiZ**: SMT solver for logic verification
- **SciRS2**: Scientific computing library
- **NumRS2**: Numerical computing primitives
- **OxiBLAS**: BLAS implementation in Pure Rust

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**COOLJAPAN OU (Team Kitasan)**
