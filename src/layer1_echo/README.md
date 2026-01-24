# Layer 1: Echo

Semantic search layer using vector embeddings.

## Overview

The Echo layer provides:
- Text-to-vector embedding conversion
- Vector storage and indexing
- Similarity-based document retrieval

## Components

### Traits

- **`EmbeddingProvider`**: Convert text to vector embeddings
- **`VectorStore`**: Store and search document embeddings
- **`Echo`**: Combined interface for indexing and search

### Implementations

- **`MockEmbeddingProvider`**: Deterministic hash-based embeddings for testing
- **`CandleEmbeddingProvider`**: BERT-based embeddings using Candle (requires `speculator` feature)
- **`InMemoryVectorStore`**: In-memory vector storage with configurable capacity

### Similarity Metrics

- **Cosine**: Angle-based similarity, range [-1, 1]
- **Euclidean**: Distance-based, converted to similarity
- **Dot Product**: Requires normalized vectors

## Usage

```rust
use oxirag::layer1_echo::*;

// Create embedding provider
let provider = MockEmbeddingProvider::new(384);

// Create vector store
let store = InMemoryVectorStore::new(384)
    .with_metric(SimilarityMetric::Cosine);

// Create Echo layer
let mut echo = EchoLayer::new(provider, store);

// Index documents
echo.index(Document::new("Hello world")).await?;

// Search
let results = echo.search("hello", 5, None).await?;
```

## Configuration

```rust
// Custom metric
let store = InMemoryVectorStore::new(dim)
    .with_metric(SimilarityMetric::DotProduct)
    .with_max_capacity(10000);
```

## File Structure

```
layer1_echo/
├── mod.rs           # EchoLayer implementation
├── traits.rs        # Core traits
├── similarity.rs    # Similarity computation
├── embedding/
│   ├── mod.rs
│   └── candle.rs    # Embedding providers
└── storage/
    ├── mod.rs
    └── memory.rs    # In-memory vector store
```
