# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-24

### Added
- Initial release of OxiRAG - A four-layer RAG engine
- **Layer 1 (Echo)**: Semantic search with vector embeddings and ANN indexing
- **Layer 2 (Speculator)**: Draft verification using small language models (Candle-based)
- **Layer 3 (Judge)**: Logic verification using SMT solvers (OxiZ integration)
- **Layer 4 (GraphRAG)**: Knowledge graph support with entity extraction and traversal
- WASM support for browser/edge deployment
- Native async runtime with Tokio
- SIMD-optimized similarity calculations
- Prefix caching with paging and invalidation
- Distillation support (teacher-student, progressive, feature-based)
- Query expansion and reranking
- Streaming results with progress reporting
- Circuit breaker and connection pooling
- Comprehensive benchmarking suite
