# ADR-0004: Monomorphized Python Facade

**Status:** Accepted
**Date:** 2026-05-17
**Deciders:** KitaSan

## Context

`Pipeline<E, S, J>` carries three generic parameters:

- `E: Echo` — the semantic search layer (Layer 1).
- `S: Speculator` — the draft verification layer (Layer 2).
- `J: Judge` — the logic verification layer (Layer 3).

PyO3 cannot expose generic Rust structs to Python: every type exposed via
`#[pyclass]` must be a concrete monomorphic type. Creating a Python-friendly
`Pipeline` class therefore requires choosing concrete type arguments and
compiling them into the extension module's binary.

A secondary constraint: the optional Candle-based providers (`CandleEmbeddingProvider`,
`CandleSlmSpeculator`) each require downloading multi-gigabyte HuggingFace model
checkpoints at first use. Forcing Python wheel users to manage a 1.5–3 GB model
cache just to `import oxirag` would be a severe ergonomic barrier.

## Decision

Expose a single `DefaultPipeline` type alias in `src/python/mod.rs`:

```rust
use oxirag::layer1_echo::{EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use oxirag::layer2_speculator::RuleBasedSpeculator;
use oxirag::layer3_judge::{AdvancedClaimExtractor, JudgeImpl, MockSmtVerifier, JudgeConfig};

pub type DefaultPipeline = Pipeline<
    EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
    RuleBasedSpeculator,
    JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
>;
```

`MockEmbeddingProvider` generates deterministic hash-based embeddings without any
model download. `RuleBasedSpeculator` verifies drafts with rule-based heuristics
at sub-microsecond latency. `MockSmtVerifier` applies pattern-based claim checking
without invoking OxiZ.

Python API:

```python
import oxirag

pipeline = oxirag.DefaultPipeline()
pipeline.index("Rust is a systems programming language.")
results = pipeline.query("What is Rust?")
print(results[0].content)
```

The `python` feature (`features = ["dep:pyo3", "dep:pyo3-async-runtimes", "native", "echo"]`)
enables this facade. It deliberately does NOT include `speculator` (Candle) or
`judge` (OxiZ) in its feature set, so the resulting wheel has no transitive
model-download requirement.

## Rationale

- Minimal friction for Python users: `pip install oxirag` and the wheel works
  immediately with no environment setup beyond the package itself.
- Ships fast: the monomorphization approach required approximately two days of
  implementation versus weeks for a full `Box<dyn>` trait-object solution.
- Covers 90% of Python use cases: most Python users want semantic search
  (`MockEmbeddingProvider` + `InMemoryVectorStore`) with lightweight verification.
  The rule-based and mock components are production-quality for well-structured
  corpora where SLM verification is unnecessary.
- The Python wheel binary size stays small (< 5 MB on x86_64 Linux) because
  Candle's BLAS/LAPACK dependencies and model checkpoint infrastructure are
  excluded.

## Consequences

- Python users cannot swap the embedding provider to `CandleEmbeddingProvider` or
  the verifier to `OxizVerifier` at runtime without writing Rust code and rebuilding
  the extension module. This is a known limitation documented in the Python README
  (`examples/python/README.md`).
- If a Python user needs real embedding quality, the recommended path is to expose
  their own Candle-backed pipeline as a new PyO3 class in a downstream crate that
  depends on `oxirag` with `features = ["speculator", "judge"]`.
- A v0.6.0 roadmap item targets a feature-gated `CandleDefaultPipeline` class
  that enables Candle providers when the `speculator` feature is active, giving
  Python users a migration path without breaking the zero-download default.

## Alternatives Considered

### `Box<dyn RagPipeline>` trait object

Define a `trait RagPipeline` with `async fn query(&self, ...)` and expose a
`PyO3`-wrapped `Box<dyn RagPipeline>`. Rejected: virtual dispatch adds a minimum
1–2 ns overhead per call, and lifetime bounds for async trait objects in PyO3
are extremely complex to express correctly across Python's GIL boundary. The
`#[async_trait]` expansion produces `Pin<Box<dyn Future>>` return types that
interact poorly with PyO3's synchronous `Python<'_>` token.

### Feature-gated Candle facade (planned for v0.6.0)

Expose two `#[pyclass]` types: `DefaultPipeline` (mock, always available) and
`CandleDefaultPipeline` (real models, requires `speculator` feature at build time).
This approach is architecturally sound and is planned for v0.6.0. It was deferred
from v0.5.0 to keep the Python feature PR minimal and reviewable.
