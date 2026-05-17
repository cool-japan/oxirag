# ADR-0001: SpanObserver / MemoryObserver / OtelSpanObserver Pattern

**Status:** Accepted
**Date:** 2026-05-17
**Deciders:** KitaSan

## Context

OxiRAG's pipeline (`Pipeline<E,S,J>`) executes up to four layers per query (Echo,
Speculator, Judge, Graph). Operators need to observe timing, item counts, and error
status for each layer without baking a specific telemetry backend into the crate's
required dependencies.

Early prototypes used `tracing::Span` directly inside the pipeline. This produced
structured log events but forced all callers to pull in the `tracing-subscriber`
and `opentelemetry-*` crates even when they only wanted simple in-process metrics.
WASM deployments in particular can't use the OTLP/gRPC stack.

## Decision

Introduce a pluggable `SpanObserver` trait in `src/observability/mod.rs`:

```rust
pub trait SpanObserver: Send + Sync {
    fn on_layer_complete(&self, record: &LayerSpanRecord);
    fn on_pipeline_complete(&self, ctx: &PipelineSpanContext);
}
```

`PipelineSpanContext` holds an ordered `Vec<Arc<dyn SpanObserver>>`. Each query
creates one `PipelineSpanContext`; layers are wrapped with RAII `LayerSpan<'_>`
guards that call `on_layer_complete` on every registered observer when committed
(success / error / skip) or dropped (panic-safe error synthesis).

Three concrete implementations ship with the crate:

- `MemoryObserver` — collects all records in a `Mutex<Vec<LayerSpanRecord>>`.
  Zero overhead path for tests and the REST `/metrics` endpoint.
- `OtelSpanObserver` (feature `otel`) — exports to an OTLP/gRPC collector or
  to stdout via `opentelemetry-stdout`. Activated by registering it via
  `PipelineBuilder::with_observers`.

Observers are registered at pipeline construction time:

```rust
use std::sync::Arc;
use oxirag::observability::MemoryObserver;
use oxirag::pipeline::PipelineBuilder;

let observer = Arc::new(MemoryObserver::new());
let pipeline = PipelineBuilder::new()
    /* ... layers ... */
    .with_observer(observer.clone())
    .build();
```

## Rationale

- Zero mandatory external dependencies: the base crate only uses `tracing` (already
  required) for structured log events. No additional crate is needed to instrument
  the pipeline.
- The OTel exporter is a clean opt-in via the `otel` feature flag; it does not
  appear in the default feature set and does not affect WASM bundle size.
- Multiple sinks can coexist: register both a `MemoryObserver` for unit-test
  assertions and an `OtelSpanObserver` for production export. Each observer
  is called in registration order.
- The `PipelineSpanContext` UUID `execution_id` enables cross-observer correlation
  without coupling observers to each other.
- `LayerSpan<'_>` is panic-safe: the `Drop` impl synthesises an `Error("dropped
  without completion")` record, so the `PipelineSpanContext` always has a complete
  history even when a layer panics.

## Consequences

- Every pipeline query allocates one `PipelineSpanContext` and one `LayerSpan`
  per executed layer. Measured overhead is less than 2 µs per layer on a modern
  x86_64 core (dominated by `Instant::now()` syscall).
- Callers that register `OtelSpanObserver` take on the OTLP export latency.
  Spans are flushed asynchronously on a background Tokio task; the `on_layer_complete`
  callback itself is non-blocking.
- `dyn SpanObserver` requires `Send + Sync`, which prevents use of unsync types
  (e.g., `Rc<Cell<_>>`) as observers. This is a deliberate safety constraint.
- The `MemoryObserver` accumulates records indefinitely; callers must call
  `MemoryObserver::clear()` periodically to prevent unbounded memory growth in
  long-running servers.

## Alternatives Considered

### Single mandatory `tracing::Span`

Using `tracing::Span` directly inside each layer (as a `tracing::instrument`
attribute or manual span) would require callers to install a `tracing-subscriber`
to receive events. The OTel bridge (`tracing-opentelemetry`) would also be a
mandatory transitive dependency. Rejected: the resulting binary bloat is
unacceptable for WASM deployments, and the coupling prevents consumers from routing
telemetry to backends other than OTel (e.g., custom ClickHouse ingestion).

### No observability

Running the pipeline as a black box with only aggregate timings logged at the
pipeline level. Rejected: debugging slow queries requires per-layer duration data.
The v0.3.0 release specifically added observability in response to production
support requests.
