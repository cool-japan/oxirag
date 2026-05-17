# OxiRAG Python Bindings

OxiRAG ships Python bindings built with [PyO3](https://pyo3.rs/) 0.28 and
[maturin](https://www.maturin.rs/).

## Setup

### Install maturin

```bash
pip install maturin
```

### Development build (editable install)

```bash
cd /path/to/oxirag
maturin develop --features python
```

This compiles the Rust extension in debug mode and installs it into the active
Python environment so `import oxirag` works immediately.

### Production wheel

```bash
maturin build --release --features python
pip install target/wheels/oxirag-*.whl
```

## Quickstart

```python
import asyncio
import oxirag


async def main():
    pipeline = oxirag.PipelineBuilder().with_dimension(384).build()

    # Index documents
    await pipeline.index(oxirag.Document("Paris is the capital of France."))
    await pipeline.index(oxirag.Document("The Eiffel Tower is in Paris."))

    # Query
    query = oxirag.Query("What is in Paris?").with_top_k(5)
    result = await pipeline.query(query)

    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.2f}")
    for r in result.search_results:
        print(f"  [{r.score:.3f}] {r.document.content}")


asyncio.run(main())
```

## Observability

```python
import asyncio
import oxirag


async def main():
    pipeline = oxirag.PipelineBuilder().with_dimension(64).build()
    await pipeline.index(oxirag.Document("Rust prevents data races."))
    await pipeline.query(oxirag.Query("What does Rust prevent?"))

    report = pipeline.span_report()
    for record in report.records:
        print(f"{record.layer_name}: {record.duration_ms} ms [{record.status}]")


asyncio.run(main())
```

## Python API reference

| Class / method | Description |
|---|---|
| `Document(content, title=None)` | Create a document for indexing. |
| `Query(text)` | Create a search query. |
| `.with_top_k(k)` | Limit the number of results. |
| `.with_min_score(s)` | Minimum similarity threshold. |
| `PipelineBuilder()` | Fluent pipeline constructor. |
| `.with_dimension(d)` | Set embedding dimension (default 384). |
| `.with_max_results(k)` | Maximum Echo-layer results (default 10). |
| `.build()` | Construct the `Pipeline` instance. |
| `await pipeline.index(doc)` | Index a single document; returns its ID. |
| `await pipeline.index_batch(docs)` | Index many documents; returns list of IDs. |
| `await pipeline.query(query)` | Run the full three-layer pipeline. |
| `await pipeline.count()` | Return the number of indexed documents. |
| `pipeline.span_report()` | Return a `SpanReport` from the `MemoryObserver`. |
| `SpanReport.records` | List of `LayerSpanRecord` objects. |
| `LayerSpanRecord.layer_name` | Layer identifier string. |
| `LayerSpanRecord.duration_ms` | Execution wall time in milliseconds. |
| `LayerSpanRecord.status` | Status string (e.g. `"Success"`). |
