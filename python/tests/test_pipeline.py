"""pytest tests for the oxirag Python module.

These tests are skipped automatically when the Rust extension has not been
built yet. To enable them, run:

    maturin develop --features python
    pytest python/tests/
"""

import asyncio

import pytest

try:
    import oxirag

    HAS_OXIRAG = True
except ImportError:
    HAS_OXIRAG = False

pytestmark = pytest.mark.skipif(
    not HAS_OXIRAG, reason="oxirag not built — run 'maturin develop --features python' first"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def pipeline():
    """Return a freshly built in-memory pipeline with a small embedding dimension."""
    return oxirag.PipelineBuilder().with_dimension(64).build()


@pytest.fixture()
def pipeline_no_fast_path():
    """Return a pipeline with fast-path disabled so all three layers always run."""
    return (
        oxirag.PipelineBuilder()
        .with_dimension(64)
        .with_fast_path(False)
        .build()
    )


# ── Synchronous unit tests ────────────────────────────────────────────────────


def test_pipeline_builds():
    p = oxirag.PipelineBuilder().with_dimension(64).build()
    assert p is not None


def test_pipeline_repr():
    p = oxirag.PipelineBuilder().with_dimension(64).build()
    r = repr(p)
    assert "Pipeline" in r


def test_builder_repr():
    b = oxirag.PipelineBuilder().with_dimension(128).with_max_results(5)
    r = repr(b)
    assert "128" in r
    assert "5" in r


def test_document_creation():
    doc = oxirag.Document("Test content")
    assert doc.content == "Test content"
    assert doc.id is not None
    assert len(doc.id) > 0
    assert doc.title is None


def test_document_with_title():
    doc = oxirag.Document("Test content", title="My Title")
    assert doc.title == "My Title"


def test_document_repr():
    doc = oxirag.Document("Hello world")
    r = repr(doc)
    assert "Document" in r


def test_document_with_metadata():
    doc = oxirag.Document("content").with_metadata("key", "value")
    assert doc.content == "content"


def test_query_creation():
    q = oxirag.Query("hello world")
    assert q.text == "hello world"
    assert q.top_k == 10  # default


def test_query_with_top_k():
    q = oxirag.Query("hello world").with_top_k(5)
    assert q.top_k == 5


def test_query_with_min_score():
    q = oxirag.Query("hello world").with_min_score(0.5)
    assert abs(q.min_score - 0.5) < 1e-6


def test_query_repr():
    q = oxirag.Query("hello world").with_top_k(3)
    r = repr(q)
    assert "hello world" in r
    assert "3" in r


# ── Async integration tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_single_returns_id(pipeline):
    doc = oxirag.Document("Rust is a systems language")
    doc_id = await pipeline.index(doc)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


@pytest.mark.asyncio
async def test_index_and_count(pipeline):
    doc = oxirag.Document("Rust is a systems language")
    await pipeline.index(doc)
    count = await pipeline.count()
    assert count == 1


@pytest.mark.asyncio
async def test_index_batch_returns_ids(pipeline):
    docs = [
        oxirag.Document("Rust fact one"),
        oxirag.Document("Rust fact two"),
        oxirag.Document("Rust fact three"),
    ]
    ids = await pipeline.index_batch(docs)
    assert len(ids) == 3
    assert all(isinstance(i, str) and len(i) > 0 for i in ids)


@pytest.mark.asyncio
async def test_count_after_batch(pipeline):
    docs = [oxirag.Document(f"Document {i}") for i in range(5)]
    await pipeline.index_batch(docs)
    count = await pipeline.count()
    assert count == 5


@pytest.mark.asyncio
async def test_query_returns_output(pipeline):
    await pipeline.index(oxirag.Document("Rust has zero-cost abstractions"))
    await pipeline.index(oxirag.Document("Rust prevents data races at compile time"))

    query = oxirag.Query("What is Rust?").with_top_k(5)
    output = await pipeline.query(query)

    assert isinstance(output.final_answer, str)
    assert len(output.final_answer) > 0
    assert 0.0 <= output.confidence <= 1.0
    assert len(output.search_results) > 0


@pytest.mark.asyncio
async def test_query_search_results_have_fields(pipeline):
    await pipeline.index(oxirag.Document("The ownership system in Rust prevents dangling pointers."))

    output = await pipeline.query(oxirag.Query("What prevents dangling pointers?"))
    assert len(output.search_results) > 0

    first = output.search_results[0]
    assert 0.0 <= first.score <= 1.0
    assert isinstance(first.rank, int)
    assert isinstance(first.document, oxirag.Document)
    assert len(first.document.content) > 0


@pytest.mark.asyncio
async def test_query_output_layers_used(pipeline):
    await pipeline.index(oxirag.Document("Cargo manages Rust dependencies."))
    output = await pipeline.query(oxirag.Query("What is Cargo?"))
    assert isinstance(output.layers_used, list)
    assert len(output.layers_used) > 0
    assert "Echo" in output.layers_used


@pytest.mark.asyncio
async def test_query_output_total_duration(pipeline):
    await pipeline.index(oxirag.Document("Rust compiles to native code."))
    output = await pipeline.query(oxirag.Query("How does Rust execute?"))
    assert output.total_duration_ms >= 0


@pytest.mark.asyncio
async def test_pipeline_draft_accessible(pipeline):
    await pipeline.index(oxirag.Document("Rust has a strong type system."))
    output = await pipeline.query(oxirag.Query("What kind of type system does Rust have?"))
    draft = output.draft
    assert isinstance(draft.content, str)
    assert 0.0 <= draft.confidence <= 1.0


@pytest.mark.asyncio
async def test_span_report_after_query(pipeline_no_fast_path):
    await pipeline_no_fast_path.index(oxirag.Document("Memory safety without GC."))
    await pipeline_no_fast_path.query(oxirag.Query("What is memory safety?"))

    report = pipeline_no_fast_path.span_report()
    assert len(report) >= 1

    for record in report.records:
        assert isinstance(record.layer_name, str)
        assert record.duration_ms >= 0
        assert isinstance(record.status, str)


@pytest.mark.asyncio
async def test_query_empty_store_returns_output(pipeline):
    """Pipeline should handle querying an empty store gracefully."""
    output = await pipeline.query(oxirag.Query("anything"))
    assert isinstance(output.final_answer, str)
    assert 0.0 <= output.confidence <= 1.0
