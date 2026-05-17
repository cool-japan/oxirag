#![cfg(feature = "python")]
//! `PyO3` wrapper for the `OxiRAG` pipeline.
//!
//! Exposes a concrete, monomorphised pipeline that uses the mock/in-memory
//! backends so that Python callers do not need to care about generics.

use std::sync::Arc;

use pyo3::PyAny;
use pyo3::prelude::*;

use crate::layer1_echo::{Echo, EchoLayer, InMemoryVectorStore, MockEmbeddingProvider};
use crate::layer2_speculator::RuleBasedSpeculator;
use crate::layer3_judge::{AdvancedClaimExtractor, JudgeImpl, MockSmtVerifier};
use crate::observability::MemoryObserver;
use crate::pipeline::{Pipeline, RagPipeline};
use crate::python::observability::PySpanReport;
use crate::python::types::{PyDocument, PyPipelineOutput, PyQuery};
use crate::types::Document;

// ────────────────────────────────────────────────────────────────────────────
// Monomorphised pipeline type alias
// ────────────────────────────────────────────────────────────────────────────

/// Concrete pipeline type used for the Python binding.
///
/// Uses mock providers so the wheel can be installed without heavy ML
/// dependencies. Production callers that need Candle / redb backends can
/// build the pipeline in Rust and wrap it separately.
pub(crate) type DefaultPipeline = Pipeline<
    EchoLayer<MockEmbeddingProvider, InMemoryVectorStore>,
    RuleBasedSpeculator,
    JudgeImpl<AdvancedClaimExtractor, MockSmtVerifier>,
>;

// ────────────────────────────────────────────────────────────────────────────
// PyPipeline
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible pipeline handle.
///
/// All async methods bridge to the Tokio runtime via
/// `pyo3_async_runtimes::tokio::future_into_py`, making them awaitable in Python as native coroutines.
///
/// # Examples (Python)
///
/// ```python
/// import asyncio, oxirag
///
/// async def main():
///     p = oxirag.PipelineBuilder().with_dimension(128).build()
///     await p.index(oxirag.Document("Rust is memory-safe."))
///     out = await p.query(oxirag.Query("What is Rust?"))
///     print(out.final_answer)
///
/// asyncio.run(main())
/// ```
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    /// Shared, async-mutex-guarded pipeline so multiple coroutines can co-exist.
    pub(crate) inner: Arc<tokio::sync::Mutex<DefaultPipeline>>,
    /// Observer that accumulates span records for later introspection.
    pub(crate) observer: Arc<MemoryObserver>,
}

#[pymethods]
impl PyPipeline {
    // ── Indexing ─────────────────────────────────────────────────────────────

    /// Index a single document and return its document ID string.
    ///
    /// The returned value is a coroutine that must be `await`ed.
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if the embedding or storage step fails.
    pub fn index<'py>(&self, py: Python<'py>, doc: PyDocument) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.inner);
        let doc_inner: Document = doc.inner;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = pipeline.lock().await;
            let doc_id = guard
                .echo_mut()
                .index(doc_inner)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(doc_id.to_string())
        })
    }

    /// Index a batch of documents and return a list of document ID strings.
    ///
    /// The returned value is a coroutine that must be `await`ed.
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if any embedding or storage step fails.
    pub fn index_batch<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<PyDocument>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.inner);
        let docs_inner: Vec<Document> = docs.into_iter().map(|d| d.inner).collect();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = pipeline.lock().await;
            let ids = guard
                .echo_mut()
                .index_batch(docs_inner)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let id_strings: Vec<String> = ids.iter().map(ToString::to_string).collect();
            Ok(id_strings)
        })
    }

    // ── Querying ──────────────────────────────────────────────────────────────

    /// Run a query through the full pipeline and return a [`PyPipelineOutput`].
    ///
    /// The returned value is a coroutine that must be `await`ed.
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if any pipeline layer fails.
    pub fn query<'py>(&self, py: Python<'py>, query: PyQuery) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.inner);
        let query_inner = query.inner;

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = pipeline.lock().await;
            let output = guard
                .process(query_inner)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyPipelineOutput { inner: output })
        })
    }

    /// Process a batch of queries and return a list of [`PyPipelineOutput`]s.
    ///
    /// The returned value is a coroutine that must be `await`ed.
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if any query in the batch fails.
    pub fn query_batch<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<PyQuery>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.inner);
        let queries_inner: Vec<crate::types::Query> =
            queries.into_iter().map(|q| q.inner).collect();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = pipeline.lock().await;
            let results = guard.process_batch(queries_inner).await;

            let mut outputs: Vec<PyPipelineOutput> = Vec::with_capacity(results.len());
            for result in results {
                let output =
                    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                outputs.push(PyPipelineOutput { inner: output });
            }
            Ok(outputs)
        })
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Return the number of documents currently indexed.
    ///
    /// The returned value is a coroutine that must be `await`ed.
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if the pipeline lock cannot be acquired.
    pub fn count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = pipeline.lock().await;
            let n = guard.echo().count().await;
            Ok(n)
        })
    }

    /// Return a [`PySpanReport`] snapshot from the embedded `MemoryObserver`.
    ///
    /// This is a synchronous call (no coroutine needed).
    ///
    /// # Errors
    ///
    /// Returns a Python `RuntimeError` if the span report cannot be converted.
    pub fn span_report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records = self.observer.records();
        Ok(PySpanReport { records }
            .into_pyobject(py)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .into())
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        "Pipeline(type=mock, backend=in-memory)".to_string()
    }
}
