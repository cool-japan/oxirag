#![cfg(feature = "python")]
//! `PyO3` wrapper types for the `OxiRAG` Python bindings.
//!
//! Each `Py*` struct is a thin newtype over the corresponding Rust type,
//! exposing only the properties needed by Python callers.

use pyo3::PyAny;
use pyo3::prelude::*;

use crate::types::{Document, DocumentId, Draft, PipelineOutput, Query, SearchResult};

// ────────────────────────────────────────────────────────────────────────────
// PyDocument
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper for [`Document`].
///
/// # Examples (Python)
///
/// ```python
/// doc = oxirag.Document("Hello, world!", title="Greeting")
/// print(doc.id, doc.content, doc.title)
/// ```
#[pyclass(name = "Document", from_py_object)]
#[derive(Clone)]
pub struct PyDocument {
    pub(crate) inner: Document,
}

#[pymethods]
impl PyDocument {
    /// Create a new document with optional title.
    #[new]
    #[pyo3(signature = (content, title=None))]
    #[must_use]
    pub fn new(content: String, title: Option<String>) -> Self {
        let mut doc = Document::new(content);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }
        Self { inner: doc }
    }

    /// The document's unique identifier (UUID string).
    #[getter]
    #[must_use]
    pub fn id(&self) -> String {
        self.inner.id.to_string()
    }

    /// The main text content of the document.
    #[getter]
    #[must_use]
    pub fn content(&self) -> &str {
        &self.inner.content
    }

    /// The document's optional title.
    #[getter]
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.inner.title.as_deref()
    }

    /// The document's optional source path or URL.
    #[getter]
    #[must_use]
    pub fn source(&self) -> Option<&str> {
        self.inner.source.as_deref()
    }

    /// Return a new document with the given metadata key-value pair added.
    #[must_use]
    pub fn with_metadata(&self, key: String, value: String) -> Self {
        Self {
            inner: self.inner.clone().with_metadata(key, value),
        }
    }

    /// Return a new document with the given source set.
    #[must_use]
    pub fn with_source(&self, source: String) -> Self {
        Self {
            inner: self.inner.clone().with_source(source),
        }
    }

    /// Return a new document with the given title set.
    #[must_use]
    pub fn with_title(&self, title: String) -> Self {
        Self {
            inner: self.inner.clone().with_title(title),
        }
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        let preview: String = self.inner.content.chars().take(40).collect();
        let ellipsis = if self.inner.content.len() > 40 {
            "..."
        } else {
            ""
        };
        format!(
            "Document(id={}, content={:?}{})",
            self.inner.id, preview, ellipsis
        )
    }

    /// Human-readable string, same as `__repr__`.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyQuery
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper for [`Query`].
///
/// # Examples (Python)
///
/// ```python
/// q = oxirag.Query("What is Rust?").with_top_k(5).with_min_score(0.1)
/// ```
#[pyclass(name = "Query", from_py_object)]
#[derive(Clone)]
pub struct PyQuery {
    pub(crate) inner: Query,
}

#[pymethods]
impl PyQuery {
    /// Create a new query from the given text.
    #[new]
    #[must_use]
    pub fn new(text: String) -> Self {
        Self {
            inner: Query::new(text),
        }
    }

    /// Return a copy of this query with `top_k` set.
    #[must_use]
    pub fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            inner: self.inner.clone().with_top_k(top_k),
        }
    }

    /// Return a copy of this query with `min_score` set.
    #[must_use]
    pub fn with_min_score(&self, score: f32) -> Self {
        Self {
            inner: self.inner.clone().with_min_score(score),
        }
    }

    /// The raw query text.
    #[getter]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.inner.text
    }

    /// Maximum number of results to retrieve.
    #[getter]
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.inner.top_k
    }

    /// Minimum similarity score threshold (`None` if not set).
    #[getter]
    #[must_use]
    pub fn min_score(&self) -> Option<f32> {
        self.inner.min_score
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!(
            "Query(text={:?}, top_k={}, min_score={:?})",
            self.inner.text, self.inner.top_k, self.inner.min_score
        )
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PySearchResult
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper for [`SearchResult`].
#[pyclass(name = "SearchResult", from_py_object)]
#[derive(Clone)]
pub struct PySearchResult {
    pub(crate) inner: SearchResult,
}

#[pymethods]
impl PySearchResult {
    /// Cosine similarity score in \[0, 1\].
    #[getter]
    #[must_use]
    pub fn score(&self) -> f32 {
        self.inner.score
    }

    /// Zero-based rank within the result set.
    #[getter]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    /// The matched document.
    #[getter]
    #[must_use]
    pub fn document(&self) -> PyDocument {
        PyDocument {
            inner: self.inner.document.clone(),
        }
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        let preview: String = self.inner.document.content.chars().take(30).collect();
        let ellipsis = if self.inner.document.content.len() > 30 {
            "..."
        } else {
            ""
        };
        format!(
            "SearchResult(rank={}, score={:.4}, content={:?}{})",
            self.inner.rank, self.inner.score, preview, ellipsis
        )
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyDraft
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper for [`Draft`].
#[pyclass(name = "Draft", from_py_object)]
#[derive(Clone)]
pub struct PyDraft {
    pub(crate) inner: Draft,
}

#[pymethods]
impl PyDraft {
    /// The generated draft answer text.
    #[getter]
    #[must_use]
    pub fn content(&self) -> &str {
        &self.inner.content
    }

    /// The original query string.
    #[getter]
    #[must_use]
    pub fn query(&self) -> &str {
        &self.inner.query
    }

    /// Draft confidence in \[0, 1\].
    #[getter]
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// List of source document IDs used to produce the draft.
    #[getter]
    #[must_use]
    pub fn sources(&self) -> Vec<String> {
        self.inner
            .sources
            .iter()
            .map(|id: &DocumentId| id.to_string())
            .collect()
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        let preview: String = self.inner.content.chars().take(40).collect();
        let ellipsis = if self.inner.content.len() > 40 {
            "..."
        } else {
            ""
        };
        format!(
            "Draft(confidence={:.3}, content={:?}{})",
            self.inner.confidence, preview, ellipsis
        )
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PyPipelineOutput
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible wrapper for [`PipelineOutput`].
///
/// Returned by `Pipeline.query()`.
#[pyclass(name = "PipelineOutput", from_py_object)]
#[derive(Clone)]
pub struct PyPipelineOutput {
    pub(crate) inner: PipelineOutput,
}

#[pymethods]
impl PyPipelineOutput {
    /// The final synthesised answer string.
    #[getter]
    #[must_use]
    pub fn final_answer(&self) -> &str {
        &self.inner.final_answer
    }

    /// Overall pipeline confidence in \[0, 1\].
    #[getter]
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    /// Total wall-clock execution time in milliseconds.
    #[getter]
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.inner.total_duration_ms
    }

    /// Names of the pipeline layers that were exercised.
    #[getter]
    #[must_use]
    pub fn layers_used(&self) -> Vec<String> {
        self.inner.layers_used.clone()
    }

    /// The list of [`SearchResult`]s from the Echo layer.
    ///
    /// # Panics
    ///
    /// Panics if `PySearchResult` cannot be converted into a Python object,
    /// which should never occur under normal circumstances.
    #[getter]
    #[must_use]
    pub fn search_results(&self, py: Python<'_>) -> Py<PyAny> {
        let list = pyo3::types::PyList::empty(py);
        for r in &self.inner.search_results {
            list.append(
                PySearchResult { inner: r.clone() }
                    .into_pyobject(py)
                    .expect("PySearchResult into_pyobject should not fail"),
            )
            .ok();
        }
        list.into()
    }

    /// The intermediate draft produced before verification.
    #[getter]
    #[must_use]
    pub fn draft(&self) -> PyDraft {
        PyDraft {
            inner: self.inner.draft.clone(),
        }
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        let preview: String = self.inner.final_answer.chars().take(50).collect();
        let ellipsis = if self.inner.final_answer.len() > 50 {
            "..."
        } else {
            ""
        };
        format!(
            "PipelineOutput(confidence={:.3}, layers={:?}, answer={:?}{})",
            self.inner.confidence, self.inner.layers_used, preview, ellipsis
        )
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
