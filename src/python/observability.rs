#![cfg(feature = "python")]
//! `PyO3` wrappers for `OxiRAG` observability / span records.
//!
//! These types expose the [`LayerSpanRecord`] data collected by a
//! `MemoryObserver` to Python callers.

use pyo3::PyAny;
use pyo3::prelude::*;

use crate::observability::LayerSpanRecord;

// ─────────────────��──────────────────────────────────────────────────────────
// PyLayerSpanRecord
// ─────────────────────────────────���────────────────────────────────���─────────

/// Python-visible wrapper for a single [`LayerSpanRecord`].
///
/// # Examples (Python)
///
/// ```python
/// report = pipeline.span_report()
/// for record in report.records:
///     print(record.layer_name, record.duration_ms, record.status)
/// ```
#[pyclass(name = "LayerSpanRecord", from_py_object)]
#[derive(Clone)]
pub struct PyLayerSpanRecord {
    pub(crate) inner: LayerSpanRecord,
}

#[pymethods]
impl PyLayerSpanRecord {
    /// Human-readable name of the pipeline layer (e.g. `"echo"`, `"judge"`).
    #[getter]
    #[must_use]
    pub fn layer_name(&self) -> &str {
        &self.inner.layer_name
    }

    /// Wall-clock duration of the layer execution in milliseconds.
    #[getter]
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.inner.duration_ms
    }

    /// Number of items processed by the layer, or `None` if not recorded.
    #[getter]
    #[must_use]
    pub fn item_count(&self) -> Option<usize> {
        self.inner.item_count
    }

    /// Status string of the layer execution (e.g. `"Success"`, `"Error(...)"`, `"Skipped"`).
    #[getter]
    #[must_use]
    pub fn status(&self) -> String {
        format!("{:?}", self.inner.status)
    }

    /// Return all recorded key-value attributes as a Python dict.
    #[getter]
    #[must_use]
    pub fn attributes(&self, py: Python<'_>) -> Py<PyAny> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.inner.attributes {
            dict.set_item(k, v).ok();
        }
        dict.into()
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!(
            "LayerSpanRecord(layer={:?}, duration_ms={}, status={:?})",
            self.inner.layer_name,
            self.inner.duration_ms,
            self.inner.status.label()
        )
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ─────────────────────────────────��──────────────────────────────────────────
// PySpanReport
// ────────────────────────────────────────────────────────────────────────────

/// Python-visible span report: a snapshot of all completed layer spans from a
/// `MemoryObserver`. Returned by `PyPipeline::span_report`.
#[pyclass(name = "SpanReport")]
pub struct PySpanReport {
    pub(crate) records: Vec<LayerSpanRecord>,
}

#[pymethods]
impl PySpanReport {
    /// All collected [`LayerSpanRecord`]s as a Python list.
    ///
    /// # Panics
    ///
    /// Panics if `PyLayerSpanRecord` cannot be converted into a Python object,
    /// which should never occur under normal circumstances.
    #[getter]
    #[must_use]
    pub fn records(&self, py: Python<'_>) -> Py<PyAny> {
        let list = pyo3::types::PyList::empty(py);
        for r in &self.records {
            list.append(
                PyLayerSpanRecord { inner: r.clone() }
                    .into_pyobject(py)
                    .expect("PyLayerSpanRecord into_pyobject should not fail"),
            )
            .ok();
        }
        list.into()
    }

    /// Number of records in this report.
    #[must_use]
    pub fn __len__(&self) -> usize {
        self.records.len()
    }

    /// Developer-friendly string representation.
    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("SpanReport({} records)", self.records.len())
    }

    /// Human-readable string.
    #[must_use]
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
