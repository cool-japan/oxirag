#![cfg(feature = "python")]
//! Python bindings for `OxiRAG` via `PyO3` 0.28.
//!
//! Build with:
//! ```bash
//! maturin develop --features python
//! ```
//!
//! Install as a wheel with:
//! ```bash
//! maturin build --features python
//! pip install target/wheels/oxirag-*.whl
//! ```
//!
//! # Quick start (Python)
//!
//! ```python
//! import asyncio
//! import oxirag
//!
//! async def main():
//!     p = oxirag.PipelineBuilder().with_dimension(128).build()
//!     await p.index(oxirag.Document("Rust is a systems language."))
//!     out = await p.query(oxirag.Query("What is Rust?"))
//!     print(out.final_answer)
//!
//! asyncio.run(main())
//! ```

pub mod builder;
pub mod observability;
pub mod pipeline;
pub mod types;

use pyo3::prelude::*;

/// Register all Python classes and free functions in the `oxirag` extension module.
///
/// # Errors
///
/// Returns a `PyResult` error if any class or function registration fails,
/// which should never occur under normal circumstances.
#[pymodule]
pub fn oxirag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Core types ────────────────────────────────────────────────────────────
    m.add_class::<types::PyDocument>()?;
    m.add_class::<types::PyQuery>()?;
    m.add_class::<types::PySearchResult>()?;
    m.add_class::<types::PyPipelineOutput>()?;
    m.add_class::<types::PyDraft>()?;

    // ── Pipeline ──────────────────────────────────────────────────────────────
    m.add_class::<pipeline::PyPipeline>()?;
    m.add_class::<builder::PyPipelineBuilder>()?;

    // ── Observability ─────────────────────────────────────────────────────────
    m.add_class::<observability::PySpanReport>()?;
    m.add_class::<observability::PyLayerSpanRecord>()?;

    Ok(())
}
