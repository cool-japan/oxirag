//! Retriever abstractions for the FLARE retrieval loop.
//!
//! The [`FlareRetriever`] trait decouples the FLARE engine from any concrete
//! vector-search backend.  Two built-in implementations are provided:
//!
//! - [`MockFlareRetriever`]: returns a fixed, pre-configured set of documents.
//! - [`QueryAugmentedRetriever`]: wraps any retriever and prepends the
//!   original query to the uncertain span before delegating.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use super::types::{ContextDoc, FlareError};

// ── FlareRetriever trait ──────────────────────────────────────────────────────

/// Async trait for retrieval backends used by the FLARE engine.
///
/// Implementors must be `Send + Sync`.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait FlareRetriever: Send + Sync {
    /// Retrieve the top-`top_k` documents most relevant to `query`.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::RetrievalFailed`] if the backend cannot service
    /// the request.
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ContextDoc>, FlareError>;
}

// ── MockFlareRetriever ────────────────────────────────────────────────────────

/// A thread-safe in-memory retriever backed by a fixed corpus of
/// [`ContextDoc`]s.
///
/// Documents are pre-sorted by score descending; each call to
/// [`retrieve`](FlareRetriever::retrieve) returns up to `top_k` of them.
///
/// The internal corpus is held in an `Arc<RwLock<…>>` so the retriever can be
/// cheaply cloned and shared across tasks.
#[derive(Debug, Clone)]
pub struct MockFlareRetriever {
    docs: Arc<RwLock<Vec<ContextDoc>>>,
}

impl MockFlareRetriever {
    /// Create a new `MockFlareRetriever` from a list of [`ContextDoc`]s.
    ///
    /// The list is sorted by score descending on construction so that
    /// `retrieve` can do a simple prefix take.
    #[must_use]
    pub fn new(mut docs: Vec<ContextDoc>) -> Self {
        docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            docs: Arc::new(RwLock::new(docs)),
        }
    }

    /// Create a `MockFlareRetriever` with no documents.
    #[must_use]
    pub fn new_empty() -> Self {
        Self::new(vec![])
    }

    /// Returns the number of documents in the corpus.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned (should never happen in
    /// normal operation — only possible if another thread panicked while
    /// holding the write lock).
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs
            .read()
            .expect("MockFlareRetriever RwLock poisoned")
            .len()
    }

    /// Returns `true` if the corpus is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl FlareRetriever for MockFlareRetriever {
    async fn retrieve(&self, _query: &str, top_k: usize) -> Result<Vec<ContextDoc>, FlareError> {
        let guard = self
            .docs
            .read()
            .map_err(|e| FlareError::RetrievalFailed(e.to_string()))?;
        // Return up to top_k docs; the corpus is already sorted by score desc.
        let results = guard.iter().take(top_k).cloned().collect();
        Ok(results)
    }
}

// ── QueryAugmentedRetriever ───────────────────────────────────────────────────

/// A retriever adapter that prepends the `original_query` to every span query
/// before delegating to the inner retriever.
///
/// This implements the FLARE "query augmentation" strategy: instead of
/// retrieving on the bare uncertain span, retrieve on
/// `"{original_query} {span}"` to maintain topical context.
///
/// # Type parameters
///
/// - `R`: any type that implements [`FlareRetriever`].
pub struct QueryAugmentedRetriever<R: FlareRetriever> {
    inner: R,
    original_query: String,
}

impl<R: FlareRetriever> QueryAugmentedRetriever<R> {
    /// Wrap `inner` with query augmentation using `original_query`.
    #[must_use]
    pub fn new(inner: R, original_query: impl Into<String>) -> Self {
        Self {
            inner,
            original_query: original_query.into(),
        }
    }
}

impl<R: FlareRetriever + std::fmt::Debug> std::fmt::Debug for QueryAugmentedRetriever<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryAugmentedRetriever")
            .field("inner", &self.inner)
            .field("original_query", &self.original_query)
            .finish()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl<R: FlareRetriever> FlareRetriever for QueryAugmentedRetriever<R> {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<ContextDoc>, FlareError> {
        let augmented = format!("{} {}", self.original_query.trim(), query.trim());
        self.inner.retrieve(&augmented, top_k).await
    }
}
