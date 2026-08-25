//! Generator abstractions for the FLARE retrieval loop.
//!
//! The [`FlareGenerator`] trait decouples the FLARE engine from any concrete
//! language model backend.  Two built-in implementations are provided for
//! testing and template-based usage:
//!
//! - [`MockFlareGenerator`]: cycles through a pre-configured list of answers.
//! - [`TemplateGenerator`]: fills `{query}` / `{context}` placeholders in a
//!   template string.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::confidence::ConfidenceEstimator;
use super::types::FlareError;

// ── FlareGenerator trait ──────────────────────────────────────────────────────

/// Async trait for text generation backends used by the FLARE engine.
///
/// Implementors must be `Send + Sync` so they can be shared across the
/// async runtime without additional wrapping.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait FlareGenerator: Send + Sync {
    /// Generate a complete response to `query`, using `context` as the
    /// retrieval-augmented background.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::GenerationFailed`] if the backend cannot produce
    /// a response.
    async fn generate(&self, query: &str, context: &str) -> Result<String, FlareError>;

    /// Generate a *partial* response (up to `max_sentences` sentences) to
    /// `query`.
    ///
    /// This is used for the incremental draft step in the FLARE loop where
    /// we need only a few sentences to evaluate confidence before deciding
    /// whether to retrieve.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::GenerationFailed`] if the backend cannot produce
    /// a response.
    async fn generate_partial(
        &self,
        query: &str,
        context: &str,
        max_sentences: usize,
    ) -> Result<String, FlareError>;
}

// ── MockFlareGenerator ────────────────────────────────────────────────────────

/// A deterministic generator that cycles through a pre-configured list of
/// answers.
///
/// Useful for unit-testing the FLARE engine without a real LLM backend.
///
/// # Thread safety
///
/// The internal call counter uses `AtomicUsize`, so `MockFlareGenerator` is
/// safe to share across threads and concurrent `await` points.
#[derive(Debug)]
pub struct MockFlareGenerator {
    base_answers: Vec<String>,
    call_counter: Arc<AtomicUsize>,
    /// Optional decay factor — kept for API compatibility; not used in the
    /// actual generation logic (a real implementation would use it to lower
    /// confidence on repeated questions).
    #[allow(dead_code)]
    confidence_decay: f32,
}

impl MockFlareGenerator {
    /// Create a new `MockFlareGenerator` that cycles through `base_answers`.
    ///
    /// # Panics
    ///
    /// Panics if `base_answers` is empty (there must be at least one answer to
    /// cycle through).
    #[must_use]
    pub fn new(base_answers: Vec<String>) -> Self {
        assert!(!base_answers.is_empty(), "base_answers must not be empty");
        Self {
            base_answers,
            call_counter: Arc::new(AtomicUsize::new(0)),
            confidence_decay: 0.0,
        }
    }

    /// Create a `MockFlareGenerator` that always returns the same `answer`.
    #[must_use]
    pub fn new_single(answer: String) -> Self {
        Self::new(vec![answer])
    }

    /// Retrieve the next answer from the cycle, incrementing the internal
    /// counter atomically.
    fn next_answer(&self) -> &str {
        let idx = self.call_counter.fetch_add(1, Ordering::Relaxed);
        &self.base_answers[idx % self.base_answers.len()]
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl FlareGenerator for MockFlareGenerator {
    async fn generate(&self, _query: &str, _context: &str) -> Result<String, FlareError> {
        Ok(self.next_answer().to_string())
    }

    async fn generate_partial(
        &self,
        _query: &str,
        _context: &str,
        max_sentences: usize,
    ) -> Result<String, FlareError> {
        let full = self.next_answer().to_string();
        // Truncate to at most `max_sentences` sentences.
        let sentences = ConfidenceEstimator::split_into_sentences(&full);
        let limited: Vec<String> = sentences.into_iter().take(max_sentences).collect();
        if limited.is_empty() {
            Ok(full)
        } else {
            Ok(limited.join(" "))
        }
    }
}

// ── TemplateGenerator ─────────────────────────────────────────────────────────

/// A generator that fills `{query}` and `{context}` placeholders in a fixed
/// template string.
///
/// Useful for inspecting what the FLARE engine passes to the generator, and
/// for lightweight integration tests where a fixed-format prompt is
/// sufficient.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "flare")]
/// # {
/// # use oxirag::retrieval_loop::generator::{FlareGenerator, TemplateGenerator};
/// # #[tokio::main]
/// # async fn main() {
/// let generator = TemplateGenerator::new("Q: {query}\nCtx: {context}");
/// let out = generator.generate("What is Rust?", "Rust is a language.").await.unwrap();
/// assert!(out.contains("What is Rust?"));
/// assert!(out.contains("Rust is a language."));
/// # }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TemplateGenerator {
    template: String,
}

impl TemplateGenerator {
    /// Create a new `TemplateGenerator` with the given `template`.
    ///
    /// The template may contain the placeholders `{query}` and `{context}`;
    /// both will be substituted on every call to [`generate`](FlareGenerator::generate).
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    fn fill(&self, query: &str, context: &str) -> String {
        self.template
            .replace("{query}", query)
            .replace("{context}", context)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl FlareGenerator for TemplateGenerator {
    async fn generate(&self, query: &str, context: &str) -> Result<String, FlareError> {
        Ok(self.fill(query, context))
    }

    async fn generate_partial(
        &self,
        query: &str,
        context: &str,
        max_sentences: usize,
    ) -> Result<String, FlareError> {
        let full = self.fill(query, context);
        let sentences = ConfidenceEstimator::split_into_sentences(&full);
        let limited: Vec<String> = sentences.into_iter().take(max_sentences).collect();
        if limited.is_empty() {
            Ok(full)
        } else {
            Ok(limited.join(" "))
        }
    }
}
