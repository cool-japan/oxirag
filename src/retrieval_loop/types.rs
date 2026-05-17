//! Core types for the FLARE (Forward-Looking Active `REtrieval`) module.
//!
//! These types model the full FLARE algorithm:
//! - Configuration via [`FlareConfig`]
//! - Token/sentence confidence via [`TokenConfidence`] and [`SentenceConfidence`]
//! - Retrieved context management via [`ContextDoc`] and [`ContextWindow`]
//! - Iteration tracing via [`IterationRecord`] and [`FlareOutput`]

use serde::{Deserialize, Serialize};

// ── FlareConfig ───────────────────────────────────────────────────────────────

/// Configuration for the [`crate::retrieval_loop::engine::FlareEngine`].
///
/// Controls the iterative generate-retrieve loop's behaviour: how many
/// iterations to allow, when to trigger retrieval, and how to manage the
/// growing context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareConfig {
    /// Maximum number of generate-retrieve iterations before forcibly stopping.
    ///
    /// Defaults to `5`.
    pub max_iterations: usize,

    /// Confidence threshold below which a sentence is considered "uncertain"
    /// and triggers retrieval.
    ///
    /// Defaults to `0.5`.
    pub confidence_threshold: f32,

    /// Minimum character length of a retrieval query span.  Spans shorter than
    /// this are skipped to avoid retrieving on trivial fragments.
    ///
    /// Defaults to `5`.
    pub min_query_length: usize,

    /// Maximum number of documents retained in the context window at any time.
    ///
    /// Defaults to `10`.
    pub max_context_docs: usize,

    /// Maximum total character budget for the serialised context window.
    /// Documents are trimmed (highest-score first) until this limit is met.
    ///
    /// Defaults to `4000`.
    pub context_window_chars: usize,

    /// Cosine-similarity threshold for de-duplicating retrieved documents.
    /// Two documents whose source IDs differ but whose content resembles one
    /// another at or above this threshold are merged (the higher-scored one
    /// wins).
    ///
    /// Defaults to `0.85`.
    pub dedup_similarity_threshold: f32,

    /// Whether to prepend the original query to the uncertain span before
    /// issuing the retrieval call.  When `true` the retriever sees
    /// `"{original_query} {uncertain_span}"`.
    ///
    /// Defaults to `true`.
    pub query_augment: bool,
}

impl Default for FlareConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            confidence_threshold: 0.5,
            min_query_length: 5,
            max_context_docs: 10,
            context_window_chars: 4000,
            dedup_similarity_threshold: 0.85,
            query_augment: true,
        }
    }
}

impl FlareConfig {
    /// Set the maximum number of generate-retrieve iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the confidence threshold that triggers retrieval.
    #[must_use]
    pub fn with_confidence_threshold(mut self, t: f32) -> Self {
        self.confidence_threshold = t;
        self
    }

    /// Set the minimum query span length.
    #[must_use]
    pub fn with_min_query_length(mut self, n: usize) -> Self {
        self.min_query_length = n;
        self
    }

    /// Set the maximum number of documents in the context window.
    #[must_use]
    pub fn with_max_context_docs(mut self, n: usize) -> Self {
        self.max_context_docs = n;
        self
    }

    /// Set the maximum context window character budget.
    #[must_use]
    pub fn with_context_window_chars(mut self, n: usize) -> Self {
        self.context_window_chars = n;
        self
    }

    /// Set the de-duplication cosine-similarity threshold.
    #[must_use]
    pub fn with_dedup_similarity_threshold(mut self, t: f32) -> Self {
        self.dedup_similarity_threshold = t;
        self
    }

    /// Enable or disable query augmentation.
    #[must_use]
    pub fn with_query_augment(mut self, enabled: bool) -> Self {
        self.query_augment = enabled;
        self
    }
}

// ── TokenConfidence ───────────────────────────────────────────────────────────

/// A single token together with its estimated pseudo-confidence.
///
/// `confidence` lies in `[0.0, 1.0]`, where `1.0` indicates the token is well
/// supported by the current context and `0.0` indicates it is unsupported.
#[derive(Debug, Clone)]
pub struct TokenConfidence {
    /// The token text (lower-cased, punctuation stripped).
    pub token: String,
    /// Estimated confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

// ── SentenceConfidence ────────────────────────────────────────────────────────

/// A sentence together with per-token confidence scores and the sentence-level
/// average.
#[derive(Debug, Clone)]
pub struct SentenceConfidence {
    /// The full sentence text as it appeared in the generated output.
    pub text: String,
    /// Per-token confidence scores.
    pub tokens: Vec<TokenConfidence>,
    /// Mean of all token confidences (`tokens.iter().map(|t| t.confidence).mean()`).
    pub avg_confidence: f32,
}

impl SentenceConfidence {
    /// Create a new `SentenceConfidence`, computing `avg_confidence`
    /// automatically from the provided token list.
    #[must_use]
    pub fn new(text: String, tokens: Vec<TokenConfidence>) -> Self {
        let avg_confidence = if tokens.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let denom = tokens.len() as f32;
            tokens.iter().map(|t| t.confidence).sum::<f32>() / denom
        };
        Self {
            text,
            tokens,
            avg_confidence,
        }
    }

    /// Returns `true` if this sentence's average confidence is below
    /// `threshold`, i.e. retrieval should be triggered.
    #[must_use]
    pub fn is_uncertain(&self, threshold: f32) -> bool {
        self.avg_confidence < threshold
    }

    /// Returns the sentence text — the span that should be used as (or
    /// augmented into) the retrieval query.
    #[must_use]
    pub fn uncertain_span(&self) -> String {
        self.text.clone()
    }
}

// ── ContextDoc ────────────────────────────────────────────────────────────────

/// A single retrieved document held inside the context window.
#[derive(Debug, Clone)]
pub struct ContextDoc {
    /// Full content of the document.
    pub content: String,
    /// Relevance score assigned by the retriever (higher is better).
    pub score: f32,
    /// Stable identifier for the source document (used for de-duplication).
    pub source_id: String,
}

impl ContextDoc {
    /// Construct a new `ContextDoc`.
    #[must_use]
    pub fn new(content: impl Into<String>, score: f32, source_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            score,
            source_id: source_id.into(),
        }
    }

    /// Render the document as a labelled context string suitable for inclusion
    /// in a generation prompt.
    ///
    /// Format:
    /// ```text
    /// [Document: <source_id> | score: <score:.3>]
    /// <content>
    /// ```
    #[must_use]
    pub fn to_context_string(&self) -> String {
        format!(
            "[Document: {} | score: {:.3}]\n{}",
            self.source_id, self.score, self.content
        )
    }
}

// ── ContextWindow ─────────────────────────────────────────────────────────────

/// A bounded, scored, de-duplicated collection of retrieved context documents.
///
/// Invariants maintained after every [`add_doc`](ContextWindow::add_doc) call:
/// - Documents are sorted by score descending.
/// - No two documents share the same `source_id`.
/// - The total character length of all `to_context_string()` values does not
///   exceed `max_chars`.
pub struct ContextWindow {
    /// Currently held documents (always sorted by score descending).
    pub docs: Vec<ContextDoc>,
    /// Character budget for the serialised window.
    pub max_chars: usize,
}

impl ContextWindow {
    /// Create an empty [`ContextWindow`] with the given character budget.
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self {
            docs: Vec::new(),
            max_chars,
        }
    }

    /// Add a document to the window, maintaining the sorted-dedup-trimmed
    /// invariants.
    ///
    /// Steps:
    /// 1. If a document with the same `source_id` already exists and its score
    ///    is at least as high, the new document is ignored.  If the new score is
    ///    higher, the old entry is replaced.
    /// 2. Documents are re-sorted by score descending.
    /// 3. Documents are trimmed from the tail until the total serialised length
    ///    is within `max_chars`.
    pub fn add_doc(&mut self, doc: ContextDoc) {
        // Step 1: de-duplicate by source_id.
        if let Some(existing) = self.docs.iter_mut().find(|d| d.source_id == doc.source_id) {
            if doc.score > existing.score {
                *existing = doc;
            }
            // else keep the existing higher-scored doc
        } else {
            self.docs.push(doc);
        }

        // Step 2: sort descending by score.
        self.docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Step 3: trim to max_chars budget.
        self.trim_to_budget();
    }

    /// Trim documents from the tail (lowest score) until the serialised length
    /// is within `max_chars`.
    fn trim_to_budget(&mut self) {
        loop {
            let total: usize = self.docs.iter().map(|d| d.to_context_string().len()).sum();
            if total <= self.max_chars || self.docs.is_empty() {
                break;
            }
            self.docs.pop();
        }
    }

    /// Serialise the entire window to a single string, separating each
    /// document with a blank line.
    #[must_use]
    pub fn as_context_string(&self) -> String {
        self.docs
            .iter()
            .map(ContextDoc::to_context_string)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Number of documents currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Returns `true` if the window contains no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}

impl std::fmt::Display for ContextWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_context_string())
    }
}

// ── IterationRecord ───────────────────────────────────────────────────────────

/// A trace record for one iteration of the FLARE loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Text generated by the generator in this iteration.
    pub generated_text: String,
    /// Whether a retrieval was triggered in this iteration.
    pub triggered_retrieval: bool,
    /// The query used for retrieval, if retrieval was triggered.
    pub retrieval_query: Option<String>,
    /// Number of documents retrieved (0 if no retrieval).
    pub docs_retrieved: usize,
    /// Average sentence confidence for the generated text in this iteration.
    pub avg_confidence: f32,
}

// ── FlareOutput ───────────────────────────────────────────────────────────────

/// The final output of a completed FLARE run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlareOutput {
    /// The final synthesised answer after all iterations.
    pub final_answer: String,
    /// Per-iteration trace records for debugging and observability.
    pub iterations: Vec<IterationRecord>,
    /// Total number of retrieval calls made across all iterations.
    pub total_retrievals: usize,
    /// Total number of context documents accumulated across all iterations.
    pub context_doc_count: usize,
}

impl FlareOutput {
    /// Fraction of iterations that triggered a retrieval call.
    ///
    /// Returns `0.0` when the run completed with zero iterations.
    #[must_use]
    pub fn retrieval_rate(&self) -> f32 {
        if self.iterations.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let numer = self.total_retrievals as f32;
        #[allow(clippy::cast_precision_loss)]
        let denom = self.iterations.len() as f32;
        numer / denom
    }
}

// ── FlareError ────────────────────────────────────────────────────────────────

/// Errors produced by the FLARE retrieval loop.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FlareError {
    /// The generator backend returned an error.
    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    /// The retriever backend returned an error.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),

    /// The loop ran for the configured maximum number of iterations without
    /// converging to a fully-confident answer.
    #[error("FLARE loop reached the maximum of {0} iterations without convergence")]
    MaxIterationsReached(usize),

    /// A retrieval query was empty or too short (below `min_query_length`).
    #[error("Retrieval query is empty or below minimum length")]
    EmptyQuery,
}
