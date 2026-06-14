//! Types and traits for the `dragin` module.
//!
//! DRAGIN (Su et al. 2024) — *Dynamic Retrieval Augmented Generation based on
//! the real-time Information Need of Large Language Models* — augments
//! generation with two cooperating mechanisms:
//!
//! - **RIND** (Real-time Information Need Detection): retrieval is triggered the
//!   moment the model is about to emit a token it is *uncertain* about, provided
//!   that token actually carries information (a **content** token, not a
//!   stopword). Low confidence on a stopword such as "the" is not an information
//!   need; low confidence on a content token signals the model is missing
//!   knowledge.
//! - **QFS** (Query Formulation by Self-attention): rather than reusing the
//!   original question, DRAGIN forms the retrieval query from the **salient
//!   content tokens** of the recently generated context. Here the self-attention
//!   signal is approximated by keeping content tokens and dropping stopwords
//!   over the most recent window of generated text.
//!
//! This contrasts with FLARE in `retrieval_loop`, which triggers on the
//! confidence of the *next sentence* as a whole and reuses the look-ahead draft
//! as the query. DRAGIN operates at **token** granularity for triggering and
//! derives its query from attention-salient tokens.
//!
//! The generator and retriever are supplied by the caller through the
//! [`UncertaintyGenerator`] and [`Retriever`] traits. Deterministic
//! [`MockUncertaintyGenerator`] and [`MockRetriever`] implementations are
//! provided for testing.

use thiserror::Error;

// ── TokenInfo ───────────────────────────────────────────────────────────────

/// A single generated token paired with the model's confidence in it.
///
/// `confidence` is expected to lie in `[0.0, 1.0]`, where higher means the
/// model is more certain. RIND treats a token as uncertain when its confidence
/// falls below the configured threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenInfo {
    /// The surface text of the generated token.
    pub token: String,
    /// The model's confidence in `token`, nominally in `[0.0, 1.0]`.
    pub confidence: f32,
}

impl TokenInfo {
    /// Create a new [`TokenInfo`] from a token and its confidence.
    #[must_use]
    pub fn new(token: impl Into<String>, confidence: f32) -> Self {
        Self {
            token: token.into(),
            confidence,
        }
    }
}

// ── UncertaintyGenerator ──────────────────────────────────────────────────────

/// A language model that generates the next segment of tokens together with a
/// per-token confidence.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete generator (e.g. wrapping an LLM that exposes token
/// log-probabilities); [`MockUncertaintyGenerator`] is provided for tests.
pub trait UncertaintyGenerator {
    /// Generate the next segment of tokens (with per-token confidence) given
    /// `context` (the original query plus everything generated and retrieved so
    /// far).
    fn generate(&self, context: &str) -> Vec<TokenInfo>;
}

// ── Retriever ─────────────────────────────────────────────────────────────────

/// Retrieves passages for a query formed by QFS.
///
/// Implementations may wrap a vector store or any other backend. They are
/// **pure sync**. [`MockRetriever`] is provided for tests.
pub trait Retriever {
    /// Retrieve passages relevant to `query`.
    fn retrieve(&self, query: &str) -> Vec<String>;
}

// ── MockUncertaintyGenerator ──────────────────────────────────────────────────

/// Deterministic [`UncertaintyGenerator`] for tests.
///
/// Holds a script of pre-built segments. Each call to
/// [`UncertaintyGenerator::generate`] consumes and returns the next segment in
/// order; once the script is exhausted, an empty segment is returned (which the
/// engine treats as "no further generation"). The `context` argument is
/// ignored, making runs fully deterministic.
///
/// Interior mutability via [`std::cell::RefCell`] lets the mock advance its
/// cursor through a shared `&self`, matching the [`UncertaintyGenerator`]
/// signature.
#[derive(Debug, Default)]
pub struct MockUncertaintyGenerator {
    /// Pre-scripted segments, consumed front-to-back across calls.
    segments: std::cell::RefCell<std::collections::VecDeque<Vec<TokenInfo>>>,
}

impl MockUncertaintyGenerator {
    /// Create a mock generator from an ordered list of scripted segments.
    #[must_use]
    pub fn new(segments: Vec<Vec<TokenInfo>>) -> Self {
        Self {
            segments: std::cell::RefCell::new(segments.into_iter().collect()),
        }
    }

    /// Number of scripted segments not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.segments.borrow().len()
    }
}

impl UncertaintyGenerator for MockUncertaintyGenerator {
    fn generate(&self, _context: &str) -> Vec<TokenInfo> {
        self.segments.borrow_mut().pop_front().unwrap_or_default()
    }
}

// ── MockRetriever ─────────────────────────────────────────────────────────────

/// Deterministic [`Retriever`] for tests.
///
/// Returns the passages of the first `(substring, passages)` mapping whose
/// `substring` is contained in the query. When no mapping matches, the query
/// itself is echoed back as a single-element passage list, so retrieval is
/// always observable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockRetriever {
    /// Ordered `(query substring, passages)` mappings.
    pub mappings: Vec<(String, Vec<String>)>,
}

impl MockRetriever {
    /// Create a mock retriever from `(query substring, passages)` mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, Vec<String>)>) -> Self {
        Self { mappings }
    }

    /// Create a mock retriever that echoes every query as its sole passage.
    #[must_use]
    pub fn echo() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }
}

impl Retriever for MockRetriever {
    fn retrieve(&self, query: &str) -> Vec<String> {
        for (needle, passages) in &self.mappings {
            if query.contains(needle.as_str()) {
                return passages.clone();
            }
        }
        vec![query.to_string()]
    }
}

// ── DraginConfig ──────────────────────────────────────────────────────────────

/// Configuration for [`DraginEngine`](crate::dragin::engine::DraginEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct DraginConfig {
    /// Confidence below which a token is considered uncertain by RIND.
    ///
    /// A token triggers retrieval when `confidence < uncertainty_threshold`
    /// **and** the token is a content token. Defaults to `0.5`.
    pub uncertainty_threshold: f32,
    /// Maximum number of retrievals performed in a single run.
    ///
    /// Once this many retrievals have been recorded the loop halts even if more
    /// uncertain content tokens would otherwise trigger. Defaults to `5`.
    pub max_retrievals: usize,
    /// Number of most-recent generated tokens QFS considers when forming a
    /// query. Defaults to `10`.
    pub query_window: usize,
}

impl Default for DraginConfig {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.5,
            max_retrievals: 5,
            query_window: 10,
        }
    }
}

impl DraginConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the uncertainty threshold used by RIND.
    #[must_use]
    pub fn with_uncertainty_threshold(mut self, uncertainty_threshold: f32) -> Self {
        self.uncertainty_threshold = uncertainty_threshold;
        self
    }

    /// Set the maximum number of retrievals per run.
    #[must_use]
    pub fn with_max_retrievals(mut self, max_retrievals: usize) -> Self {
        self.max_retrievals = max_retrievals;
        self
    }

    /// Set the QFS query window (number of recent tokens considered).
    #[must_use]
    pub fn with_query_window(mut self, query_window: usize) -> Self {
        self.query_window = query_window;
        self
    }
}

// ── RetrievalTrigger ──────────────────────────────────────────────────────────

/// A single retrieval event fired by RIND during a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalTrigger {
    /// Index of the triggering token within the segment that produced it.
    pub position: usize,
    /// The content token whose low confidence triggered retrieval.
    pub trigger_token: String,
    /// The query formed by QFS from the salient recent tokens.
    pub formed_query: String,
    /// Passages returned by the retriever for [`RetrievalTrigger::formed_query`].
    pub retrieved: Vec<String>,
}

// ── DraginTrace ───────────────────────────────────────────────────────────────

/// The full record of a DRAGIN run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraginTrace {
    /// Retrieval events fired, in the order they occurred.
    pub triggers: Vec<RetrievalTrigger>,
    /// The accumulated generated text across all segments.
    pub generated: String,
    /// Number of retrievals performed (equal to `triggers.len()`).
    pub num_retrievals: usize,
}

// ── DraginError ───────────────────────────────────────────────────────────────

/// Errors from the `dragin` module.
#[derive(Debug, Error)]
pub enum DraginError {
    /// The original query was empty after trimming.
    #[error("query must not be empty")]
    EmptyQuery,
}
