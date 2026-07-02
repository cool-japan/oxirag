//! Types and traits for the `retrieval_augmented_thoughts` module.
//!
//! Retrieval-Augmented Thoughts (RAT; Wang et al., 2024) first drafts an
//! ordinary chain-of-thought (`CoT`) for a question, then walks the draft
//! **left-to-right**, revising each thought step with evidence retrieved
//! specifically for that step. The retrieval query for step `i` is built
//! from the original question, the thoughts already revised (`0..i`), and
//! the current draft thought `i` — so retrieval is conditioned on the
//! cumulative reasoning so far, not just the bare question.
//!
//! This is deliberately different from:
//! - `iterative_rag` (ITER-RETGEN), which regenerates the *whole* answer
//!   each round from a query expanded with terms pulled from the previous
//!   draft, and
//! - `self_ask`, which interleaves explicit follow-up sub-questions with
//!   sub-answers rather than revising a pre-drafted reasoning chain.
//!
//! The generator and retriever are supplied by the caller as the
//! [`RatGenerator`] and [`RatRetriever`] traits. Deterministic
//! [`MockRatGenerator`] and [`MockRatRetriever`] implementations are
//! provided for tests and doctests.

use std::collections::HashSet;

use thiserror::Error;

// ── RatConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`RatEngine`](crate::retrieval_augmented_thoughts::RatEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatConfig {
    /// Maximum number of thought steps to draft and revise.
    ///
    /// Defaults to `5`. When the generator drafts more steps than this, the
    /// draft chain is truncated to the first `max_thoughts` steps before
    /// revision begins.
    pub max_thoughts: usize,
    /// Number of passages to retrieve per thought step.
    ///
    /// Defaults to `3`.
    pub top_k: usize,
    /// Minimum trimmed length (in bytes) a draft thought must have before the
    /// engine bothers retrieving evidence and revising it.
    ///
    /// Defaults to `0`, meaning every drafted thought is revised. Raising
    /// this lets trivial or filler draft thoughts (e.g. a bare punctuation
    /// mark) pass through unchanged, with no retrieval query issued and no
    /// passages recorded.
    pub min_thought_len: usize,
}

impl Default for RatConfig {
    fn default() -> Self {
        Self {
            max_thoughts: 5,
            top_k: 3,
            min_thought_len: 0,
        }
    }
}

impl RatConfig {
    /// Create a new [`RatConfig`] with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of thought steps.
    #[must_use]
    pub fn with_max_thoughts(mut self, max_thoughts: usize) -> Self {
        self.max_thoughts = max_thoughts;
        self
    }

    /// Set the number of passages retrieved per thought step.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the minimum draft-thought length required before revision.
    #[must_use]
    pub fn with_min_thought_len(mut self, min_thought_len: usize) -> Self {
        self.min_thought_len = min_thought_len;
        self
    }

    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RatError::InvalidConfig`] when `max_thoughts` is `0`, since
    /// no reasoning chain could ever be produced (the algorithm requires at
    /// least one thought step to draft and revise).
    pub fn validate(&self) -> Result<(), RatError> {
        if self.max_thoughts == 0 {
            return Err(RatError::InvalidConfig(
                "max_thoughts must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

// ── RatThought ────────────────────────────────────────────────────────────────

/// One thought step in a RAT reasoning chain: the initial draft, the
/// retrieval query built for it, the passages retrieved, and the revised
/// text produced from that evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatThought {
    /// Zero-based position of this thought within the chain.
    pub index: usize,
    /// The initial (unrevised) draft text for this step.
    pub draft: String,
    /// The revised text for this step, corrected using retrieved evidence.
    pub revised: String,
    /// The retrieval query issued for this step (empty when retrieval was
    /// skipped because the draft was shorter than
    /// [`RatConfig::min_thought_len`]).
    pub retrieval_query: String,
    /// Passages retrieved for this step (empty when retrieval was skipped).
    pub passages: Vec<String>,
}

impl RatThought {
    /// Returns `true` if revision changed the text of this thought.
    #[must_use]
    pub fn was_revised(&self) -> bool {
        self.draft != self.revised
    }

    /// Returns `true` if any passages were retrieved for this thought.
    #[must_use]
    pub fn has_passages(&self) -> bool {
        !self.passages.is_empty()
    }
}

// ── RatTrace ──────────────────────────────────────────────────────────────────

/// The full record of a RAT run: the question, every thought step (draft,
/// retrieval query, passages, revision), and the synthesised final answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatTrace {
    /// The original question, trimmed.
    pub question: String,
    /// Thought steps in left-to-right order.
    pub thoughts: Vec<RatThought>,
    /// The final answer synthesised from the revised reasoning chain.
    pub final_answer: String,
}

impl RatTrace {
    /// Number of thought steps in the chain.
    #[must_use]
    pub fn num_thoughts(&self) -> usize {
        self.thoughts.len()
    }

    /// The revised text of every thought step, in order.
    #[must_use]
    pub fn revised_thoughts(&self) -> Vec<&str> {
        self.thoughts.iter().map(|t| t.revised.as_str()).collect()
    }
}

// ── RatError ──────────────────────────────────────────────────────────────────

/// Errors from the `retrieval_augmented_thoughts` module.
#[derive(Debug, Error)]
pub enum RatError {
    /// The question was empty or contained only whitespace.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// The generator drafted zero thought steps.
    #[error("no thoughts were drafted")]
    NoThoughts,
    /// The supplied [`RatConfig`] was invalid.
    #[error("invalid RAT configuration: {0}")]
    InvalidConfig(String),
}

// ── RatGenerator ──────────────────────────────────────────────────────────────

/// A language model that drives the RAT drafting/revision/finalisation
/// steps.
///
/// Implementations are **pure sync** — no I/O, no async — mirroring
/// `self_ask::SelfAskModel` and `iterative_rag`'s draft-building helpers.
/// The caller supplies a concrete generator (e.g. wrapping an LLM);
/// [`MockRatGenerator`] is provided for tests.
pub trait RatGenerator {
    /// Draft an initial, ordered chain-of-thought for `question`.
    ///
    /// The returned steps are later revised left-to-right by
    /// [`RatGenerator::revise`]; this method does not see any retrieved
    /// evidence.
    fn draft_cot(&self, question: &str) -> Vec<String>;

    /// Revise a single draft thought using step-targeted retrieved
    /// evidence.
    ///
    /// `prior` holds the *already-revised* thoughts `0..i` (in order);
    /// `draft` is the current step's unrevised text; `passages` are the
    /// passages retrieved for this step's query.
    fn revise(&self, question: &str, prior: &[String], draft: &str, passages: &[String]) -> String;

    /// Synthesise the final answer from `question` and the fully revised
    /// `thoughts` chain.
    fn finalize(&self, question: &str, thoughts: &[String]) -> String;
}

// ── RatRetriever ──────────────────────────────────────────────────────────────

/// Retrieves supporting passages for a step-targeted query.
///
/// Implementations may wrap a vector index, a lexical search engine, or any
/// other retrieval backend. They are **pure sync**. [`MockRatRetriever`] is
/// provided for tests.
pub trait RatRetriever {
    /// Retrieve up to `top_k` passages relevant to `query`.
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<String>;
}

// ── Blanket reference impls (enable `&dyn RatGenerator` / `&dyn RatRetriever`) ──

impl<T> RatGenerator for &T
where
    T: RatGenerator + ?Sized,
{
    fn draft_cot(&self, question: &str) -> Vec<String> {
        (**self).draft_cot(question)
    }

    fn revise(&self, question: &str, prior: &[String], draft: &str, passages: &[String]) -> String {
        (**self).revise(question, prior, draft, passages)
    }

    fn finalize(&self, question: &str, thoughts: &[String]) -> String {
        (**self).finalize(question, thoughts)
    }
}

impl<T> RatRetriever for &T
where
    T: RatRetriever + ?Sized,
{
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<String> {
        (**self).retrieve(query, top_k)
    }
}

// ── lexical overlap helper (used by MockRatRetriever) ───────────────────────────

/// Normalise a token: lowercase and strip leading/trailing punctuation.
fn normalize_token(token: &str) -> String {
    token
        .to_lowercase()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// Tokenise `text` into a set of normalised, non-empty tokens.
fn lexical_tokens(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect()
}

// ── MockRatGenerator ──────────────────────────────────────────────────────────

/// Deterministic [`RatGenerator`] for tests and doctests.
///
/// [`RatGenerator::draft_cot`] returns the pre-scripted `drafts` unchanged
/// (the `question` argument is ignored, matching the scripted-mock idiom
/// used by `self_ask::MockSelfAskModel`).
///
/// [`RatGenerator::revise`] is template-based: when `passages` is empty the
/// draft is returned unchanged (nothing to revise with); otherwise the
/// draft is followed by a deterministic `" (confirmed by: ...)"` suffix
/// listing the retrieved passages, so tests can assert exact output and
/// confirm that revision actually incorporates retrieval.
///
/// [`RatGenerator::finalize`] joins the revised thoughts with `" -> "` under
/// a fixed `"Final answer based on: "` prefix.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockRatGenerator {
    /// Pre-scripted draft chain-of-thought steps, returned verbatim by
    /// [`RatGenerator::draft_cot`].
    pub drafts: Vec<String>,
}

impl MockRatGenerator {
    /// Create a mock generator that drafts the given `drafts` script.
    #[must_use]
    pub fn new(drafts: Vec<String>) -> Self {
        Self { drafts }
    }

    /// Create a mock generator that drafts zero thoughts.
    #[must_use]
    pub fn empty() -> Self {
        Self { drafts: Vec::new() }
    }
}

impl RatGenerator for MockRatGenerator {
    fn draft_cot(&self, _question: &str) -> Vec<String> {
        self.drafts.clone()
    }

    fn revise(
        &self,
        _question: &str,
        _prior: &[String],
        draft: &str,
        passages: &[String],
    ) -> String {
        if passages.is_empty() {
            draft.to_string()
        } else {
            format!("{draft} (confirmed by: {})", passages.join("; "))
        }
    }

    fn finalize(&self, _question: &str, thoughts: &[String]) -> String {
        format!("Final answer based on: {}", thoughts.join(" -> "))
    }
}

// ── MockRatRetriever ──────────────────────────────────────────────────────────

/// Deterministic [`RatRetriever`] for tests and doctests.
///
/// Holds a fixed `corpus` of passages. [`RatRetriever::retrieve`] scores
/// every passage by the size of the token overlap between `query` and the
/// passage (case-insensitive, punctuation-stripped whitespace tokens),
/// keeps only passages with a positive overlap, ranks them by descending
/// overlap (ties broken by ascending corpus order for determinism), and
/// returns the top `top_k`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockRatRetriever {
    /// The fixed passage corpus searched by [`RatRetriever::retrieve`].
    pub corpus: Vec<String>,
}

impl MockRatRetriever {
    /// Create a mock retriever backed by `corpus`.
    #[must_use]
    pub fn new(corpus: Vec<String>) -> Self {
        Self { corpus }
    }

    /// Create a mock retriever with an empty corpus (always retrieves
    /// nothing).
    #[must_use]
    pub fn empty() -> Self {
        Self { corpus: Vec::new() }
    }
}

impl RatRetriever for MockRatRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<String> {
        if top_k == 0 || self.corpus.is_empty() {
            return Vec::new();
        }

        let query_tokens = lexical_tokens(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, usize, &String)> = self
            .corpus
            .iter()
            .enumerate()
            .map(|(i, passage)| {
                let passage_tokens = lexical_tokens(passage);
                let overlap = query_tokens.intersection(&passage_tokens).count();
                (overlap, i, passage)
            })
            .filter(|(overlap, _, _)| *overlap > 0)
            .collect();

        // Rank by descending overlap; break ties by ascending corpus index
        // so the result order is fully deterministic.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        scored
            .into_iter()
            .take(top_k)
            .map(|(_, _, passage)| passage.clone())
            .collect()
    }
}
