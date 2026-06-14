//! Types and traits for the `self_ask` module.
//!
//! Self-Ask (Press et al. 2022) drives **compositional** multi-hop reasoning by
//! having a language model repeatedly decide whether a follow-up question is
//! needed, ask it, answer it (optionally via retrieval), and finally compose the
//! overall answer. The prompting scaffold mirrors the paper:
//!
//! ```text
//! Are follow up questions needed here: Yes
//! Follow up: <q1>
//! Intermediate answer: <a1>
//! Follow up: <q2>
//! Intermediate answer: <a2>
//! So the final answer is: <final>
//! ```
//!
//! Unlike upfront `query_decomposition` (which splits a query once, in advance)
//! or graph-based `multi_hop` traversal, Self-Ask interleaves *self-questioning*
//! with answering: each follow-up may depend on the answers to prior follow-ups.
//!
//! The model and the sub-answerer are supplied by the caller as the
//! [`SelfAskModel`] and [`SubAnswerer`] traits. Deterministic [`MockSelfAskModel`]
//! and [`MockSubAnswerer`] implementations are provided for testing.

use thiserror::Error;

// ── FollowUp ──────────────────────────────────────────────────────────────────

/// One resolved follow-up step: a self-asked question paired with its answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FollowUp {
    /// The follow-up question the model decided to ask.
    pub question: String,
    /// The answer produced for [`FollowUp::question`] by the [`SubAnswerer`].
    pub answer: String,
}

impl FollowUp {
    /// Create a new follow-up from a question and its answer.
    #[must_use]
    pub fn new(question: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            answer: answer.into(),
        }
    }
}

// ── SelfAskTrace ──────────────────────────────────────────────────────────────

/// The full record of a Self-Ask run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfAskTrace {
    /// Resolved follow-up steps, in the order they were asked.
    pub follow_ups: Vec<FollowUp>,
    /// The composed final answer to the original question.
    pub final_answer: String,
    /// Number of follow-up hops taken (equal to `follow_ups.len()`).
    pub num_hops: usize,
}

// ── SelfAskConfig ─────────────────────────────────────────────────────────────

/// Configuration for [`SelfAskEngine`](crate::self_ask::engine::SelfAskEngine).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfAskConfig {
    /// Maximum number of follow-up questions to ask before composing the answer.
    ///
    /// Defaults to `5`. The engine halts the self-questioning loop once this many
    /// follow-ups have been recorded, even if the model would keep asking.
    pub max_follow_ups: usize,
}

impl Default for SelfAskConfig {
    fn default() -> Self {
        Self { max_follow_ups: 5 }
    }
}

impl SelfAskConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of follow-up questions.
    #[must_use]
    pub fn with_max_follow_ups(mut self, max_follow_ups: usize) -> Self {
        self.max_follow_ups = max_follow_ups;
        self
    }
}

// ── SelfAskModel ──────────────────────────────────────────────────────────────

/// A language model that drives the Self-Ask self-questioning loop.
///
/// Implementations are **pure sync** — no I/O, no async. The caller supplies a
/// concrete model (e.g. wrapping an LLM); [`MockSelfAskModel`] is provided for
/// tests.
pub trait SelfAskModel {
    /// Decide the next follow-up question given the original `question` and the
    /// `history` of resolved follow-ups so far.
    ///
    /// Returns `Some(follow_up)` when another follow-up is needed, or `None` when
    /// the model is ready to compose the final answer.
    fn next_follow_up(&self, question: &str, history: &[FollowUp]) -> Option<String>;

    /// Compose the final answer from the original `question` and the `history`
    /// of resolved follow-ups.
    fn compose_final(&self, question: &str, history: &[FollowUp]) -> String;
}

// ── SubAnswerer ───────────────────────────────────────────────────────────────

/// Answers a single follow-up question.
///
/// Implementations may wrap a retriever (answering from retrieved documents) or
/// any other source. They are **pure sync**. [`MockSubAnswerer`] is provided for
/// tests.
pub trait SubAnswerer {
    /// Answer a single `follow_up` question.
    fn answer(&self, follow_up: &str) -> String;
}

// ── MockSelfAskModel ──────────────────────────────────────────────────────────

/// Deterministic [`SelfAskModel`] for tests.
///
/// Returns the pre-scripted follow-up at `follow_ups[history.len()]` until the
/// scripted questions are exhausted, after which [`SelfAskModel::next_follow_up`]
/// returns `None`. [`SelfAskModel::compose_final`] always returns the configured
/// `final_answer`. An empty `follow_ups` script means the model answers
/// immediately (zero hops).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockSelfAskModel {
    /// Pre-scripted follow-up questions, consumed in order by hop index.
    pub follow_ups: Vec<String>,
    /// The final answer returned by [`SelfAskModel::compose_final`].
    pub final_answer: String,
}

impl MockSelfAskModel {
    /// Create a mock model from scripted follow-ups and a final answer.
    #[must_use]
    pub fn new(follow_ups: Vec<String>, final_answer: impl Into<String>) -> Self {
        Self {
            follow_ups,
            final_answer: final_answer.into(),
        }
    }

    /// Create a mock model that answers immediately with no follow-ups.
    #[must_use]
    pub fn immediate(final_answer: impl Into<String>) -> Self {
        Self {
            follow_ups: Vec::new(),
            final_answer: final_answer.into(),
        }
    }
}

impl SelfAskModel for MockSelfAskModel {
    fn next_follow_up(&self, _question: &str, history: &[FollowUp]) -> Option<String> {
        self.follow_ups.get(history.len()).cloned()
    }

    fn compose_final(&self, _question: &str, _history: &[FollowUp]) -> String {
        self.final_answer.clone()
    }
}

// ── MockSubAnswerer ───────────────────────────────────────────────────────────

/// Deterministic [`SubAnswerer`] for tests.
///
/// Answers a follow-up by returning the value of the first `(substring, answer)`
/// pair whose `substring` is contained in the follow-up question. When no pair
/// matches, the follow-up question is echoed back unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockSubAnswerer {
    /// Ordered `(follow-up substring, answer)` mappings.
    pub answers: Vec<(String, String)>,
}

impl MockSubAnswerer {
    /// Create a mock sub-answerer from `(substring, answer)` mappings.
    #[must_use]
    pub fn new(answers: Vec<(String, String)>) -> Self {
        Self { answers }
    }

    /// Create an empty mock sub-answerer that echoes every follow-up.
    #[must_use]
    pub fn echo() -> Self {
        Self {
            answers: Vec::new(),
        }
    }
}

impl SubAnswerer for MockSubAnswerer {
    fn answer(&self, follow_up: &str) -> String {
        for (needle, answer) in &self.answers {
            if follow_up.contains(needle.as_str()) {
                return answer.clone();
            }
        }
        follow_up.to_string()
    }
}

// ── SelfAskError ──────────────────────────────────────────────────────────────

/// Errors from the `self_ask` module.
#[derive(Debug, Error)]
pub enum SelfAskError {
    /// The original question was empty after trimming.
    #[error("question must not be empty")]
    EmptyQuestion,
}
