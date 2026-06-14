//! Types for the `synthetic_eval` module.
//!
//! These define the configuration, the generated question/answer records, and
//! the error type used throughout synthetic evaluation-set generation.

use crate::types::DocumentId;
use thiserror::Error;

// ── QuestionType ──────────────────────────────────────────────────────────────

/// The kind of question produced for a synthetic evaluation tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// A factoid question of the form "What did `<subject>` ...?" derived from a
    /// leading subject and its predicate.
    Factoid,
    /// A definitional question of the form "What is `<entity>`?" whose answer is
    /// the salient sentence describing the entity.
    Definitional,
    /// A cloze (fill-in-the-blank) question where the most salient term is masked
    /// with `___` and the answer is the masked term.
    Cloze,
    /// A relational question of the form
    /// "What is the relationship between `<a>` and `<b>`?" requiring two entities.
    Relational,
}

impl QuestionType {
    /// All four question types, in their canonical (template) order.
    #[must_use]
    pub fn all() -> Vec<QuestionType> {
        vec![
            QuestionType::Factoid,
            QuestionType::Definitional,
            QuestionType::Cloze,
            QuestionType::Relational,
        ]
    }

    /// A short, stable, lower-case label for this question type.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionType::Factoid => "factoid",
            QuestionType::Definitional => "definitional",
            QuestionType::Cloze => "cloze",
            QuestionType::Relational => "relational",
        }
    }
}

impl std::fmt::Display for QuestionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── SyntheticQa ───────────────────────────────────────────────────────────────

/// A single generated (question, ground-truth answer, source, distractors) tuple.
///
/// This is the unit of an offline RAG evaluation set: the `question` is asked of
/// a retrieval pipeline, `answer` is the expected ground truth, `source_id` is the
/// document the answer was derived from, and `distractor_ids` are hard negatives —
/// lexically related documents that nonetheless do not contain the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticQa {
    /// The generated natural-language question.
    pub question: String,
    /// The ground-truth answer, derived from the source sentence.
    pub answer: String,
    /// The identifier of the document the question was generated from.
    pub source_id: DocumentId,
    /// The exact source sentence the question and answer were derived from.
    pub source_sentence: String,
    /// Identifiers of hard-negative distractor documents (never the source).
    pub distractor_ids: Vec<DocumentId>,
    /// The kind of question that was generated.
    pub question_type: QuestionType,
}

// ── SyntheticEvalConfig ───────────────────────────────────────────────────────

/// Configuration for synthetic evaluation-set generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticEvalConfig {
    /// Maximum number of question/answer tuples to emit per document. Default `3`.
    pub questions_per_doc: usize,
    /// Number of hard-negative distractors to mine per question. Default `2`.
    pub num_distractors: usize,
    /// Minimum number of tokens a sentence must have to be considered. Default `5`.
    pub min_sentence_tokens: usize,
    /// The set of question types that are allowed to be generated. Default: all four.
    pub types: Vec<QuestionType>,
}

impl Default for SyntheticEvalConfig {
    fn default() -> Self {
        Self {
            questions_per_doc: 3,
            num_distractors: 2,
            min_sentence_tokens: 5,
            types: QuestionType::all(),
        }
    }
}

impl SyntheticEvalConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of questions emitted per document.
    #[must_use]
    pub fn with_questions_per_doc(mut self, value: usize) -> Self {
        self.questions_per_doc = value;
        self
    }

    /// Set the number of distractors mined per question.
    #[must_use]
    pub fn with_num_distractors(mut self, value: usize) -> Self {
        self.num_distractors = value;
        self
    }

    /// Set the minimum number of tokens a candidate sentence must contain.
    #[must_use]
    pub fn with_min_sentence_tokens(mut self, value: usize) -> Self {
        self.min_sentence_tokens = value;
        self
    }

    /// Restrict generation to the given question types.
    #[must_use]
    pub fn with_types(mut self, types: Vec<QuestionType>) -> Self {
        self.types = types;
        self
    }
}

// ── SyntheticEvalError ────────────────────────────────────────────────────────

/// Errors produced by the `synthetic_eval` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SyntheticEvalError {
    /// The provided corpus contained no documents.
    #[error("corpus is empty")]
    EmptyCorpus,
}
