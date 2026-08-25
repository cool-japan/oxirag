//! Synthetic evaluation-set generation (RAGAS / ARES style).
//!
//! This module *generates* offline RAG evaluation sets from a corpus. Given a
//! collection of [`Document`](crate::types::Document)s it produces
//! [`SyntheticQa`] tuples — each pairing a generated question with its
//! ground-truth answer, the source document and sentence, and a set of mined
//! hard-negative distractors.
//!
//! It is intentionally *distinct* from the `evaluation`, `retrieval_eval`, and
//! `llm_judge` modules: those **score** an existing answer, whereas this module
//! **creates** the test set those scorers consume.
//!
//! # Pipeline
//!
//! 1. Each document is split into sentences, which are ranked by term-frequency
//!    salience (stopwords discounted).
//! 2. A [`QuestionTemplater`] turns each salient sentence into
//!    `(question, answer, type)` tuples — factoid, definitional, cloze, and
//!    relational forms.
//! 3. For every question, hard-negative distractors are mined from other
//!    documents that share salient terms with the question but do not contain
//!    the answer.
//!
//! All steps are purely lexical and deterministic — no randomness, no external
//! models.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use oxirag::synthetic_eval::{SyntheticEvalConfig, SyntheticEvalGenerator};
//! use oxirag::types::Document;
//!
//! let corpus = vec![
//!     Document::new("Ada Lovelace wrote the first algorithm for the Analytical Engine."),
//!     Document::new("Charles Babbage designed the Analytical Engine in the 1830s."),
//! ];
//! let generator = SyntheticEvalGenerator::new(SyntheticEvalConfig::default());
//! let qa_set = generator.generate(&corpus).expect("non-empty corpus");
//! for qa in &qa_set {
//!     println!("Q: {}\nA: {}", qa.question, qa.answer);
//! }
//! ```

pub mod generator;
pub mod templater;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use generator::SyntheticEvalGenerator;
pub use templater::{HeuristicTemplater, QuestionTemplater, sentences, tokenize};
pub use types::{QuestionType, SyntheticEvalConfig, SyntheticEvalError, SyntheticQa};
