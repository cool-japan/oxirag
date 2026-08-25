//! Extractive question answering: `SQuAD`-style answer-span extraction.
//!
//! Given an interrogative `question` and a source `passage`,
//! [`ExtractiveQaEngine::answer`] returns the single contiguous span of the
//! passage that best answers the question — an [`AnswerSpan`] carrying the
//! verbatim text, its token offsets, the classified [`QuestionType`], and
//! the full [`SpanScore`] breakdown that produced it.
//! [`ExtractiveQaEngine::top_k_spans`] returns multiple ranked candidates
//! instead of just the best one.
//!
//! # How this differs from its closest neighbours
//!
//! This module is easy to confuse with two others in the crate that also
//! end up selecting a contiguous span of source text. The *task* each one
//! solves is different, even though the mechanics (pick a contiguous run of
//! tokens) can look superficially similar:
//!
//! * [`quote_grounding`](crate::quote_grounding) starts from a **claim** — a
//!   sentence already asserted by some generated answer — and finds the
//!   verbatim source span that *substantiates* it. This is a
//!   **claim → supporting-quote** framing: there is no question anywhere in
//!   the pipeline, and the output's job is to be evidence for a statement
//!   someone (or something) already made.
//! * `extractive_qa` (this module) starts from a **question** and finds the
//!   verbatim span that *answers* it — a **question → answer-span**
//!   framing. Boundary selection is driven by question-type-aware
//!   heuristics that have no equivalent in claim-to-quote matching: a `Who`
//!   question biases towards capitalized spans, a `When` question biases
//!   towards dates, a `HowMany`/`HowMuch` question biases towards numeric
//!   tokens, and so on (see [`QuestionType`]).
//! * [`self_ask`](crate::self_ask) and [`long_rag`](crate::long_rag) also
//!   work from a question, but they **generate or synthesise** answer text
//!   rather than extracting an exact span of a *given* passage.
//!   `self_ask` composes its final answer from a language model over a
//!   chain of self-asked follow-up questions; `long_rag` ranks long
//!   retrieval units and leaves answer synthesis to a downstream generator
//!   entirely. Neither guarantees — or even attempts — that its output is a
//!   contiguous, offset-addressable substring of an input passage the way
//!   every [`AnswerSpan`] here provably is.
//!
//! In short: claim-support vs. question-answering is one axis of
//! distinction (`quote_grounding` vs. everything else here); *extracted*
//! vs. *synthesised* answer text is the other (`extractive_qa` vs.
//! `self_ask`/`long_rag`). This module is the only one that is both
//! question-answering **and** extraction.
//!
//! # Algorithm
//!
//! 1. **Question-type classification** ([`QuestionType`]) — the question is
//!    classified by its interrogative word/pattern (`who`, `what`, `when`,
//!    `where`, `how many`, `how much`, `why`, `which`, falling back to
//!    `Other`) via configurable leftmost keyword matching (see
//!    [`ExtractiveQaConfig::type_keywords`]).
//! 2. **Candidate generation** ([`SpanCandidate`]) — the passage is
//!    tokenized on whitespace, and every contiguous token window up to
//!    [`ExtractiveQaConfig::max_span_tokens`] tokens long is generated. This
//!    bounds candidate generation to `O(passage_len * max_span_tokens)`
//!    rather than the `O(passage_len^2)` of considering every possible
//!    span, at the cost of being unable to return an answer longer than
//!    `max_span_tokens` — `SQuAD`-style extractive answers are almost always
//!    short, so this is a favourable trade-off; raise the cap for passages
//!    that plausibly need longer answers.
//! 3. **Boundary scoring** ([`SpanScore`]) — every candidate is scored as a
//!    configurably-weighted blend of three signals: a question-type-aware
//!    content match against the span itself, lexical overlap between the
//!    question's content words and the span's immediate surrounding
//!    context (not the span itself), and a length prior that mildly
//!    penalises long (or degenerate zero-length) spans. See
//!    [`ExtractiveQaEngine::score_span`] for the exact formula.
//! 4. **Selection** — [`ExtractiveQaEngine::answer`] returns the
//!    single highest-scoring span; [`ExtractiveQaEngine::top_k_spans`]
//!    returns the `k` highest-scoring spans, descending by score, with no
//!    duplicate offsets.
//!
//! # Example
//!
//! ```
//! use oxirag::extractive_qa::{ExtractiveQaConfig, ExtractiveQaEngine, QuestionType};
//!
//! let engine = ExtractiveQaEngine::new(ExtractiveQaConfig::default());
//! let passage = "Marie Curie discovered radium in 1898 alongside Pierre Curie.";
//!
//! let answer = engine
//!     .answer("When was radium discovered?", passage)
//!     .unwrap();
//! assert_eq!(answer.question_type, QuestionType::When);
//! assert!(answer.text.contains("1898"));
//!
//! // `top_k_spans` exposes the ranked alternatives, not just the winner.
//! let ranked = engine
//!     .top_k_spans("When was radium discovered?", passage, 3)
//!     .unwrap();
//! assert!(ranked.len() <= 3);
//! assert_eq!(ranked[0].text, answer.text);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::ExtractiveQaEngine;
pub use types::{
    AnswerSpan, ExtractiveQaConfig, ExtractiveQaError, ExtractiveQaResult, QuestionType,
    SpanCandidate, SpanScore,
};
