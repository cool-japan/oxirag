//! Buffer of Thoughts (Yang et al. 2024, "Buffer of Thoughts: Thought-Augmented
//! Reasoning with Large Language Models") — a **persistent, growing**
//! meta-buffer of distilled thought templates.
//!
//! # How this differs from `self_discover` and `analogical_prompting`
//!
//! - [`crate::self_discover`] SELECTs atomic reasoning modules from a
//!   **fixed, built-in bank** (8 static modules) that never grows or changes
//!   across runs.
//! - [`crate::analogical_prompting`] asks the model to **self-generate**
//!   fresh exemplars for every single problem, with no memory carried between
//!   calls — nothing persists.
//! - **Buffer of Thoughts** maintains a [`ThoughtBuffer`] that starts empty
//!   and **grows over time**: every problem solved successfully is a
//!   candidate for *distillation* into a new, reusable
//!   [`ThoughtTemplate`], and every problem that *reuses* an existing
//!   template *reinforces* it (updating `usage_count` / `success_rate`). The
//!   buffer is the module's entire value proposition — it gets more useful
//!   the more problems it has processed, and that value compounds across
//!   sessions when the caller persists a [`ThoughtBuffer`]'s templates
//!   externally.
//!
//! # Pipeline
//!
//! [`BotEngine`] orchestrates four stages per problem:
//!
//! 1. **Retrieve** — compute the new problem's [`ProblemSignature`] (a
//!    structural fingerprint, *not* the literal text — see
//!    [`crate::buffer_of_thoughts::types`] for the exact scheme) and look up
//!    the closest-matching buffered template by weighted-Jaccard tag overlap.
//!    Below [`BotConfig::similarity_threshold`], retrieval honestly reports
//!    "no match" rather than forcing a bad one.
//! 2. **Instantiate** — substitute the matched template's `{problem}` and
//!    `{opN}` placeholders with the new problem's specifics (or instantiate
//!    the generic [`crate::buffer_of_thoughts::types::DEFAULT_TEMPLATE_TEXT`]
//!    fallback when nothing matched).
//! 3. **Solve** — ask a pluggable [`BotGenerator`] (e.g. [`MockBotGenerator`]
//!    for tests) to produce an answer conditioned on the instantiated
//!    scaffold.
//! 4. **Distill** — once the caller supplies a [`SolveOutcome`] for the
//!    answer, either reinforce the reused template, distill a brand-new one
//!    from a successful unmatched solve, or skip write-back for an
//!    unsuccessful unmatched solve.
//!
//! # Example — the grow-then-reuse lifecycle
//!
//! ```
//! use oxirag::buffer_of_thoughts::{
//!     BotConfig, BotEngine, DistillAction, MockBotGenerator, SolveOutcome,
//! };
//!
//! let mut engine = BotEngine::new(BotConfig::default(), MockBotGenerator);
//! assert!(engine.buffer.is_empty());
//!
//! // Nothing buffered yet: falls back to the generic scaffold, succeeds, and
//! // distills a brand-new template — the buffer grows from 0 to 1.
//! let (first, action) = engine
//!     .solve_and_record("What is the sum of 12 and 7?", |_| SolveOutcome::Success)
//!     .unwrap();
//! assert!(first.matched_template_id.is_none());
//! assert!(matches!(action, DistillAction::Distilled(_)));
//! assert_eq!(engine.buffer.len(), 1);
//!
//! // A structurally similar problem now retrieves and reuses that template
//! // instead of starting from scratch — the buffer's value compounds without
//! // growing further.
//! let (second, action) = engine
//!     .solve_and_record("What is the sum of 30 and 5?", |_| SolveOutcome::Success)
//!     .unwrap();
//! assert!(second.matched_template_id.is_some());
//! assert!(matches!(action, DistillAction::Reinforced(_)));
//! assert_eq!(engine.buffer.len(), 1);
//! assert_eq!(engine.buffer.templates()[0].usage_count, 2);
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::{BotEngine, ThoughtBuffer};
pub use types::{
    BotConfig, BotError, BotGenerator, BotSolveResult, DEFAULT_TEMPLATE_TEXT, DistillAction,
    EvictionPolicy, MockBotGenerator, ProblemSignature, SolveOutcome, ThoughtTemplate,
};
