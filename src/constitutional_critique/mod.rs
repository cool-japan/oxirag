//! Constitutional AI-style critique-and-revise (Bai et al. 2022,
//! "Constitutional AI: Harmlessness from AI Feedback").
//!
//! A **fixed, named set of principles** — the "constitution" — is applied
//! to a draft one principle at a time, in order. For each principle, a
//! [`ConstitutionalReviser`] first
//! [`critique`](ConstitutionalReviser::critique)s the *current* draft
//! against that single principle; if the principle is violated, the
//! reviser [`revise`](ConstitutionalReviser::revise)s the draft to address
//! JUST that critique, and the revision becomes the current draft for
//! every later principle. If the principle is not violated, the draft
//! passes through that step unchanged. The full per-principle trace —
//! `(principle, critique, revision)` for every principle applied — is
//! accumulated and returned alongside the final draft.
//!
//! ```text
//! draft_0   --[principle 1: critique]--> violated? --[revise]--> draft_1
//! draft_1   --[principle 2: critique]--> violated? --[revise]--> draft_2
//! ...
//! draft_n-1 --[principle n: critique]--> violated? --[revise]--> draft_n
//! ```
//!
//! # Differentiator: vs `self_refine` and `guardrails`
//!
//! This module sits between two neighbours it is easy to confuse it with:
//!
//! - `self_refine` performs **open-ended** self-critique: a single model
//!   freely critiques its own output against no fixed rubric, in a
//!   feedback-then-refine loop bounded only by an iteration cap and a
//!   score threshold. There is no principle *list*, no per-step targeting
//!   of a specific named concern, and no per-principle trace — just a
//!   running critique and score.
//! - `guardrails` is a **detection/blocking** validation layer: it scans
//!   text once for PII, prompt-injection, and moderation signals and
//!   reports a pass/fail-style report (`blocked`, `risk_score`). It does
//!   not iterate, does not revise the text against each concern in turn,
//!   and does not target a violation with a rewrite — it only flags (and
//!   optionally redacts PII in one shot).
//! - `constitutional_critique` (this module), by contrast, walks a
//!   **fixed, named** list of [`ConstitutionalPrinciple`]s *in order*,
//!   applying a dedicated critique-then-revise step to **each one
//!   individually** — every principle gets its own
//!   [`ConstitutionalCritique`] against the draft *as it stands after
//!   every earlier principle's revision*, and, on violation, its own
//!   targeted [`ConstitutionalRevision`]. The result is an ordered,
//!   auditable trace of exactly which principle changed what, and in what
//!   sequence.
//!
//! # Built-in constitution
//!
//! [`ConstitutionalPrinciple::default_constitution`] provides five
//! principles spanning the three trigger heuristics in
//! [`ConstitutionalTrigger`]: `avoid_harm`, `be_respectful`, and
//! `avoid_illegal_activity` (keyword lists), `be_truthful`
//! (absolute/overconfident phrase paired with a hedge), and
//! `respect_privacy` (hand-rolled PII-shaped token scanning — no regex).
//! Callers may also build a fully custom constitution from scratch with
//! [`ConstitutionalPrinciple::new`].
//!
//! [`MockConstitutionalReviser`] implements real (not scripted) versions
//! of these heuristics deterministically, for use without a live LLM: it
//! scans the draft against each principle's own trigger definitions and,
//! on a hit, applies the fix recorded on the match — strip/redact a
//! keyword, substitute a hedge phrase, or redact a PII-shaped span.
//!
//! # Example
//!
//! ```
//! use oxirag::constitutional_critique::{
//!     ConstitutionalConfig, ConstitutionalEngine, ConstitutionalPrinciple,
//!     MockConstitutionalReviser,
//! };
//!
//! let draft = "This will definitely kill the bug in your code.";
//! let principles = ConstitutionalPrinciple::default_constitution();
//! let reviser = MockConstitutionalReviser::new();
//!
//! let engine = ConstitutionalEngine::new(ConstitutionalConfig::default());
//! let result = engine.run(draft, &principles, &reviser).unwrap();
//!
//! // `avoid_harm` (on "kill") and `be_truthful` (on "definitely") both
//! // fire; the other three default principles do not.
//! assert_eq!(result.revisions_applied, 2);
//! assert_eq!(result.trace.len(), principles.len());
//! assert_ne!(result.final_draft, draft);
//! assert!(!result.final_draft.contains("kill"));
//! assert!(!result.final_draft.contains("definitely"));
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::ConstitutionalEngine;
pub use types::{
    ConstitutionalConfig, ConstitutionalCritique, ConstitutionalError, ConstitutionalMatch,
    ConstitutionalPrinciple, ConstitutionalResult, ConstitutionalReviser, ConstitutionalRevision,
    ConstitutionalTraceEntry, ConstitutionalTrigger, MockConstitutionalReviser,
};
