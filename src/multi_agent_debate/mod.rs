//! Adversarial multi-agent debate — two or more personas argue opposing
//! positions across multiple rounds, each seeing and explicitly rebutting
//! the others' arguments, before a separate judge persona decides a winner.
//!
//! ## How this differs from its neighbours
//!
//! - **`crate::self_consistency`** samples `K` reasoning chains
//!   *independently* — no chain ever sees another chain's reasoning — and
//!   marginalizes over their final answers by majority vote. There is no
//!   dialogue, no rebuttal, and no adversarial framing: the chains do not
//!   know each other exist. `multi_agent_debate`'s personas, by contrast,
//!   are handed the *complete* transcript of every argument made so far —
//!   including every opponent's — before producing their next argument
//!   (see [`DebatePersona::argue`]), and are expected to engage with, not
//!   merely coexist alongside, what an opponent just said.
//! - **`crate::tree_of_thought`** and **`crate::graph_of_thought`** are
//!   *single-agent* search: one reasoning process branches into a tree (or
//!   a DAG, with `graph_of_thought`'s aggregation/refinement operations) of
//!   candidate thoughts, evaluated and pruned by a scorer. There is no
//!   second party, no opposing position, and no rebuttal — just one agent
//!   exploring more of the solution space. `multi_agent_debate` instead has
//!   `N >= 2` *adversarial* personas, each committed to a distinct position
//!   for the entire debate, who read and directly respond to each other.
//! - A separate [`DebateJudge`] — never one of the arguing personas — makes
//!   the final call after reviewing the whole multi-round transcript,
//!   mirroring how AI-safety debate protocols keep the debaters and the
//!   judge separate.
//!
//! ## Structure
//!
//! [`DebateEngine::run`] drives `N >= 2` participants (each a
//! [`DebatePersona`] paired with a distinct position, via
//! [`DebateParticipant`]) through [`DebateConfig::max_rounds`] rounds. Each
//! round, every participant is called exactly once and is handed every
//! argument from every strictly earlier round (see [`DebatePersona::argue`]
//! for the exact visibility contract); its returned [`DebateArgument`] is
//! appended to that round's [`DebateRound`]. When
//! [`DebateConfig::early_stop`] is enabled, the debate can end before
//! `max_rounds` once a participant's self-reported confidence stops
//! improving for [`DebateConfig::flat_confidence_window`] consecutive
//! rounds. Once the transcript is complete, the caller-supplied
//! [`DebateJudge`] reviews it and returns a [`DebateVerdict`] (winning
//! position, rationale, and a per-position [`DebatePositionScore`]
//! breakdown), assembled together with the transcript into the final
//! [`DebateResult`].
//!
//! [`MockDebatePersona`] and [`MockDebateJudge`] provide deterministic
//! implementations of [`DebatePersona`] and [`DebateJudge`] (`FNV-1a`-hash
//! and lexical-overlap driven — no live model), for use in tests and
//! examples.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "multi-agent-debate")]
//! # {
//! use oxirag::multi_agent_debate::{
//!     DebateConfig, DebateEngine, DebateParticipant, MockDebateJudge, MockDebatePersona,
//! };
//!
//! let optimist = MockDebatePersona::new("Optimist");
//! let skeptic = MockDebatePersona::new("Skeptic");
//! let participants = [
//!     DebateParticipant::new(&optimist, "renewable energy will dominate by 2040"),
//!     DebateParticipant::new(&skeptic, "fossil fuels will remain dominant past 2040"),
//! ];
//!
//! let engine = DebateEngine::new(DebateConfig::new().with_max_rounds(3));
//! let judge = MockDebateJudge::default();
//! let result = engine
//!     .run(
//!         "Will renewable energy dominate global supply by 2040?",
//!         &participants,
//!         &judge,
//!     )
//!     .expect("debate should run");
//!
//! assert_eq!(result.transcript.len(), 3);
//! assert!(result.transcript.iter().all(|round| round.arguments.len() == 2));
//! assert!(!result.verdict.rationale.is_empty());
//! # }
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::{DebateEngine, MockDebateJudge, MockDebatePersona};
pub use types::{
    DebateArgument, DebateConfig, DebateError, DebateJudge, DebateJudgeWeights, DebateParticipant,
    DebatePersona, DebatePositionScore, DebateResult, DebateRound, DebateVerdict,
};
