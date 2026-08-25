//! Mixture-of-Agents (Wang et al., 2024, "Mixture-of-Agents Enhances Large
//! Language Model Capabilities"): a **layered**, collaborative architecture
//! in which several proposer agents independently answer a query, a
//! separate aggregator *synthesizes* their answers into one improved
//! response, and that synthesis is fed back to a fresh layer of proposers as
//! auxiliary context — repeating for `L` layers.
//!
//! # Structure
//!
//! - **Layer 1**: `N` [`MoaProposer`] agents each independently produce a
//!   candidate response to the query, with no context from one another
//!   ([`MoaProposer::propose`] receives an empty `prior_responses`).
//! - **Aggregator**: a distinct [`MoaAggregator`] **synthesizes** the `N`
//!   candidates into one improved response. It does not pick a winner — it
//!   merges their complementary content (see
//!   [`MoaSynthesisAggregator`]'s algorithm documentation).
//! - **Layer 2..L**: the aggregator's synthesis (and, depending on
//!   [`MoaContextMode`], the previous layer's individual proposals too) is
//!   fed back to a *fresh* set of proposer calls as auxiliary context —
//!   "here is what was previously said; produce a better response" — and
//!   the propose-then-aggregate cycle repeats.
//!
//! [`MoaEngine::run`] drives this whole loop and records a complete
//! [`MoaTrace`]: every layer's individual proposals, its aggregate, and its
//! [`MoaLayerStats`] (inter-proposal agreement, distinct-claim coverage, and
//! change from the previous layer's aggregate). When
//! [`MoaConfig::early_stop_similarity`] is set, the run can end before
//! `num_layers` once a layer's aggregate stops changing materially relative
//! to the previous one.
//!
//! The defining property — and the paper's central empirical claim, which
//! is directly testable here — is **synthesis-then-next-layer-of-proposers**:
//! each layer's proposers can see the *previous* layer's aggregated output,
//! so quality (informational coverage, in particular) should improve
//! monotonically with depth rather than staying flat the way independent
//! resampling would.
//!
//! # How this differs from its neighbours
//!
//! Two modules in this crate sound adjacent but are structurally different:
//!
//! - **`crate::self_consistency`** samples `K` reasoning chains
//!   **independently** — no chain ever sees another chain's reasoning, in
//!   any round, at any point — and marginalizes over their final answers by
//!   majority vote / clustering. There is no cross-agent communication at
//!   all. Mixture-of-agents' layer-`1` proposers *also* answer
//!   independently (so a single-layer run looks superficially similar), but
//!   from layer `2` onward its proposers are explicitly handed the
//!   *previous layer's synthesized output* via `prior_responses` — that
//!   inter-layer communication, entirely absent from `self_consistency`, is
//!   the whole point of a layered mixture-of-agents run (see
//!   [`MoaProposer::propose`] and the "layering is real" tests in this
//!   module's test suite, which record exactly what a layer-`2` proposer is
//!   handed).
//! - **`crate::multi_agent_debate`** binds each participant permanently to a
//!   fixed, distinct, validated-unique position for the *entire* debate;
//!   participants *rebut* each other across rounds over the raw multi-round
//!   transcript; and a separate judge **picks a winning position** at the
//!   end (`DebateVerdict::winning_position`). It is adversarial and
//!   selective. Mixture-of-agents is **collaborative and synthetic**:
//!   [`MoaProposer`]s have no fixed stance and are not arguing against one
//!   another, and [`MoaAggregator::aggregate`] must **merge** proposals'
//!   complementary content rather than choosing one and discarding the
//!   rest — a "pick the best proposal" aggregator would violate the trait's
//!   contract (see [`MoaAggregator`]'s documentation and the
//!   "synthesis, not selection" tests in this module's test suite, which
//!   construct proposals where two proposers each know a fact the other
//!   does not and assert the aggregate contains **both**).
//!
//! | | `self_consistency` | `multi_agent_debate` | `mixture_of_agents` |
//! |---|---|---|---|
//! | Cross-agent visibility | none, ever | full raw transcript, every round | previous layer's *synthesized* output (+ optionally its proposals) |
//! | Stance | none (same task, sampled) | fixed, adversarial, unique per participant | none (collaborative) |
//! | Combining step | majority vote / clustering over final answers | judge **selects** a winner | aggregator **synthesizes** a merged response |
//! | Structure | flat (`K` independent samples) | flat-over-rounds (same `N` participants argue every round) | layered (`L` layers, each conditioned on the last) |
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "mixture-of-agents")]
//! # {
//! use oxirag::mixture_of_agents::{
//!     MoaConfig, MoaEngine, MoaProposer, MoaSynthesisAggregator, MockMoaProposer,
//! };
//!
//! let physicist = MockMoaProposer::new("Physicist")
//!     .with_fact("Light behaves as both a wave and a particle.");
//! let biologist = MockMoaProposer::new("Biologist")
//!     .with_fact("Photosynthesis converts light into chemical energy.");
//! let proposers: [&dyn MoaProposer; 2] = [&physicist, &biologist];
//!
//! let engine = MoaEngine::new(MoaConfig::new().with_num_layers(2).with_proposers_per_layer(2));
//! let aggregator = MoaSynthesisAggregator::default();
//! let trace = engine
//!     .run("What role does light play in nature?", &proposers, &aggregator)
//!     .expect("mixture-of-agents run should succeed");
//!
//! // Synthesis, not selection: the final response retains both proposers'
//! // distinct contributions, not just whichever "won".
//! assert!(trace.final_response.to_lowercase().contains("wave"));
//! assert!(trace.final_response.to_lowercase().contains("photosynthesis"));
//! assert_eq!(trace.layers.len(), 2);
//! # }
//! ```

pub mod aggregator;
pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use aggregator::MoaSynthesisAggregator;
pub use engine::{MoaEngine, MockMoaProposer};
pub use types::{
    MoaAggregator, MoaConfig, MoaContextMode, MoaError, MoaLayer, MoaLayerStats, MoaProposer,
    MoaResponse, MoaTrace,
};
