//! Core data types, configuration, error enum, and the pluggable
//! [`MoaProposer`] / [`MoaAggregator`] traits for the `mixture_of_agents`
//! module.
//!
//! The layer-orchestration loop (round-robin proposer calls, context
//! assembly, early stopping) and the deterministic
//! [`MockMoaProposer`](crate::mixture_of_agents::MockMoaProposer)
//! implementation live in [`super::engine`]; the real synthesis algorithm
//! backing [`MoaAggregator`] — [`MoaSynthesisAggregator`](crate::mixture_of_agents::MoaSynthesisAggregator) —
//! lives in [`super::aggregator`].

use thiserror::Error;

// ── MoaError ─────────────────────────────────────────────────────────────────

/// Errors produced by the `mixture_of_agents` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MoaError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,

    /// [`MoaConfig::num_layers`] was `0`.
    #[error("num_layers must be at least 1, got 0")]
    ZeroLayers,

    /// No proposers were supplied; a mixture-of-agents run requires at
    /// least one.
    #[error("at least 1 proposer is required, got 0")]
    ZeroProposers,

    /// The number of proposers supplied to
    /// [`MoaEngine::run`](crate::mixture_of_agents::MoaEngine::run) did not
    /// match [`MoaConfig::proposers_per_layer`].
    #[error("expected {expected} proposer(s) (per MoaConfig::proposers_per_layer), got {actual}")]
    ProposerCountMismatch {
        /// The expected proposer count, from [`MoaConfig::proposers_per_layer`].
        expected: usize,
        /// The proposer count actually supplied.
        actual: usize,
    },

    /// [`MoaConfig::early_stop_similarity`] was `Some(value)` with `value`
    /// outside `[0.0, 1.0]`.
    #[error("early_stop_similarity must be within [0.0, 1.0], got {value}")]
    InvalidSimilarityThreshold {
        /// The out-of-range threshold that was supplied.
        value: f32,
    },

    /// A [`MoaProposer`] implementation failed to produce a response.
    ///
    /// Provided for custom (non-mock) implementations backed by a real
    /// model or service, so a failure (timeout, refusal, malformed
    /// response, ...) can be reported without extending this enum.
    #[error("proposer failed to produce a response: {reason}")]
    ProposerFailed {
        /// A human-readable reason for the failure.
        reason: String,
    },

    /// A [`MoaAggregator`] implementation failed to synthesize a response.
    #[error("aggregator failed to synthesize a response: {reason}")]
    AggregatorFailed {
        /// A human-readable reason for the failure.
        reason: String,
    },

    /// A [`MoaAggregator`] was asked to synthesize an empty slice of
    /// proposals.
    #[error("aggregator was called with zero proposals to synthesize")]
    EmptyProposals,
}

// ── MoaContextMode ───────────────────────────────────────────────────────────

/// Controls what a completed layer hands forward to the *next* layer's
/// proposers as `prior_responses` (see [`MoaProposer::propose`]).
///
/// This is the configurable "auxiliary context" mechanism that makes
/// mixture-of-agents a genuinely *layered* architecture rather than a single
/// flat round of independent sampling — see the
/// [module-level documentation](crate::mixture_of_agents) for why that
/// distinction matters relative to `crate::self_consistency`.
///
/// Regardless of this setting, the *aggregator* for a given layer always
/// sees every one of that layer's own proposals in full (via
/// [`MoaAggregator::aggregate`]) — `MoaContextMode` only governs what is
/// threaded forward into the *next* layer's `prior_responses`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoaContextMode {
    /// Only the completed layer's synthesized aggregate is exposed to the
    /// next layer's proposers, as a single-element `prior_responses` slice.
    AggregateOnly,
    /// Only the completed layer's individual proposals (not the aggregate)
    /// are exposed, as an `N`-element `prior_responses` slice.
    ProposalsOnly,
    /// Both the completed layer's aggregate and its individual proposals
    /// are exposed, as an `N + 1`-element `prior_responses` slice. This is
    /// the richest signal and matches how the original Mixture-of-Agents
    /// paper's reference implementations expose every prior-layer model
    /// output to the next layer. Default.
    #[default]
    AggregateAndProposals,
}

// ── MoaConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`MoaEngine`](crate::mixture_of_agents::MoaEngine).
///
/// Proposers and the aggregator are supplied per-call to
/// [`MoaEngine::run`](crate::mixture_of_agents::MoaEngine::run), not stored
/// here — mirroring the caller-supplies-executor convention used throughout
/// this crate (e.g. `crate::multi_agent_debate::DebateEngine::run`).
#[derive(Debug, Clone, PartialEq)]
pub struct MoaConfig {
    /// Number of layers to run. Layer `0` always runs with empty
    /// `prior_responses`; every subsequent layer's proposers see the
    /// previous layer's context per [`MoaConfig::context_mode`]. Must be at
    /// least `1`. Defaults to `3`.
    pub num_layers: usize,
    /// The number of proposers that must be supplied to
    /// [`MoaEngine::run`](crate::mixture_of_agents::MoaEngine::run) — every
    /// layer calls exactly this many proposers. Defaults to `3`.
    pub proposers_per_layer: usize,
    /// What a completed layer exposes to the next layer's proposers. See
    /// [`MoaContextMode`]. Defaults to
    /// [`MoaContextMode::AggregateAndProposals`].
    pub context_mode: MoaContextMode,
    /// When `Some(threshold)`, the run may stop before `num_layers` once a
    /// layer's aggregate is at least `threshold`-similar (lexical-overlap
    /// based; see [`MoaLayerStats::change_from_previous`]) to the previous
    /// layer's aggregate — i.e. the synthesis has stopped changing
    /// materially. Must be within `[0.0, 1.0]`. `None` disables early
    /// stopping. Defaults to `None`.
    pub early_stop_similarity: Option<f32>,
}

impl Default for MoaConfig {
    fn default() -> Self {
        Self {
            num_layers: 3,
            proposers_per_layer: 3,
            context_mode: MoaContextMode::default(),
            early_stop_similarity: None,
        }
    }
}

impl MoaConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of layers.
    #[must_use]
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set the number of proposers per layer.
    #[must_use]
    pub fn with_proposers_per_layer(mut self, proposers_per_layer: usize) -> Self {
        self.proposers_per_layer = proposers_per_layer;
        self
    }

    /// Set the inter-layer context mode.
    #[must_use]
    pub fn with_context_mode(mut self, context_mode: MoaContextMode) -> Self {
        self.context_mode = context_mode;
        self
    }

    /// Enable convergence-based early stopping with the given similarity
    /// threshold (`[0.0, 1.0]`).
    #[must_use]
    pub fn with_early_stop_similarity(mut self, threshold: f32) -> Self {
        self.early_stop_similarity = Some(threshold);
        self
    }

    /// Disable early stopping (the default).
    #[must_use]
    pub fn without_early_stop(mut self) -> Self {
        self.early_stop_similarity = None;
        self
    }
}

// ── MoaResponse ──────────────────────────────────────────────────────────────

/// A single piece of text produced somewhere in a mixture-of-agents run:
/// either one proposer's individual proposal, or a layer's synthesized
/// aggregate.
///
/// Bookkeeping fields (`proposer_id`, `layer`, `is_aggregate`) are always
/// owned and set by [`MoaEngine`](crate::mixture_of_agents::MoaEngine)
/// itself — [`MoaProposer::propose`] and [`MoaAggregator::aggregate`]
/// implementations only return the response text (`String`); the engine
/// wraps it into a `MoaResponse` with ground-truth bookkeeping afterward
/// (mirroring how `crate::multi_agent_debate::DebateEngine` owns
/// `DebateArgument`'s identity fields rather than trusting a persona to get
/// them right).
#[derive(Debug, Clone, PartialEq)]
pub struct MoaResponse {
    /// Zero-based index, into the proposer slice a layer was run with, of
    /// the proposer that produced this response. For an aggregate response
    /// (`is_aggregate == true`), this is `usize::MAX` — an aggregate is not
    /// attributable to any single proposer slot.
    pub proposer_id: usize,
    /// Zero-based layer index this response was produced in.
    pub layer: usize,
    /// The response text.
    pub text: String,
    /// `true` when this is a layer's synthesized aggregate (produced by a
    /// [`MoaAggregator`]); `false` when this is one proposer's individual
    /// proposal (produced by a [`MoaProposer`]).
    pub is_aggregate: bool,
}

impl MoaResponse {
    /// Build an individual proposer's response.
    #[must_use]
    pub fn proposal(proposer_id: usize, layer: usize, text: impl Into<String>) -> Self {
        Self {
            proposer_id,
            layer,
            text: text.into(),
            is_aggregate: false,
        }
    }

    /// Build a layer's synthesized aggregate response.
    #[must_use]
    pub fn aggregate(layer: usize, text: impl Into<String>) -> Self {
        Self {
            proposer_id: usize::MAX,
            layer,
            text: text.into(),
            is_aggregate: true,
        }
    }
}

// ── MoaLayerStats ────────────────────────────────────────────────────────────

/// Agreement, coverage, and change signals for a single completed layer, as
/// computed by [`MoaEngine`](crate::mixture_of_agents::MoaEngine).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoaLayerStats {
    /// The layer this statistic belongs to.
    pub layer_index: usize,
    /// Mean pairwise lexical-overlap (Jaccard, over content terms) agreement
    /// across this layer's individual proposals, in `[0.0, 1.0]`. `1.0` when
    /// fewer than two proposals were produced (nothing to disagree with).
    pub agreement: f32,
    /// The number of distinct sentence-level claims found across this
    /// layer's proposals, after lexical-overlap clustering (see
    /// [`crate::mixture_of_agents::MoaSynthesisAggregator`]'s
    /// documentation for the clustering procedure this reuses).
    pub distinct_claims: usize,
    /// Of `distinct_claims`, how many have at least one sentence in the
    /// layer's aggregate whose content-term overlap with the claim clears
    /// the same clustering threshold — i.e. how many distinct claims the
    /// aggregate actually retained.
    pub covered_claims: usize,
    /// `covered_claims / distinct_claims`, in `[0.0, 1.0]`. `1.0` when
    /// `distinct_claims` is `0` (nothing to have dropped). A synthesizing
    /// aggregator (like [`crate::mixture_of_agents::MoaSynthesisAggregator`])
    /// should keep this at (or very near) `1.0`; a *selecting* aggregator
    /// that discards minority claims would show a materially lower value.
    pub coverage_ratio: f32,
    /// How much this layer's aggregate differs from the *previous* layer's
    /// aggregate: `1.0 - jaccard(content_terms(previous), content_terms(current))`,
    /// in `[0.0, 1.0]` (`0.0` = identical content-term sets, `1.0` =
    /// completely disjoint). `None` for layer `0`, which has no previous
    /// layer to compare against. This is the signal
    /// [`MoaConfig::early_stop_similarity`] is checked against (as
    /// `1.0 - change_from_previous`).
    pub change_from_previous: Option<f32>,
}

// ── MoaLayer / MoaTrace ──────────────────────────────────────────────────────

/// One completed layer of a mixture-of-agents run: every proposer's
/// individual proposal, the aggregator's synthesis of them, and this
/// layer's [`MoaLayerStats`].
#[derive(Debug, Clone, PartialEq)]
pub struct MoaLayer {
    /// Zero-based layer index.
    pub layer_index: usize,
    /// Every proposer's individual proposal for this layer, in
    /// proposer-slice order (i.e. `proposals[i].proposer_id == i`).
    pub proposals: Vec<MoaResponse>,
    /// The aggregator's synthesis of `proposals` for this layer.
    pub aggregate: MoaResponse,
    /// Agreement/coverage/change statistics for this layer.
    pub stats: MoaLayerStats,
}

/// The complete output of a
/// [`MoaEngine::run`](crate::mixture_of_agents::MoaEngine::run) call: every
/// layer that was actually run, in order, together with the final
/// synthesized response.
#[derive(Debug, Clone, PartialEq)]
pub struct MoaTrace {
    /// The original query.
    pub query: String,
    /// Every layer that was run, in layer order. Always non-empty for a
    /// successful run (a run either errors before layer `0` or produces at
    /// least one layer).
    pub layers: Vec<MoaLayer>,
    /// The final response: the last layer's aggregate text. Shortcut for
    /// `layers.last().unwrap().aggregate.text`.
    pub final_response: String,
    /// Number of layers actually run. Less than or equal to
    /// [`MoaConfig::num_layers`]; strictly less than it only when early
    /// stopping fired.
    pub layers_run: usize,
    /// `true` when the run stopped before [`MoaConfig::num_layers`] because
    /// the early-stop convergence signal fired.
    pub stopped_early: bool,
    /// A human-readable explanation of why the run stopped early, or `None`
    /// when it was not stopped early.
    pub early_stop_reason: Option<String>,
}

impl MoaTrace {
    /// The last layer that was run, if any.
    #[must_use]
    pub fn final_layer(&self) -> Option<&MoaLayer> {
        self.layers.last()
    }

    /// The layer at `index`, if it was run.
    #[must_use]
    pub fn layer(&self, index: usize) -> Option<&MoaLayer> {
        self.layers.get(index)
    }
}

// ── MoaProposer ──────────────────────────────────────────────────────────────

/// A pluggable Layer-1..L proposer in a mixture-of-agents run.
///
/// Implementations produce one candidate response per call, given the query
/// and (from layer `1` onward) `prior_responses` — the previous layer's
/// context, shaped per [`MoaConfig::context_mode`]. In layer `0`,
/// `prior_responses` is always empty: layer `0` proposers answer completely
/// independently, exactly like `crate::self_consistency`'s sampled reasoning
/// chains. What makes mixture-of-agents structurally different is layer `1`
/// onward: a proposer there can see — and is expected to build on — the
/// *synthesized* output (and/or individual proposals, depending on
/// [`MoaConfig::context_mode`]) of the *previous* layer, which
/// `self_consistency` chains never do (see the
/// [module-level documentation](crate::mixture_of_agents)).
///
/// A `MoaProposer` has no fixed adversarial stance (unlike
/// `crate::multi_agent_debate::DebatePersona`, which is permanently bound to
/// one position for an entire debate) — it is simply asked to produce the
/// best response it can, optionally informed by what came before.
///
/// [`MockMoaProposer`](crate::mixture_of_agents::MockMoaProposer) provides a
/// deterministic implementation for tests and examples.
pub trait MoaProposer {
    /// Produce this proposer's candidate response.
    ///
    /// `prior_responses` holds the previous layer's context (empty in layer
    /// `0`) — see the trait-level documentation and [`MoaContextMode`].
    ///
    /// # Errors
    ///
    /// Implementations may return [`MoaError::ProposerFailed`] (or any
    /// other [`MoaError`] variant that genuinely applies) when they cannot
    /// produce a response.
    fn propose(&self, query: &str, prior_responses: &[MoaResponse]) -> Result<String, MoaError>;
}

// ── MoaAggregator ────────────────────────────────────────────────────────────

/// A pluggable aggregator that synthesizes one layer's proposals into a
/// single improved response.
///
/// The defining contract of a `MoaAggregator` is **synthesis, not
/// selection**: given `N` proposals, it must not simply pick the "best" one
/// and discard the rest (that would make this module a variant of
/// `crate::multi_agent_debate`'s judge, which explicitly *does* pick a
/// winner). Instead it must merge the proposals' complementary informational
/// content — a claim asserted by only one proposal should still surface in
/// the output if it is substantive, not silently dropped because another
/// proposal "won". See the [module-level documentation](crate::mixture_of_agents)
/// for why this distinction is the core of what makes mixture-of-agents a
/// distinct pattern.
///
/// [`MoaSynthesisAggregator`](crate::mixture_of_agents::MoaSynthesisAggregator)
/// provides a genuine sentence-level synthesis algorithm (deterministic, no
/// live model) satisfying this contract, and is the recommended default.
pub trait MoaAggregator {
    /// Synthesize `proposals` (one layer's individual proposer outputs)
    /// into a single response.
    ///
    /// # Errors
    ///
    /// Implementations may return [`MoaError::AggregatorFailed`],
    /// [`MoaError::EmptyProposals`], or any other [`MoaError`] variant that
    /// genuinely applies.
    fn aggregate(&self, query: &str, proposals: &[MoaResponse]) -> Result<String, MoaError>;
}
