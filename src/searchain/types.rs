//! Types and traits for the `searchain` module.
//!
//! `SearChain` (Xu et al. 2024, "Search-in-the-Chain: Interactively Enhancing
//! Large Language Models with Search for Knowledge-intensive Tasks") is a
//! two-phase, backtracking pipeline:
//!
//! - **Phase 1 — Chain-of-Query construction.** A [`ChainGenerator`] builds
//!   the *complete* global reasoning chain upfront, before any retrieval
//!   happens: a sequence of [`ChainNode`]s, each carrying a sub-query and a
//!   tentative (parametric-knowledge-only) answer, plus the ids of earlier
//!   nodes it causally depends on.
//! - **Phase 2 — Interactive Reasoning-Verification (IRV).** A [`Retriever`]
//!   fetches evidence for every node's sub-query; the evidence is compared
//!   against the node's current answer, producing a [`NodeVerdict`]. A
//!   [`NodeVerdict::Conflicting`] verdict triggers *backtracking*: the node's
//!   answer is revised from evidence, and every node that causally depends on
//!   it is explicitly re-verified, since its own tentative answer may have
//!   been built on the now-invalidated premise.
//! - **Phase 3 — Final-answer assembly.** A [`Generator`] synthesises the
//!   final answer from the verified chain.
//!
//! [`MockChainGenerator`], [`MockSearchainRetriever`], and
//! [`MockSearchainGenerator`] provide deterministic implementations of the
//! three pluggable traits, for use in tests.

use thiserror::Error;

// ── NodeVerdict ──────────────────────────────────────────────────────────────

/// The outcome of comparing a chain node's current answer against freshly
/// retrieved evidence during Interactive Reasoning-Verification (IRV).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeVerdict {
    /// The retrieved evidence supports the node's current answer; the answer
    /// is kept unchanged.
    Verified,
    /// The retrieved evidence is insufficient or inconclusive: it neither
    /// clearly supports nor contradicts the current answer. The answer is
    /// replaced with the best answer extractable from the evidence (when
    /// evidence was returned at all).
    Unverified,
    /// The retrieved evidence directly contradicts the node's current
    /// answer. The answer is replaced with evidence, and every node that
    /// causally depends on this node is scheduled for re-verification
    /// (backtracking).
    Conflicting,
}

impl NodeVerdict {
    /// Return a stable lowercase string representation of the verdict.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Unverified => "unverified",
            Self::Conflicting => "conflicting",
        }
    }
}

// ── ChainNode ────────────────────────────────────────────────────────────────

/// A single node in a `SearChain` chain-of-query.
///
/// Produced upfront (Phase 1, before any retrieval) by a [`ChainGenerator`]:
/// `sub_query` is the question this node answers, `tentative_answer` is the
/// parametric-knowledge guess for it (no retrieval involved yet), and
/// `depends_on` lists the ids of earlier nodes this node's premise was built
/// on. During Phase 2 (Interactive Reasoning-Verification), the engine fills
/// in `verdict` and `evidence`, and — whenever the tentative answer does not
/// hold up, or a dependency is later invalidated — overwrites `final_answer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainNode {
    /// Zero-based identifier; must equal this node's position in the chain.
    pub id: usize,
    /// The sub-question this node answers.
    pub sub_query: String,
    /// The initial, parametric-knowledge-only answer produced during chain
    /// construction (Phase 1), before any retrieval has occurred. Immutable
    /// after construction — used as the stable reference text that
    /// downstream nodes' tentative answers may textually embed.
    pub tentative_answer: String,
    /// Ids of earlier nodes (`id` values strictly less than this node's
    /// `id`) whose answers this node's sub-query or tentative answer was
    /// built on.
    pub depends_on: Vec<usize>,
    /// The node's current best answer. Starts equal to `tentative_answer`
    /// and is overwritten during Phase 2 whenever the verdict is not
    /// `Verified`, or when backtracking patches it after an ancestor node
    /// was revised.
    pub final_answer: String,
    /// Evidence passages retrieved for `sub_query` during the most recent
    /// verification of this node. Empty until the node has been verified at
    /// least once.
    pub evidence: Vec<String>,
    /// The verdict from the most recent verification, or `None` before
    /// Phase 2 has visited this node.
    pub verdict: Option<NodeVerdict>,
    /// `true` once `final_answer` has diverged from `tentative_answer`,
    /// whether due to this node's own verification or a backtracking patch
    /// propagated from an ancestor.
    pub revised: bool,
}

impl ChainNode {
    /// Construct a new node with no dependencies.
    ///
    /// `final_answer` starts equal to `tentative_answer`; `evidence` starts
    /// empty, `verdict` starts `None`, and `revised` starts `false`.
    #[must_use]
    pub fn new(
        id: usize,
        sub_query: impl Into<String>,
        tentative_answer: impl Into<String>,
    ) -> Self {
        let tentative_answer = tentative_answer.into();
        Self {
            id,
            sub_query: sub_query.into(),
            final_answer: tentative_answer.clone(),
            tentative_answer,
            depends_on: Vec::new(),
            evidence: Vec::new(),
            verdict: None,
            revised: false,
        }
    }

    /// Attach dependency node ids.
    #[must_use]
    pub fn with_depends_on(mut self, depends_on: Vec<usize>) -> Self {
        self.depends_on = depends_on;
        self
    }

    /// Return `true` when this node has no dependencies (a chain root).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.depends_on.is_empty()
    }

    /// Return `true` when this node's most recent verdict is
    /// [`NodeVerdict::Conflicting`].
    #[must_use]
    pub fn is_conflicting(&self) -> bool {
        matches!(self.verdict, Some(NodeVerdict::Conflicting))
    }

    /// Return `true` when this node has reached a terminal (non-conflicting)
    /// verdict — [`NodeVerdict::Verified`] or [`NodeVerdict::Unverified`].
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.verdict,
            Some(NodeVerdict::Verified | NodeVerdict::Unverified)
        )
    }
}

// ── ChainGenerator ───────────────────────────────────────────────────────────

/// Phase 1: builds the complete chain-of-query for a complex `query`, upfront
/// and with no retrieval involved.
///
/// Implementations draw only on parametric knowledge to produce every node's
/// `sub_query` and `tentative_answer` in a single pass, before Phase 2
/// (Interactive Reasoning-Verification) begins. [`MockChainGenerator`] is
/// provided for tests.
pub trait ChainGenerator {
    /// Generate the full reasoning chain for `query`.
    ///
    /// Implementations should return at least one node, with ids
    /// `0, 1, 2, ...` matching each node's position in the returned vector;
    /// [`crate::searchain::SearChainEngine::build_chain`] validates this.
    fn generate_chain(&self, query: &str) -> Vec<ChainNode>;
}

// ── Retriever ─────────────────────────────────────────────────────────────────

/// Phase 2: retrieves evidence passages for a chain node's sub-query.
///
/// Implementations may wrap a vector store, a web search API, or any other
/// backend. [`MockSearchainRetriever`] is provided for tests.
pub trait Retriever {
    /// Retrieve evidence passages relevant to `sub_query`, ranked
    /// best-first. An empty vector signals "no supporting evidence found".
    fn retrieve(&self, sub_query: &str) -> Vec<String>;
}

// ── Generator ─────────────────────────────────────────────────────────────────

/// Phase 3: assembles the final answer to the original query from the
/// verified (and possibly backtracking-revised) chain.
///
/// [`MockSearchainGenerator`] is provided for tests.
pub trait Generator {
    /// Synthesise the final answer from `original_query` and the fully
    /// verified `chain`.
    fn generate(&self, original_query: &str, chain: &[ChainNode]) -> String;
}

// ── MockChainGenerator ────────────────────────────────────────────────────────

/// Deterministic [`ChainGenerator`] for tests.
///
/// Returns the chain of the first `(query substring, chain)` mapping whose
/// substring is contained in the query. When nothing matches, a trivial
/// single-node fallback chain is returned whose sub-query and tentative
/// answer both equal the query text, so `generate_chain` never returns an
/// empty chain.
#[derive(Debug, Clone, Default)]
pub struct MockChainGenerator {
    /// Ordered `(query substring, chain)` mappings.
    pub mappings: Vec<(String, Vec<ChainNode>)>,
}

impl MockChainGenerator {
    /// Create a mock generator from `(query substring, chain)` mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, Vec<ChainNode>)>) -> Self {
        Self { mappings }
    }
}

impl ChainGenerator for MockChainGenerator {
    fn generate_chain(&self, query: &str) -> Vec<ChainNode> {
        for (needle, chain) in &self.mappings {
            if query.contains(needle.as_str()) {
                return chain.clone();
            }
        }
        vec![ChainNode::new(0, query, query)]
    }
}

// ── MockSearchainRetriever ──────────────────────────────────────────────────

/// Deterministic [`Retriever`] for tests.
///
/// Returns the evidence of the first `(sub-query substring, evidence)`
/// mapping whose substring is contained in the sub-query. When nothing
/// matches, an empty evidence list is returned, modelling "no supporting
/// evidence found" (which the engine treats as [`NodeVerdict::Unverified`]).
#[derive(Debug, Clone, Default)]
pub struct MockSearchainRetriever {
    /// Ordered `(sub-query substring, evidence passages)` mappings.
    pub mappings: Vec<(String, Vec<String>)>,
}

impl MockSearchainRetriever {
    /// Create a mock retriever from `(sub-query substring, evidence)`
    /// mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, Vec<String>)>) -> Self {
        Self { mappings }
    }
}

impl Retriever for MockSearchainRetriever {
    fn retrieve(&self, sub_query: &str) -> Vec<String> {
        for (needle, evidence) in &self.mappings {
            if sub_query.contains(needle.as_str()) {
                return evidence.clone();
            }
        }
        Vec::new()
    }
}

// ── AnswerAssemblyStrategy / MockSearchainGenerator ─────────────────────────

/// Strategy used by [`MockSearchainGenerator`] to assemble the final answer
/// from a verified chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnswerAssemblyStrategy {
    /// Use the last node's `final_answer` as the overall answer — the common
    /// case, since the last sub-query in a chain-of-query typically resolves
    /// the original complex query.
    #[default]
    LastNode,
    /// Join every node's `final_answer` with `"; "`.
    JoinAll,
}

/// Deterministic [`Generator`] for tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct MockSearchainGenerator {
    /// The assembly strategy to apply.
    pub strategy: AnswerAssemblyStrategy,
}

impl MockSearchainGenerator {
    /// Create a mock generator using the given assembly strategy.
    #[must_use]
    pub fn new(strategy: AnswerAssemblyStrategy) -> Self {
        Self { strategy }
    }
}

impl Generator for MockSearchainGenerator {
    fn generate(&self, _original_query: &str, chain: &[ChainNode]) -> String {
        match self.strategy {
            AnswerAssemblyStrategy::LastNode => chain
                .last()
                .map(|n| n.final_answer.clone())
                .unwrap_or_default(),
            AnswerAssemblyStrategy::JoinAll => chain
                .iter()
                .map(|n| n.final_answer.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        }
    }
}

// ── SearChainConfig ──────────────────────────────────────────────────────────

/// Configuration for [`crate::searchain::SearChainEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearChainConfig {
    /// Minimum number of shared content terms required before a negation,
    /// numeric, or temporal mismatch between an answer and its evidence is
    /// treated as a contradiction (a subject-overlap gate that avoids
    /// flagging coincidental negations about unrelated subjects). Defaults
    /// to `2`.
    pub min_shared_terms: usize,
    /// Minimum fraction (in `[0.0, 1.0]`) of an answer's content terms that
    /// must also appear in the evidence for the evidence to count as
    /// *supporting* the answer. Defaults to `0.5`.
    pub min_support_overlap: f32,
    /// Maximum number of backtracking-triggered re-verifications performed
    /// across an entire Interactive Reasoning-Verification run. Guards
    /// against unbounded cascades; once exceeded, verification stops early
    /// and the result carries
    /// [`SearChainStatus::PartialBacktrackExhausted`]. Defaults to `8`.
    pub max_backtrack_iterations: usize,
}

impl Default for SearChainConfig {
    fn default() -> Self {
        Self {
            min_shared_terms: 2,
            min_support_overlap: 0.5,
            max_backtrack_iterations: 8,
        }
    }
}

impl SearChainConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum shared-term gate for contradiction detection.
    #[must_use]
    pub fn with_min_shared_terms(mut self, min_shared_terms: usize) -> Self {
        self.min_shared_terms = min_shared_terms;
        self
    }

    /// Set the minimum support-overlap ratio required for
    /// [`NodeVerdict::Verified`].
    #[must_use]
    pub fn with_min_support_overlap(mut self, min_support_overlap: f32) -> Self {
        self.min_support_overlap = min_support_overlap;
        self
    }

    /// Set the maximum number of backtracking-triggered re-verifications.
    #[must_use]
    pub fn with_max_backtrack_iterations(mut self, max_backtrack_iterations: usize) -> Self {
        self.max_backtrack_iterations = max_backtrack_iterations;
        self
    }
}

// ── SearChainStatus ──────────────────────────────────────────────────────────

/// The termination status of an Interactive Reasoning-Verification (IRV)
/// run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearChainStatus {
    /// Every node reached a terminal (non-conflicting) verdict and every
    /// backtracking cascade fully resolved.
    Complete,
    /// The [`SearChainConfig::max_backtrack_iterations`] budget was
    /// exhausted before every cascade could be resolved. `unresolved_node_ids`
    /// honestly lists every node that never got a chance to be re-verified
    /// against a corrected upstream premise, and is therefore not
    /// trustworthy as-is.
    PartialBacktrackExhausted {
        /// Ids of nodes that never reached a settled, re-verified state,
        /// including the transitive dependents of any node that was left
        /// unresolved.
        unresolved_node_ids: Vec<usize>,
    },
}

impl SearChainStatus {
    /// Return `true` for [`SearChainStatus::Complete`].
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }
}

// ── SearChainResult ──────────────────────────────────────────────────────────

/// The complete output of a [`crate::searchain::SearChainEngine`] run: the
/// original query, the fully processed chain, the assembled final answer,
/// and how the Interactive Reasoning-Verification pass terminated.
#[derive(Debug, Clone, PartialEq)]
pub struct SearChainResult {
    /// The original complex query.
    pub original_query: String,
    /// The chain after Phase 1 construction and Phase 2 verification.
    pub chain: Vec<ChainNode>,
    /// The synthesised final answer (Phase 3).
    pub final_answer: String,
    /// How the Interactive Reasoning-Verification pass terminated.
    pub status: SearChainStatus,
    /// Total number of backtracking-triggered re-verifications performed.
    pub backtrack_events: usize,
}

impl SearChainResult {
    /// Return `true` when [`SearChainResult::status`] is
    /// [`SearChainStatus::Complete`].
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.status.is_complete()
    }

    /// Return the ids of every node whose `final_answer` diverged from its
    /// `tentative_answer`, in ascending order.
    #[must_use]
    pub fn revised_node_ids(&self) -> Vec<usize> {
        self.chain
            .iter()
            .filter(|n| n.revised)
            .map(|n| n.id)
            .collect()
    }

    /// Return the ids of every node left unresolved when the backtracking
    /// budget was exhausted, or an empty slice when the run completed.
    #[must_use]
    pub fn unresolved_node_ids(&self) -> &[usize] {
        match &self.status {
            SearChainStatus::Complete => &[],
            SearChainStatus::PartialBacktrackExhausted {
                unresolved_node_ids,
            } => unresolved_node_ids,
        }
    }
}

// ── SearChainError ───────────────────────────────────────────────────────────

/// Errors from the `searchain` module.
#[derive(Debug, Error)]
pub enum SearChainError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The [`ChainGenerator`] produced a chain with zero nodes.
    #[error("chain generator produced an empty chain")]
    EmptyChain,
    /// A node's ids were not sequential starting at zero.
    #[error(
        "chain node ids must be sequential starting at 0: expected id {expected}, found {found}"
    )]
    NonSequentialIds {
        /// The id expected at this position (the position's index).
        expected: usize,
        /// The id actually found.
        found: usize,
    },
    /// A node declared a dependency on itself or a later node, which would
    /// make the dependency non-causal.
    #[error("node {node_id} declares a dependency on node {dep_id}, which is not an earlier node")]
    InvalidDependency {
        /// The id of the node declaring the invalid dependency.
        node_id: usize,
        /// The id of the (non-earlier) node it depends on.
        dep_id: usize,
    },
}
