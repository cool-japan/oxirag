//! Core types for the self-taught reasoner: problems, generations, kept
//! rationales, the accumulated bootstrapped set, per-round statistics, the run
//! outcome, configuration, and errors.

use std::collections::{BTreeSet, HashSet};

use thiserror::Error;

// ── Problem ────────────────────────────────────────────────────────────────────

/// A problem the reasoner must solve: a statement and the gold answer that is
/// both the correctness oracle and — for problems the forward pass fails — the
/// hint fed to backward rationalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarProblem {
    /// Stable identifier, used to deduplicate the accumulated set and to key the
    /// per-round solved-id records.
    pub id: String,
    /// The problem statement shown to the model.
    pub statement: String,
    /// The known-correct answer.
    pub gold_answer: String,
}

impl StarProblem {
    /// Create a problem.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        statement: impl Into<String>,
        gold_answer: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            statement: statement.into(),
            gold_answer: gold_answer.into(),
        }
    }
}

// ── Generation ─────────────────────────────────────────────────────────────────

/// One rationale + answer a model produced for a problem — the output of both
/// forward generation and backward rationalization.
///
/// It carries no provenance and no problem id: whether a generation becomes a
/// *kept* [`StarRationale`], and with which [`RationaleSource`], is the engine's
/// decision, taken after the correctness filter (and, for rationalizations, the
/// cheat check) have run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarGeneration {
    /// The reasoning trace the model produced.
    pub rationale: String,
    /// The final answer the reasoning arrived at.
    pub answer: String,
}

impl StarGeneration {
    /// Create a generation.
    #[must_use]
    pub fn new(rationale: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            rationale: rationale.into(),
            answer: answer.into(),
        }
    }
}

// ── Kept rationale ─────────────────────────────────────────────────────────────

/// How a kept rationale entered the bootstrapped set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RationaleSource {
    /// Produced by forward generation and kept because its answer was correct.
    Forward,
    /// Produced by backward rationalization from the gold answer, and kept
    /// because it independently reached that answer *without cheating*.
    Rationalized,
}

impl RationaleSource {
    /// Short human-readable label.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Rationalized => "rationalized",
        }
    }
}

/// A kept `(problem, rationale, answer)` triple in the bootstrapped training set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StarRationale {
    /// The problem this rationale solves.
    pub problem_id: String,
    /// The problem statement, retained so the fine-tuning step has the full
    /// training example without a back-reference to the problem list.
    pub statement: String,
    /// The reasoning trace.
    pub rationale: String,
    /// The answer the rationale reached — always equivalent to the gold answer,
    /// because that is the condition under which the engine kept it.
    pub answer: String,
    /// Whether this rationale came from forward generation or backward
    /// rationalization.
    pub source: RationaleSource,
}

// ── Accumulated set ─────────────────────────────────────────────────────────────

/// The accumulated bootstrapped training set: the kept rationales that the
/// generator is fine-tuned on, holding **at most one rationale per problem id**.
///
/// The dedup-by-problem-id policy is what guarantees the loop terminates: the set
/// only ever grows, and it is bounded by the number of problems, so it must reach
/// a fixed point in at most `problems + 1` rounds (see [`crate::self_taught_reasoner`]).
/// The first kept rationale for a problem wins; a later round that re-solves the
/// same problem does not displace it, keeping the set's growth monotone and its
/// contents stable.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RationaleSet {
    entries: Vec<StarRationale>,
    ids: HashSet<String>,
}

impl RationaleSet {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of kept rationales.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a rationale for `problem_id` is present.
    #[must_use]
    pub fn contains(&self, problem_id: &str) -> bool {
        self.ids.contains(problem_id)
    }

    /// The kept rationale for `problem_id`, if any.
    #[must_use]
    pub fn get(&self, problem_id: &str) -> Option<&StarRationale> {
        self.entries.iter().find(|r| r.problem_id == problem_id)
    }

    /// The kept rationales, in insertion order.
    #[must_use]
    pub fn rationales(&self) -> &[StarRationale] {
        &self.entries
    }

    /// The set of problem ids that have a kept rationale, sorted.
    #[must_use]
    pub fn problem_ids(&self) -> BTreeSet<String> {
        self.entries.iter().map(|r| r.problem_id.clone()).collect()
    }

    /// How many kept rationales came from the given source.
    #[must_use]
    pub fn count_by_source(&self, source: RationaleSource) -> usize {
        self.entries.iter().filter(|r| r.source == source).count()
    }

    /// Insert `rationale` unless its problem id is already present.
    ///
    /// Returns `true` when the rationale was newly inserted, `false` when a
    /// rationale for that problem was already in the set (in which case the set
    /// is unchanged — the first-seen rationale is retained).
    pub fn insert(&mut self, rationale: StarRationale) -> bool {
        if self.ids.contains(&rationale.problem_id) {
            return false;
        }
        self.ids.insert(rationale.problem_id.clone());
        self.entries.push(rationale);
        true
    }
}

// ── Per-round record ────────────────────────────────────────────────────────────

/// A record of one bootstrapping round.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapRound {
    /// Zero-based round index.
    pub round_index: usize,
    /// **Forward-generation accuracy at the start of this round**, i.e. before
    /// this round's fine-tuning — the fraction of problems the generator solved
    /// on its own. This is the headline "did bootstrapping help?" curve: it
    /// reflects everything the model learned in rounds `0..round_index`.
    pub forward_accuracy: f64,
    /// How many problems the forward pass solved this round.
    pub forward_solved: usize,
    /// How many problems backward rationalization solved this round.
    pub rationalized_solved: usize,
    /// Kept rationales produced this round (forward + rationalized, before
    /// deduplication against the accumulated set).
    pub kept_this_round: usize,
    /// How many of `kept_this_round` were *new* to the accumulated set.
    pub newly_accumulated: usize,
    /// The accumulated set's size after this round.
    pub accumulated_size: usize,
    /// Problem ids the forward pass solved this round, sorted.
    pub forward_solved_ids: Vec<String>,
    /// Problem ids backward rationalization solved this round, sorted.
    pub rationalized_solved_ids: Vec<String>,
    /// Rationalizations rejected because they merely echoed the hint.
    pub cheats_rejected: usize,
    /// Generations (forward or rationalized) rejected because their answer was
    /// wrong.
    pub incorrect_rejected: usize,
}

// ── Outcome ─────────────────────────────────────────────────────────────────────

/// The full outcome of a `STaR` run.
#[derive(Debug, Clone, PartialEq)]
pub struct StarOutcome {
    /// One record per round, in order. The last round is the one at which the
    /// accumulated set stopped growing (when `converged`) or the round cap was
    /// reached.
    pub rounds: Vec<BootstrapRound>,
    /// The accumulated bootstrapped training set at the fixed point.
    pub final_set: RationaleSet,
    /// Whether the loop reached a fixed point (the accumulated set stopped
    /// growing) rather than hitting the round cap.
    pub converged: bool,
    /// The number of problems the run was given.
    pub total_problems: usize,
}

impl StarOutcome {
    /// The per-round forward-accuracy curve — one entry per round.
    #[must_use]
    pub fn accuracy_curve(&self) -> Vec<f64> {
        self.rounds.iter().map(|r| r.forward_accuracy).collect()
    }

    /// Forward accuracy at the final recorded round.
    #[must_use]
    pub fn final_forward_accuracy(&self) -> f64 {
        self.rounds.last().map_or(0.0, |r| r.forward_accuracy)
    }

    /// The fraction of problems that ended up with a kept rationale — the
    /// coverage of the bootstrapped set over the problem list.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn final_coverage(&self) -> f64 {
        if self.total_problems == 0 {
            return 0.0;
        }
        self.final_set.len() as f64 / self.total_problems as f64
    }

    /// The ids of every problem that ended up solved (has a kept rationale),
    /// sorted.
    #[must_use]
    pub fn solved_problem_ids(&self) -> BTreeSet<String> {
        self.final_set.problem_ids()
    }

    /// The number of rounds the loop ran.
    #[must_use]
    pub fn num_rounds(&self) -> usize {
        self.rounds.len()
    }
}

// ── Config ─────────────────────────────────────────────────────────────────────

/// Configuration for [`SelfTaughtReasoner`](crate::self_taught_reasoner::SelfTaughtReasoner).
#[derive(Debug, Clone, PartialEq)]
pub struct StarConfig {
    /// Hard cap on the number of bootstrapping rounds. The loop normally stops
    /// earlier, when the accumulated set stops growing. Defaults to `32`.
    pub max_rounds: usize,
    /// Token-set Jaccard threshold above which two answers count as equivalent,
    /// used by the correctness oracle. Defaults to `0.8`.
    pub equivalence_threshold: f32,
    /// Whether to run the backward-rationalization pass. Turning this off yields
    /// the *no-rationalization* ablation baseline. Defaults to `true`.
    pub use_rationalization: bool,
    /// Whether to shuffle the problem processing order each round (deterministic,
    /// from [`Self::seed`]). Defaults to `false`, giving a fixed `0..n` order.
    pub shuffle_each_round: bool,
    /// Seed for the per-round problem-order shuffle. Defaults to `0`.
    pub seed: u64,
}

impl Default for StarConfig {
    fn default() -> Self {
        Self {
            max_rounds: 32,
            equivalence_threshold: 0.8,
            use_rationalization: true,
            shuffle_each_round: false,
            seed: 0,
        }
    }
}

impl StarConfig {
    /// A config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the round cap.
    #[must_use]
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = max_rounds;
        self
    }

    /// Set the answer-equivalence threshold.
    #[must_use]
    pub fn with_equivalence_threshold(mut self, threshold: f32) -> Self {
        self.equivalence_threshold = threshold;
        self
    }

    /// Enable or disable backward rationalization.
    #[must_use]
    pub fn with_rationalization(mut self, use_rationalization: bool) -> Self {
        self.use_rationalization = use_rationalization;
        self
    }

    /// Enable or disable per-round problem-order shuffling.
    #[must_use]
    pub fn with_shuffle_each_round(mut self, shuffle: bool) -> Self {
        self.shuffle_each_round = shuffle;
        self
    }

    /// Set the shuffle seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Reject a configuration the loop cannot run with.
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidConfig`] when `max_rounds` is zero or the
    /// equivalence threshold is not a finite value in `[0, 1]`.
    pub fn validate(&self) -> Result<(), StarError> {
        if self.max_rounds == 0 {
            return Err(StarError::InvalidConfig {
                reason: "max_rounds must be at least 1".to_string(),
            });
        }
        if !self.equivalence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.equivalence_threshold)
        {
            return Err(StarError::InvalidConfig {
                reason: format!(
                    "equivalence_threshold must be finite and in [0, 1], got {}",
                    self.equivalence_threshold
                ),
            });
        }
        Ok(())
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────────

/// Everything that can go wrong when running the bootstrapping loop.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StarError {
    /// The loop was given no problems to bootstrap from.
    #[error("STaR requires at least one problem")]
    NoProblems,

    /// Two problems shared an id, which would make the accumulated set's
    /// dedup-by-id policy silently drop one of them.
    #[error("duplicate problem id: {id}")]
    DuplicateProblemId {
        /// The id that appeared more than once.
        id: String,
    },

    /// A configuration value was outside its valid range.
    #[error("invalid STaR configuration: {reason}")]
    InvalidConfig {
        /// Why the configuration was rejected.
        reason: String,
    },
}
