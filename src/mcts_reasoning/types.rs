//! Types for the `mcts_reasoning` module: the search tree arena, the selection and
//! rollout policies, the progressive-widening schedule, the plug-in traits, and the
//! configuration.

use thiserror::Error;

use crate::types::SearchResult;

// ── MctsError ─────────────────────────────────────────────────────────────────

/// Errors from the `mcts_reasoning` module.
#[derive(Debug, Error)]
pub enum MctsError {
    /// The query string was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
    /// The underlying retrieval step failed.
    #[error("Retrieval failed: {0}")]
    RetrievalFailed(String),
    /// The step generator proposed no candidate steps at the root, so there is
    /// nothing to search over.
    #[error("The step generator proposed no candidate steps at the root")]
    NoCandidates,
    /// The configuration is not a valid search problem.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ── MctsCandidate ─────────────────────────────────────────────────────────────

/// A candidate next reasoning step, proposed by an [`MctsStepGenerator`].
///
/// The `prior` is the generator's *unnormalized* belief that this step is worth
/// taking — a policy logit's exponential, a retrieval score, a rule's confidence,
/// or simply `1.0` if the generator has no opinion. The engine normalizes the
/// priors of a node's candidates to sum to `1` when the node is created (see
/// [`MctsNode::candidates`]), so a generator never has to.
///
/// The prior is consumed by [`MctsSelectionPolicy::Puct`] and
/// [`MctsSelectionPolicy::PriorUct`], and by
/// [`MctsRolloutPolicy::PriorWeighted`]. Under [`MctsSelectionPolicy::Uct`] and
/// [`MctsRolloutPolicy::Uniform`] it is ignored entirely, which is exactly the
/// point of having both.
#[derive(Debug, Clone, PartialEq)]
pub struct MctsCandidate {
    /// The reasoning step's content.
    pub content: String,
    /// The generator's unnormalized prior weight for this step. Must be finite and
    /// non-negative; see [`MctsNode::candidates`] for what happens if it is not.
    pub prior: f64,
}

impl MctsCandidate {
    /// A candidate with no prior opinion (weight `1.0`).
    ///
    /// A node all of whose candidates are built this way ends up with an exactly
    /// **uniform** prior, `1/K` — which is the setting in which
    /// [`MctsSelectionPolicy::PriorUct`] provably degenerates to plain
    /// [`MctsSelectionPolicy::Uct`].
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            prior: 1.0,
        }
    }

    /// A candidate carrying an explicit unnormalized prior weight.
    #[must_use]
    pub fn with_prior(content: impl Into<String>, prior: f64) -> Self {
        Self {
            content: content.into(),
            prior,
        }
    }
}

// ── MctsStepGenerator ─────────────────────────────────────────────────────────

/// Proposes the candidate next steps of a reasoning path.
///
/// Called **once per tree node**, at the moment the node is created, and once per
/// step of every rollout. The `path` runs from the root's content to the current
/// state's content, inclusive, so an implementation sees the whole reasoning
/// prefix — not merely the last step.
///
/// Returning an empty vector marks the state **terminal**: it is a dead end, no
/// child will ever be attached to it, and every subsequent visit re-evaluates it in
/// place. That is a load-bearing contract — see [`MctsNode::terminal`].
pub trait MctsStepGenerator {
    /// Propose the candidate steps that could follow `path`.
    fn propose(&self, query: &str, path: &[String], context: &[SearchResult])
    -> Vec<MctsCandidate>;
}

// ── MctsTerminalEvaluator ─────────────────────────────────────────────────────

/// Scores a completed reasoning path — the **return** of a rollout.
///
/// Called once at the end of every simulation, on the terminal (or depth-capped)
/// path the rollout reached. The value it returns is what gets backpropagated, so
/// it is the only signal the whole search has about what is good.
///
/// Implementations should return a value in `[0, 1]`. The engine **clamps** to that
/// range, and maps a non-finite score to `0.0` — a `NaN` return would otherwise
/// propagate into every `Q` on the path and silently poison every `partial_cmp` in
/// the selection step, turning the search into a first-child walk with no error
/// anywhere. See [`MctsEngine`](crate::mcts_reasoning::MctsEngine).
pub trait MctsTerminalEvaluator {
    /// Score the completed reasoning `path`, in `[0, 1]`.
    fn score(&self, query: &str, path: &[String], context: &[SearchResult]) -> f64;
}

// ── MctsSelectionPolicy ───────────────────────────────────────────────────────

/// The **tree policy** — how the search picks which child to descend into.
///
/// Every variant scores a child `a` of a node `s` as `Q(s,a) + bonus`, where
/// `Q(s,a)` is the child's running mean return (see [`MctsNode::mean_value`]) and
/// the bonus is what makes it a *search* rather than a hill-climb. `N(s)` is the
/// parent's visit count and `N(s,a)` the child's, both read **before** the current
/// simulation is backed up.
///
/// Because a child is created *and immediately rolled out*, `N(s,a) >= 1` for every
/// child that exists, and `N(s) = 1 + sum_a N(s,a) >= 2` for every node that has
/// one. Neither the division nor the logarithm below can therefore see a zero, and
/// there is no "infinity for an unvisited child" special case to get wrong.
///
/// # The variants, and how they relate
///
/// | Variant | Bonus | Prior? |
/// |---|---|---|
/// | [`Uct`](MctsSelectionPolicy::Uct) | `c * sqrt(ln N(s) / N(s,a))` | no |
/// | [`PriorUct`](MctsSelectionPolicy::PriorUct) | `c * K * P(s,a) * sqrt(ln N(s) / N(s,a))` | yes |
/// | [`Puct`](MctsSelectionPolicy::Puct) | `c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))` | yes |
/// | [`UniformRandom`](MctsSelectionPolicy::UniformRandom) | — (ablation baseline) | no |
///
/// `K` is the number of candidate steps the generator proposed at `s`, so `K*P` is
/// the child's prior **relative to uniform**. That normalization is what makes
/// `PriorUct` collapse *exactly* onto `Uct` when the prior is uniform (`P = 1/K`),
/// which is the sense in which a prior-guided search is a strict generalization of
/// an uninformed one.
///
/// **`Puct` does not, and cannot, collapse onto `Uct`.** Its bonus decays like
/// `1/(1 + N(s,a))` and grows like `sqrt(N(s))`; `Uct`'s decays like
/// `1/sqrt(N(s,a))` and grows like `sqrt(ln N(s))`. No choice of `c` reconciles two
/// different functions of two free variables, and setting `P = 1/K` only makes the
/// bonus *constant across children* — it does not change its shape. This is not a
/// deficiency in either; they are different algorithms with different regret
/// arguments, and this module ships both rather than pretending one is the other.
#[derive(Debug, Clone, PartialEq)]
pub enum MctsSelectionPolicy {
    /// **`UCT`** — Kocsis & Szepesvári (2006), "Bandit based Monte-Carlo Planning".
    ///
    /// `Q(s,a) + c * sqrt(ln N(s) / N(s,a))`
    ///
    /// This is `UCB1` applied recursively at every node of the tree. The bonus is
    /// the width of a Hoeffding confidence interval on `Q(s,a)`: it is large for a
    /// child that has been tried rarely relative to its parent, and it shrinks like
    /// `1/sqrt(N(s,a))` as evidence accumulates. Selecting the argmax is therefore
    /// "act as though the most uncertain plausible child is as good as it could
    /// be" — *optimism in the face of uncertainty* — and it is what bounds the
    /// number of pulls of a suboptimal child at `O(ln n)`.
    ///
    /// `exploration` is `c`. The canonical value for returns in `[0, 1]` is
    /// `sqrt(2)`, which is what [`MctsConfig::default`] uses. Setting `c = 0`
    /// reduces the tree policy to greedy exploitation of `Q`, which is a useful
    /// thing to be able to say and a terrible thing to deploy.
    Uct {
        /// The exploration constant `c`. Must be finite and non-negative.
        exploration: f64,
    },

    /// **Prior-scaled `UCT`** — the exploration bonus of [`Uct`](MctsSelectionPolicy::Uct),
    /// weighted by the generator's policy prior.
    ///
    /// `Q(s,a) + c * K * P(s,a) * sqrt(ln N(s) / N(s,a))`
    ///
    /// The idea is Rosin's (2011, "Multi-armed bandits with episode context"): a
    /// prior over the actions should bias *how much you are willing to explore*
    /// each one, not what you believe it is worth. A step the generator considers
    /// twice as plausible as uniform gets twice the exploration budget; a step it
    /// considers implausible is still reachable, but only once `Q` gives the search
    /// a reason to look.
    ///
    /// The factor `K` (the number of candidates at `s`) normalizes the prior
    /// against the uniform distribution, so that **a uniform prior recovers
    /// [`Uct`](MctsSelectionPolicy::Uct) exactly**: `K * (1/K) = 1`, and multiplying
    /// a bonus by an exact `1.0` is exact.
    ///
    /// In exact arithmetic that is an identity for every `K`. In `f64` it is an
    /// identity for every `K < 49` — verified over a grid of exploration constants,
    /// visit counts and values, bit for bit. It is *not* quite one beyond that:
    /// `49 * fl(1/49) = 0.9999999999999999`, the first of 82 such `K` below 1000
    /// (the others start `98, 103, 107, 161, ...`), and for those the two policies'
    /// scores can differ by up to **2 ulps** — never more, and at most in the
    /// direction the rounding of `1/K` already went.
    ///
    /// This is documented rather than papered over because a search that is "the same
    /// except sometimes in the last bit" is exactly the kind of thing that becomes a
    /// flaky test three months later. The module's tests assert the bit-exact
    /// reduction below `K = 49` and *pin* the 1-ulp discrepancy at `K = 49` itself.
    PriorUct {
        /// The exploration constant `c`. Must be finite and non-negative.
        exploration: f64,
    },

    /// **`PUCT`** — the `AlphaGo` / `AlphaZero` variant (Silver et al., 2016/2017).
    ///
    /// `Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
    ///
    /// The prior multiplies the bonus directly, and the `1/(1 + N(s,a))` decay is
    /// far more aggressive than `UCT`'s `1/sqrt(N(s,a))`: a strong prior is trusted
    /// hard and early, and is then abandoned quickly as real returns arrive. The
    /// `sqrt(N(s))` numerator keeps the bonus from vanishing as the parent's
    /// evidence piles up, so a child with a high prior and no visits stays
    /// attractive.
    ///
    /// This is the policy to use when the generator is a *trained* proposer whose
    /// priors mean something. With a flat or meaningless prior it is strictly worse
    /// than [`Uct`](MctsSelectionPolicy::Uct) — it has no `ln` to temper it, and no
    /// regret bound to fall back on.
    ///
    /// See [`MctsSelectionPolicy`] for why this does *not* reduce to
    /// [`Uct`](MctsSelectionPolicy::Uct) under a uniform prior, and
    /// [`PriorUct`](MctsSelectionPolicy::PriorUct) for the variant that does.
    Puct {
        /// The exploration constant `c_puct`. Must be finite and non-negative.
        /// `AlphaZero` used values around `1.0`–`4.0`.
        exploration: f64,
    },

    /// **Uniformly random descent** — the ablation baseline, not a search policy.
    ///
    /// Picks a child uniformly at random, ignoring `Q` and ignoring the visit
    /// counts. Expansion, rollout and backpropagation are otherwise unchanged, so
    /// this is "the same Monte-Carlo estimator with the *tree policy removed*" —
    /// which is precisely the thing `UCT`'s advantage has to be measured against.
    ///
    /// It is public because the comparison is the whole point: a tree search that
    /// cannot beat random descent on a problem with a learnable gradient is not
    /// searching, and no amount of internally-consistent visit counts will reveal
    /// that. This module's tests pit `UCT` against it on a needle-in-a-haystack
    /// reasoning tree and assert the margin. (It is the same role `epsilon = 1`
    /// plays in `bandit_ranker`.)
    ///
    /// Note that under this policy the visit counts carry **no signal**, so
    /// selecting the final action by max-visit is meaningless; a caller using it as
    /// a baseline should read [`MctsNode::mean_value`] instead.
    UniformRandom,
}

impl Default for MctsSelectionPolicy {
    /// `UCT` with `c = sqrt(2)`, the canonical constant for returns in `[0, 1]`.
    fn default() -> Self {
        Self::Uct {
            exploration: std::f64::consts::SQRT_2,
        }
    }
}

impl MctsSelectionPolicy {
    /// The exploration constant, or `None` for
    /// [`UniformRandom`](MctsSelectionPolicy::UniformRandom), which has none.
    #[must_use]
    pub fn exploration(&self) -> Option<f64> {
        match *self {
            Self::Uct { exploration }
            | Self::PriorUct { exploration }
            | Self::Puct { exploration } => Some(exploration),
            Self::UniformRandom => None,
        }
    }

    /// Whether this policy reads [`MctsCandidate::prior`].
    #[must_use]
    pub fn uses_prior(&self) -> bool {
        matches!(self, Self::PriorUct { .. } | Self::Puct { .. })
    }

    /// The score this policy assigns to `child` as a child of `parent` — the
    /// quantity the tree policy takes the argmax of.
    ///
    /// `N(s)` and `N(s,a)` are read from the nodes as they stand, which during
    /// selection means **before** the current simulation has been backed up. Every
    /// child that exists has `N(s,a) >= 1` (it was rolled out when it was created)
    /// and every parent with a child has `N(s) >= 2` (by the invariant
    /// `N(s) = 1 + sum_c N(c)`), so neither the division nor the logarithm below can
    /// see a zero.
    ///
    /// This is public because a search that will not tell you *why* it went where it
    /// went is very hard to trust: given the tree in [`MctsOutput`], this recovers
    /// the exact number that decided every descent.
    ///
    /// [`UniformRandom`](Self::UniformRandom) has no score function — it ignores every
    /// statistic by construction — and returns `0.0` for every child.
    #[must_use]
    pub fn score(&self, parent: &MctsNode, child: &MctsNode) -> f64 {
        let exploit = child.mean_value();
        let parent_visits = f64::from(parent.visits);
        let child_visits = f64::from(child.visits);

        match *self {
            Self::Uct { exploration } => {
                exploit + exploration * (parent_visits.ln() / child_visits).sqrt()
            }
            Self::PriorUct { exploration } => {
                // `K * P` — the child's prior *relative to uniform*. A uniform prior
                // makes this exactly `1.0` (for every `K` whose reciprocal
                // round-trips, which is every `K < 49`), and multiplying by an exact
                // `1.0` is exact, so the expression then collapses term for term —
                // and bit for bit — onto the `Uct` arm above.
                //
                // The candidate count is bounded by what one generator call returns,
                // many orders of magnitude below `2^53`, so the cast is exact.
                #[allow(clippy::cast_precision_loss)]
                let relative_prior = child.prior * parent.action_count() as f64;
                exploit + exploration * relative_prior * (parent_visits.ln() / child_visits).sqrt()
            }
            Self::Puct { exploration } => {
                exploit + exploration * child.prior * parent_visits.sqrt() / (1.0 + child_visits)
            }
            Self::UniformRandom => 0.0,
        }
    }
}

// ── MctsRolloutPolicy ─────────────────────────────────────────────────────────

/// The **default policy** — how a rollout picks its next step once it has left the
/// tree.
///
/// The tree policy ([`MctsSelectionPolicy`]) governs the part of the state space the
/// search has already built nodes for. Beyond the frontier there are no statistics
/// to act on, and *something* still has to choose. That something is this.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MctsRolloutPolicy {
    /// Choose uniformly at random among the proposed steps — a "light" playout.
    ///
    /// Cheap, unbiased, and the classical choice. Its estimate of a state's value
    /// is the mean return under random continuation, which is a *lower bound* on
    /// the state's true (best-play) value and a noisy one — but it is unbiased in
    /// the sense that matters: it does not systematically prefer any subtree, so
    /// the tree policy above it is the only thing steering the search.
    #[default]
    Uniform,
    /// Sample a step with probability proportional to its
    /// [prior](MctsCandidate::prior) — a "heavy" playout.
    ///
    /// Strictly better when the generator's priors are informative, and strictly
    /// worse when they are confidently wrong: a heavy playout inherits the
    /// generator's blind spots and hands them to the search as evidence. If the
    /// priors are uniform this is identical in distribution to
    /// [`Uniform`](MctsRolloutPolicy::Uniform) (though it consumes the `RNG`
    /// differently, so the two do not produce the same stream).
    PriorWeighted,
}

// ── MctsWidening ──────────────────────────────────────────────────────────────

/// **Progressive widening** — bounding a node's branching factor by its visit count.
///
/// A reasoning-step generator can propose a great many continuations, and in the
/// limit (a sampled language model) unboundedly many. Attaching them all to a node
/// is the fastest way to destroy a tree search: the visit budget is spread so thin
/// across the children that no child ever accumulates the evidence needed to be
/// preferred, `Q` stays noise, and `UCT` degenerates into round-robin sampling of a
/// giant flat bandit. The search becomes *broad and blind*.
///
/// Progressive widening (Coulom 2007; Chaslot et al. 2008; Couëtoux et al. 2011)
/// fixes this by rationing children against evidence. A node with `N` visits is
/// allowed at most
///
/// ```text
/// k(N) = ceil(coefficient * N^exponent)
/// ```
///
/// children. A new child is only attached when the node's existing children are
/// fewer than `k(N)`; otherwise the search must descend into a child it already
/// has. With `exponent < 1` the allowance grows *sublinearly* — so the visits per
/// child, `N / k(N) ~ N^(1 - exponent) / coefficient`, grows without bound, and
/// every child that does exist eventually gets enough samples for its `Q` to mean
/// something. That is the whole trick, and it is why the exponent must be strictly
/// below `1`.
///
/// The candidates are drawn in the order the generator proposed them, so a
/// generator that returns its steps in descending prior order gets the most
/// plausible ones admitted first — which is usually what you want, and costs
/// nothing.
///
/// # Numerics
///
/// `k(N)` is a `ceil` of a `powf`, and `powf` is not required by the Rust standard
/// library to be correctly rounded — a platform whose `pow` returns
/// `3.0000000000000004` for `9^0.5` would silently allow a *fourth* child where the
/// mathematics allows three, and the bug would be invisible on the machine it was
/// written on. The limit is therefore snapped to the nearest integer when it lies
/// within `1e-9` of one, before the `ceil`. (On this codebase's platform glibc's
/// `pow` happens to be exact for these cases; the guard is there so the answer does
/// not depend on that.)
#[derive(Debug, Clone, PartialEq)]
pub struct MctsWidening {
    /// The multiplier `C` in `k(N) = ceil(C * N^alpha)`. Must be finite and
    /// strictly positive.
    pub coefficient: f64,
    /// The exponent `alpha` in `k(N) = ceil(C * N^alpha)`. Must lie in `(0, 1)` —
    /// at `alpha >= 1` the allowance grows at least linearly in the visits and the
    /// per-child sample count stops growing, which defeats the purpose.
    pub exponent: f64,
}

impl Default for MctsWidening {
    /// `k(N) = ceil(1.0 * N^0.5)` — the square-root schedule.
    ///
    /// A node's `k`-th child is unlocked at `N = (k-1)^2 + 1` visits: the second
    /// child at `2` visits, the third at `5`, the fourth at `10`, the tenth at `82`.
    /// Aggressive enough to keep a deep reasoning tree narrow, mild enough that a
    /// genuinely promising node does widen.
    fn default() -> Self {
        Self {
            coefficient: 1.0,
            exponent: 0.5,
        }
    }
}

impl MctsWidening {
    /// Create a widening schedule `k(N) = ceil(coefficient * N^exponent)`.
    #[must_use]
    pub fn new(coefficient: f64, exponent: f64) -> Self {
        Self {
            coefficient,
            exponent,
        }
    }

    /// The maximum number of children a node with `visits` visits may have.
    ///
    /// Always at least `1`: a non-terminal node that has been visited must be able
    /// to acquire a child, or the search would have nowhere to go and the
    /// simulation would be unable to make progress.
    ///
    /// See the [type documentation](MctsWidening) for the `ceil`/`powf` rounding
    /// guard.
    #[must_use]
    pub fn limit(&self, visits: u32) -> usize {
        let raw = self.coefficient * f64::from(visits).powf(self.exponent);
        if !raw.is_finite() {
            return usize::MAX;
        }
        // Snap to an exact integer when `powf`'s last bit is the only thing
        // standing between us and one, so that `k(9)` under the square-root
        // schedule is 3 on every platform rather than 3-or-4 depending on libm.
        let snapped = if (raw - raw.round()).abs() < 1e-9 {
            raw.round()
        } else {
            raw
        };
        let bounded = snapped.ceil().clamp(1.0, 4_294_967_295.0);
        // `bounded` is in [1, 2^32 - 1] by the clamp above, so the cast is exact on
        // every platform this crate supports (`usize` is at least 32 bits).
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            bounded as usize
        }
    }
}

// ── MctsNode ──────────────────────────────────────────────────────────────────

/// A node of the search tree: one reasoning state, and everything the search has
/// learned about it.
///
/// The statistics are the entire difference between this and a
/// `tree_of_thought` node. There, `value` is written once,
/// at expansion, from a heuristic, and is never revised. Here, [`visits`](Self::visits)
/// and [`total_value`](Self::total_value) are *accumulators*: every simulation that
/// passes through this node adds its rollout's return to `total_value` and one to
/// `visits`, so [`mean_value`](Self::mean_value) is a running Monte-Carlo estimate
/// of what lies beneath — and it gets better the more the search looks.
#[derive(Debug, Clone, PartialEq)]
pub struct MctsNode {
    /// Unique id, and index into [`MctsTree::nodes`].
    pub id: usize,
    /// Parent's id; `None` for the root.
    pub parent: Option<usize>,
    /// Ids of the materialized children, in the order the candidates were proposed.
    ///
    /// Under progressive widening this is a *prefix* of [`candidates`](Self::candidates):
    /// `children[i]` is the node grown from `candidates[i]`, and the candidates from
    /// `children.len()` onward have not been tried yet.
    pub children: Vec<usize>,
    /// Every step the generator proposed here, with priors **normalized to sum to
    /// `1`**.
    ///
    /// Computed once, when the node is created, and never recomputed — one generator
    /// call per node. Normalization maps each candidate's weight `w_i` to
    /// `max(w_i, 0) / sum_j max(w_j, 0)`; if that denominator is not finite and
    /// strictly positive (all weights zero, negative, `NaN`, or infinite) the engine
    /// falls back to the **uniform** prior `1/K` rather than propagating the
    /// nonsense into the selection score.
    pub candidates: Vec<MctsCandidate>,
    /// The reasoning step taken to reach this state. The root's content is the
    /// framing of the question itself.
    pub content: String,
    /// Depth in the tree; the root is `0`.
    pub depth: usize,
    /// `N(s)` — how many simulations have been backed up through this node.
    ///
    /// Bounded by `simulations + 1`, and [`MctsConfig::validate`] refuses a budget
    /// that could overflow a `u32`.
    pub visits: u32,
    /// `W(s)` — the sum of the returns of every rollout backed up through this node.
    pub total_value: f64,
    /// `P(s)` — this node's normalized prior, as assigned by its parent's generator.
    /// The root's is `1.0`.
    pub prior: f64,
    /// Whether this state is an end of the reasoning: either it hit
    /// [`MctsConfig::max_depth`], or the generator proposed nothing here.
    ///
    /// A terminal node is never expanded. When selection reaches one it is
    /// re-evaluated in place and the return backed up, so a terminal node's visit
    /// count keeps growing while its child count stays `0` — which is why the
    /// backpropagation invariant `N(s) = 1 + sum_c N(c)` is stated for nodes that
    /// *have* children.
    pub terminal: bool,
}

impl MctsNode {
    /// `Q(s) = W(s) / N(s)` — the mean return of every rollout backed up through
    /// this node, and `0.0` for a node that has never been visited.
    ///
    /// This is exactly the quantity [`MctsSelectionPolicy`] exploits, and the
    /// quantity that a `tree_of_thought` node does not
    /// have.
    #[must_use]
    pub fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_value / f64::from(self.visits)
    }

    /// `K` — the number of steps the generator proposed here, materialized or not.
    ///
    /// This is the denominator of the uniform prior, and hence the `K` in
    /// [`MctsSelectionPolicy::PriorUct`]'s bonus.
    #[must_use]
    pub fn action_count(&self) -> usize {
        self.candidates.len()
    }

    /// The number of proposed steps that have not been grown into children yet.
    #[must_use]
    pub fn untried_count(&self) -> usize {
        self.candidates.len().saturating_sub(self.children.len())
    }

    /// Whether every proposed step has been grown into a child.
    #[must_use]
    pub fn fully_expanded(&self) -> bool {
        self.untried_count() == 0
    }
}

// ── MctsTree ──────────────────────────────────────────────────────────────────

/// The search tree: a flat arena of [`MctsNode`]s whose ids are their indices.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MctsTree {
    /// Every node, indexed by [`MctsNode::id`].
    pub nodes: Vec<MctsNode>,
}

impl MctsTree {
    /// The root node, or `None` for an empty tree.
    #[must_use]
    pub fn root(&self) -> Option<&MctsNode> {
        self.nodes.first()
    }

    /// Borrow a node by id.
    #[must_use]
    pub fn node(&self, id: usize) -> Option<&MctsNode> {
        self.nodes.get(id)
    }

    /// The path from the root down to `id`, inclusive, root **first**.
    ///
    /// (Note the direction: `tree_of_thought`'s
    /// `path_to_root` returns the reverse. A rollout needs the reasoning prefix in
    /// reading order, and so does an evaluator, so this module returns it that way
    /// rather than making every caller reverse it.)
    #[must_use]
    pub fn path_from_root(&self, id: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut cursor = Some(id);
        while let Some(current) = cursor {
            match self.nodes.get(current) {
                Some(node) => {
                    path.push(current);
                    cursor = node.parent;
                }
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// The reasoning steps from the root down to `id`, inclusive, in reading order.
    #[must_use]
    pub fn path_contents(&self, id: usize) -> Vec<String> {
        self.path_from_root(id)
            .into_iter()
            .filter_map(|i| self.nodes.get(i).map(|n| n.content.clone()))
            .collect()
    }

    /// The child of `id` with the most visits — the **final action selection** rule.
    ///
    /// # Why visits and not value
    ///
    /// The obvious choice is `argmax Q`, and it is wrong. `Q` is a mean over however
    /// many rollouts the child happened to get, so a child visited *twice* that got
    /// lucky both times has a higher `Q` than one visited four hundred times whose
    /// mean has actually converged — and the search will hand you the lucky one. The
    /// visit count is the search's own revealed confidence: `UCT` only keeps
    /// spending visits on a child while its upper confidence bound stays on top, so
    /// the most-visited child is the one the tree policy kept coming back to *after*
    /// its uncertainty had shrunk. It is the robust statistic, it is what `AlphaGo`
    /// and every serious `MCTS` engine ship, and the gap between the two is not
    /// academic: this module's tests contain a tree where they disagree.
    ///
    /// # Tie-break (exact, deterministic, documented)
    ///
    /// Ties are broken by, in order:
    ///
    /// 1. **higher visit count** `N`;
    /// 2. then **higher mean value** `Q`;
    /// 3. then **lower node id** — i.e. the candidate the generator proposed first.
    ///
    /// The last rule matters more than it looks. The obvious implementation,
    /// `children.max_by_key(|c| c.visits)`, returns the **last** maximal element,
    /// which makes the search's answer depend on the order the arena happened to be
    /// filled in — a silent, reproducible-but-wrong tie-break. This is a fold with a
    /// strict comparison, so the *first* maximal child wins, and "first" means
    /// "lowest id", which means "the step the generator proposed earliest".
    #[must_use]
    pub fn best_child_by_visits(&self, id: usize) -> Option<usize> {
        let parent = self.nodes.get(id)?;
        let mut best: Option<(usize, u32, f64)> = None;
        for &child_id in &parent.children {
            let Some(child) = self.nodes.get(child_id) else {
                continue;
            };
            let candidate = (child_id, child.visits, child.mean_value());
            best = match best {
                None => Some(candidate),
                Some(current) => {
                    // Replace only on a *strict* improvement, so equal keys keep the
                    // earlier (lower-id) child. `children` is in ascending id order.
                    let better = candidate.1 > current.1
                        || (candidate.1 == current.1 && candidate.2 > current.2);
                    if better {
                        Some(candidate)
                    } else {
                        Some(current)
                    }
                }
            };
        }
        best.map(|(child_id, _, _)| child_id)
    }

    /// The **principal variation**: the path obtained by repeatedly descending into
    /// the most-visited child from the root, until a node with no children.
    ///
    /// This is the reasoning chain the search actually endorses. It is *not* the
    /// highest-scoring leaf, and it is not the path the best single rollout took —
    /// it is the path down the part of the tree the search chose to spend its budget
    /// on. Each step uses the tie-break of [`best_child_by_visits`](Self::best_child_by_visits).
    #[must_use]
    pub fn principal_variation(&self) -> Vec<usize> {
        let mut path = Vec::new();
        if self.nodes.is_empty() {
            return path;
        }
        let mut cursor = 0usize;
        path.push(cursor);
        while let Some(next) = self.best_child_by_visits(cursor) {
            path.push(next);
            cursor = next;
        }
        path
    }

    /// The reasoning steps along the [principal variation](Self::principal_variation).
    #[must_use]
    pub fn principal_variation_contents(&self) -> Vec<String> {
        self.principal_variation()
            .into_iter()
            .filter_map(|i| self.nodes.get(i).map(|n| n.content.clone()))
            .collect()
    }

    /// Whether the backpropagation invariant holds at every internal node.
    ///
    /// For every node `s` that has at least one child:
    ///
    /// ```text
    /// N(s) == 1 + sum over children c of N(c)
    /// ```
    ///
    /// The `1` is the node's **own** creation rollout: a node is created, rolled out
    /// once, and its return backed up, all in the simulation that expanded it. Every
    /// *later* simulation that reaches `s` must descend into exactly one child — it
    /// cannot stop at `s`, because `s` is not terminal (it has children, so it was
    /// expandable) and it cannot descend into two. So each of `s`'s later visits is
    /// accounted for by exactly one visit to exactly one child, and the identity is
    /// exact, not approximate.
    ///
    /// It holds at the **root** too, which is the reason the root is rolled out once
    /// when the tree is built: without that, the root would be the one node in the
    /// tree with `N(s) = sum_c N(c)` and the invariant would need a special case.
    /// The price is one extra rollout per search — `simulations + 1` in total.
    ///
    /// The invariant is stated for nodes *with children* because a **terminal** node
    /// accumulates visits without ever acquiring one: selection reaches it,
    /// re-evaluates it in place, and backs the return up. Its `N` counts rollouts,
    /// and there is nothing beneath it to sum.
    ///
    /// This is a real check with a real failure mode — a backup that misses the root,
    /// double-counts the expanded node, or stops one short of the leaf will still
    /// produce a plausible-looking tree with plausible-looking values, and will fail
    /// here.
    #[must_use]
    pub fn check_visit_invariant(&self) -> bool {
        self.nodes.iter().all(|node| {
            if node.children.is_empty() {
                return true;
            }
            let child_visits: u32 = node
                .children
                .iter()
                .filter_map(|&c| self.nodes.get(c))
                .map(|c| c.visits)
                .sum();
            node.visits == 1 + child_visits
        })
    }
}

// ── MctsStats ─────────────────────────────────────────────────────────────────

/// What the search did, as opposed to what it concluded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MctsStats {
    /// The number of simulations run — [`MctsConfig::simulations`].
    pub simulations: usize,
    /// The number of nodes in the final tree, root included.
    pub nodes_created: usize,
    /// The return of every rollout, in the order they happened.
    ///
    /// Length is `simulations + 1`: element `0` is the **root's** creation rollout
    /// (see [`MctsTree::check_visit_invariant`]), and element `i` is the rollout of
    /// simulation `i`. Because the root is on the backup path of every single one of
    /// them, and backpropagation adds each return to `total_value` in exactly this
    /// order, the root's [`MctsNode::total_value`] is the in-order sum of this
    /// vector — **bit for bit**, not merely to within a tolerance. That is a
    /// genuinely independent check on the backup, and the tests use it as one.
    pub rollout_returns: Vec<f64>,
    /// The root child that each simulation descended into (or created), by node id.
    ///
    /// Length is `simulations`. This is the search's empirical action distribution
    /// over time, and it is what makes cumulative **regret** measurable: pair each
    /// entry with the true value of the action and sum the shortfall against the
    /// best one. `UCT`'s central claim is that this sum grows like `O(log n)`, and
    /// this vector is what lets a caller — or a test — actually check that rather
    /// than take it on faith.
    pub root_action_history: Vec<usize>,
    /// The greatest depth any node in the tree reached.
    pub max_depth_reached: usize,
}

// ── MctsOutput ────────────────────────────────────────────────────────────────

/// The result of a Monte-Carlo tree search.
#[derive(Debug, Clone, PartialEq)]
pub struct MctsOutput {
    /// The synthesized answer, from the principal variation and the retrieved
    /// context.
    pub answer: String,
    /// The reasoning steps along the [principal variation](MctsTree::principal_variation),
    /// root first.
    pub best_path: Vec<String>,
    /// The node id of the chosen first step — the most-visited child of the root.
    ///
    /// `None` only if the root never acquired a child, which requires a zero
    /// simulation budget (and [`MctsConfig::validate`] rejects that).
    pub best_root_action: Option<usize>,
    /// The full tree, with every visit count and value the search accumulated.
    pub tree: MctsTree,
    /// What the search did on the way there.
    pub stats: MctsStats,
}

// ── MctsConfig ────────────────────────────────────────────────────────────────

/// Configuration for [`MctsEngine`](crate::mcts_reasoning::MctsEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct MctsConfig {
    /// How many simulations to run. Each one selects, expands at most one node,
    /// rolls out once, and backs up once. Defaults to `256`.
    pub simulations: usize,
    /// The depth at which a reasoning path is complete. A node at this depth is
    /// [terminal](MctsNode::terminal). Defaults to `4`.
    pub max_depth: usize,
    /// How many steps a rollout may take *beyond* the node it starts from, or `None`
    /// to play out all the way to [`max_depth`](Self::max_depth).
    ///
    /// `Some(0)` turns the rollout off entirely: the return becomes the evaluator's
    /// score of the node's own path, and the search degenerates into a best-first
    /// search over a static heuristic — which is a legitimate configuration when the
    /// evaluator is a trained value function (`AlphaZero` does exactly this), and a
    /// pointless one when it is not. Defaults to `None`.
    pub rollout_depth_cap: Option<usize>,
    /// The tree policy. Defaults to [`MctsSelectionPolicy::default`] — `UCT` with
    /// `c = sqrt(2)`.
    pub selection: MctsSelectionPolicy,
    /// The default policy used inside rollouts. Defaults to
    /// [`MctsRolloutPolicy::Uniform`].
    pub rollout: MctsRolloutPolicy,
    /// The progressive-widening schedule, or `None` to attach one new child per
    /// visit until every proposed candidate is a child (classical `MCTS` expansion).
    ///
    /// Defaults to `Some(MctsWidening::default())`, the square-root schedule.
    pub widening: Option<MctsWidening>,
    /// Seed for the rollout policy's [`MctsRng`](crate::mcts_reasoning::MctsRng).
    /// Defaults to `0x5EED`.
    pub seed: u64,
    /// How many documents to retrieve for the search's context. Defaults to `5`.
    pub top_k: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 256,
            max_depth: 4,
            rollout_depth_cap: None,
            selection: MctsSelectionPolicy::default(),
            rollout: MctsRolloutPolicy::default(),
            widening: Some(MctsWidening::default()),
            seed: 0x5EED,
            top_k: 5,
        }
    }
}

impl MctsConfig {
    /// Set the simulation budget.
    #[must_use]
    pub fn with_simulations(mut self, v: usize) -> Self {
        self.simulations = v;
        self
    }

    /// Set the terminal depth.
    #[must_use]
    pub fn with_max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }

    /// Set the rollout depth cap.
    #[must_use]
    pub fn with_rollout_depth_cap(mut self, v: Option<usize>) -> Self {
        self.rollout_depth_cap = v;
        self
    }

    /// Set the tree policy.
    #[must_use]
    pub fn with_selection(mut self, v: MctsSelectionPolicy) -> Self {
        self.selection = v;
        self
    }

    /// Set the rollout policy.
    #[must_use]
    pub fn with_rollout(mut self, v: MctsRolloutPolicy) -> Self {
        self.rollout = v;
        self
    }

    /// Set the progressive-widening schedule (`None` disables widening).
    #[must_use]
    pub fn with_widening(mut self, v: Option<MctsWidening>) -> Self {
        self.widening = v;
        self
    }

    /// Set the rollout seed.
    #[must_use]
    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Set the retrieval top-k.
    #[must_use]
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.top_k = v;
        self
    }

    /// Check that this configuration describes a search that can actually run.
    ///
    /// # Errors
    ///
    /// Returns [`MctsError::InvalidConfig`] if the simulation budget is zero or
    /// large enough to overflow a `u32` visit count; if `max_depth` is zero (the
    /// root would be terminal and there would be nothing to search); if an
    /// exploration constant is negative or not finite; or if the widening schedule's
    /// coefficient is not finite and positive, or its exponent does not lie strictly
    /// inside `(0, 1)` — at `alpha >= 1` the per-child sample count stops growing and
    /// widening no longer does its job (see [`MctsWidening`]).
    pub fn validate(&self) -> Result<(), MctsError> {
        if self.simulations == 0 {
            return Err(MctsError::InvalidConfig(
                "simulations must be at least 1".into(),
            ));
        }
        // Every node's visit count is bounded by `simulations + 1`, so this bound is
        // exactly what makes the `u32` accumulator in `MctsNode::visits` provably
        // overflow-free.
        if self.simulations >= u32::MAX as usize {
            return Err(MctsError::InvalidConfig(format!(
                "simulations must be below {} to fit a u32 visit count, got {}",
                u32::MAX,
                self.simulations
            )));
        }
        if self.max_depth == 0 {
            return Err(MctsError::InvalidConfig(
                "max_depth must be at least 1, or the root is already terminal".into(),
            ));
        }
        if let Some(c) = self.selection.exploration()
            && (!c.is_finite() || c < 0.0)
        {
            return Err(MctsError::InvalidConfig(format!(
                "the exploration constant must be finite and non-negative, got {c}"
            )));
        }
        if let Some(widening) = &self.widening {
            if !widening.coefficient.is_finite() || widening.coefficient <= 0.0 {
                return Err(MctsError::InvalidConfig(format!(
                    "the widening coefficient must be finite and positive, got {}",
                    widening.coefficient
                )));
            }
            if !widening.exponent.is_finite()
                || widening.exponent <= 0.0
                || widening.exponent >= 1.0
            {
                return Err(MctsError::InvalidConfig(format!(
                    "the widening exponent must lie strictly inside (0, 1), got {}",
                    widening.exponent
                )));
            }
        }
        Ok(())
    }
}

// ── MctsHeuristicGenerator ────────────────────────────────────────────────────

/// A lexical, dependency-free [`MctsStepGenerator`] — the default proposer.
///
/// It is a *baseline*, and it is worth being precise about what it does rather than
/// leaving the impression that it reasons. For each of the most frequent content
/// terms in the retrieved context that the path has not already used, it proposes
/// the step `"Examine: {term}"`, with a prior proportional to how often that term
/// occurs. Terms already used in the path are not proposed again, so a path never
/// revisits a term and the branching factor falls as the path lengthens.
///
/// That is enough for the engine to be exercised end to end against a real corpus —
/// the priors are informative, the paths are distinct, and the whole thing is
/// deterministic — and it is nowhere near enough to be mistaken for a language
/// model. Plug in a real proposer for real work; the trait is the point of the
/// module, this is the thing that makes it runnable without one.
#[derive(Debug, Clone, Default)]
pub struct MctsHeuristicGenerator {
    /// The most candidates to propose at any one node. Defaults to `4`.
    pub max_candidates: usize,
}

impl MctsHeuristicGenerator {
    /// A generator proposing at most `max_candidates` steps per node.
    #[must_use]
    pub fn new(max_candidates: usize) -> Self {
        Self { max_candidates }
    }

    /// The default budget when `max_candidates` is left at `0`.
    const DEFAULT_CANDIDATES: usize = 4;

    /// The prefix every proposed step carries. Used both to format a proposal and to
    /// recognize one already on the path, so the two can never drift apart.
    const STEP_PREFIX: &'static str = "Examine: ";
}

/// Lowercase alphanumeric terms of at least four characters — long enough to skip
/// most function words without needing a stopword list to maintain.
fn content_terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .map(str::to_lowercase)
        .collect()
}

impl MctsStepGenerator for MctsHeuristicGenerator {
    fn propose(
        &self,
        query: &str,
        path: &[String],
        context: &[SearchResult],
    ) -> Vec<MctsCandidate> {
        let budget = if self.max_candidates == 0 {
            Self::DEFAULT_CANDIDATES
        } else {
            self.max_candidates
        };

        // Term -> occurrences, over the retrieved context and the query.
        let mut counts: Vec<(String, usize)> = Vec::new();
        let corpus = context
            .iter()
            .map(|r| r.document.content.as_str())
            .chain(std::iter::once(query));
        for term in corpus.flat_map(content_terms) {
            match counts.iter_mut().find(|(t, _)| *t == term) {
                Some((_, n)) => *n += 1,
                None => counts.push((term, 1)),
            }
        }

        // A term already *examined* on this path is not a new step — but "examined"
        // means "the subject of a prior `Examine:` step this generator produced", not
        // "mentioned anywhere on the path". Keying off the generator's own proposal
        // format keeps the root's `Question: {query}` framing from masking every query
        // term (which would strand a search that has retrieved no external context, so
        // that the query's own terms are all it has to reason about).
        let used: Vec<String> = path
            .iter()
            .filter_map(|step| step.strip_prefix(Self::STEP_PREFIX))
            .map(str::to_lowercase)
            .collect();
        counts.retain(|(term, _)| !used.contains(term));

        // Descending frequency, then alphabetical — a total order, so the proposal
        // order (and hence which candidates progressive widening admits first) does
        // not depend on the iteration order of anything.
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        counts.truncate(budget);

        if counts.is_empty() {
            // Never strand the root: a corpus with no long terms would otherwise make
            // the root terminal and the search would have nothing to do.
            return Vec::new();
        }

        counts
            .into_iter()
            .map(|(term, n)| {
                // `n` is a term count, bounded by the corpus length; the cast is
                // exact for any corpus smaller than 2^53 terms.
                #[allow(clippy::cast_precision_loss)]
                MctsCandidate::with_prior(format!("{}{term}", Self::STEP_PREFIX), n as f64)
            })
            .collect()
    }
}

// ── MctsHeuristicEvaluator ────────────────────────────────────────────────────

/// A lexical, dependency-free [`MctsTerminalEvaluator`] — the default scorer.
///
/// The return of a path is the fraction of the query's content terms that the path's
/// steps, taken together, manage to cover — a crude *coverage* signal. It is
/// deliberately a signal and not a lookup: it rewards a path that gets to the query's
/// terms and it rewards getting to more of them, so it has a gradient for the search
/// to climb, which is what a rollout return has to have to be worth backing up at
/// all. It knows nothing about whether the reasoning is *correct*.
#[derive(Debug, Clone, Default)]
pub struct MctsHeuristicEvaluator;

impl MctsTerminalEvaluator for MctsHeuristicEvaluator {
    fn score(&self, query: &str, path: &[String], _context: &[SearchResult]) -> f64 {
        let wanted = content_terms(query);
        if wanted.is_empty() {
            return 0.0;
        }
        let covered: Vec<String> = path.iter().flat_map(|step| content_terms(step)).collect();
        let hits = wanted.iter().filter(|w| covered.contains(w)).count();
        // Both counts are term counts of a finite string; exact well below 2^53.
        #[allow(clippy::cast_precision_loss)]
        {
            hits as f64 / wanted.len() as f64
        }
    }
}
