//! **Monte-Carlo tree search over reasoning steps** — `UCT` and `PUCT` selection,
//! progressive widening, rollout simulation, and value backpropagation.
//!
//! Every other search in this crate decides where to look *next* using only what it
//! can see from where it is standing. This one decides using what it has learned from
//! everywhere it has already been.
//!
//! # The loop
//!
//! A reasoning problem is a tree: the root is the question, an edge is a reasoning
//! step, and a leaf is a completed chain that can be scored. The tree is far too
//! large to enumerate — with `B` plausible steps and `D` steps to make there are
//! `B^D` chains — so the only question that matters is **where to spend the next
//! unit of effort**. `MCTS` answers it by running the same four phases, over and
//! over, growing the tree asymmetrically toward whatever is paying off:
//!
//! 1. **Select.** From the root, walk down the tree it has already built, at each
//!    node choosing the child that maximizes a score which trades the child's
//!    *estimated value* against how *uncertain* that estimate still is. That trade is
//!    [`MctsSelectionPolicy`], and it is the entire algorithm in one line.
//! 2. **Expand.** At the frontier, attach one new child — but only if the node has
//!    earned it. See *progressive widening*, below.
//! 3. **Simulate.** From the new node, play the reasoning out to a conclusion using a
//!    cheap default policy, and score the completed chain. That score is the
//!    simulation's **return**.
//! 4. **Backpropagate.** Add that return to *every node on the path back to the
//!    root*, and count a visit at each. Every ancestor's value estimate is now a
//!    running average over everything the search has discovered beneath it.
//!
//! Then do it again. The estimates sharpen, the selection follows them, and the tree
//! deepens exactly where the returns are.
//!
//! # Distinct from the crate's other tree searches
//!
//! Distinct from `tree_of_thought`, whose `BFS`/beam and
//! `DFS` assign each node a value **once, at expansion, and never revise it**: this
//! module *backpropagates* simulated rollout returns up the tree, so a node's value is
//! a running average over everything discovered beneath it, and `UCT` spends the next
//! expansion where that average is most uncertain. Distinct from
//! `bandit_ranker`, whose `LinUCB` explores a **flat, fixed
//! arm set** via a ridge-regression confidence width; here the action set is the
//! tree's growing frontier and the bonus is driven by **visit counts**.
//!
//! The difference is not stylistic. A beam search that scores a node `0.4` at
//! expansion will still believe `0.4` after exploring a subtree full of `0.9`s
//! underneath it, because it has no mechanism for the discovery to travel back up.
//! [`MctsNode::visits`] and [`MctsNode::total_value`] *are* that mechanism, and
//! [`MctsTree::check_visit_invariant`] is the exact identity they must satisfy for it
//! to be working.
//!
//! # Selection: `UCT`, `PUCT`, and the one that is not `AlphaZero`'s
//!
//! [`MctsSelectionPolicy`] ships three real policies and one ablation:
//!
//! * [`Uct`](MctsSelectionPolicy::Uct) — `Q + c * sqrt(ln N(s) / N(s,a))`. Kocsis &
//!   Szepesvári (2006): `UCB1`, applied recursively at every node. The bonus is a
//!   confidence width, so the argmax is *optimism in the face of uncertainty*, and
//!   the number of pulls of a suboptimal child is bounded at `O(ln n)`.
//! * [`Puct`](MctsSelectionPolicy::Puct) — `Q + c * P * sqrt(N(s)) / (1 + N(s,a))`.
//!   The `AlphaGo`/`AlphaZero` form: a policy prior multiplies the bonus directly, so
//!   a trained proposer's opinion is trusted hard and early, then abandoned fast as
//!   real returns arrive.
//! * [`PriorUct`](MctsSelectionPolicy::PriorUct) — `UCT`'s bonus, scaled by the prior
//!   *relative to uniform*. This one **provably degenerates to
//!   [`Uct`](MctsSelectionPolicy::Uct) under a uniform prior**, bit for bit.
//! * [`UniformRandom`](MctsSelectionPolicy::UniformRandom) — descend at random. Not a
//!   search policy; the yardstick that `UCT`'s advantage is measured against, in the
//!   same role `epsilon = 1` plays in `bandit_ranker`.
//!
//! A note on a folk claim, because it is load-bearing here and it is false:
//! **`AlphaZero`'s `PUCT` does not reduce to `UCT` under a uniform prior**, for any
//! choice of exploration constant. Its bonus decays like `1/(1 + N(s,a))` and grows
//! like `sqrt(N(s))`; `UCT`'s decays like `1/sqrt(N(s,a))` and grows like
//! `sqrt(ln N(s))`. A uniform prior makes `PUCT`'s bonus *constant across the
//! children of a node*, which is not the same thing as making it `UCT`'s, and no
//! scalar reconciles two different functions of two free variables. That is why
//! [`PriorUct`](MctsSelectionPolicy::PriorUct) exists as a separate variant: it is the
//! member of the family that *does* nest `UCT`, and this module's tests assert both
//! facts — the reduction, and the non-reduction.
//!
//! # Progressive widening
//!
//! A reasoning-step generator can propose many continuations, and a sampled language
//! model can propose unboundedly many. Attaching them all to a node is the fastest way
//! to destroy a tree search: the visit budget spreads so thin that no child ever
//! accumulates the evidence to be preferred, `Q` stays noise, and `UCT` degenerates
//! into round-robin over a giant flat bandit. [`MctsWidening`] rations children against
//! evidence — a node with `N` visits may have at most `ceil(C * N^alpha)` children — so
//! that the samples per child grow without bound even as the tree widens.
//!
//! # Choosing the answer: visits, not value
//!
//! The final move is the root's **most-visited** child, not its highest-valued one.
//! `Q` is a mean over however many rollouts a child happened to receive, so a child
//! visited twice that got lucky twice outranks one visited four hundred times whose
//! mean has actually converged. The visit count is the search's own revealed
//! confidence. [`MctsTree::best_child_by_visits`] documents the exact, deterministic
//! tie-break — and *why* the obvious `max_by_key` is wrong.
//!
//! # Determinism
//!
//! The rollout is random and this crate takes no dependency on `rand`, so the module
//! carries [`MctsRng`], a seeded `SplitMix64`. Given [`MctsConfig::seed`], a search
//! reproduces its tree **bit-for-bit**: every node, every visit count, every
//! accumulated value. Tie-breaks are resolved by node id and are not randomized, so
//! the `RNG` is consumed by exactly two things — the rollout policy and the
//! `UniformRandom` ablation.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "mcts-reasoning")]
//! # {
//! use oxirag::mcts_reasoning::{
//!     MctsCandidate, MctsConfig, MctsEngine, MctsSelectionPolicy, MctsStepGenerator,
//!     MctsTerminalEvaluator,
//! };
//! use oxirag::types::SearchResult;
//!
//! // A three-step reasoning problem: pick a digit at each step, and be rewarded for
//! // how large the number you build is.
//! struct Digits;
//! impl MctsStepGenerator for Digits {
//!     fn propose(&self, _q: &str, _path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
//!         (0..3).map(|d| MctsCandidate::new(d.to_string())).collect()
//!     }
//! }
//! struct Bigger;
//! impl MctsTerminalEvaluator for Bigger {
//!     fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
//!         // Skip the root (the question); reward the digits chosen after it.
//!         let sum: f64 = path[1..].iter().filter_map(|s| s.parse::<f64>().ok()).sum();
//!         sum / (2.0 * (path.len() - 1) as f64)
//!     }
//! }
//!
//! let engine = MctsEngine::new(
//!     MctsConfig::default()
//!         .with_simulations(200)
//!         .with_max_depth(3)
//!         .with_selection(MctsSelectionPolicy::Uct { exploration: std::f64::consts::SQRT_2 }),
//! );
//! let out = engine.search_with("build the largest number", &Digits, &Bigger, &[]).unwrap();
//!
//! // The search should endorse the largest digit as its first step.
//! assert_eq!(out.best_path[1], "2");
//! // And the backpropagation identity holds exactly, everywhere.
//! assert!(out.tree.check_visit_invariant());
//! # }
//! ```
//!
//! # References
//!
//! * Kocsis, L. & Szepesvári, C. (2006). *Bandit based Monte-Carlo Planning.* `ECML`.
//! * Coulom, R. (2007). *Computing Elo Ratings of Move Patterns in the Game of Go.*
//!   (progressive widening)
//! * Rosin, C. (2011). *Multi-armed bandits with episode context.* `AMAI`. (`PUCB`)
//! * Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* `IEEE
//!   TCIAIG`.
//! * Silver, D. et al. (2017). *Mastering the game of Go without human knowledge.*
//!   `Nature`. (`AlphaZero`'s `PUCT`)

pub mod engine;
pub mod rng;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::MctsEngine;
pub use rng::MctsRng;
pub use types::{
    MctsCandidate, MctsConfig, MctsError, MctsHeuristicEvaluator, MctsHeuristicGenerator, MctsNode,
    MctsOutput, MctsRolloutPolicy, MctsSelectionPolicy, MctsStats, MctsStepGenerator,
    MctsTerminalEvaluator, MctsTree, MctsWidening,
};
