//! Contextual multi-armed bandits for **online** ranking: choosing, adapting, and
//! learning which retrieval strategy to use — per request, from feedback, while
//! serving traffic.
//!
//! Every other ranking module in this crate answers "given these candidates, what
//! is the best order?" This one answers a different and strictly harder question:
//! **"which of my strategies should I use right now, when I do not yet know which
//! is best, and the only way to find out is to try one and see what happens?"**
//!
//! That is the *exploration/exploitation dilemma*, and it is unavoidable in any
//! system that learns from its own decisions. Exploit — always serve the strategy
//! that looks best so far — and you may lock forever onto a mediocre one, because
//! you never gathered the evidence that would have changed your mind. Explore —
//! keep trying everything — and you keep paying for choices you already know are
//! worse. The bandit algorithms here are the principled resolutions of that
//! tension, and the property that distinguishes them from guessing is
//! **sublinear regret**: the guarantee that the policy's average per-round payoff
//! converges to that of an oracle who knew the right answer from the start.
//!
//! # The setting
//!
//! On each request the system observes a **context** `x` (a [`BanditContext`] — a
//! feature vector describing the query: embedding statistics, length, user
//! segment, whatever is available), chooses one of `K` **arms** (a [`BanditArm`] —
//! a retrieval strategy, a reranker, a prompt template, a candidate document for
//! the top slot), serves it, and later observes a **reward** (a click, a
//! thumbs-up, a downstream answer-quality score).
//!
//! All three policies here model each arm `a` as a **disjoint linear** payoff,
//! `E[reward | a, x] = theta*_a^T x`, and maintain a per-arm ridge regression
//!
//! ```text
//! A_a = lambda * I + sum x x^T ,   b_a = sum r * x ,   theta_hat_a = A_a^-1 b_a
//! ```
//!
//! They differ *only* in how they turn that shared model into a score:
//!
//! | Policy | Score for arm `a` | How it explores |
//! |---|---|---|
//! | [`LinUcbRanker`] | `theta_hat^T x + alpha * sqrt(x^T A^-1 x)` | **Optimism** — act on the top of the confidence interval, so an uncertain arm looks *better* than it is until it has been checked. |
//! | [`ThompsonSamplingRanker`] | `theta_tilde^T x`, `theta_tilde ~ N(theta_hat, v^2 A^-1)` | **Posterior sampling** — pull each arm with exactly the probability that it is optimal. |
//! | [`EpsilonGreedyRanker`] | `theta_hat^T x` | **A coin** — with probability `eps_t`, ignore the model and rank at random. |
//!
//! The first two explore *where the uncertainty actually is*: `A_a^-1` is small in
//! directions that have been observed often and large in directions that have not,
//! so their exploration is aimed. ε-greedy explores blindly and uniformly, which is
//! why it is here as a **baseline** — the yardstick against which the others'
//! advantage is measured, and whose `eps = 0` setting is the pure-greedy policy
//! this module's tests trap into a deceptive arm to prove exploration is doing real
//! work.
//!
//! # How this differs from `ab_eval` and `active_learning_retrieval`
//!
//! Two modules in this crate sound adjacent and are not. Neither is a bandit;
//! neither makes a sequential decision under uncertainty; and this module does not
//! replace, wrap, or compete with either.
//!
//! | Module | Regime | Decides | Learns from | Has a reward? | Has regret? |
//! |---|---|---|---|---|---|
//! | [`ab_eval`](crate::ab_eval) | **offline, one-shot** | *Is system A better than B?* | two pre-computed score vectors over the same queries | no — it consumes scores, it does not receive payoffs | no |
//! | [`active_learning_retrieval`](crate::active_learning_retrieval) | **offline, pool-based** | *Which items should a human label next?* | a fixed pool of already-scored candidates | no — it maximizes *informativeness*, not payoff | no |
//! | `bandit_ranker` (this module) | **online, sequential** | *Which arm do I serve on this very request?* | its own actions' rewards, one at a time, forever | **yes** — that is the whole input | **yes** — and it is the thing being minimized |
//!
//! [`ab_eval`](crate::ab_eval) is a paired significance test: given two vectors of
//! per-query scores it reports a bootstrap confidence interval and a p-value. It
//! runs *once*, after both systems have already been evaluated; it never chooses an
//! action and it never sees a reward. It answers "did A win?", which is the question
//! you ask *after* an experiment. This module answers "which should I serve next?",
//! which is the question you ask *during* one — and, unlike a fixed A/B split, it
//! shifts traffic toward the winner *while the experiment is still running*, which
//! is precisely the cost that sublinear regret bounds.
//!
//! [`active_learning_retrieval`](crate::active_learning_retrieval) is uncertainty
//! sampling over a static pool: it scores candidates by margin or entropy and
//! returns the most *ambiguous* batch for a human to label. Its objective is
//! information gain and nothing else — it is *indifferent* to whether the items it
//! picks are any good, which is exactly right for choosing labeling work and
//! exactly wrong for choosing what to serve a user. A bandit cannot be indifferent
//! to that: every exploratory pull is served to a real request and paid for in real
//! reward. The exploration/exploitation *tradeoff* — the thing this module exists
//! to manage — simply does not arise in a pure-information objective, because there
//! is nothing to trade off against.
//!
//! # Determinism
//!
//! Thompson sampling and ε-greedy are randomized, and this crate takes no
//! dependency on `rand`. Both draw from [`SplitMix64Rng`], a seeded `SplitMix64`
//! generator with Box–Muller normals, so a given [`BanditConfig::seed`] replayed
//! against the same contexts and rewards produces a **byte-identical** action
//! sequence. A production exploration trajectory can be reproduced exactly from its
//! logged seed. (This follows the convention [`ab_eval`](crate::ab_eval) already
//! sets, deriving its bootstrap resamples from a hash rather than from an RNG.)
//!
//! # Numerics
//!
//! The inverse `A_a^-1` is what both the score and the exploration bonus need, on
//! every round, for every arm — and it is **never** recomputed by inverting `A`.
//! It is maintained incrementally in `O(d^2)` by the **Sherman–Morrison** rank-1
//! identity, with a symmetrization step that prevents floating-point drift out of
//! the symmetric (and hence out of the positive-definite) cone, a clamp on the
//! quadratic form before every `sqrt`, and a scale-aware Cholesky jitter for the
//! posterior draw. Every one of those guards is derived and justified in
//! [`linalg`], and the incremental inverse is checked against an independent
//! Gauss–Jordan inverse to ~1e-9 in this module's tests. The linear algebra is
//! hand-rolled `std`-only `f64` (no `ndarray`), which the small `d` of a contextual
//! bandit makes not merely acceptable but correct.
//!
//! # Evaluating one
//!
//! * [`BanditRegretTracker`] — cumulative regret against a known oracle, plus the
//!   window-average curve that reveals whether regret is genuinely sublinear. A
//!   bandit with linear regret still returns rankings, still fits a model, and
//!   still reports respectable-looking statistics; the *shape of this curve* is
//!   what exposes it.
//! * [`BanditOffPolicyEvaluator`] — unbiased offline evaluation by **replay** (Li
//!   et al., 2011): walk a log of uniformly-random actions, keep only the events
//!   where the policy's choice matches the logged one, and average their rewards.
//!   Because the logged action was uniform, the acceptance probability is a
//!   constant `1/K` — independent of the context *and* of the arm the policy chose
//!   — so the accepted subsequence is a genuine sample path of the policy and its
//!   mean reward is an unbiased estimate of the policy's online value. Evaluate a
//!   new ranker against last month's traffic without deploying it.
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "bandit-ranker")]
//! # {
//! use oxirag::bandit_ranker::{
//!     BanditConfig, BanditContext, BanditRanker, BanditRegretTracker, LinUcbRanker,
//! };
//!
//! // Three retrieval strategies, contexts of dimension 3.
//! let config = BanditConfig::with_dimension(3)
//!     .expect("dimension 3 is valid")
//!     .set_alpha(0.6);
//! let mut ranker = LinUcbRanker::from_ids(config, ["dense", "bm25", "hybrid"])
//!     .expect("three distinct arm ids");
//!
//! // The environment's true coefficients — unknown to the ranker.
//! let truth = [
//!     ("dense", [0.20_f64, 0.10, 0.05]),
//!     ("bm25", [0.05, 0.30, 0.05]),
//!     ("hybrid", [0.30, 0.30, 0.30]), // strictly best on non-negative contexts
//! ];
//! let expected = |arm: &str, x: &[f64]| -> f64 {
//!     let theta = truth.iter().find(|(id, _)| *id == arm).expect("known arm").1;
//!     theta.iter().zip(x).map(|(t, xi)| t * xi).sum()
//! };
//!
//! let mut regret = BanditRegretTracker::new();
//! for step in 0..400 {
//!     // A deterministic sweep of contexts, standing in for real traffic.
//!     let phase = f64::from(step) * 0.37;
//!     let features = vec![phase.sin().abs(), phase.cos().abs(), 0.5];
//!     let context = BanditContext::new(features).expect("finite features");
//!
//!     let ranking = ranker.select(&context).expect("arms exist");
//!     let chosen = ranking.top_arm_id().expect("non-empty ranking").to_string();
//!
//!     let mean = expected(&chosen, context.features());
//!     let best = truth
//!         .iter()
//!         .map(|(id, _)| expected(id, context.features()))
//!         .fold(f64::MIN, f64::max);
//!     regret.record(best, mean);
//!
//!     ranker.update(&chosen, &context, mean).expect("known arm");
//! }
//!
//! // It found the dominant strategy, and its average regret is falling.
//! assert_eq!(ranker.stats().most_pulled_arm(), Some("hybrid"));
//! let windows = regret.window_average_regret(4);
//! assert!(windows[3] < windows[0]);
//! # }
//! ```

pub mod epsilon_greedy;
pub mod evaluation;
pub mod linalg;
pub mod linucb;
pub mod model;
pub mod rng;
pub mod thompson;
pub mod types;

#[cfg(test)]
mod tests;

pub use epsilon_greedy::EpsilonGreedyRanker;
pub use evaluation::{
    BanditLoggedEvent, BanditOffPolicyEstimate, BanditOffPolicyEvaluator, BanditRegretTracker,
};
pub use linalg::LinalgError;
pub use linucb::LinUcbRanker;
pub use model::{BanditArmSet, BanditLinearModel};
pub use rng::SplitMix64Rng;
pub use thompson::ThompsonSamplingRanker;
pub use types::{
    BanditArm, BanditArmStats, BanditConfig, BanditContext, BanditError, BanditRankedArm,
    BanditRanker, BanditRanking, BanditResult, BanditStats,
};
