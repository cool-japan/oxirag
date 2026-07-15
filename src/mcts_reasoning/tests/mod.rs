#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::doc_markdown
)]
//! Tests for Monte-Carlo tree search over reasoning steps.
//!
//! Six of these carry the module's actual claims, and none of them is satisfiable by
//! an implementation that merely compiles and returns a plausible tree:
//!
//! * `uct_scores_match_hand_computed_arithmetic_bit_for_bit` — the UCT score of a
//!   three-child fixture is checked against constants derived offline, by `to_bits`
//!   equality. The fixture is chosen so the argmax is the *under-explored* child and
//!   not the best-mean one: an implementation that dropped the exploration term, or
//!   put the visit counts the wrong way up, still produces a working-looking search
//!   and fails here.
//! * `uct_converges_to_the_known_optimal_leaf` and
//!   `uct_finds_the_optimum_with_far_fewer_simulations_than_random_descent` — on a
//!   tree whose optimal leaf is known by construction, UCT must find it, and must
//!   find it *much* sooner than the same estimator with the tree policy removed.
//!   That ablation is the only thing separating "a search" from "an expensive random
//!   sampler that keeps statistics".
//! * `uct_cumulative_regret_is_sublinear` — regret at `4n` must be less than `4x`
//!   regret at `n`, while the random baseline's is `~4x` on the nose. Linear regret
//!   means the tree policy is not learning, and it is invisible in every other
//!   statistic the search reports.
//! * `visit_invariant_holds_exactly_after_every_search` — `N(s) = 1 + sum_c N(c)` at
//!   every internal node, and the root's accumulated value equals the in-order sum of
//!   every rollout return, *bit for bit*. A backup that skips the root, double-counts
//!   the expanded node, or stops one short of the leaf still produces a plausible
//!   tree and fails here.
//! * `prior_uct_reduces_to_uct_exactly_under_a_uniform_prior` and
//!   `alphazero_puct_does_not_reduce_to_uct_under_a_uniform_prior` — the reduction
//!   this module claims, and the reduction it explicitly refuses to claim, are *both*
//!   asserted. The second is the interesting one: on one fixture, with a uniform
//!   prior and the same constant, PUCT and UCT choose **different children**.
//! * `same_seed_reproduces_the_tree_bit_for_bit` — the reproducibility contract.

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::mcts_reasoning::engine::MctsEngine;
use crate::mcts_reasoning::rng::MctsRng;
use crate::mcts_reasoning::types::{
    MctsCandidate, MctsConfig, MctsError, MctsHeuristicEvaluator, MctsHeuristicGenerator, MctsNode,
    MctsRolloutPolicy, MctsSelectionPolicy, MctsStepGenerator, MctsTerminalEvaluator, MctsTree,
    MctsWidening,
};
use crate::types::{Document, DocumentId, SearchResult};
use async_trait::async_trait;
use std::f64::consts::SQRT_2;

// ═════════════════════════════════════════════════════════════════════════════
// Fixtures: a mock Echo, and two worlds with known answers
// ═════════════════════════════════════════════════════════════════════════════

struct MockEcho {
    results: Vec<SearchResult>,
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Echo for MockEcho {
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError> {
        Ok(document.id.clone())
    }
    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        Ok(documents.into_iter().map(|d| d.id).collect())
    }
    async fn search(
        &self,
        _query: &str,
        _top_k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        Ok(self.results.clone())
    }
    async fn get(&self, _id: &DocumentId) -> Result<Option<Document>, EmbeddingError> {
        Ok(None)
    }
    async fn delete(&mut self, _id: &DocumentId) -> Result<bool, EmbeddingError> {
        Ok(false)
    }
    async fn count(&self) -> usize {
        self.results.len()
    }
    async fn clear(&mut self) -> Result<(), EmbeddingError> {
        Ok(())
    }
}

fn make_result(content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content),
        score,
        rank: 0,
    }
}

/// **World 1 — a flat bandit.** The root's children are `K` terminal arms with known,
/// deterministic values. This is exactly the setting `UCB1`'s regret bound is stated
/// for, so it is the setting in which that bound can be *measured*.
struct FlatArms {
    values: Vec<f64>,
}

impl FlatArms {
    fn arm_index(step: &str) -> Option<usize> {
        step.strip_prefix("arm")?.parse().ok()
    }
    /// The value of the arm a root action leads to.
    fn value_of(&self, node_content: &str) -> f64 {
        Self::arm_index(node_content).map_or(0.0, |i| self.values[i])
    }
    fn best(&self) -> f64 {
        self.values.iter().copied().fold(f64::MIN, f64::max)
    }
}

impl MctsStepGenerator for FlatArms {
    fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
        if path.len() == 1 {
            (0..self.values.len())
                .map(|i| MctsCandidate::new(format!("arm{i}")))
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl MctsTerminalEvaluator for FlatArms {
    fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
        path.get(1).map_or(0.0, |s| self.value_of(s))
    }
}

/// **World 2 — a golden path.** A `depth`-deep, `branching`-wide tree in which exactly
/// one chain of choices is correct. A path's return is the fraction of its *leading*
/// steps that are correct, so the optimal leaf has value exactly `1.0` and is unique,
/// and there is a gradient at every level for a search to climb.
///
/// The golden choice is the **last** candidate proposed at every level, which makes
/// the fixture adversarial on purpose: the "lowest id wins" tie-break and progressive
/// widening both reach for the *first* candidate, so neither can stumble onto the
/// answer by accident. Only the returns can find it.
///
/// A uniformly random playout reaches the optimal leaf with probability
/// `(1/branching)^depth` — `1/243` at the settings used below.
struct GoldenPath {
    depth: usize,
    branching: usize,
}

impl GoldenPath {
    /// The correct choice at every level: the last one proposed.
    fn golden(&self) -> usize {
        self.branching - 1
    }
    fn golden_step(&self, level: usize) -> String {
        format!("s{level}c{}", self.golden())
    }
    /// The full optimal chain, root excluded — the thing the search has to find.
    fn golden_steps(&self) -> Vec<String> {
        (0..self.depth).map(|l| self.golden_step(l)).collect()
    }
    /// The probability a uniformly random playout lands on the optimal leaf.
    fn random_hit_rate(&self) -> f64 {
        (1.0 / self.branching as f64).powi(self.depth as i32)
    }
}

impl MctsStepGenerator for GoldenPath {
    fn propose(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> Vec<MctsCandidate> {
        let taken = path.len() - 1;
        if taken >= self.depth {
            return Vec::new();
        }
        (0..self.branching)
            .map(|i| MctsCandidate::new(format!("s{taken}c{i}")))
            .collect()
    }
}

impl MctsTerminalEvaluator for GoldenPath {
    fn score(&self, _q: &str, path: &[String], _c: &[SearchResult]) -> f64 {
        let mut correct = 0usize;
        for (level, step) in path.iter().skip(1).enumerate() {
            if *step == self.golden_step(level) {
                correct += 1;
            } else {
                break;
            }
        }
        correct as f64 / self.depth as f64
    }
}

/// A node built by hand, for the arithmetic fixtures.
fn make_node(id: usize, visits: u32, total_value: f64, prior: f64, candidates: usize) -> MctsNode {
    MctsNode {
        id,
        parent: if id == 0 { None } else { Some(0) },
        children: Vec::new(),
        candidates: (0..candidates)
            .map(|i| MctsCandidate::new(format!("c{i}")))
            .collect(),
        content: format!("n{id}"),
        depth: usize::from(id != 0),
        visits,
        total_value,
        prior,
        terminal: false,
    }
}

/// The **canonical fixture**: a parent with `N(s) = 6` and three children whose visit
/// counts sum to `5`. It satisfies the backpropagation invariant by construction
/// (`6 = 1 + 3 + 1 + 1`), so it is a tree the engine could actually have produced.
///
/// | child | `N(s,a)` | `W` | `Q` |
/// |---|---|---|---|
/// | `a` (id 1) | 3 | 2.4 | 0.8 — the **best mean**, and heavily exploited |
/// | `b` (id 2) | 1 | 0.5 | 0.5 — mediocre, and barely tried |
/// | `c` (id 3) | 1 | 0.2 | 0.2 — poor, and barely tried |
fn canonical_fixture() -> MctsTree {
    let mut parent = make_node(0, 6, 0.0, 1.0, 3);
    parent.children = vec![1, 2, 3];
    MctsTree {
        nodes: vec![
            parent,
            make_node(1, 3, 2.4, 1.0 / 3.0, 0),
            make_node(2, 1, 0.5, 1.0 / 3.0, 0),
            make_node(3, 1, 0.2, 1.0 / 3.0, 0),
        ],
    }
}

/// The argmax of a policy over the canonical fixture's children.
fn argmax_child(tree: &MctsTree, policy: &MctsSelectionPolicy) -> usize {
    let parent = &tree.nodes[0];
    let mut best = (usize::MAX, f64::NEG_INFINITY);
    for &id in &parent.children {
        let score = policy.score(parent, &tree.nodes[id]);
        if score > best.1 {
            best = (id, score);
        }
    }
    best.0
}

mod robustness;
mod search;
mod selection;
