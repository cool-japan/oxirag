//! The Monte-Carlo tree search engine: select, expand, simulate, backpropagate.

use crate::mcts_reasoning::rng::MctsRng;
use crate::mcts_reasoning::types::{
    MctsCandidate, MctsConfig, MctsError, MctsHeuristicEvaluator, MctsHeuristicGenerator, MctsNode,
    MctsOutput, MctsRolloutPolicy, MctsSelectionPolicy, MctsStats, MctsStepGenerator,
    MctsTerminalEvaluator, MctsTree,
};
use crate::types::SearchResult;

/// The root is always node `0` — the arena is filled root-first and never reordered.
const ROOT: usize = 0;

// ── MctsEngine ────────────────────────────────────────────────────────────────

/// Monte-Carlo tree search over reasoning steps.
///
/// One [`search_with`](MctsEngine::search_with) call runs
/// [`MctsConfig::simulations`] iterations of the four-phase loop — **select**,
/// **expand**, **simulate**, **backpropagate** — and returns the tree it built along
/// with the reasoning path it endorses.
///
/// The [`Echo`] retrieval layer is supplied *per call* to the async
/// [`run`](MctsEngine::run) / [`run_with`](MctsEngine::run_with) methods; the
/// synchronous [`search_with`](MctsEngine::search_with) takes an already-retrieved
/// context and is the whole algorithm, testable without any I/O.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
#[derive(Debug, Clone, Default)]
pub struct MctsEngine {
    /// Configuration for this engine.
    pub config: MctsConfig,
}

impl MctsEngine {
    /// Create an engine with the given config.
    #[must_use]
    pub fn new(config: MctsConfig) -> Self {
        Self { config }
    }

    /// Run the search against an explicit generator, evaluator, and context.
    ///
    /// This is the algorithm, with no retrieval and no I/O: everything the search
    /// knows comes from the two traits and the `context` slice.
    ///
    /// # Errors
    ///
    /// Returns [`MctsError::InvalidConfig`] if [`MctsConfig::validate`] rejects the
    /// configuration, [`MctsError::EmptyQuery`] if `query` is blank, and
    /// [`MctsError::NoCandidates`] if `generator` proposes no step at the root — a
    /// search with no first move is not a search.
    pub fn search_with<G, V>(
        &self,
        query: &str,
        generator: &G,
        evaluator: &V,
        context: &[SearchResult],
    ) -> Result<MctsOutput, MctsError>
    where
        G: MctsStepGenerator + ?Sized,
        V: MctsTerminalEvaluator + ?Sized,
    {
        self.config.validate()?;
        if query.trim().is_empty() {
            return Err(MctsError::EmptyQuery);
        }

        let mut run = MctsRun {
            config: &self.config,
            query,
            generator,
            evaluator,
            context,
            tree: MctsTree::default(),
            rng: MctsRng::new(self.config.seed),
            stats: MctsStats::default(),
        };

        run.build_root()?;
        for _ in 0..self.config.simulations {
            run.simulate();
        }
        Ok(run.finish())
    }

    /// Retrieve context for `query`, then search with the default lexical generator
    /// and evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`MctsError::RetrievalFailed`] if the [`Echo`] layer fails, plus
    /// everything [`search_with`](MctsEngine::search_with) can return.
    ///
    /// [`Echo`]: crate::layer1_echo::traits::Echo
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<MctsOutput, MctsError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        self.run_with(
            query,
            echo,
            &MctsHeuristicGenerator::default(),
            &MctsHeuristicEvaluator,
        )
        .await
    }

    /// Retrieve context for `query`, then search with the supplied generator and
    /// evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`MctsError::RetrievalFailed`] if the [`Echo`] layer fails, plus
    /// everything [`search_with`](MctsEngine::search_with) can return.
    ///
    /// [`Echo`]: crate::layer1_echo::traits::Echo
    pub async fn run_with<E, G, V>(
        &self,
        query: &str,
        echo: &E,
        generator: &G,
        evaluator: &V,
    ) -> Result<MctsOutput, MctsError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
        G: MctsStepGenerator + ?Sized,
        V: MctsTerminalEvaluator + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(MctsError::EmptyQuery);
        }
        let context: Vec<SearchResult> = echo
            .search(query, self.config.top_k, None)
            .await
            .map_err(|e| MctsError::RetrievalFailed(e.to_string()))?;
        self.search_with(query, generator, evaluator, &context)
    }
}

// ── MctsRun ───────────────────────────────────────────────────────────────────

/// One search in progress: the tree, the `RNG`, the statistics, and borrows of
/// everything the four phases need.
///
/// This exists so the phases can be methods with two arguments instead of free
/// functions with eight.
struct MctsRun<'a, G: ?Sized, V: ?Sized> {
    config: &'a MctsConfig,
    query: &'a str,
    generator: &'a G,
    evaluator: &'a V,
    context: &'a [SearchResult],
    tree: MctsTree,
    rng: MctsRng,
    stats: MctsStats,
}

impl<G, V> MctsRun<'_, G, V>
where
    G: MctsStepGenerator + ?Sized,
    V: MctsTerminalEvaluator + ?Sized,
{
    // ── setup ─────────────────────────────────────────────────────────────────

    /// Create the root, and roll out from it once.
    ///
    /// That first rollout is not decoration. It is what makes the backpropagation
    /// invariant `N(s) = 1 + sum_c N(c)` hold at the **root** as well as everywhere
    /// else (see [`MctsTree::check_visit_invariant`]), and it means the root's `Q` is
    /// defined before the first simulation asks for it. The cost is one extra rollout
    /// per search.
    fn build_root(&mut self) -> Result<(), MctsError> {
        let content = format!("Question: {}", self.query);
        let root = self.create_node(None, content, 1.0, 0);

        // The root is terminal only if the generator proposed nothing (its depth is
        // `0`, and `validate` has already established `max_depth >= 1`).
        if self.tree.nodes[root].terminal {
            return Err(MctsError::NoCandidates);
        }

        let value = self.rollout(root);
        self.backpropagate(root, value);
        self.stats.rollout_returns.push(value);
        Ok(())
    }

    /// Push a node into the arena, asking the generator for its candidate steps.
    ///
    /// The generator is called **once per node**, here, and never again for that
    /// node: progressive widening draws from the list this produces. A node at
    /// [`MctsConfig::max_depth`] is terminal by depth and the generator is not called
    /// at all.
    fn create_node(
        &mut self,
        parent: Option<usize>,
        content: String,
        prior: f64,
        depth: usize,
    ) -> usize {
        let id = self.tree.nodes.len();

        let candidates = if depth >= self.config.max_depth {
            Vec::new()
        } else {
            let mut path = match parent {
                Some(p) => self.tree.path_contents(p),
                None => Vec::new(),
            };
            path.push(content.clone());
            normalize_priors(self.generator.propose(self.query, &path, self.context))
        };

        // A state with no proposed continuation is a dead end, exactly as a state at
        // the depth limit is: no child will ever hang off it, and every future visit
        // re-evaluates it in place.
        let terminal = candidates.is_empty();

        self.tree.nodes.push(MctsNode {
            id,
            parent,
            children: Vec::new(),
            candidates,
            content,
            depth,
            visits: 0,
            total_value: 0.0,
            prior,
            terminal,
        });
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
        id
    }

    // ── the four phases ───────────────────────────────────────────────────────

    /// One simulation: select down to a leaf, expand it, roll out, back up.
    fn simulate(&mut self) {
        // ── 1. SELECTION (and 2. EXPANSION, which ends the descent) ────────────
        let mut node_id = ROOT;
        let mut root_action: Option<usize> = None;

        loop {
            if self.tree.nodes[node_id].terminal {
                // A terminal leaf is re-evaluated in place; there is nowhere to go.
                break;
            }

            let limit = self.child_limit(node_id);
            let node = &self.tree.nodes[node_id];

            if node.children.len() < limit && !node.fully_expanded() {
                let child = self.expand(node_id);
                if node_id == ROOT {
                    root_action = Some(child);
                }
                node_id = child;
                // The freshly created node is this simulation's leaf: roll out from
                // it rather than descending further. This is what gives every node
                // exactly one "creation rollout".
                break;
            }

            let Some(child) = self.select_child(node_id) else {
                break;
            };
            if node_id == ROOT {
                root_action = Some(child);
            }
            node_id = child;
        }

        // ── 3. SIMULATION ─────────────────────────────────────────────────────
        let value = self.rollout(node_id);

        // ── 4. BACKPROPAGATION ────────────────────────────────────────────────
        self.backpropagate(node_id, value);

        self.stats.rollout_returns.push(value);
        if let Some(action) = root_action {
            self.stats.root_action_history.push(action);
        }
    }

    /// How many children this node is currently allowed, under progressive widening.
    ///
    /// Never zero for a non-terminal node: [`MctsWidening::limit`] floors at `1`, and
    /// a non-terminal node has at least one candidate to spend that allowance on. So
    /// the descent can always either widen or select, and can never stall — which is
    /// the precondition of the visit invariant.
    ///
    /// [`MctsWidening::limit`]: crate::mcts_reasoning::MctsWidening::limit
    fn child_limit(&self, node_id: usize) -> usize {
        let node = &self.tree.nodes[node_id];
        match &self.config.widening {
            Some(widening) => widening.limit(node.visits).min(node.action_count()),
            // No widening: attach one new child per visit until every candidate is a
            // child — classical MCTS expansion.
            None => node.action_count(),
        }
    }

    /// Grow the next untried candidate of `parent_id` into a child.
    ///
    /// Candidates are admitted in the order the generator proposed them, so
    /// `children[i]` always corresponds to `candidates[i]` and `children` is in
    /// ascending id order — which is what the "lowest id wins" tie-breaks rely on.
    fn expand(&mut self, parent_id: usize) -> usize {
        let (content, prior, depth) = {
            let parent = &self.tree.nodes[parent_id];
            let next = parent.children.len();
            let candidate = &parent.candidates[next];
            (candidate.content.clone(), candidate.prior, parent.depth + 1)
        };

        let child_id = self.create_node(Some(parent_id), content, prior, depth);
        self.tree.nodes[parent_id].children.push(child_id);
        child_id
    }

    /// The tree policy: score every child and take the argmax.
    ///
    /// Returns `None` only when `node_id` has no children at all.
    ///
    /// # Tie-break
    ///
    /// The fold replaces the incumbent only on a **strict** improvement, and
    /// `children` is in ascending id order, so the **first** (lowest-id, earliest
    /// proposed) maximal child wins. `max_by_key` would return the *last* one, which
    /// would make the search's behaviour depend on arena fill order.
    fn select_child(&mut self, node_id: usize) -> Option<usize> {
        // `self.tree` and `self.rng` are disjoint fields, so both borrows coexist.
        let node = &self.tree.nodes[node_id];
        if node.children.is_empty() {
            return None;
        }

        if matches!(self.config.selection, MctsSelectionPolicy::UniformRandom) {
            let index = self.rng.next_usize_below(node.children.len())?;
            return node.children.get(index).copied();
        }

        let mut best: Option<(usize, f64)> = None;
        for &child_id in &node.children {
            let Some(child) = self.tree.nodes.get(child_id) else {
                continue;
            };
            let score = self.config.selection.score(node, child);
            let better = match best {
                // A non-finite score is unreachable — returns are clamped to `[0, 1]`
                // and every existing child has `N >= 1` — but if one ever arose it
                // must not be allowed to win an argmax by poisoning the comparison.
                None => score.is_finite(),
                Some((_, incumbent)) => score > incumbent,
            };
            if better {
                best = Some((child_id, score));
            }
        }

        // The `or_else` cannot fire in practice (see above); it is here so that the
        // descent is *total*. A `None` return would stop the simulation at an
        // internal node, which would silently break the visit invariant rather than
        // failing loudly.
        best.map(|(id, _)| id)
            .or_else(|| node.children.first().copied())
    }

    /// Play a random continuation from `node_id` to a terminal state and score it.
    ///
    /// The node's own candidate list is reused for the first step (it was computed
    /// when the node was created), so a rollout of length `L` costs `L` generator
    /// calls rather than `L + 1`.
    ///
    /// The return is **clamped into `[0, 1]`**, and a non-finite score is mapped to
    /// `0.0`. See [`clamp_return`].
    fn rollout(&mut self, node_id: usize) -> f64 {
        let mut path = self.tree.path_contents(node_id);
        let (mut depth, mut candidates) = {
            let node = &self.tree.nodes[node_id];
            (node.depth, node.candidates.clone())
        };

        let ceiling = match self.config.rollout_depth_cap {
            Some(cap) => self.config.max_depth.min(depth.saturating_add(cap)),
            None => self.config.max_depth,
        };

        while depth < ceiling && !candidates.is_empty() {
            let Some(index) = self.sample_step(&candidates) else {
                break;
            };
            let Some(step) = candidates.get(index) else {
                break;
            };
            path.push(step.content.clone());
            depth += 1;

            candidates = if depth < ceiling {
                normalize_priors(self.generator.propose(self.query, &path, self.context))
            } else {
                Vec::new()
            };
        }

        clamp_return(self.evaluator.score(self.query, &path, self.context))
    }

    /// The default policy: which candidate the rollout takes.
    fn sample_step(&mut self, candidates: &[MctsCandidate]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        match self.config.rollout {
            MctsRolloutPolicy::Uniform => self.rng.next_usize_below(candidates.len()),
            MctsRolloutPolicy::PriorWeighted => {
                let total: f64 = candidates.iter().map(|c| c.prior.max(0.0)).sum();
                if !total.is_finite() || total <= 0.0 {
                    // No usable prior mass — fall back to uniform rather than
                    // silently always taking the first step.
                    return self.rng.next_usize_below(candidates.len());
                }
                // Inverse-transform sampling on the cumulative prior. `next_f64` is
                // in `[0, 1)`, so `mass` is in `[0, total)` and some candidate always
                // claims it.
                let mut mass = self.rng.next_f64() * total;
                for (index, candidate) in candidates.iter().enumerate() {
                    mass -= candidate.prior.max(0.0);
                    if mass < 0.0 {
                        return Some(index);
                    }
                }
                // Unreachable except for accumulated rounding in the running
                // subtraction; the last candidate owns the residue.
                Some(candidates.len() - 1)
            }
        }
    }

    /// Add `value` to every node from `from` up to the root, and count the visit.
    ///
    /// This is the phase that makes the whole thing a *search* rather than an
    /// enumeration: one rollout's return updates the value estimate of every state on
    /// the path that produced it, so a node's `Q` becomes a running average over
    /// everything discovered beneath it, and the tree policy above it immediately
    /// starts acting on the new evidence.
    ///
    /// The `visits` counter is a `u32`; [`MctsConfig::validate`] has already refused
    /// any budget that could overflow it (a node's visit count is bounded by
    /// `simulations + 1`).
    fn backpropagate(&mut self, from: usize, value: f64) {
        let mut cursor = Some(from);
        while let Some(id) = cursor {
            let Some(node) = self.tree.nodes.get_mut(id) else {
                break;
            };
            node.visits += 1;
            node.total_value += value;
            cursor = node.parent;
        }
    }

    // ── teardown ──────────────────────────────────────────────────────────────

    /// Read the answer off the finished tree.
    fn finish(mut self) -> MctsOutput {
        self.stats.simulations = self.config.simulations;
        self.stats.nodes_created = self.tree.nodes.len();

        let best_root_action = self.tree.best_child_by_visits(ROOT);
        let best_path = self.tree.principal_variation_contents();
        let answer = synthesize(self.query, &best_path, self.context);

        MctsOutput {
            answer,
            best_path,
            best_root_action,
            tree: self.tree,
            stats: self.stats,
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Normalize a generator's candidate priors to a probability distribution.
///
/// Each weight is floored at `0` and divided by their sum. If that sum is not finite
/// and strictly positive — every weight zero, negative, `NaN`, or infinite — the
/// priors are replaced by the **uniform** distribution `1/K`.
///
/// The fallback is not a formality. A prior of `NaN` reaches
/// [`MctsSelectionPolicy::Puct`]'s bonus, makes every comparison in the argmax
/// return `false`, and turns the tree policy into "always take the first child" with
/// no error and no symptom other than a search that quietly stops working. Uniform is
/// the honest reading of "the generator told us nothing".
///
/// Uniform *weights* (the [`MctsCandidate::new`] default of `1.0`) normalize to
/// exactly `1/K` through this path: the sum of `K` ones is exactly `K`, so each prior
/// is `fl(1/K)`. That exactness is what
/// [`PriorUct`](MctsSelectionPolicy::PriorUct)'s reduction to
/// [`Uct`](MctsSelectionPolicy::Uct) rests on.
fn normalize_priors(mut candidates: Vec<MctsCandidate>) -> Vec<MctsCandidate> {
    if candidates.is_empty() {
        return candidates;
    }

    let usable = |w: f64| if w.is_finite() && w > 0.0 { w } else { 0.0 };
    let total: f64 = candidates.iter().map(|c| usable(c.prior)).sum();

    if total.is_finite() && total > 0.0 {
        for candidate in &mut candidates {
            candidate.prior = usable(candidate.prior) / total;
        }
    } else {
        // The candidate count is tiny; the cast is exact.
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / candidates.len() as f64;
        for candidate in &mut candidates {
            candidate.prior = uniform;
        }
    }
    candidates
}

/// Force an evaluator's score into `[0, 1]`, mapping a non-finite score to `0.0`.
///
/// `NaN` is the dangerous one. It would flow from the evaluator into `total_value`,
/// out of `mean_value`, into every selection score on the path, and there it would
/// silently lose every `>` comparison — the search would keep running, keep counting
/// visits, and keep returning a confident answer produced by a tree policy that had
/// stopped functioning. Mapping it to the *worst* possible return means a broken
/// evaluator degrades the path it broke on, and nothing else.
fn clamp_return(raw: f64) -> f64 {
    if raw.is_finite() {
        raw.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compose the answer from the endorsed reasoning path and the retrieved context.
fn synthesize(query: &str, path: &[String], context: &[SearchResult]) -> String {
    let conclusion = path.last().cloned().unwrap_or_else(|| query.to_string());
    if context.is_empty() {
        return conclusion;
    }
    let evidence = context
        .iter()
        .take(2)
        .map(|r| r.document.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{conclusion} {evidence}")
}
