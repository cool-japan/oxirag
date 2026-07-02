//! [`ThoughtBuffer`] (persistent template storage, retrieval, eviction) and
//! [`BotEngine`] (the retrieve → instantiate → solve → distill pipeline).

use crate::buffer_of_thoughts::types::{
    BotConfig, BotError, BotGenerator, BotSolveResult, DEFAULT_TEMPLATE_TEXT, DistillAction,
    EvictionPolicy, ProblemSignature, SolveOutcome, ThoughtTemplate, instantiate_template,
};

// ── ThoughtBuffer ──────────────────────────────────────────────────────────────

/// A persistent, growing store of [`ThoughtTemplate`]s.
///
/// Unlike a fixed, built-in module bank, a [`ThoughtBuffer`] starts empty and
/// accumulates templates over time as problems are solved and distilled via
/// [`BotEngine`]. Retrieval is a linear scan over buffered templates scored by
/// [`ProblemSignature::similarity`] — buffers are bounded by
/// [`BotConfig::max_buffer_size`] (typically hundreds of entries at most), so
/// this is fast in practice and keeps the retrieval logic fully transparent.
///
/// A monotonically-increasing logical clock (`tick`, not wall-clock time)
/// drives LRU eviction: the clock advances on every buffer *mutation*
/// (`insert` or `reinforce`), not on read-only similarity queries, so
/// "recently used" means "recently relied upon in a completed solve/outcome
/// cycle" rather than merely considered as a retrieval candidate.
#[derive(Debug, Clone, Default)]
pub struct ThoughtBuffer {
    templates: Vec<ThoughtTemplate>,
    next_id: u64,
    tick: u64,
}

impl ThoughtBuffer {
    /// Create a new, empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of templates currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Returns `true` when the buffer holds no templates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// All templates currently stored, in insertion order.
    #[must_use]
    pub fn templates(&self) -> &[ThoughtTemplate] {
        &self.templates
    }

    /// Look up a template by `id`.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&ThoughtTemplate> {
        self.templates.iter().find(|t| t.id == id)
    }

    /// Retrieve the best-matching buffered template for `signature`, if its
    /// similarity meets or exceeds `threshold`.
    ///
    /// Returns `Some((id, similarity))` for the closest match, or `None`
    /// either when the buffer is empty or when the best similarity found is
    /// below `threshold` — retrieval never forces a bad match. Ties in
    /// similarity are broken by higher `success_rate`, then by the lowest
    /// `id` (the more established template), for determinism.
    #[must_use]
    pub fn retrieve_best(
        &self,
        signature: &ProblemSignature,
        threshold: f32,
    ) -> Option<(u64, f32)> {
        let mut best: Option<(u64, f32, f32)> = None; // (id, similarity, success_rate)

        for template in &self.templates {
            let sim = template.signature.similarity(signature);
            let is_better = match best {
                None => true,
                Some((best_id, best_sim, best_rate)) => {
                    match sim
                        .partial_cmp(&best_sim)
                        .unwrap_or(std::cmp::Ordering::Equal)
                    {
                        std::cmp::Ordering::Greater => true,
                        std::cmp::Ordering::Less => false,
                        std::cmp::Ordering::Equal => {
                            match template
                                .success_rate
                                .partial_cmp(&best_rate)
                                .unwrap_or(std::cmp::Ordering::Equal)
                            {
                                std::cmp::Ordering::Greater => true,
                                std::cmp::Ordering::Less => false,
                                std::cmp::Ordering::Equal => template.id < best_id,
                            }
                        }
                    }
                }
            };
            if is_better {
                best = Some((template.id, sim, template.success_rate));
            }
        }

        best.filter(|&(_, sim, _)| sim >= threshold)
            .map(|(id, sim, _)| (id, sim))
    }

    /// Choose the index of the "worst" template to evict under `policy`.
    ///
    /// Returns `None` when the buffer is empty.
    fn evict_index(&self, policy: EvictionPolicy) -> Option<usize> {
        if self.templates.is_empty() {
            return None;
        }
        let mut worst_idx = 0usize;
        for (idx, candidate) in self.templates.iter().enumerate().skip(1) {
            let current = &self.templates[worst_idx];
            let should_replace = match policy {
                EvictionPolicy::Lru => {
                    candidate.last_used_tick < current.last_used_tick
                        || (candidate.last_used_tick == current.last_used_tick
                            && candidate.id < current.id)
                }
                EvictionPolicy::LowestSuccessRate => {
                    match candidate
                        .success_rate
                        .partial_cmp(&current.success_rate)
                        .unwrap_or(std::cmp::Ordering::Equal)
                    {
                        std::cmp::Ordering::Less => true,
                        std::cmp::Ordering::Equal => {
                            candidate.usage_count < current.usage_count
                                || (candidate.usage_count == current.usage_count
                                    && candidate.id < current.id)
                        }
                        std::cmp::Ordering::Greater => false,
                    }
                }
            };
            if should_replace {
                worst_idx = idx;
            }
        }
        Some(worst_idx)
    }

    /// Insert a freshly-distilled template, evicting under `config`'s
    /// [`EvictionPolicy`] first if the buffer is already at
    /// [`BotConfig::max_buffer_size`]. Returns the new template's `id`.
    pub(crate) fn insert(
        &mut self,
        signature: ProblemSignature,
        template_text: String,
        config: &BotConfig,
    ) -> u64 {
        self.tick = self.tick.saturating_add(1);

        while self.templates.len() >= config.max_buffer_size {
            match self.evict_index(config.eviction_policy) {
                Some(idx) => {
                    self.templates.remove(idx);
                }
                None => break,
            }
        }

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.templates.push(ThoughtTemplate::new(
            id,
            signature,
            template_text,
            self.tick,
        ));
        id
    }

    /// Record one more (re)use of the template with `id`.
    ///
    /// # Errors
    ///
    /// Returns [`BotError::TemplateNotFound`] if no template with `id` is
    /// currently buffered (e.g. it was evicted since it was retrieved).
    pub(crate) fn reinforce(&mut self, id: u64, success: bool) -> Result<(), BotError> {
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        match self.templates.iter_mut().find(|t| t.id == id) {
            Some(template) => {
                template.reinforce(success, tick);
                Ok(())
            }
            None => Err(BotError::TemplateNotFound(id)),
        }
    }
}

// ── BotEngine ──────────────────────────────────────────────────────────────────

/// Drives the Buffer-of-Thoughts pipeline: retrieve → instantiate → solve →
/// distill.
///
/// The engine owns both the persistent [`ThoughtBuffer`] and the generator
/// `G` (mirroring `self_discover::SelfDiscoverEngine`'s owned-model
/// pattern), so a single long-lived `BotEngine` accumulates templates across
/// many calls to [`BotEngine::solve`] / [`BotEngine::record_outcome`] (or the
/// combined [`BotEngine::solve_and_record`]).
///
/// # Example
///
/// ```
/// use oxirag::buffer_of_thoughts::{BotConfig, BotEngine, DistillAction, MockBotGenerator, SolveOutcome};
///
/// let mut engine = BotEngine::new(BotConfig::default(), MockBotGenerator);
/// assert!(engine.buffer.is_empty());
///
/// // First problem: the buffer is empty, so BoT falls back to the generic
/// // scaffold. Since it succeeds, a brand-new template is distilled.
/// let (first, action) = engine
///     .solve_and_record("What is the sum of 12 and 7?", |_| SolveOutcome::Success)
///     .unwrap();
/// assert!(first.matched_template_id.is_none());
/// assert!(matches!(action, DistillAction::Distilled(_)));
/// assert_eq!(engine.buffer.len(), 1);
///
/// // A structurally similar problem now retrieves and reuses the distilled
/// // template instead of starting from scratch — the buffer does not grow.
/// let (second, action) = engine
///     .solve_and_record("What is the sum of 30 and 5?", |_| SolveOutcome::Success)
///     .unwrap();
/// assert!(second.matched_template_id.is_some());
/// assert!(matches!(action, DistillAction::Reinforced(_)));
/// assert_eq!(engine.buffer.len(), 1);
/// ```
pub struct BotEngine<G: BotGenerator> {
    /// Configuration for this engine.
    pub config: BotConfig,
    /// The persistent, growing buffer of thought templates.
    pub buffer: ThoughtBuffer,
    /// The generator used for `solve` and `distill`.
    pub generator: G,
}

impl<G: BotGenerator> BotEngine<G> {
    /// Create a new engine with an empty buffer.
    #[must_use]
    pub fn new(config: BotConfig, generator: G) -> Self {
        Self {
            config,
            buffer: ThoughtBuffer::new(),
            generator,
        }
    }

    /// Run retrieve → instantiate → solve for `problem`, without recording an
    /// outcome or mutating the buffer.
    ///
    /// 1. **Retrieve**: compute `problem`'s [`ProblemSignature`] and look up
    ///    the closest-matching buffered template via
    ///    [`ThoughtBuffer::retrieve_best`].
    /// 2. **Instantiate**: substitute the matched template's placeholders (or
    ///    the generic [`DEFAULT_TEMPLATE_TEXT`] fallback, when nothing
    ///    matched) with values from `problem`.
    /// 3. **Solve**: ask [`BotGenerator::generate`] for an answer conditioned
    ///    on the instantiated scaffold.
    ///
    /// Call [`BotEngine::record_outcome`] with the returned
    /// [`BotSolveResult`] once the caller has judged the answer, to trigger
    /// distillation or reinforcement.
    ///
    /// # Errors
    ///
    /// Returns [`BotError::EmptyProblem`] if `problem` is empty after
    /// trimming.
    pub fn solve(&self, problem: &str) -> Result<BotSolveResult, BotError> {
        if problem.trim().is_empty() {
            return Err(BotError::EmptyProblem);
        }

        let signature = ProblemSignature::compute(problem);
        let retrieved = self
            .buffer
            .retrieve_best(&signature, self.config.similarity_threshold);

        let (instantiated_scaffold, matched_template_id, similarity) = match retrieved {
            Some((id, sim)) => {
                let template = self.buffer.get(id).ok_or(BotError::TemplateNotFound(id))?;
                (
                    instantiate_template(&template.template_text, problem),
                    Some(id),
                    Some(sim),
                )
            }
            None => (
                instantiate_template(DEFAULT_TEMPLATE_TEXT, problem),
                None,
                None,
            ),
        };

        let answer = self.generator.generate(problem, &instantiated_scaffold);

        Ok(BotSolveResult {
            problem: problem.to_string(),
            signature,
            matched_template_id,
            similarity,
            instantiated_scaffold,
            answer,
        })
    }

    /// Distill (write-back) step: given a judged `outcome` for `solve_result`,
    /// grow or reinforce the buffer.
    ///
    /// - If `solve_result` reused a buffered template
    ///   (`matched_template_id = Some(id)`), that template is reinforced
    ///   regardless of `outcome` (both successes and failures update
    ///   `usage_count`; only successes increment `success_count`), returning
    ///   [`DistillAction::Reinforced`].
    /// - If `solve_result` matched nothing and `outcome` is
    ///   [`SolveOutcome::Success`], [`BotGenerator::distill`] is asked to
    ///   generalize a brand-new template, which is inserted into the buffer
    ///   (evicting under [`BotConfig::eviction_policy`] if at capacity),
    ///   returning [`DistillAction::Distilled`].
    /// - If `solve_result` matched nothing and `outcome` is
    ///   [`SolveOutcome::Failure`], nothing is written back
    ///   ([`DistillAction::Skipped`]) — distilling a template from an
    ///   unsuccessful, unmatched solve would pollute the buffer with an
    ///   ungeneralizable strategy.
    ///
    /// # Errors
    ///
    /// Returns [`BotError::TemplateNotFound`] if the matched template was
    /// evicted between `solve` and `record_outcome`, or
    /// [`BotError::EmptyTemplateText`] if [`BotGenerator::distill`] returns
    /// an empty (or all-whitespace) string.
    pub fn record_outcome(
        &mut self,
        solve_result: &BotSolveResult,
        outcome: SolveOutcome,
    ) -> Result<DistillAction, BotError> {
        match solve_result.matched_template_id {
            Some(id) => {
                self.buffer
                    .reinforce(id, outcome == SolveOutcome::Success)?;
                Ok(DistillAction::Reinforced(id))
            }
            None => {
                if outcome == SolveOutcome::Success {
                    let template_text = self.generator.distill(
                        &solve_result.problem,
                        &solve_result.answer,
                        &solve_result.signature,
                    );
                    if template_text.trim().is_empty() {
                        return Err(BotError::EmptyTemplateText);
                    }
                    let id = self.buffer.insert(
                        solve_result.signature.clone(),
                        template_text,
                        &self.config,
                    );
                    Ok(DistillAction::Distilled(id))
                } else {
                    Ok(DistillAction::Skipped)
                }
            }
        }
    }

    /// Convenience wrapper combining [`BotEngine::solve`] and
    /// [`BotEngine::record_outcome`] in a single call.
    ///
    /// `judge` inspects the [`BotSolveResult`] (e.g. checking the answer
    /// against a known ground truth) and returns the [`SolveOutcome`] to
    /// record.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`BotEngine::solve`] or
    /// [`BotEngine::record_outcome`].
    pub fn solve_and_record<F>(
        &mut self,
        problem: &str,
        judge: F,
    ) -> Result<(BotSolveResult, DistillAction), BotError>
    where
        F: FnOnce(&BotSolveResult) -> SolveOutcome,
    {
        let result = self.solve(problem)?;
        let outcome = judge(&result);
        let action = self.record_outcome(&result, outcome)?;
        Ok((result, action))
    }
}

impl<G: BotGenerator + Default> Default for BotEngine<G> {
    fn default() -> Self {
        Self::new(BotConfig::default(), G::default())
    }
}
