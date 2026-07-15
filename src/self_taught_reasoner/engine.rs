//! [`SelfTaughtReasoner`] — the `STaR` bootstrapping loop.

use std::collections::HashSet;

use super::model::ReasoningModel;
use super::rng::{StarRng, mix_seed};
use super::text::{answers_equivalent, is_cheating_rationale};
use super::types::{
    BootstrapRound, RationaleSet, RationaleSource, StarConfig, StarError, StarGeneration,
    StarOutcome, StarProblem, StarRationale,
};

/// The self-taught reasoner: runs the forward → filter → rationalize → accumulate
/// → fine-tune loop to a fixed point.
///
/// Each round it (1) has the generator solve every problem forward and keeps the
/// correct ones, (2) for every problem the forward pass missed, has the generator
/// rationalize backward from the gold answer and keeps the rationales that
/// **reach the gold answer without cheating**, (3) accumulates the kept rationales
/// into the bootstrapped set (deduplicated by problem id), and (4) fine-tunes the
/// generator on the full accumulated set. The loop stops when a round adds nothing
/// new — the accumulated set has reached a fixed point.
#[derive(Debug, Clone)]
pub struct SelfTaughtReasoner {
    /// Configuration for this run.
    pub config: StarConfig,
}

impl SelfTaughtReasoner {
    /// Create a reasoner with the given configuration.
    #[must_use]
    pub fn new(config: StarConfig) -> Self {
        Self { config }
    }

    /// Run the bootstrapping loop over `problems`, fine-tuning `model` in place.
    ///
    /// Returns a [`StarOutcome`] with the per-round record, the final
    /// bootstrapped set, and whether the loop converged. `model` is left in its
    /// fully fine-tuned state.
    ///
    /// # Errors
    ///
    /// - [`StarError::NoProblems`] when `problems` is empty.
    /// - [`StarError::DuplicateProblemId`] when two problems share an id (the
    ///   accumulated set deduplicates by id, so duplicates would silently drop
    ///   one problem's rationale).
    /// - [`StarError::InvalidConfig`] when the configuration fails
    ///   [`StarConfig::validate`].
    pub fn run<M: ReasoningModel>(
        &self,
        problems: &[StarProblem],
        model: &mut M,
    ) -> Result<StarOutcome, StarError> {
        self.config.validate()?;
        if problems.is_empty() {
            return Err(StarError::NoProblems);
        }
        let mut seen_ids: HashSet<&str> = HashSet::with_capacity(problems.len());
        for problem in problems {
            if !seen_ids.insert(problem.id.as_str()) {
                return Err(StarError::DuplicateProblemId {
                    id: problem.id.clone(),
                });
            }
        }

        let n = problems.len();
        let mut accumulated = RationaleSet::new();
        let mut rounds: Vec<BootstrapRound> = Vec::new();
        let mut converged = false;

        for round_index in 0..self.config.max_rounds {
            let round = self.execute_round(round_index, problems, model, &mut accumulated);
            // Fixed point: the accumulated set stopped growing, so the next round
            // would fine-tune on the same set and reproduce this one exactly. The
            // set only ever grows and is bounded by the problem count, so this is
            // reached in at most `n + 1` rounds and never oscillates.
            let reached_fixed_point = round.newly_accumulated == 0;
            rounds.push(round);
            if reached_fixed_point {
                converged = true;
                break;
            }
        }

        Ok(StarOutcome {
            rounds,
            final_set: accumulated,
            converged,
            total_problems: n,
        })
    }

    /// Run one bootstrapping round in place: forward-generate and filter,
    /// rationalize the failures, accumulate the survivors, fine-tune on the full
    /// accumulated set, and return the round's record.
    fn execute_round<M: ReasoningModel>(
        &self,
        round_index: usize,
        problems: &[StarProblem],
        model: &mut M,
        accumulated: &mut RationaleSet,
    ) -> BootstrapRound {
        let threshold = self.config.equivalence_threshold;
        let n = problems.len();
        let order = self.processing_order(round_index, n);

        // 1. Forward generation + correctness filter.
        let mut forward_solved_ids: Vec<String> = Vec::new();
        let mut failed: Vec<usize> = Vec::new();
        let mut round_kept: Vec<StarRationale> = Vec::new();
        let mut incorrect_rejected = 0;
        for &i in &order {
            let problem = &problems[i];
            let generation = model.generate(problem);
            if answers_equivalent(&generation.answer, &problem.gold_answer, threshold) {
                forward_solved_ids.push(problem.id.clone());
                round_kept.push(keep(problem, generation, RationaleSource::Forward));
            } else {
                incorrect_rejected += 1;
                failed.push(i);
            }
        }

        // 2. Backward rationalization over the problems the forward pass missed,
        //    guarded by both the correctness oracle and the cheat detector.
        let mut rationalized_solved_ids: Vec<String> = Vec::new();
        let mut cheats_rejected = 0;
        if self.config.use_rationalization {
            for &i in &failed {
                let problem = &problems[i];
                let generation = model.rationalize(problem, &problem.gold_answer);
                if !answers_equivalent(&generation.answer, &problem.gold_answer, threshold) {
                    // Rationalization did not even reach the gold answer.
                    incorrect_rejected += 1;
                    continue;
                }
                if is_cheating_rationale(
                    &generation.rationale,
                    &problem.statement,
                    &problem.gold_answer,
                ) {
                    // Reached the answer only by echoing the hint — the cheat
                    // STaR's validity depends on excluding.
                    cheats_rejected += 1;
                    continue;
                }
                rationalized_solved_ids.push(problem.id.clone());
                round_kept.push(keep(problem, generation, RationaleSource::Rationalized));
            }
        }

        // 3. Accumulate into the bootstrapped set (dedup by problem id).
        let kept_this_round = round_kept.len();
        let mut newly_accumulated = 0;
        for rationale in round_kept {
            if accumulated.insert(rationale) {
                newly_accumulated += 1;
            }
        }

        // 4. Fine-tune from the full accumulated set.
        model.update_from(accumulated.rationales());

        // 5. Assemble the round record with sorted id lists, for a deterministic
        //    record independent of the processing order.
        forward_solved_ids.sort();
        rationalized_solved_ids.sort();
        #[allow(clippy::cast_precision_loss)]
        let forward_accuracy = forward_solved_ids.len() as f64 / n as f64;
        BootstrapRound {
            round_index,
            forward_accuracy,
            forward_solved: forward_solved_ids.len(),
            rationalized_solved: rationalized_solved_ids.len(),
            kept_this_round,
            newly_accumulated,
            accumulated_size: accumulated.len(),
            forward_solved_ids,
            rationalized_solved_ids,
            cheats_rejected,
            incorrect_rejected,
        }
    }

    /// The order in which to process problems this round.
    ///
    /// A fixed `0..n` order unless [`StarConfig::shuffle_each_round`] is set, in
    /// which case a deterministic per-round shuffle derived from the seed and the
    /// round index is applied. The order never changes *which* problems get
    /// solved (the correctness filter is order-independent), only the recorded
    /// sequence — which is why the fixed point is reproducible across seeds.
    fn processing_order(&self, round_index: usize, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        if self.config.shuffle_each_round {
            let round_seed = mix_seed(self.config.seed, round_index as u64);
            let mut rng = StarRng::new(round_seed);
            rng.shuffle(&mut order);
        }
        order
    }
}

/// Wrap a kept generation into a [`StarRationale`], tagging its provenance.
fn keep(
    problem: &StarProblem,
    generation: StarGeneration,
    source: RationaleSource,
) -> StarRationale {
    StarRationale {
        problem_id: problem.id.clone(),
        statement: problem.statement.clone(),
        rationale: generation.rationale,
        answer: generation.answer,
        source,
    }
}
