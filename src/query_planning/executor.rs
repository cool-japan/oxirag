//! Plan executor: topologically sorts a [`QueryPlan`] and runs each step
//! against an [`Echo`] backend, collecting [`PlanResult`]s.

use std::collections::HashMap;

use async_trait::async_trait;

use super::types::{PlanResult, PlanStep, PlanStepKind, QueryPlan, QueryPlanningError};
use crate::layer1_echo::traits::Echo;

// ── Topological sort (Kahn's algorithm) ──────────────────────────────────────

/// Return a valid topological ordering of step IDs, or
/// [`QueryPlanningError::CyclicDependency`] when a cycle is detected.
pub(crate) fn topo_sort(steps: &[PlanStep]) -> Result<Vec<usize>, QueryPlanningError> {
    let n = steps.len();
    // Map step.id → index in the slice for fast look-up
    let id_to_idx: HashMap<usize, usize> =
        steps.iter().enumerate().map(|(i, s)| (s.id, i)).collect();

    // Build adjacency and in-degree
    let mut in_degree = vec![0usize; n];
    let mut children: Vec<Vec<usize>> = vec![vec![]; n]; // children[idx] = list of dependent idxs

    for (idx, step) in steps.iter().enumerate() {
        for &dep_id in &step.depends_on {
            if let Some(&dep_idx) = id_to_idx.get(&dep_id) {
                children[dep_idx].push(idx);
                in_degree[idx] += 1;
            }
        }
    }

    // Kahn's BFS
    let mut queue: std::collections::VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| if d == 0 { Some(i) } else { None })
        .collect();

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        order.push(steps[idx].id);
        for &child_idx in &children[idx] {
            in_degree[child_idx] -= 1;
            if in_degree[child_idx] == 0 {
                queue.push_back(child_idx);
            }
        }
    }

    if order.len() != n {
        return Err(QueryPlanningError::CyclicDependency);
    }
    Ok(order)
}

// ── PlanExecutor ──────────────────────────────────────────────────────────────

/// Executes a [`QueryPlan`] against an [`Echo`] backend.
#[derive(Debug, Clone, Default)]
pub struct PlanExecutor {}

impl PlanExecutor {
    /// Create a new [`PlanExecutor`].
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl PlanExecutor {
    /// Execute all steps in `plan` against `echo`, returning a [`PlanResult`]
    /// for each step in topological order.
    ///
    /// # Errors
    ///
    /// Returns [`QueryPlanningError::CyclicDependency`] if the plan contains a
    /// cycle, or [`QueryPlanningError::StepFailed`] if an individual step fails.
    #[allow(clippy::unused_self)]
    pub async fn run<E>(
        &self,
        plan: &QueryPlan,
        echo: &E,
    ) -> Result<Vec<PlanResult>, QueryPlanningError>
    where
        E: Echo + ?Sized,
    {
        if plan.is_empty() {
            return Ok(vec![]);
        }

        let order = topo_sort(&plan.steps)?;

        // Build a map from step_id → PlanResult for dependency look-up
        let id_to_step: HashMap<usize, &PlanStep> = plan.steps.iter().map(|s| (s.id, s)).collect();

        let mut results: HashMap<usize, PlanResult> = HashMap::new();

        for step_id in order {
            let step = id_to_step[&step_id];

            let result = match step.kind {
                PlanStepKind::Retrieve => execute_retrieve(step, echo).await?,
                PlanStepKind::Filter => execute_filter(step, &results),
                PlanStepKind::Aggregate => execute_aggregate(step, &results),
                PlanStepKind::Compare => execute_compare(step, &results),
                PlanStepKind::Verify => execute_verify(step, &results),
            };

            results.insert(step_id, result);
        }

        // Return results in plan order
        Ok(plan
            .steps
            .iter()
            .filter_map(|s| results.remove(&s.id))
            .collect())
    }
}

// ── Step execution helpers ────────────────────────────────────────────────────

async fn execute_retrieve<E>(step: &PlanStep, echo: &E) -> Result<PlanResult, QueryPlanningError>
where
    E: Echo + ?Sized,
{
    let search_results = echo
        .search(&step.query, 5, None)
        .await
        .map_err(|e| QueryPlanningError::StepFailed(e.to_string()))?;

    let answer = search_results
        .iter()
        .take(3)
        .map(|r| r.document.content.clone())
        .collect::<Vec<_>>()
        .join(" ");

    let sources: Vec<String> = search_results
        .iter()
        .take(3)
        .map(|r| r.document.id.to_string())
        .collect();

    Ok(PlanResult::new(step.id, answer).with_sources(sources))
}

fn execute_filter(step: &PlanStep, results: &HashMap<usize, PlanResult>) -> PlanResult {
    // Collect text from all dependency results
    let dep_text: String = step
        .depends_on
        .iter()
        .filter_map(|id| results.get(id))
        .map(|r| r.answer.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Keep only sentences that contain at least one query keyword
    let keywords: Vec<&str> = step.query.split_whitespace().collect();
    let lower_dep = dep_text.to_lowercase();

    let filtered: String = dep_text
        .split(". ")
        .filter(|sentence| {
            let lower_s = sentence.to_lowercase();
            keywords.iter().any(|kw| lower_s.contains(*kw))
        })
        .collect::<Vec<_>>()
        .join(". ");

    let answer = if filtered.is_empty() {
        lower_dep
    } else {
        filtered
    };
    PlanResult::new(step.id, answer)
}

fn execute_aggregate(step: &PlanStep, results: &HashMap<usize, PlanResult>) -> PlanResult {
    let combined: String = step
        .depends_on
        .iter()
        .filter_map(|id| results.get(id))
        .map(|r| r.answer.as_str())
        .collect::<Vec<_>>()
        .join("; ");

    PlanResult::new(step.id, combined)
}

fn execute_compare(step: &PlanStep, results: &HashMap<usize, PlanResult>) -> PlanResult {
    let dep_answers: Vec<&str> = step
        .depends_on
        .iter()
        .filter_map(|id| results.get(id))
        .map(|r| r.answer.as_str())
        .collect();

    let answer = match dep_answers.as_slice() {
        [a, b, ..] => format!("A: {a} vs B: {b}"),
        [a] => a.to_string(),
        [] => String::new(),
    };

    PlanResult::new(step.id, answer)
}

fn execute_verify(step: &PlanStep, results: &HashMap<usize, PlanResult>) -> PlanResult {
    let dep_text: String = step
        .depends_on
        .iter()
        .filter_map(|id| results.get(id))
        .map(|r| r.answer.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    PlanResult::new(step.id, format!("Verified: {dep_text}"))
}

// ── Async-trait workaround: expose run() behind a dyn-compatible wrapper ──────
//
// PlanExecutor::run is a generic method (E: Echo + ?Sized) and cannot be made
// object-safe. The PlanExecutor struct is kept as a plain `new()/default()`
// value; callers invoke `run` directly.

/// Sealed async trait so that `PlanExecutor::run` can be called via a shared
/// reference without boxing.  Only implemented for `PlanExecutor`.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait RunPlan {
    /// Execute `plan` against `echo`.
    async fn run_plan<E: Echo + ?Sized + Send + Sync>(
        &self,
        plan: &QueryPlan,
        echo: &E,
    ) -> Result<Vec<PlanResult>, QueryPlanningError>;
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl RunPlan for PlanExecutor {
    async fn run_plan<E: Echo + ?Sized + Send + Sync>(
        &self,
        plan: &QueryPlan,
        echo: &E,
    ) -> Result<Vec<PlanResult>, QueryPlanningError> {
        self.run(plan, echo).await
    }
}
