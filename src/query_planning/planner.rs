//! Heuristic query planner that classifies a natural-language query and
//! produces a [`QueryPlan`] DAG ready for execution.

use super::types::{PlanStep, PlanStepKind, QueryPlan, QueryPlanningError, SynthesisStrategy};

// ── QueryPlanner ──────────────────────────────────────────────────────────────

/// Heuristic planner that inspects keyword signals to classify a query and
/// decompose it into an executable [`QueryPlan`].
#[derive(Debug, Clone, Default)]
pub struct QueryPlanner {}

impl QueryPlanner {
    /// Create a new [`QueryPlanner`].
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Plan `query` into a [`QueryPlan`].
    ///
    /// # Errors
    ///
    /// Returns [`QueryPlanningError::EmptyQuery`] when `query` is blank.
    pub fn plan(&self, query: &str) -> Result<QueryPlan, QueryPlanningError> {
        if query.trim().is_empty() {
            return Err(QueryPlanningError::EmptyQuery);
        }

        let lower = query.to_lowercase();

        // ── Classify ──────────────────────────────────────────────────────────
        let is_comparative = lower.contains("compare")
            || lower.contains(" versus ")
            || lower.contains(" vs ")
            || lower.contains("difference between");

        let is_aggregative = lower.contains("how many")
            || lower.contains(" count ")
            || lower.contains(" total ")
            || lower.contains(" sum ")
            || lower.contains(" average ");

        let has_and = lower.contains(" and ");

        let needs_verify = lower.contains("if ")
            || lower.contains("whether ")
            || lower.contains("does ")
            || lower.contains("is it");

        // ── Build plan ────────────────────────────────────────────────────────
        if is_comparative {
            return Ok(self.build_comparative_plan(query));
        }
        if is_aggregative {
            return Ok(self.build_aggregative_plan(query));
        }
        if has_and {
            return Ok(self.build_multi_step_plan(query));
        }
        if needs_verify {
            return Ok(self.build_verify_plan(query));
        }

        // Default: simple single Retrieve
        Ok(QueryPlan::new(
            vec![PlanStep::new(0, PlanStepKind::Retrieve, query)],
            SynthesisStrategy::Simple,
        ))
    }

    // ── Plan builders ─────────────────────────────────────────────────────────

    #[allow(clippy::unused_self)]
    fn build_comparative_plan(&self, query: &str) -> QueryPlan {
        QueryPlan::new(
            vec![
                PlanStep::new(0, PlanStepKind::Retrieve, format!("aspect A of {query}")),
                PlanStep::new(1, PlanStepKind::Retrieve, format!("aspect B of {query}")),
                PlanStep::new(2, PlanStepKind::Compare, query).with_depends_on(vec![0, 1]),
            ],
            SynthesisStrategy::Comparative,
        )
    }

    #[allow(clippy::unused_self)]
    fn build_aggregative_plan(&self, query: &str) -> QueryPlan {
        QueryPlan::new(
            vec![
                PlanStep::new(0, PlanStepKind::Retrieve, query),
                PlanStep::new(1, PlanStepKind::Aggregate, query).with_depends_on(vec![0]),
            ],
            SynthesisStrategy::Aggregative,
        )
    }

    #[allow(clippy::unused_self)]
    fn build_multi_step_plan(&self, query: &str) -> QueryPlan {
        // Split on " and " to produce one Retrieve step per part
        let parts: Vec<&str> = query.splitn(10, " and ").collect();
        let retrieve_count = parts.len();

        let mut steps: Vec<PlanStep> = parts
            .into_iter()
            .enumerate()
            .map(|(i, part)| PlanStep::new(i, PlanStepKind::Retrieve, part.trim()))
            .collect();

        let dep_ids: Vec<usize> = (0..retrieve_count).collect();
        steps.push(
            PlanStep::new(retrieve_count, PlanStepKind::Aggregate, query).with_depends_on(dep_ids),
        );

        QueryPlan::new(steps, SynthesisStrategy::Aggregative)
    }

    #[allow(clippy::unused_self)]
    fn build_verify_plan(&self, query: &str) -> QueryPlan {
        QueryPlan::new(
            vec![
                PlanStep::new(0, PlanStepKind::Retrieve, query),
                PlanStep::new(1, PlanStepKind::Verify, query).with_depends_on(vec![0]),
            ],
            SynthesisStrategy::Simple,
        )
    }
}
