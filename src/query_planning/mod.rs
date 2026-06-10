//! DAG-structured query execution plans.
//!
//! Classifies natural-language queries into execution DAGs and runs them
//! against an [`Echo`] backend, collecting a [`PlanResult`] per step.
//!
//! [`Echo`]: crate::layer1_echo::traits::Echo

pub mod executor;
pub mod planner;
#[cfg(test)]
mod tests;
pub mod types;

pub use executor::PlanExecutor;
pub use planner::QueryPlanner;
pub use types::{
    PlanResult, PlanStep, PlanStepKind, QueryPlan, QueryPlanningError, SynthesisStrategy,
};
