//! `ReWOO`: Decoupling Reasoning from Observations for Efficient Augmented
//! Language Models (Xu et al. 2023).
//!
//! `ReWOO`'s central idea is that planning and evidence-gathering do **not**
//! need to be interleaved. A single upfront [`RewooPlanner`] call produces a
//! complete, ordered [`RewooPlan`] in which every future retrieval result is
//! represented by a placeholder token (`#E1`, `#E2`, ...) — the planner never
//! sees any actual evidence, only the shape of the reasoning. Only *after*
//! the whole plan exists does a [`RewooWorker`] walk it in order, resolving
//! one placeholder at a time by executing that step's action. Finally a
//! [`RewooSolver`] substitutes every resolved placeholder back into the
//! plan's reasoning text and synthesizes the answer.
//!
//! # Distinct from `agentic` and `query_planning`
//!
//! | Module | Planning happens | Model/planner calls | Evidence known while planning? |
//! |--------|-------------------|----------------------|----------------------------------|
//! | [`crate::agentic`] (`ReAct`) | Interleaved: one Thought→Action→Observation turn at a time | One per step | Yes — each turn sees the previous step's real observation |
//! | [`crate::query_planning`] | A runtime DAG is scheduled and executed by an [`crate::query_planning::PlanExecutor`] | Varies with DAG shape | Partially — sibling/downstream nodes may see upstream results as the DAG runs |
//! | **`rewoo`** (this module) | A single upfront pass, entirely before any evidence-gathering | **Exactly one**, regardless of plan length | No — placeholders stand in for evidence that does not exist yet |
//!
//! This decoupling is `ReWOO`'s efficiency property: a [`RewooWorker`] makes
//! exactly one [`RewooRetriever::retrieve`] call per
//! [`RewooAction::Search`] step and
//! **zero** additional planning calls, no matter how long the plan is —
//! whereas a ReAct-style loop needs one model call per step because planning
//! and acting alternate.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|-----------------|
//! | [`PlaceholderVar`] | A named evidence slot (`#E1`, `#E2`, ...) produced by one step and consumed by later ones |
//! | [`RewooAction`] | What a step does to resolve its placeholder: [`Search`](RewooAction::Search) or [`Compute`](RewooAction::Compute) |
//! | [`RewooStep`] / [`RewooPlan`] | One planned step / the complete ordered plan |
//! | [`RewooPlanSource`] + [`MockRewooPlanSource`] | Pluggable single-call plan generation |
//! | [`RewooRetriever`] + [`MockRewooRetriever`] | Pluggable evidence retrieval for `Search` actions |
//! | [`RewooGenerator`] + [`MockRewooGenerator`] | Pluggable final-answer synthesis |
//! | [`RewooPlanner`] | Generates + structurally validates a plan (one plan-source call) |
//! | [`RewooWorker`] | Resolves every placeholder in plan order (one retriever call per `Search` step) |
//! | [`RewooSolver`] | Substitutes resolved evidence into the plan and synthesizes the answer |
//! | [`RewooPipeline`] | Convenience wrapper chaining planner → worker → solver |
//! | [`substitute_placeholders`] | Whole-token (`#E1` vs `#E10`-safe) placeholder substitution |
//! | [`evaluate_expression`] | Recursive-descent arithmetic evaluator for `Compute` actions |
//!
//! # Quick start
//!
//! ```
//! use oxirag::rewoo::{
//!     MockRewooGenerator, MockRewooPlanSource, MockRewooRetriever, PlaceholderVar, RewooAction,
//!     RewooConfig, RewooPipeline, RewooPlan, RewooStep,
//! };
//!
//! // The planner is invoked exactly once, entirely up front: placeholders
//! // stand in for evidence that does not exist yet, and step 2's action
//! // references step 1's not-yet-resolved placeholder `#E1`.
//! let plan = RewooPlan::new(
//!     "Who invented the telephone, and what year was he born?",
//!     vec![
//!         RewooStep::new(
//!             "Find who invented the telephone.",
//!             RewooAction::Search("who invented the telephone".to_string()),
//!             PlaceholderVar::new(1),
//!         ),
//!         RewooStep::new(
//!             "Find the birth year of #E1.",
//!             RewooAction::Search("birth year of #E1".to_string()),
//!             PlaceholderVar::new(2),
//!         ),
//!     ],
//! );
//! let plan_source = MockRewooPlanSource::single(plan);
//!
//! // The worker resolves step 2's query only after substituting step 1's
//! // resolved evidence into it.
//! let retriever = MockRewooRetriever::new(vec![
//!     (
//!         "who invented the telephone".to_string(),
//!         "Alexander Graham Bell".to_string(),
//!     ),
//!     (
//!         "birth year of Alexander Graham Bell".to_string(),
//!         "1847".to_string(),
//!     ),
//! ]);
//! let generator = MockRewooGenerator::default();
//!
//! let pipeline = RewooPipeline::new(RewooConfig::default());
//! let outcome = pipeline
//!     .run(
//!         "Who invented the telephone, and what year was he born?",
//!         &plan_source,
//!         &retriever,
//!         &generator,
//!     )
//!     .unwrap();
//!
//! assert_eq!(
//!     outcome.evidence.get(PlaceholderVar::new(1)),
//!     Some("Alexander Graham Bell")
//! );
//! assert_eq!(outcome.evidence.get(PlaceholderVar::new(2)), Some("1847"));
//! assert!(outcome.answer.contains("1847"));
//! ```

pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::{
    MockRewooGenerator, MockRewooPlanSource, MockRewooRetriever, RewooGenerator, RewooPipeline,
    RewooPlanSource, RewooPlanner, RewooRetriever, RewooSolver, RewooWorker, evaluate_expression,
    substitute_placeholders,
};
pub use types::{
    PlaceholderVar, RewooAction, RewooConfig, RewooError, RewooEvidence, RewooOutcome, RewooPlan,
    RewooStep,
};
