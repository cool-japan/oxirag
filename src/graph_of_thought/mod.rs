//! Graph-of-Thoughts — model reasoning as a sequence of operations over a thought DAG.
//!
//! Implements *Graph of Thoughts* (Besta et al. 2023, "Graph of Thoughts:
//! Solving Elaborate Problems with Large Language Models"). It generalizes
//! Tree-of-Thoughts: individual thoughts are vertices of a directed acyclic
//! [`ThoughtGraph`], and a problem is solved by executing a *plan* of graph
//! [`GotOperation`]s over them:
//!
//! - **Generate** — branch `k` new thoughts from the current frontier.
//! - **Aggregate** — merge several thoughts into one synthesized thought.
//! - **Refine** — improve a thought in place (a self-edge).
//! - **`KeepBest`** — prune the frontier to the top-`n` thoughts by score.
//!
//! Unlike `tree_of_thought` (a tree explored by search), Graph-of-Thoughts
//! permits *aggregation* — combining independent reasoning branches into one
//! thought — yielding a true DAG rather than a tree.
//!
//! The language model is supplied by the caller through three **pure-sync**
//! traits — [`ThoughtGenerator`], [`ThoughtScorer`], and [`ThoughtAggregator`] —
//! mirroring the caller-supplies-executor pattern used by `self_consistency` and
//! `self_ask`. Deterministic [`MockThoughtGenerator`], [`MockThoughtScorer`], and
//! [`MockThoughtAggregator`] implementations are provided for testing. There is
//! no async and no I/O.
//!
//! # Example
//!
//! ```
//! use oxirag::graph_of_thought::{
//!     GotConfig, GotOperation, GraphOfThoughtEngine, MockThoughtAggregator,
//!     MockThoughtGenerator, MockThoughtScorer,
//! };
//!
//! let generator = MockThoughtGenerator::new("idea");
//! let scorer = MockThoughtScorer::default();
//! let aggregator = MockThoughtAggregator::default();
//!
//! let plan = [
//!     GotOperation::Generate { k: 3 },
//!     GotOperation::KeepBest { n: 2 },
//!     GotOperation::Aggregate,
//! ];
//!
//! let engine = GraphOfThoughtEngine::new(GotConfig::default());
//! let out = engine
//!     .run("solve the puzzle", &plan, &generator, &scorer, &aggregator)
//!     .unwrap();
//!
//! assert!(!out.final_answer.is_empty());
//! assert_eq!(out.best_thought, out.final_answer);
//! ```

pub mod engine;
pub mod graph;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::GraphOfThoughtEngine;
pub use graph::{Thought, ThoughtGraph};
pub use types::{
    GotConfig, GotError, GotOperation, GotOutput, MockThoughtAggregator, MockThoughtGenerator,
    MockThoughtScorer, ThoughtAggregator, ThoughtGenerator, ThoughtScorer,
};
