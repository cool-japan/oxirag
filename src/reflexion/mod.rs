//! Reflexion — verbal self-reflection with episodic memory across retry attempts.
pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;
pub use engine::ReflexionEngine;
pub use types::{
    Attempt, AttemptEvaluator, AttemptScore, EpisodicMemory, HeuristicEvaluator,
    HeuristicSelfReflector, Reflection, ReflexionConfig, ReflexionError, ReflexionOutcome,
    SelfReflector,
};
