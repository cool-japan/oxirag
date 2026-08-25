//! LLM-as-judge evaluation — pointwise, pairwise, and reference-based scoring.
pub mod judge;
pub mod rubric;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub use judge::LlmJudge;
pub use types::{
    Criterion, CriterionScores, HeuristicJudge, JudgeContext, JudgeMode, JudgeModel,
    LlmJudgeConfig, LlmJudgeError, PairwiseVerdict, PointwiseVerdict, Pref, Rubric,
};
