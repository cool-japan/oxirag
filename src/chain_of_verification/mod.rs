//! Chain-of-Verification (`CoVe`) — verify draft answers with targeted re-retrieval.
pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;
pub mod verifier;
pub use engine::ChainOfVerificationEngine;
pub use types::{
    ChainOfVerificationError, ClaimVerdict, CoVeConfig, CoVeOutput, HeuristicQuestionPlanner,
    QuestionPlanner, VerificationAnswer, VerificationQuestion,
};
