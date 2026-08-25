//! Self-Ask (Press et al. 2022) — compositional multi-hop reasoning via explicit
//! self-questioning.
//!
//! The model iteratively decides whether a follow-up question is needed, asks it,
//! answers it (optionally via retrieval through a [`SubAnswerer`]), and finally
//! composes the overall answer. This interleaves self-questioning with answering,
//! so each follow-up can depend on earlier answers — distinct from upfront
//! `query_decomposition` and from graph-based `multi_hop` traversal.
//!
//! # Example
//!
//! ```
//! use oxirag::self_ask::{
//!     MockSelfAskModel, MockSubAnswerer, SelfAskConfig, SelfAskEngine,
//! };
//!
//! let model = MockSelfAskModel::new(
//!     vec![
//!         "Who directed Inception?".to_string(),
//!         "When was that director born?".to_string(),
//!     ],
//!     "Christopher Nolan was born in 1970.",
//! );
//! let answerer = MockSubAnswerer::new(vec![
//!     ("directed".to_string(), "Christopher Nolan".to_string()),
//!     ("born".to_string(), "1970".to_string()),
//! ]);
//!
//! let engine = SelfAskEngine::new(SelfAskConfig::default());
//! let trace = engine
//!     .run("How old was Inception's director in 1970?", &model, &answerer)
//!     .unwrap();
//!
//! assert_eq!(trace.num_hops, 2);
//! assert_eq!(trace.follow_ups[0].answer, "Christopher Nolan");
//! assert_eq!(trace.final_answer, "Christopher Nolan was born in 1970.");
//! ```

pub mod engine;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use engine::SelfAskEngine;
pub use types::{
    FollowUp, MockSelfAskModel, MockSubAnswerer, SelfAskConfig, SelfAskError, SelfAskModel,
    SelfAskTrace, SubAnswerer,
};
