//! Query intent classification and adaptive retrieval-strategy routing.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`IntentClassifier`] | Classify a query into an [`IntentScores`] distribution |
//! | [`HeuristicIntentClassifier`] | Keyword/pattern heuristic classifier |
//! | [`MockIntentClassifier`] | Scripted test double |
//! | [`QueryRouter`] | Map intent scores to a [`RoutingDecision`] |
//! | [`RouterConfig`] | Routing table + fallbacks + top-k overrides |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "query-routing")] {
//! use oxirag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let router = QueryRouter::new(
//!     HeuristicIntentClassifier::new(),
//!     RouterConfig::default(),
//! );
//! let decision = router.route("compare Rust vs Go").await.unwrap();
//! assert_eq!(decision.strategy, RoutingStrategy::HybridSearch);
//! # }
//! # }
//! ```

pub mod classifier;
pub mod router;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use classifier::{HeuristicIntentClassifier, IntentClassifier, MockIntentClassifier};
pub use router::QueryRouter;
pub use types::{
    IntentScores, QueryIntent, QueryRouterError, RouterConfig, RoutingDecision, RoutingStrategy,
};
