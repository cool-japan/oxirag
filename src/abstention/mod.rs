//! Abstention: selective prediction (answer-or-refuse) with a risk–coverage tradeoff.
//!
//! This module implements **selective prediction**: rather than always emitting
//! an answer, the system may *abstain* (refuse) when it is not confident enough
//! or not sufficiently grounded in retrieved evidence. Abstaining on the queries
//! it is least sure of lets a system trade a little *coverage* (how many queries
//! it answers) for a large reduction in *risk* (how often the answers it does
//! give are wrong).
//!
//! # Distinct from `self_rag` and `adaptive_rag`
//!
//! `self_rag` *critiques* an answer with reflection tokens (relevance, support,
//! utility), and `adaptive_rag` *routes* a query to a retrieval **depth** by its
//! complexity. Neither refuses to answer. Abstention is orthogonal to both: it
//! takes whatever answer confidence and retrieval support those (or any other)
//! components produced and makes the single binary decision of whether to commit
//! to an answer at all — and it quantifies the consequences of that decision
//! across a dataset with a risk–coverage curve.
//!
//! # Components
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`AbstentionDecision`] | The answer-or-abstain verdict |
//! | [`AbstentionConfig`] | Thresholds and signal blending |
//! | [`AbstentionPolicy`] | Per-query decision + risk–coverage analysis |
//! | [`AbstentionAssessment`] | Full per-query result (decision, scores, reason) |
//! | [`RiskCoverage`] | One point on the risk–coverage curve |
//!
//! Everything is pure Rust, deterministic, and free of randomness or external ML.
//!
//! # Per-query policy
//!
//! ```text
//! combined = support_weight * support + (1 - support_weight) * confidence
//! decision = Answer  iff  combined >= confidence_threshold  AND  support >= min_support
//! ```
//!
//! # Example
//!
//! ```
//! use oxirag::abstention::{AbstentionConfig, AbstentionDecision, AbstentionPolicy};
//!
//! let policy = AbstentionPolicy::new(AbstentionConfig::default());
//!
//! // Confident and well-grounded ⇒ answer.
//! let yes = policy.assess(0.9, 0.8);
//! assert_eq!(yes.decision, AbstentionDecision::Answer);
//!
//! // Plenty of support but low confidence ⇒ abstain.
//! let no = policy.assess(0.1, 0.9);
//! assert_eq!(no.decision, AbstentionDecision::Abstain);
//!
//! // Characterise a labelled dataset: at threshold 0, coverage is full.
//! let scored = [(0.9_f32, true), (0.8, true), (0.3, false), (0.1, false)];
//! let curve = policy.risk_coverage_curve(&scored, 5);
//! assert_eq!(curve.first().unwrap().coverage, 1.0);
//! ```

pub mod policy;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use policy::AbstentionPolicy;
pub use types::{
    AbstentionAssessment, AbstentionConfig, AbstentionDecision, AbstentionError, RiskCoverage,
};
