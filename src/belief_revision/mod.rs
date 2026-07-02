//! Bayesian belief revision over a stream of retrieved evidence.
//!
//! This module maintains an explicit **posterior probability distribution**
//! over a fixed set of candidate hypotheses (answers) and revises it — via
//! genuine Bayesian likelihood-ratio math — as each new piece of retrieved
//! evidence arrives. Given hypotheses `h_1 .. h_k`, a prior `P(h)`, and
//! evidence `e` with per-hypothesis likelihoods `P(e | h)`, it computes the
//! posterior `P(h | e) ∝ P(h) * P(e | h)` and folds evidence in sequentially,
//! so the posterior after `t` observations is the prior for observation
//! `t + 1`.
//!
//! The update is carried out in **log-probability / log-odds space** for
//! numerical stability: multiplying many small likelihoods together in linear
//! space underflows to zero after enough updates (at which point the final
//! normalization is `0 / 0 = NaN`), whereas adding their logarithms never
//! does. Renormalization uses a stable `logsumexp`. See [`bayes`] for the
//! math and [`engine`] for the lifecycle.
//!
//! # How it differs from its neighbours
//!
//! Three modules in this crate touch "which retrieved claim should I trust?",
//! but only this one maintains a probability distribution and updates it with
//! Bayes' rule:
//!
//! * [`knowledge_conflict`](crate::knowledge_conflict) detects **pairwise
//!   contradictions** between passages (negation / numeric / temporal
//!   disagreements) and resolves them under a **fixed policy** (recency,
//!   authority, or majority). There is no probability distribution at all —
//!   the output is a discrete winner per conflict, not a posterior.
//! * [`conformal_rag`](crate::conformal_rag) turns a calibration set into a
//!   **static threshold** (an order statistic) and emits a coverage-guaranteed
//!   **prediction set** — the set of candidates whose nonconformity is below
//!   that threshold. The threshold does not move as new evidence arrives, and
//!   the output is a set membership decision, not an incrementally-updated
//!   probability.
//! * `belief_revision` (this module), by contrast, holds an explicit
//!   **posterior probability distribution** over the candidate hypotheses and
//!   updates it with **genuine Bayesian likelihood-ratio math** every time a
//!   new piece of evidence arrives. The value it delivers is a live,
//!   normalized `P(h | e_1, .., e_t)` that shifts continuously with the
//!   accumulating evidence — not a fixed policy verdict and not a static
//!   calibration threshold.
//!
//! # Example
//!
//! ```
//! use oxirag::belief_revision::{
//!     BeliefHypothesis, BeliefRevisionConfig, BeliefRevisionEngine, EvidenceUpdate,
//! };
//!
//! let engine = BeliefRevisionEngine::new(BeliefRevisionConfig::default());
//! let hypotheses = vec![
//!     BeliefHypothesis::new("paris", "The capital of France is Paris."),
//!     BeliefHypothesis::new("lyon", "The capital of France is Lyon."),
//! ];
//!
//! // Uniform prior (`None`) over the two hypotheses.
//! let mut state = engine.init(hypotheses, None).expect("valid init");
//!
//! // Evidence that strongly supports "paris": P(e | paris) = 0.9,
//! // P(e | lyon) = 0.1.
//! let evidence = EvidenceUpdate::new("Most sources name Paris.")
//!     .with_likelihood("paris", 0.9)
//!     .with_likelihood("lyon", 0.1);
//! engine.update(&mut state, &evidence).expect("valid update");
//!
//! // The posterior is a valid probability distribution...
//! let posterior = state.posterior();
//! let total: f64 = posterior.iter().map(|(_, p)| p).sum();
//! assert!((total - 1.0).abs() < 1e-9);
//!
//! // ...and it now favors "paris".
//! let best = engine.most_likely(&state).expect("non-empty state");
//! assert_eq!(best.id, "paris");
//! assert!(state.probability_of("paris").expect("present") > 0.8);
//! ```

pub mod bayes;
pub mod engine;
#[cfg(test)]
mod tests;
pub mod types;

pub use bayes::LikelihoodRatio;
pub use engine::BeliefRevisionEngine;
pub use types::{
    BeliefHypothesis, BeliefRevisionConfig, BeliefRevisionError, BeliefRevisionResult, BeliefState,
    EvidenceUpdate,
};
