//! `learning_to_rank` — feature-based, **trained** Learning-to-Rank
//! (`RankNet`/`LambdaMART`-lite).
//!
//! This module fits a ranking model from labelled *preference-pair* data with
//! a real gradient-descent training loop. Each `(query, document)` pair is
//! reduced to a fixed-order [`LtrFeatureVector`] of heterogeneous numeric
//! retrieval-metadata signals — a self-computed `BM25`-style relevance score, a
//! recency signal, an `FNV-1a` pseudo-embedding cosine similarity, a
//! popularity count, and a length signal (see [`LtrFeatureExtractor`]). The
//! [`LtrEngine`] then learns one weight per signal by minimising the pairwise
//! logistic (`RankNet`) loss over labelled preferences (see the `train` module)
//! and ranks candidates by the resulting linear score.
//!
//! # How this differs from `cross_encoder`
//!
//! [`crate::cross_encoder`] is a **fixed, default-weight linear combiner** over
//! query/document **text-interaction** features (exact-match ratio, term
//! overlap, IDF-weighted overlap, bigram match, …). Its `FeatureWeights` are
//! hand-set constants that are **never trained** — no labels, no optimisation,
//! no gradient descent.
//!
//! `learning_to_rank` is the opposite: its weights are **fitted from labelled
//! preference-pair data** via genuine gradient descent, and its features are
//! not text comparisons but **heterogeneous numeric retrieval-metadata
//! signals** (relevance score, recency, embedding similarity, popularity,
//! length). The model *learns* how to trade those signals off from data rather
//! than assuming a fixed weighting.
//!
//! # The algorithm
//!
//! For a preference pair with feature vectors `x_a`, `x_b` and target
//! probability `y` that `a` outranks `b`, the model scores each document
//! linearly, `s = w · x`, and predicts `P(a≻b) = sigmoid(w · (x_a − x_b))`.
//! Training minimises the mean pairwise binary cross-entropy (plus an optional
//! `L2` penalty), which is convex in `w`, so full-batch gradient descent
//! decreases the loss monotonically for a small enough learning rate and
//! converges to the unique optimum. Training is fully deterministic: the same
//! [`LtrTrainingSet`] and [`LtrConfig`] always yield byte-identical weights.
//!
//! # Example
//!
//! ```
//! use oxirag::learning_to_rank::{
//!     LtrConfig, LtrEngine, LtrFeatureVector, LtrTrainingPair, LtrTrainingSet,
//! };
//!
//! // A two-signal problem where the first signal is decisive: whenever it is
//! // larger, that document should rank higher.
//! let mut training_set = LtrTrainingSet::new();
//! for &(hi, lo) in &[(0.9_f64, 0.1_f64), (0.8, 0.2), (0.7, 0.3), (0.95, 0.05)] {
//!     training_set.push(LtrTrainingPair::preferred(
//!         0,
//!         LtrFeatureVector::new(vec![hi, 0.5]),
//!         LtrFeatureVector::new(vec![lo, 0.5]),
//!     ));
//! }
//!
//! let config = LtrConfig::new()
//!     .with_feature_dim(2)
//!     .with_learning_rate(0.3)
//!     .with_epochs(500);
//!
//! let engine = LtrEngine::new();
//! let model = engine.train(&training_set, &config).expect("training succeeds");
//!
//! // The loss went down: the model learned something.
//! assert!(model.final_loss < model.loss_history[0]);
//!
//! // It now ranks a held-out candidate list by the decisive signal.
//! let candidates = vec![
//!     LtrFeatureVector::new(vec![0.2, 0.9]),
//!     LtrFeatureVector::new(vec![0.85, 0.1]),
//!     LtrFeatureVector::new(vec![0.5, 0.5]),
//! ];
//! let ranked = engine.rank(&model, &candidates);
//! assert_eq!(ranked[0].0, 1); // the candidate with the largest first signal
//! ```

pub mod engine;
pub mod features;
#[cfg(test)]
mod tests;
pub mod train;
pub mod types;

pub use engine::LtrEngine;
pub use features::LtrFeatureExtractor;
pub use types::{
    LTR_DEFAULT_FEATURE_DIM, LTR_FEATURE_BM25, LTR_FEATURE_EMBEDDING_SIM, LTR_FEATURE_LENGTH,
    LTR_FEATURE_POPULARITY, LTR_FEATURE_RECENCY, LtrConfig, LtrDocument, LtrError,
    LtrFeatureVector, LtrModel, LtrResult, LtrTrainingPair, LtrTrainingSet,
};
