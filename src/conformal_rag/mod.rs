//! Split conformal prediction for calibrated RAG answer/retrieval confidence.
//!
//! Conformal prediction wraps **any** underlying scoring function — an LLM-judge
//! score, a retrieval-relevance score, a bespoke closure — and turns it into a
//! **prediction set** carrying a *distribution-free, finite-sample marginal
//! coverage guarantee*:
//!
//! ```text
//! P(true_label in prediction_set) >= 1 - alpha
//! ```
//!
//! The guarantee holds as long as the calibration data and the test data are
//! **exchangeable** (e.g. i.i.d. draws from the same distribution). Crucially,
//! *no* assumption is made about the underlying score's distribution: it may be
//! arbitrarily miscalibrated, and the coverage guarantee still holds exactly.
//!
//! # How it differs from the `abstention` module
//!
//! Both modules can decide to *refuse to answer*, but they are fundamentally
//! different. `abstention` thresholds a heuristic blend of confidence and
//! retrieval support — a sensible rule of thumb with **no** statistical
//! guarantee. Conformal prediction's threshold instead comes from a
//! **calibration-set order statistic** and comes with a **proven** coverage
//! guarantee. Abstention answers "does this feel confident enough?"; conformal
//! prediction answers "with what provable probability does the admissible set
//! contain the truth?".
//!
//! # The algorithm (split / inductive conformal prediction)
//!
//! 1. **Nonconformity score** `s(query, candidate) -> f64` — higher means the
//!    candidate is a *worse* fit for the query. Any [`NonconformityScorer`]
//!    (including a plain closure) supplies this; [`LexicalOverlapScorer`] is a
//!    dependency-free default.
//! 2. **Calibrate** ([`ConformalCalibrator::calibrate`]) — given `n`
//!    nonconformity scores of held-out **known-correct** examples and a target
//!    miscoverage rate `alpha`, the threshold is the
//!    `ceil((n + 1) * (1 - alpha))`-th smallest of those `n` scores. The
//!    `(n + 1)` — not `n` — is the finite-sample correction that makes the
//!    guarantee hold *exactly* at finite `n`; the naive empirical
//!    `(1 - alpha)`-quantile under-covers by about `1 / (n + 1)`. When
//!    `ceil((n + 1) * (1 - alpha)) > n` (i.e. `alpha < 1 / (n + 1)`) the
//!    threshold is conceptually `+infinity`: the set admits everything.
//! 3. **Predict** ([`ConformalCalibrator::predict_set`]) — the prediction set is
//!    `{ candidate : s(query, candidate) <= threshold }`. For RAG:
//!    - **empty** set ⇒ *abstain* (nothing passes calibrated scrutiny);
//!    - **singleton** ⇒ answer directly with that candidate;
//!    - **multiple** ⇒ a calibrated ambiguity set (pick top-1 via
//!      [`PredictionSet::best`], or surface the ambiguity).
//!
//! # Group-conditional (Mondrian) coverage
//!
//! [`MondrianConformalCalibrator`] calibrates a separate threshold per
//! caller-supplied group key (e.g. a query-difficulty band), delivering tighter
//! per-group coverage than a single global threshold.
//!
//! # Example
//!
//! ```
//! use oxirag::conformal_rag::{
//!     ConformalCalibrator, ConformalConfig, LexicalOverlapScorer,
//! };
//!
//! // Calibrate on nonconformity scores of known-correct held-out answers.
//! // (Here, small illustrative scores; real calibration sets are larger.)
//! let calibration = [0.05, 0.10, 0.12, 0.20, 0.22, 0.30, 0.33, 0.41, 0.55, 0.70];
//! let config = ConformalConfig::new().with_alpha(0.2);
//! let calibrator = ConformalCalibrator::calibrate(config, &calibration)
//!     .expect("valid calibration");
//!
//! // Score fresh candidate answers with the default lexical scorer.
//! let scorer = LexicalOverlapScorer::new();
//! let query = "capital of france";
//! let candidates = [
//!     "The capital of France is Paris.", // shares many query tokens -> low s
//!     "Bananas are rich in potassium.",  // shares nothing            -> high s
//! ];
//! let set = calibrator.predict_set(&scorer, query, &candidates);
//!
//! // The relevant answer is admitted; the irrelevant one is filtered out.
//! assert!(set.best().is_some());
//! assert_eq!(set.best().unwrap().index, 0);
//! ```

pub mod calibrator;
pub mod mondrian;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use calibrator::ConformalCalibrator;
pub use mondrian::MondrianConformalCalibrator;
pub use types::{
    ConformalConfig, ConformalError, ConformalMember, LexicalOverlapScorer, NonconformityScorer,
    PredictionKind, PredictionSet,
};
