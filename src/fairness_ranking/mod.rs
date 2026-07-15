//! **Group exposure fairness in ranked lists**: measuring, and repairing, the
//! position-discounted attention a ranking allocates to protected groups.
//!
//! Every other ranking module in this crate optimizes a ranking *for the user* —
//! the most relevant document first, the most diverse set, the most credible
//! source. This module optimizes it *for the ranked*. It is the only place in the
//! crate that treats a rank position as a **resource being handed out** rather than
//! a utility being collected, and asks whether that resource is distributed
//! fairly across demographic groups.
//!
//! # The one primitive, read backwards
//!
//! The position discount `1 / log2(1 + rank)` appears throughout this crate —
//! `retrieval_eval::dcg_at_k`, `retrieval_diversity`'s `alpha-nDCG`, every `nDCG` —
//! always as a **gain to the searcher**, a weight on how much a document at that
//! rank is worth to the person reading. This module reads the identical arithmetic
//! as the **attention paid out** to whoever occupies the slot: a quantity that can
//! be over- or under-allocated to a group, and whose fair allocation is the whole
//! subject. Nothing about the formula changes; everything about what it denotes
//! does. See [`exposure`].
//!
//! # Three constructions, three scopes
//!
//! | Construction | Scope | What it does |
//! |---|---|---|
//! | `FA*IR` ([`fair`]) | one ranking, **binary** groups | constrains every prefix's protected count to a corrected binomial floor |
//! | `DELTR` ([`deltr`]) | one **scorer**, binary groups | trains a linear model whose rankings are un-disparate to begin with |
//! | Amortized equity ([`amortized`]) | a **sequence** of rankings, any groups | equalizes cumulative attention-per-relevance across queries, as an assignment problem |
//!
//! `FA*IR` is the ranked group fairness test of Zehlike et al. (`CIKM` 2017). Its
//! decision boundary is the `alpha`-quantile of an **exact binomial** law, adjusted
//! for the fact that the test is applied at every prefix by a recursive
//! multiple-test correction. Both the exact binomial `CDF` and the correction are
//! hand-rolled in [`binomial`] and [`fair`] — this crate has no binomial `CDF`
//! elsewhere, and the normal approximation it does have
//! (`watermarking::stats::normal_cdf`) is wrong for the short prefixes that matter
//! most.
//!
//! `DELTR` (Zehlike & Castillo, `WWW` 2020) adds a one-sided disparate-exposure
//! penalty to `ListNet`'s listwise loss and descends on the sum. See [`deltr`].
//!
//! Amortized equity of attention (Biega et al., `SIGIR` 2018) is the recognition
//! that a *single* ranking of two equally-relevant documents is unavoidably
//! unfair — one of them must take the smaller payout — so fairness has to be
//! amortized across a *sequence*, each query served by solving a linear
//! **assignment problem** ([`assignment`]) that best repairs the accumulated
//! inequity. See [`amortized`].
//!
//! # Distinct from `diversity_rank` and `retrieval_diversity`
//!
//! Distinct from `diversity_rank` (greedy `DPP` selection maximizing *content*
//! dissimilarity) and `retrieval_diversity` (`ILD` / S-recall / alpha-`nDCG`, which
//! measure *subtopic coverage for the user* — and whose alpha-`nDCG` *discounts*
//! re-covering a subtopic, the **opposite** of exposure equity): `fairness_ranking`
//! is the only module that reasons about **protected groups** and the
//! **position-discounted exposure allocated to them**.
//!
//! # Determinism and numerics
//!
//! `DELTR`'s weight initialization and the module's tests draw from
//! [`FairnessRng`], a seeded `SplitMix64` generator — named with the module prefix
//! because `bandit_ranker` already exports a bare `SplitMix64Rng` and this crate's
//! prelude is flat. All linear algebra is hand-rolled `std`-only `f64` over
//! row-major `Vec<f64>`, in the style of `bandit_ranker::linalg`; there is no
//! `ndarray` and no `rand` dependency.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "fairness-ranking")]
//! # {
//! use oxirag::fairness_ranking::{
//!     FairnessConfig, FairnessPolicy, FairnessRanker, GroupId, ProtectedAttribute,
//! };
//!
//! // A protected group (0) and a non-protected group (1).
//! let attribute = ProtectedAttribute::binary("gender", "female", "male").expect("valid");
//! // FA*IR aiming for parity: every prefix must hold at least the corrected
//! // binomial floor of protected candidates.
//! let policy = FairnessPolicy::RankedGroupFairness {
//!     target_proportion: 0.5,
//!     significance: 0.2,
//! };
//! let ranker = FairnessRanker::new(FairnessConfig::new(attribute, policy, 0)).expect("valid");
//!
//! // Six candidates: the protected group holds the lower-scoring ones, so the
//! // unconstrained ranking buries it.
//! let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
//! let groups = vec![
//!     GroupId(1), GroupId(1), GroupId(1), GroupId(0), GroupId(0), GroupId(0),
//! ];
//!
//! let report = ranker.report(&scores, &groups).expect("well-formed");
//! // The fair ranking reduces the exposure-per-relevance gap and passes its audit.
//! assert!(report.adjusted_gap < report.baseline_gap);
//! assert!(report.audit.expect("ranked group fairness was requested").satisfied);
//! # }
//! ```

pub mod amortized;
pub mod assignment;
pub mod binomial;
pub mod deltr;
pub mod exposure;
pub mod fair;
pub mod ranker;
pub mod rng;
pub mod types;

#[cfg(test)]
mod tests;

pub use amortized::{EquityOfAttention, ExposureReport};
pub use assignment::{AssignmentError, AssignmentSolution, solve_assignment};
pub use binomial::{
    binomial_cdf, binomial_pmf, binomial_pmf_table, binomial_quantile, exact_binomial_coefficient,
    log_binomial_coefficient, log_binomial_pmf, log_gamma,
};
pub use deltr::{DeltrConfig, DeltrLoss, DeltrModel, DeltrSample};
pub use exposure::{
    DisparateExposure, ExposureMetrics, GroupExposure, exposure_at_rank, fairness_dcg_at_k,
    fairness_ndcg_at_k, total_exposure,
};
pub use fair::{
    FairnessAudit, MTable, adjusted_significance, audit_ranking, failure_probability, fair_top_k,
    prefix_protected_counts, raw_table,
};
pub use ranker::{FairnessRanker, FairnessReport};
pub use rng::FairnessRng;
pub use types::{
    ExposureTarget, FairnessConfig, FairnessError, FairnessMetric, FairnessPolicy, FairnessResult,
    GroupId, ProtectedAttribute, ProtectedGroup,
};
