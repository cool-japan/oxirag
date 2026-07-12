//! Types, configuration, and numerical constants for language-model retrieval.
//!
//! This file holds the *vocabulary* of the module: the three query-likelihood
//! smoothing schemes ([`LmSmoothing`]), the two Divergence-From-Randomness
//! models ([`DfrModel`]), the scoring-model selector ([`LmScoringModel`]), the
//! index configuration ([`LmRetrievalConfig`]), the RM3 feedback configuration
//! ([`Rm3Config`]) and its output ([`Rm3ExpandedQuery`] / [`Rm3Term`]), the
//! search result ([`LmHit`]), and the error type ([`LmRetrievalError`]).
//!
//! The *mathematics* of each of these lives next door: smoothing in
//! [`super::smoothing`], the `DFR` models in [`super::dfr`], and the relevance
//! model in [`super::rm3`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Numerical constants ──────────────────────────────────────────────────────

/// The smallest probability any document language model is allowed to assign
/// to a query term before its logarithm is taken.
///
/// Every smoothing scheme in this module is *designed* to keep
/// `P(w | D) > 0` for every term in the collection vocabulary — that is the
/// entire point of smoothing, and it is why `log P(w | D)` is finite in the
/// well-posed case. But the schemes degenerate at the boundaries of their
/// parameter ranges:
///
/// - Dirichlet with `μ = 0` is the unsmoothed maximum-likelihood estimate, so
///   `P(w | D) = 0` for any term absent from `D`.
/// - Jelinek-Mercer with `λ = 0` is likewise the bare `MLE`.
/// - Absolute discounting with `δ = 0` is, again, the bare `MLE`.
///
/// Those are legitimate (if useless) configurations, and a retrieval library
/// must not answer them with `-inf` or `NaN`. Clamping the probability at
/// `1e-12` bounds `log P(w | D)` below at `ln(1e-12) ≈ -27.63`: still a
/// crushing penalty that ranks any document missing the term below every
/// document containing it, but a *finite* one that composes safely with the
/// rest of the score.
///
/// The value is chosen far below any probability a realistic collection model
/// can produce — `P(w | C) = cf(w) / |C|` only reaches `1e-12` for a
/// once-occurring term in a collection of a *trillion* tokens — so the clamp
/// is inert on every well-posed configuration and never perturbs a score that
/// the theory defines.
pub const LM_MIN_PROBABILITY: f64 = 1e-12;

/// Lower bound applied to the normalized term frequency `tfn` inside `PL2`.
///
/// `PL2`'s information content is a *Stirling approximation* of the Poisson
/// surprisal (see [`super::dfr`]), and the Stirling series is invalid as
/// `tfn → 0`: the surrogate `0.5 · log2(2π · tfn)` diverges to `-∞`, even
/// though the exact surprisal converges to the perfectly finite `λ · log2(e)`
/// (because `0! = 1`, not `√(2π·0) = 0`). Flooring `tfn` at `1e-9` bounds the
/// surrogate's error instead of silently emitting `-inf`.
///
/// The floor is unreachable in practice: `tfn = tf · log2(1 + c·avgdl/dl)`
/// falls below `1e-9` only when `dl > 1.4e9 · c · avgdl` — i.e. for a document
/// a billion times longer than the collection average. It exists purely as a
/// numerical guard against pathological configurations (`c → 0`) and
/// adversarial inputs.
pub const PL2_MIN_TFN: f64 = 1e-9;

// ── LmSmoothing ──────────────────────────────────────────────────────────────

/// How a document's unigram language model `P(w | D)` is smoothed against the
/// collection model `P(w | C) = cf(w) / |C|`.
///
/// Every scheme has the same shape — a *foreground* term derived from the
/// document's own counts, plus a *background* term that is some coefficient
/// times `P(w | C)`:
///
/// ```text
/// P(w | D) = S_D(w) + K_D · P(w | C)
/// ```
///
/// where `S_D(w) = 0` whenever `tf(w, D) = 0`. It is this decomposition that
/// makes both the zero-frequency problem and `RM1` tractable: the background
/// coefficient `K_D` does not depend on `w` at all (see
/// [`super::smoothing`] and [`super::rm3`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LmSmoothing {
    /// Bayesian smoothing with a Dirichlet prior (Zhai & Lafferty, 2001):
    ///
    /// ```text
    /// P(w | D) = (tf(w, D) + μ · P(w | C)) / (|D| + μ)
    /// ```
    ///
    /// The prior contributes `μ` pseudo-tokens distributed according to the
    /// collection model, so the amount of smoothing a document receives is
    /// *inversely proportional to its length*: the background coefficient is
    /// `K_D = μ / (|D| + μ)`, which shrinks as `|D|` grows. This is exactly
    /// the behaviour one wants — a long document is strong evidence in its own
    /// right and needs little help from the prior — and it is why Dirichlet
    /// smoothing rewards *absolute* within-document evidence, in contrast to
    /// [`LmSmoothing::JelinekMercer`], whose smoothing weight is a constant.
    Dirichlet {
        /// The Dirichlet prior mass, in pseudo-tokens. Must be finite and
        /// non-negative. `2000` is the standard default for prose collections.
        /// `μ = 0` degenerates to the unsmoothed maximum-likelihood estimate.
        mu: f64,
    },
    /// Linear interpolation between the document's maximum-likelihood estimate
    /// and the collection model (Jelinek & Mercer, 1980):
    ///
    /// ```text
    /// P(w | D) = (1 - λ) · tf(w, D) / |D| + λ · P(w | C)
    /// ```
    ///
    /// The background coefficient is the constant `K_D = λ`, *independent of
    /// document length*, so this scheme scores documents on the *relative*
    /// frequency of query terms. A short document in which a query term makes
    /// up a large fraction of the text can therefore beat a long document that
    /// contains the term many more times — the opposite of Dirichlet's
    /// preference.
    JelinekMercer {
        /// The weight placed on the collection model. Must be finite and in
        /// `[0, 1]`. `λ = 0` is the bare maximum-likelihood estimate; `λ = 1`
        /// is the pure collection model, under which every document receives
        /// an identical score.
        lambda: f64,
    },
    /// Absolute discounting (Ney, Essen & Kneser, 1994):
    ///
    /// ```text
    /// P(w | D) = max(tf(w, D) - δ, 0) / |D| + (δ · |D|_unique / |D|) · P(w | C)
    /// ```
    ///
    /// A fixed mass `δ` is subtracted from the count of *each distinct term
    /// observed in the document*, and the total mass so freed —
    /// `δ · |D|_unique / |D|` — is redistributed over the whole vocabulary
    /// according to the collection model. Because the escaped mass is
    /// *exactly* the mass removed, the result is a proper distribution; see
    /// [`super::smoothing`] for the normalization proof, which is also the
    /// reason `δ` is constrained to `[0, 1]`.
    AbsoluteDiscounting {
        /// The count discounted from each distinct observed term. Must be
        /// finite and in `[0, 1]`; values above `1` would truncate
        /// `max(tf - δ, 0)` to zero for singleton terms and break
        /// normalization. `δ = 0` is the bare maximum-likelihood estimate.
        delta: f64,
    },
}

impl Default for LmSmoothing {
    /// Dirichlet smoothing with `μ = 2000`, the standard setting from Zhai &
    /// Lafferty's smoothing study.
    fn default() -> Self {
        Self::Dirichlet { mu: 2000.0 }
    }
}

// ── DfrModel ─────────────────────────────────────────────────────────────────

/// A Divergence-From-Randomness weighting model (Amati & van Rijsbergen,
/// 2002).
///
/// Unlike query likelihood — which is a *generative* model, asking "how likely
/// is this query under this document's language model?" — `DFR` asks how far a
/// term's observed within-document frequency *diverges* from what a random
/// process would produce, and weights that divergence by how much of it can be
/// attributed to the term being *elite* in the document. See [`super::dfr`]
/// for the derivations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DfrModel {
    /// Poisson randomness model with Laplace after-effect and normalization 2.
    ///
    /// The randomness model is a Poisson process with rate
    /// `λ_w = cf(w) / N`; the "amount of information" is its surprisal,
    /// evaluated via Stirling's approximation; the after-effect factor
    /// `1 / (tfn + 1)` comes from Laplace's law of succession. Term
    /// frequencies are length-normalized with normalization 2:
    /// `tfn = tf · log2(1 + c · avgdl / dl)`.
    Pl2 {
        /// The length-normalization hyper-parameter. Must be finite and
        /// strictly positive. Larger `c` normalizes long documents more
        /// aggressively. `1.0` is a reasonable default for short documents,
        /// `7.0` for long ones.
        c: f64,
    },
    /// `DPH` — the hyper-geometric `DFR` model with Popper's normalization.
    ///
    /// Completely parameter-free: it derives its length normalization from the
    /// within-document relative frequency `tf / dl` rather than from a tunable
    /// constant, which makes it a strong baseline when no training data is
    /// available to tune `PL2`'s `c`.
    Dph,
}

impl Default for DfrModel {
    /// `PL2` with `c = 1.0`.
    fn default() -> Self {
        Self::Pl2 { c: 1.0 }
    }
}

// ── LmScoringModel ───────────────────────────────────────────────────────────

/// Which of the two scoring families ranks documents in an
/// [`LmRetrievalIndex`](super::LmRetrievalIndex).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LmScoringModel {
    /// Query likelihood: `score(D, Q) = Σ_w qtf(w) · log P(w | D)`, with
    /// `P(w | D)` smoothed according to
    /// [`LmRetrievalConfig::smoothing`].
    ///
    /// Because a smoothed language model assigns non-zero probability to
    /// *every* term in the vocabulary, query likelihood is well defined for
    /// every document in the collection — including documents containing none
    /// of the query's terms — and the index therefore scores all of them. That
    /// is not merely a formality: a document containing a query term exactly
    /// once can legitimately score *below* one that does not contain it at
    /// all, if the former is enormously longer (`(1 + μp) / (|D| + μ)` can
    /// fall below `μp / (|D'| + μ)` when `|D| ≫ |D'|`), so restricting the
    /// candidate set to the posting lists would silently change the ranking.
    QueryLikelihood,
    /// Divergence From Randomness, using the given model.
    ///
    /// `DFR` is not a generative model over documents; it measures the
    /// divergence of an *observed* term frequency from a randomness model, and
    /// a term that does not occur in a document contributes exactly zero. The
    /// index therefore scores only the union of the query terms' posting
    /// lists — documents outside it would all receive an identical score of
    /// `0` and carry no evidence at all.
    Dfr(DfrModel),
}

impl Default for LmScoringModel {
    /// Query likelihood.
    fn default() -> Self {
        Self::QueryLikelihood
    }
}

// ── LmRetrievalConfig ────────────────────────────────────────────────────────

/// Configuration of an [`LmRetrievalIndex`](super::LmRetrievalIndex).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LmRetrievalConfig {
    /// Which scoring family ranks documents in
    /// [`search`](super::LmRetrievalIndex::search).
    pub model: LmScoringModel,
    /// The smoothing scheme for document language models.
    ///
    /// This field is **always** meaningful, even when [`Self::model`] selects
    /// [`LmScoringModel::Dfr`]: `RM3` estimates its relevance model from
    /// *smoothed document language models* weighted by *query likelihood*
    /// (see [`super::rm3`]), so it needs a smoothing scheme regardless of what
    /// produced the first-pass ranking.
    pub smoothing: LmSmoothing,
    /// Whether the built-in tokenizer lower-cases terms. Documents added as
    /// pre-tokenized term lists bypass the tokenizer entirely and are indexed
    /// verbatim.
    pub lowercase: bool,
}

impl Default for LmRetrievalConfig {
    fn default() -> Self {
        Self {
            model: LmScoringModel::default(),
            smoothing: LmSmoothing::default(),
            lowercase: true,
        }
    }
}

impl LmRetrievalConfig {
    /// A query-likelihood configuration with the given smoothing scheme.
    #[must_use]
    pub fn query_likelihood(smoothing: LmSmoothing) -> Self {
        Self {
            model: LmScoringModel::QueryLikelihood,
            smoothing,
            lowercase: true,
        }
    }

    /// A `DFR` configuration with the given model.
    ///
    /// [`Self::smoothing`] is left at its default, because `RM3` still needs a
    /// smoothing scheme even when `DFR` produces the first-pass ranking.
    #[must_use]
    pub fn dfr(model: DfrModel) -> Self {
        Self {
            model: LmScoringModel::Dfr(model),
            smoothing: LmSmoothing::default(),
            lowercase: true,
        }
    }

    /// Override the scoring model.
    #[must_use]
    pub fn with_model(mut self, model: LmScoringModel) -> Self {
        self.model = model;
        self
    }

    /// Override the smoothing scheme.
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: LmSmoothing) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Override whether the built-in tokenizer lower-cases terms.
    #[must_use]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Validate every hyper-parameter this configuration carries.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] if the smoothing or
    /// `DFR` hyper-parameters lie outside the range in which their model is
    /// mathematically well-posed.
    pub fn validate(&self) -> Result<(), LmRetrievalError> {
        self.smoothing.validate()?;
        if let LmScoringModel::Dfr(model) = self.model {
            model.validate()?;
        }
        Ok(())
    }
}

// ── Rm3Config ────────────────────────────────────────────────────────────────

/// Configuration of `RM3` pseudo-relevance feedback.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rm3Config {
    /// How many top-ranked documents form the feedback set `F`. Must be at
    /// least `1`.
    pub fb_docs: usize,
    /// How many terms the expanded query retains. Must be at least `1`.
    pub fb_terms: usize,
    /// The interpolation weight on the *original* query in
    /// `P'(w | Q) = α · P_ML(w | Q) + (1 - α) · P(w | R)`. Must be finite and
    /// in `[0, 1]`. `α = 1` reproduces the original query exactly; `α = 0` is
    /// pure `RM1`.
    pub alpha: f64,
}

impl Default for Rm3Config {
    /// The classic setting: 10 feedback documents, 10 expansion terms,
    /// `α = 0.5`.
    fn default() -> Self {
        Self {
            fb_docs: 10,
            fb_terms: 10,
            alpha: 0.5,
        }
    }
}

impl Rm3Config {
    /// Create an `RM3` configuration.
    #[must_use]
    pub fn new(fb_docs: usize, fb_terms: usize, alpha: f64) -> Self {
        Self {
            fb_docs,
            fb_terms,
            alpha,
        }
    }

    /// Validate the feedback-set size, expansion width, and interpolation
    /// weight.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] if `fb_docs` or
    /// `fb_terms` is zero, or if `alpha` is not a finite value in `[0, 1]`.
    pub fn validate(&self) -> Result<(), LmRetrievalError> {
        if self.fb_docs == 0 {
            return Err(LmRetrievalError::InvalidParameter {
                parameter: "fb_docs".to_string(),
                value: 0.0,
                reason: "the feedback set must contain at least one document".to_string(),
            });
        }
        if self.fb_terms == 0 {
            return Err(LmRetrievalError::InvalidParameter {
                parameter: "fb_terms".to_string(),
                value: 0.0,
                reason: "the expanded query must retain at least one term".to_string(),
            });
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(LmRetrievalError::InvalidParameter {
                parameter: "alpha".to_string(),
                value: self.alpha,
                reason: "the RM3 interpolation weight must be a finite value in [0, 1]".to_string(),
            });
        }
        Ok(())
    }
}

// ── Rm3Term / Rm3ExpandedQuery ───────────────────────────────────────────────

/// One term of an `RM3`-expanded query, with its (renormalized) weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rm3Term {
    /// The term itself.
    pub term: String,
    /// The term's weight in the expanded query. Weights over an
    /// [`Rm3ExpandedQuery`] sum to `1`.
    pub weight: f64,
}

/// The output of `RM3`: a weighted bag of terms, ready to be scored by
/// [`LmRetrievalIndex::search_expanded`](super::LmRetrievalIndex::search_expanded).
///
/// Terms are ordered by descending weight, ties broken by ascending term, so
/// the representation is deterministic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Rm3ExpandedQuery {
    /// The weighted terms, heaviest first.
    pub terms: Vec<Rm3Term>,
}

impl Rm3ExpandedQuery {
    /// Build an expanded query from a term-weight list, verbatim (no sorting,
    /// no renormalization).
    #[must_use]
    pub fn new(terms: Vec<Rm3Term>) -> Self {
        Self { terms }
    }

    /// The number of terms in the expanded query.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the expanded query is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The weight assigned to `term`, or `0.0` if it is not present.
    #[must_use]
    pub fn weight_of(&self, term: &str) -> f64 {
        self.terms
            .iter()
            .find(|t| t.term == term)
            .map_or(0.0, |t| t.weight)
    }

    /// Whether `term` survived expansion.
    #[must_use]
    pub fn contains(&self, term: &str) -> bool {
        self.terms.iter().any(|t| t.term == term)
    }

    /// The total weight, which is `1` for any query produced by `RM3`.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.terms.iter().map(|t| t.weight).sum()
    }

    /// The expanded query as `(term, weight)` pairs, the form the scorers
    /// consume.
    #[must_use]
    pub fn as_weighted_terms(&self) -> Vec<(String, f64)> {
        self.terms
            .iter()
            .map(|t| (t.term.clone(), t.weight))
            .collect()
    }
}

// ── LmHit ────────────────────────────────────────────────────────────────────

/// One ranked document returned by an
/// [`LmRetrievalIndex`](super::LmRetrievalIndex) search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LmHit {
    /// The document's identifier, as supplied to
    /// [`add_document`](super::LmRetrievalIndex::add_document).
    pub document_id: String,
    /// The model's score.
    ///
    /// For [`LmScoringModel::QueryLikelihood`] this is a *log*-likelihood and
    /// is therefore negative (larger, i.e. closer to zero, is better). For
    /// [`LmScoringModel::Dfr`] it is a divergence weight, usually but not
    /// always positive. In both cases the ranking is by descending score.
    pub score: f64,
    /// The document's zero-based rank in the returned list.
    pub rank: usize,
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the `language_model_retrieval` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LmRetrievalError {
    /// A search or feedback operation was attempted before
    /// [`build`](super::LmRetrievalIndex::build) was called, or after a
    /// document was added and the derived collection statistics were
    /// invalidated.
    #[error("index has not been built; call build() before searching")]
    NotBuilt,
    /// Two documents were added under the same identifier.
    #[error("duplicate document id: {document_id}")]
    DuplicateDocumentId {
        /// The identifier that was added twice.
        document_id: String,
    },
    /// A query contained no terms at all (it was empty, or the tokenizer
    /// discarded everything in it).
    #[error("query must contain at least one term")]
    EmptyQuery,
    /// A model hyper-parameter lay outside the range in which its model is
    /// well-posed.
    #[error("invalid parameter {parameter} = {value}: {reason}")]
    InvalidParameter {
        /// The offending parameter's name.
        parameter: String,
        /// The value that was rejected.
        value: f64,
        /// Why the value is not admissible.
        reason: String,
    },
}

/// Convenience alias for this module's fallible return type.
pub type LmRetrievalResult<T> = Result<T, LmRetrievalError>;

// ── Tokenization ─────────────────────────────────────────────────────────────

/// The module's tokenizer: split on every non-alphanumeric character, drop
/// empty pieces, and optionally lower-case.
///
/// Deliberately minimal and deterministic. Real deployments will want
/// stemming and stop-word removal (stop words matter a great deal to `RM3`,
/// whose relevance model happily promotes high-collection-probability function
/// words); pre-tokenize with
/// [`add_document_terms`](super::LmRetrievalIndex::add_document_terms) to plug
/// in a better analyzer.
#[must_use]
pub fn lm_tokenize(text: &str, lowercase: bool) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|piece| !piece.is_empty())
        .map(|piece| {
            if lowercase {
                piece.to_lowercase()
            } else {
                piece.to_string()
            }
        })
        .collect()
}
