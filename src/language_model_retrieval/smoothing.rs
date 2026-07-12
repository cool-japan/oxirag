//! The three query-likelihood smoothing schemes, and the algebra that makes
//! them safe in log space.
//!
//! # The zero-frequency problem
//!
//! Query likelihood ranks a document `D` by the probability its unigram
//! language model assigns to the query:
//!
//! ```text
//! score(D, Q) = Σ_{w ∈ Q} qtf(w) · log P(w | D)
//! ```
//!
//! Estimate `P(w | D)` by maximum likelihood — `tf(w, D) / |D|` — and the
//! whole thing collapses: *one* query term missing from `D` sends
//! `log P(w | D)` to `-∞` and the document's score with it, regardless of how
//! well the rest of the query matches. Worse, this is not a rare corner case
//! but the *typical* case, since most documents contain most of the vocabulary
//! zero times.
//!
//! Smoothing is the fix, and every scheme here implements the same idea: take
//! probability mass away from the terms the document *did* produce and give it
//! to the terms it did not, in proportion to how common those terms are in the
//! collection as a whole. The collection model is the maximum-likelihood
//! estimate over the concatenation of every document,
//!
//! ```text
//! P(w | C) = cf(w) / |C|
//! ```
//!
//! where `cf(w)` is the *collection term frequency* (`w`'s total occurrences,
//! not the number of documents containing it) and `|C| = Σ_w cf(w)` is the
//! total token count. Since `P(w | C) > 0` for every term the collection
//! contains, backing off to it guarantees `P(w | D) > 0`, and the logarithm is
//! finite.
//!
//! # The common form
//!
//! Every scheme in this module decomposes as
//!
//! ```text
//! P(w | D) = S_D(w) + K_D · P(w | C)
//! ```
//!
//! — a *foreground* part `S_D(w)` that reads the document's own counts and is
//! **zero whenever `tf(w, D) = 0`**, plus a *background* part whose
//! coefficient `K_D` does not depend on `w`:
//!
//! | scheme | `S_D(w)` | `K_D` |
//! |---|---|---|
//! | Dirichlet, prior `μ` | `tf / (\|D\| + μ)` | `μ / (\|D\| + μ)` |
//! | Jelinek-Mercer, `λ` | `(1 - λ) · tf / \|D\|` | `λ` |
//! | Absolute discounting, `δ` | `max(tf - δ, 0) / \|D\|` | `δ · \|D\|_unique / \|D\|` |
//!
//! Adding the columns back together reproduces the three textbook formulas
//! exactly. Two things fall out of the decomposition for free.
//!
//! **It proves each scheme is a proper distribution.** Summing over the
//! vocabulary `V`, and using `Σ_{w ∈ V} P(w | C) = 1`:
//!
//! ```text
//! Dirichlet:  Σ_V tf / (|D| + μ)          + μ / (|D| + μ)          = |D| / (|D| + μ) + μ / (|D| + μ) = 1
//! J-M:        Σ_V (1 - λ) · tf / |D|      + λ                      = (1 - λ) + λ                     = 1
//! Abs. disc.: Σ_V max(tf - δ, 0) / |D|    + δ · |D|_unique / |D|   = (|D| - δ·|D|_u)/|D| + δ·|D|_u/|D| = 1
//! ```
//!
//! The absolute-discounting line is the interesting one, and it is *why* the
//! escape mass has the form `δ · |D|_unique / |D|`. Discounting removes `δ`
//! from each of the `|D|_unique` **distinct** terms the document contains, so
//! the total mass removed is `δ · |D|_unique` counts, i.e. a probability of
//! `δ · |D|_unique / |D|`. Handing exactly that much back to the collection
//! model — no more, no less — is what makes the columns cancel. The
//! cancellation requires `max(tf - δ, 0) = tf - δ` for every observed term,
//! which holds iff `δ ≤ 1` (every observed term has `tf ≥ 1`); for `δ > 1` a
//! singleton term would be truncated at zero, less mass would actually be
//! removed than is handed out, and the "distribution" would sum to more than
//! one. That is the whole reason [`LmSmoothing::AbsoluteDiscounting`] rejects
//! `δ > 1`.
//!
//! **It makes `RM1` cheap and exact.** Because `K_D` is `w`-free, the
//! relevance model `Σ_D P(D | Q) · P(w | D)` splits into a sparse part
//! (supported only on terms the feedback documents actually contain) plus a
//! single scalar multiple of the collection model. See [`super::rm3`].
//!
//! # Out-of-vocabulary terms
//!
//! A query term with `cf(w) = 0` — one the collection has never seen — has
//! `P(w | C) = 0`, and the whole edifice above gives it `P(w | D) = 0` and
//! `log P(w | D) = -∞`. Smoothing against the collection model cannot help,
//! because the collection model itself has no mass to lend.
//!
//! [`LmCollectionModel::probability`] therefore floors such a term at
//! `1 / (|C| + 1)`. The choice is not arbitrary: it is strictly smaller than
//! `1 / |C|`, the probability of the rarest term the collection *could* have
//! produced (one that occurs exactly once), so an unseen term is always
//! treated as rarer than any seen term — while remaining strictly positive, so
//! the logarithm is finite. The collection distribution over the known
//! vocabulary still sums to exactly `1`; the floor is escape mass for an open
//! vocabulary, and it is not renormalized into it.
//!
//! (The degenerate case `|C| = 0` — a collection consisting solely of empty
//! documents — floors every term at `1 / 1 = 1`, giving every document a
//! log-probability of `0` and a total score of `0`. Every document ties. That
//! is meaningless, but it is meaningless *finitely*, which is the contract
//! this module owes its caller.)

use std::collections::HashMap;

use super::types::{LM_MIN_PROBABILITY, LmRetrievalError, LmSmoothing};

// ── LmCollectionModel ────────────────────────────────────────────────────────

/// The collection language model `P(w | C) = cf(w) / |C|`, with an explicit
/// floor for out-of-vocabulary terms.
///
/// This is a thin, borrowing view over an index's collection statistics: it
/// owns nothing and exists so the smoothing math can be exercised (and
/// hand-verified) without an index around it.
#[derive(Debug, Clone, Copy)]
pub struct LmCollectionModel<'a> {
    /// `cf(w)` for every term in the vocabulary.
    collection_frequencies: &'a HashMap<String, u64>,
    /// `|C|`, the total number of tokens in the collection.
    total_tokens: u64,
}

impl<'a> LmCollectionModel<'a> {
    /// Wrap a collection-frequency table and its token total.
    #[must_use]
    pub fn new(collection_frequencies: &'a HashMap<String, u64>, total_tokens: u64) -> Self {
        Self {
            collection_frequencies,
            total_tokens,
        }
    }

    /// `|C|` — the total token count of the collection.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// `cf(w)` — `w`'s total number of occurrences across the collection.
    #[must_use]
    pub fn collection_frequency(&self, term: &str) -> u64 {
        self.collection_frequencies.get(term).copied().unwrap_or(0)
    }

    /// The probability the collection model assigns to an out-of-vocabulary
    /// term: `1 / (|C| + 1)`.
    ///
    /// Strictly below `1 / |C|` (the probability of a term occurring exactly
    /// once) and strictly positive. See the module documentation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn oov_probability(&self) -> f64 {
        1.0 / (self.total_tokens as f64 + 1.0)
    }

    /// `P(w | C)`, floored at [`Self::oov_probability`] for terms the
    /// collection has never seen.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn probability(&self, term: &str) -> f64 {
        let cf = self.collection_frequency(term);
        if cf == 0 || self.total_tokens == 0 {
            self.oov_probability()
        } else {
            cf as f64 / self.total_tokens as f64
        }
    }
}

// ── LmSmoothingComponents ────────────────────────────────────────────────────

/// The `(S_D(w), K_D)` decomposition of a smoothed document language model:
/// `P(w | D) = foreground + background_coefficient · P(w | C)`.
///
/// See the module documentation for the table of the three schemes, and
/// [`super::rm3`] for why splitting the probability this way is what makes the
/// relevance model computable in closed form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmSmoothingComponents {
    /// `S_D(w)` — the document's own contribution. Exactly `0` when
    /// `tf(w, D) = 0`, and (for absolute discounting) also when
    /// `tf(w, D) ≤ δ`.
    pub foreground: f64,
    /// `K_D` — the coefficient multiplying `P(w | C)`. Independent of `w`.
    pub background_coefficient: f64,
}

impl LmSmoothingComponents {
    /// Reassemble `P(w | D) = S_D(w) + K_D · P(w | C)`, *unclamped*.
    ///
    /// This is the raw value the theory prescribes. It can be exactly zero at
    /// the degenerate parameter settings (`μ = 0`, `λ = 0`, `δ = 0` — all of
    /// which reduce the scheme to the unsmoothed maximum-likelihood estimate)
    /// when the term is absent from the document. Use
    /// [`LmSmoothing::document_probability`] for the clamped value that is
    /// safe to take a logarithm of.
    #[must_use]
    pub fn probability(&self, collection_probability: f64) -> f64 {
        self.background_coefficient
            .mul_add(collection_probability, self.foreground)
    }
}

// ── LmSmoothing ──────────────────────────────────────────────────────────────

impl LmSmoothing {
    /// Validate the scheme's hyper-parameter.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] when `μ` is negative or
    /// non-finite, when `λ` is outside `[0, 1]`, or when `δ` is outside
    /// `[0, 1]` (see the module documentation for why `δ > 1` breaks
    /// normalization).
    pub fn validate(&self) -> Result<(), LmRetrievalError> {
        match *self {
            Self::Dirichlet { mu } => {
                if !mu.is_finite() || mu < 0.0 {
                    return Err(LmRetrievalError::InvalidParameter {
                        parameter: "mu".to_string(),
                        value: mu,
                        reason: "the Dirichlet prior mass must be finite and non-negative"
                            .to_string(),
                    });
                }
            }
            Self::JelinekMercer { lambda } => {
                if !lambda.is_finite() || !(0.0..=1.0).contains(&lambda) {
                    return Err(LmRetrievalError::InvalidParameter {
                        parameter: "lambda".to_string(),
                        value: lambda,
                        reason:
                            "the Jelinek-Mercer interpolation weight must be a finite value in [0, 1]"
                                .to_string(),
                    });
                }
            }
            Self::AbsoluteDiscounting { delta } => {
                if !delta.is_finite() || !(0.0..=1.0).contains(&delta) {
                    return Err(LmRetrievalError::InvalidParameter {
                        parameter: "delta".to_string(),
                        value: delta,
                        reason: "the absolute-discounting parameter must be a finite value in \
                                 [0, 1]; delta > 1 would truncate singleton terms to zero and \
                                 hand out more escape mass than it removed, so P(.|D) would no \
                                 longer sum to one"
                            .to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Decompose `P(w | D)` into its foreground and background parts.
    ///
    /// `doc_length` is `|D|` and `unique_terms` is `|D|_unique`, the number of
    /// *distinct* terms in `D` (used only by absolute discounting).
    ///
    /// # Empty documents
    ///
    /// A document of length zero has no maximum-likelihood estimate at all
    /// (`tf / |D|` is `0 / 0`), so all three schemes fall back to the
    /// collection model: `S_D(w) = 0` and `K_D = 1`, giving
    /// `P(w | D) = P(w | C)`. This is the right answer as well as the safe
    /// one — an empty document is exactly as informative as no document, and
    /// the prior is all that remains. It also keeps the normalization identity
    /// `Σ_V S_D(w) + K_D = 1` intact, which `RM1` depends on.
    ///
    /// Dirichlet's `|D| + μ = 0` (empty document *and* `μ = 0`) is covered by
    /// the same branch.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn components(
        &self,
        term_frequency: u64,
        doc_length: u64,
        unique_terms: u64,
    ) -> LmSmoothingComponents {
        let tf = term_frequency as f64;
        let dl = doc_length as f64;
        let unique = unique_terms as f64;

        match *self {
            Self::Dirichlet { mu } => {
                let denominator = dl + mu;
                if denominator > 0.0 {
                    LmSmoothingComponents {
                        foreground: tf / denominator,
                        background_coefficient: mu / denominator,
                    }
                } else {
                    // |D| = 0 and mu = 0: no evidence and no prior. Back off
                    // entirely to the collection model.
                    LmSmoothingComponents {
                        foreground: 0.0,
                        background_coefficient: 1.0,
                    }
                }
            }
            Self::JelinekMercer { lambda } => {
                if dl > 0.0 {
                    LmSmoothingComponents {
                        foreground: (1.0 - lambda) * tf / dl,
                        background_coefficient: lambda,
                    }
                } else {
                    LmSmoothingComponents {
                        foreground: 0.0,
                        background_coefficient: 1.0,
                    }
                }
            }
            Self::AbsoluteDiscounting { delta } => {
                if dl > 0.0 {
                    LmSmoothingComponents {
                        foreground: (tf - delta).max(0.0) / dl,
                        background_coefficient: delta * unique / dl,
                    }
                } else {
                    LmSmoothingComponents {
                        foreground: 0.0,
                        background_coefficient: 1.0,
                    }
                }
            }
        }
    }

    /// `P(w | D)`, clamped at [`LM_MIN_PROBABILITY`] so that its logarithm is
    /// always finite.
    ///
    /// The clamp binds only at the degenerate settings `μ = 0` / `λ = 0` /
    /// `δ = 0`, where the scheme collapses to the unsmoothed
    /// maximum-likelihood estimate and genuinely assigns zero probability to
    /// unseen terms. Every well-posed configuration produces a probability
    /// many orders of magnitude above the clamp, which is therefore inert.
    #[must_use]
    pub fn document_probability(
        &self,
        term_frequency: u64,
        doc_length: u64,
        unique_terms: u64,
        collection_probability: f64,
    ) -> f64 {
        self.components(term_frequency, doc_length, unique_terms)
            .probability(collection_probability)
            .max(LM_MIN_PROBABILITY)
    }

    /// `log P(w | D)` — natural log, always finite.
    ///
    /// The base of the logarithm rescales every document's score by the same
    /// positive constant and so cannot change the ranking; the natural log is
    /// used because it is what the relevance model of [`super::rm3`] needs
    /// (`P(D | Q) ∝ exp(score)` is only meaningful for `exp`/`ln`).
    #[must_use]
    pub fn log_document_probability(
        &self,
        term_frequency: u64,
        doc_length: u64,
        unique_terms: u64,
        collection_probability: f64,
    ) -> f64 {
        self.document_probability(
            term_frequency,
            doc_length,
            unique_terms,
            collection_probability,
        )
        .ln()
    }
}
