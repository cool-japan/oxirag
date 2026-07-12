//! Configuration, inputs, outputs, statistics and errors for the `REPLUG`
//! ensemble and its `LSR` retriever-feedback signal.
//!
//! Two conventions run through this file and are worth stating once:
//!
//! - **Distributions are `f64`.** Logits arrive from a model as `f32` (see
//!   [`super::model::ReplugLanguageModel`]) because that is what the model
//!   natively produces; but a *probability distribution* that we are going to
//!   assert sums to `1`, take logarithms of, and difference into a `KL` is
//!   carried in `f64`. The rationale is in [`super::math`].
//! - **Log-probabilities are first-class.** Every distribution this module
//!   hands back carries its log-space form alongside (or instead of) the
//!   probability-space form, so a caller never has to call `.ln()` on a
//!   probability that may have underflowed.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Everything that can go wrong in a `REPLUG` ensemble or `LSR` update.
///
/// Note the absence of any "recovered by falling back to a uniform
/// distribution" case: a `REPLUG` ensemble over zero documents, or over a model
/// that emitted a `NaN` logit, has no defensible answer, and inventing one
/// would produce a plausible-looking distribution that means nothing.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ReplugError {
    /// The ensemble was asked to run over an empty document set.
    ///
    /// `REPLUG` is *defined* as a mixture over retrieved documents; with none
    /// retrieved there is no mixture, and falling back to the bare
    /// (unconditioned) language model would silently change the semantics of
    /// the call from "retrieval-augmented" to "not".
    #[error("REPLUG ensemble requires at least one retrieved document, got zero")]
    NoDocuments,

    /// The language model returned an empty logit vector, or a distribution
    /// over an empty vocabulary was requested.
    #[error("empty logit vector: a next-token distribution needs a non-empty vocabulary")]
    EmptyLogits,

    /// The language model emitted a `NaN` or infinite logit.
    #[error("non-finite logit at index {index}: {value}")]
    NonFiniteLogit {
        /// Position of the offending logit in the vocabulary.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// Per-document distributions disagreed on the vocabulary size.
    #[error(
        "vocabulary size mismatch: expected {expected}, got {actual} \
         (document index {document_index})"
    )]
    VocabSizeMismatch {
        /// The vocabulary size established by the model or the first document.
        expected: usize,
        /// The size actually returned.
        actual: usize,
        /// Which document produced the mismatched vector.
        document_index: usize,
    },

    /// The number of mixture weights did not match the number of documents.
    #[error("weight/document count mismatch: {weights} weights for {documents} documents")]
    WeightCountMismatch {
        /// How many weights were supplied.
        weights: usize,
        /// How many documents were supplied.
        documents: usize,
    },

    /// A temperature was zero, negative, or `NaN`.
    ///
    /// `τ = 0` is *not* silently promoted to "arg-max": `softmax(x / 0)`
    /// evaluates `0 / 0 = NaN` at the maximizing element. The arg-max limit is
    /// reached by a small positive `τ`, which
    /// [`super::math::temperature_log_softmax`] handles exactly and without
    /// overflow.
    #[error("temperature must be finite and strictly positive, got {temperature}")]
    InvalidTemperature {
        /// The rejected value.
        temperature: f64,
    },

    /// A token id fell outside the model's vocabulary.
    #[error("token id {token_id} is outside the vocabulary of size {vocab_size}")]
    TokenOutOfRange {
        /// The offending token id.
        token_id: usize,
        /// The model's reported vocabulary size.
        vocab_size: usize,
    },

    /// `LSR` was asked to score an empty ground-truth continuation.
    ///
    /// `Q_LM(d | q) = softmax(log P_LM(y* | d ⊕ q) / β)` is meaningless when
    /// `y*` is empty: every document scores `log P = 0` and the "LM preference"
    /// is uniform by construction, which would produce a confident-looking but
    /// entirely content-free gradient.
    #[error("LSR requires a non-empty ground-truth continuation y*")]
    EmptyTarget,

    /// A configuration value was outside its valid range.
    #[error("invalid REPLUG configuration: {reason}")]
    InvalidConfig {
        /// Why the configuration was rejected.
        reason: String,
    },

    /// The underlying language model failed.
    #[error("language model error: {message}")]
    Model {
        /// The model's own error message.
        message: String,
    },
}

/// Convenience alias for results in this module.
pub type ReplugResult<T> = Result<T, ReplugError>;

// ── Decoding strategy ────────────────────────────────────────────────────────

/// How the next token is picked from the **ensembled** distribution.
///
/// Both strategies act on `p(y | q) = Σ_i λ_i p(y | d_i ⊕ q)` — the mixture —
/// and never on any individual document's distribution. That is the whole
/// point of the architecture: the documents vote, and the decoder reads only
/// the tally.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ReplugDecoding {
    /// Take `arg max_y p(y | q)`. Deterministic; ties go to the lowest token id.
    #[default]
    Greedy,
    /// Sample from the mixture, optionally re-tempered.
    ///
    /// The sampling temperature `T` is applied to the **mixture**, not to any
    /// set of logits — because after mixing there *are* no logits, only a
    /// distribution. Re-tempering a distribution means
    /// `p_T(y) ∝ p(y)^{1/T}`, which this module performs in log-space as
    /// `log_softmax(log p / T)`. `T = 1` is exact sampling from the mixture;
    /// `T → 0` degenerates to greedy; `T > 1` flattens.
    Sampling {
        /// The re-tempering exponent `T` (must be finite and strictly positive).
        temperature: f64,
        /// `SplitMix64` seed, so that a "sampled" run is still reproducible.
        seed: u64,
    },
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Tunables for the `REPLUG` ensemble and the `REPLUG`-`LSR` update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugConfig {
    /// **τ** — the retrieval-likelihood temperature in
    /// `λ(d_i | q) = softmax(s_i / τ)`.
    ///
    /// Controls how sharply the ensemble concentrates on its best-retrieved
    /// document. `τ → 0⁺` puts all mass on the top-scored document (the
    /// ensemble degenerates to single-document RAG); `τ → ∞` weights every
    /// retrieved document equally (the ensemble ignores the retriever's
    /// ranking). Must be finite and strictly positive.
    ///
    /// The right scale for `τ` depends on the scale of your retrieval scores:
    /// cosine similarities live in `[-1, 1]` and want a small `τ` (~`0.1`) for
    /// the softmax to discriminate at all, whereas raw inner products in the
    /// tens want a `τ` of comparable magnitude.
    pub temperature: f64,

    /// **β** — the LM-likelihood temperature in
    /// `Q_LM(d | q) = softmax(log P_LM(y* | d ⊕ q) / β)`.
    ///
    /// Controls how sharply the *LM's* preference over documents is expressed
    /// before it is distilled into the retriever. Must be finite and strictly
    /// positive. Because its argument is a **log**-likelihood (a large negative
    /// number whose magnitude grows with `|y*|`), `β` is the knob that puts
    /// that quantity on a scale where a softmax is informative rather than
    /// saturated — see [`ReplugConfig::lsr_length_normalize`].
    pub lsr_beta: f64,

    /// How many retrieved documents to actually ensemble, keeping the
    /// highest-scoring ones.
    ///
    /// Truncation happens **before** λ is computed, so λ is a softmax over the
    /// *retained* scores and sums to `1` over exactly the documents that
    /// participate in the mixture. `None` keeps everything.
    pub top_k_documents: Option<usize>,

    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Greedy or sampled decoding from the mixture.
    pub decoding: ReplugDecoding,

    /// Whether to divide the sequence log-likelihood by `|y*|` before the `β`
    /// softmax in `LSR`.
    ///
    /// Off by default, which is faithful to the paper's
    /// `Q_LM = softmax(log P_LM(y* | d ⊕ q) / β)`. Turn it on when your `y*`
    /// lengths vary a lot: the raw sequence log-likelihood scales roughly
    /// linearly with `|y*|`, so a fixed `β` that is well-calibrated for a
    /// 5-token continuation will saturate `Q_LM` into a one-hot for a 50-token
    /// one (the *differences* between documents grow with length too). Dividing
    /// by `|y*|` turns the argument into a mean per-token log-likelihood, whose
    /// scale is length-invariant, making a single `β` transferable across
    /// examples.
    pub lsr_length_normalize: bool,

    /// Step size for the `LSR` retrieval-score update
    /// `s_i ← s_i − η · ∂KL/∂s_i`.
    pub lsr_learning_rate: f64,

    /// Text placed between a document and the query when building the
    /// per-document context `d_i ⊕ q`.
    pub document_separator: String,

    /// Whether [`ReplugEnsembleOutput`] retains the full ensembled distribution
    /// for every decoding step.
    ///
    /// On by default because these distributions *are* the object of study for
    /// this module. Be aware of the cost before turning `max_tokens` up on a
    /// real model: a 128k vocabulary over 512 steps is ~512 MB of `f64` per
    /// call. Set to `false` to keep only the chosen token and its log-
    /// probability.
    pub record_distributions: bool,
}

impl Default for ReplugConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            lsr_beta: 1.0,
            top_k_documents: Some(10),
            max_tokens: 32,
            decoding: ReplugDecoding::Greedy,
            lsr_length_normalize: false,
            lsr_learning_rate: 0.1,
            document_separator: "\n\n".to_string(),
            record_distributions: true,
        }
    }
}

impl ReplugConfig {
    /// Reject a configuration that cannot produce a meaningful distribution.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::InvalidTemperature`] for a non-positive or
    /// non-finite `τ`, `β`, or sampling temperature, and
    /// [`ReplugError::InvalidConfig`] for a zero `top_k_documents`, a zero
    /// `max_tokens`, or a non-finite learning rate.
    pub fn validate(&self) -> ReplugResult<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(ReplugError::InvalidTemperature {
                temperature: self.temperature,
            });
        }
        if !self.lsr_beta.is_finite() || self.lsr_beta <= 0.0 {
            return Err(ReplugError::InvalidTemperature {
                temperature: self.lsr_beta,
            });
        }
        if let ReplugDecoding::Sampling { temperature, .. } = self.decoding
            && (!temperature.is_finite() || temperature <= 0.0)
        {
            return Err(ReplugError::InvalidTemperature { temperature });
        }
        if self.top_k_documents == Some(0) {
            return Err(ReplugError::InvalidConfig {
                reason: "top_k_documents must be at least 1 when set".to_string(),
            });
        }
        if self.max_tokens == 0 {
            return Err(ReplugError::InvalidConfig {
                reason: "max_tokens must be at least 1".to_string(),
            });
        }
        if !self.lsr_learning_rate.is_finite() {
            return Err(ReplugError::InvalidConfig {
                reason: format!(
                    "lsr_learning_rate must be finite, got {}",
                    self.lsr_learning_rate
                ),
            });
        }
        Ok(())
    }

    /// Set `τ`, the retrieval-likelihood temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set `β`, the LM-likelihood temperature used by `LSR`.
    #[must_use]
    pub fn with_lsr_beta(mut self, lsr_beta: f64) -> Self {
        self.lsr_beta = lsr_beta;
        self
    }

    /// Set how many top-scored documents participate in the mixture.
    #[must_use]
    pub fn with_top_k_documents(mut self, top_k: Option<usize>) -> Self {
        self.top_k_documents = top_k;
        self
    }

    /// Set the generation length cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the decoding strategy.
    #[must_use]
    pub fn with_decoding(mut self, decoding: ReplugDecoding) -> Self {
        self.decoding = decoding;
        self
    }

    /// Set the `LSR` score-update step size.
    #[must_use]
    pub fn with_lsr_learning_rate(mut self, learning_rate: f64) -> Self {
        self.lsr_learning_rate = learning_rate;
        self
    }
}

// ── Documents ────────────────────────────────────────────────────────────────

/// One retrieved document, with the retrieval score that determines its share
/// of the mixture.
///
/// The `score` is whatever your retriever produced — a cosine similarity, an
/// inner product, a `BM25` score. It is never interpreted as a probability;
/// it is fed through `softmax(· / τ)` precisely *because* it is an
/// uncalibrated real number, and `τ` is the knob that adapts its scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugDocument {
    /// Stable identifier, used to key the `LSR` gradients and the reranking.
    pub id: String,
    /// The document text, prepended to the query to form `d ⊕ q`.
    pub text: String,
    /// The retriever's similarity score `s(q, d)`.
    pub score: f64,
}

impl ReplugDocument {
    /// Create a retrieved document.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>, score: f64) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            score,
        }
    }
}

// ── Per-step output ──────────────────────────────────────────────────────────

/// One decoding step of the ensemble.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugStep {
    /// 0-based index of this decoding step.
    pub step: usize,
    /// The ensembled distribution `p(y | q) = Σ_i λ_i p(y | d_i ⊕ q)`, present
    /// only when [`ReplugConfig::record_distributions`] is set.
    ///
    /// Sums to `1` up to rounding — the module's tests assert this.
    pub distribution: Option<Vec<f64>>,
    /// The same distribution in log-space, `log p(y | q)`, present under the
    /// same condition. This is the *primary* representation: it is finite even
    /// where `distribution` has underflowed to `0.0`.
    pub log_distribution: Option<Vec<f64>>,
    /// The token this step emitted.
    pub token_id: usize,
    /// `log p(token_id | q)` under the **ensembled** distribution.
    pub token_log_prob: f64,
    /// Entropy of the ensembled distribution, in nats. A useful diagnostic: it
    /// rises when the retrieved documents disagree about the next token.
    pub entropy_nats: f64,
}

/// Per-run statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugStats {
    /// Documents supplied to the ensemble.
    pub documents_supplied: usize,
    /// Documents that actually participated, after `top_k_documents`.
    pub documents_retained: usize,
    /// The model's vocabulary size.
    pub vocab_size: usize,
    /// How many tokens were generated.
    pub generated_tokens: usize,
    /// Forward passes performed. Exactly `documents_retained × steps` — the
    /// cost `REPLUG` pays for its ensemble, and worth surfacing rather than
    /// hiding.
    pub lm_calls: usize,
    /// `Σ_t log p(y_t | q)` under the ensembled distributions.
    pub sequence_log_prob: f64,
    /// Mean per-token log-probability, i.e. `sequence_log_prob / generated_tokens`.
    pub mean_token_log_prob: f64,
    /// Mean entropy (nats) of the ensembled distributions across steps.
    pub mean_entropy_nats: f64,
    /// Entropy (nats) of the document-weight distribution λ. `0` means the
    /// ensemble collapsed onto one document; `ln(k)` means it weighted all `k`
    /// equally.
    pub weight_entropy_nats: f64,
    /// The largest λ.
    pub max_doc_weight: f64,
    /// The smallest λ.
    pub min_doc_weight: f64,
}

/// The full result of a `REPLUG` generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugEnsembleOutput {
    /// The decoded text.
    pub text: String,
    /// The generated token ids.
    pub token_ids: Vec<usize>,
    /// Per-step record, including the ensembled distributions.
    pub steps: Vec<ReplugStep>,
    /// Ids of the documents that participated, in the order their weights are
    /// listed.
    pub document_ids: Vec<String>,
    /// The mixture weights `λ(d_i | q) = softmax(s_i / τ)`, aligned with
    /// `document_ids`. Computed once from the query and the retrieval scores,
    /// and held fixed for every decoding step — λ conditions on `q`, not on the
    /// partial generation.
    pub document_weights: Vec<f64>,
    /// Statistics for the run.
    pub stats: ReplugStats,
}

// ── LSR output ───────────────────────────────────────────────────────────────

/// The per-document learning signal produced by `REPLUG`-`LSR`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugDocumentGradient {
    /// Which document this row is about.
    pub document_id: String,
    /// `log P_LM(y* | d ⊕ q)` — how much this document actually *helped* the
    /// frozen LM assign probability to the ground truth. Length-normalized when
    /// [`ReplugConfig::lsr_length_normalize`] is set.
    pub target_log_likelihood: f64,
    /// `P_R(d | q) = softmax(s / τ)` — what the retriever currently believes.
    pub retrieval_prob: f64,
    /// `Q_LM(d | q) = softmax(log P_LM(y* | d ⊕ q) / β)` — what the LM's
    /// behaviour implies it *should* believe.
    pub lm_prob: f64,
    /// `∂ KL(Q_LM ‖ P_R) / ∂ s = (P_R − Q_LM) / τ`.
    ///
    /// Read the sign: when the LM liked this document more than the retriever
    /// did (`Q > P`) the gradient is **negative**, and a descent step
    /// `s ← s − η · grad` therefore **raises** the document's score. That is
    /// the entire mechanism by which the LM teaches the retriever.
    pub score_gradient: f64,
    /// The retrieval score before the update.
    pub original_score: f64,
    /// The retrieval score after one descent step.
    pub updated_score: f64,
}

/// The result of one `REPLUG`-`LSR` feedback computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplugLsrSignal {
    /// `KL(Q_LM ‖ P_R) ≥ 0`, the quantity being minimized. Zero exactly when
    /// the retriever's ranking already matches the LM's preference.
    pub kl_loss: f64,
    /// Per-document breakdown, in the order the documents were supplied.
    pub gradients: Vec<ReplugDocumentGradient>,
}

impl ReplugLsrSignal {
    /// The document ids re-ranked by the **updated** scores, best first.
    ///
    /// This is where the `LSR` loss becomes visible as behaviour: if the LM
    /// found document `B` more useful than the retriever's top-ranked `A`, this
    /// ordering is where `B` overtakes it. Ties break by ascending id, so the
    /// ranking is deterministic.
    #[must_use]
    pub fn reranked_document_ids(&self) -> Vec<String> {
        let mut rows: Vec<&ReplugDocumentGradient> = self.gradients.iter().collect();
        rows.sort_by(|left, right| {
            right
                .updated_score
                .total_cmp(&left.updated_score)
                .then_with(|| left.document_id.cmp(&right.document_id))
        });
        rows.into_iter()
            .map(|row| row.document_id.clone())
            .collect()
    }

    /// The updated retrieval scores, in the order the documents were supplied.
    #[must_use]
    pub fn updated_scores(&self) -> Vec<f64> {
        self.gradients.iter().map(|row| row.updated_score).collect()
    }
}
