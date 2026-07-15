//! Configuration, errors, per-step records and outputs for decode-time
//! distribution contrast.
//!
//! Two conventions run through this file, both inherited from
//! [`crate::replug::types`] and both load-bearing:
//!
//! - **Distributions are `f64`, and they are carried in log-space.** A model
//!   emits `f32` logits — that is what its final projection natively produces —
//!   but the arithmetic this module performs on them is a *difference of two
//!   logarithms*, which is catastrophically cancellation-prone. Contrast is
//!   subtraction; subtraction in `f32` throws away exactly the precision the
//!   method depends on.
//! - **Nothing is silently repaired.** There is no "fell back to the unmodified
//!   distribution" case in [`ContextAwareError`]. A contrast against a model
//!   that emitted a `NaN`, or against a layer that does not exist, has no
//!   defensible answer, and inventing one would produce a plausible-looking
//!   distribution that means nothing.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::replug::ReplugError;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Everything that can go wrong while contrasting two next-token distributions.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ContextAwareError {
    /// A score vector was empty, or the vocabulary is empty.
    #[error("empty score vector: a next-token distribution needs a non-empty vocabulary")]
    EmptyLogits,

    /// A score vector contained a `NaN` or an infinity where a finite value was
    /// required.
    ///
    /// This is checked, once, at the trait boundary, and the reason it is
    /// checked *at all* is specific to contrast: the combination
    /// `w⁺ · s⁺ + w⁻ · s⁻` multiplies a score by a possibly-**negative** weight.
    /// `-0.0 · -inf` is `NaN` and `-1.0 · -inf` is `+inf` — an infinitely
    /// confident vote for the one token the negative model considered
    /// impossible. Neither is a distribution, and neither would be visible
    /// downstream. A convex mixture, whose weights are all non-negative, never
    /// has to worry about this.
    #[error("non-finite score at index {index} of the {origin} distribution: {value}")]
    NonFiniteScore {
        /// Position of the offending score in the vocabulary.
        index: usize,
        /// Which of the two distributions it came from.
        origin: String,
        /// The offending value.
        value: f64,
    },

    /// Two distributions disagreed about the size of the vocabulary.
    #[error("vocabulary size mismatch: expected {expected}, got {actual} (from the {origin})")]
    VocabSizeMismatch {
        /// The size established by the model or by the first vector.
        expected: usize,
        /// The size actually seen.
        actual: usize,
        /// Which input produced the mismatched vector.
        origin: String,
    },

    /// A vector handed to a divergence was not a normalized log-probability
    /// vector.
    ///
    /// Reported by [`super::math::jensen_shannon_divergence_from_log_probs`],
    /// and it exists to catch one specific, silent, extremely easy mistake:
    /// passing **raw logits** to a divergence. A Jensen–Shannon divergence is
    /// defined between *distributions*; feed it logits and it will happily
    /// return a finite, plausible-looking number that is not the divergence of
    /// anything. The check is one `logsumexp`, and it turns that into a loud
    /// error.
    #[error(
        "not a normalized log-probability vector: log Σ exp(·) = {log_total_mass}, expected 0 \
         (did you pass raw logits to a divergence instead of `log_softmax` output?)"
    )]
    UnnormalizedDistribution {
        /// `log Σ_y exp(v_y)`, which is `0` for a normalized distribution.
        log_total_mass: f64,
    },

    /// The plausibility threshold `alpha` was outside `[0, 1]`, or `NaN`.
    ///
    /// `alpha > 1` would make the constraint `p(y) ≥ alpha · max p` unsatisfiable
    /// even for the arg-max, i.e. it could empty the vocabulary and leave nothing
    /// to decode. `alpha < 0` is not a mass fraction. Both are rejected rather
    /// than clamped.
    #[error("plausibility alpha must lie in [0, 1], got {alpha}")]
    InvalidPlausibilityAlpha {
        /// The rejected value.
        alpha: f64,
    },

    /// A layer index was outside the model's stack.
    #[error("layer {layer} is outside the model's {num_layers}-layer stack")]
    LayerOutOfRange {
        /// The offending layer index.
        layer: usize,
        /// The model's reported depth.
        num_layers: usize,
    },

    /// The mature layer appeared in the premature-layer candidate set.
    ///
    /// Contrasting a layer against **itself** yields the all-zero score vector,
    /// whose softmax is the *uniform* distribution over the plausible set — a
    /// catastrophic and completely silent failure. `DoLa` excludes the final
    /// layer from the candidate bucket for exactly this reason, and so does this
    /// module, loudly.
    #[error(
        "the mature layer {layer} cannot also be a premature candidate: contrasting a layer \
         against itself yields the uniform distribution"
    )]
    MatureLayerInCandidates {
        /// The layer that appeared on both sides.
        layer: usize,
    },

    /// There was no premature layer to select from.
    #[error(
        "no premature-layer candidates: `DoLa` needs a model with at least two layers, and at \
         least one candidate that is not the mature layer"
    )]
    EmptyCandidateSet,

    /// A token id fell outside the model's vocabulary.
    #[error("token id {token_id} is outside the vocabulary of size {vocab_size}")]
    TokenOutOfRange {
        /// The offending token id.
        token_id: usize,
        /// The model's reported vocabulary size.
        vocab_size: usize,
    },

    /// A configuration value was outside its valid range.
    #[error("invalid context-aware decoding configuration: {reason}")]
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

    /// A shared log-space primitive from [`crate::replug::math`] rejected its
    /// input.
    ///
    /// This module reuses `REPLUG`'s numerically-stable `f64` log-space kernels
    /// rather than growing a second, subtly-different copy of them, so their
    /// errors surface here unchanged.
    #[error("log-space primitive failed: {0}")]
    Math(#[from] ReplugError),
}

/// Convenience alias for results in this module.
pub type ContextAwareResult<T> = Result<T, ContextAwareError>;

// ── Which contrast ───────────────────────────────────────────────────────────

/// Which of the three contrast methods produced a result.
///
/// The three differ **only** in where the two distributions come from. The
/// combination rule, the plausibility constraint and the renormalization are
/// literally the same code:
///
/// | Mode | Positive distribution | Negative distribution |
/// |---|---|---|
/// | [`DecodeMode::ContextAware`] | `p(y | c ⊕ x)` — one model, **with** the retrieved context | `p(y | x)` — the *same* model, **without** it |
/// | [`DecodeMode::Contrastive`] | `p_expert(y | x)` — the large model | `p_amateur(y | x)` — a *smaller* model, same context |
/// | [`DecodeMode::Dola`] | `q_mature(y | x)` — the **final** layer | `q_premature(y | x)` — an **earlier** layer of the same forward pass |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodeMode {
    /// Context-aware decoding (Shi et al., 2024): contrast one model against
    /// itself, with and without the retrieved context.
    ContextAware,
    /// Contrastive decoding (Li et al., 2023): contrast an expert model against
    /// an amateur one.
    Contrastive,
    /// `DoLa` (Chuang et al., 2024): contrast a mature layer against a
    /// Jensen–Shannon-selected premature layer.
    Dola,
}

// ── Decoding strategy ────────────────────────────────────────────────────────

/// How the next token is picked from the **contrasted** distribution.
///
/// Both strategies act on the contrasted, plausibility-constrained,
/// renormalized distribution, never on either of the two inputs to the contrast.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum DecodingStrategy {
    /// Take `arg max_y p(y)`. Deterministic; ties go to the lowest token id.
    #[default]
    Greedy,
    /// Sample from the contrasted distribution, optionally re-tempered.
    ///
    /// Tokens excluded by the plausibility constraint carry `log p = -inf` and
    /// are therefore unreachable — the constraint is a *hard* one, and
    /// re-tempering cannot resurrect what it removed.
    Sampling {
        /// The re-tempering exponent `T`: `p_T(y) ∝ p(y)^{1/T}`. Must be finite
        /// and strictly positive. `T = 1` samples from the contrasted
        /// distribution exactly; `T → 0` degenerates to greedy.
        temperature: f64,
        /// `SplitMix64` seed, so a sampled run is still reproducible.
        seed: u64,
    },
}

// ── Shared config ────────────────────────────────────────────────────────────

/// The decode-time settings every contrast method carries.
///
/// Embedded as the `shared` field of [`CadConfig`], [`ContrastiveConfig`] and
/// [`DolaConfig`], because the plausibility constraint, the token-selection rule
/// and the generation cap are properties of *contrast itself*, not of any one
/// method's choice of the two distributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAwareConfig {
    /// **`alpha_plaus`** — the adaptive plausibility constraint.
    ///
    /// Only tokens satisfying
    ///
    /// ```text
    ///   p⁺(y)  ≥  alpha_plaus · max_{y'} p⁺(y')
    /// ```
    ///
    /// under the **positive** distribution survive the contrast; the rest are
    /// set to `-inf` *before* renormalization. This is the one piece of
    /// machinery that a convex mixture (such as `REPLUG`'s) never needs and that
    /// an extrapolation cannot do without — see the module documentation.
    ///
    /// The two extremes are exact rather than approximate:
    ///
    /// - **`0.0`** — the constraint `p⁺(y) ≥ 0` holds for every token, so the
    ///   constraint is *vacuous* and the published, unconstrained formula is
    ///   reproduced bit-for-bit. This is what Shi et al.'s `CAD` actually is.
    /// - **`1.0`** — only the arg-max (and anything exactly tied with it)
    ///   survives, i.e. the contrast can reorder nothing.
    ///
    /// Defaults to `0.1`, which is Li et al.'s and Chuang et al.'s published
    /// value. Note that this default is *not* faithful to `CAD`, which publishes
    /// no constraint at all; the module defaults it on anyway, because without it
    /// the extrapolation diverges on tokens the negative model considers
    /// impossible, and the module's tests exhibit that divergence with numbers.
    /// Set it to `0.0` to recover the published `CAD`.
    pub plausibility_alpha: f64,

    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Greedy or sampled selection from the contrasted distribution.
    pub strategy: DecodingStrategy,

    /// Whether each [`ContextAwareStep`] retains the full distributions.
    ///
    /// On by default because these distributions *are* the object of study. Be
    /// aware of the cost on a real model before turning `max_tokens` up: three
    /// `f64` vectors over a 128k vocabulary, for 512 steps, is ~1.5 GB.
    pub record_distributions: bool,
}

impl Default for ContextAwareConfig {
    fn default() -> Self {
        Self {
            plausibility_alpha: 0.1,
            max_tokens: 32,
            strategy: DecodingStrategy::Greedy,
            record_distributions: true,
        }
    }
}

impl ContextAwareConfig {
    /// Reject a configuration that cannot produce a meaningful distribution.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidPlausibilityAlpha`] when
    /// `plausibility_alpha` is outside `[0, 1]` or `NaN`, and
    /// [`ContextAwareError::InvalidConfig`] for a zero `max_tokens` or a
    /// non-positive sampling temperature.
    pub fn validate(&self) -> ContextAwareResult<()> {
        // `NaN` fails every comparison, so `contains` rejects it for free.
        if !(0.0..=1.0).contains(&self.plausibility_alpha) {
            return Err(ContextAwareError::InvalidPlausibilityAlpha {
                alpha: self.plausibility_alpha,
            });
        }
        if self.max_tokens == 0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: "max_tokens must be at least 1".to_string(),
            });
        }
        if let DecodingStrategy::Sampling { temperature, .. } = self.strategy
            && (!temperature.is_finite() || temperature <= 0.0)
        {
            return Err(ContextAwareError::InvalidConfig {
                reason: format!(
                    "sampling temperature must be finite and strictly positive, got {temperature}"
                ),
            });
        }
        Ok(())
    }

    /// Set the adaptive plausibility threshold.
    #[must_use]
    pub fn with_plausibility_alpha(mut self, alpha: f64) -> Self {
        self.plausibility_alpha = alpha;
        self
    }

    /// Set the generation length cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set greedy or sampled selection.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DecodingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set whether the full distributions are retained per step.
    #[must_use]
    pub fn with_record_distributions(mut self, record: bool) -> Self {
        self.record_distributions = record;
        self
    }
}

// ── CAD ──────────────────────────────────────────────────────────────────────

/// Tunables for **context-aware decoding** (Shi et al., 2024).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CadConfig {
    /// **`alpha`** — how hard to push away from the model's context-free prior.
    ///
    /// ```text
    ///   score(y)  =  (1 + alpha) · z(y | c, x)  −  alpha · z(y | x)
    /// ```
    ///
    /// equivalently `p_cad(y) ∝ p(y | c, x) · ( p(y | c, x) / p(y | x) )^alpha`:
    /// the model's context-conditioned belief, re-weighted by how much the
    /// context *changed its mind*. Must be finite and non-negative.
    ///
    /// - `alpha = 0` is the plain context-conditioned distribution, exactly
    ///   (the module's tests pin this bit-for-bit).
    /// - `alpha > 0` amplifies the context's influence. Shi et al. report
    ///   `alpha ∈ [0.5, 1.0]` across their tasks; `1.0` is the default here.
    /// - Large `alpha` is *not* free: the factor `1 / p(y | x)^alpha` diverges as
    ///   `p(y | x) → 0`, so a token the context-free model considers impossible
    ///   is amplified without bound. That is what `plausibility_alpha` is for.
    pub alpha: f64,

    /// Text placed between the retrieved context and the query when building the
    /// **positive** prompt `c ⊕ separator ⊕ x`.
    pub context_separator: String,

    /// Plausibility constraint, generation cap, and token selection.
    pub shared: ContextAwareConfig,
}

impl Default for CadConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            context_separator: "\n\n".to_string(),
            shared: ContextAwareConfig::default(),
        }
    }
}

impl CadConfig {
    /// Reject a configuration that cannot produce a meaningful distribution.
    ///
    /// # Errors
    ///
    /// Propagates [`ContextAwareConfig::validate`], and returns
    /// [`ContextAwareError::InvalidConfig`] when `alpha` is negative or
    /// non-finite.
    pub fn validate(&self) -> ContextAwareResult<()> {
        self.shared.validate()?;
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: format!(
                    "CAD alpha must be finite and non-negative, got {}; a negative alpha would \
                     push the output *toward* the context-free prior, which is the opposite of \
                     what this method is for",
                    self.alpha
                ),
            });
        }
        Ok(())
    }

    /// The two contrast weights `(1 + alpha, −alpha)`.
    ///
    /// This pair *is* the method. Their sum is `1`, which is exactly why the
    /// operation is an **extrapolation** along the line through the two
    /// distributions rather than an interpolation between them: at `alpha > 0`
    /// the second coefficient is negative, so the result lies beyond the
    /// positive endpoint, outside the segment the two distributions span.
    #[must_use]
    pub fn contrast_weights(&self) -> (f64, f64) {
        (1.0 + self.alpha, -self.alpha)
    }

    /// Set the contrast strength `alpha`.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the text joining the context to the query.
    #[must_use]
    pub fn with_context_separator(mut self, separator: impl Into<String>) -> Self {
        self.context_separator = separator.into();
        self
    }

    /// Set the adaptive plausibility threshold.
    #[must_use]
    pub fn with_plausibility_alpha(mut self, alpha: f64) -> Self {
        self.shared.plausibility_alpha = alpha;
        self
    }

    /// Set the generation length cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.shared.max_tokens = max_tokens;
        self
    }

    /// Set greedy or sampled selection.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DecodingStrategy) -> Self {
        self.shared.strategy = strategy;
        self
    }
}

// ── Contrastive decoding ─────────────────────────────────────────────────────

/// Tunables for **contrastive decoding** (Li et al., 2023).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastiveConfig {
    /// **`beta`** — how much of the amateur's belief to subtract.
    ///
    /// ```text
    ///   score(y)  =  log p_expert(y | x)  −  beta · log p_amateur(y | x)
    /// ```
    ///
    /// `beta = 1` is Li et al.'s published objective, the plain log-ratio
    /// `log( p_expert / p_amateur )`, and is the default. Must be finite and
    /// non-negative; `beta = 0` reduces to the expert's own distribution, exactly.
    ///
    /// Note the asymmetry with [`CadConfig::alpha`]: `CAD` weights the positive
    /// distribution by `1 + alpha`, `CD` weights it by `1`. That is not an
    /// oversight — it is what the two papers write, and the two objectives are
    /// genuinely different functions, not reparameterizations of each other.
    pub amateur_weight: f64,

    /// **`tau`** — a temperature applied to the amateur *before* subtracting it.
    ///
    /// `p_amateur^{1/tau}`, computed as `log_softmax(z_amateur / tau)` without
    /// ever forming the quotient (see `crate::replug::math`). `tau > 1` flattens
    /// the amateur, which weakens what it can veto; `tau < 1` sharpens it, which
    /// concentrates the subtraction on the amateur's own favourites. Must be
    /// finite and strictly positive. `tau = 1` leaves the amateur untouched and
    /// is the default.
    pub amateur_temperature: f64,

    /// Plausibility constraint, generation cap, and token selection.
    ///
    /// Li et al.'s published `alpha` for the adaptive plausibility constraint is
    /// `0.1`, which is this module's default.
    pub shared: ContextAwareConfig,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            amateur_weight: 1.0,
            amateur_temperature: 1.0,
            shared: ContextAwareConfig::default(),
        }
    }
}

impl ContrastiveConfig {
    /// Reject a configuration that cannot produce a meaningful distribution.
    ///
    /// # Errors
    ///
    /// Propagates [`ContextAwareConfig::validate`], and returns
    /// [`ContextAwareError::InvalidConfig`] for a negative or non-finite
    /// `amateur_weight`, or a non-positive or non-finite `amateur_temperature`.
    pub fn validate(&self) -> ContextAwareResult<()> {
        self.shared.validate()?;
        if !self.amateur_weight.is_finite() || self.amateur_weight < 0.0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: format!(
                    "amateur_weight must be finite and non-negative, got {}",
                    self.amateur_weight
                ),
            });
        }
        if !self.amateur_temperature.is_finite() || self.amateur_temperature <= 0.0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: format!(
                    "amateur_temperature must be finite and strictly positive, got {}",
                    self.amateur_temperature
                ),
            });
        }
        Ok(())
    }

    /// The two contrast weights `(1, −beta)`.
    #[must_use]
    pub fn contrast_weights(&self) -> (f64, f64) {
        (1.0, -self.amateur_weight)
    }

    /// Set `beta`, the amateur's subtraction weight.
    #[must_use]
    pub fn with_amateur_weight(mut self, weight: f64) -> Self {
        self.amateur_weight = weight;
        self
    }

    /// Set `tau`, the amateur's temperature.
    #[must_use]
    pub fn with_amateur_temperature(mut self, temperature: f64) -> Self {
        self.amateur_temperature = temperature;
        self
    }

    /// Set the adaptive plausibility threshold.
    #[must_use]
    pub fn with_plausibility_alpha(mut self, alpha: f64) -> Self {
        self.shared.plausibility_alpha = alpha;
        self
    }

    /// Set the generation length cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.shared.max_tokens = max_tokens;
        self
    }

    /// Set greedy or sampled selection.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DecodingStrategy) -> Self {
        self.shared.strategy = strategy;
        self
    }
}

// ── DoLa ─────────────────────────────────────────────────────────────────────

/// How `DoLa` picks the premature layer to contrast the mature layer against.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerSelector {
    /// `DoLa`-static: always contrast against this one layer.
    ///
    /// Chuang et al. report that the *best* static layer varies by task and by
    /// model, which is precisely the observation that motivates the dynamic
    /// selection below. This variant exists so that claim can be measured rather
    /// than believed.
    Fixed(usize),

    /// `DoLa` (dynamic): contrast against
    /// `arg max_j JSD( q_mature ‖ q_j )` over a candidate bucket, **recomputed at
    /// every decoding step**.
    ///
    /// The premise: the layer that disagrees *most* with the final layer is the
    /// layer across which the model did the most work, and subtracting it isolates
    /// what the upper layers actually contributed. A token whose probability was
    /// already settled in the early layers (a function word, a continuation of a
    /// name) has a small divergence and is barely touched; a token that only the
    /// upper layers became confident about (a fact) is amplified.
    ///
    /// Because it is recomputed per step, different tokens in the same generation
    /// contrast against different layers.
    MaxJensenShannon {
        /// The candidate premature layers.
        ///
        /// **Empty means "every layer except the mature one"** — the exhaustive
        /// selection, and the default. Chuang et al. instead partition the stack
        /// into *buckets* (e.g. even-numbered layers in the lower or upper half)
        /// to cut the divergence computations; [`LayerSelector::bucket`] builds
        /// one.
        ///
        /// The mature layer must not appear here: see
        /// [`ContextAwareError::MatureLayerInCandidates`].
        candidates: Vec<usize>,
    },
}

impl Default for LayerSelector {
    /// The exhaustive dynamic selection: every layer but the mature one.
    fn default() -> Self {
        Self::MaxJensenShannon {
            candidates: Vec::new(),
        }
    }
}

impl LayerSelector {
    /// A dynamic selector over `start, start + stride, …` while `< end`.
    ///
    /// `LayerSelector::bucket(0, 16, 2)` is Chuang et al.'s "lower bucket" of a
    /// 32-layer model: the even layers of its bottom half.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidConfig`] when `stride` is zero, and
    /// [`ContextAwareError::EmptyCandidateSet`] when the range is empty.
    pub fn bucket(start: usize, end: usize, stride: usize) -> ContextAwareResult<Self> {
        if stride == 0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: "layer bucket stride must be at least 1".to_string(),
            });
        }
        let candidates: Vec<usize> = (start..end).step_by(stride).collect();
        if candidates.is_empty() {
            return Err(ContextAwareError::EmptyCandidateSet);
        }
        Ok(Self::MaxJensenShannon { candidates })
    }
}

/// Tunables for **`DoLa`** (Chuang et al., 2024).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DolaConfig {
    /// Which layer is the **mature** one. `None` means the last, which is the
    /// model's actual output layer and is what the paper uses.
    ///
    /// Overriding it is useful for ablations ("what if we treated layer 24 as
    /// final?"), and for that reason it is validated against the model's depth
    /// rather than assumed.
    pub mature_layer: Option<usize>,

    /// How to pick the premature layer.
    pub selector: LayerSelector,

    /// Plausibility constraint, generation cap, and token selection.
    ///
    /// Chuang et al.'s published plausibility `alpha` is `0.1`, this module's
    /// default. In `DoLa` the constraint is not optional garnish: the contrast
    /// weights are fixed at `(1, −1)`, so `p ∝ q_mature / q_premature`, and a
    /// token that an early layer has all but ruled out has an *enormous* ratio
    /// no matter how little the final layer thinks of it.
    pub shared: ContextAwareConfig,
}

impl DolaConfig {
    /// A configuration with the given selector and everything else defaulted.
    #[must_use]
    pub fn new(selector: LayerSelector) -> Self {
        Self {
            mature_layer: None,
            selector,
            shared: ContextAwareConfig::default(),
        }
    }

    /// Reject a configuration that cannot produce a meaningful distribution.
    ///
    /// Layer indices are *not* checked here — that needs the model's depth, and
    /// happens in `DoLaDecoder::resolve_candidates`.
    ///
    /// # Errors
    ///
    /// Propagates [`ContextAwareConfig::validate`].
    pub fn validate(&self) -> ContextAwareResult<()> {
        self.shared.validate()
    }

    /// The two contrast weights, `(1, −1)`.
    ///
    /// Fixed by the method, and deliberately not configurable: `DoLa`'s objective
    /// is the plain log-ratio `log q_mature − log q_premature`. The knob `DoLa`
    /// exposes is not *how hard* to contrast but *what to contrast against* —
    /// [`DolaConfig::selector`].
    #[must_use]
    pub fn contrast_weights(&self) -> (f64, f64) {
        (1.0, -1.0)
    }

    /// Set which layer counts as mature.
    #[must_use]
    pub fn with_mature_layer(mut self, layer: Option<usize>) -> Self {
        self.mature_layer = layer;
        self
    }

    /// Set the adaptive plausibility threshold.
    #[must_use]
    pub fn with_plausibility_alpha(mut self, alpha: f64) -> Self {
        self.shared.plausibility_alpha = alpha;
        self
    }

    /// Set the generation length cap.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.shared.max_tokens = max_tokens;
        self
    }

    /// Set greedy or sampled selection.
    #[must_use]
    pub fn with_strategy(mut self, strategy: DecodingStrategy) -> Self {
        self.shared.strategy = strategy;
        self
    }
}

/// The premature layer `DoLa` chose for one decoding step, and the evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrematureLayer {
    /// The selected layer index.
    pub layer: usize,

    /// `JSD( q_mature ‖ q_layer )` in **nats**, the quantity that won the
    /// selection. Bounded above by `ln 2 ≈ 0.693`, attained only when the two
    /// layers put their mass on disjoint token sets.
    pub divergence_nats: f64,

    /// Every candidate that was considered, as `(layer, JSD in nats)`, in
    /// ascending layer order.
    ///
    /// The audit trail, and the reason this type exists rather than a bare
    /// `usize`: the *shape* of this curve across the stack is the diagnostic that
    /// tells you whether the model settled the token early (flat, small
    /// divergences) or late (a pronounced peak).
    pub considered: Vec<(usize, f64)>,
}

impl PrematureLayer {
    /// The largest divergence among the candidates, i.e. `divergence_nats`.
    #[must_use]
    pub fn max_divergence_nats(&self) -> f64 {
        self.divergence_nats
    }

    /// The smallest divergence among the candidates — the layer that had already
    /// converged on the final answer.
    #[must_use]
    pub fn min_divergence_nats(&self) -> f64 {
        self.considered
            .iter()
            .map(|&(_, divergence)| divergence)
            .fold(f64::INFINITY, f64::min)
    }
}

// ── One contrasted step ──────────────────────────────────────────────────────

/// One contrasted next-token distribution, with both of its inputs and the
/// plausibility mask that shaped it.
///
/// Everything needed to audit a single decoding step, because the whole point of
/// this module is that the output is *not* the model's distribution and you
/// should be able to see exactly how it differs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastOutcome {
    /// Which method produced this.
    pub mode: DecodeMode,

    /// The contrasted, constrained, renormalized `log p(y)`.
    ///
    /// Entries excluded by the plausibility constraint are exactly
    /// `f64::NEG_INFINITY`, not merely small: the constraint is a hard mask, and
    /// a hard mask should be visible as one.
    pub log_probs: Vec<f64>,

    /// `log p⁺(y)` — the **positive** distribution, normalized, before any
    /// contrast. For `CAD` this is the context-conditioned distribution; the
    /// quantity `alpha = 0` reproduces.
    pub positive_log_probs: Vec<f64>,

    /// `log p⁻(y)` — the **negative** distribution, normalized. The thing being
    /// subtracted.
    pub negative_log_probs: Vec<f64>,

    /// The adaptive plausibility mask `V_valid`, computed from `p⁺`.
    ///
    /// `plausible[y]` is `false` exactly when `y` was set to `-inf`. At least one
    /// entry is always `true` — the arg-max of `p⁺` satisfies
    /// `p⁺(y) ≥ alpha · max p⁺` for every `alpha ≤ 1`, so the constraint can
    /// never empty the vocabulary.
    pub plausible: Vec<bool>,

    /// Which premature layer was contrasted against, for
    /// [`DecodeMode::Dola`]. `None` for the other two modes.
    pub premature_layer: Option<PrematureLayer>,
}

impl ContrastOutcome {
    /// The contrasted distribution in probability space.
    ///
    /// Excluded tokens are exactly `0.0`.
    #[must_use]
    pub fn probs(&self) -> Vec<f64> {
        self.log_probs.iter().copied().map(f64::exp).collect()
    }

    /// The positive distribution in probability space — what the model would have
    /// said without any contrast.
    #[must_use]
    pub fn positive_probs(&self) -> Vec<f64> {
        self.positive_log_probs
            .iter()
            .copied()
            .map(f64::exp)
            .collect()
    }

    /// `|V_valid|` — how many tokens survived the plausibility constraint.
    #[must_use]
    pub fn plausible_count(&self) -> usize {
        self.plausible.iter().filter(|&&keep| keep).count()
    }

    /// Whether the constraint actually removed anything on this step.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.plausible.iter().any(|&keep| !keep)
    }

    /// The greedy choice: `arg max_y log p(y)`, ties to the lowest token id.
    #[must_use]
    pub fn argmax(&self) -> Option<usize> {
        crate::replug::math::arg_max(&self.log_probs)
    }

    /// The greedy choice under the **positive** distribution alone — what the
    /// model would have emitted with no contrast at all.
    ///
    /// The pair `(positive_argmax, argmax)` is the headline measurement of this
    /// module: when they differ, the contrast changed the model's mind.
    #[must_use]
    pub fn positive_argmax(&self) -> Option<usize> {
        crate::replug::math::arg_max(&self.positive_log_probs)
    }

    /// Shannon entropy of the contrasted distribution, in nats.
    #[must_use]
    pub fn entropy_nats(&self) -> f64 {
        crate::replug::math::entropy_from_log_probs(&self.log_probs)
    }
}

// ── Generation output ────────────────────────────────────────────────────────

/// One step of a contrasted generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAwareStep {
    /// 0-based index of this decoding step.
    pub step: usize,

    /// The token this step emitted.
    pub token_id: usize,

    /// `log p(token_id)` under the **contrasted** distribution.
    pub token_log_prob: f64,

    /// `log p⁺(token_id)` — what the positive distribution alone gave it.
    ///
    /// The difference `token_log_prob − positive_token_log_prob` is, in nats,
    /// exactly how much the contrast moved this token.
    pub positive_token_log_prob: f64,

    /// `log p⁻(token_id)` — what the negative distribution gave it.
    pub negative_token_log_prob: f64,

    /// Entropy of the contrasted distribution, in nats.
    pub entropy_nats: f64,

    /// `|V_valid|` at this step.
    pub plausible_tokens: usize,

    /// The premature layer chosen at this step, for [`DecodeMode::Dola`].
    pub premature_layer: Option<PrematureLayer>,

    /// The full contrasted `log p(·)`, present only when
    /// [`ContextAwareConfig::record_distributions`] is set.
    pub log_distribution: Option<Vec<f64>>,

    /// The full positive `log p⁺(·)`, under the same condition.
    pub positive_log_distribution: Option<Vec<f64>>,
}

/// Per-run statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAwareStats {
    /// Which contrast produced the run.
    pub mode: DecodeMode,

    /// The model's vocabulary size.
    pub vocab_size: usize,

    /// How many tokens were generated.
    pub generated_tokens: usize,

    /// Logit queries issued to the model(s).
    ///
    /// Two per step for `CAD` (two prompts) and for contrastive decoding (two
    /// models); `1 + |candidates|` per step for `DoLa`. The `DoLa` figure counts
    /// *trait calls*, not forward passes: a real layered backend serves every
    /// layer from the **one** forward pass it already ran, which is exactly why
    /// `DoLa` is the cheapest of the three at inference time even though it looks
    /// like the most expensive here.
    pub lm_calls: usize,

    /// `Σ_t log p(y_t)` under the contrasted distributions.
    pub sequence_log_prob: f64,

    /// `Σ_t log p⁺(y_t)` — the same tokens, scored by the *uncontrasted* positive
    /// distribution.
    ///
    /// The gap between this and `sequence_log_prob` is the total evidence, in
    /// nats, that the contrast supplied.
    pub positive_sequence_log_prob: f64,

    /// Mean per-token log-probability under the contrasted distributions.
    pub mean_token_log_prob: f64,

    /// Mean entropy (nats) of the contrasted distributions.
    pub mean_entropy_nats: f64,

    /// Mean `|V_valid|` across steps.
    pub mean_plausible_tokens: f64,

    /// How many steps the plausibility constraint actually removed a token on.
    ///
    /// `0` means the constraint never bound, and the run is bit-for-bit what the
    /// unconstrained extrapolation would have produced.
    pub truncated_steps: usize,
}

/// The full result of a contrasted generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAwareOutput {
    /// The decoded text.
    pub text: String,

    /// The generated token ids.
    pub token_ids: Vec<usize>,

    /// Per-step record.
    pub steps: Vec<ContextAwareStep>,

    /// Statistics for the run.
    pub stats: ContextAwareStats,
}
