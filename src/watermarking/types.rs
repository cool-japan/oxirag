//! Types for the `watermarking` module: configuration, mode, detection
//! outcome, token-id alias, and errors.
//!
//! Nothing in this file performs any hashing or statistics — it only
//! describes the tunable parameters ([`WatermarkConfig`]), the two embedding
//! strategies ([`WatermarkMode`]), the outcome of a detection run
//! ([`WatermarkDetection`]), and the failure modes ([`WatermarkError`]). The
//! actual green-list hashing lives in [`crate::watermarking::hasher`], the
//! generation-side biasing in [`crate::watermarking::generator`], and the
//! detection statistics in [`crate::watermarking::detector`] /
//! [`crate::watermarking::stats`].

use thiserror::Error;

// ── WatermarkTokenId ─────────────────────────────────────────────────────────

/// A vocabulary token id.
///
/// `u32` comfortably covers every LLM vocabulary in current use (the
/// largest published tokenizers top out in the low hundreds of thousands),
/// while keeping green-list bookkeeping ([`Vec<bool>`] masks indexed by
/// token id) compact.
pub type WatermarkTokenId = u32;

// ── WatermarkMode ────────────────────────────────────────────────────────────

/// Which watermark-embedding strategy [`crate::watermarking::WatermarkGenerator`]
/// applies at each generation step.
///
/// Both modes derive the same per-step green list (see
/// [`crate::watermarking::WatermarkHasher`]) from the preceding
/// [`WatermarkConfig::context_width`] tokens and the secret key; they differ
/// only in how that green list is used to influence the next token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatermarkMode {
    /// Add [`WatermarkConfig::delta`] to every green-list token's logit
    /// before the argmax/softmax decision, but never forbid a red-list
    /// token outright.
    ///
    /// This is the variant that matters in practice (Kirchenbauer et al.,
    /// 2023): because the bias is additive rather than exclusionary, a
    /// token whose logit already leads every other token by more than
    /// `delta` is picked identically whether or not it happens to be green
    /// — the watermark can only ever change the *outcome* of a decoding
    /// step where the model itself was already close to indifferent between
    /// two or more candidates. Put differently, the watermark is imprinted
    /// only where the model has genuine entropy to spend; it never corrupts
    /// a step the model was confident about. The direct, documented
    /// consequence is that **low-entropy text carries a weak watermark**
    /// (few or no steps have enough spare entropy for `delta` to flip the
    /// decision), which degrades detection power on such text. That is a
    /// real, inherent limitation of soft watermarking, not a bug in this
    /// implementation.
    Soft,
    /// Restrict sampling to the green list entirely: every red-list token's
    /// logit is driven to negative infinity before the decision.
    ///
    /// This is the theoretical baseline the soft variant improves on. It
    /// maximizes detectability (every scored token is green by
    /// construction, whenever the green list is non-empty) at the cost of
    /// quality: on a step where the model's single best token happens to be
    /// red, hard mode forces a strictly worse — sometimes semantically
    /// wrong — substitute. Included here mainly so [`WatermarkGenerator`]
    /// callers can quantify that quality/detectability trade-off, not
    /// because it is recommended for production use.
    ///
    /// [`WatermarkGenerator`]: crate::watermarking::WatermarkGenerator
    Hard,
}

// ── WatermarkConfig ──────────────────────────────────────────────────────────

/// Configuration shared by [`crate::watermarking::WatermarkHasher`],
/// [`crate::watermarking::WatermarkGenerator`], and
/// [`crate::watermarking::WatermarkDetector`].
///
/// Generation and detection **must** use the same configuration (in
/// particular the same [`secret_key`](Self::secret_key),
/// [`gamma`](Self::gamma), and [`context_width`](Self::context_width)) —
/// the whole scheme works because both sides recompute the identical
/// deterministic green list from the identical inputs; there is no shared
/// RNG state to synchronize.
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkConfig {
    /// Fraction of the vocabulary placed on the green list at every step,
    /// in `[0.0, 1.0]`. The paper's default is `0.25`. `0.0` makes every
    /// token red (no green list at all, so soft mode degenerates to
    /// unbiased generation and detection carries no signal); `1.0` makes
    /// every token green (soft mode's bias becomes a constant shift that
    /// changes nothing, and detection again carries no signal, since every
    /// token is green regardless of the model's choice). Both extremes are
    /// valid configurations, not errors — see [`Self::validate`].
    pub gamma: f64,
    /// Logit bias added to every green-list token in [`WatermarkMode::Soft`]
    /// mode. The paper's default is `2.0`. Ignored by [`WatermarkMode::Hard`].
    /// Must be finite; conventionally positive (a negative delta would
    /// bias *away* from the green list, which is a valid but inverted
    /// scheme this module does not specifically optimize for).
    pub delta: f64,
    /// Number of preceding tokens `h` hashed (together with
    /// [`secret_key`](Self::secret_key)) to seed each step's green list.
    /// Must be at least `1`.
    ///
    /// This is a robustness/fragility trade-off:
    ///
    /// * `h = 1` (the paper's default) seeds each step's green list from a
    ///   single preceding token. It is the most robust to text edits: a
    ///   substitution at position `i` can only corrupt the scoring of
    ///   positions `i` and `i + 1` (its own green/red status, and the
    ///   context used to score its immediate successor).
    /// * Larger `h` mixes more preceding tokens into the seed, which is
    ///   harder for an attacker to reverse-engineer (a `h = 1` scheme leaks
    ///   the green/red list for a token's *specific* predecessor to anyone
    ///   who observes enough watermarked text; a wider context is more
    ///   expensive to reconstruct) — but it is correspondingly *more*
    ///   fragile to ordinary edits, since a single substitution now
    ///   corrupts the context (and hence the recomputed green list) of the
    ///   following `h` tokens, not just `1`. See the `robustness_*` tests
    ///   in this module for a direct measurement of this trade-off.
    pub context_width: usize,
    /// The shared secret that seeds every green-list hash. Detection must
    /// use the same key generation did; without it, an adversary cannot
    /// reconstruct any step's green list (and so cannot forge or scrub the
    /// watermark without simply degrading the text, see the `robustness_*`
    /// tests).
    pub secret_key: u64,
    /// Size of the vocabulary being watermarked. Every token id passed to
    /// this module's types must lie in `0..vocab_size`. Must be at least
    /// `1`.
    pub vocab_size: usize,
    /// The z-score threshold [`crate::watermarking::WatermarkDetector::detect`]
    /// compares against to set [`WatermarkDetection::is_watermarked`].
    ///
    /// Because the null distribution of the z-statistic is (by
    /// construction) asymptotically standard normal, this threshold has a
    /// direct false-positive-rate interpretation: the probability that
    /// *non*-watermarked text is flagged as watermarked is approximately
    /// `1 - Φ(z_threshold)` (the upper tail of the standard normal). For
    /// example `z_threshold = 4.0` implies a false-positive rate of
    /// approximately `3.2e-5`; the calibration tests in this module verify
    /// that this theoretical rate matches the empirically observed one.
    pub z_threshold: f64,
}

impl WatermarkConfig {
    /// Create a configuration for a `vocab_size`-token vocabulary and the
    /// given `secret_key`, with the paper's default `gamma = 0.25`,
    /// `delta = 2.0`, `context_width = 1`, and `z_threshold = 4.0`.
    #[must_use]
    pub fn new(vocab_size: usize, secret_key: u64) -> Self {
        Self {
            gamma: 0.25,
            delta: 2.0,
            context_width: 1,
            secret_key,
            vocab_size,
            z_threshold: 4.0,
        }
    }

    /// Override the green-list fraction (default `0.25`).
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Override the soft-mode logit bias (default `2.0`).
    #[must_use]
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Override the context width `h` (default `1`).
    #[must_use]
    pub fn with_context_width(mut self, context_width: usize) -> Self {
        self.context_width = context_width;
        self
    }

    /// Override the secret key.
    #[must_use]
    pub fn with_secret_key(mut self, secret_key: u64) -> Self {
        self.secret_key = secret_key;
        self
    }

    /// Override the vocabulary size.
    #[must_use]
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Override the detection z-score threshold (default `4.0`).
    #[must_use]
    pub fn with_z_threshold(mut self, z_threshold: f64) -> Self {
        self.z_threshold = z_threshold;
        self
    }

    /// Validate this configuration, returning the specific
    /// [`WatermarkError`] variant describing the first problem found.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::InvalidGamma`] unless `gamma` is finite and
    /// within `[0.0, 1.0]`; [`WatermarkError::InvalidDelta`] unless `delta`
    /// is finite; [`WatermarkError::InvalidContextWidth`] unless
    /// `context_width >= 1`; [`WatermarkError::InvalidVocabSize`] unless
    /// `vocab_size >= 1`; [`WatermarkError::InvalidZThreshold`] unless
    /// `z_threshold` is finite.
    pub fn validate(&self) -> Result<(), WatermarkError> {
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(WatermarkError::InvalidGamma { gamma: self.gamma });
        }
        if !self.delta.is_finite() {
            return Err(WatermarkError::InvalidDelta { delta: self.delta });
        }
        if self.context_width < 1 {
            return Err(WatermarkError::InvalidContextWidth {
                context_width: self.context_width,
            });
        }
        if self.vocab_size < 1 {
            return Err(WatermarkError::InvalidVocabSize {
                vocab_size: self.vocab_size,
            });
        }
        if !self.z_threshold.is_finite() {
            return Err(WatermarkError::InvalidZThreshold {
                z_threshold: self.z_threshold,
            });
        }
        Ok(())
    }
}

// ── WatermarkDetection ───────────────────────────────────────────────────────

/// The outcome of [`crate::watermarking::WatermarkDetector::detect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WatermarkDetection {
    /// Number of scored tokens whose green-list membership (recomputed from
    /// their own predecessor context) was green.
    pub green_count: usize,
    /// Number of tokens actually scored — the input length minus
    /// [`WatermarkConfig::context_width`] (the leading tokens excluded for
    /// lacking predecessor context; see
    /// [`crate::watermarking::WatermarkDetector::detect`]).
    pub total_scored: usize,
    /// The test statistic
    /// `z = (green_count - gamma * total_scored) / sqrt(total_scored * gamma * (1 - gamma))`
    /// under the null hypothesis that the text was produced without
    /// knowledge of the green lists (each scored token is independently
    /// green with probability `gamma`).
    ///
    /// Degenerate case: when `gamma` is exactly `0.0` or `1.0` the null
    /// distribution has zero variance (every token is deterministically
    /// red, or deterministically green, regardless of watermarking), so no
    /// statistical test is possible; this field is defined as `0.0` in that
    /// case (see [`crate::watermarking::stats::z_score_and_p_value`]).
    pub z_score: f64,
    /// One-sided p-value `1 - Φ(z_score)` (the standard normal upper tail),
    /// i.e. the probability, under the null hypothesis, of seeing a green
    /// count at least this large by chance. Small values are evidence of
    /// watermarking.
    pub p_value: f64,
    /// Whether `z_score >= `[`WatermarkConfig::z_threshold`].
    pub is_watermarked: bool,
}

impl WatermarkDetection {
    /// The empirical green fraction `green_count / total_scored`, or `0.0`
    /// when nothing was scored.
    #[must_use]
    pub fn green_fraction(&self) -> f64 {
        if self.total_scored == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let fraction = self.green_count as f64 / self.total_scored as f64;
            fraction
        }
    }
}

// ── WatermarkError ───────────────────────────────────────────────────────────

/// Errors produced by the `watermarking` module.
#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum WatermarkError {
    /// [`WatermarkConfig::gamma`] was not finite or not in `[0.0, 1.0]`.
    #[error("gamma must be finite and within [0.0, 1.0], got {gamma}")]
    InvalidGamma {
        /// The offending value.
        gamma: f64,
    },
    /// [`WatermarkConfig::delta`] was not finite.
    #[error("delta must be finite, got {delta}")]
    InvalidDelta {
        /// The offending value.
        delta: f64,
    },
    /// [`WatermarkConfig::context_width`] was `0`.
    #[error("context_width must be at least 1, got {context_width}")]
    InvalidContextWidth {
        /// The offending value.
        context_width: usize,
    },
    /// [`WatermarkConfig::vocab_size`] was `0`.
    #[error("vocab_size must be at least 1, got {vocab_size}")]
    InvalidVocabSize {
        /// The offending value.
        vocab_size: usize,
    },
    /// [`WatermarkConfig::z_threshold`] was not finite.
    #[error("z_threshold must be finite, got {z_threshold}")]
    InvalidZThreshold {
        /// The offending value.
        z_threshold: f64,
    },
    /// A logit vector's length did not match the configured `vocab_size`.
    #[error("logits length {actual} does not match configured vocab_size {expected}")]
    LogitsVocabMismatch {
        /// The configured `vocab_size`.
        expected: usize,
        /// The logit vector's actual length.
        actual: usize,
    },
    /// A context slice's length did not match the configured
    /// `context_width` exactly.
    #[error("context length {actual} does not match configured context_width {expected}")]
    ContextLengthMismatch {
        /// The configured `context_width`.
        expected: usize,
        /// The context slice's actual length.
        actual: usize,
    },
    /// [`crate::watermarking::WatermarkDetector::detect`] was called with an
    /// empty token sequence.
    #[error("text is empty: nothing to detect")]
    EmptyText,
    /// The token sequence had too few tokens for even one to have a full
    /// predecessor context: [`crate::watermarking::WatermarkDetector::detect`]
    /// needs at least `context_width + 1` tokens (the leading `context_width`
    /// tokens are never scorable, see [`WatermarkDetection::total_scored`]).
    #[error(
        "text has {actual} token(s), fewer than the {required} needed to score at least one \
         token (the first context_width token(s) have no predecessor context)"
    )]
    InsufficientContext {
        /// The minimum number of tokens needed (`context_width + 1`).
        required: usize,
        /// The actual number of tokens supplied.
        actual: usize,
    },
    /// A token id in the input sequence was outside `0..vocab_size`.
    #[error("token id {token_id} is out of range for vocab_size {vocab_size}")]
    TokenOutOfRange {
        /// The offending token id.
        token_id: WatermarkTokenId,
        /// The configured `vocab_size`.
        vocab_size: usize,
    },
    /// [`WatermarkMode::Hard`] generation was attempted at a step whose
    /// green list is empty (only possible when `gamma == 0.0`), so no token
    /// can legally be sampled: every logit would have to be driven to
    /// negative infinity.
    #[error(
        "hard-mode generation requires a non-empty green list, but gamma={gamma} yields none \
         for vocab_size={vocab_size}"
    )]
    EmptyGreenList {
        /// The configured `gamma` that produced an empty green list.
        gamma: f64,
        /// The configured `vocab_size`.
        vocab_size: usize,
    },
}

/// Convenience alias for this module's fallible return type.
pub type WatermarkResult<T> = Result<T, WatermarkError>;
