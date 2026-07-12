//! [`WatermarkGenerator`] — generation-side green-list biasing.
//!
//! This crate has no language model of its own, so `WatermarkGenerator`
//! operates on caller-supplied logit vectors (as any external decoder would
//! hand it, one step at a time) rather than driving a model directly. It
//! also never samples stochastically: per this crate's no-`rand` convention
//! there is no temperature/nucleus sampler here, only deterministic greedy
//! (arg-max) decoding after the configured bias has been applied. Greedy
//! decoding is sufficient to demonstrate — and to test — both watermarking
//! properties this module cares about: soft mode's bias only changes the
//! decision on steps where the model was already close to indifferent
//! (see [`crate::watermarking::WatermarkMode::Soft`]), and both modes
//! systematically prefer the green list often enough for
//! [`crate::watermarking::WatermarkDetector`] to detect it.

use super::hasher::WatermarkHasher;
use super::types::{
    WatermarkConfig, WatermarkError, WatermarkMode, WatermarkResult, WatermarkTokenId,
};

/// Applies watermark bias to logit vectors and greedily decodes the result.
///
/// Construct once per `(config, mode)` pair; `bias_logits`/`generate_token`
/// can then be called once per decoding step, or [`generate_sequence`] can
/// drive a whole precomputed table of per-step logits at once.
///
/// [`generate_sequence`]: Self::generate_sequence
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkGenerator {
    config: WatermarkConfig,
    mode: WatermarkMode,
    hasher: WatermarkHasher,
}

impl WatermarkGenerator {
    /// Build a generator for `config` and `mode`.
    ///
    /// # Errors
    ///
    /// Propagates [`WatermarkConfig::validate`]'s error if `config` is
    /// invalid.
    pub fn new(config: WatermarkConfig, mode: WatermarkMode) -> WatermarkResult<Self> {
        let hasher = WatermarkHasher::new(&config)?;
        Ok(Self {
            config,
            mode,
            hasher,
        })
    }

    /// The configuration this generator was built with.
    #[must_use]
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }

    /// The embedding strategy this generator applies.
    #[must_use]
    pub fn mode(&self) -> WatermarkMode {
        self.mode
    }

    /// Apply this generator's watermark bias to `logits` in place, using the
    /// green list derived from `context` (the tokens immediately preceding
    /// the position being decoded, in order, oldest first).
    ///
    /// If `context` has fewer than `context_width` tokens — which happens
    /// only for the first `context_width` positions of any sequence, before
    /// enough predecessor tokens exist — `logits` is left **unchanged**: no
    /// green list can be computed yet, so no bias can be applied. This
    /// mirrors [`crate::watermarking::WatermarkDetector::detect`]'s
    /// exclusion of those same leading positions from scoring; it is
    /// expected behavior at the start of every sequence, not an error.
    ///
    /// When `context` has *more* than `context_width` tokens, only the
    /// trailing `context_width` are used (the caller may simply pass the
    /// full generated-so-far history and let this method take the suffix it
    /// needs).
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::LogitsVocabMismatch`] if
    /// `logits.len() != config.vocab_size`. Returns
    /// [`WatermarkError::EmptyGreenList`] in [`WatermarkMode::Hard`] mode if
    /// the green list for this context is empty (only possible when
    /// `gamma == 0.0`): every logit would have to be forced to negative
    /// infinity, leaving nothing legal to decode.
    pub fn bias_logits(
        &self,
        logits: &mut [f64],
        context: &[WatermarkTokenId],
    ) -> WatermarkResult<()> {
        if logits.len() != self.config.vocab_size {
            return Err(WatermarkError::LogitsVocabMismatch {
                expected: self.config.vocab_size,
                actual: logits.len(),
            });
        }
        if context.len() < self.config.context_width {
            // Not enough predecessor context yet: no bias this early.
            return Ok(());
        }
        let trailing = &context[context.len() - self.config.context_width..];
        let mask = self.hasher.green_mask(trailing)?;

        match self.mode {
            WatermarkMode::Soft => {
                for (logit, &is_green) in logits.iter_mut().zip(mask.iter()) {
                    if is_green {
                        *logit += self.config.delta;
                    }
                }
            }
            WatermarkMode::Hard => {
                if self.hasher.green_size() == 0 {
                    return Err(WatermarkError::EmptyGreenList {
                        gamma: self.config.gamma,
                        vocab_size: self.config.vocab_size,
                    });
                }
                for (logit, &is_green) in logits.iter_mut().zip(mask.iter()) {
                    if !is_green {
                        *logit = f64::NEG_INFINITY;
                    }
                }
            }
        }
        Ok(())
    }

    /// Bias `logits` per [`bias_logits`](Self::bias_logits) and greedily
    /// decode the result: the token id with the largest (post-bias) logit,
    /// ties broken toward the lowest token id.
    ///
    /// `logits` is not mutated; a scratch copy is biased internally.
    ///
    /// # Errors
    ///
    /// Same as [`bias_logits`](Self::bias_logits).
    pub fn generate_token(
        &self,
        logits: &[f64],
        context: &[WatermarkTokenId],
    ) -> WatermarkResult<WatermarkTokenId> {
        let mut biased = logits.to_vec();
        self.bias_logits(&mut biased, context)?;
        Ok(argmax(&biased))
    }

    /// Generate a full token sequence from a table of per-step logit
    /// vectors (`logits_per_step[i]` is the vocabulary-length logit vector
    /// for decoding position `i`), growing the predecessor context out of
    /// the tokens already produced.
    ///
    /// This is the convenience entry point most callers (and this module's
    /// tests) should reach for: it is exactly the loop
    /// [`generate_token`](Self::generate_token) would need driven manually,
    /// with the trailing-context bookkeeping done once, here.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::LogitsVocabMismatch`] if any step's logit
    /// vector does not have length `vocab_size`, or (in
    /// [`WatermarkMode::Hard`]) [`WatermarkError::EmptyGreenList`] if some
    /// fully-contexted step's green list is empty.
    pub fn generate_sequence(
        &self,
        logits_per_step: &[Vec<f64>],
    ) -> WatermarkResult<Vec<WatermarkTokenId>> {
        let mut tokens: Vec<WatermarkTokenId> = Vec::with_capacity(logits_per_step.len());
        for logits in logits_per_step {
            let start = tokens.len().saturating_sub(self.config.context_width);
            let token = self.generate_token(logits, &tokens[start..])?;
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// The index of the largest value in `values` (ties broken toward the
/// lowest index), via [`f64::total_cmp`] so `NEG_INFINITY` (used by
/// [`WatermarkMode::Hard`] to exclude red-list tokens) orders correctly and
/// no float comparison ever panics or silently mis-orders `NaN`.
///
/// Callers guarantee `values` is non-empty (`vocab_size >= 1` is enforced by
/// [`WatermarkConfig::validate`]); an empty slice returns `0`.
fn argmax(values: &[f64]) -> WatermarkTokenId {
    let mut best_idx: usize = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value.total_cmp(&best_val) == std::cmp::Ordering::Greater {
            best_val = value;
            best_idx = idx;
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let best_idx = best_idx as WatermarkTokenId;
    best_idx
}
