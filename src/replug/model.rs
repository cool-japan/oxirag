//! The language-model abstraction `REPLUG` needs — and a deterministic,
//! table-driven implementation of it for tests, doctests, and pipeline
//! validation.
//!
//! # Why this module defines its own LM trait
//!
//! `REPLUG` is an operation on **next-token probability distributions**. It
//! cannot be built on top of an interface that returns text, or even one that
//! returns a scalar log-probability per *sampled* token: the mixture
//! `Σ_i λ_i p(y | d_i ⊕ q)` must be evaluated at every `y` in the vocabulary,
//! so the ensemble needs the *whole* distribution from each document's forward
//! pass, before any token has been chosen.
//!
//! No existing trait in this crate exposes that. `SmallLanguageModel` (the
//! speculator's LM interface) returns generated text plus per-token
//! log-probabilities for the tokens it happened to emit — the full logit vector
//! is computed inside the Candle backend and discarded on the way out. Rather
//! than reach across module boundaries and widen someone else's trait,
//! `REPLUG` states its own, minimal requirement here:
//! [`ReplugLanguageModel`] is *exactly* "give me the next-token logits for this
//! context, and tell me how to turn a token id back into text".
//!
//! Any real backend (Candle, an HTTP endpoint, a quantized on-device model) can
//! implement it in a few lines, and this module never needs to know which.

use super::types::{ReplugError, ReplugResult};

// ── The trait ────────────────────────────────────────────────────────────────

/// A language model that can expose a full next-token distribution.
///
/// The contract is deliberately small, and every clause of it is load-bearing:
///
/// - [`next_token_logits`](ReplugLanguageModel::next_token_logits) returns
///   **unnormalized logits**, not probabilities. The engine normalizes them
///   itself, in log-space, because it needs `log_softmax` anyway and because a
///   backend that pre-normalizes has already thrown away the precision the
///   ensemble depends on (see [`super::math`]).
/// - The returned vector must have length
///   [`vocab_size`](ReplugLanguageModel::vocab_size) and contain only finite
///   values. Both are checked by the engine; a violation is a
///   [`ReplugError::VocabSizeMismatch`] or [`ReplugError::NonFiniteLogit`],
///   never a silently-repaired distribution.
/// - `f32` is the right type here — it is what a model's final projection
///   natively produces, and there is no precision below it to preserve. The
///   engine promotes to `f64` at this boundary and does all *distribution*
///   arithmetic there.
///
/// `Send + Sync` so an engine holding one can be shared across threads, matching
/// the rest of this crate's model traits.
pub trait ReplugLanguageModel: Send + Sync {
    /// The size of the vocabulary, i.e. the length of every logit vector.
    fn vocab_size(&self) -> usize;

    /// The next-token logits for `context`.
    ///
    /// `context` is the fully-formed prompt — in `REPLUG` that is
    /// `d_i ⊕ q ⊕ (tokens generated so far)`, assembled by the engine.
    ///
    /// # Errors
    ///
    /// Returns a [`ReplugError`] when the model cannot produce logits for the
    /// given context.
    fn next_token_logits(&self, context: &str) -> ReplugResult<Vec<f32>>;

    /// The surface text of a token id, for assembling the generated string and
    /// for extending the context on the next step.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::TokenOutOfRange`] when `token_id` is not in the
    /// vocabulary.
    fn decode(&self, token_id: usize) -> ReplugResult<String>;

    /// The end-of-sequence token, if the model has one. Generation halts when
    /// the ensemble selects it. Defaults to "no `EOS`", i.e. generate until
    /// `max_tokens`.
    fn eos_token_id(&self) -> Option<usize> {
        None
    }
}

// ── A deterministic fixture model ────────────────────────────────────────────

/// A conditional rule of [`ReplugStaticLanguageModel`]: "when the context
/// contains *all* of these substrings, the next-token logits are these".
#[derive(Debug, Clone, PartialEq)]
pub struct ReplugContextRule {
    /// Every one of these must occur in the context for the rule to fire (a
    /// conjunction, so a rule can key on "this document **and** this
    /// already-generated prefix").
    pub required_substrings: Vec<String>,
    /// The logits to return when the rule fires.
    pub logits: Vec<f32>,
}

/// A fully-deterministic, table-driven language model.
///
/// **This is a test fixture, not a language model.** It does no inference and
/// has learned nothing: it looks up logits you supplied, by substring-matching
/// the context against rules you wrote. It exists so that every claim this
/// module makes about the `REPLUG` mixture — that it is an arithmetic mean and
/// not a geometric one, that it beats its own components, that it survives
/// logits of `±1e4` — can be checked against *hand-computed* numbers, with no
/// model weights, no randomness, and no network anywhere in the loop. It is
/// public because those checks are as useful to a caller wiring up their own
/// [`ReplugLanguageModel`] as they are to this crate's own test suite.
///
/// # Rule selection
///
/// Among all rules whose `required_substrings` are *all* present in the
/// context, the winner is the **most specific**: most required substrings
/// first, then greatest total substring length, then earliest insertion. If no
/// rule fires, the model returns its default logits. The ordering is a total
/// one, so the model is deterministic for any context.
#[derive(Debug, Clone)]
pub struct ReplugStaticLanguageModel {
    token_texts: Vec<String>,
    rules: Vec<ReplugContextRule>,
    default_logits: Vec<f32>,
    eos_token_id: Option<usize>,
}

impl ReplugStaticLanguageModel {
    /// Build a model over the given token surface strings, with a uniform
    /// default distribution (all-zero logits).
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::EmptyLogits`] when the vocabulary is empty.
    pub fn new(token_texts: Vec<String>) -> ReplugResult<Self> {
        if token_texts.is_empty() {
            return Err(ReplugError::EmptyLogits);
        }
        let vocab_size = token_texts.len();
        Ok(Self {
            token_texts,
            rules: Vec::new(),
            default_logits: vec![0.0; vocab_size],
            eos_token_id: None,
        })
    }

    /// Convert a probability vector into logits by taking natural logarithms.
    ///
    /// `softmax(ln p) == p` exactly (up to rounding) whenever `p` is a
    /// normalized distribution, which is what makes hand-computed tests
    /// possible: you write down the distribution you want the model to have,
    /// and the model has it.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::InvalidConfig`] when any probability is not
    /// strictly positive — `ln(0) = -inf` is not a logit. To express a
    /// *point mass*, do not pass a zero probability: pass raw logits with a
    /// large gap (e.g. `[1e4, 0.0, 0.0]`), whose softmax underflows to an exact
    /// `[1, 0, 0]` in probability space while remaining perfectly finite in
    /// log-space. That is what a real model's point mass looks like, and it is
    /// the case the ensemble has to survive.
    pub fn logits_from_probabilities(probabilities: &[f64]) -> ReplugResult<Vec<f32>> {
        if probabilities.is_empty() {
            return Err(ReplugError::EmptyLogits);
        }
        let mut logits = Vec::with_capacity(probabilities.len());
        for (index, &probability) in probabilities.iter().enumerate() {
            if !probability.is_finite() || probability <= 0.0 {
                return Err(ReplugError::InvalidConfig {
                    reason: format!(
                        "probability at index {index} must be strictly positive and finite, \
                         got {probability}; use raw logits to express a point mass"
                    ),
                });
            }
            #[allow(clippy::cast_possible_truncation)] // f64 -> f32 logit, by design.
            logits.push(probability.ln() as f32);
        }
        Ok(logits)
    }

    /// Set the logits returned when no rule fires.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::VocabSizeMismatch`] when the length is wrong.
    pub fn with_default_logits(mut self, logits: Vec<f32>) -> ReplugResult<Self> {
        self.check_len(&logits, 0)?;
        self.default_logits = logits;
        Ok(self)
    }

    /// Add a rule: when the context contains every substring in `required`,
    /// return `logits`.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::VocabSizeMismatch`] when `logits` has the wrong
    /// length, and [`ReplugError::InvalidConfig`] when `required` is empty (a
    /// rule that always fires is what `with_default_logits` is for).
    pub fn with_rule(
        mut self,
        required: Vec<impl Into<String>>,
        logits: Vec<f32>,
    ) -> ReplugResult<Self> {
        self.check_len(&logits, self.rules.len())?;
        let required_substrings: Vec<String> = required
            .into_iter()
            .map(Into::into)
            .collect::<Vec<String>>();
        if required_substrings.is_empty() {
            return Err(ReplugError::InvalidConfig {
                reason: "a context rule needs at least one required substring".to_string(),
            });
        }
        self.rules.push(ReplugContextRule {
            required_substrings,
            logits,
        });
        Ok(self)
    }

    /// Add a rule whose distribution is given as probabilities.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::logits_from_probabilities`] and [`Self::with_rule`].
    pub fn with_probability_rule(
        self,
        required: Vec<impl Into<String>>,
        probabilities: &[f64],
    ) -> ReplugResult<Self> {
        let logits = Self::logits_from_probabilities(probabilities)?;
        self.with_rule(required, logits)
    }

    /// Declare an end-of-sequence token.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::TokenOutOfRange`] when the id is not in the
    /// vocabulary.
    pub fn with_eos_token(mut self, token_id: usize) -> ReplugResult<Self> {
        if token_id >= self.token_texts.len() {
            return Err(ReplugError::TokenOutOfRange {
                token_id,
                vocab_size: self.token_texts.len(),
            });
        }
        self.eos_token_id = Some(token_id);
        Ok(self)
    }

    /// The token id of a surface string, if present. Handy for building a
    /// ground-truth continuation `y*` in `LSR` tests.
    #[must_use]
    pub fn token_id(&self, text: &str) -> Option<usize> {
        self.token_texts
            .iter()
            .position(|candidate| candidate == text)
    }

    /// Encode a whole continuation into token ids by exact surface match.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::InvalidConfig`] naming the first token text that
    /// is not in the vocabulary.
    pub fn encode(&self, texts: &[&str]) -> ReplugResult<Vec<usize>> {
        texts
            .iter()
            .map(|text| {
                self.token_id(text)
                    .ok_or_else(|| ReplugError::InvalidConfig {
                        reason: format!("token text {text:?} is not in the fixture vocabulary"),
                    })
            })
            .collect()
    }

    fn check_len(&self, logits: &[f32], document_index: usize) -> ReplugResult<()> {
        if logits.len() == self.token_texts.len() {
            Ok(())
        } else {
            Err(ReplugError::VocabSizeMismatch {
                expected: self.token_texts.len(),
                actual: logits.len(),
                document_index,
            })
        }
    }

    /// The winning rule for a context, by the specificity order documented on
    /// the type. Returns `None` when nothing matches.
    fn select_rule(&self, context: &str) -> Option<&ReplugContextRule> {
        self.rules
            .iter()
            .filter(|rule| {
                rule.required_substrings
                    .iter()
                    .all(|needle| context.contains(needle.as_str()))
            })
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                let left_length: usize = left
                    .required_substrings
                    .iter()
                    .map(String::len)
                    .sum::<usize>();
                let right_length: usize = right
                    .required_substrings
                    .iter()
                    .map(String::len)
                    .sum::<usize>();
                left.required_substrings
                    .len()
                    .cmp(&right.required_substrings.len())
                    .then_with(|| left_length.cmp(&right_length))
                    // `max_by` keeps the last maximum; reverse the index order
                    // so that earliest-inserted wins a full tie.
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(_, rule)| rule)
    }
}

impl ReplugLanguageModel for ReplugStaticLanguageModel {
    fn vocab_size(&self) -> usize {
        self.token_texts.len()
    }

    fn next_token_logits(&self, context: &str) -> ReplugResult<Vec<f32>> {
        Ok(self
            .select_rule(context)
            .map_or_else(|| self.default_logits.clone(), |rule| rule.logits.clone()))
    }

    fn decode(&self, token_id: usize) -> ReplugResult<String> {
        self.token_texts
            .get(token_id)
            .cloned()
            .ok_or(ReplugError::TokenOutOfRange {
                token_id,
                vocab_size: self.token_texts.len(),
            })
    }

    fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id
    }
}
