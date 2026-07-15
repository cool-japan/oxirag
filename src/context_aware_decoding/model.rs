//! The language-model abstraction decode-time contrast needs — and a
//! deterministic, table-driven implementation of it for tests, doctests and
//! wiring validation.
//!
//! # Why this module defines its own LM traits
//!
//! The house precedent is explicit. `crate::replug::model` opens with a section
//! titled *"Why this module defines its own LM trait"*, whose conclusion is:
//! rather than reach across module boundaries and widen someone else's trait,
//! state your own, minimal requirement. [`crate::replug::ReplugLanguageModel`]
//! is that requirement for `REPLUG`, and its shape — "give me the full next-token
//! logits for this context" — is *almost* the shape this module needs. All three
//! contrasts are two calls to a model:
//!
//! | | positive call | negative call |
//! |---|---|---|
//! | `CAD` | `next_token_logits(c ⊕ x)` | `next_token_logits(x)` |
//! | contrastive decoding | `expert.next_token_logits(x)` | `amateur.next_token_logits(x)` |
//!
//! Both of those fit `ReplugLanguageModel` exactly, and if the module stopped
//! there it would have no business declaring a new trait.
//!
//! **`DoLa` does not fit.** Its two distributions come from *the same forward
//! pass on the same context*, read out at two different depths of the network —
//! the final layer, and an earlier one projected through the model's output head
//! (the "logit lens"). No interface that maps a *context* to *one* logit vector
//! can express that, however it is composed: the second distribution is not a
//! function of any context you could construct. It is a function of the model's
//! internals.
//!
//! So this module states its own requirement, in two parts, and the split is not
//! cosmetic:
//!
//! - [`ContextAwareLanguageModel`] is the **black-box** interface. An HTTP
//!   endpoint that returns logits satisfies it. `CAD` and contrastive decoding
//!   need nothing more, which is precisely why they work on models you did not
//!   train and cannot open.
//! - [`LayeredLanguageModel`] is the **white-box** interface, and it is a
//!   strictly stronger claim about the backend: that you can read out an
//!   intermediate layer. `DoLa` requires it. That requirement is a real
//!   deployment constraint, and encoding it in the type system means the compiler
//!   states it instead of a paragraph of documentation nobody reads.

use super::types::{ContextAwareError, ContextAwareResult};

// ── The traits ───────────────────────────────────────────────────────────────

/// A language model that can expose a full next-token distribution — the
/// black-box interface.
///
/// The contract is deliberately small, and matches
/// [`crate::replug::ReplugLanguageModel`] clause for clause:
///
/// - `next_token_logits` returns **unnormalized logits**, not probabilities. The
///   decoders normalize in log-space themselves, because they need
///   `log_softmax` anyway and because a backend that pre-normalizes has already
///   discarded the precision a contrast depends on.
/// - The returned vector must have length `vocab_size` and contain only finite
///   values. Both are checked; a violation is a
///   [`ContextAwareError::VocabSizeMismatch`] or
///   [`ContextAwareError::NonFiniteScore`], never a silently-repaired
///   distribution.
/// - `f32` is the right type at this boundary — it is what a model's final
///   projection natively produces. The decoders promote to `f64` here and do all
///   distribution arithmetic there.
///
/// `Send + Sync` so a decoder holding one can be shared across threads, matching
/// the rest of this crate's model traits.
pub trait ContextAwareLanguageModel: Send + Sync {
    /// The size of the vocabulary, i.e. the length of every logit vector.
    fn vocab_size(&self) -> usize;

    /// The next-token logits for `context`.
    ///
    /// # Errors
    ///
    /// Returns a [`ContextAwareError`] when the model cannot produce logits for
    /// the given context.
    fn next_token_logits(&self, context: &str) -> ContextAwareResult<Vec<f32>>;

    /// The surface text of a token id, for assembling the generated string and
    /// for extending the context on the next step.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::TokenOutOfRange`] when `token_id` is not in
    /// the vocabulary.
    fn decode(&self, token_id: usize) -> ContextAwareResult<String>;

    /// The end-of-sequence token, if the model has one. Generation halts when the
    /// contrasted distribution selects it. Defaults to "no `EOS`".
    fn eos_token_id(&self) -> Option<usize> {
        None
    }
}

/// A language model whose **intermediate layers** can be read out through the
/// output head — the white-box interface `DoLa` requires.
///
/// Layers are indexed `0 .. num_layers()`, shallowest first, and
/// `num_layers() - 1` is the **mature** layer: the one whose logits the model
/// actually emits. The contract is therefore
///
/// ```text
///   layer_logits(context, num_layers() - 1)  ==  next_token_logits(context)
/// ```
///
/// and an implementation that violates it will produce a `DoLa` distribution that
/// contrasts against something the model never said.
/// [`ContextAwareStaticLanguageModel`] satisfies it *structurally* — its
/// `next_token_logits` is defined as a call to its own `layer_logits` — rather
/// than by convention, and this module's tests pin that.
///
/// A real backend implements this by applying the final layer norm and the
/// unembedding matrix to the residual stream at layer `j`, which is one matrix
/// multiply against activations the forward pass already computed. Hence the
/// accounting note on `ContextAwareStats::lm_calls`: `DoLa` issues many *trait*
/// calls per token but only **one** forward pass, which is why it is the cheapest
/// of the three contrasts at inference time despite querying the most
/// distributions.
pub trait LayeredLanguageModel: ContextAwareLanguageModel {
    /// How many layers can be read out. Must be at least `1`; `DoLa` needs at
    /// least `2`.
    fn num_layers(&self) -> usize;

    /// The next-token logits obtained by projecting layer `layer`'s hidden state
    /// through the output head.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::LayerOutOfRange`] when `layer` is not in
    /// `0 .. num_layers()`, and whatever the model itself returns.
    fn layer_logits(&self, context: &str, layer: usize) -> ContextAwareResult<Vec<f32>>;
}

// ── A deterministic fixture model ────────────────────────────────────────────

/// A conditional rule of [`ContextAwareStaticLanguageModel`]: "when the context
/// contains *all* of these substrings, the per-layer logits are these".
#[derive(Debug, Clone, PartialEq)]
pub struct ContextAwareRule {
    /// Every one of these must occur in the context for the rule to fire — a
    /// conjunction, so a rule can key on "the retrieved context **and** the query"
    /// while a less specific rule keys on the query alone. That is exactly how a
    /// `CAD` fixture distinguishes its two prompts, since the with-context prompt
    /// contains the query as a substring.
    pub required_substrings: Vec<String>,

    /// One logit vector per layer, shallowest first. The last entry is the mature
    /// layer, and it is what `next_token_logits` returns.
    pub layer_logits: Vec<Vec<f32>>,
}

/// A fully-deterministic, table-driven language model with a readable layer
/// stack.
///
/// **This is a test fixture, not a language model.** It does no inference and has
/// learned nothing: it looks up logits you supplied, by substring-matching the
/// context against rules you wrote. It exists so that every claim this module
/// makes — that `CAD` at `alpha = 0` is the identity, that the plausibility
/// constraint forbids the token the unconstrained extrapolation would have
/// emitted, that `DoLa` selects the layer whose Jensen–Shannon divergence from
/// the final layer is largest — can be checked against **hand-computed numbers**,
/// with no model weights, no randomness and no network anywhere in the loop. It
/// is public because those checks are as useful to a caller wiring up their own
/// backend as they are to this crate's test suite.
///
/// It implements [`LayeredLanguageModel`], and therefore
/// [`ContextAwareLanguageModel`], with the mature-layer contract satisfied by
/// construction: `next_token_logits(c)` *is* `layer_logits(c, num_layers - 1)`,
/// the same code path, not a parallel one that could drift.
///
/// # Rule selection
///
/// Among all rules whose `required_substrings` are *all* present in the context,
/// the winner is the **most specific**: most required substrings first, then
/// greatest total substring length, then earliest insertion. If no rule fires,
/// the model returns its default logits. The ordering is total, so the model is
/// deterministic for any context.
#[derive(Debug, Clone)]
pub struct ContextAwareStaticLanguageModel {
    token_texts: Vec<String>,
    num_layers: usize,
    rules: Vec<ContextAwareRule>,
    default_layer_logits: Vec<Vec<f32>>,
    eos_token_id: Option<usize>,
}

impl ContextAwareStaticLanguageModel {
    /// A single-layer model over the given token surface strings, with a uniform
    /// default distribution (all-zero logits).
    ///
    /// One layer is all `CAD` and contrastive decoding need. `DoLa` needs
    /// [`Self::with_layer_count`].
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::EmptyLogits`] when the vocabulary is empty.
    pub fn new(token_texts: Vec<String>) -> ContextAwareResult<Self> {
        Self::with_layer_count(token_texts, 1)
    }

    /// A model with `num_layers` readable layers, all defaulting to uniform.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::EmptyLogits`] when the vocabulary is empty,
    /// and [`ContextAwareError::InvalidConfig`] when `num_layers` is zero.
    pub fn with_layer_count(
        token_texts: Vec<String>,
        num_layers: usize,
    ) -> ContextAwareResult<Self> {
        if token_texts.is_empty() {
            return Err(ContextAwareError::EmptyLogits);
        }
        if num_layers == 0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: "a model needs at least one layer".to_string(),
            });
        }
        let vocab_size = token_texts.len();
        Ok(Self {
            token_texts,
            num_layers,
            rules: Vec::new(),
            default_layer_logits: vec![vec![0.0; vocab_size]; num_layers],
            eos_token_id: None,
        })
    }

    /// Convert a probability vector into logits by taking natural logarithms.
    ///
    /// `softmax(ln p) == p` exactly (up to rounding) whenever `p` is normalized,
    /// which is what makes hand-computed tests possible: you write down the
    /// distribution you want the model to have, and the model has it.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidConfig`] when any probability is not
    /// strictly positive — `ln(0) = -inf` is not a logit. To express a *point
    /// mass*, pass raw logits with a large gap (e.g. `[1e4, 0.0, 0.0]`), whose
    /// softmax underflows to an exact `[1, 0, 0]` in probability space while
    /// remaining finite in log-space. That is what a real model's point mass looks
    /// like, and it is the case a contrast has to survive.
    pub fn logits_from_probabilities(probabilities: &[f64]) -> ContextAwareResult<Vec<f32>> {
        if probabilities.is_empty() {
            return Err(ContextAwareError::EmptyLogits);
        }
        let mut logits = Vec::with_capacity(probabilities.len());
        for (index, &probability) in probabilities.iter().enumerate() {
            if !probability.is_finite() || probability <= 0.0 {
                return Err(ContextAwareError::InvalidConfig {
                    reason: format!(
                        "probability at index {index} must be strictly positive and finite, got \
                         {probability}; use raw logits to express a point mass"
                    ),
                });
            }
            #[allow(clippy::cast_possible_truncation)] // f64 -> f32 logit, by design.
            logits.push(probability.ln() as f32);
        }
        Ok(logits)
    }

    /// Set the logits returned when no rule fires, for a single-layer model.
    ///
    /// # Errors
    ///
    /// As [`Self::with_default_layer_logits`].
    pub fn with_default_logits(self, logits: Vec<f32>) -> ContextAwareResult<Self> {
        self.with_default_layer_logits(vec![logits])
    }

    /// Set the per-layer logits returned when no rule fires.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidConfig`] when the number of layer
    /// vectors is not `num_layers`, and
    /// [`ContextAwareError::VocabSizeMismatch`] when any of them has the wrong
    /// length.
    pub fn with_default_layer_logits(
        mut self,
        layer_logits: Vec<Vec<f32>>,
    ) -> ContextAwareResult<Self> {
        self.check_layer_logits(&layer_logits, "default logits")?;
        self.default_layer_logits = layer_logits;
        Ok(self)
    }

    /// Add a rule for a single-layer model: when the context contains every
    /// substring in `required`, return `logits`.
    ///
    /// # Errors
    ///
    /// As [`Self::with_layered_rule`]. In particular, calling this on a
    /// multi-layer model is an [`ContextAwareError::InvalidConfig`] rather than a
    /// silent broadcast of the same logits to every layer — which would make every
    /// layer identical, every Jensen–Shannon divergence zero, and `DoLa`'s layer
    /// selection meaningless while still returning a number.
    pub fn with_rule(
        self,
        required: Vec<impl Into<String>>,
        logits: Vec<f32>,
    ) -> ContextAwareResult<Self> {
        self.with_layered_rule(required, vec![logits])
    }

    /// Add a single-layer rule whose distribution is given as probabilities.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::logits_from_probabilities`] and [`Self::with_rule`].
    pub fn with_probability_rule(
        self,
        required: Vec<impl Into<String>>,
        probabilities: &[f64],
    ) -> ContextAwareResult<Self> {
        let logits = Self::logits_from_probabilities(probabilities)?;
        self.with_rule(required, logits)
    }

    /// Add a rule supplying logits for **every** layer, shallowest first.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidConfig`] when `required` is empty (a
    /// rule that always fires is what [`Self::with_default_layer_logits`] is for)
    /// or when the number of layer vectors is not `num_layers`, and
    /// [`ContextAwareError::VocabSizeMismatch`] when any of them has the wrong
    /// length.
    pub fn with_layered_rule(
        mut self,
        required: Vec<impl Into<String>>,
        layer_logits: Vec<Vec<f32>>,
    ) -> ContextAwareResult<Self> {
        let required_substrings: Vec<String> = required.into_iter().map(Into::into).collect();
        if required_substrings.is_empty() {
            return Err(ContextAwareError::InvalidConfig {
                reason: "a context rule needs at least one required substring".to_string(),
            });
        }
        self.check_layer_logits(&layer_logits, "context rule")?;
        self.rules.push(ContextAwareRule {
            required_substrings,
            layer_logits,
        });
        Ok(self)
    }

    /// Add a layered rule whose per-layer distributions are given as
    /// probabilities.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::logits_from_probabilities`] and
    /// [`Self::with_layered_rule`].
    pub fn with_layered_probability_rule(
        self,
        required: Vec<impl Into<String>>,
        layer_probabilities: &[Vec<f64>],
    ) -> ContextAwareResult<Self> {
        let mut layer_logits = Vec::with_capacity(layer_probabilities.len());
        for probabilities in layer_probabilities {
            layer_logits.push(Self::logits_from_probabilities(probabilities)?);
        }
        self.with_layered_rule(required, layer_logits)
    }

    /// Declare an end-of-sequence token.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::TokenOutOfRange`] when the id is not in the
    /// vocabulary.
    pub fn with_eos_token(mut self, token_id: usize) -> ContextAwareResult<Self> {
        if token_id >= self.token_texts.len() {
            return Err(ContextAwareError::TokenOutOfRange {
                token_id,
                vocab_size: self.token_texts.len(),
            });
        }
        self.eos_token_id = Some(token_id);
        Ok(self)
    }

    /// The token id of a surface string, if present.
    #[must_use]
    pub fn token_id(&self, text: &str) -> Option<usize> {
        self.token_texts
            .iter()
            .position(|candidate| candidate == text)
    }

    /// Encode a continuation into token ids by exact surface match.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::InvalidConfig`] naming the first token text
    /// that is not in the fixture vocabulary.
    pub fn encode(&self, texts: &[&str]) -> ContextAwareResult<Vec<usize>> {
        texts
            .iter()
            .map(|text| {
                self.token_id(text)
                    .ok_or_else(|| ContextAwareError::InvalidConfig {
                        reason: format!("token text {text:?} is not in the fixture vocabulary"),
                    })
            })
            .collect()
    }

    /// Validate a per-layer logit table against the model's shape.
    fn check_layer_logits(&self, layer_logits: &[Vec<f32>], what: &str) -> ContextAwareResult<()> {
        if layer_logits.len() != self.num_layers {
            return Err(ContextAwareError::InvalidConfig {
                reason: format!(
                    "{what}: expected one logit vector per layer ({}), got {}",
                    self.num_layers,
                    layer_logits.len()
                ),
            });
        }
        for (layer, logits) in layer_logits.iter().enumerate() {
            if logits.len() != self.token_texts.len() {
                return Err(ContextAwareError::VocabSizeMismatch {
                    expected: self.token_texts.len(),
                    actual: logits.len(),
                    origin: format!("{what} at layer {layer}"),
                });
            }
        }
        Ok(())
    }

    /// The winning rule for a context, by the specificity order documented on the
    /// type. `None` when nothing matches.
    fn select_rule(&self, context: &str) -> Option<&ContextAwareRule> {
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
                    // `max_by` keeps the last maximum; reverse the index order so
                    // that earliest-inserted wins a full tie.
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(_, rule)| rule)
    }

    /// The single code path behind both trait methods.
    fn logits_at(&self, context: &str, layer: usize) -> ContextAwareResult<Vec<f32>> {
        if layer >= self.num_layers {
            return Err(ContextAwareError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        let table = self
            .select_rule(context)
            .map_or(&self.default_layer_logits, |rule| &rule.layer_logits);
        Ok(table[layer].clone())
    }
}

impl ContextAwareLanguageModel for ContextAwareStaticLanguageModel {
    fn vocab_size(&self) -> usize {
        self.token_texts.len()
    }

    fn next_token_logits(&self, context: &str) -> ContextAwareResult<Vec<f32>> {
        // The mature-layer contract, discharged by construction rather than by
        // convention: there is no second table for the model to disagree with
        // itself from.
        self.logits_at(context, self.num_layers - 1)
    }

    fn decode(&self, token_id: usize) -> ContextAwareResult<String> {
        self.token_texts
            .get(token_id)
            .cloned()
            .ok_or(ContextAwareError::TokenOutOfRange {
                token_id,
                vocab_size: self.token_texts.len(),
            })
    }

    fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id
    }
}

impl LayeredLanguageModel for ContextAwareStaticLanguageModel {
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn layer_logits(&self, context: &str, layer: usize) -> ContextAwareResult<Vec<f32>> {
        self.logits_at(context, layer)
    }
}
