//! The three decoders: [`ContextAwareDecoder`], [`ContrastiveDecoder`] and
//! [`DoLaDecoder`].
//!
//! They share one kernel and differ only in where their two distributions come
//! from. Everything below the dashed line — combine with a negative weight,
//! constrain to the plausible set, renormalize, select a token, extend the
//! context, repeat — is literally the same code for all three; see
//! [`super::math::masked_contrast_log_probs`] and `run_generation`.
//!
//! ```text
//!   CAD          one model,  two prompts   :  p(y | c ⊕ x)   vs  p(y | x)
//!   contrastive  two models, one prompt    :  p_exp(y | x)   vs  p_ama(y | x)
//!   DoLa         one model,  two depths    :  q_N(y | x)     vs  q_M(y | x)
//!   ────────────────────────────────────────────────────────────────────────
//!   shared       s = w⁺ v⁺ + w⁻ v⁻ ;  s[y ∉ V_valid] = -inf ;  log_softmax(s)
//! ```

use crate::replug::ReplugError;
use crate::replug::math::{
    ReplugRng, arg_max, log_softmax, promote_logits, temperature_log_softmax,
};

use super::math::{jensen_shannon_divergence_from_log_probs, masked_contrast_log_probs};
use super::model::{ContextAwareLanguageModel, LayeredLanguageModel};
use super::types::{
    CadConfig, ContextAwareConfig, ContextAwareError, ContextAwareOutput, ContextAwareResult,
    ContextAwareStats, ContextAwareStep, ContrastOutcome, ContrastiveConfig, DecodeMode,
    DecodingStrategy, DolaConfig, LayerSelector, PrematureLayer,
};

// ── Shared plumbing ──────────────────────────────────────────────────────────

/// Promote a model's `f32` logits to `f64`, naming which distribution they came
/// from if they are unusable.
///
/// Reuses `REPLUG`'s [`promote_logits`], which is the one place a `NaN` or an
/// infinity is caught, and re-labels its error with the contrast-specific origin
/// ("with-context prompt", "amateur model", "layer 12") — because in a contrast
/// there are always *two* distributions and "the model emitted a `NaN`" is only
/// half an answer.
fn promote(logits: &[f32], origin: &str) -> ContextAwareResult<Vec<f64>> {
    promote_logits(logits).map_err(|error| match error {
        ReplugError::NonFiniteLogit { index, value } => ContextAwareError::NonFiniteScore {
            index,
            origin: origin.to_string(),
            value,
        },
        ReplugError::EmptyLogits => ContextAwareError::EmptyLogits,
        other => ContextAwareError::Math(other),
    })
}

/// Fetch, validate and promote one next-token logit vector.
fn scores_for<M: ContextAwareLanguageModel + ?Sized>(
    model: &M,
    context: &str,
    origin: &str,
) -> ContextAwareResult<Vec<f64>> {
    let vocab_size = model.vocab_size();
    if vocab_size == 0 {
        return Err(ContextAwareError::EmptyLogits);
    }
    let logits = model.next_token_logits(context)?;
    if logits.len() != vocab_size {
        return Err(ContextAwareError::VocabSizeMismatch {
            expected: vocab_size,
            actual: logits.len(),
            origin: origin.to_string(),
        });
    }
    promote(&logits, origin)
}

/// Fetch, validate and promote one *layer's* logit vector.
fn layer_scores_for<M: LayeredLanguageModel + ?Sized>(
    model: &M,
    context: &str,
    layer: usize,
) -> ContextAwareResult<Vec<f64>> {
    let vocab_size = model.vocab_size();
    if vocab_size == 0 {
        return Err(ContextAwareError::EmptyLogits);
    }
    let logits = model.layer_logits(context, layer)?;
    if logits.len() != vocab_size {
        return Err(ContextAwareError::VocabSizeMismatch {
            expected: vocab_size,
            actual: logits.len(),
            origin: format!("layer {layer}"),
        });
    }
    promote(&logits, &format!("layer {layer}"))
}

/// Combine, constrain, renormalize, and record both inputs.
///
/// The single point through which all three decoders pass, so that "which
/// distribution the plausibility mask is read off" is a property of the *module*
/// and not of each method's copy of the loop. It is read off the **positive**
/// one, always.
fn build_outcome(
    mode: DecodeMode,
    positive: &[f64],
    negative: &[f64],
    weights: (f64, f64),
    plausibility_alpha: f64,
    premature_layer: Option<PrematureLayer>,
) -> ContextAwareResult<ContrastOutcome> {
    let (log_probs, plausible) =
        masked_contrast_log_probs(positive, negative, weights.0, weights.1, plausibility_alpha)?;

    Ok(ContrastOutcome {
        mode,
        log_probs,
        positive_log_probs: log_softmax(positive),
        negative_log_probs: log_softmax(negative),
        plausible,
        premature_layer,
    })
}

/// Pick a token from the contrasted distribution.
///
/// Greedy takes the arg-max. Sampling re-tempers first: `p_T(y) ∝ p(y)^{1/T}`,
/// which is `log_softmax(log p / T)`, done in log-space via
/// [`temperature_log_softmax`] so that a tiny `T` degenerates to greedy without
/// overflowing on the way. Tokens the plausibility constraint removed carry
/// `log p = -inf`, so `exp` of them is exactly `0` and the inverse-CDF search can
/// never land on one — the constraint survives re-tempering, which is the whole
/// reason it is applied as a hard mask rather than as a penalty.
fn select_token(
    log_probs: &[f64],
    strategy: DecodingStrategy,
    rng: Option<&mut ReplugRng>,
) -> ContextAwareResult<usize> {
    match (strategy, rng) {
        (DecodingStrategy::Greedy, _) | (DecodingStrategy::Sampling { .. }, None) => {
            arg_max(log_probs).ok_or(ContextAwareError::EmptyLogits)
        }
        (DecodingStrategy::Sampling { temperature, .. }, Some(rng)) => {
            let retempered = temperature_log_softmax(log_probs, temperature)?;
            rng.sample_from_log_probs(&retempered)
                .ok_or(ContextAwareError::EmptyLogits)
        }
    }
}

/// The autoregressive loop, shared by all three decoders.
///
/// `step` is handed the text generated so far and returns that step's contrast.
/// Everything the loop does with it — entropy, plausible-set size, token
/// selection, book-keeping, `EOS` — is method-independent, which is exactly the
/// claim this module makes about the three papers.
fn run_generation<M, F>(
    shared: &ContextAwareConfig,
    mode: DecodeMode,
    model: &M,
    lm_calls_per_step: usize,
    mut step: F,
) -> ContextAwareResult<ContextAwareOutput>
where
    M: ContextAwareLanguageModel + ?Sized,
    F: FnMut(&str) -> ContextAwareResult<ContrastOutcome>,
{
    shared.validate()?;
    let vocab_size = model.vocab_size();
    let eos = model.eos_token_id();

    let mut rng = match shared.strategy {
        DecodingStrategy::Sampling { seed, .. } => Some(ReplugRng::new(seed)),
        DecodingStrategy::Greedy => None,
    };

    let mut generated = String::new();
    let mut token_ids: Vec<usize> = Vec::new();
    let mut steps: Vec<ContextAwareStep> = Vec::new();
    let mut lm_calls = 0_usize;
    let mut sequence_log_prob = 0.0_f64;
    let mut positive_sequence_log_prob = 0.0_f64;
    let mut entropy_total = 0.0_f64;
    let mut plausible_total = 0_usize;
    let mut truncated_steps = 0_usize;

    for step_index in 0..shared.max_tokens {
        let outcome = step(&generated)?;
        lm_calls += lm_calls_per_step;

        let entropy_nats = outcome.entropy_nats();
        entropy_total += entropy_nats;
        let plausible_tokens = outcome.plausible_count();
        plausible_total += plausible_tokens;
        if outcome.truncated() {
            truncated_steps += 1;
        }

        let token_id = select_token(&outcome.log_probs, shared.strategy, rng.as_mut())?;
        if token_id >= vocab_size {
            return Err(ContextAwareError::TokenOutOfRange {
                token_id,
                vocab_size,
            });
        }

        let token_log_prob = outcome.log_probs[token_id];
        let positive_token_log_prob = outcome.positive_log_probs[token_id];
        let negative_token_log_prob = outcome.negative_log_probs[token_id];
        sequence_log_prob += token_log_prob;
        positive_sequence_log_prob += positive_token_log_prob;

        let (log_distribution, positive_log_distribution) = if shared.record_distributions {
            (
                Some(outcome.log_probs.clone()),
                Some(outcome.positive_log_probs.clone()),
            )
        } else {
            (None, None)
        };

        steps.push(ContextAwareStep {
            step: step_index,
            token_id,
            token_log_prob,
            positive_token_log_prob,
            negative_token_log_prob,
            entropy_nats,
            plausible_tokens,
            premature_layer: outcome.premature_layer,
            log_distribution,
            positive_log_distribution,
        });
        token_ids.push(token_id);

        if Some(token_id) == eos {
            break;
        }
        generated.push_str(&model.decode(token_id)?);
    }

    let stats = build_stats(
        mode,
        vocab_size,
        &token_ids,
        lm_calls,
        (sequence_log_prob, positive_sequence_log_prob),
        (entropy_total, plausible_total),
        truncated_steps,
    );

    Ok(ContextAwareOutput {
        text: generated,
        token_ids,
        steps,
        stats,
    })
}

#[allow(clippy::cast_precision_loss)] // Step/token counts, far below 2^53.
fn build_stats(
    mode: DecodeMode,
    vocab_size: usize,
    token_ids: &[usize],
    lm_calls: usize,
    log_probs: (f64, f64),
    totals: (f64, usize),
    truncated_steps: usize,
) -> ContextAwareStats {
    let (sequence_log_prob, positive_sequence_log_prob) = log_probs;
    let (entropy_total, plausible_total) = totals;
    let generated_tokens = token_ids.len();
    let divisor = generated_tokens.max(1) as f64;

    ContextAwareStats {
        mode,
        vocab_size,
        generated_tokens,
        lm_calls,
        sequence_log_prob,
        positive_sequence_log_prob,
        mean_token_log_prob: if generated_tokens == 0 {
            0.0
        } else {
            sequence_log_prob / divisor
        },
        mean_entropy_nats: if generated_tokens == 0 {
            0.0
        } else {
            entropy_total / divisor
        },
        mean_plausible_tokens: if generated_tokens == 0 {
            0.0
        } else {
            plausible_total as f64 / divisor
        },
        truncated_steps,
    }
}

// ── CAD ──────────────────────────────────────────────────────────────────────

/// **Context-aware decoding** (Shi et al., 2024): one model, two prompts.
///
/// The model is run twice on the same query — once with the retrieved context in
/// front of it and once without — and the two next-token distributions are
/// extrapolated apart:
///
/// ```text
///   score(y)  =  (1 + alpha) · z(y | c ⊕ x)  −  alpha · z(y | x)
/// ```
///
/// What the negative pass buys is a *measurement of the model's prior*. Its
/// answer to the bare question is what it would have said from memory alone, and
/// subtracting it leaves only what the context contributed. The failure this
/// addresses is specific and is the one that makes retrieval-augmented systems
/// untrustworthy: a model that has memorized a fact will keep asserting it even
/// when the retrieved passage says otherwise, because the retrieved passage is
/// worth a few logits and the memory is worth many.
///
/// Holds only configuration; the model is passed in per call.
#[derive(Debug, Clone, Default)]
pub struct ContextAwareDecoder {
    /// Contrast strength, prompt assembly, and the shared decode-time settings.
    pub config: CadConfig,
}

impl ContextAwareDecoder {
    /// Create a decoder, validating the configuration up front.
    ///
    /// # Errors
    ///
    /// Propagates [`CadConfig::validate`].
    pub fn new(config: CadConfig) -> ContextAwareResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// The **positive** prompt: `c ⊕ separator ⊕ x ⊕ generated`.
    ///
    /// The context goes first, exactly as in `crate::replug::ReplugEngine` — with
    /// a causal model and a `KV` cache, a prefix that starts with the document
    /// lets the query's attention read it, and lets the document's prefix be
    /// cached across queries.
    #[must_use]
    pub fn positive_context(&self, context: &str, query: &str, generated: &str) -> String {
        let mut prompt = String::with_capacity(
            context.len() + self.config.context_separator.len() + query.len() + generated.len(),
        );
        prompt.push_str(context);
        prompt.push_str(&self.config.context_separator);
        prompt.push_str(query);
        prompt.push_str(generated);
        prompt
    }

    /// The **negative** prompt: `x ⊕ generated`, with the context removed.
    ///
    /// The generated prefix is *kept*. That is not an oversight: the two
    /// distributions must be next-token distributions for the **same** position in
    /// the same sequence, or their difference is not a difference of anything. The
    /// only thing that differs between the two passes is the presence of `c`.
    #[must_use]
    pub fn negative_context(&self, query: &str, generated: &str) -> String {
        let mut prompt = String::with_capacity(query.len() + generated.len());
        prompt.push_str(query);
        prompt.push_str(generated);
        prompt
    }

    /// One contrasted step, continuing an existing generated prefix.
    ///
    /// # Errors
    ///
    /// [`ContextAwareError::VocabSizeMismatch`] when the model contradicts its own
    /// `vocab_size`, [`ContextAwareError::NonFiniteScore`] when it emits a `NaN`
    /// or an infinity, and whatever the model itself returns.
    pub fn next_token_with_prefix<M: ContextAwareLanguageModel + ?Sized>(
        &self,
        model: &M,
        context: &str,
        query: &str,
        generated: &str,
    ) -> ContextAwareResult<ContrastOutcome> {
        let positive_prompt = self.positive_context(context, query, generated);
        let negative_prompt = self.negative_context(query, generated);

        let positive = scores_for(model, &positive_prompt, "with-context prompt")?;
        let negative = scores_for(model, &negative_prompt, "context-free prompt")?;

        build_outcome(
            DecodeMode::ContextAware,
            &positive,
            &negative,
            self.config.contrast_weights(),
            self.config.shared.plausibility_alpha,
            None,
        )
    }

    /// The contrasted next-token distribution for a context and a query — one
    /// step, no generation.
    ///
    /// # Errors
    ///
    /// As [`Self::next_token_with_prefix`].
    pub fn next_token<M: ContextAwareLanguageModel + ?Sized>(
        &self,
        model: &M,
        context: &str,
        query: &str,
    ) -> ContextAwareResult<ContrastOutcome> {
        self.next_token_with_prefix(model, context, query, "")
    }

    /// Generate autoregressively under the contrast.
    ///
    /// Two logit queries per token: the with-context prompt and the context-free
    /// one.
    ///
    /// # Errors
    ///
    /// Propagates [`CadConfig::validate`] and [`Self::next_token_with_prefix`].
    pub fn generate<M: ContextAwareLanguageModel + ?Sized>(
        &self,
        model: &M,
        context: &str,
        query: &str,
    ) -> ContextAwareResult<ContextAwareOutput> {
        self.config.validate()?;
        run_generation(
            &self.config.shared,
            DecodeMode::ContextAware,
            model,
            2,
            |generated| self.next_token_with_prefix(model, context, query, generated),
        )
    }
}

// ── Contrastive decoding ─────────────────────────────────────────────────────

/// **Contrastive decoding** (Li et al., 2023): two models, one prompt.
///
/// ```text
///   score(y)  =  log p_expert(y | x)  −  beta · log p_amateur(y | x)
///   V_valid   =  { y : p_expert(y) ≥ alpha_plaus · max_{y'} p_expert(y') }
/// ```
///
/// The amateur is a *smaller model of the same family*, and the premise is that
/// the failure modes of language models — repetition, genericness, the
/// short-sighted grab for the locally likely token — are **shared** across scale,
/// while competence is not. Subtracting the amateur therefore removes what both
/// models want for bad reasons and keeps what only the expert wants.
///
/// # The plausibility constraint is not optional here
///
/// The log-ratio `log(p_exp / p_ama)` is maximized by tokens the *amateur* finds
/// impossible, and a small model finds a great many things impossible for no good
/// reason at all — rare tokens, proper nouns, anything outside its capacity. Left
/// unconstrained, contrastive decoding does not select the token the expert is
/// confident about; it selects the token the amateur has never heard of. Li et
/// al.'s adaptive plausibility constraint, which this module applies by default at
/// `alpha_plaus = 0.1`, is what makes the objective usable, and the module's tests
/// exhibit exactly what removing it does.
#[derive(Debug, Clone, Default)]
pub struct ContrastiveDecoder {
    /// Amateur weight, amateur temperature, and the shared decode-time settings.
    pub config: ContrastiveConfig,
}

impl ContrastiveDecoder {
    /// Create a decoder, validating the configuration up front.
    ///
    /// # Errors
    ///
    /// Propagates [`ContrastiveConfig::validate`].
    pub fn new(config: ContrastiveConfig) -> ContextAwareResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// One contrasted step, continuing an existing generated prefix.
    ///
    /// # Errors
    ///
    /// [`ContextAwareError::VocabSizeMismatch`] when the two models disagree about
    /// the size of the vocabulary — they must share a tokenizer, or their
    /// distributions are not defined over the same set and their difference means
    /// nothing — or when either contradicts its own `vocab_size`;
    /// [`ContextAwareError::NonFiniteScore`] when either emits a `NaN`; and
    /// whatever the models themselves return.
    pub fn next_token_with_prefix<E, A>(
        &self,
        expert: &E,
        amateur: &A,
        prompt: &str,
        generated: &str,
    ) -> ContextAwareResult<ContrastOutcome>
    where
        E: ContextAwareLanguageModel + ?Sized,
        A: ContextAwareLanguageModel + ?Sized,
    {
        if expert.vocab_size() != amateur.vocab_size() {
            return Err(ContextAwareError::VocabSizeMismatch {
                expected: expert.vocab_size(),
                actual: amateur.vocab_size(),
                origin: "amateur model (expert and amateur must share a tokenizer)".to_string(),
            });
        }

        let context = format!("{prompt}{generated}");
        let expert_scores = scores_for(expert, &context, "expert model")?;
        let amateur_logits = scores_for(amateur, &context, "amateur model")?;

        // The amateur's temperature genuinely reshapes its distribution — unlike a
        // renormalization, which the final `log_softmax` would absorb — so it must
        // be applied here, before the subtraction. At `tau = 1` it reduces to a
        // renormalization and is therefore inert.
        let amateur_scores =
            temperature_log_softmax(&amateur_logits, self.config.amateur_temperature)?;

        build_outcome(
            DecodeMode::Contrastive,
            &expert_scores,
            &amateur_scores,
            self.config.contrast_weights(),
            self.config.shared.plausibility_alpha,
            None,
        )
    }

    /// The contrasted next-token distribution for one prompt — one step, no
    /// generation.
    ///
    /// # Errors
    ///
    /// As [`Self::next_token_with_prefix`].
    pub fn next_token<E, A>(
        &self,
        expert: &E,
        amateur: &A,
        prompt: &str,
    ) -> ContextAwareResult<ContrastOutcome>
    where
        E: ContextAwareLanguageModel + ?Sized,
        A: ContextAwareLanguageModel + ?Sized,
    {
        self.next_token_with_prefix(expert, amateur, prompt, "")
    }

    /// Generate autoregressively under the contrast.
    ///
    /// Two logit queries per token, one from each model. The token stream, the
    /// `EOS` and the vocabulary all come from the **expert** — the amateur is a
    /// critic, never an author.
    ///
    /// # Errors
    ///
    /// Propagates [`ContrastiveConfig::validate`] and
    /// [`Self::next_token_with_prefix`].
    pub fn generate<E, A>(
        &self,
        expert: &E,
        amateur: &A,
        prompt: &str,
    ) -> ContextAwareResult<ContextAwareOutput>
    where
        E: ContextAwareLanguageModel + ?Sized,
        A: ContextAwareLanguageModel + ?Sized,
    {
        self.config.validate()?;
        run_generation(
            &self.config.shared,
            DecodeMode::Contrastive,
            expert,
            2,
            |generated| self.next_token_with_prefix(expert, amateur, prompt, generated),
        )
    }
}

// ── DoLa ─────────────────────────────────────────────────────────────────────

/// **`DoLa`** (Chuang et al., 2024): one model, one prompt, two depths.
///
/// ```text
///   M         =  arg max_{j ∈ candidates}  JSD( q_N(· | x) ‖ q_j(· | x) )
///   score(y)  =  log q_N(y | x)  −  log q_M(y | x)
///   V_valid   =  { y : q_N(y) ≥ alpha_plaus · max_{y'} q_N(y') }
/// ```
///
/// where `q_j` is the distribution obtained by projecting layer `j`'s hidden state
/// through the model's own output head — the "logit lens". No second model and no
/// second prompt: `DoLa` contrasts the network against **an earlier version of
/// itself**, on the same forward pass.
///
/// The premise is a claim about where factual knowledge lives. Chuang et al.
/// observe that a model's distribution over *function* words and syntactic
/// continuations is settled in its lower layers and barely changes afterwards,
/// while its distribution over *factual* tokens keeps moving all the way to the
/// top. Subtracting a lower layer therefore cancels the part of the prediction
/// that was never in doubt and leaves the part the upper layers actually
/// contributed — which, on a factual token, is the fact.
///
/// The premature layer is chosen **per token** by maximum Jensen–Shannon
/// divergence from the mature layer, on the reasoning that the layer that
/// disagrees most with the final answer is the layer across which the most work
/// was done. Different tokens of one generation contrast against different layers,
/// and [`PrematureLayer::considered`] records the whole curve so the choice can be
/// audited rather than trusted.
#[derive(Debug, Clone, Default)]
pub struct DoLaDecoder {
    /// Mature layer, premature-layer selector, and the shared decode-time
    /// settings.
    pub config: DolaConfig,
}

impl DoLaDecoder {
    /// Create a decoder, validating the configuration up front.
    ///
    /// Layer indices cannot be validated here — they need the model's depth, and
    /// are checked by [`Self::resolve_candidates`] on first use.
    ///
    /// # Errors
    ///
    /// Propagates [`DolaConfig::validate`].
    pub fn new(config: DolaConfig) -> ContextAwareResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Which layer this configuration treats as mature, for this model.
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::LayerOutOfRange`] when an explicit
    /// [`DolaConfig::mature_layer`] is outside the model's stack, and
    /// [`ContextAwareError::InvalidConfig`] for a zero-layer model.
    pub fn resolve_mature_layer<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
    ) -> ContextAwareResult<usize> {
        let num_layers = model.num_layers();
        if num_layers == 0 {
            return Err(ContextAwareError::InvalidConfig {
                reason: "a layered model must report at least one layer".to_string(),
            });
        }
        let mature = self.config.mature_layer.unwrap_or(num_layers - 1);
        if mature >= num_layers {
            return Err(ContextAwareError::LayerOutOfRange {
                layer: mature,
                num_layers,
            });
        }
        Ok(mature)
    }

    /// The premature-layer candidates for this model, sorted ascending and
    /// deduplicated.
    ///
    /// An empty [`LayerSelector::MaxJensenShannon`] candidate list expands to
    /// "every layer except the mature one".
    ///
    /// # Errors
    ///
    /// Returns [`ContextAwareError::LayerOutOfRange`] for a candidate outside the
    /// stack, [`ContextAwareError::MatureLayerInCandidates`] when the mature layer
    /// is also a candidate (contrasting a layer against itself yields the uniform
    /// distribution — silently), and [`ContextAwareError::EmptyCandidateSet`] when
    /// nothing is left to choose from, which is what a single-layer model gives
    /// you.
    pub fn resolve_candidates<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
    ) -> ContextAwareResult<Vec<usize>> {
        let num_layers = model.num_layers();
        let mature = self.resolve_mature_layer(model)?;

        let candidates: Vec<usize> = match &self.config.selector {
            LayerSelector::Fixed(layer) => vec![*layer],
            LayerSelector::MaxJensenShannon { candidates } if candidates.is_empty() => {
                (0..num_layers).filter(|&layer| layer != mature).collect()
            }
            LayerSelector::MaxJensenShannon { candidates } => {
                let mut sorted = candidates.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted
            }
        };

        if candidates.is_empty() {
            return Err(ContextAwareError::EmptyCandidateSet);
        }
        for &layer in &candidates {
            if layer >= num_layers {
                return Err(ContextAwareError::LayerOutOfRange { layer, num_layers });
            }
            if layer == mature {
                return Err(ContextAwareError::MatureLayerInCandidates { layer });
            }
        }
        Ok(candidates)
    }

    /// `arg max_j JSD( q_mature ‖ q_j )`, with the full divergence curve.
    ///
    /// Ties go to the **lowest** layer index, which makes the selection total and
    /// therefore reproducible; the paper does not specify a rule, and a
    /// nondeterministic one would make a `DoLa` generation irreproducible for no
    /// benefit.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve_candidates`], plus
    /// [`super::math::jensen_shannon_divergence_from_log_probs`] and whatever the
    /// model returns.
    pub fn select_premature_layer<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
        context: &str,
    ) -> ContextAwareResult<PrematureLayer> {
        let mature = self.resolve_mature_layer(model)?;
        let candidates = self.resolve_candidates(model)?;
        let mature_log_probs = log_softmax(&layer_scores_for(model, context, mature)?);

        let mut considered = Vec::with_capacity(candidates.len());
        for &layer in &candidates {
            let layer_log_probs = log_softmax(&layer_scores_for(model, context, layer)?);
            considered.push((
                layer,
                jensen_shannon_divergence_from_log_probs(&mature_log_probs, &layer_log_probs)?,
            ));
        }

        Self::choose(considered).map(|(_, premature)| premature)
    }

    /// Pick the maximally-divergent candidate, returning its position in the
    /// candidate list alongside the record.
    fn choose(considered: Vec<(usize, f64)>) -> ContextAwareResult<(usize, PrematureLayer)> {
        let divergences: Vec<f64> = considered
            .iter()
            .map(|&(_, divergence)| divergence)
            .collect();
        // `arg_max` returns the *first* maximum, and `considered` is in ascending
        // layer order, so a tie resolves to the lowest layer.
        let position = arg_max(&divergences).ok_or(ContextAwareError::EmptyCandidateSet)?;
        let (layer, divergence_nats) = considered[position];
        Ok((
            position,
            PrematureLayer {
                layer,
                divergence_nats,
                considered,
            },
        ))
    }

    /// One contrasted step, continuing an existing generated prefix.
    ///
    /// # Errors
    ///
    /// As [`Self::select_premature_layer`], plus
    /// [`ContextAwareError::VocabSizeMismatch`] when a layer contradicts the
    /// model's `vocab_size` and [`ContextAwareError::NonFiniteScore`] when one
    /// emits a `NaN`.
    pub fn next_token_with_prefix<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
        prompt: &str,
        generated: &str,
    ) -> ContextAwareResult<ContrastOutcome> {
        let context = format!("{prompt}{generated}");
        let mature = self.resolve_mature_layer(model)?;
        let candidates = self.resolve_candidates(model)?;

        let mature_scores = layer_scores_for(model, &context, mature)?;
        let mature_log_probs = log_softmax(&mature_scores);

        let mut considered = Vec::with_capacity(candidates.len());
        let mut candidate_scores: Vec<Vec<f64>> = Vec::with_capacity(candidates.len());
        for &layer in &candidates {
            let scores = layer_scores_for(model, &context, layer)?;
            let divergence =
                jensen_shannon_divergence_from_log_probs(&mature_log_probs, &log_softmax(&scores))?;
            considered.push((layer, divergence));
            candidate_scores.push(scores);
        }

        let (position, premature) = Self::choose(considered)?;

        build_outcome(
            DecodeMode::Dola,
            &mature_scores,
            &candidate_scores[position],
            self.config.contrast_weights(),
            self.config.shared.plausibility_alpha,
            Some(premature),
        )
    }

    /// The contrasted next-token distribution for one prompt — one step, no
    /// generation.
    ///
    /// # Errors
    ///
    /// As [`Self::next_token_with_prefix`].
    pub fn next_token<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
        prompt: &str,
    ) -> ContextAwareResult<ContrastOutcome> {
        self.next_token_with_prefix(model, prompt, "")
    }

    /// Generate autoregressively under the contrast.
    ///
    /// `1 + |candidates|` logit queries per token — but **one** forward pass, on a
    /// backend that reads its layers out of the pass it already ran. See the note
    /// on [`ContextAwareStats::lm_calls`].
    ///
    /// # Errors
    ///
    /// Propagates [`DolaConfig::validate`] and [`Self::next_token_with_prefix`].
    pub fn generate<M: LayeredLanguageModel + ?Sized>(
        &self,
        model: &M,
        prompt: &str,
    ) -> ContextAwareResult<ContextAwareOutput> {
        self.config.validate()?;
        let queries_per_step = 1 + self.resolve_candidates(model)?.len();
        run_generation(
            &self.config.shared,
            DecodeMode::Dola,
            model,
            queries_per_step,
            |generated| self.next_token_with_prefix(model, prompt, generated),
        )
    }
}
