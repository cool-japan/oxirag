//! [`ReplugEngine`] — the `REPLUG` output-distribution ensemble.
//!
//! # The algorithm, in full
//!
//! Given a query `q` and retrieved documents `d_1 … d_k` with retrieval scores
//! `s_1 … s_k`:
//!
//! 1. **Truncate** to the top-`k` documents by score
//!    ([`ReplugConfig::top_k_documents`]).
//! 2. **Weight** them by a temperature-controlled softmax over those scores:
//!
//!    ```text
//!      λ(d_i | q) = softmax(s_i / τ)_i
//!    ```
//!
//!    computed **once**, from `q` alone. λ conditions on the query, not on the
//!    partial generation, so it is fixed for the whole decode — a document does
//!    not get a bigger vote just because the tokens emitted so far happen to
//!    look like it.
//! 3. **Run the LM `k` times, separately.** Each document is prepended to the
//!    query on its own — `d_i ⊕ q` — and produces its own next-token
//!    distribution `p(y | d_i ⊕ q)`. The documents never see each other. This is
//!    the structural difference from every "stuff all the passages into one
//!    prompt" scheme: `k` independent forward passes, `k` independent
//!    distributions, no cross-document attention and no context-length blow-up.
//! 4. **Mix the distributions:**
//!
//!    ```text
//!      p(y | q) = Σ_i λ(d_i | q) · p(y | d_i ⊕ q)
//!    ```
//!
//!    an *arithmetic* mean of distributions — see
//!    [`super::math::log_linear_pool_log_probs`] for why averaging the
//!    **logits** instead computes a *geometric* mean and is a different (and
//!    wrong) operation.
//! 5. **Decode** one token from the mixture, append its text to *every*
//!    document's context, and go back to step 3. Only step 3 onward repeats;
//!    λ from step 2 is reused unchanged.
//!
//! The cost is explicit and unhidden: `k` forward passes per token, reported as
//! [`super::types::ReplugStats::lm_calls`].

use super::math::{
    ReplugRng, arg_max, entropy_from_log_probs, kl_divergence_from_log_probs,
    log_linear_pool_log_probs, mixture_log_probs, promote_logits, temperature_log_softmax,
};
use super::model::ReplugLanguageModel;
use super::types::{
    ReplugConfig, ReplugDecoding, ReplugDocument, ReplugEnsembleOutput, ReplugError, ReplugResult,
    ReplugStats, ReplugStep,
};

/// The `REPLUG` ensemble engine.
///
/// Holds only configuration; the language model is passed in per call, so one
/// engine can serve many models and a model can be shared across engines.
#[derive(Debug, Clone, Default)]
pub struct ReplugEngine {
    /// Temperatures, top-`k`, decoding strategy and `LSR` hyper-parameters.
    pub config: ReplugConfig,
}

impl ReplugEngine {
    /// Create an engine, validating the configuration up front.
    ///
    /// # Errors
    ///
    /// Propagates [`ReplugConfig::validate`].
    pub fn new(config: ReplugConfig) -> ReplugResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    // ── Document selection and weighting ─────────────────────────────────────

    /// Keep the [`ReplugConfig::top_k_documents`] highest-scoring documents.
    ///
    /// Sorted by descending score, ties broken by ascending id so the retained
    /// set — and therefore λ, and therefore the whole generation — is
    /// deterministic for a given input. Truncation happens **before** λ is
    /// computed, so the weights sum to `1` over exactly the documents that
    /// participate.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::NoDocuments`] when `documents` is empty.
    pub fn select_documents(
        &self,
        documents: &[ReplugDocument],
    ) -> ReplugResult<Vec<ReplugDocument>> {
        if documents.is_empty() {
            return Err(ReplugError::NoDocuments);
        }
        for (index, document) in documents.iter().enumerate() {
            if !document.score.is_finite() {
                return Err(ReplugError::NonFiniteLogit {
                    index,
                    value: document.score,
                });
            }
        }

        let mut retained: Vec<ReplugDocument> = documents.to_vec();
        retained.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.id.cmp(&right.id))
        });
        if let Some(top_k) = self.config.top_k_documents {
            retained.truncate(top_k);
        }
        Ok(retained)
    }

    /// `log λ(d_i | q) = log softmax(s_i / τ)_i` for the given documents.
    ///
    /// Returned in log-space because that is what the mixture consumes: a
    /// document whose weight underflows to `0.0` still carries a finite
    /// `log λ ≈ -800`, and forcing it through probability space first would
    /// throw that away for no reason.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::NoDocuments`] for an empty set, and propagates
    /// [`ReplugError::InvalidTemperature`].
    pub fn document_log_weights(&self, documents: &[ReplugDocument]) -> ReplugResult<Vec<f64>> {
        if documents.is_empty() {
            return Err(ReplugError::NoDocuments);
        }
        let scores: Vec<f64> = documents.iter().map(|document| document.score).collect();
        temperature_log_softmax(&scores, self.config.temperature)
    }

    /// `λ(d_i | q) = softmax(s_i / τ)_i`, in probability space.
    ///
    /// `τ → 0⁺` collapses onto the top-scored document; `τ → ∞` becomes
    /// uniform. Both limits are exact — see
    /// [`super::math::temperature_log_softmax`].
    ///
    /// # Errors
    ///
    /// As [`Self::document_log_weights`].
    pub fn document_weights(&self, documents: &[ReplugDocument]) -> ReplugResult<Vec<f64>> {
        Ok(self
            .document_log_weights(documents)?
            .into_iter()
            .map(f64::exp)
            .collect())
    }

    // ── Contexts ─────────────────────────────────────────────────────────────

    /// Assemble one document's context: `d ⊕ separator ⊕ q ⊕ generated`.
    ///
    /// The document goes **first**. That is not cosmetic: with a causal LM and a
    /// KV cache, a prefix that starts with the document lets the query's
    /// attention read it, and — in a production deployment — lets the document's
    /// prefix be cached across queries.
    #[must_use]
    pub fn build_context(&self, document_text: &str, query: &str, generated: &str) -> String {
        let mut context = String::with_capacity(
            document_text.len()
                + self.config.document_separator.len()
                + query.len()
                + generated.len(),
        );
        context.push_str(document_text);
        context.push_str(&self.config.document_separator);
        context.push_str(query);
        context.push_str(generated);
        context
    }

    // ── One ensembled step ───────────────────────────────────────────────────

    /// Fetch, validate and log-normalize each document's next-token
    /// distribution for the given contexts.
    ///
    /// This is where the model's output is checked once and for all: right
    /// length, no `NaN`, no `inf`. Everything downstream may assume finiteness.
    ///
    /// # Errors
    ///
    /// [`ReplugError::VocabSizeMismatch`] when the model contradicts its own
    /// `vocab_size`, [`ReplugError::NonFiniteLogit`] when it emits a `NaN` or
    /// infinity, and whatever the model itself returns.
    fn per_document_log_probs<M: ReplugLanguageModel + ?Sized>(
        model: &M,
        contexts: &[String],
    ) -> ReplugResult<Vec<Vec<f64>>> {
        let vocab_size = model.vocab_size();
        if vocab_size == 0 {
            return Err(ReplugError::EmptyLogits);
        }

        let mut per_document = Vec::with_capacity(contexts.len());
        for (document_index, context) in contexts.iter().enumerate() {
            let logits = model.next_token_logits(context)?;
            if logits.len() != vocab_size {
                return Err(ReplugError::VocabSizeMismatch {
                    expected: vocab_size,
                    actual: logits.len(),
                    document_index,
                });
            }
            let promoted = promote_logits(&logits)?;
            per_document.push(super::math::log_softmax(&promoted));
        }
        Ok(per_document)
    }

    /// The ensembled next-token log-distribution for one step, given the
    /// already-computed `log_weights` and the per-document contexts.
    ///
    /// # Errors
    ///
    /// [`ReplugError::VocabSizeMismatch`] when the model contradicts its own
    /// `vocab_size` or the per-document distributions are ragged,
    /// [`ReplugError::NonFiniteLogit`] when the model emits a `NaN` or infinity,
    /// [`ReplugError::WeightCountMismatch`] when `log_weights` and `contexts`
    /// disagree in length, and whatever the model itself returns.
    pub fn ensemble_log_distribution<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        contexts: &[String],
        log_weights: &[f64],
    ) -> ReplugResult<Vec<f64>> {
        let per_document = Self::per_document_log_probs(model, contexts)?;
        mixture_log_probs(log_weights, &per_document)
    }

    /// The ensembled next-token distribution for a query and its retrieved
    /// documents — one step, no generation.
    ///
    /// The workhorse for inspecting what `REPLUG` actually computes. Returns
    /// `(log p(· | q), λ, retained documents)`.
    ///
    /// # Errors
    ///
    /// As [`Self::select_documents`], [`Self::document_log_weights`] and
    /// [`Self::ensemble_log_distribution`].
    pub fn ensemble_next_token<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
    ) -> ReplugResult<(Vec<f64>, Vec<f64>, Vec<ReplugDocument>)> {
        let retained = self.select_documents(documents)?;
        let log_weights = self.document_log_weights(&retained)?;
        let contexts: Vec<String> = retained
            .iter()
            .map(|document| self.build_context(&document.text, query, ""))
            .collect();
        let log_distribution = self.ensemble_log_distribution(model, &contexts, &log_weights)?;
        let weights = log_weights.iter().copied().map(f64::exp).collect();
        Ok((log_distribution, weights, retained))
    }

    /// The **log-linear (geometric) pool** of the same per-document
    /// distributions — the thing `REPLUG` is *not*.
    ///
    /// Exposed so the difference can be measured rather than taken on faith:
    /// feed the same query and documents to this and to
    /// [`Self::ensemble_next_token`], and the two log-distributions will differ
    /// — and can even disagree on the arg-max, because one pools by **OR** and
    /// the other by **AND**. The derivation is in
    /// [`super::math::log_linear_pool_log_probs`].
    ///
    /// # Errors
    ///
    /// As [`Self::ensemble_next_token`].
    pub fn log_linear_pool_next_token<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
    ) -> ReplugResult<Vec<f64>> {
        let retained = self.select_documents(documents)?;
        let weights = self.document_weights(&retained)?;
        let vocab_size = model.vocab_size();

        let mut per_document_logits = Vec::with_capacity(retained.len());
        for (document_index, document) in retained.iter().enumerate() {
            let context = self.build_context(&document.text, query, "");
            let logits = model.next_token_logits(&context)?;
            if logits.len() != vocab_size {
                return Err(ReplugError::VocabSizeMismatch {
                    expected: vocab_size,
                    actual: logits.len(),
                    document_index,
                });
            }
            per_document_logits.push(promote_logits(&logits)?);
        }

        log_linear_pool_log_probs(&weights, &per_document_logits)
    }

    /// `KL( REPLUG mixture ‖ log-linear pool )` for one step — a scalar measure
    /// of how much the "average the logits" shortcut would have changed the
    /// answer on *this* input.
    ///
    /// # Errors
    ///
    /// As [`Self::ensemble_next_token`] and
    /// [`super::math::kl_divergence_from_log_probs`].
    pub fn pooling_divergence<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
    ) -> ReplugResult<f64> {
        let (mixture, _, _) = self.ensemble_next_token(model, query, documents)?;
        let pooled = self.log_linear_pool_next_token(model, query, documents)?;
        kl_divergence_from_log_probs(&mixture, &pooled)
    }

    // ── Generation ───────────────────────────────────────────────────────────

    /// Run `REPLUG` autoregressively: ensemble, decode a token, extend *every*
    /// document's context with it, repeat.
    ///
    /// # Errors
    ///
    /// As [`Self::ensemble_next_token`], plus [`ReplugError::TokenOutOfRange`]
    /// if the selected token cannot be decoded.
    pub fn generate<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
    ) -> ReplugResult<ReplugEnsembleOutput> {
        self.config.validate()?;

        let retained = self.select_documents(documents)?;
        let log_weights = self.document_log_weights(&retained)?;
        let weights: Vec<f64> = log_weights.iter().copied().map(f64::exp).collect();
        let vocab_size = model.vocab_size();
        let eos = model.eos_token_id();

        // λ is computed once, from `q` and the retrieval scores alone, and is
        // reused unchanged at every step. See the module docs.
        let mut rng = match self.config.decoding {
            ReplugDecoding::Sampling { seed, .. } => Some(ReplugRng::new(seed)),
            ReplugDecoding::Greedy => None,
        };

        let mut generated = String::new();
        let mut token_ids: Vec<usize> = Vec::new();
        let mut steps: Vec<ReplugStep> = Vec::new();
        let mut lm_calls = 0_usize;
        let mut sequence_log_prob = 0.0_f64;
        let mut entropy_total = 0.0_f64;

        for step in 0..self.config.max_tokens {
            let contexts: Vec<String> = retained
                .iter()
                .map(|document| self.build_context(&document.text, query, &generated))
                .collect();

            let log_distribution =
                self.ensemble_log_distribution(model, &contexts, &log_weights)?;
            lm_calls += retained.len();

            let entropy_nats = entropy_from_log_probs(&log_distribution);
            entropy_total += entropy_nats;

            let token_id = self.select_token(&log_distribution, rng.as_mut())?;
            if token_id >= vocab_size {
                return Err(ReplugError::TokenOutOfRange {
                    token_id,
                    vocab_size,
                });
            }

            // Reported under the *mixture*, never under the re-tempered
            // sampling distribution: `token_log_prob` is a property of the
            // model's belief, not of how aggressively we chose to sample from
            // it.
            let token_log_prob = log_distribution[token_id];
            sequence_log_prob += token_log_prob;

            let (distribution, log_distribution_out) = if self.config.record_distributions {
                (
                    Some(log_distribution.iter().copied().map(f64::exp).collect()),
                    Some(log_distribution.clone()),
                )
            } else {
                (None, None)
            };

            steps.push(ReplugStep {
                step,
                distribution,
                log_distribution: log_distribution_out,
                token_id,
                token_log_prob,
                entropy_nats,
            });
            token_ids.push(token_id);

            if Some(token_id) == eos {
                break;
            }

            generated.push_str(&model.decode(token_id)?);
        }

        let stats = Self::build_stats(
            documents.len(),
            &weights,
            vocab_size,
            &token_ids,
            lm_calls,
            sequence_log_prob,
            entropy_total,
        );

        Ok(ReplugEnsembleOutput {
            text: generated,
            token_ids,
            steps,
            document_ids: retained
                .iter()
                .map(|document| document.id.clone())
                .collect(),
            document_weights: weights,
            stats,
        })
    }

    /// Pick a token from the ensembled log-distribution.
    ///
    /// Greedy takes the arg-max. Sampling **re-tempers the mixture** first:
    /// after mixing there are no logits left to divide, only a distribution, so
    /// a sampling temperature `T` means `p_T(y) ∝ p(y)^{1/T}`, which is
    /// `log_softmax(log p / T)` — done in log-space, and via
    /// [`super::math::temperature_log_softmax`] so that a tiny `T` degenerates
    /// to greedy without overflowing on the way.
    fn select_token(
        &self,
        log_distribution: &[f64],
        rng: Option<&mut ReplugRng>,
    ) -> ReplugResult<usize> {
        match (self.config.decoding, rng) {
            (ReplugDecoding::Greedy, _) | (ReplugDecoding::Sampling { .. }, None) => {
                arg_max(log_distribution).ok_or(ReplugError::EmptyLogits)
            }
            (ReplugDecoding::Sampling { temperature, .. }, Some(rng)) => {
                let retempered = temperature_log_softmax(log_distribution, temperature)?;
                rng.sample_from_log_probs(&retempered)
                    .ok_or(ReplugError::EmptyLogits)
            }
        }
    }

    #[allow(clippy::cast_precision_loss)] // Step/token counts, far below 2^53.
    fn build_stats(
        documents_supplied: usize,
        weights: &[f64],
        vocab_size: usize,
        token_ids: &[usize],
        lm_calls: usize,
        sequence_log_prob: f64,
        entropy_total: f64,
    ) -> ReplugStats {
        let generated_tokens = token_ids.len();
        let steps = generated_tokens.max(1) as f64;

        // The λ entropy is computed from λ directly rather than from log λ,
        // because λ here is already materialized; the `p == 0` guard is the
        // same `0 · log 0 = 0` convention as everywhere else in this module.
        let weight_entropy_nats = weights
            .iter()
            .filter(|&&weight| weight > 0.0)
            .map(|&weight| -weight * weight.ln())
            .sum::<f64>()
            .max(0.0);

        ReplugStats {
            documents_supplied,
            documents_retained: weights.len(),
            vocab_size,
            generated_tokens,
            lm_calls,
            sequence_log_prob,
            mean_token_log_prob: if generated_tokens == 0 {
                0.0
            } else {
                sequence_log_prob / steps
            },
            mean_entropy_nats: if generated_tokens == 0 {
                0.0
            } else {
                entropy_total / steps
            },
            weight_entropy_nats,
            max_doc_weight: weights.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            min_doc_weight: weights.iter().copied().fold(f64::INFINITY, f64::min),
        }
    }

    // ── Teacher forcing (shared with LSR) ────────────────────────────────────

    /// `log P_LM(y* | context_prefix)` — the teacher-forced sequence
    /// log-likelihood of a ground-truth continuation under **one** context.
    ///
    /// ```text
    ///   log P_LM(y* | c) = Σ_t log p(y*_t | c ⊕ y*_{<t})
    /// ```
    ///
    /// Each term comes from a [`super::math::log_softmax`], so the sum is a sum
    /// of finite negatives and stays finite even when the individual token
    /// probabilities underflow — which, over a continuation of any length, they
    /// eventually will. Computing this as `ln(Π_t p_t)` instead would underflow
    /// the product to `0` after a few dozen tokens and report `-inf` for every
    /// document, making the whole `LSR` signal vanish. This is the single most
    /// common way to silently break `REPLUG`-`LSR`.
    ///
    /// # Errors
    ///
    /// [`ReplugError::EmptyTarget`] for an empty `y*`,
    /// [`ReplugError::TokenOutOfRange`] for an out-of-vocabulary target token,
    /// and whatever the model returns.
    pub fn sequence_log_likelihood<M: ReplugLanguageModel + ?Sized>(
        model: &M,
        context_prefix: &str,
        target_token_ids: &[usize],
    ) -> ReplugResult<f64> {
        if target_token_ids.is_empty() {
            return Err(ReplugError::EmptyTarget);
        }
        let vocab_size = model.vocab_size();

        let mut context = context_prefix.to_string();
        let mut total = 0.0_f64;

        for &target in target_token_ids {
            if target >= vocab_size {
                return Err(ReplugError::TokenOutOfRange {
                    token_id: target,
                    vocab_size,
                });
            }
            let logits = model.next_token_logits(&context)?;
            if logits.len() != vocab_size {
                return Err(ReplugError::VocabSizeMismatch {
                    expected: vocab_size,
                    actual: logits.len(),
                    document_index: 0,
                });
            }
            let promoted = promote_logits(&logits)?;
            let log_probs = super::math::log_softmax(&promoted);
            total += log_probs[target];

            // Teacher forcing: the *ground truth* token is appended, not the
            // model's own preference.
            context.push_str(&model.decode(target)?);
        }

        Ok(total)
    }

    /// `log P_LM(y* | q)` under the **ensembled** distribution — the quantity
    /// `REPLUG` actually improves, as opposed to the per-document
    /// likelihoods that [`Self::sequence_log_likelihood`] measures.
    ///
    /// At each step the mixture is formed over all retained documents (with the
    /// same fixed λ), the ground-truth token's ensembled log-probability is
    /// accumulated, and that token is teacher-forced into every document's
    /// context.
    ///
    /// This is the honest way to compare "the ensemble" against "the best single
    /// document": both are log-likelihoods of the *same* `y*`, under the same
    /// frozen LM.
    ///
    /// # Errors
    ///
    /// As [`Self::ensemble_next_token`] and [`Self::sequence_log_likelihood`].
    pub fn ensemble_target_log_likelihood<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
        target_token_ids: &[usize],
    ) -> ReplugResult<f64> {
        if target_token_ids.is_empty() {
            return Err(ReplugError::EmptyTarget);
        }
        let retained = self.select_documents(documents)?;
        let log_weights = self.document_log_weights(&retained)?;
        let vocab_size = model.vocab_size();

        let mut generated = String::new();
        let mut total = 0.0_f64;

        for &target in target_token_ids {
            if target >= vocab_size {
                return Err(ReplugError::TokenOutOfRange {
                    token_id: target,
                    vocab_size,
                });
            }
            let contexts: Vec<String> = retained
                .iter()
                .map(|document| self.build_context(&document.text, query, &generated))
                .collect();
            let log_distribution =
                self.ensemble_log_distribution(model, &contexts, &log_weights)?;
            total += log_distribution[target];
            generated.push_str(&model.decode(target)?);
        }

        Ok(total)
    }
}
