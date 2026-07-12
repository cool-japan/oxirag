//! `REPLUG`-`LSR` — **LM-Supervised Retrieval**: using the frozen language
//! model as the teacher, and the retriever as the student.
//!
//! # The idea
//!
//! The ensemble in [`super::ensemble`] takes the retriever's scores as given.
//! But the retriever was trained on *its own* objective (contrastive similarity,
//! usually) and nobody ever asked whether the documents it likes are the
//! documents that actually make the **LM** better at the task. `LSR` closes that
//! loop. The LM stays frozen — it is a black box we are not allowed to fine-tune
//! — and its *behaviour* becomes the supervision signal for the retriever.
//!
//! # The two distributions
//!
//! Over the retrieved set `{d_1 … d_k}`:
//!
//! - **What the retriever believes.** Its scores, softmaxed:
//!
//!   ```text
//!     P_R(d_i | q) = softmax(s(q, d_i) / τ)_i
//!   ```
//!
//!   This is literally the same λ the ensemble uses — the retrieval likelihood
//!   *is* the mixture weight, which is what makes the loop close.
//!
//! - **What the LM's behaviour implies it should believe.** Take a ground-truth
//!   continuation `y*` and ask each document, one at a time, how much
//!   probability the frozen LM assigns to `y*` when *that* document is in
//!   context:
//!
//!   ```text
//!     Q_LM(d_i | q) = softmax( log P_LM(y* | d_i ⊕ q) / β )_i
//!   ```
//!
//!   A document that made `y*` likely gets a high `Q`; a document that made no
//!   difference — or actively distracted the model — gets a low one. Note this
//!   is a *relative* judgment across the retrieved set, not an absolute one:
//!   `Q` is a softmax, so it says which documents helped **more than the
//!   others**, which is exactly the kind of signal a ranker can consume.
//!
//! # The loss and its gradient (derived)
//!
//! Minimize the divergence from the LM's preference to the retriever's:
//!
//! ```text
//!   L = KL( Q_LM ‖ P_R ) = Σ_i Q_i · (log Q_i − log P_i)
//! ```
//!
//! `Q` is the *target* and is held fixed (it depends only on the frozen LM). The
//! gradient with respect to a retrieval score `s_j` is short and worth writing
//! out, because its **sign** is the whole mechanism:
//!
//! ```text
//!   L            = Σ_i Q_i log Q_i  −  Σ_i Q_i log P_i
//!                = const(Q)         −  Σ_i Q_i log P_i
//!
//!   log P_i      = s_i / τ  −  logsumexp_m( s_m / τ )
//!
//!   ∂ log P_i    1
//!   ───────   =  ─ · ( δ_ij  −  P_j )
//!   ∂ s_j        τ
//!
//!   ∂L           1                             1
//!   ───   =  − ─ · Σ_i Q_i ( δ_ij − P_j )  = − ─ · ( Q_j − P_j · Σ_i Q_i )
//!   ∂s_j        τ                              τ
//!
//!                                          P_j − Q_j
//!                                       =  ─────────        (since Σ_i Q_i = 1)
//!                                              τ
//! ```
//!
//! So the gradient is the classic softmax cross-entropy residual `(P − Q) / τ`,
//! and a descent step
//!
//! ```text
//!   s_j  ←  s_j  −  η · (P_j − Q_j) / τ
//! ```
//!
//! **raises** the score of any document the LM valued more than the retriever did
//! (`Q_j > P_j` ⟹ negative gradient ⟹ score goes up) and **lowers** the score of
//! any document the retriever over-rated. Iterate, and the retriever's ranking
//! rotates toward the LM's preference. `1/τ` appears as a factor: a sharp
//! retrieval softmax produces correspondingly larger gradients, which is why `η`
//! and `τ` have to be tuned together.
//!
//! # Why `KL(Q ‖ P)` and not `KL(P ‖ Q)`
//!
//! The order is not arbitrary. `KL(Q ‖ P)` is *mass-covering* in `P`: it is
//! `+inf` wherever `Q` has mass and `P` has none, so the retriever is punished
//! hard for assigning ~zero score to a document the LM found genuinely useful.
//! That is the failure we care about — a useful document that never gets
//! retrieved is invisible. The reverse, `KL(P ‖ Q)`, is mode-seeking and would
//! happily let the retriever collapse onto one document the LM likes while
//! ignoring the others.

use super::ensemble::ReplugEngine;
use super::math::{kl_divergence_from_log_probs, temperature_log_softmax};
use super::model::ReplugLanguageModel;
use super::types::{
    ReplugDocument, ReplugDocumentGradient, ReplugError, ReplugLsrSignal, ReplugResult,
};

impl ReplugEngine {
    /// `log P_LM(y* | d_i ⊕ q)` for every document — how much each one actually
    /// helped the frozen LM produce the ground truth.
    ///
    /// Length-normalized into a mean per-token log-likelihood when
    /// [`super::types::ReplugConfig::lsr_length_normalize`] is set.
    ///
    /// # Errors
    ///
    /// [`ReplugError::NoDocuments`], [`ReplugError::EmptyTarget`], or whatever
    /// the model returns.
    #[allow(clippy::cast_precision_loss)] // Continuation length, far below 2^53.
    pub fn document_target_log_likelihoods<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
        target_token_ids: &[usize],
    ) -> ReplugResult<Vec<f64>> {
        if documents.is_empty() {
            return Err(ReplugError::NoDocuments);
        }
        if target_token_ids.is_empty() {
            return Err(ReplugError::EmptyTarget);
        }

        let divisor = if self.config.lsr_length_normalize {
            target_token_ids.len() as f64
        } else {
            1.0
        };

        documents
            .iter()
            .map(|document| {
                let prefix = self.build_context(&document.text, query, "");
                let log_likelihood =
                    Self::sequence_log_likelihood(model, &prefix, target_token_ids)?;
                Ok(log_likelihood / divisor)
            })
            .collect()
    }

    /// Compute the full `REPLUG`-`LSR` feedback signal for one `(q, y*)` pair.
    ///
    /// Runs the frozen LM once per document per target token (teacher-forced),
    /// forms `P_R` and `Q_LM`, computes `KL(Q_LM ‖ P_R)` and the per-document
    /// gradient `(P − Q) / τ`, and applies one descent step to every retrieval
    /// score.
    ///
    /// Documents are **not** truncated to `top_k_documents` here: `LSR` is a
    /// learning signal over the candidate set the retriever produced, and
    /// silently dropping the low-ranked candidates would remove exactly the
    /// documents whose scores most need correcting.
    ///
    /// # Errors
    ///
    /// [`ReplugError::NoDocuments`] for an empty candidate set,
    /// [`ReplugError::EmptyTarget`] for an empty `y*`,
    /// [`ReplugError::InvalidTemperature`] for a bad `τ` or `β`, and whatever
    /// the model returns.
    pub fn lsr_signal<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
        target_token_ids: &[usize],
    ) -> ReplugResult<ReplugLsrSignal> {
        self.config.validate()?;
        if documents.is_empty() {
            return Err(ReplugError::NoDocuments);
        }
        if target_token_ids.is_empty() {
            return Err(ReplugError::EmptyTarget);
        }
        for (index, document) in documents.iter().enumerate() {
            if !document.score.is_finite() {
                return Err(ReplugError::NonFiniteLogit {
                    index,
                    value: document.score,
                });
            }
        }

        // What the retriever believes: P_R(d | q) = softmax(s / τ).
        let scores: Vec<f64> = documents.iter().map(|document| document.score).collect();
        let log_retrieval = temperature_log_softmax(&scores, self.config.temperature)?;

        // What the LM's behaviour implies: Q_LM(d | q) = softmax(log P_LM(y* | d ⊕ q) / β).
        let target_log_likelihoods =
            self.document_target_log_likelihoods(model, query, documents, target_token_ids)?;
        let log_lm = temperature_log_softmax(&target_log_likelihoods, self.config.lsr_beta)?;

        // L = KL(Q_LM ‖ P_R), entirely in log-space — neither `log_retrieval`
        // nor `log_lm` is ever exponentiated-then-logged, so a document whose
        // retrieval probability underflows to 0.0 still contributes a finite
        // (large) penalty rather than a NaN.
        let kl_loss = kl_divergence_from_log_probs(&log_lm, &log_retrieval)?;

        let gradients = documents
            .iter()
            .zip(&target_log_likelihoods)
            .zip(log_retrieval.iter().zip(&log_lm))
            .map(|((document, &target_log_likelihood), (&log_p, &log_q))| {
                let retrieval_prob = log_p.exp();
                let lm_prob = log_q.exp();

                // ∂ KL(Q ‖ P) / ∂ s_j = (P_j − Q_j) / τ — see the module docs.
                let score_gradient = (retrieval_prob - lm_prob) / self.config.temperature;
                let updated_score = document.score - self.config.lsr_learning_rate * score_gradient;

                ReplugDocumentGradient {
                    document_id: document.id.clone(),
                    target_log_likelihood,
                    retrieval_prob,
                    lm_prob,
                    score_gradient,
                    original_score: document.score,
                    updated_score,
                }
            })
            .collect();

        Ok(ReplugLsrSignal { kl_loss, gradients })
    }

    /// Apply a [`ReplugLsrSignal`] to the documents it was computed from,
    /// returning a new set carrying the updated retrieval scores.
    ///
    /// # Errors
    ///
    /// Returns [`ReplugError::WeightCountMismatch`] when the signal does not
    /// correspond to this document set (different length), and
    /// [`ReplugError::InvalidConfig`] when the ids do not line up — an update
    /// applied to the wrong documents would produce a plausible-looking and
    /// completely meaningless ranking, so it is refused rather than guessed at.
    pub fn apply_lsr_update(
        documents: &[ReplugDocument],
        signal: &ReplugLsrSignal,
    ) -> ReplugResult<Vec<ReplugDocument>> {
        if documents.len() != signal.gradients.len() {
            return Err(ReplugError::WeightCountMismatch {
                weights: signal.gradients.len(),
                documents: documents.len(),
            });
        }
        documents
            .iter()
            .zip(&signal.gradients)
            .map(|(document, gradient)| {
                if document.id != gradient.document_id {
                    return Err(ReplugError::InvalidConfig {
                        reason: format!(
                            "LSR signal is not aligned with the document set: expected id {:?}, \
                             found {:?}",
                            document.id, gradient.document_id
                        ),
                    });
                }
                Ok(ReplugDocument {
                    id: document.id.clone(),
                    text: document.text.clone(),
                    score: gradient.updated_score,
                })
            })
            .collect()
    }

    /// Run `steps` rounds of `LSR` on the same `(q, y*)` pair, returning the
    /// adapted documents and the signal from every round.
    ///
    /// Because `Q_LM` depends only on the frozen LM and is therefore **constant**
    /// across rounds, while `P_R` moves toward it, the reported `kl_loss`
    /// decreases monotonically for a sufficiently small `η` — this is plain
    /// gradient descent on a convex function of the scores (the softmax
    /// cross-entropy `−Σ_i Q_i log P_i(s)` is convex in `s`). The module's tests
    /// assert exactly that, which is the sharpest available check that the
    /// derived gradient is the true one: a sign error or a missing `1/τ` would
    /// break monotonicity immediately.
    ///
    /// # Errors
    ///
    /// As [`Self::lsr_signal`] and [`Self::apply_lsr_update`].
    pub fn lsr_adapt<M: ReplugLanguageModel + ?Sized>(
        &self,
        model: &M,
        query: &str,
        documents: &[ReplugDocument],
        target_token_ids: &[usize],
        steps: usize,
    ) -> ReplugResult<(Vec<ReplugDocument>, Vec<ReplugLsrSignal>)> {
        if steps == 0 {
            return Err(ReplugError::InvalidConfig {
                reason: "lsr_adapt needs at least one step".to_string(),
            });
        }

        let mut current = documents.to_vec();
        let mut history = Vec::with_capacity(steps);

        for _ in 0..steps {
            let signal = self.lsr_signal(model, query, &current, target_token_ids)?;
            current = Self::apply_lsr_update(&current, &signal)?;
            history.push(signal);
        }

        Ok((current, history))
    }
}
