//! **Activation steering**: fit a direction from contrastive activation pairs and
//! add it back *during* the forward pass, changing what the model says.
//!
//! Every other module in this crate that touches a model's internals **reads**
//! them. This one **writes** to them. That is the entire difference, and it is a
//! large one: reading a model's activations tells you what it is thinking, and
//! there is a great deal you can do with that — but it leaves the model's output
//! exactly as it was. Writing to them changes the output. The model does not know
//! it has been edited, there is no prompt to which it can object, and the edit
//! applies to every token it goes on to generate.
//!
//! Two algorithms are implemented, and they differ in *where* they write:
//!
//! | | Site | Direction | Shift |
//! |---|---|---|---|
//! | **`ITI`** (Li et al., 2023) | the output of a single **attention head**, `head_dim` wide | a linear probe's direction, unit-norm | `alpha * sigma`, `sigma` = std-dev of the activations' projections onto that direction |
//! | **`CAA`** (Rimsky et al., 2024) | the **residual stream** leaving a block, `hidden_dim` wide | `mean(positive) - mean(negative)`, raw | `alpha` |
//!
//! # Inference-Time Intervention
//!
//! `ITI` starts from an observation about where a concept lives. A truthfulness
//! direction is not smeared uniformly across a model: a *few* attention heads
//! represent it strongly and most do not. So:
//!
//! 1. **Probe every head.** For each of the `num_layers * num_heads` head slots,
//!    fit a binary [`LinearProbe`] on the head's output activations for a set of
//!    [`ContrastivePair`]s — one statement that exhibits the concept, one that does
//!    not. This module fits that probe two ways: by gradient descent on the
//!    logistic loss, and in closed form as the **mass-mean** difference
//!    `mean(pos) - mean(neg)`. See [`ProbeMethod`].
//! 2. **Rank the heads by *validation* accuracy** and keep the top `K`. Validation,
//!    never training: with `head_dim` free parameters and a few dozen examples, a
//!    probe on a head that knows nothing will still fit its training set well above
//!    chance. The held-out split is the only thing that tells a head that *knows*
//!    something from a head that *memorized* something.
//! 3. **Shift.** During the forward pass, add `alpha * sigma * theta` to each
//!    selected head's output, where `theta` is its unit direction and `sigma` is the
//!    standard deviation of the training activations' projections onto `theta`.
//!    Dividing the shift that way is what makes `alpha` **unitless** and comparable
//!    across heads whose activations differ in scale by orders of magnitude.
//!
//! # Contrastive Activation Addition
//!
//! `CAA` is blunter and, for many behaviours, works better. It skips the heads
//! entirely: take the residual stream at one layer, average the difference between
//! the positive and negative members of each pair, and add `alpha` times that
//! vector back into the residual stream at the same layer. There is no probe, no
//! selection, no `sigma` — the vector's own norm *is* the effect size, and it is
//! applied raw.
//!
//! For a *complete* pairing, `CAA`'s mean-of-paired-differences and `ITI`'s
//! difference-of-means are the same number. They are computed differently anyway,
//! because the pairing is the whole point of the construction: the two prompts of a
//! pair differ in the behaviour and, ideally, in nothing else, and that is what
//! cancels the nuisance variation which would otherwise swamp the contrast.
//!
//! # Two limits of this crate's hidden-state machinery
//!
//! This module defines its own model trait, [`SteerableModel`], rather than
//! building on [`HiddenStateProvider`](crate::hidden_states::HiddenStateProvider).
//! It does so because that trait cannot express what steering needs — and because
//! quietly pretending otherwise would be worse than saying so.
//!
//! ## Limit 1: there is no mid-forward hook
//!
//! [`HiddenStateProvider::extract_hidden_states`](crate::hidden_states::HiddenStateProvider::extract_hidden_states)
//! returns an **owned** [`ModelHiddenStates`](crate::hidden_states::ModelHiddenStates).
//! By the time a caller can touch a single number, the forward pass is over. Mutating
//! that snapshot is a **post-hoc edit of a captured forward pass**: it changes the
//! stored tensor for layer `l` and *nothing else*. Layer `l + 1`'s stored tensor was
//! computed from the pre-edit value and is not recomputed; the `LM` head is never
//! consulted, because no logits were kept.
//!
//! A faithful `ITI` cannot be built on that. `ITI`'s claim — the *only* claim that
//! makes it an intervention rather than a statistic — is that the edit changes what
//! every subsequent layer computes and therefore what the model emits.
//!
//! This module does not paper over the gap; it *demonstrates* it. The test
//! `captured_edit_does_not_propagate_to_a_later_layer` edits a captured
//! `ModelHiddenStates` at layer `l`, shows that layer `l` changed, and shows that
//! layer `l + 1` is **bit-identical** to what it was. The same edit applied through
//! [`SteerableModel::forward_pass`] changes layer `l + 1` *and* the logits. Both
//! operations are offered — [`ActivationSteering::edit_captured_residual`] for the
//! first, [`ActivationSteering::steer`] for the second — and the first one's
//! documentation says, at length, what it cannot do.
//!
//! ## Limit 2: the crate has no per-head activations
//!
//! `ITI` probes *head outputs*: `head_dim`-wide vectors, `num_heads` of them per
//! layer, with `num_heads * head_dim == hidden_dim`. Nothing in this crate produces
//! them. The closest thing is
//! [`LayerHiddenState::attention_weights`](crate::hidden_states::LayerHiddenState::attention_weights),
//! and it is not close at all: it is shaped `[1, num_heads, seq_len, seq_len]` — an
//! attention **pattern**, a matrix of one weight per (query, key) pair. It says
//! *where each head looked*. It does not contain *what each head returned*, and no
//! reshaping of a `seq_len x seq_len` matrix will produce a `head_dim`-wide vector.
//! (The one provider that could populate it, `candle_provider`, in fact never does:
//! the field is always `None`.)
//!
//! So [`SteerableModel::forward_pass`] reports [`HeadActivations`] itself, and a
//! backend that wants `ITI` has to expose them. There is no way around this that
//! does not involve inventing the numbers.
//!
//! # Distinct from
//!
//! Distinct from [`hidden_states`](crate::hidden_states), which *captures and
//! caches* a forward pass's per-layer activations for reuse but never modifies
//! them, and from `eigenscore`, which eigen-decomposes the covariance of `K`
//! sampled **response embeddings** to score hallucination and never reads a model's
//! internal activations at all. This module **writes** to the activations: it fits a
//! direction from contrastive activation pairs and adds it back during the forward
//! pass, changing what the model says.
//!
//! # Quick start
//!
//! ```
//! # #[cfg(feature = "activation-steering")]
//! # {
//! use oxirag::activation_steering::{
//!     ActivationSteering, ContrastivePair, HeadIndex, Intervention, InterventionConfig,
//!     InterventionSite, ProbeMethod, SteerableModel, SteeringConfig, SteeringFixtureModel,
//!     SteeringGeometry, SteeringVector,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // A 2-layer, 4-head fixture, with a "truthfulness" concept planted in exactly
//! // one head: whenever the input carries the TRUTHFUL marker, head L1H2's output
//! // is shifted by 3.0 along its first basis direction.
//! let geometry = SteeringGeometry::new(2, 4, 8)?;
//! let planted = HeadIndex::new(1, 2);
//! let direction = SteeringVector::unit(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?;
//! let model = SteeringFixtureModel::new(geometry, 12, 0xA11CE)?.with_concept(
//!     "TRUTHFUL",
//!     Intervention::new(InterventionSite::Head(planted), direction.clone(), 3.0)?,
//! )?;
//!
//! // Contrastive pairs: same content, one carries the marker.
//! let pairs: Vec<ContrastivePair> = (0..12)
//!     .map(|i| ContrastivePair::new(
//!         format!("TRUTHFUL claim number {i}"),
//!         format!("claim number {i}"),
//!     ))
//!     .collect();
//!
//! let config = SteeringConfig::default()
//!     .with_probe_method(ProbeMethod::MassMean)
//!     .with_top_k_heads(1);
//! let mut steering = ActivationSteering::new(geometry, config)?;
//! let report = steering.fit_iti(&model, &pairs)?;
//!
//! // It found the head the concept was planted in, and recovered the direction.
//! assert_eq!(report.selected, vec![planted]);
//! let fitted = report.head(planted).expect("the selected head is reported");
//! assert!(fitted.direction.cosine(&direction)? > 0.999);
//!
//! // Steering changes what the model emits...
//! let clean = model.forward("claim number 42")?;
//! let steered = steering.steer(&model, "claim number 42", &InterventionConfig::with_alpha(2.0)?)?;
//! assert!(clean != steered);
//!
//! // ...and alpha = 0 is *exactly* the identity, not approximately.
//! let ablated = steering.steer(&model, "claim number 42", &InterventionConfig::with_alpha(0.0)?)?;
//! assert_eq!(clean, ablated);
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! # References
//!
//! * Li, Patel, Viégas, Pfister & Wattenberg (2023), *Inference-Time Intervention:
//!   Eliciting Truthful Answers from a Language Model*.
//! * Rimsky, Gabrieli, Schulz, Tong, Hubinger & Turner (2024), *Steering Llama 2 via
//!   Contrastive Activation Addition*.
//! * Burns, Ye, Klein & Steinhardt (2023), *Discovering Latent Knowledge in Language
//!   Models Without Supervision* — the contrastive-pair construction these build on.

pub mod engine;
pub mod model;
pub mod probe;
pub mod rng;
pub mod types;

#[cfg(test)]
mod tests;

pub use engine::{ActivationSteering, HeadProbeReport, SteeringReport, split_pairs};
pub use model::{SteerableModel, SteeringConceptRule, SteeringFixtureModel};
pub use probe::{LinearProbe, SteeringProbeResult};
pub use rng::SteeringRng;
pub use types::{
    ActivationPair, CaaVector, ContrastivePair, HeadActivations, HeadIndex, Intervention,
    InterventionConfig, InterventionSite, MIN_DIRECTION_NORM, ProbeAccuracy, ProbeMethod,
    ResidualStream, SteeringConfig, SteeringError, SteeringForwardPass, SteeringGeometry,
    SteeringPositions, SteeringResult, SteeringVector,
};
