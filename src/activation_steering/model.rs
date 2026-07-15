//! [`SteerableModel`] — the model abstraction activation steering needs — and a
//! deterministic, table-driven implementation of it for tests, doctests, and
//! pipeline validation.
//!
//! # Why this module defines its own model trait
//!
//! Steering is an operation *inside* a forward pass. It is not a transformation of
//! a forward pass's output, and it is not a transformation of a forward pass's
//! *recorded* output either — it has to happen while the model is still running,
//! because the entire claim of `ITI` and `CAA` is that the edit changes what
//! every *subsequent* layer computes, and therefore changes the tokens the model
//! emits.
//!
//! No existing trait in this crate can express that.
//! [`HiddenStateProvider::extract_hidden_states`](crate::hidden_states::HiddenStateProvider::extract_hidden_states)
//! is `async fn(&self, &str) -> Result<ModelHiddenStates>`: it runs the model to
//! completion and hands back an **owned snapshot**. There is no callback, no hook
//! list, no borrow of the live activations — by the time the caller can touch a
//! single number, the forward pass is over and the logits, if there were any, have
//! been discarded. Writing into that snapshot changes the snapshot. It cannot
//! change the model.
//!
//! So this module states its own, minimal requirement:
//! [`SteerableModel::forward_pass`] is *exactly* "run yourself on this text, and
//! while you do, apply these edits". Everything else in the trait is a
//! convenience derived from it. A real backend (Candle, a remote endpoint, a
//! quantized on-device model) implements the one method and gets the rest.
//!
//! # The fixture
//!
//! [`SteeringFixtureModel`] is **a test fixture, not a transformer.** Its
//! "attention" is a uniform causal mean, its output projection is the identity,
//! its weights are hashed constants, and it has learned nothing. It exists so that
//! every claim this module makes about an intervention —
//!
//! * that the activation at the site moves by *exactly* `alpha * sigma` along the
//!   fitted direction,
//! * that the edit propagates into later layers and into the logits,
//! * that `alpha = 0` is **bit-identical** to no intervention at all,
//! * and that a probe fitted on its head activations recovers the direction that
//!   was planted in them,
//!
//! — can be checked against hand-computed numbers, with no model weights, no
//! randomness, and no network anywhere in the loop. It is public because those
//! checks are as useful to a caller wiring up their own [`SteerableModel`] as they
//! are to this crate's own test suite.
//!
//! ## The concept channel
//!
//! The fixture plants ground truth with [`SteeringConceptRule`]s: "when the input
//! contains the token `marker`, this head's output is shifted by this vector".
//! A rule is literally an [`Intervention`] — the same type the engine *produces* —
//! so a test can plant a direction and then demand the engine hand it back.
//!
//! One deliberate un-realism makes that ground truth clean: **marker tokens are
//! excluded from the content the fixture embeds.** In a real model the marker word
//! would leave its own footprint in every head, and a probe on *any* head could
//! separate the classes by finding it — which would make "only these `K` heads
//! carry the concept" false by construction. The fixture separates the two
//! channels so the statement is true and testable. A real model has no such
//! separation. This one does, because it is a fixture whose purpose is to make the
//! ground truth known.

// The fixture turns hashed `u64`s into `f64` weights and rounds the result to the
// `f32` that a model's activation tensors are actually stored in. Both casts are
// the point of the code, not an accident of it.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use super::rng::SteeringRng;
use super::types::{
    HeadActivations, HeadIndex, Intervention, InterventionSite, ResidualStream, SteeringError,
    SteeringForwardPass, SteeringGeometry, SteeringResult,
};

/// Offset folded into the seed for a layer's head matrices, so that the head
/// weights, the embeddings, and the unembedding draw from decorrelated streams.
const HEAD_WEIGHT_STREAM: u64 = 0x0000_0000_0000_0001;

/// Offset folded into the seed for the unembedding matrix.
const UNEMBEDDING_STREAM: u64 = 0x0000_0000_DEAD_BEEF;

/// Offset folded into the seed for the positional vectors.
const POSITION_STREAM: u64 = 0x0000_0000_0000_00F0;

/// The scale of a positional vector's entries, relative to a token embedding's.
///
/// Small but non-zero: enough to make two occurrences of the same token at
/// different positions distinguishable, not enough to swamp the token's identity.
const POSITION_SCALE: f64 = 0.1;

/// A model whose forward pass can be **edited while it runs**.
///
/// The contract is deliberately small, and every clause of it is load-bearing:
///
/// * [`forward_pass`](SteerableModel::forward_pass) applies the interventions
///   **during** the forward pass, not after it. An implementation that ran the
///   model, then edited the returned activations, would satisfy the type signature
///   and violate the contract: the edit must be visible to every computation
///   downstream of the site, including the logits.
/// * The activations it reports back are **post-intervention** — what the model
///   actually computed. That is what makes the effect of an intervention
///   measurable *at its own site*, and it is what the module's magnitude test
///   relies on.
/// * Per-head activations are `head_dim` wide, and the `num_heads` of a layer
///   partition its `hidden_dim`-wide residual stream. See
///   [`SteeringGeometry`].
///
/// `Send + Sync` so an engine holding one can be shared across threads, matching
/// the rest of this crate's model traits.
pub trait SteerableModel: Send + Sync {
    /// The model's shape.
    fn geometry(&self) -> SteeringGeometry;

    /// Run the model on `text`, applying `interventions` **mid-forward**.
    ///
    /// An empty `interventions` slice is a clean forward pass.
    ///
    /// # Errors
    ///
    /// Returns a [`SteeringError`] when an intervention names a site outside the
    /// model, when its direction is the wrong width for that site, or when the
    /// backend fails.
    fn forward_pass(
        &self,
        text: &str,
        interventions: &[Intervention],
    ) -> SteeringResult<SteeringForwardPass>;

    /// The per-head activations of a **clean** forward pass — what the probes are
    /// fitted on.
    ///
    /// # Errors
    ///
    /// Propagates [`SteerableModel::forward_pass`].
    fn head_activations(&self, text: &str) -> SteeringResult<HeadActivations> {
        Ok(self.forward_pass(text, &[])?.heads)
    }

    /// The per-layer residual stream of a **clean** forward pass — what `CAA`
    /// vectors are fitted on.
    ///
    /// # Errors
    ///
    /// Propagates [`SteerableModel::forward_pass`].
    fn residual_activations(&self, text: &str) -> SteeringResult<ResidualStream> {
        Ok(self.forward_pass(text, &[])?.residual)
    }

    /// The logits the model emits under `interventions`.
    ///
    /// # Errors
    ///
    /// Propagates [`SteerableModel::forward_pass`].
    fn forward_with_interventions(
        &self,
        text: &str,
        interventions: &[Intervention],
    ) -> SteeringResult<Vec<f32>> {
        Ok(self.forward_pass(text, interventions)?.logits)
    }

    /// The logits the model emits with no steering at all.
    ///
    /// # Errors
    ///
    /// Propagates [`SteerableModel::forward_pass`].
    fn forward(&self, text: &str) -> SteeringResult<Vec<f32>> {
        self.forward_with_interventions(text, &[])
    }
}

/// A concept planted in a [`SteeringFixtureModel`]: "when the input contains the
/// token `marker`, apply `intervention`".
///
/// The payload is an [`Intervention`] — the very type the engine produces — so a
/// test can plant a direction at a head and then demand that the fitted engine
/// recover it. The marker must be a single whitespace-free token, and it is
/// **excluded** from the text the fixture embeds; see the
/// [module documentation](self#the-concept-channel).
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringConceptRule {
    /// The token whose presence fires this rule.
    pub marker: String,
    /// What the model then does to its own activations.
    pub intervention: Intervention,
}

/// A fully-deterministic, table-driven steerable model.
///
/// **This is a test fixture, not a transformer.** See the
/// [module documentation](self#the-fixture).
///
/// # What it computes
///
/// With `x_t` the embedding of content token `t` (plus a positional vector), and
/// `W_{l,h}` a hashed `head_dim x head_dim` matrix:
///
/// ```text
/// r_0[t]        = embed(token_t) + pos(t)
/// ctx_{l,h}[t]  = mean over t' <= t of r_l[t'] restricted to head h's slice
/// o_{l,h}[t]    = W_{l,h} . ctx_{l,h}[t]            <- the ITI site
/// r_{l+1}[t]    = r_l[t] + concat over h of o_{l,h}[t]   <- the CAA site
/// logits        = U . r_L[last]
/// ```
///
/// The "attention" is a uniform causal mean and the output projection is the
/// identity, so head `h`'s output is written straight into head `h`'s slice of the
/// residual stream. Three consequences make the fixture *measurable*, and they are
/// the reason it is shaped this way:
///
/// 1. A head intervention `delta` at layer `l` moves `r_{l+1}` by exactly
///    `place_h(delta)` — a closed form, with no forward-pass arithmetic to
///    re-derive.
/// 2. That perturbation is carried into `r_{l+2}`, `r_{l+3}`, ... by the residual
///    connection alone, so **propagation is guaranteed**, not a lucky property of
///    the hashed weights.
/// 3. At the last layer the residual *is* the unembedding's input, so the logit
///    change is exactly `U . place_h(delta)` — computable from
///    [`SteeringFixtureModel::unembedding_row`] in three lines.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "activation-steering")]
/// # {
/// use oxirag::activation_steering::{
///     Intervention, InterventionSite, HeadIndex, SteerableModel, SteeringFixtureModel,
///     SteeringGeometry, SteeringVector,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let geometry = SteeringGeometry::new(3, 4, 8)?;
/// let model = SteeringFixtureModel::new(geometry, 16, 7)?;
///
/// // A clean pass, and a pass with the last layer's head 2 shifted.
/// let head = HeadIndex::new(2, 2);
/// let direction = SteeringVector::unit(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?;
/// let intervention = Intervention::new(InterventionSite::Head(head), direction, 2.5)?;
///
/// let clean = model.forward_pass("the capital of france", &[])?;
/// let steered = model.forward_pass("the capital of france", &[intervention])?;
///
/// // The head moved by exactly the magnitude, along exactly the direction.
/// let before = clean.heads.head(head)?[0];
/// let after = steered.heads.head(head)?[0];
/// assert!((f64::from(after - before) - 2.5).abs() < 1e-5);
///
/// // And the logits moved with it.
/// assert!(clean.logits != steered.logits);
/// # Ok(())
/// # }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SteeringFixtureModel {
    geometry: SteeringGeometry,
    vocab_size: usize,
    /// `[layer * num_heads + head]` -> a `head_dim x head_dim` row-major matrix.
    head_weights: Vec<Vec<f64>>,
    /// `vocab_size x hidden_dim`, row-major.
    unembedding: Vec<f64>,
    concepts: Vec<SteeringConceptRule>,
}

impl SteeringFixtureModel {
    /// Build a fixture with hashed weights derived from `seed`.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::InvalidConfig`] when `vocab_size` is zero.
    pub fn new(geometry: SteeringGeometry, vocab_size: usize, seed: u64) -> SteeringResult<Self> {
        if vocab_size == 0 {
            return Err(SteeringError::InvalidConfig {
                reason: "vocab_size must be at least 1".to_string(),
            });
        }
        let head_dim = geometry.head_dim();
        let hidden_dim = geometry.hidden_dim();

        let mut head_weights = Vec::with_capacity(geometry.num_head_slots());
        // Entries scaled by `0.5 / sqrt(head_dim)` so that a head's output has an
        // O(1) norm for an O(1) input and the residual stream neither explodes nor
        // vanishes across layers.
        let head_scale = 0.5 / (head_dim as f64).sqrt();
        for slot in 0..geometry.num_head_slots() {
            let mut rng = SteeringRng::derive(seed, HEAD_WEIGHT_STREAM ^ (slot as u64));
            let matrix = (0..head_dim * head_dim)
                .map(|_| rng.next_symmetric() * head_scale)
                .collect();
            head_weights.push(matrix);
        }

        let mut rng = SteeringRng::derive(seed, UNEMBEDDING_STREAM);
        let unembedding_scale = 1.0 / (hidden_dim as f64).sqrt();
        let unembedding = (0..vocab_size * hidden_dim)
            .map(|_| rng.next_symmetric() * unembedding_scale)
            .collect();

        Ok(Self {
            geometry,
            vocab_size,
            head_weights,
            unembedding,
            concepts: Vec::new(),
        })
    }

    /// Plant a concept: when the input contains the whitespace-delimited token
    /// `marker`, the model applies `intervention` to its own activations.
    ///
    /// The marker is **not** embedded as content — see the
    /// [module documentation](self#the-concept-channel).
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::InvalidConfig`] when `marker` is empty or contains
    /// whitespace (it must be a single token, or the exclusion rule would be
    /// ambiguous), and propagates [`Intervention::check`] when the intervention
    /// does not fit the geometry.
    pub fn with_concept(
        mut self,
        marker: impl Into<String>,
        intervention: Intervention,
    ) -> SteeringResult<Self> {
        let marker = marker.into();
        if marker.is_empty() || marker.split_whitespace().count() != 1 {
            return Err(SteeringError::InvalidConfig {
                reason: format!(
                    "a concept marker must be exactly one whitespace-free token, got {marker:?}"
                ),
            });
        }
        intervention.check(self.geometry)?;
        self.concepts.push(SteeringConceptRule {
            marker,
            intervention,
        });
        Ok(self)
    }

    /// The model's shape.
    #[must_use]
    pub const fn geometry(&self) -> SteeringGeometry {
        self.geometry
    }

    /// The number of output logits.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The concepts planted in this fixture.
    #[must_use]
    pub fn concepts(&self) -> &[SteeringConceptRule] {
        &self.concepts
    }

    /// Row `token` of the unembedding matrix — the `hidden_dim`-wide vector whose
    /// dot product with the final residual is `logits[token]`.
    ///
    /// Public so that a test can predict the *logit* change a residual change
    /// causes, in closed form, without re-deriving the forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::InvalidConfig`] when `token >= vocab_size`.
    pub fn unembedding_row(&self, token: usize) -> SteeringResult<&[f64]> {
        if token >= self.vocab_size {
            return Err(SteeringError::InvalidConfig {
                reason: format!(
                    "token {token} is outside a vocabulary of {} tokens",
                    self.vocab_size
                ),
            });
        }
        let hidden_dim = self.geometry.hidden_dim();
        Ok(&self.unembedding[token * hidden_dim..(token + 1) * hidden_dim])
    }

    /// The content tokens of `text`: its whitespace-delimited tokens, with every
    /// registered concept marker removed.
    #[must_use]
    pub fn content_tokens<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace()
            .filter(|token| !self.concepts.iter().any(|rule| rule.marker == *token))
            .collect()
    }

    /// The sequence length the fixture will use for `text` — the number of content
    /// tokens, or `1` for a text with none (a forward pass over an empty sequence
    /// is not a thing, so a text with no content is embedded as a single
    /// zero-token).
    #[must_use]
    pub fn sequence_length(&self, text: &str) -> usize {
        self.content_tokens(text).len().max(1)
    }

    /// Whether `text` fires the rule for `marker`.
    #[must_use]
    pub fn has_marker(&self, text: &str, marker: &str) -> bool {
        text.split_whitespace().any(|token| token == marker)
    }

    /// The deterministic embedding of one content token.
    fn embed(&self, token: &str) -> Vec<f64> {
        let mut rng = SteeringRng::new(fnv1a(token.as_bytes()));
        (0..self.geometry.hidden_dim())
            .map(|_| rng.next_symmetric())
            .collect()
    }

    /// The deterministic positional vector for position `position`.
    fn position_vector(&self, position: usize) -> Vec<f64> {
        let mut rng = SteeringRng::derive(POSITION_STREAM, position as u64);
        (0..self.geometry.hidden_dim())
            .map(|_| rng.next_symmetric() * POSITION_SCALE)
            .collect()
    }

    /// The concept rules that `text` fires.
    fn fired_concepts(&self, text: &str) -> Vec<&Intervention> {
        self.concepts
            .iter()
            .filter(|rule| self.has_marker(text, &rule.marker))
            .map(|rule| &rule.intervention)
            .collect()
    }
}

impl SteerableModel for SteeringFixtureModel {
    fn geometry(&self) -> SteeringGeometry {
        self.geometry
    }

    fn forward_pass(
        &self,
        text: &str,
        interventions: &[Intervention],
    ) -> SteeringResult<SteeringForwardPass> {
        for intervention in interventions {
            intervention.check(self.geometry)?;
        }

        let num_layers = self.geometry.num_layers();
        let num_heads = self.geometry.num_heads();
        let head_dim = self.geometry.head_dim();
        let hidden_dim = self.geometry.hidden_dim();

        let tokens = self.content_tokens(text);
        let sequence_length = tokens.len().max(1);
        let concepts = self.fired_concepts(text);

        // r_0[t] = embed(token_t) + pos(t). A text with no content tokens is
        // embedded as a single all-zero token plus its positional vector, so the
        // pass is still well-defined.
        let mut residual: Vec<Vec<f64>> = (0..sequence_length)
            .map(|position| {
                let mut vector = tokens
                    .get(position)
                    .map_or_else(|| vec![0.0; hidden_dim], |token| self.embed(token));
                for (slot, offset) in vector.iter_mut().zip(self.position_vector(position)) {
                    *slot += offset;
                }
                vector
            })
            .collect();

        let last = sequence_length - 1;
        let mut head_activations: Vec<Vec<f32>> =
            Vec::with_capacity(self.geometry.num_head_slots());
        let mut residual_stream: Vec<Vec<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            // `outputs[head][position]` — every head's output vector at every
            // position, before it is written back into the residual stream. This is
            // the ITI site.
            let mut outputs: Vec<Vec<Vec<f64>>> =
                vec![vec![vec![0.0; head_dim]; sequence_length]; num_heads];

            for (head, output) in outputs.iter_mut().enumerate() {
                let slice = self.geometry.head_slice(head)?;
                let weights = &self.head_weights[layer * num_heads + head];

                // Uniform causal "attention": ctx[t] is the running mean of this
                // head's slice of the residual over positions 0..=t.
                let mut prefix_sum = vec![0.0_f64; head_dim];
                for (position, out) in output.iter_mut().enumerate() {
                    for (slot, value) in prefix_sum
                        .iter_mut()
                        .zip(&residual[position][slice.clone()])
                    {
                        *slot += value;
                    }
                    let divisor = (position + 1) as f64;
                    for (row, out_slot) in out.iter_mut().enumerate() {
                        let weight_row = &weights[row * head_dim..(row + 1) * head_dim];
                        *out_slot = weight_row
                            .iter()
                            .zip(&prefix_sum)
                            .map(|(w, ctx)| w * ctx / divisor)
                            .sum();
                    }
                }

                let index = HeadIndex::new(layer, head);
                // Concepts first — they are what the model *computes*, part of its
                // ground truth. The caller's interventions are applied last, so that
                // an intervention of magnitude zero adds an exact `0.0` to a value
                // that is already final, and the pass stays bit-identical to a clean
                // one.
                for concept in &concepts {
                    apply_head(concept, index, output, sequence_length);
                }
                for intervention in interventions {
                    apply_head(intervention, index, output, sequence_length);
                }
            }

            // r_{l+1}[t] = r_l[t] + concat_h(o_{l,h}[t]): the output projection is
            // the identity, so head h writes into head h's slice.
            for (position, vector) in residual.iter_mut().enumerate() {
                for (head, output) in outputs.iter().enumerate() {
                    let base = head * head_dim;
                    for (offset, value) in output[position].iter().enumerate() {
                        vector[base + offset] += value;
                    }
                }
            }

            for concept in &concepts {
                apply_residual(concept, layer, &mut residual, sequence_length);
            }
            for intervention in interventions {
                apply_residual(intervention, layer, &mut residual, sequence_length);
            }

            for output in &outputs {
                head_activations.push(output[last].iter().map(|&v| v as f32).collect());
            }
            residual_stream.push(residual[last].iter().map(|&v| v as f32).collect());
        }

        let logits: Vec<f32> = (0..self.vocab_size)
            .map(|token| {
                let row = &self.unembedding[token * hidden_dim..(token + 1) * hidden_dim];
                row.iter()
                    .zip(&residual[last])
                    .map(|(u, r)| u * r)
                    .sum::<f64>() as f32
            })
            .collect();

        Ok(SteeringForwardPass {
            heads: HeadActivations::new(self.geometry, head_activations)?,
            residual: ResidualStream::new(hidden_dim, residual_stream)?,
            logits,
        })
    }
}

/// Add `intervention`'s delta to `output` if it targets head `index`.
fn apply_head(
    intervention: &Intervention,
    index: HeadIndex,
    output: &mut [Vec<f64>],
    sequence_length: usize,
) {
    if intervention.site != InterventionSite::Head(index) {
        return;
    }
    let delta = intervention.delta();
    for (position, vector) in output.iter_mut().enumerate() {
        if intervention.positions.includes(position, sequence_length) {
            for (slot, d) in vector.iter_mut().zip(&delta) {
                *slot += d;
            }
        }
    }
}

/// Add `intervention`'s delta to the residual stream if it targets `layer`.
fn apply_residual(
    intervention: &Intervention,
    layer: usize,
    residual: &mut [Vec<f64>],
    sequence_length: usize,
) {
    if intervention.site != (InterventionSite::Residual { layer }) {
        return;
    }
    let delta = intervention.delta();
    for (position, vector) in residual.iter_mut().enumerate() {
        if intervention.positions.includes(position, sequence_length) {
            for (slot, d) in vector.iter_mut().zip(&delta) {
                *slot += d;
            }
        }
    }
}

/// FNV-1a over the token's bytes.
///
/// Used rather than [`std::collections::hash_map::DefaultHasher`] because the
/// latter's output is explicitly **not** stable across Rust releases, and a
/// fixture whose embeddings silently changed under a toolchain upgrade would take
/// every hand-computed number in the test suite with it.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
