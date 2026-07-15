//! Core types for [`activation_steering`](super): model geometry, steering
//! directions, contrastive data, interventions, and configuration.
//!
//! The one piece of arithmetic that lives here rather than in
//! [`probe`](super::probe) or [`engine`](super::engine) is [`SteeringVector`],
//! which owns the handful of vector operations — norm, unit, dot, cosine,
//! projection — that every other file needs and that must round *identically*
//! wherever they are used. A cosine computed two different ways is a cosine you
//! cannot assert on.

use std::fmt;
use std::ops::Range;

use thiserror::Error;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors produced by the `activation_steering` module.
#[derive(Debug, Error)]
pub enum SteeringError {
    /// A [`SteeringGeometry`] had a zero-sized dimension. A model with no
    /// layers, no heads, or heads of width zero has nothing to steer.
    #[error("invalid steering geometry: {reason}")]
    InvalidGeometry {
        /// What was wrong with the geometry.
        reason: String,
    },

    /// A [`SteeringConfig`] or [`InterventionConfig`] value was outside its
    /// admissible range.
    #[error("invalid steering configuration: {reason}")]
    InvalidConfig {
        /// A human-readable explanation of what was invalid.
        reason: String,
    },

    /// A vector was empty. There is no such thing as a zero-dimensional
    /// direction.
    #[error("empty {what}")]
    EmptyVector {
        /// Which vector was empty.
        what: &'static str,
    },

    /// A value that must be finite was `NaN` or infinite.
    ///
    /// Rejected up front and loudly. A single non-finite activation folded into
    /// a probe's gradient would make every subsequent weight `NaN`, and the
    /// resulting "direction" would be a confident-looking vector of `NaN`s that
    /// silently no-ops every intervention built from it.
    #[error("non-finite {what} at index {index}: {value}")]
    NonFinite {
        /// Which quantity was non-finite.
        what: &'static str,
        /// Where in the vector it was found.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// Two vectors that had to agree in length did not.
    #[error("{what} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Which quantity had the wrong width.
        what: &'static str,
        /// The width that was required.
        expected: usize,
        /// The width that was supplied.
        actual: usize,
    },

    /// A direction's norm was below [`MIN_DIRECTION_NORM`], so it cannot be
    /// normalized: dividing by it would amplify rounding noise into a
    /// confident-looking unit vector pointing nowhere in particular.
    #[error("cannot normalize a {what} of norm {norm:e} (below {MIN_DIRECTION_NORM:e})")]
    ZeroDirection {
        /// Which quantity was degenerate.
        what: &'static str,
        /// Its measured norm.
        norm: f64,
    },

    /// A layer index was outside the model.
    #[error("layer {layer} out of range for a model with {num_layers} layers")]
    LayerOutOfRange {
        /// The offending index.
        layer: usize,
        /// How many layers the model has.
        num_layers: usize,
    },

    /// A head index was outside the model.
    #[error("head {head} out of range for a model with {num_heads} heads per layer")]
    HeadOutOfRange {
        /// The offending index.
        head: usize,
        /// How many heads each layer has.
        num_heads: usize,
    },

    /// A fitting routine was handed too few contrastive pairs to form both a
    /// non-empty training split and a non-empty validation split.
    ///
    /// Reported rather than silently degraded: a probe with no validation split
    /// still *reports* a validation accuracy, and that number would be a
    /// fabrication.
    #[error(
        "need at least {needed} contrastive pairs to form a train/validation split \
         at train_fraction {train_fraction}, got {actual}"
    )]
    InsufficientPairs {
        /// The minimum number of pairs required.
        needed: usize,
        /// How many were supplied.
        actual: usize,
        /// The configured training fraction that implies `needed`.
        train_fraction: f64,
    },

    /// Top-K head selection was asked for more heads than carry any signal at
    /// all.
    ///
    /// Degenerate heads (see [`HeadProbeReport::degenerate`](super::HeadProbeReport::degenerate))
    /// are excluded from selection, because an intervention along a zero
    /// direction is a silent no-op that would nonetheless be reported as a
    /// steering vector. Rather than pad the selection with them, this is an
    /// error.
    #[error(
        "cannot select {requested} heads: only {available} of {total} carry a \
         non-degenerate direction"
    )]
    InsufficientHeads {
        /// How many heads were requested (`SteeringConfig::top_k_heads`).
        requested: usize,
        /// How many heads have a usable direction.
        available: usize,
        /// How many head slots the model has in total.
        total: usize,
    },

    /// A probe's loss or gradient became non-finite during gradient descent.
    #[error("logistic probe diverged at iteration {iteration}: loss = {loss}")]
    ProbeDiverged {
        /// The iteration at which the loss stopped being finite.
        iteration: usize,
        /// The offending loss value.
        loss: f64,
    },

    /// An [`ActivationSteering`](super::ActivationSteering) method that needs a
    /// fitted model was called before the corresponding `fit_*`.
    #[error("activation steering has not been fitted: call {expected} first")]
    NotFitted {
        /// The method that must be called first.
        expected: &'static str,
    },

    /// A contrastive dataset was empty.
    #[error("no contrastive pairs supplied")]
    NoPairs,

    /// A [`SteerableModel`](super::SteerableModel) failed to produce a forward
    /// pass.
    #[error("steerable model error: {reason}")]
    Model {
        /// What the backend reported.
        reason: String,
    },

    /// An edit into a captured [`ModelHiddenStates`](crate::hidden_states::ModelHiddenStates)
    /// could not be applied because the captured tensor was not shaped
    /// `[1, seq_len, hidden_dim]`.
    #[error("captured layer {layer} has shape {shape:?}, expected [1, seq_len, {hidden_dim}]")]
    UnsteerableCapture {
        /// The layer whose tensor was malformed.
        layer: usize,
        /// The shape that was found.
        shape: Vec<usize>,
        /// The hidden width the steering vector expects.
        hidden_dim: usize,
    },
}

/// Convenience alias for this module's fallible return type.
pub type SteeringResult<T> = Result<T, SteeringError>;

/// The smallest norm a direction may have and still be normalizable.
///
/// Below this, a "direction" is numerically indistinguishable from the zero
/// vector, and normalizing it would amplify rounding noise into a
/// confident-looking unit vector pointing nowhere in particular. Heads whose
/// fitted direction falls below this bar are reported as
/// [degenerate](super::HeadProbeReport::degenerate) and are **never** selected
/// for intervention.
pub const MIN_DIRECTION_NORM: f64 = 1e-9;

// ── Geometry ─────────────────────────────────────────────────────────────────

/// The shape of the model being steered: how many layers, how many attention
/// heads per layer, and how wide each head is.
///
/// The invariant that makes per-head steering meaningful is
/// `num_heads * head_dim == hidden_dim`: the heads of a layer *partition* the
/// residual stream's width, so writing to head `h` writes to a known, disjoint
/// slice of the residual vector (see [`SteeringGeometry::head_slice`]). This is
/// the same convention the crate's
/// [`KVCache`](crate::hidden_states::KVCache) already uses for its
/// `[batch, heads, seq_len, head_dim]` tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SteeringGeometry {
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl SteeringGeometry {
    /// Build a geometry.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::InvalidGeometry`] when any dimension is zero.
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> SteeringResult<Self> {
        for (value, name) in [
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (head_dim, "head_dim"),
        ] {
            if value == 0 {
                return Err(SteeringError::InvalidGeometry {
                    reason: format!("{name} must be non-zero"),
                });
            }
        }
        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
        })
    }

    /// Number of transformer blocks.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of attention heads per block.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Width of one head's output vector.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Width of the residual stream, `num_heads * head_dim`.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Total number of head slots across the whole model,
    /// `num_layers * num_heads` — the number of probes `ITI` fits.
    #[must_use]
    pub const fn num_head_slots(&self) -> usize {
        self.num_layers * self.num_heads
    }

    /// The slice of the residual stream that head `head` of any layer occupies.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::HeadOutOfRange`] when `head >= num_heads`.
    pub fn head_slice(&self, head: usize) -> SteeringResult<Range<usize>> {
        if head >= self.num_heads {
            return Err(SteeringError::HeadOutOfRange {
                head,
                num_heads: self.num_heads,
            });
        }
        Ok(head * self.head_dim..(head + 1) * self.head_dim)
    }

    /// The flat index of a head, `layer * num_heads + head` — the index into a
    /// [`HeadActivations`] block.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::LayerOutOfRange`] or
    /// [`SteeringError::HeadOutOfRange`] when the index is outside the model.
    pub fn flat_index(&self, head: HeadIndex) -> SteeringResult<usize> {
        self.check_layer(head.layer)?;
        if head.head >= self.num_heads {
            return Err(SteeringError::HeadOutOfRange {
                head: head.head,
                num_heads: self.num_heads,
            });
        }
        Ok(head.layer * self.num_heads + head.head)
    }

    /// The inverse of [`SteeringGeometry::flat_index`].
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::HeadOutOfRange`] when `flat >= num_head_slots`.
    pub fn head_at(&self, flat: usize) -> SteeringResult<HeadIndex> {
        if flat >= self.num_head_slots() {
            return Err(SteeringError::HeadOutOfRange {
                head: flat,
                num_heads: self.num_head_slots(),
            });
        }
        Ok(HeadIndex::new(flat / self.num_heads, flat % self.num_heads))
    }

    /// Validate a layer index.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::LayerOutOfRange`] when `layer >= num_layers`.
    pub fn check_layer(&self, layer: usize) -> SteeringResult<()> {
        if layer >= self.num_layers {
            return Err(SteeringError::LayerOutOfRange {
                layer,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }

    /// The width of the activation an intervention at `site` writes to:
    /// `head_dim` for a head, `hidden_dim` for the residual stream.
    ///
    /// # Errors
    ///
    /// Returns an error when `site` names a layer or head outside the model.
    pub fn site_width(&self, site: InterventionSite) -> SteeringResult<usize> {
        match site {
            InterventionSite::Head(head) => {
                self.flat_index(head)?;
                Ok(self.head_dim)
            }
            InterventionSite::Residual { layer } => {
                self.check_layer(layer)?;
                Ok(self.hidden_dim())
            }
        }
    }
}

/// One attention head, addressed by its layer and its position within that
/// layer.
///
/// The derived `Ord` sorts by layer, then by head — the same order as the flat
/// index `layer * num_heads + head`, which is what the module's tie-breaks rely
/// on to be a *strict total order*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HeadIndex {
    /// The transformer block this head lives in.
    pub layer: usize,
    /// The head's position within its block.
    pub head: usize,
}

impl HeadIndex {
    /// Address head `head` of layer `layer`.
    #[must_use]
    pub const fn new(layer: usize, head: usize) -> Self {
        Self { layer, head }
    }
}

impl fmt::Display for HeadIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}H{}", self.layer, self.head)
    }
}

// ── Directions ───────────────────────────────────────────────────────────────

/// A direction in activation space: what a steering intervention adds.
///
/// Kept in `f64` even though activations are `f32`, because a steering vector is
/// a *fitted* quantity — a mean of hundreds of activations, or the weight vector
/// of a gradient descent — and the arithmetic that produces it should not round
/// to `f32` on the way. The single conversion back to `f32` happens at the point
/// of application, in [`Intervention::delta`].
///
/// A `SteeringVector` is **not** necessarily unit-norm. `ITI` directions are
/// (the magnitude lives in `alpha * sigma`); `CAA` vectors are not (their norm
/// *is* the effect size). Use [`SteeringVector::normalized`] when you need the
/// unit version.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringVector {
    values: Vec<f64>,
}

impl SteeringVector {
    /// Wrap a vector of coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when `values` is empty, and
    /// [`SteeringError::NonFinite`] when any coefficient is `NaN` or infinite.
    pub fn new(values: Vec<f64>) -> SteeringResult<Self> {
        if values.is_empty() {
            return Err(SteeringError::EmptyVector {
                what: "steering vector",
            });
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SteeringError::NonFinite {
                    what: "steering vector coefficient",
                    index,
                    value,
                });
            }
        }
        Ok(Self { values })
    }

    /// Wrap a vector and scale it to unit norm.
    ///
    /// # Errors
    ///
    /// Propagates [`SteeringVector::new`], and returns
    /// [`SteeringError::ZeroDirection`] when the norm is below
    /// [`MIN_DIRECTION_NORM`].
    pub fn unit(values: Vec<f64>) -> SteeringResult<Self> {
        Self::new(values)?.normalized()
    }

    /// The all-zero vector of width `dim` — the direction of a **degenerate**
    /// head, which carries no signal and is never steered along.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when `dim == 0`.
    pub fn zeros(dim: usize) -> SteeringResult<Self> {
        Self::new(vec![0.0; dim])
    }

    /// The coefficients.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// The width of the activation this direction lives in.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false` — a `SteeringVector` cannot be constructed empty. Present
    /// because `len` without `is_empty` is a lint, and because a caller writing
    /// generic code should not have to know that.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The Euclidean norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Whether this direction is too small to be normalized — the definition of
    /// a degenerate direction.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.norm() < MIN_DIRECTION_NORM
    }

    /// This direction, scaled to unit norm.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::ZeroDirection`] when the norm is below
    /// [`MIN_DIRECTION_NORM`].
    pub fn normalized(&self) -> SteeringResult<Self> {
        let norm = self.norm();
        if norm < MIN_DIRECTION_NORM {
            return Err(SteeringError::ZeroDirection {
                what: "steering vector",
                norm,
            });
        }
        Ok(Self {
            values: self.values.iter().map(|v| v / norm).collect(),
        })
    }

    /// This direction, multiplied by `factor`.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NonFinite`] when `factor` is not finite, or when
    /// the product overflows to infinity.
    pub fn scaled(&self, factor: f64) -> SteeringResult<Self> {
        if !factor.is_finite() {
            return Err(SteeringError::NonFinite {
                what: "scale factor",
                index: 0,
                value: factor,
            });
        }
        Self::new(self.values.iter().map(|v| v * factor).collect())
    }

    /// The dot product with another direction of the same width.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::DimensionMismatch`] when the widths differ.
    pub fn dot(&self, other: &Self) -> SteeringResult<f64> {
        if self.len() != other.len() {
            return Err(SteeringError::DimensionMismatch {
                what: "steering vector",
                expected: self.len(),
                actual: other.len(),
            });
        }
        Ok(self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum())
    }

    /// The cosine of the angle to another direction.
    ///
    /// This is the module's measure of "did the probe recover the direction the
    /// data was generated along?", so it is defined in exactly one place.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::DimensionMismatch`] when the widths differ, and
    /// [`SteeringError::ZeroDirection`] when either vector is degenerate (the
    /// angle to the zero vector is undefined, and returning `0.0` for it would
    /// quietly report "orthogonal" for "no direction at all").
    pub fn cosine(&self, other: &Self) -> SteeringResult<f64> {
        let dot = self.dot(other)?;
        let (left, right) = (self.norm(), other.norm());
        if left < MIN_DIRECTION_NORM {
            return Err(SteeringError::ZeroDirection {
                what: "left steering vector",
                norm: left,
            });
        }
        if right < MIN_DIRECTION_NORM {
            return Err(SteeringError::ZeroDirection {
                what: "right steering vector",
                norm: right,
            });
        }
        Ok(dot / (left * right))
    }

    /// The projection of an `f32` activation onto this direction, computed in
    /// `f64`.
    ///
    /// The `f32 -> f64` promotion is exact (`f64::from`), so this loses nothing:
    /// it is the accumulation that needs the wider type.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::DimensionMismatch`] when `activation` has the
    /// wrong width.
    pub fn project(&self, activation: &[f32]) -> SteeringResult<f64> {
        if activation.len() != self.len() {
            return Err(SteeringError::DimensionMismatch {
                what: "activation",
                expected: self.len(),
                actual: activation.len(),
            });
        }
        Ok(self
            .values
            .iter()
            .zip(activation)
            .map(|(w, &x)| w * f64::from(x))
            .sum())
    }
}

// ── Activations ──────────────────────────────────────────────────────────────

/// The per-head output activations of one forward pass, at the position `ITI`
/// hooks (the final token).
///
/// Indexed by [`SteeringGeometry::flat_index`]; each entry is `head_dim` wide.
///
/// **These do not exist anywhere else in this crate.** See the
/// [module documentation](super#limit-2-the-crate-has-no-per-head-activations)
/// for why `LayerHiddenState::attention_weights` — which is shaped
/// `[1, num_heads, seq_len, seq_len]` — is not this object and cannot be made
/// into it.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadActivations {
    geometry: SteeringGeometry,
    values: Vec<Vec<f32>>,
}

impl HeadActivations {
    /// Wrap a block of per-head activations.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::DimensionMismatch`] when `values` does not have
    /// exactly [`SteeringGeometry::num_head_slots`] entries, or when any entry
    /// is not `head_dim` wide.
    pub fn new(geometry: SteeringGeometry, values: Vec<Vec<f32>>) -> SteeringResult<Self> {
        if values.len() != geometry.num_head_slots() {
            return Err(SteeringError::DimensionMismatch {
                what: "head activation block",
                expected: geometry.num_head_slots(),
                actual: values.len(),
            });
        }
        for slot in &values {
            if slot.len() != geometry.head_dim() {
                return Err(SteeringError::DimensionMismatch {
                    what: "head activation",
                    expected: geometry.head_dim(),
                    actual: slot.len(),
                });
            }
        }
        Ok(Self { geometry, values })
    }

    /// One head's activation.
    ///
    /// # Errors
    ///
    /// Returns an error when `head` is outside the geometry.
    pub fn head(&self, head: HeadIndex) -> SteeringResult<&[f32]> {
        let flat = self.geometry.flat_index(head)?;
        Ok(&self.values[flat])
    }

    /// The geometry these activations came from.
    #[must_use]
    pub const fn geometry(&self) -> SteeringGeometry {
        self.geometry
    }

    /// All slots, in flat-index order.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec<f32>] {
        &self.values
    }

    /// Consume `self` and yield the raw slots.
    #[must_use]
    pub fn into_inner(self) -> Vec<Vec<f32>> {
        self.values
    }

    /// The number of head slots, `num_layers * num_heads`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Always `false` — a geometry has at least one layer and one head.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The residual stream of one forward pass, one `hidden_dim`-wide vector per
/// layer, at the position `CAA` hooks (the final token).
///
/// `layer(l)` is the residual **leaving** block `l` — the value a `CAA` hook on
/// block `l`'s output reads, and the value an
/// [`InterventionSite::Residual`] at layer `l` writes to.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualStream {
    hidden_dim: usize,
    values: Vec<Vec<f32>>,
}

impl ResidualStream {
    /// Wrap a per-layer residual block.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when there are no layers, and
    /// [`SteeringError::DimensionMismatch`] when any layer's vector is not
    /// `hidden_dim` wide.
    pub fn new(hidden_dim: usize, values: Vec<Vec<f32>>) -> SteeringResult<Self> {
        if values.is_empty() {
            return Err(SteeringError::EmptyVector {
                what: "residual stream",
            });
        }
        for layer in &values {
            if layer.len() != hidden_dim {
                return Err(SteeringError::DimensionMismatch {
                    what: "residual stream layer",
                    expected: hidden_dim,
                    actual: layer.len(),
                });
            }
        }
        Ok(Self { hidden_dim, values })
    }

    /// The residual vector leaving block `layer`.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::LayerOutOfRange`] when `layer` is outside the
    /// stream.
    pub fn layer(&self, layer: usize) -> SteeringResult<&[f32]> {
        self.values
            .get(layer)
            .map(Vec::as_slice)
            .ok_or(SteeringError::LayerOutOfRange {
                layer,
                num_layers: self.values.len(),
            })
    }

    /// How many layers the stream has.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.values.len()
    }

    /// The residual width.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// All layers, in order.
    #[must_use]
    pub fn as_slice(&self) -> &[Vec<f32>] {
        &self.values
    }

    /// Consume `self` and yield the raw layers.
    #[must_use]
    pub fn into_inner(self) -> Vec<Vec<f32>> {
        self.values
    }
}

/// Everything one forward pass of a [`SteerableModel`](super::SteerableModel)
/// produces — with any [`Intervention`]s **already applied**.
///
/// The activations reported here are therefore *post*-intervention: they are
/// what the model actually computed, not what it would have computed. That is
/// what makes the effect of an intervention measurable at its own site.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringForwardPass {
    /// Per-head outputs at the final token.
    pub heads: HeadActivations,
    /// Per-layer residual stream at the final token.
    pub residual: ResidualStream,
    /// Next-token logits, `vocab_size` wide.
    pub logits: Vec<f32>,
}

// ── Contrastive data ─────────────────────────────────────────────────────────

/// One contrastive example: two inputs that differ in the concept being steered
/// and — ideally — in nothing else.
///
/// In `ITI` these are a truthful and an untruthful completion of the same
/// question. In `CAA` they are the same multiple-choice prompt with the
/// behaviour-exhibiting and behaviour-avoiding answer appended. In both cases
/// the *pairing* is the point: it is what cancels the nuisance variation that
/// would otherwise dominate the difference of activations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContrastivePair {
    /// The input that exhibits the concept.
    pub positive: String,
    /// The input that does not.
    pub negative: String,
}

impl ContrastivePair {
    /// Build a pair.
    #[must_use]
    pub fn new(positive: impl Into<String>, negative: impl Into<String>) -> Self {
        Self {
            positive: positive.into(),
            negative: negative.into(),
        }
    }
}

/// One contrastive example's activations, already collected: the positive
/// input's activation at every site, and the negative input's.
///
/// A *site* is a head slot (for `ITI`, indexed by
/// [`SteeringGeometry::flat_index`]) or a layer (for `CAA`). The two vectors
/// must agree in length and site-by-site width; nothing else is assumed.
///
/// This is the type that lets the module be fitted **without a model** — from a
/// table of activations dumped by some other tool, or from a synthetic
/// generating process whose ground truth is known. The module's headline
/// recovery tests use exactly that: they build `ActivationPair`s in which a
/// known unit direction separates the classes, and demand it back.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationPair {
    /// The positive input's activation at every site.
    pub positive: Vec<Vec<f32>>,
    /// The negative input's activation at every site.
    pub negative: Vec<Vec<f32>>,
}

impl ActivationPair {
    /// Build a pair, checking that the two sides line up and are finite.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::EmptyVector`] when there are no sites,
    /// [`SteeringError::DimensionMismatch`] when the two sides disagree on the
    /// number of sites or on any site's width, and [`SteeringError::NonFinite`]
    /// when any activation is `NaN` or infinite.
    pub fn new(positive: Vec<Vec<f32>>, negative: Vec<Vec<f32>>) -> SteeringResult<Self> {
        if positive.is_empty() {
            return Err(SteeringError::EmptyVector {
                what: "activation pair",
            });
        }
        if positive.len() != negative.len() {
            return Err(SteeringError::DimensionMismatch {
                what: "activation pair site count",
                expected: positive.len(),
                actual: negative.len(),
            });
        }
        for (site, (pos, neg)) in positive.iter().zip(&negative).enumerate() {
            if pos.len() != neg.len() {
                return Err(SteeringError::DimensionMismatch {
                    what: "activation pair site width",
                    expected: pos.len(),
                    actual: neg.len(),
                });
            }
            if pos.is_empty() {
                return Err(SteeringError::EmptyVector {
                    what: "activation pair site",
                });
            }
            for (side, values) in [("positive", pos), ("negative", neg)] {
                for (index, &value) in values.iter().enumerate() {
                    if !value.is_finite() {
                        let _ = site;
                        return Err(SteeringError::NonFinite {
                            what: if side == "positive" {
                                "positive activation"
                            } else {
                                "negative activation"
                            },
                            index,
                            value: f64::from(value),
                        });
                    }
                }
            }
        }
        Ok(Self { positive, negative })
    }

    /// How many sites this pair covers.
    #[must_use]
    pub fn num_sites(&self) -> usize {
        self.positive.len()
    }
}

// ── Interventions ────────────────────────────────────────────────────────────

/// Where an intervention writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterventionSite {
    /// One attention head's **output** vector, `head_dim` wide, *before* it is
    /// written back into the residual stream. This is `ITI`'s site (Li et al.,
    /// 2023).
    Head(HeadIndex),
    /// The residual stream **leaving** a transformer block, `hidden_dim` wide.
    /// This is `CAA`'s site (Rimsky et al., 2024).
    Residual {
        /// The block whose output residual is edited.
        layer: usize,
    },
}

impl InterventionSite {
    /// The layer this site lives in.
    #[must_use]
    pub const fn layer(&self) -> usize {
        match self {
            Self::Head(head) => head.layer,
            Self::Residual { layer } => *layer,
        }
    }
}

impl fmt::Display for InterventionSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Head(head) => write!(f, "head {head}"),
            Self::Residual { layer } => write!(f, "residual L{layer}"),
        }
    }
}

/// Which token positions an intervention is applied at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SteeringPositions {
    /// Every position in the sequence.
    ///
    /// This is what `CAA` does (Rimsky et al. add their vector at every token
    /// from the prompt onward), and it is the default.
    #[default]
    All,
    /// Only the final position — the one whose activations the probes were fit
    /// on.
    LastToken,
}

impl SteeringPositions {
    /// Whether position `position` of a `sequence_length`-long sequence is
    /// steered.
    #[must_use]
    pub const fn includes(&self, position: usize, sequence_length: usize) -> bool {
        match self {
            Self::All => position < sequence_length,
            Self::LastToken => sequence_length > 0 && position + 1 == sequence_length,
        }
    }
}

/// A single edit to a forward pass: at `site`, add `magnitude * direction`.
///
/// The two clients of this type parameterize it differently, and the difference
/// is the whole difference between the two algorithms:
///
/// | | `direction` | `magnitude` |
/// |---|---|---|
/// | `ITI` | the probe's direction, **unit-norm** | `alpha * sigma`, where `sigma` is the standard deviation of the training activations' projections onto that direction |
/// | `CAA` | the raw mean-difference vector, **whose norm is the effect size** | `alpha` |
///
/// `ITI` normalizes the direction and puts the scale in `sigma` precisely so that
/// `alpha` is *unitless* and comparable across heads whose activations have wildly
/// different scales. `CAA` keeps the raw vector because its norm is a meaningful,
/// dataset-level measure of how far apart the two behaviours sit.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervention {
    /// Where to write.
    pub site: InterventionSite,
    /// What to add, up to scale.
    pub direction: SteeringVector,
    /// How much of it to add.
    pub magnitude: f64,
    /// At which token positions.
    pub positions: SteeringPositions,
}

impl Intervention {
    /// Build an intervention that writes at every position.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NonFinite`] when `magnitude` is `NaN` or
    /// infinite.
    pub fn new(
        site: InterventionSite,
        direction: SteeringVector,
        magnitude: f64,
    ) -> SteeringResult<Self> {
        if !magnitude.is_finite() {
            return Err(SteeringError::NonFinite {
                what: "intervention magnitude",
                index: 0,
                value: magnitude,
            });
        }
        Ok(Self {
            site,
            direction,
            magnitude,
            positions: SteeringPositions::All,
        })
    }

    /// Restrict this intervention to `positions`.
    #[must_use]
    pub fn at_positions(mut self, positions: SteeringPositions) -> Self {
        self.positions = positions;
        self
    }

    /// The vector actually added to the activation: `magnitude * direction`, in
    /// `f64`.
    ///
    /// This is the single definition of "what an intervention does", and both
    /// the fixture model and any real backend must agree with it.
    #[must_use]
    pub fn delta(&self) -> Vec<f64> {
        self.direction
            .as_slice()
            .iter()
            .map(|d| self.magnitude * d)
            .collect()
    }

    /// [`Intervention::delta`], rounded to the `f32` a real backend's activation
    /// tensors are stored in.
    ///
    /// Provided as the one place the `f64 -> f32` narrowing happens, so a backend
    /// never has to invent its own.
    #[must_use]
    pub fn delta_f32(&self) -> Vec<f32> {
        // The narrowing is the point: activations are `f32`, so the shift must
        // become one before it can be added to them. The fitted direction is
        // kept in `f64` right up to this line so that only *one* rounding
        // separates the fit from the edit.
        #[allow(clippy::cast_possible_truncation)]
        self.delta().into_iter().map(|d| d as f32).collect()
    }

    /// The Euclidean length of the shift, `|magnitude| * ||direction||`.
    ///
    /// For an `ITI` intervention this is exactly `alpha * sigma`, because the
    /// direction is unit-norm — which is the quantity the module's
    /// intervention-magnitude test measures.
    #[must_use]
    pub fn delta_norm(&self) -> f64 {
        self.magnitude.abs() * self.direction.norm()
    }

    /// Check that this intervention fits `geometry`.
    ///
    /// # Errors
    ///
    /// Returns an error when the site is outside the model, or when the
    /// direction's width does not match the site's.
    pub fn check(&self, geometry: SteeringGeometry) -> SteeringResult<()> {
        let width = geometry.site_width(self.site)?;
        if self.direction.len() != width {
            return Err(SteeringError::DimensionMismatch {
                what: "intervention direction",
                expected: width,
                actual: self.direction.len(),
            });
        }
        Ok(())
    }
}

/// How strongly, and where, to steer at inference time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterventionConfig {
    /// The steering strength.
    ///
    /// `0.0` is the identity — and it is *exactly* the identity, not
    /// approximately: with `alpha = 0` every [`Intervention::delta`] is the zero
    /// vector, and the module's ablation test asserts the steered forward pass is
    /// **bit-identical** to the clean one.
    ///
    /// Negative values steer *away* from the positive class, which is a
    /// legitimate and frequently used mode (`CAA` reports that negative
    /// coefficients suppress a behaviour as reliably as positive ones elicit it).
    pub alpha: f64,
    /// Which token positions to steer.
    pub positions: SteeringPositions,
}

impl Default for InterventionConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            positions: SteeringPositions::All,
        }
    }
}

impl InterventionConfig {
    /// A configuration at strength `alpha`, steering every position.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NonFinite`] when `alpha` is `NaN` or infinite.
    pub fn with_alpha(alpha: f64) -> SteeringResult<Self> {
        if !alpha.is_finite() {
            return Err(SteeringError::NonFinite {
                what: "alpha",
                index: 0,
                value: alpha,
            });
        }
        Ok(Self {
            alpha,
            positions: SteeringPositions::All,
        })
    }

    /// Restrict steering to `positions`.
    #[must_use]
    pub const fn at_positions(mut self, positions: SteeringPositions) -> Self {
        self.positions = positions;
        self
    }
}

// ── Probes ───────────────────────────────────────────────────────────────────

/// Which linear probe is fitted on the contrastive activations — and therefore
/// which direction the intervention runs along.
///
/// Both are from Li et al. (2023), which fits probes to *rank* heads and then
/// offers two choices of direction to *shift* along. They are not
/// interchangeable, and the paper finds the second works better:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProbeMethod {
    /// **Logistic regression**, fitted by gradient descent on the (convex)
    /// logistic loss with an `L2` penalty on the weights.
    ///
    /// The direction is the normalized weight vector — the direction of steepest
    /// increase in the log-odds of the positive class. It is the *discriminative*
    /// direction: it points where the classes are most separable, which is not
    /// generally where their means differ, because a discriminative fit will
    /// happily exploit a low-variance nuisance dimension.
    #[default]
    Logistic,
    /// **Mass-mean**: the direction is `mean(positive) - mean(negative)`,
    /// normalized, and the classifier is the nearest-centroid rule that direction
    /// induces.
    ///
    /// Closed-form, no optimization, no hyper-parameters. Li et al. find that
    /// shifting along the mass-mean direction beats shifting along the probe's
    /// weights even when the probe classifies better, and the reason is exactly
    /// the asymmetry above: a probe's weight vector is optimized to *separate*,
    /// while the mass-mean shift is the direction in which the positive class
    /// actually *lies*. You are intervening, not classifying.
    MassMean,
}

impl fmt::Display for ProbeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Logistic => f.write_str("logistic"),
            Self::MassMean => f.write_str("mass-mean"),
        }
    }
}

/// A probe's accuracy on the two splits.
///
/// Head selection ranks on [`ProbeAccuracy::validation`] and never on
/// [`ProbeAccuracy::train`]. With `head_dim` free parameters and a few dozen
/// examples, *training* accuracy is nearly free — a probe on a head that carries
/// no signal at all will still fit its training set well above chance. The
/// validation split is the only thing that distinguishes a head that knows
/// something from a head that memorized something.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeAccuracy {
    /// Fraction correct on the pairs the probe was fitted on.
    pub train: f64,
    /// Fraction correct on the held-out pairs.
    pub validation: f64,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// How to fit the probes, split the data, and select the heads.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringConfig {
    /// Which probe to fit, and therefore which direction to steer along.
    pub probe_method: ProbeMethod,
    /// `K` in "top-`K` heads by validation accuracy".
    ///
    /// Li et al. sweep this and settle on 48 of a `LLaMA`-7B's 1024 heads — a
    /// deliberately *sparse* intervention, on the finding that steering every
    /// head degrades fluency without improving truthfulness.
    pub top_k_heads: usize,
    /// The fraction of contrastive **pairs** in the training split.
    ///
    /// Pairs, not activations: both halves of a pair go to the same side of the
    /// split. Splitting at the activation level would put a pair's positive in
    /// train and its negative in validation, and since the two share everything
    /// except the concept, that is a textbook leak — the validation accuracy it
    /// produced would be an overstatement, and head selection would be ranking on
    /// a fabricated number.
    pub train_fraction: f64,
    /// Seed for the train/validation split. See [`SteeringRng`](super::SteeringRng).
    pub seed: u64,
    /// The `L2` penalty on the logistic probe's **weights** (never on its bias:
    /// penalizing the intercept would drag the decision boundary toward the
    /// origin, which is a statement about the activations' absolute position that
    /// nobody intends to make).
    pub l2_penalty: f64,
    /// Maximum gradient-descent iterations per probe.
    pub max_iterations: usize,
    /// Convergence tolerance: descent stops when the gradient's infinity-norm
    /// falls below this.
    pub tolerance: f64,
    /// Which layers `CAA` fits a residual-stream vector for. Empty means *all*
    /// layers.
    pub caa_layers: Vec<usize>,
}

impl Default for SteeringConfig {
    fn default() -> Self {
        Self {
            probe_method: ProbeMethod::Logistic,
            top_k_heads: 8,
            train_fraction: 0.8,
            seed: 0x5EED_0BED,
            l2_penalty: 1e-3,
            max_iterations: 500,
            tolerance: 1e-7,
            caa_layers: Vec::new(),
        }
    }
}

impl SteeringConfig {
    /// Check every field.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::InvalidConfig`] naming the offending field.
    pub fn validate(&self) -> SteeringResult<()> {
        if self.top_k_heads == 0 {
            return Err(SteeringError::InvalidConfig {
                reason: "top_k_heads must be at least 1".to_string(),
            });
        }
        if !self.train_fraction.is_finite()
            || self.train_fraction <= 0.0
            || self.train_fraction >= 1.0
        {
            return Err(SteeringError::InvalidConfig {
                reason: format!(
                    "train_fraction must lie strictly inside (0, 1), got {}",
                    self.train_fraction
                ),
            });
        }
        if !self.l2_penalty.is_finite() || self.l2_penalty < 0.0 {
            return Err(SteeringError::InvalidConfig {
                reason: format!(
                    "l2_penalty must be finite and non-negative, got {}",
                    self.l2_penalty
                ),
            });
        }
        if self.max_iterations == 0 {
            return Err(SteeringError::InvalidConfig {
                reason: "max_iterations must be at least 1".to_string(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(SteeringError::InvalidConfig {
                reason: format!(
                    "tolerance must be finite and strictly positive, got {}",
                    self.tolerance
                ),
            });
        }
        Ok(())
    }

    /// Set the probe method.
    #[must_use]
    pub fn with_probe_method(mut self, probe_method: ProbeMethod) -> Self {
        self.probe_method = probe_method;
        self
    }

    /// Set `K`.
    #[must_use]
    pub fn with_top_k_heads(mut self, top_k_heads: usize) -> Self {
        self.top_k_heads = top_k_heads;
        self
    }

    /// Set the training fraction.
    #[must_use]
    pub fn with_train_fraction(mut self, train_fraction: f64) -> Self {
        self.train_fraction = train_fraction;
        self
    }

    /// Set the split seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the `L2` penalty.
    #[must_use]
    pub fn with_l2_penalty(mut self, l2_penalty: f64) -> Self {
        self.l2_penalty = l2_penalty;
        self
    }

    /// Set the layers `CAA` fits vectors for.
    #[must_use]
    pub fn with_caa_layers(mut self, caa_layers: Vec<usize>) -> Self {
        self.caa_layers = caa_layers;
        self
    }
}

/// A `CAA` steering vector: the mean difference between the positive and
/// negative residual streams at one layer.
///
/// Its **norm is not incidental** — unlike an `ITI` direction, a `CaaVector` is
/// applied raw, scaled only by `alpha`. The norm is the dataset's answer to "how
/// far apart, in the residual stream, do these two behaviours actually sit at
/// this layer?", and comparing it across layers is the standard way of choosing
/// which layer to steer.
#[derive(Debug, Clone, PartialEq)]
pub struct CaaVector {
    /// The block whose output residual this vector is added to.
    pub layer: usize,
    /// `mean(positive residual) - mean(negative residual)`, unnormalized.
    pub vector: SteeringVector,
    /// How many contrastive pairs went into the mean.
    pub num_pairs: usize,
}

impl CaaVector {
    /// The effect size: `||mean(positive) - mean(negative)||`.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.vector.norm()
    }

    /// The intervention that adds `alpha * self` to layer `self.layer`'s output
    /// residual.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NonFinite`] when `config.alpha` is not finite.
    pub fn intervention(&self, config: &InterventionConfig) -> SteeringResult<Intervention> {
        Intervention::new(
            InterventionSite::Residual { layer: self.layer },
            self.vector.clone(),
            config.alpha,
        )
        .map(|intervention| intervention.at_positions(config.positions))
    }
}
