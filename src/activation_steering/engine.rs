//! [`ActivationSteering`] — fit the directions, rank the heads, build the
//! interventions, run the steered forward pass.
//!
//! # The pipeline
//!
//! ```text
//!  contrastive pairs
//!         │
//!         ├─ SteerableModel::head_activations ──► one head_dim vector per head slot, per example
//!         │                                                │
//!         │                          split *pairs* (seeded) ├──► train ──► LinearProbe::fit
//!         │                                                └──► validation ──► accuracy
//!         │                                                                        │
//!         │                                        rank by VALIDATION accuracy ◄────┘
//!         │                                                │
//!         │                                        take top-K (excluding degenerate)
//!         │                                                │
//!         │                    direction = unit(probe weights);  sigma = std of projections
//!         │                                                │
//!         └─ SteerableModel::residual_activations ──► CAA: mean(pos) - mean(neg) per layer
//!                                                          │
//!                                        Intervention { site, direction, magnitude }
//!                                                          │
//!                                        SteerableModel::forward_pass  ◄── the mid-forward hook
//! ```
//!
//! # `sigma`, and why `alpha` is unitless
//!
//! An `ITI` intervention shifts a head's activation by `alpha * sigma * theta`,
//! where `theta` is the unit direction and `sigma` is **the standard deviation of
//! the training activations' projections onto `theta`**. Without that `sigma` the
//! same `alpha` would mean something different on every head, because heads differ
//! in activation scale by orders of magnitude — an `alpha` that barely nudges one
//! head would obliterate another. Dividing the shift into "a direction" and "how
//! many standard deviations along it" is what makes a single `alpha` a meaningful
//! knob across all `K` selected heads at once.
//!
//! Two details of `sigma` are worth stating because they are choices, not
//! inevitabilities, and because getting either wrong produces a number that still
//! *looks* like a standard deviation:
//!
//! * It is computed over the **training split only**. Using the whole dataset
//!   would leak the validation split into the intervention.
//! * It is the **pooled** standard deviation over both classes, with a `1/n`
//!   denominator (population, not sample) — matching Li et al.'s reference
//!   implementation. Pooled over a balanced two-class mixture whose means sit at
//!   `±mu` along `theta` with within-class spread `s`, this converges to
//!   `sqrt(s^2 + mu^2)`, not to `s`: the class separation is *part* of the spread
//!   being measured. That is intentional, and the module's tests assert the
//!   `sqrt(s^2 + mu^2)` value rather than the `s` one.

// Sample counts become `f64` to average over them, and the split point is a
// rounded fraction of a count. Neither can lose anything a contrastive dataset
// would notice.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::hidden_states::ModelHiddenStates;

use super::model::SteerableModel;
use super::probe::{LinearProbe, SteeringProbeResult};
use super::rng::SteeringRng;
use super::types::{
    ActivationPair, CaaVector, ContrastivePair, HeadIndex, Intervention, InterventionConfig,
    InterventionSite, ProbeMethod, SteeringConfig, SteeringError, SteeringGeometry, SteeringResult,
    SteeringVector,
};

/// Everything that was learned about one attention head.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadProbeReport {
    /// Which head.
    pub head: HeadIndex,
    /// The probe fitted on it, and what it scored.
    pub result: SteeringProbeResult,
    /// The **unit** direction the intervention runs along: the probe's weight
    /// vector, normalized. All-zero for a [degenerate](HeadProbeReport::degenerate)
    /// head.
    pub direction: SteeringVector,
    /// The standard deviation of the training activations' projections onto
    /// [`direction`](HeadProbeReport::direction) — the `sigma` in `alpha * sigma`.
    /// Zero for a degenerate head.
    pub sigma: f64,
    /// Whether the fitted direction is numerically indistinguishable from zero.
    ///
    /// This happens exactly when the head cannot tell the classes apart at all —
    /// most starkly when its activations for a pair's positive and negative are
    /// *identical*, which makes `mean(pos) - mean(neg)` the zero vector and makes
    /// the logistic probe's optimum `w = 0`.
    ///
    /// A degenerate head is **never selected**. Steering along a zero direction is
    /// a no-op that would nonetheless be reported as a steering vector, with a
    /// direction, a sigma, and a rank — a result that looks exactly like a real one
    /// and does nothing. Rather than emit that, the engine excludes such heads and
    /// raises [`SteeringError::InsufficientHeads`] if too few remain.
    pub degenerate: bool,
}

/// The outcome of fitting `ITI`: every head's probe, and the top-`K` that were
/// selected.
#[derive(Debug, Clone, PartialEq)]
pub struct SteeringReport {
    /// The model's shape.
    pub geometry: SteeringGeometry,
    /// Which probe was fitted.
    pub probe_method: ProbeMethod,
    /// How many contrastive pairs were supplied.
    pub num_pairs: usize,
    /// How many went into the training split.
    pub num_train_pairs: usize,
    /// How many were held out.
    pub num_validation_pairs: usize,
    /// Every head slot, ranked **best validation accuracy first**. Ties are broken
    /// by training accuracy, then by flat head index — a strict total order, so the
    /// ranking is a single well-defined sequence rather than one of several
    /// acceptable ones.
    pub heads: Vec<HeadProbeReport>,
    /// The selected heads, best first. Never contains a degenerate head.
    pub selected: Vec<HeadIndex>,
    /// How many heads carried no usable direction at all.
    pub num_degenerate: usize,
}

impl SteeringReport {
    /// The report for one head.
    #[must_use]
    pub fn head(&self, head: HeadIndex) -> Option<&HeadProbeReport> {
        self.heads.iter().find(|report| report.head == head)
    }

    /// Whether `head` made the top-`K`.
    #[must_use]
    pub fn is_selected(&self, head: HeadIndex) -> bool {
        self.selected.contains(&head)
    }

    /// The reports of the selected heads, best first.
    pub fn selected_reports(&self) -> impl Iterator<Item = &HeadProbeReport> {
        self.selected.iter().filter_map(|head| self.head(*head))
    }

    /// The mean validation accuracy across *all* heads — the baseline against
    /// which the selected heads' accuracies should be read. On a model where only
    /// a few heads encode the concept this sits near chance, and that is the
    /// point.
    #[must_use]
    pub fn mean_validation_accuracy(&self) -> f64 {
        if self.heads.is_empty() {
            return 0.0;
        }
        self.heads
            .iter()
            .map(|report| report.result.accuracy.validation)
            .sum::<f64>()
            / self.heads.len() as f64
    }
}

/// Fits steering directions from contrastive activations, and applies them.
///
/// See the [module documentation](super) for what `ITI` and `CAA` are, and for the
/// two limits of the crate's existing hidden-state machinery that make this module
/// define its own [`SteerableModel`].
#[derive(Debug, Clone)]
pub struct ActivationSteering {
    geometry: SteeringGeometry,
    config: SteeringConfig,
    report: Option<SteeringReport>,
    caa_vectors: Vec<CaaVector>,
}

impl ActivationSteering {
    /// Build an engine for a model of shape `geometry`.
    ///
    /// # Errors
    ///
    /// Propagates [`SteeringConfig::validate`].
    pub fn new(geometry: SteeringGeometry, config: SteeringConfig) -> SteeringResult<Self> {
        config.validate()?;
        Ok(Self {
            geometry,
            config,
            report: None,
            caa_vectors: Vec::new(),
        })
    }

    /// The geometry this engine was built for.
    #[must_use]
    pub const fn geometry(&self) -> SteeringGeometry {
        self.geometry
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &SteeringConfig {
        &self.config
    }

    // ── ITI ──────────────────────────────────────────────────────────────────

    /// Fit `ITI` end to end: collect each pair's per-head activations from
    /// `model`, fit a probe per head, and select the top-`K`.
    ///
    /// # Errors
    ///
    /// Propagates the model's forward passes and
    /// [`ActivationSteering::fit_iti_from_activations`].
    pub fn fit_iti(
        &mut self,
        model: &dyn SteerableModel,
        pairs: &[ContrastivePair],
    ) -> SteeringResult<&SteeringReport> {
        if pairs.is_empty() {
            return Err(SteeringError::NoPairs);
        }
        let mut activations = Vec::with_capacity(pairs.len());
        for pair in pairs {
            activations.push(ActivationPair::new(
                model.head_activations(&pair.positive)?.into_inner(),
                model.head_activations(&pair.negative)?.into_inner(),
            )?);
        }
        self.fit_iti_from_activations(&activations)
    }

    /// Fit `ITI` from activations that have already been collected — from a dump,
    /// from another tool, or from a synthetic generating process whose ground truth
    /// is known.
    ///
    /// Each [`ActivationPair`] must carry one `head_dim`-wide vector per head slot,
    /// indexed by [`SteeringGeometry::flat_index`].
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] on an empty dataset,
    /// [`SteeringError::InsufficientPairs`] when there are too few pairs to split,
    /// [`SteeringError::DimensionMismatch`] when a pair has the wrong number of
    /// sites or a site of the wrong width, and
    /// [`SteeringError::InsufficientHeads`] when fewer than `top_k_heads` heads
    /// carry a usable direction. Propagates [`LinearProbe::fit`].
    pub fn fit_iti_from_activations(
        &mut self,
        pairs: &[ActivationPair],
    ) -> SteeringResult<&SteeringReport> {
        let slots = self.geometry.num_head_slots();
        let head_dim = self.geometry.head_dim();
        check_sites(pairs, slots, head_dim, "head activation")?;

        let (train, validation) =
            split_pairs(pairs.len(), self.config.train_fraction, self.config.seed)?;

        let mut reports = Vec::with_capacity(slots);
        for slot in 0..slots {
            let head = self.geometry.head_at(slot)?;

            let train_positive = gather(pairs, &train, slot, true);
            let train_negative = gather(pairs, &train, slot, false);
            let validation_positive = gather(pairs, &validation, slot, true);
            let validation_negative = gather(pairs, &validation, slot, false);

            let result = LinearProbe::fit(
                self.config.probe_method,
                &train_positive,
                &train_negative,
                &validation_positive,
                &validation_negative,
                &self.config,
            )?;

            // The probe's weight vector *is* the raw direction, for both methods:
            // gradient descent leaves the discriminative direction in `w`, and the
            // mass-mean fit *defines* `w` as `mean(pos) - mean(neg)`.
            let raw = result.probe.weight_vector()?;
            let degenerate = raw.is_degenerate();
            let (direction, sigma) = if degenerate {
                (SteeringVector::zeros(head_dim)?, 0.0)
            } else {
                let direction = raw.normalized()?;
                let sigma = projection_std(&direction, &train_positive, &train_negative)?;
                (direction, sigma)
            };

            reports.push(HeadProbeReport {
                head,
                result,
                direction,
                sigma,
                degenerate,
            });
        }

        // Rank by validation accuracy, breaking ties by training accuracy and then
        // by head index. `total_cmp` rather than `partial_cmp`: an `f64` comparison
        // that can return `None` has no place in a selection whose whole job is to
        // be reproducible, and the two accuracies here are counts over a sample, so
        // exact ties are common rather than exotic.
        reports.sort_by(|left, right| {
            right
                .result
                .accuracy
                .validation
                .total_cmp(&left.result.accuracy.validation)
                .then_with(|| {
                    right
                        .result
                        .accuracy
                        .train
                        .total_cmp(&left.result.accuracy.train)
                })
                .then_with(|| left.head.cmp(&right.head))
        });

        let num_degenerate = reports.iter().filter(|report| report.degenerate).count();
        let available = reports.len() - num_degenerate;
        if available < self.config.top_k_heads {
            return Err(SteeringError::InsufficientHeads {
                requested: self.config.top_k_heads,
                available,
                total: reports.len(),
            });
        }
        let selected: Vec<HeadIndex> = reports
            .iter()
            .filter(|report| !report.degenerate)
            .take(self.config.top_k_heads)
            .map(|report| report.head)
            .collect();

        self.report = Some(SteeringReport {
            geometry: self.geometry,
            probe_method: self.config.probe_method,
            num_pairs: pairs.len(),
            num_train_pairs: train.len(),
            num_validation_pairs: validation.len(),
            heads: reports,
            selected,
            num_degenerate,
        });
        self.report()
    }

    /// The `ITI` report.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NotFitted`] before `fit_iti*` has been called.
    pub fn report(&self) -> SteeringResult<&SteeringReport> {
        self.report.as_ref().ok_or(SteeringError::NotFitted {
            expected: "fit_iti",
        })
    }

    /// The selected heads, best first.
    ///
    /// # Errors
    ///
    /// Propagates [`ActivationSteering::report`].
    pub fn selected_heads(&self) -> SteeringResult<&[HeadIndex]> {
        Ok(&self.report()?.selected)
    }

    /// The `ITI` interventions: one per selected head, each shifting that head's
    /// output by `alpha * sigma` along its unit direction.
    ///
    /// # Errors
    ///
    /// Propagates [`ActivationSteering::report`], and returns
    /// [`SteeringError::NonFinite`] when `alpha * sigma` is not finite.
    pub fn iti_interventions(
        &self,
        config: &InterventionConfig,
    ) -> SteeringResult<Vec<Intervention>> {
        let report = self.report()?;
        let mut interventions = Vec::with_capacity(report.selected.len());
        for head in &report.selected {
            let Some(entry) = report.head(*head) else {
                // Unreachable: `selected` is built from `heads`.
                return Err(SteeringError::NotFitted {
                    expected: "fit_iti",
                });
            };
            let intervention = Intervention::new(
                InterventionSite::Head(entry.head),
                entry.direction.clone(),
                config.alpha * entry.sigma,
            )?
            .at_positions(config.positions);
            interventions.push(intervention);
        }
        Ok(interventions)
    }

    // ── CAA ──────────────────────────────────────────────────────────────────

    /// Fit `CAA` end to end: collect each pair's per-layer residual stream from
    /// `model` and average the differences.
    ///
    /// # Errors
    ///
    /// Propagates the model's forward passes and
    /// [`ActivationSteering::fit_caa_from_activations`].
    pub fn fit_caa(
        &mut self,
        model: &dyn SteerableModel,
        pairs: &[ContrastivePair],
    ) -> SteeringResult<&[CaaVector]> {
        if pairs.is_empty() {
            return Err(SteeringError::NoPairs);
        }
        let mut activations = Vec::with_capacity(pairs.len());
        for pair in pairs {
            activations.push(ActivationPair::new(
                model.residual_activations(&pair.positive)?.into_inner(),
                model.residual_activations(&pair.negative)?.into_inner(),
            )?);
        }
        self.fit_caa_from_activations(&activations)
    }

    /// Fit `CAA` from residual streams that have already been collected.
    ///
    /// Each [`ActivationPair`] must carry one `hidden_dim`-wide vector per layer.
    ///
    /// # Why there is no train/validation split here
    ///
    /// `CAA` fits no classifier and selects nothing. Its vector is a *descriptive
    /// statistic* of the dataset — a mean — so there is nothing to overfit and
    /// nothing to rank. The split exists in `ITI` to keep head **selection**
    /// honest, and holding out pairs here would only make the mean noisier for no
    /// gain. Every pair contributes.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NoPairs`] on an empty dataset,
    /// [`SteeringError::DimensionMismatch`] when a pair has the wrong number of
    /// layers or a layer of the wrong width, and
    /// [`SteeringError::LayerOutOfRange`] when a configured `caa_layer` is outside
    /// the model.
    pub fn fit_caa_from_activations(
        &mut self,
        pairs: &[ActivationPair],
    ) -> SteeringResult<&[CaaVector]> {
        let num_layers = self.geometry.num_layers();
        let hidden_dim = self.geometry.hidden_dim();
        check_sites(pairs, num_layers, hidden_dim, "residual stream")?;

        let layers: Vec<usize> = if self.config.caa_layers.is_empty() {
            (0..num_layers).collect()
        } else {
            for &layer in &self.config.caa_layers {
                self.geometry.check_layer(layer)?;
            }
            let mut layers = self.config.caa_layers.clone();
            layers.sort_unstable();
            layers.dedup();
            layers
        };

        let count = pairs.len() as f64;
        let mut vectors = Vec::with_capacity(layers.len());
        for layer in layers {
            // The mean of the *paired* differences. For a complete pairing this
            // equals the difference of the class means exactly — a fact the module's
            // tests assert — but it is computed pair-wise because that is what CAA
            // *is*: the contrast is defined within a pair, where the two prompts
            // differ in the behaviour and in nothing else.
            let mut mean = vec![0.0_f64; hidden_dim];
            for pair in pairs {
                for ((slot, positive), negative) in mean
                    .iter_mut()
                    .zip(&pair.positive[layer])
                    .zip(&pair.negative[layer])
                {
                    *slot += f64::from(*positive) - f64::from(*negative);
                }
            }
            for slot in &mut mean {
                *slot /= count;
            }
            vectors.push(CaaVector {
                layer,
                vector: SteeringVector::new(mean)?,
                num_pairs: pairs.len(),
            });
        }

        self.caa_vectors = vectors;
        Ok(&self.caa_vectors)
    }

    /// The fitted `CAA` vectors, one per configured layer.
    #[must_use]
    pub fn caa_vectors(&self) -> &[CaaVector] {
        &self.caa_vectors
    }

    /// The `CAA` vector for one layer.
    #[must_use]
    pub fn caa_vector(&self, layer: usize) -> Option<&CaaVector> {
        self.caa_vectors.iter().find(|vector| vector.layer == layer)
    }

    /// The `CAA` interventions: one per fitted layer, each adding `alpha * v` to
    /// that layer's output residual.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NotFitted`] before `fit_caa*` has been called, and
    /// propagates [`CaaVector::intervention`].
    pub fn caa_interventions(
        &self,
        config: &InterventionConfig,
    ) -> SteeringResult<Vec<Intervention>> {
        if self.caa_vectors.is_empty() {
            return Err(SteeringError::NotFitted {
                expected: "fit_caa",
            });
        }
        self.caa_vectors
            .iter()
            .map(|vector| vector.intervention(config))
            .collect()
    }

    // ── Applying them ────────────────────────────────────────────────────────

    /// Every intervention this engine has been fitted for — `ITI`'s if `fit_iti`
    /// ran, `CAA`'s if `fit_caa` ran, both if both did.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NotFitted`] when neither has been fitted.
    pub fn all_interventions(
        &self,
        config: &InterventionConfig,
    ) -> SteeringResult<Vec<Intervention>> {
        let mut interventions = Vec::new();
        if self.report.is_some() {
            interventions.extend(self.iti_interventions(config)?);
        }
        if !self.caa_vectors.is_empty() {
            interventions.extend(self.caa_interventions(config)?);
        }
        if interventions.is_empty() {
            return Err(SteeringError::NotFitted {
                expected: "fit_iti or fit_caa",
            });
        }
        Ok(interventions)
    }

    /// Run `model` on `text` with every fitted intervention applied
    /// **mid-forward**, and return the resulting logits.
    ///
    /// # Errors
    ///
    /// Propagates [`ActivationSteering::all_interventions`] and the model's
    /// forward pass.
    pub fn steer(
        &self,
        model: &dyn SteerableModel,
        text: &str,
        config: &InterventionConfig,
    ) -> SteeringResult<Vec<f32>> {
        let interventions = self.all_interventions(config)?;
        model.forward_with_interventions(text, &interventions)
    }

    // ── The captured-state bridge, and its limit ─────────────────────────────

    /// Add `alpha * v` to the residual vectors stored in a **captured**
    /// [`ModelHiddenStates`], at layer `layer`.
    ///
    /// Returns how many token positions were edited.
    ///
    /// # This is a post-hoc edit, not a mid-forward hook
    ///
    /// [`ModelHiddenStates`] is an *owned snapshot*: by the time
    /// [`HiddenStateProvider::extract_hidden_states`](crate::hidden_states::HiddenStateProvider::extract_hidden_states)
    /// returns one, the forward pass is over. Writing into `layers[l]` changes
    /// **that stored tensor and nothing else**. Layer `l + 1`'s stored tensor was
    /// computed from the *pre-edit* value of layer `l` and is not recomputed; no
    /// logits are produced at all, because none were kept. The module's test
    /// `captured_edit_does_not_propagate_to_a_later_layer` demonstrates precisely
    /// that, rather than asking you to take it on faith.
    ///
    /// So this method is useful — and only useful — when the **captured
    /// representation itself is the thing you consume**: a pooled embedding fed to
    /// a retriever, a classifier, or a similarity search. Steering it steers those.
    /// It does not steer what the model *says*. For that you need
    /// [`SteerableModel::forward_pass`], which is why this module defines that
    /// trait at all.
    ///
    /// # Errors
    ///
    /// Returns [`SteeringError::NotFitted`] when `fit_caa*` has not run,
    /// [`SteeringError::LayerOutOfRange`] when no `CAA` vector was fitted for
    /// `layer` or the capture has no such layer, and
    /// [`SteeringError::UnsteerableCapture`] when the captured tensor is not shaped
    /// `[1, seq_len, hidden_dim]`.
    pub fn edit_captured_residual(
        &self,
        states: &mut ModelHiddenStates,
        layer: usize,
        config: &InterventionConfig,
    ) -> SteeringResult<usize> {
        let hidden_dim = self.geometry.hidden_dim();
        let vector = self
            .caa_vector(layer)
            .ok_or(if self.caa_vectors.is_empty() {
                SteeringError::NotFitted {
                    expected: "fit_caa",
                }
            } else {
                SteeringError::LayerOutOfRange {
                    layer,
                    num_layers: self.geometry.num_layers(),
                }
            })?;
        let delta = vector.intervention(config)?.delta_f32();

        let captured_layers = states.layers.len();
        let captured = states
            .layers
            .iter_mut()
            .find(|entry| entry.layer_idx == layer)
            .ok_or(SteeringError::LayerOutOfRange {
                layer,
                num_layers: captured_layers,
            })?;

        // `[1, seq_len, hidden_dim]` is what the crate's extractors produce, and it
        // is the only layout in which "position t's residual vector" is a
        // well-defined contiguous slice.
        let dims = &captured.hidden_state.shape.dims;
        if dims.len() != 3 || dims[0] != 1 || dims[2] != hidden_dim {
            return Err(SteeringError::UnsteerableCapture {
                layer,
                shape: dims.clone(),
                hidden_dim,
            });
        }
        let sequence_length = dims[1];

        let mut edited = 0usize;
        for position in 0..sequence_length {
            if !config.positions.includes(position, sequence_length) {
                continue;
            }
            for (offset, shift) in delta.iter().enumerate() {
                let index = position * hidden_dim + offset;
                let slot = captured.hidden_state.data.get_mut(index).ok_or(
                    SteeringError::UnsteerableCapture {
                        layer,
                        shape: captured.hidden_state.shape.dims.clone(),
                        hidden_dim,
                    },
                )?;
                *slot += shift;
            }
            edited += 1;
        }
        Ok(edited)
    }
}

// ── Free functions ───────────────────────────────────────────────────────────

/// Split `num_pairs` **pairs** into a training and a validation index set.
///
/// Pairs, not activations: both halves of a contrastive pair land on the same side
/// of the split. Splitting at the activation level would put a pair's positive in
/// train and its negative in validation, and since the two differ only in the
/// concept, that is a textbook leak — the validation accuracy it produced would be
/// an overstatement, and head selection ranks on exactly that number.
///
/// The training size is `clamp(round(n * train_fraction), 1, n - 1)`, so **both
/// splits are always non-empty**: a validation accuracy measured on zero examples
/// is not a small sample, it is a fabrication.
///
/// # Errors
///
/// Returns [`SteeringError::InsufficientPairs`] when `num_pairs < 2`.
pub fn split_pairs(
    num_pairs: usize,
    train_fraction: f64,
    seed: u64,
) -> SteeringResult<(Vec<usize>, Vec<usize>)> {
    if num_pairs < 2 {
        return Err(SteeringError::InsufficientPairs {
            needed: 2,
            actual: num_pairs,
            train_fraction,
        });
    }
    let mut indices: Vec<usize> = (0..num_pairs).collect();
    SteeringRng::new(seed).shuffle(&mut indices);

    let requested = (num_pairs as f64 * train_fraction).round();
    let train_size = (requested.max(1.0) as usize).min(num_pairs - 1);
    let validation = indices.split_off(train_size);
    Ok((indices, validation))
}

/// Collect one site's activations for one class over a set of pair indices.
fn gather<'a>(
    pairs: &'a [ActivationPair],
    indices: &[usize],
    site: usize,
    positive: bool,
) -> Vec<&'a [f32]> {
    indices
        .iter()
        .filter_map(|&index| pairs.get(index))
        .map(|pair| {
            let class = if positive {
                &pair.positive
            } else {
                &pair.negative
            };
            class[site].as_slice()
        })
        .collect()
}

/// Check that every pair carries `sites` activations of width `width`.
fn check_sites(
    pairs: &[ActivationPair],
    sites: usize,
    width: usize,
    what: &'static str,
) -> SteeringResult<()> {
    if pairs.is_empty() {
        return Err(SteeringError::NoPairs);
    }
    for pair in pairs {
        if pair.num_sites() != sites {
            return Err(SteeringError::DimensionMismatch {
                what,
                expected: sites,
                actual: pair.num_sites(),
            });
        }
        for class in [&pair.positive, &pair.negative] {
            for activation in class {
                if activation.len() != width {
                    return Err(SteeringError::DimensionMismatch {
                        what,
                        expected: width,
                        actual: activation.len(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// The **population** standard deviation (`1/n`, not `1/(n-1)`) of the training
/// activations' projections onto `direction`, pooled over both classes.
///
/// This is `sigma`. See the [module documentation](self#sigma-and-why-alpha-is-unitless)
/// for why it is pooled, why it is over the training split only, and why it
/// converges to `sqrt(s^2 + mu^2)` rather than to the within-class spread `s`.
fn projection_std(
    direction: &SteeringVector,
    positive: &[&[f32]],
    negative: &[&[f32]],
) -> SteeringResult<f64> {
    let count = positive.len() + negative.len();
    if count == 0 {
        return Err(SteeringError::NoPairs);
    }
    let mut projections = Vec::with_capacity(count);
    for activation in positive.iter().chain(negative.iter()) {
        projections.push(direction.project(activation)?);
    }
    let count_f = count as f64;
    let mean = projections.iter().sum::<f64>() / count_f;
    let variance = projections
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count_f;
    Ok(variance.sqrt())
}
