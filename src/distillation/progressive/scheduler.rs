//! Progressive distillation scheduler.
//!
//! A `ProgressiveScheduler` defines how model size evolves across the sequence
//! of distillation stages.  Three strategies are available:
//!
//! - **Linear** – model parameters decrease linearly from teacher to target.
//! - **Exponential** – model parameters decrease along an exponential curve
//!   controlled by a decay factor.
//! - **Custom** – an explicit list of boundary sizes supplied by the caller.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use serde::{Deserialize, Serialize};

use super::types::{ModelSize, StageConfig};

// ────────────────────────────────────────────────────────────────────────────
// ProgressiveScheduler
// ────────────────────────────────────────────────────────────────────────────

/// Determines how model sizes evolve across progressive distillation stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressiveScheduler {
    /// Model parameters decrease linearly from `start_size` to `end_size`
    /// over `num_stages` stages.
    Linear {
        /// Model size at the beginning (teacher).
        start_size: ModelSize,
        /// Target model size at the end (final student).
        end_size: ModelSize,
        /// Number of distillation stages.
        num_stages: usize,
    },
    /// Model parameters decrease exponentially from `start_size` to
    /// `end_size` over `num_stages` stages, with curvature controlled by
    /// `decay_factor`.
    Exponential {
        /// Model size at the beginning (teacher).
        start_size: ModelSize,
        /// Target model size at the end (final student).
        end_size: ModelSize,
        /// Number of distillation stages.
        num_stages: usize,
        /// Decay factor in `(0.0, 1.0)`.  Lower values produce a steeper
        /// initial drop.
        decay_factor: f64,
    },
    /// Caller-supplied sequence of boundary sizes.
    ///
    /// The number of stages equals `sizes.len() - 1`.
    Custom {
        /// Sequence of model sizes from teacher to final student (inclusive
        /// of both endpoints).
        sizes: Vec<ModelSize>,
    },
}

impl ProgressiveScheduler {
    /// Create a linear scheduler.
    #[must_use]
    pub const fn linear(start_size: ModelSize, end_size: ModelSize, num_stages: usize) -> Self {
        Self::Linear {
            start_size,
            end_size,
            num_stages,
        }
    }

    /// Create an exponential scheduler.
    #[must_use]
    pub const fn exponential(
        start_size: ModelSize,
        end_size: ModelSize,
        num_stages: usize,
        decay_factor: f64,
    ) -> Self {
        Self::Exponential {
            start_size,
            end_size,
            num_stages,
            decay_factor,
        }
    }

    /// Create a custom scheduler from an explicit list of boundary sizes.
    #[must_use]
    pub const fn custom(sizes: Vec<ModelSize>) -> Self {
        Self::Custom { sizes }
    }

    /// Number of distillation stages produced by this scheduler.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        match self {
            Self::Linear { num_stages, .. } | Self::Exponential { num_stages, .. } => *num_stages,
            Self::Custom { sizes } => sizes.len().saturating_sub(1),
        }
    }

    /// Generate the full sequence of `ModelSize` boundary points.
    ///
    /// For `Linear` and `Exponential` schedulers this produces
    /// `num_stages + 1` entries (start through end, inclusive).
    /// For `Custom` it returns the sizes as provided.
    #[must_use]
    pub fn generate_sizes(&self) -> Vec<ModelSize> {
        match self {
            Self::Linear {
                start_size,
                end_size,
                num_stages,
            } => {
                let mut sizes = Vec::with_capacity(*num_stages + 1);
                for i in 0..=*num_stages {
                    let t = if *num_stages == 0 {
                        1.0
                    } else {
                        i as f64 / *num_stages as f64
                    };
                    let params =
                        start_size.params_millions * (1.0 - t) + end_size.params_millions * t;
                    let layers = ((start_size.num_layers as f64 * (1.0 - t)
                        + end_size.num_layers as f64 * t)
                        .round()) as usize;
                    let hidden = ((start_size.hidden_dim as f64 * (1.0 - t)
                        + end_size.hidden_dim as f64 * t)
                        .round()) as usize;
                    sizes.push(ModelSize::new(params, layers, hidden));
                }
                sizes
            }
            Self::Exponential {
                start_size,
                end_size,
                num_stages,
                decay_factor,
            } => {
                let mut sizes = Vec::with_capacity(*num_stages + 1);
                for i in 0..=*num_stages {
                    let t = if *num_stages == 0 {
                        1.0
                    } else {
                        i as f64 / *num_stages as f64
                    };
                    // Exponential interpolation: faster reduction early.
                    let factor = (1.0 - decay_factor).powf(t);
                    let params = end_size.params_millions
                        + (start_size.params_millions - end_size.params_millions) * factor;
                    let layers = ((end_size.num_layers as f64
                        + (start_size.num_layers as f64 - end_size.num_layers as f64) * factor)
                        .round()) as usize;
                    let hidden = ((end_size.hidden_dim as f64
                        + (start_size.hidden_dim as f64 - end_size.hidden_dim as f64) * factor)
                        .round()) as usize;
                    sizes.push(ModelSize::new(params, layers, hidden));
                }
                sizes
            }
            Self::Custom { sizes } => sizes.clone(),
        }
    }

    /// Generate a `Vec<StageConfig>` by sliding a two-element window over the
    /// boundary sizes and cloning `base_config` for each window pair.
    #[must_use]
    pub fn generate_stage_configs(&self, base_config: &StageConfig) -> Vec<StageConfig> {
        let sizes = self.generate_sizes();
        if sizes.len() < 2 {
            return Vec::new();
        }

        sizes
            .windows(2)
            .enumerate()
            .map(|(idx, window)| {
                let teacher_size = window[0];
                let student_size = window[1];
                StageConfig {
                    teacher_size,
                    student_size,
                    stage_name: Some(format!("Stage {}", idx + 1)),
                    ..base_config.clone()
                }
            })
            .collect()
    }
}

impl Default for ProgressiveScheduler {
    fn default() -> Self {
        Self::Linear {
            start_size: ModelSize::from_params(7000.0),
            end_size: ModelSize::from_params(1000.0),
            num_stages: 3,
        }
    }
}
