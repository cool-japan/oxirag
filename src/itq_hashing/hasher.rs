//! The ITQ hasher: PCA dimensionality reduction, the alternating-minimization
//! engine that *learns* the optimal binary-coding rotation via the orthogonal
//! Procrustes problem, and per-vector encoding.
//!
//! # The algorithm (Gong & Lazebnik, CVPR 2011)
//!
//! Given training vectors `X` (`n x D`):
//!
//! 1. **PCA.** Center `X`, form the `D x D` covariance
//!    `C = (1/n) X_c^T X_c`, and take the top-`k` eigenvectors `V` (the
//!    directions of maximal variance) via the module's own symmetric
//!    eigensolver. Project the data: `Z = X_c · V` (`n x k`).
//! 2. **Learn the rotation** by alternating minimization of the quantization
//!    loss `‖B − Z R‖_F^2`, where `B ∈ {-1, +1}^{n×k}` and `R` is `k x k`
//!    orthogonal. Each iteration performs two closed-form global minimizations.
//!    The B-step fixes `R` and sets `B = sign(Z R)` (with the `sign(0) := +1`
//!    convention). The R-step fixes `B` and solves the orthogonal Procrustes
//!    problem `M = Z^T B = U Σ V_m^T`, `R = U V_m^T`. Because each sub-step is
//!    a global minimizer, the objective is non-increasing across iterations and
//!    the loop converges.
//! 3. **Encode.** For a new vector `x`: `z = (x − mu) · V`, `t = z R`, and the
//!    code bit is `t_i >= 0` (again `sign(0) := +1`).
//!
//! Unlike a fixed random rotation (as in the sibling `rabitq` module), `R`
//! here is *learned from the data distribution* — that data-driven refinement
//! of the quantization axes is the entire point of ITQ.
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use super::linalg::{jacobi_symmetric, orthogonal_procrustes, random_orthogonal};
use super::types::{ItqCode, ItqConfig, ItqError};

// ── ItqHasher ─────────────────────────────────────────────────────────────────

/// A trained ITQ hasher: holds the PCA centering mean, the top-`k` principal
/// components `V`, and the learned orthogonal rotation `R`, and turns a vector
/// into an [`ItqCode`].
#[derive(Debug, Clone)]
pub struct ItqHasher {
    dim: usize,
    num_bits: usize,
    /// Per-column mean of the training data (length `D`) subtracted on encode.
    mean: Vec<f64>,
    /// The top-`k` PCA eigenvectors, one per row: `components[j]` is the `j`-th
    /// principal component (length `D`). Projection is `z_j = <x − mu, V_j>`.
    components: Vec<Vec<f64>>,
    /// The learned `k x k` orthogonal rotation `R`, row-major: rotated
    /// coordinate `t_j = sum_i z_i · R[i][j]`.
    rotation: Vec<Vec<f64>>,
    /// The Frobenius quantization objective `‖B − Z R‖_F^2` recorded after each
    /// alternating-minimization iteration (non-increasing; useful for
    /// diagnostics and convergence checks).
    loss_history: Vec<f64>,
}

impl ItqHasher {
    /// Train an ITQ hasher on `vectors`, learning both the PCA basis and the
    /// binary-coding rotation.
    ///
    /// # Errors
    ///
    /// - [`ItqError::InvalidConfig`] if `config` fails validation, or if the
    ///   vectors have zero dimensionality.
    /// - [`ItqError::EmptyDataset`] if `vectors` is empty.
    /// - [`ItqError::DimensionMismatch`] if the vectors have inconsistent
    ///   lengths.
    /// - [`ItqError::NonFinite`] if any coordinate is `NaN`/infinite.
    /// - [`ItqError::NumBitsExceedsDimension`] if `config.num_bits > D`.
    /// - [`ItqError::Numerical`] on an internal eigensolver/SVD failure.
    pub fn train(vectors: &[Vec<f32>], config: &ItqConfig) -> Result<Self, ItqError> {
        config.validate()?;
        if vectors.is_empty() {
            return Err(ItqError::EmptyDataset);
        }

        let dim = vectors[0].len();
        if dim == 0 {
            return Err(ItqError::InvalidConfig(
                "vector dimensionality must be greater than zero".into(),
            ));
        }
        for v in vectors {
            if v.len() != dim {
                return Err(ItqError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            for &value in v {
                if !value.is_finite() {
                    return Err(ItqError::NonFinite);
                }
            }
        }

        let num_bits = config.num_bits;
        if num_bits > dim {
            return Err(ItqError::NumBitsExceedsDimension { num_bits, dim });
        }

        // Step 1a: per-column mean and centered data (in f64).
        let n = vectors.len();
        let inv_n = 1.0 / n as f64;
        let mut mean = vec![0.0_f64; dim];
        for v in vectors {
            for (m, &value) in mean.iter_mut().zip(v.iter()) {
                *m += f64::from(value);
            }
        }
        for m in &mut mean {
            *m *= inv_n;
        }
        let centered: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| {
                v.iter()
                    .zip(mean.iter())
                    .map(|(&value, &m)| f64::from(value) - m)
                    .collect()
            })
            .collect();

        // Step 1b: D x D covariance C = (1/n) X_c^T X_c (symmetric PSD).
        let mut covariance = vec![vec![0.0_f64; dim]; dim];
        for row in &centered {
            for i in 0..dim {
                let ri = row[i];
                for j in i..dim {
                    covariance[i][j] += ri * row[j];
                }
            }
        }
        for i in 0..dim {
            for j in i..dim {
                let value = covariance[i][j] * inv_n;
                covariance[i][j] = value;
                covariance[j][i] = value;
            }
        }

        // Step 1c: top-k eigenvectors (descending eigenvalue order).
        let (_eigenvalues, eigenvectors) =
            jacobi_symmetric(&covariance, config.jacobi_max_sweeps, config.jacobi_tol)?;
        let mut components = vec![vec![0.0_f64; dim]; num_bits];
        for j in 0..num_bits {
            for d in 0..dim {
                components[j][d] = eigenvectors[d][j];
            }
        }

        // Step 1d: project the training data, Z = X_c · V (n x k).
        let z: Vec<Vec<f64>> = centered
            .iter()
            .map(|row| {
                components
                    .iter()
                    .map(|component| dot_slices(row, component))
                    .collect()
            })
            .collect();

        // Step 2: alternating minimization to learn R.
        let (rotation, loss_history) = learn_rotation(&z, num_bits, config)?;

        Ok(Self {
            dim,
            num_bits,
            mean,
            components,
            rotation,
            loss_history,
        })
    }

    /// The input dimensionality `D` this hasher was trained on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The code length `k` (number of bits) this hasher produces.
    #[must_use]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// The PCA centering mean (length `D`).
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The top-`k` PCA components, one per row (`components()[j]` has length
    /// `D`).
    #[must_use]
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// The learned `k x k` orthogonal rotation `R` (row-major).
    #[must_use]
    pub fn rotation(&self) -> &[Vec<f64>] {
        &self.rotation
    }

    /// The Frobenius quantization objective recorded after each
    /// alternating-minimization iteration (non-increasing).
    #[must_use]
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Project `x` through PCA and the learned rotation, returning the rotated
    /// coordinates `t` (length `k`) before thresholding. Exposed primarily for
    /// diagnostics and tests.
    ///
    /// # Errors
    ///
    /// - [`ItqError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`ItqError::NonFinite`] if any coordinate is `NaN`/infinite.
    pub fn project(&self, x: &[f32]) -> Result<Vec<f64>, ItqError> {
        if x.len() != self.dim {
            return Err(ItqError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        for &value in x {
            if !value.is_finite() {
                return Err(ItqError::NonFinite);
            }
        }
        Ok(self.project_unchecked(x))
    }

    /// Encode `x` into its ITQ [`ItqCode`].
    ///
    /// Bit `i` is set when the `i`-th rotated coordinate is non-negative
    /// (`sign(0) := +1`), matching the `{-1, +1}` code matrix used during
    /// training.
    ///
    /// # Errors
    ///
    /// - [`ItqError::DimensionMismatch`] if `x.len() != dim`.
    /// - [`ItqError::NonFinite`] if any coordinate is `NaN`/infinite.
    pub fn encode(&self, x: &[f32]) -> Result<ItqCode, ItqError> {
        let rotated = self.project(x)?;
        let word_count = self.num_bits.div_ceil(64);
        let mut bits = vec![0u64; word_count];
        for (j, &value) in rotated.iter().enumerate() {
            if value >= 0.0 {
                bits[j / 64] |= 1u64 << (j % 64);
            }
        }
        Ok(ItqCode::new(bits, self.num_bits))
    }

    /// Project + rotate without input validation (callers guarantee the length
    /// and finiteness invariants).
    fn project_unchecked(&self, x: &[f32]) -> Vec<f64> {
        // z_j = <x − mu, V_j>
        let z: Vec<f64> = self
            .components
            .iter()
            .map(|component| {
                component
                    .iter()
                    .zip(x.iter())
                    .zip(self.mean.iter())
                    .map(|((&c, &xi), &m)| c * (f64::from(xi) - m))
                    .sum()
            })
            .collect();
        // t_j = sum_i z_i · R[i][j]
        (0..self.num_bits)
            .map(|j| (0..self.num_bits).map(|i| z[i] * self.rotation[i][j]).sum())
            .collect()
    }
}

// ── alternating minimization ──────────────────────────────────────────────────

/// Dot product of two equal-length `f64` slices.
fn dot_slices(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// The binary code matrix `B = sign(Z R)` for the current rotation, with
/// `sign(0) := +1`. Result rows hold `{-1.0, +1.0}` entries.
fn binarize(z: &[Vec<f64>], r: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    z.iter()
        .map(|z_row| {
            (0..k)
                .map(|j| {
                    let t: f64 = (0..k).map(|i| z_row[i] * r[i][j]).sum();
                    if t >= 0.0 { 1.0 } else { -1.0 }
                })
                .collect()
        })
        .collect()
}

/// The `k x k` matrix `M = Z^T B`.
fn z_transpose_b(z: &[Vec<f64>], b: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; k]; k];
    for (z_row, b_row) in z.iter().zip(b.iter()) {
        for i in 0..k {
            let zi = z_row[i];
            for j in 0..k {
                m[i][j] += zi * b_row[j];
            }
        }
    }
    m
}

/// The Frobenius quantization objective `‖B − Z R‖_F^2`.
fn frobenius_loss(z: &[Vec<f64>], r: &[Vec<f64>], b: &[Vec<f64>], k: usize) -> f64 {
    let mut acc = 0.0_f64;
    for (z_row, b_row) in z.iter().zip(b.iter()) {
        for j in 0..k {
            let t: f64 = (0..k).map(|i| z_row[i] * r[i][j]).sum();
            let diff = b_row[j] - t;
            acc += diff * diff;
        }
    }
    acc
}

/// Learn the `k x k` orthogonal rotation `R` by alternating minimization,
/// returning `R` together with the per-iteration objective trajectory.
///
/// The rotation is initialized to a deterministic pseudo-random orthogonal
/// matrix; each iteration then performs the closed-form B-step followed by the
/// closed-form Procrustes R-step, records `‖B − Z R‖_F^2`, and stops early once
/// the objective's decrease falls below `config.tolerance`.
fn learn_rotation(
    z: &[Vec<f64>],
    k: usize,
    config: &ItqConfig,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), ItqError> {
    let mut r = random_orthogonal(k, config.seed)?;
    let mut history = Vec::with_capacity(config.max_iterations);
    let mut previous = f64::INFINITY;

    for _ in 0..config.max_iterations {
        // B-step: minimize over B for the current R.
        let b = binarize(z, &r, k);
        // R-step: minimize over orthogonal R for the current B (Procrustes).
        let m = z_transpose_b(z, &b, k);
        let r_new = orthogonal_procrustes(&m, config.jacobi_max_sweeps, config.jacobi_tol)?;
        let loss = frobenius_loss(z, &r_new, &b, k);
        history.push(loss);
        r = r_new;

        if previous - loss < config.tolerance {
            break;
        }
        previous = loss;
    }

    Ok((r, history))
}
