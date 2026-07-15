//! [`EditableMemory`] — one linear associative memory site, plus the second-moment matrix of
//! the keys it must preserve.
//!
//! A transformer's `MLP` down-projection is, structurally, a linear key → value store: the
//! post-activation vector `k` is a key, `W k` is the value read out of it, and the fact
//! "`subject` has `relation` `object`" lives in the matrix as the association between one key
//! and one value. That is the whole premise of `ROME` (Meng et al., 2022, *Locating and Editing
//! Factual Associations in GPT*). This type is that store, in isolation and in `f64`, with the
//! two objects an edit needs:
//!
//! * `W` — the `m × d` weight matrix, held twice: the pristine `W_0` it was constructed with,
//!   and the current `W'` after however many edits. Keeping `W_0` is not sentimentality; the
//!   `MEMIT`-style joint re-solve *requires* it, because the cumulative delta is defined as the
//!   minimum-norm matrix taking `W_0` to something satisfying all accumulated constraints at
//!   once. It is also what makes the collateral drift measurable at all: drift is
//!   `||(W' - W_0) k_i||`, and without `W_0` there is nothing to measure against.
//! * `C` — the `d × d` second-moment matrix `E[k k^T] + ridge * I` of the keys the edit must
//!   *not* disturb, factored once by Cholesky and applied forever after by triangular
//!   substitution.

use serde::{Deserialize, Serialize};

use super::linalg::{
    self, KnowledgeEditLinalgError, cholesky_with_jitter, frobenius_norm_sq, mat_vec,
    scaled_identity, spd_quadratic_form, spd_solve, symmetrize,
    weighted_frobenius_norm_sq_from_cholesky,
};
use super::types::KnowledgeEditError;

/// One linear memory site: an `m × d` weight matrix plus the second-moment matrix of the keys
/// it is required to preserve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditableMemory {
    label: String,
    rows: usize,
    cols: usize,
    base_weights: Vec<f64>,
    weights: Vec<f64>,
    covariance: Vec<f64>,
    cholesky: Vec<f64>,
    ridge: f64,
    jitter: f64,
    preserved_keys: Vec<Vec<f64>>,
}

impl EditableMemory {
    /// Build a site from an explicit set of **preserved keys** — the keys whose readouts the
    /// edit must leave alone.
    ///
    /// The second-moment matrix is the *mean* outer product plus a **relative** ridge:
    ///
    /// ```text
    /// M = (1/N) * sum_i k_i k_i^T
    /// C = M + ridge_factor * (trace(M) / d) * I
    /// ```
    ///
    /// The ridge is relative rather than absolute on purpose. An absolute `1e-6` means one
    /// thing for keys of norm `1` and something completely different for keys of norm `1000`;
    /// tying it to the mean diagonal of `M` makes the resulting edit **exactly invariant** to a
    /// rescaling of the key set, because scaling every `k_i` by `a` scales `C` by `a^2`, and
    /// the rank-1 formula `Delta = r (C^-1 k*)^T / (k*^T C^-1 k*)` has `C^-1` in both numerator
    /// and denominator. (When `M` is degenerate — no keys, or all of them zero — the scale
    /// falls back to `1.0` so that `C` is still positive definite and the edit degrades
    /// gracefully to the minimum-Euclidean-norm update.)
    ///
    /// The keys are **retained**, which is what lets [`Self::drift_report`] report a *measured*
    /// collateral drift rather than only the analytic bound.
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::InvalidConfig`] if a dimension is zero or the ridge factor is
    ///   not finite and positive.
    /// * [`KnowledgeEditError::DimensionMismatch`] if `weights` is not `rows * cols` long, or a
    ///   key is not `cols` long.
    /// * [`KnowledgeEditError::NonFinite`] if any weight or key coordinate is a `NaN` or an
    ///   infinity.
    /// * [`KnowledgeEditError::Linalg`] if `C` will not factor even with a ridge — which,
    ///   given the ridge above, means the inputs were pathological.
    pub fn from_preserved_keys(
        label: impl Into<String>,
        weights: Vec<f64>,
        rows: usize,
        cols: usize,
        preserved_keys: &[Vec<f64>],
        ridge_factor: f64,
    ) -> Result<Self, KnowledgeEditError> {
        check_shape(&weights, rows, cols)?;
        if !ridge_factor.is_finite() || ridge_factor <= 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "covariance ridge factor must be finite and positive, got {ridge_factor}"
            )));
        }
        for key in preserved_keys {
            if key.len() != cols {
                return Err(KnowledgeEditError::DimensionMismatch {
                    what: "preserved key",
                    expected: cols,
                    actual: key.len(),
                });
            }
            if key.iter().any(|v| !v.is_finite()) {
                return Err(KnowledgeEditError::NonFinite {
                    what: "preserved key",
                });
            }
        }

        // M = (1/N) sum_i k_i k_i^T.
        let mut moment = vec![0.0; cols * cols];
        if !preserved_keys.is_empty() {
            for key in preserved_keys {
                for i in 0..cols {
                    let ki = key[i];
                    for j in 0..cols {
                        moment[i * cols + j] += ki * key[j];
                    }
                }
            }
            #[allow(clippy::cast_precision_loss)] // A key set of 2^53 vectors is not a thing.
            let inverse_count = 1.0 / preserved_keys.len() as f64;
            for entry in &mut moment {
                *entry *= inverse_count;
            }
        }
        symmetrize(&mut moment, cols);

        let trace: f64 = (0..cols).map(|i| moment[i * cols + i]).sum();
        #[allow(clippy::cast_precision_loss)] // `cols` is a memory width.
        let mean_diagonal = trace / cols as f64;
        let scale = if mean_diagonal.is_finite() && mean_diagonal > 0.0 {
            mean_diagonal
        } else {
            1.0
        };
        let ridge = ridge_factor * scale;

        let mut covariance = moment;
        for i in 0..cols {
            covariance[i * cols + i] += ridge;
        }

        Self::assemble(
            label.into(),
            weights,
            rows,
            cols,
            covariance,
            ridge,
            preserved_keys.to_vec(),
        )
    }

    /// Build a site from a **caller-supplied** second-moment matrix — the realistic case, where
    /// `C` was accumulated over a corpus far too large to keep around.
    ///
    /// `known_ridge` is the absolute ridge already sitting on `covariance`'s diagonal. It is
    /// used only to report [`EditResult::predicted_rms_drift`](super::EditResult::predicted_rms_drift),
    /// which is exact only if the ridge is known; pass `0.0` for a pure second moment, and the
    /// prediction degrades to the (still valid) upper bound `||Delta||_C`.
    ///
    /// No preserved keys are retained, so [`Self::drift_report`] returns `None` and
    /// `EditResult`'s measured-drift fields are `None`. The analytic bound is unaffected.
    ///
    /// # Errors
    ///
    /// As [`Self::from_preserved_keys`], plus [`KnowledgeEditError::Linalg`] if `covariance` is
    /// not positive definite even after the jitter ladder — i.e. if it was not a second-moment
    /// matrix at all.
    pub fn with_covariance(
        label: impl Into<String>,
        weights: Vec<f64>,
        rows: usize,
        cols: usize,
        covariance: Vec<f64>,
        known_ridge: f64,
    ) -> Result<Self, KnowledgeEditError> {
        check_shape(&weights, rows, cols)?;
        if covariance.len() != cols * cols {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "covariance",
                expected: cols * cols,
                actual: covariance.len(),
            });
        }
        if covariance.iter().any(|v| !v.is_finite()) {
            return Err(KnowledgeEditError::NonFinite { what: "covariance" });
        }
        if !known_ridge.is_finite() || known_ridge < 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "known ridge must be finite and non-negative, got {known_ridge}"
            )));
        }
        let mut covariance = covariance;
        symmetrize(&mut covariance, cols);
        Self::assemble(
            label.into(),
            weights,
            rows,
            cols,
            covariance,
            known_ridge,
            Vec::new(),
        )
    }

    /// An identity-covariance site: `C = I`. The edit then reduces to the minimum-Frobenius-norm
    /// update `Delta = r k*^T / ||k*||^2`, which is the right thing when nothing is known about
    /// the key distribution — and a useful baseline against which the `C`-weighted edit's
    /// advantage can be measured.
    ///
    /// # Errors
    ///
    /// As [`Self::with_covariance`].
    pub fn with_identity_covariance(
        label: impl Into<String>,
        weights: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, KnowledgeEditError> {
        let covariance = scaled_identity(cols, 1.0);
        Self::with_covariance(label, weights, rows, cols, covariance, 0.0)
    }

    fn assemble(
        label: String,
        weights: Vec<f64>,
        rows: usize,
        cols: usize,
        covariance: Vec<f64>,
        ridge: f64,
        preserved_keys: Vec<Vec<f64>>,
    ) -> Result<Self, KnowledgeEditError> {
        let (cholesky, jitter) = cholesky_with_jitter(&covariance, cols)?;
        Ok(Self {
            label,
            rows,
            cols,
            base_weights: weights.clone(),
            weights,
            covariance,
            cholesky,
            ridge,
            jitter,
            preserved_keys,
        })
    }

    /// The site's label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Value width `m`.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Key width `d`.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The current weights `W'`, row-major, `rows * cols` long.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The pristine weights `W_0` the site was constructed with.
    #[must_use]
    pub fn base_weights(&self) -> &[f64] {
        &self.base_weights
    }

    /// The second-moment matrix `C`, row-major, `cols * cols` long.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// The Cholesky factor `L` with `L L^T = C`.
    #[must_use]
    pub fn cholesky(&self) -> &[f64] {
        &self.cholesky
    }

    /// The absolute ridge sitting on `C`'s diagonal.
    #[must_use]
    pub fn ridge(&self) -> f64 {
        self.ridge
    }

    /// The extra ridge the Cholesky factorization needed. `0.0` for a well-formed second-moment
    /// matrix, which is what a correctly ridged `C` always is.
    #[must_use]
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// The preserved keys, if the site retained them.
    #[must_use]
    pub fn preserved_keys(&self) -> &[Vec<f64>] {
        &self.preserved_keys
    }

    /// Read the current memory: `W' k`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `key` is not `cols` long.
    pub fn read(&self, key: &[f64]) -> Result<Vec<f64>, KnowledgeEditError> {
        Ok(mat_vec(&self.weights, key, self.rows, self.cols)?)
    }

    /// Read the pristine memory: `W_0 k`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `key` is not `cols` long.
    pub fn read_base(&self, key: &[f64]) -> Result<Vec<f64>, KnowledgeEditError> {
        Ok(mat_vec(&self.base_weights, key, self.rows, self.cols)?)
    }

    /// Read arbitrary weights of this site's shape: `W k`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] on a shape mismatch.
    pub fn read_with(&self, weights: &[f64], key: &[f64]) -> Result<Vec<f64>, KnowledgeEditError> {
        Ok(mat_vec(weights, key, self.rows, self.cols)?)
    }

    /// Apply `C^-1` to a vector, by triangular substitution against the cached Cholesky factor.
    /// `C^-1` itself is never formed.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `key` is not `cols` long, or the solve overflows.
    pub fn apply_inverse_covariance(&self, key: &[f64]) -> Result<Vec<f64>, KnowledgeEditError> {
        Ok(spd_solve(&self.cholesky, key, self.cols)?)
    }

    /// The Mahalanobis energy `k^T C^-1 k` of a key — the denominator of the rank-1 edit.
    ///
    /// Computed as `||L^-1 k||^2`, so it is a sum of squares: non-negative by construction and
    /// strictly positive for every `k != 0`. A key with *large* energy is one the preserved
    /// distribution rarely visits, and it is the cheapest key to edit, because the collateral
    /// cost of an edit is `||v* - W k*|| / sqrt(k*^T C^-1 k*)`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `key` is not `cols` long, or the solve overflows.
    pub fn mahalanobis_energy(&self, key: &[f64]) -> Result<f64, KnowledgeEditError> {
        Ok(spd_quadratic_form(&self.cholesky, key, self.cols)?)
    }

    /// The cumulative delta `W' - W_0`, densely.
    #[must_use]
    pub fn cumulative_delta(&self) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.base_weights)
            .map(|(w, b)| w - b)
            .collect()
    }

    /// The `C`-weighted norm `||W' - W_0||_C = sqrt(tr(D C D^T))` of the cumulative delta.
    ///
    /// This is the exact upper bound on the root-mean-square drift the edits so far have
    /// inflicted on the preserved keys — see the identity documented on
    /// [`weighted_frobenius_norm_sq_from_cholesky`].
    /// It is recomputed from the committed weights, so it is a *measurement of the memory* and
    /// not a number a solve reported about itself.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if the computation overflows.
    pub fn cumulative_weighted_norm(&self) -> Result<f64, KnowledgeEditError> {
        let delta = self.cumulative_delta();
        Ok(self.weighted_norm_of(&delta)?)
    }

    /// The `C`-weighted norm of an arbitrary delta of this site's shape.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditLinalgError::DimensionMismatch`] if `delta` is not `rows * cols` long.
    pub fn weighted_norm_of(&self, delta: &[f64]) -> Result<f64, KnowledgeEditLinalgError> {
        let squared =
            weighted_frobenius_norm_sq_from_cholesky(delta, &self.cholesky, self.rows, self.cols)?;
        Ok(squared.sqrt())
    }

    /// The Frobenius norm of the cumulative delta.
    #[must_use]
    pub fn cumulative_frobenius_norm(&self) -> f64 {
        frobenius_norm_sq(&self.cumulative_delta()).sqrt()
    }

    /// The **measured** collateral drift on the retained preserved keys, as
    /// `(root_mean_square, maximum)` of `||(W' - W_0) k_i||`.
    ///
    /// `None` when the site was built from a caller-supplied covariance and therefore has no
    /// key set to measure against — in which case the analytic bound from
    /// [`Self::cumulative_weighted_norm`] is all there is, and the module says so rather than
    /// inventing a number.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if a readout overflows.
    pub fn drift_report(&self) -> Result<Option<(f64, f64)>, KnowledgeEditError> {
        if self.preserved_keys.is_empty() {
            return Ok(None);
        }
        let delta = self.cumulative_delta();
        let mut sum_squares = 0.0;
        let mut maximum: f64 = 0.0;
        for key in &self.preserved_keys {
            let shift = mat_vec(&delta, key, self.rows, self.cols)?;
            let norm = linalg::l2_norm(&shift);
            sum_squares += norm * norm;
            maximum = maximum.max(norm);
        }
        #[allow(clippy::cast_precision_loss)] // A key set of 2^53 vectors is not a thing.
        let count = self.preserved_keys.len() as f64;
        Ok(Some(((sum_squares / count).sqrt(), maximum)))
    }

    /// Overwrite the current weights. The base weights `W_0` are untouched, so the cumulative
    /// delta and every drift measurement remain meaningful.
    pub(crate) fn set_weights(&mut self, weights: Vec<f64>) {
        debug_assert_eq!(weights.len(), self.rows * self.cols);
        self.weights = weights;
    }

    /// Discard every edit: `W' <- W_0`.
    pub fn reset(&mut self) {
        self.weights.clone_from(&self.base_weights);
    }
}

fn check_shape(weights: &[f64], rows: usize, cols: usize) -> Result<(), KnowledgeEditError> {
    if rows == 0 || cols == 0 {
        return Err(KnowledgeEditError::InvalidConfig(format!(
            "memory dimensions must be at least 1 x 1, got {rows} x {cols}"
        )));
    }
    if weights.len() != rows * cols {
        return Err(KnowledgeEditError::DimensionMismatch {
            what: "memory weights",
            expected: rows * cols,
            actual: weights.len(),
        });
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(KnowledgeEditError::NonFinite {
            what: "memory weights",
        });
    }
    Ok(())
}
