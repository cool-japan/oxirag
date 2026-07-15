//! The closed forms: a rank-1 edit of a single key ([`RankOneEdit`], `ROME`) and the joint
//! rank-`E` edit of many keys at once ([`EditBatch`] → [`MultiEdit`], `MEMIT`).
//!
//! # The problem both of them solve
//!
//! A linear memory `W` maps keys to values. We want it to answer `v*` for a key `k*` while
//! disturbing its answers on a distribution of *preserved* keys as little as possible. Writing
//! `Delta = W' - W`, "as little as possible" is
//!
//! ```text
//! minimize   E_k [ || Delta k ||^2 ] = tr(Delta C Delta^T),     C = E[k k^T]
//! subject to Delta k* = v* - W k*  =:  r
//! ```
//!
//! a convex quadratic under a linear constraint, so it has a closed form. Introduce a
//! Lagrange multiplier `lambda` (an `m`-vector, one per output coordinate):
//!
//! ```text
//! L(Delta, lambda) = tr(Delta C Delta^T) - 2 lambda^T (Delta k* - r)
//! dL/dDelta = 2 Delta C - 2 lambda k*^T = 0    =>    Delta = lambda k*^T C^-1
//! ```
//!
//! Substituting back into the constraint gives `lambda (k*^T C^-1 k*) = r`, hence
//!
//! ```text
//!             (v* - W k*) (C^-1 k*)^T
//! Delta  =  ---------------------------          (C is symmetric, so k*^T C^-1 = (C^-1 k*)^T)
//!                 k*^T C^-1 k*
//! ```
//!
//! which is a **rank-1** matrix: an outer product of the residual `r` with the whitened key
//! `u = C^-1 k*`, scaled by the reciprocal of the Mahalanobis energy `q = k*^T C^-1 k*`.
//!
//! # The two post-conditions, and where the second one comes from
//!
//! **Exactness.** `W' k* = W k* + Delta k* = W k* + r (u^T k*) / q = W k* + r = v*`, since
//! `u^T k* = q` by definition. This is an identity, not an approximation.
//!
//! **Bounded collateral damage.** For any preserved key `k_i`,
//!
//! ```text
//! || W' k_i - W k_i ||  =  || Delta k_i ||  =  ||r|| * |u^T k_i| / q
//! ```
//!
//! and because `C^-1` is positive definite, `<a, b> = a^T C^-1 b` is a genuine inner product, so
//! Cauchy–Schwarz applies to it:
//!
//! ```text
//! |u^T k_i| = |k_i^T C^-1 k*| <= sqrt(k_i^T C^-1 k_i) * sqrt(k*^T C^-1 k*) = sqrt(q_i * q)
//!
//!                                    ||r|| * sqrt(q_i)
//! =>   || W' k_i - W k_i ||   <=   --------------------- ,       q_i = k_i^T C^-1 k_i
//!                                       sqrt(q)
//! ```
//!
//! This is *derived*, it is **tight** (equality exactly when `k_i` is parallel to `k*`), and it
//! is what [`RankOneEdit::collateral_bound`] returns. It says something intuitive: an edit
//! damages a preserved key in proportion to how aligned that key is with the edit key, in the
//! geometry `C` induces — and it damages the whole key set in inverse proportion to how *rare*
//! the edit key is, since `q` is large exactly for keys the preserved distribution avoids.
//!
//! **In aggregate**, the bound sharpens into an identity. The objective value at the optimum is
//!
//! ```text
//! ||Delta||_C^2 := tr(Delta C Delta^T) = ||r||^2 / q
//! ```
//!
//! and for the ridged empirical second moment `C = (1/N) sum_i k_i k_i^T + ridge * I` this
//! expands to
//!
//! ```text
//! ||Delta||_C^2  =  (1/N) * sum_i || Delta k_i ||^2  +  ridge * ||Delta||_F^2
//! ```
//!
//! so the **root-mean-square drift over the preserved key set is exactly**
//! `sqrt(||r||^2/q - ridge * ||Delta||_F^2)`, and `||r|| / sqrt(q)` bounds it from above with a
//! slack of precisely `ridge * ||Delta||_F^2`. The module asserts this identity to `1e-12`
//! rather than asserting an inequality it could not have failed.
//!
//! # `MEMIT`: many edits at once
//!
//! With `E` constraints, stack the keys as the columns of a `d × E` matrix `K` and the residuals
//! as the columns of an `m × E` matrix `R`. The same Lagrangian, now with `E` multipliers, gives
//!
//! ```text
//! Delta = R G^-1 K^T C^-1 ,      G = K^T C^-1 K       (the E × E Gram matrix of the edit
//!                                                       keys in the C^-1 inner product)
//! ```
//!
//! which for `E = 1` collapses back to the rank-1 formula (`G = [q]`). `Delta` is rank `E`: it
//! is `sum_e lambda_e u_e^T` for the columns `lambda_e` of `R G^-1` and `u_e = C^-1 k_e`, and
//! [`MultiEdit`] stores exactly those `2E` vectors rather than an `m × d` dense matrix.
//!
//! The exact cost is `||Delta||_C^2 = tr(R G^-1 R^T)`, and this is where multi-edit methods
//! actually fail: `G` is the Gram matrix of the edit keys, so **correlated edit keys make `G`
//! ill-conditioned and `G^-1` enormous**, and the collateral damage explodes. When the edit keys
//! are `C^-1`-orthogonal, `G` is diagonal and the cost is simply additive —
//! `||Delta_E||_C^2 = sum_e ||Delta_e||_C^2`, i.e. drift grows like `sqrt(E)`. When they are
//! not, it grows without bound. [`MultiEdit::drift_bound`] reports the rigorous, computable
//! bound `sqrt(||R||_F^2 * trace(G^-1))`, and [`MultiEdit::jitter`] reports whether `G` was so
//! near-singular that the factorization had to be propped up at all.

use serde::{Deserialize, Serialize};

use super::linalg::{
    add_scaled_outer_product, cholesky_with_jitter, dot, inverse_trace_from_cholesky, l2_norm,
    solve_lower, solve_upper_transposed, symmetrize,
};
use super::memory::EditableMemory;
use super::types::{EditKey, EditValue, KnowledgeEditError};

/// The closed-form rank-1 solution for a single key, stored in its natural factored form.
///
/// The dense `m × d` delta is never materialized on the production path: [`Self::apply`]
/// accumulates the outer product straight into the weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankOneEdit {
    residual: Vec<f64>,
    direction: Vec<f64>,
    denominator: f64,
    weighted_norm: f64,
}

impl RankOneEdit {
    /// Solve `min tr(Delta C Delta^T)` subject to `(W + Delta) k* = v*`, against the memory's
    /// **current** weights `W'`.
    ///
    /// This is the `ROME` semantics: the edit is made to the model you have. For a sequence of
    /// edits that must *all* hold at once, use [`EditBatch`] instead — see [`EditStrategy`] for
    /// why the difference matters.
    ///
    /// [`EditStrategy`]: super::EditStrategy
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::DimensionMismatch`] if the key or value does not match the
    ///   memory's shape.
    /// * [`KnowledgeEditError::DegenerateEditKey`] if `k*^T C^-1 k* <= 0`, which for a
    ///   positive-definite `C^-1` proves the key is the zero vector — and no rank-1 update of
    ///   `W` can change `W k` when `k` is zero.
    /// * [`KnowledgeEditError::Linalg`] if a solve overflows.
    pub fn solve(
        memory: &EditableMemory,
        key: &EditKey,
        value: &EditValue,
    ) -> Result<Self, KnowledgeEditError> {
        Self::solve_against(memory, memory.weights(), key, value)
    }

    /// As [`Self::solve`], but against an explicit weight matrix of the memory's shape rather
    /// than the memory's current one.
    ///
    /// This is what lets an editor evaluate a *trial* write without mutating anything.
    ///
    /// # Errors
    ///
    /// As [`Self::solve`].
    pub fn solve_against(
        memory: &EditableMemory,
        weights: &[f64],
        key: &EditKey,
        value: &EditValue,
    ) -> Result<Self, KnowledgeEditError> {
        check_key_value(memory, key, value)?;

        // r = v* - W k*
        let current = memory.read_with(weights, key.as_slice())?;
        let residual: Vec<f64> = value
            .as_slice()
            .iter()
            .zip(&current)
            .map(|(v, w)| v - w)
            .collect();

        // u = C^-1 k*, and q = k*^T C^-1 k* = ||L^-1 k*||^2 (a sum of squares: see `linalg`).
        let direction = memory.apply_inverse_covariance(key.as_slice())?;
        let denominator = memory.mahalanobis_energy(key.as_slice())?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(KnowledgeEditError::DegenerateEditKey { denominator });
        }

        let weighted_norm = l2_norm(&residual) / denominator.sqrt();
        Ok(Self {
            residual,
            direction,
            denominator,
            weighted_norm,
        })
    }

    /// The residual `r = v* - W k*` this edit has to install.
    #[must_use]
    pub fn residual(&self) -> &[f64] {
        &self.residual
    }

    /// The whitened key `u = C^-1 k*` — the direction, in key space, along which the memory is
    /// perturbed.
    #[must_use]
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// The Mahalanobis energy `q = k*^T C^-1 k*`. Strictly positive for every non-zero key.
    #[must_use]
    pub fn denominator(&self) -> f64 {
        self.denominator
    }

    /// The exact optimal objective value's square root, `||Delta||_C = ||r|| / sqrt(q)`.
    ///
    /// This *is* the collateral cost of the edit: the root-mean-square drift it inflicts on the
    /// preserved keys is at most this, with slack exactly `ridge * ||Delta||_F^2`.
    #[must_use]
    pub fn weighted_norm(&self) -> f64 {
        self.weighted_norm
    }

    /// Write the edit into a weight matrix: `W += r u^T / q`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] on a shape mismatch, or if the write would introduce a
    /// non-finite weight.
    pub fn apply(
        &self,
        weights: &mut [f64],
        rows: usize,
        cols: usize,
    ) -> Result<(), KnowledgeEditError> {
        add_scaled_outer_product(
            weights,
            1.0 / self.denominator,
            &self.residual,
            &self.direction,
            rows,
            cols,
        )?;
        Ok(())
    }

    /// Materialize the dense `rows × cols` delta. For inspection and testing; the production
    /// path uses [`Self::apply`].
    #[must_use]
    pub fn dense_delta(&self, rows: usize, cols: usize) -> Vec<f64> {
        let mut delta = vec![0.0; rows * cols];
        let inverse = 1.0 / self.denominator;
        for (row, &r) in delta.chunks_exact_mut(cols).zip(&self.residual) {
            let scaled = inverse * r;
            for (slot, &u) in row.iter_mut().zip(&self.direction) {
                *slot = scaled * u;
            }
        }
        delta
    }

    /// The **exact** drift this edit inflicts on one preserved key: `||Delta k_i||`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `preserved_key` is not the memory's key width.
    pub fn collateral_drift(&self, preserved_key: &[f64]) -> Result<f64, KnowledgeEditError> {
        let alignment = dot(&self.direction, preserved_key)?;
        let scale = alignment / self.denominator;
        Ok(l2_norm(&self.residual) * scale.abs())
    }

    /// The **derived Cauchy–Schwarz bound** on that drift: `||r|| * sqrt(q_i / q)`.
    ///
    /// Always at least [`Self::collateral_drift`], with equality exactly when the preserved key
    /// is parallel to the edit key. Derived in the [module documentation](self); it is not a
    /// guess.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if `preserved_key` is not the memory's key width, or the
    /// solve overflows.
    pub fn collateral_bound(
        &self,
        memory: &EditableMemory,
        preserved_key: &[f64],
    ) -> Result<f64, KnowledgeEditError> {
        let energy = memory.mahalanobis_energy(preserved_key)?;
        Ok(l2_norm(&self.residual) * (energy / self.denominator).sqrt())
    }
}

/// An accumulating set of `(key, value)` constraints, solved **jointly**.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EditBatch {
    keys: Vec<EditKey>,
    values: Vec<EditValue>,
}

impl EditBatch {
    /// An empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::DimensionMismatch`] if the key or value width disagrees with the
    /// constraints already in the batch.
    pub fn push(&mut self, key: EditKey, value: EditValue) -> Result<(), KnowledgeEditError> {
        if let Some(first) = self.keys.first()
            && first.dim() != key.dim()
        {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "batch edit key",
                expected: first.dim(),
                actual: key.dim(),
            });
        }
        if let Some(first) = self.values.first()
            && first.dim() != value.dim()
        {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "batch edit value",
                expected: first.dim(),
                actual: value.dim(),
            });
        }
        self.keys.push(key);
        self.values.push(value);
        Ok(())
    }

    /// Drop the constraint at `index`, returning it.
    ///
    /// Used by [`KnowledgeEditor::retract`](super::KnowledgeEditor::retract): removing a
    /// constraint and re-solving is how an edit is *undone*, as opposed to being silently
    /// evicted.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::InvalidConfig`] if `index` is out of range.
    pub fn remove(&mut self, index: usize) -> Result<(EditKey, EditValue), KnowledgeEditError> {
        if index >= self.keys.len() {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "batch index {index} is out of range ({} constraints)",
                self.keys.len()
            )));
        }
        Ok((self.keys.remove(index), self.values.remove(index)))
    }

    /// The number of constraints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether there are no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The constrained keys, in insertion order.
    #[must_use]
    pub fn keys(&self) -> &[EditKey] {
        &self.keys
    }

    /// The constrained values, in insertion order.
    #[must_use]
    pub fn values(&self) -> &[EditValue] {
        &self.values
    }

    /// Solve every constraint jointly against the memory's **base** weights `W_0`.
    ///
    /// This is the `MEMIT` semantics, and it is what makes a *sequence* of edits behave: the
    /// cumulative delta is re-derived from scratch each time, so every constraint — not just the
    /// newest — is satisfied exactly.
    ///
    /// # Feasibility, and what happens past it
    ///
    /// "Satisfied exactly" holds **only when the edit keys are linearly independent**, which
    /// requires `E <= d` (you cannot have more than `d` independent vectors in `d`-dimensional key
    /// space). In that regime the Gram matrix `G = K^T C^-1 K` is positive definite, its Cholesky
    /// needs no jitter, and every `Delta k_e = r_e` holds to machine precision.
    ///
    /// Past it — `E > d`, or `E <= d` but with near-dependent keys — the constraints are generically
    /// *infeasible*: if `k_E = sum_e a_e k_e` then `Delta k_E = sum_e a_e r_e` is **forced**, and a
    /// caller-chosen `r_E` cannot be honoured. `G` is then singular, [`cholesky_with_jitter`] props
    /// it up with a ridge, and this method returns the resulting **ridge-regularized least-squares
    /// solution** — a real, well-defined matrix, but one that no longer hits every target exactly.
    /// The returned [`MultiEdit::jitter`] is non-zero in exactly this case and is the signal that
    /// feasibility was exceeded; [`MultiEdit::weighted_norm`] remains the honest `C`-norm of
    /// whatever delta was actually produced. This is why [`super::KnowledgeEditor`] **verifies the
    /// post-condition after every write and rolls back** rather than trusting the solve: an
    /// infeasible batch is caught there and reported as
    /// [`KnowledgeEditError::PostconditionViolated`], never silently accepted.
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::DimensionMismatch`] if a key or value does not match the memory.
    /// * [`KnowledgeEditError::DegenerateEditKey`] if an edit key is the zero vector.
    /// * [`KnowledgeEditError::Linalg`] if the Gram matrix will not factor even after the full
    ///   jitter ladder — grossly degenerate keys.
    pub fn solve(&self, memory: &EditableMemory) -> Result<MultiEdit, KnowledgeEditError> {
        self.solve_against(memory, memory.base_weights())
    }

    /// As [`Self::solve`], but against an explicit weight matrix of the memory's shape.
    ///
    /// # Errors
    ///
    /// As [`Self::solve`].
    pub fn solve_against(
        &self,
        memory: &EditableMemory,
        weights: &[f64],
    ) -> Result<MultiEdit, KnowledgeEditError> {
        let count = self.keys.len();
        if count == 0 {
            return Ok(MultiEdit::empty());
        }
        for (key, value) in self.keys.iter().zip(&self.values) {
            check_key_value(memory, key, value)?;
        }

        // R (m x E), column e = v_e - W k_e; and U (d x E), column e = C^-1 k_e.
        let mut residuals: Vec<Vec<f64>> = Vec::with_capacity(count);
        let mut directions: Vec<Vec<f64>> = Vec::with_capacity(count);
        for (key, value) in self.keys.iter().zip(&self.values) {
            let current = memory.read_with(weights, key.as_slice())?;
            residuals.push(
                value
                    .as_slice()
                    .iter()
                    .zip(&current)
                    .map(|(v, w)| v - w)
                    .collect(),
            );
            directions.push(memory.apply_inverse_covariance(key.as_slice())?);
        }

        // G[a][b] = k_a^T C^-1 k_b = k_a . u_b. Symmetric in exact arithmetic; symmetrized here
        // so the factorization does not depend on which triangle it reads.
        let mut gram = vec![0.0; count * count];
        for (a, key) in self.keys.iter().enumerate() {
            for (b, direction) in directions.iter().enumerate() {
                gram[a * count + b] = dot(key.as_slice(), direction)?;
            }
        }
        symmetrize(&mut gram, count);

        // A zero edit key would make a whole row and column of G vanish, which is singular and
        // would be "repaired" into nonsense by the jitter ladder. Catch it for what it is.
        for a in 0..count {
            let energy = gram[a * count + a];
            if !energy.is_finite() || energy <= 0.0 {
                return Err(KnowledgeEditError::DegenerateEditKey {
                    denominator: energy,
                });
            }
        }

        let (gram_factor, jitter) = cholesky_with_jitter(&gram, count)?;

        // Lambda = R G^-1, computed one output coordinate at a time: for output row i, solve
        // `G x = R[i, :]`. The forward-substitution intermediate `y = L_G^-1 R[i, :]` doubles as
        // the exact objective value, since
        //     tr(R G^-1 R^T) = sum_i R[i,:]^T G^-1 R[i,:] = sum_i ||L_G^-1 R[i,:]||^2,
        // a sum of squares — so the reported cost cannot come out negative.
        let rows = memory.rows();
        let mut coefficients: Vec<Vec<f64>> = vec![vec![0.0; rows]; count];
        let mut weighted_norm_sq = 0.0;
        for i in 0..rows {
            let rhs: Vec<f64> = residuals.iter().map(|r| r[i]).collect();
            let forward = solve_lower(&gram_factor, &rhs, count)?;
            weighted_norm_sq += forward.iter().map(|v| v * v).sum::<f64>();
            let solved = solve_upper_transposed(&gram_factor, &forward, count)?;
            for (e, coefficient) in coefficients.iter_mut().enumerate() {
                coefficient[i] = solved[e];
            }
        }

        let residual_frobenius_sq: f64 = residuals
            .iter()
            .map(|r| r.iter().map(|v| v * v).sum::<f64>())
            .sum();
        let gram_inverse_trace = inverse_trace_from_cholesky(&gram_factor, count)?;

        Ok(MultiEdit {
            coefficients,
            directions,
            weighted_norm: weighted_norm_sq.sqrt(),
            gram_inverse_trace,
            residual_frobenius_sq,
            jitter,
        })
    }
}

/// The closed-form rank-`E` solution: `Delta = sum_e lambda_e u_e^T`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiEdit {
    coefficients: Vec<Vec<f64>>,
    directions: Vec<Vec<f64>>,
    weighted_norm: f64,
    gram_inverse_trace: f64,
    residual_frobenius_sq: f64,
    jitter: f64,
}

impl MultiEdit {
    /// The delta that changes nothing — the solution of an empty constraint set.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            coefficients: Vec::new(),
            directions: Vec::new(),
            weighted_norm: 0.0,
            gram_inverse_trace: 0.0,
            residual_frobenius_sq: 0.0,
            jitter: 0.0,
        }
    }

    /// The rank `E` of the delta — the number of constraints it satisfies.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.coefficients.len()
    }

    /// Whether the delta changes nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The exact cost `||Delta||_C = sqrt(tr(R G^-1 R^T))`.
    #[must_use]
    pub fn weighted_norm(&self) -> f64 {
        self.weighted_norm
    }

    /// `trace(G^-1)` — how badly conditioned the edit keys' Gram matrix is. Large means the edit
    /// keys are close to linearly dependent, which is the regime where multi-edit collateral
    /// damage explodes.
    #[must_use]
    pub fn gram_inverse_trace(&self) -> f64 {
        self.gram_inverse_trace
    }

    /// A rigorous, computable upper bound on [`Self::weighted_norm`]:
    /// `sqrt(||R||_F^2 * trace(G^-1))`.
    ///
    /// Every eigenvalue of the positive-definite `G^-1` is at most its trace (they are all
    /// positive and they sum to it), so `tr(R G^-1 R^T) <= trace(G^-1) * ||R||_F^2`. Unlike the
    /// exact cost, this bound can be evaluated from the *keys alone* — it does not need the
    /// residuals to have been solved — which makes it usable as an admission test before an edit
    /// is committed.
    #[must_use]
    pub fn drift_bound(&self) -> f64 {
        (self.residual_frobenius_sq * self.gram_inverse_trace).sqrt()
    }

    /// The ridge the Gram-matrix Cholesky needed. Anything above zero means the edit keys were
    /// numerically linearly dependent and the joint solve is on thin ice — the editor still
    /// verifies the post-conditions afterwards, so a jittered solve that nevertheless satisfies
    /// every constraint is honest, and one that does not is rejected.
    #[must_use]
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Write the edit into a weight matrix: `W += sum_e lambda_e u_e^T`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] on a shape mismatch, or if the write would introduce a
    /// non-finite weight.
    pub fn apply(
        &self,
        weights: &mut [f64],
        rows: usize,
        cols: usize,
    ) -> Result<(), KnowledgeEditError> {
        for (coefficient, direction) in self.coefficients.iter().zip(&self.directions) {
            add_scaled_outer_product(weights, 1.0, coefficient, direction, rows, cols)?;
        }
        Ok(())
    }

    /// Materialize the dense `rows × cols` delta. For inspection and testing; the production
    /// path uses [`Self::apply`].
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] on a shape mismatch.
    pub fn dense_delta(&self, rows: usize, cols: usize) -> Result<Vec<f64>, KnowledgeEditError> {
        let mut delta = vec![0.0; rows * cols];
        self.apply(&mut delta, rows, cols)?;
        Ok(delta)
    }
}

fn check_key_value(
    memory: &EditableMemory,
    key: &EditKey,
    value: &EditValue,
) -> Result<(), KnowledgeEditError> {
    if key.dim() != memory.cols() {
        return Err(KnowledgeEditError::DimensionMismatch {
            what: "edit key",
            expected: memory.cols(),
            actual: key.dim(),
        });
    }
    if value.dim() != memory.rows() {
        return Err(KnowledgeEditError::DimensionMismatch {
            what: "edit value",
            expected: memory.rows(),
            actual: value.dim(),
        });
    }
    Ok(())
}
