//! Deterministic pseudo-random orthogonal rotation matrix `P`.
//!
//! `RaBitQ` applies the *same* random rotation to every indexed vector and to
//! every query. Two properties of that rotation make the whole scheme work:
//!
//! 1. **Isometry.** Because `P` is orthogonal (`PᵀP = P Pᵀ = I`),
//!    `⟨P a, P b⟩ = ⟨a, b⟩` for *any* vectors `a, b`. So a 1-bit code computed
//!    entirely in *rotated* space can still be used to estimate inner
//!    products in the *original* space.
//! 2. **Isotropy.** A (pseudo-)random rotation spreads the mass of any fixed
//!    input direction roughly evenly across all `D` axes, so the coordinate
//!    truncation error introduced by 1-bit quantization does not depend on
//!    the data distribution — this is what gives `RaBitQ` its distribution-free
//!    error bound of `O(1/√(D-1))`.
//!
//! No random-number crate is used. `P` is derived deterministically from a
//! `u64` seed:
//!
//! 1. Generate a `D×D` matrix of pseudo-Gaussian entries: an FNV-1a hash of
//!    `(seed, index)` produces a stream of pseudo-uniform `u64`s, which are
//!    converted to standard-normal samples via the Box–Muller transform.
//! 2. Orthonormalize the columns with the modified Gram–Schmidt process
//!    (numerically stabler than classical Gram–Schmidt) to obtain an
//!    orthogonal matrix `P` with `PᵀP = P Pᵀ = I`.
//!
//! All arithmetic during construction is performed in `f64` for numerical
//! stability; the resulting matrix is stored as `f32`.

use super::types::RaBitQError;

// ── FNV-1a based pseudo-Gaussian stream ──────────────────────────────────────

/// FNV-1a 64-bit hash (offset basis `14695981039346656037`, prime
/// `1099511628211`) of the little-endian byte concatenation of `(seed, index)`.
fn fnv1a_u64(seed: u64, index: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for &b in seed.to_le_bytes().iter().chain(index.to_le_bytes().iter()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Map a 64-bit hash to an open-interval uniform `f64` in `(0, 1)`.
///
/// The open interval avoids exactly `0.0` (which would make `ln(u1)` diverge
/// in the Box–Muller transform below) and exactly `1.0`.
fn hash_to_unit_open(h: u64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let v = (h as f64 + 1.0) / (u64::MAX as f64 + 2.0);
    v.clamp(1e-15, 1.0 - 1e-15)
}

/// The `k`-th (0-indexed) standard-normal pseudo-random sample, deterministic
/// in `(seed, k)`, produced via the Box–Muller transform:
///
/// ```text
/// z0 = sqrt(-2 ln u1) * cos(2*pi*u2)
/// z1 = sqrt(-2 ln u1) * sin(2*pi*u2)
/// ```
///
/// Each pair `(u1, u2)` of independent uniforms yields one pair `(z0, z1)` of
/// independent standard-normal samples; `k` selects which element of pair
/// `k / 2` to return.
fn box_muller_sample(seed: u64, k: u64) -> f64 {
    let pair = k / 2;
    let u1 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair));
    let u2 = hash_to_unit_open(fnv1a_u64(seed, 2 * pair + 1));
    let radius = (-2.0 * u1.ln()).sqrt();
    if k.is_multiple_of(2) {
        radius * (2.0 * std::f64::consts::PI * u2).cos()
    } else {
        radius * (2.0 * std::f64::consts::PI * u2).sin()
    }
}

// ── Rotation ──────────────────────────────────────────────────────────────────

/// A deterministic pseudo-random `D×D` orthogonal rotation matrix `P`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotation {
    dim: usize,
    seed: u64,
    /// Row-major orthogonal matrix: `rows[i][j]` holds `P_{ij}`.
    rows: Vec<Vec<f32>>,
}

impl Rotation {
    /// Deterministically build a `dim × dim` orthogonal matrix from `seed`.
    ///
    /// # Errors
    ///
    /// - [`RaBitQError::InvalidConfig`] if `dim == 0`.
    /// - [`RaBitQError::InvalidConfig`] in the astronomically unlikely event
    ///   that a Gram–Schmidt column collapses to (numerically) zero norm for
    ///   the given `(seed, dim)` pair; callers should pick a different seed.
    pub fn generate(dim: usize, seed: u64) -> Result<Self, RaBitQError> {
        if dim == 0 {
            return Err(RaBitQError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }

        // Column-major pseudo-Gaussian matrix: cols[c][r] is entry (r, c).
        let mut cols: Vec<Vec<f64>> = (0..dim)
            .map(|c| {
                (0..dim)
                    .map(|r| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = (c * dim + r) as u64;
                        box_muller_sample(seed, idx)
                    })
                    .collect()
            })
            .collect();

        // Modified Gram-Schmidt: orthonormalize columns in place, in order,
        // each column projected against every already-finalized predecessor.
        for k in 0..dim {
            for j in 0..k {
                let q_j = cols[j].clone();
                let dot: f64 = cols[k].iter().zip(q_j.iter()).map(|(a, b)| a * b).sum();
                for (a, &b) in cols[k].iter_mut().zip(q_j.iter()) {
                    *a -= dot * b;
                }
            }
            let norm = cols[k].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-9 {
                return Err(RaBitQError::InvalidConfig(format!(
                    "degenerate rotation basis for seed {seed} and dim {dim} \
                     (column {k} collapsed during Gram-Schmidt); pick a different seed"
                )));
            }
            for a in &mut cols[k] {
                *a /= norm;
            }
        }

        // Transpose to row-major P where P[i][j] = cols[j][i].
        let mut rows = vec![vec![0.0f32; dim]; dim];
        for (j, col) in cols.iter().enumerate() {
            for (i, &val) in col.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    rows[i][j] = val as f32;
                }
            }
        }

        Ok(Self { dim, seed, rows })
    }

    /// The configured dimensionality `D`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The seed this rotation was generated from.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrow the row-major matrix: `rows()[i][j]` is `P_{ij}`.
    #[must_use]
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    /// Apply the rotation: returns `P · v`.
    ///
    /// Callers are expected to have already validated `v.len() == self.dim()`
    /// (the public quantizer API does so); if the lengths disagree, only the
    /// shared prefix is used, matching the module's zip-based convention.
    #[must_use]
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        self.rows
            .iter()
            .map(|row| row.iter().zip(v.iter()).map(|(&p, &x)| p * x).sum())
            .collect()
    }
}
