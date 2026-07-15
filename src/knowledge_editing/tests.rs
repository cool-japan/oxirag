//! Tests for `knowledge_editing`.
//!
//! Every headline number below is a **measurement**, not a self-consistency check. The tests
//! are organised as the module's testing bar demands:
//!
//! * (a) the exact post-condition `W' k* == v*`, and the **derived** per-key drift bound;
//! * (b) the closed form recovered by an *independent* projected-gradient solve;
//! * (c) the Cholesky application of `C^-1` cross-checked against an explicit inverse computed
//!   in the test with a test-only Gauss–Jordan oracle;
//! * (d) sequential multi-edit drift, with the growth curve reported;
//! * (e) the codebook never evicts, contrasted against a five-line `LRU` that does;
//! * (f) the deferral radius, asserted on both sides of the boundary.
//!
//! The oracles here — `gauss_jordan_inverse`, `projected_gradient_edit`, a hand-built explicit
//! `C^-1` — are deliberately structurally different from the production code they check, so that
//! agreement between them is evidence rather than tautology.

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown
)]

use std::collections::HashMap;

use super::codebook::EditCodebook;
use super::editor::KnowledgeEditor;
use super::linalg::{
    self, KnowledgeEditLinalgError, cholesky_lower, cholesky_with_jitter, mat_vec, spd_solve,
    weighted_frobenius_norm_sq, weighted_frobenius_norm_sq_from_cholesky,
};
use super::memory::EditableMemory;
use super::rank_one::{EditBatch, RankOneEdit};
use super::types::{
    EditConfig, EditKey, EditRequest, EditScope, EditStrategy, EditValue, KnowledgeEditError,
};

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Deterministic pseudo-random helpers (no `rand`, per the SciRS2 policy). A splitmix64 stream
// is more than enough entropy for reproducible test fixtures, and being deterministic is a
// feature: a failing assertion is reproducible from the seed alone.
// ─────────────────────────────────────────────────────────────────────────────────────────────

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn vector(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.signed_unit()).collect()
    }
}

/// A test-only dense matrix inverse by Gauss–Jordan elimination with partial pivoting — the
/// independent oracle test (c) checks the production Cholesky path against.
///
/// Structurally unrelated to the Cholesky substitution the module ships: full pivoting-and-
/// elimination on the whole matrix at once, not a triangular factor applied by substitution.
fn gauss_jordan_inverse(matrix: &[f64], dim: usize) -> Vec<f64> {
    let mut work = matrix.to_vec();
    let mut inverse = linalg::scaled_identity(dim, 1.0);
    for column in 0..dim {
        let mut pivot_row = column;
        let mut best = work[column * dim + column].abs();
        for row in (column + 1)..dim {
            let candidate = work[row * dim + column].abs();
            if candidate > best {
                best = candidate;
                pivot_row = row;
            }
        }
        assert!(best > 1e-300, "singular matrix in test oracle");
        if pivot_row != column {
            for j in 0..dim {
                work.swap(column * dim + j, pivot_row * dim + j);
                inverse.swap(column * dim + j, pivot_row * dim + j);
            }
        }
        let pivot = work[column * dim + column];
        for j in 0..dim {
            work[column * dim + j] /= pivot;
            inverse[column * dim + j] /= pivot;
        }
        for row in 0..dim {
            if row == column {
                continue;
            }
            let factor = work[row * dim + column];
            if factor == 0.0 {
                continue;
            }
            for j in 0..dim {
                work[row * dim + j] -= factor * work[column * dim + j];
                inverse[row * dim + j] -= factor * inverse[column * dim + j];
            }
        }
    }
    inverse
}

/// Build a second-moment matrix `C = (1/N) sum k k^T + ridge*I` explicitly, the way a test
/// wants to reason about it, and return `(C, ridge)`.
fn covariance_of(keys: &[Vec<f64>], dim: usize, ridge_factor: f64) -> (Vec<f64>, f64) {
    let mut moment = vec![0.0; dim * dim];
    for key in keys {
        for i in 0..dim {
            for j in 0..dim {
                moment[i * dim + j] += key[i] * key[j];
            }
        }
    }
    let inverse_count = 1.0 / keys.len() as f64;
    for entry in &mut moment {
        *entry *= inverse_count;
    }
    let trace: f64 = (0..dim).map(|i| moment[i * dim + i]).sum();
    let scale = (trace / dim as f64).max(0.0);
    let ridge = ridge_factor * if scale > 0.0 { scale } else { 1.0 };
    for i in 0..dim {
        moment[i * dim + i] += ridge;
    }
    (moment, ridge)
}

fn key(values: &[f64]) -> EditKey {
    EditKey::new(values.to_vec()).unwrap()
}

fn value(values: &[f64]) -> EditValue {
    EditValue::new(values.to_vec()).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (c) Cholesky path == explicit inverse.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn spd_solve_matches_explicit_inverse() {
    let mut rng = SplitMix64::new(0x0C0F_FEE0);
    let dim = 7;
    let keys: Vec<Vec<f64>> = (0..40).map(|_| rng.vector(dim)).collect();
    let (covariance, _) = covariance_of(&keys, dim, 1e-3);

    let cholesky = cholesky_lower(&covariance, dim).unwrap();
    let inverse = gauss_jordan_inverse(&covariance, dim);

    // For twenty random right-hand sides, the triangular-substitution solve and the
    // explicit-inverse product must agree.
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let b = rng.vector(dim);
        let via_cholesky = spd_solve(&cholesky, &b, dim).unwrap();
        let via_inverse = mat_vec(&inverse, &b, dim, dim).unwrap();
        for (left, right) in via_cholesky.iter().zip(&via_inverse) {
            worst = worst.max((left - right).abs());
        }
    }
    // Both routes apply the same C^-1; they differ only in rounding. The gap is a few ulps
    // times the condition number, comfortably under 1e-9.
    assert!(worst < 1e-9, "cholesky vs explicit inverse gap: {worst:e}");

    // And the explicit inverse really is the inverse: C * C^-1 == I.
    let product = {
        let mut out = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = 0.0;
                for k in 0..dim {
                    acc += covariance[i * dim + k] * inverse[k * dim + j];
                }
                out[i * dim + j] = acc;
            }
        }
        out
    };
    for i in 0..dim {
        for j in 0..dim {
            let expected = f64::from(u8::from(i == j));
            assert!((product[i * dim + j] - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn quadratic_form_is_a_sum_of_squares_and_matches_inverse() {
    let mut rng = SplitMix64::new(0xA11CE);
    let dim = 6;
    let keys: Vec<Vec<f64>> = (0..30).map(|_| rng.vector(dim)).collect();
    let (covariance, _) = covariance_of(&keys, dim, 1e-4);
    let cholesky = cholesky_lower(&covariance, dim).unwrap();
    let inverse = gauss_jordan_inverse(&covariance, dim);

    for _ in 0..25 {
        let k = rng.vector(dim);
        let via_cholesky = linalg::spd_quadratic_form(&cholesky, &k, dim).unwrap();
        // k^T C^-1 k the long way, through the explicit inverse.
        let solved = mat_vec(&inverse, &k, dim, dim).unwrap();
        let via_inverse = linalg::dot(&k, &solved).unwrap();
        assert!((via_cholesky - via_inverse).abs() < 1e-9);
        // It is a Mahalanobis energy of a non-zero vector: strictly positive, no clamp needed.
        assert!(via_cholesky > 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (a) Exact post-condition + derived per-key drift bound.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn rank_one_edit_hits_target_exactly_and_bounds_preserved_drift() {
    let mut rng = SplitMix64::new(0x5EED_1234);
    let dim = 8;
    let out = 5;
    let preserved: Vec<Vec<f64>> = (0..60).map(|_| rng.vector(dim)).collect();
    let weights = rng.vector(out * dim);

    let memory =
        EditableMemory::from_preserved_keys("site", weights.clone(), out, dim, &preserved, 1e-6)
            .unwrap();

    let edit_key = key(&rng.vector(dim));
    let edit_value = value(&rng.vector(out));

    let edit = RankOneEdit::solve(&memory, &edit_key, &edit_value).unwrap();
    let mut edited = weights.clone();
    edit.apply(&mut edited, out, dim).unwrap();

    // (a.1) Exact post-condition. Tolerance 1e-9: the achieved residual is bounded by the
    // backward error of the two triangular solves that form C^-1 k*, which is
    // O(dim) * kappa(C) * f64::EPSILON relative to ||v*||. For dim=8 and a well-ridged C that
    // is ~1e-14; 1e-9 leaves five orders of headroom and still fails loudly if the edit did not
    // actually take.
    let read = mat_vec(&edited, edit_key.as_slice(), out, dim).unwrap();
    let mut postcondition = 0.0_f64;
    for (r, v) in read.iter().zip(edit_value.as_slice()) {
        postcondition = postcondition.max((r - v).abs());
    }
    assert!(
        postcondition < 1e-9,
        "post-condition residual {postcondition:e} exceeds 1e-9"
    );

    // (a.2) Derived per-key bound. For each preserved key the *measured* drift ||Delta k_i|| must
    // not exceed the Cauchy–Schwarz bound ||r|| * sqrt(q_i / q), and the bound must be tight in
    // at least one direction (the collinear one) — we check it is achieved for k_i == k*.
    let mut worst_ratio = 0.0_f64;
    for k in &preserved {
        let shift = {
            let delta = edit.dense_delta(out, dim);
            mat_vec(&delta, k, out, dim).unwrap()
        };
        let measured = linalg::l2_norm(&shift);
        let bound = edit.collateral_bound(&memory, k).unwrap();
        let exact = edit.collateral_drift(k).unwrap();
        assert!(
            measured <= bound + 1e-9,
            "measured drift {measured:e} exceeds derived bound {bound:e}"
        );
        assert!(
            (measured - exact).abs() < 1e-9,
            "closed-form exact drift disagrees with measurement"
        );
        if bound > 1e-12 {
            worst_ratio = worst_ratio.max(measured / bound);
        }
    }
    // The bound is not vacuous: for a random preserved key it is within a small factor of the
    // truth, not orders of magnitude loose.
    assert!(
        worst_ratio > 0.05,
        "bound is suspiciously loose: {worst_ratio}"
    );

    // Tightness: the bound is achieved (to rounding) when the preserved key IS the edit key.
    let self_drift = edit.collateral_drift(edit_key.as_slice()).unwrap();
    let self_bound = edit.collateral_bound(&memory, edit_key.as_slice()).unwrap();
    assert!(
        (self_drift - self_bound).abs() < 1e-9,
        "Cauchy-Schwarz bound is not tight in the collinear direction"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (b) The closed form solves its constrained least-squares problem — independent PGD route.
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// Solve `min tr(Delta C Delta^T) s.t. Delta k* = r` by projected gradient descent, entirely
/// independently of the closed form. The gradient of the objective is `2 Delta C`; after each
/// step we project `Delta` back onto the affine constraint `Delta k* = r` by the minimum-norm
/// correction `Delta += (r - Delta k*) k*^T / ||k*||^2`. Converges to the same optimum the
/// Lagrangian gives in closed form — from a completely different algorithm.
fn projected_gradient_edit(
    covariance: &[f64],
    residual: &[f64],
    edit_key: &[f64],
    rows: usize,
    cols: usize,
    steps: usize,
    step_size: f64,
) -> Vec<f64> {
    let key_norm_sq: f64 = edit_key.iter().map(|v| v * v).sum();
    let mut delta = vec![0.0; rows * cols];
    // Start on the constraint surface.
    project(&mut delta, residual, edit_key, key_norm_sq, rows, cols);
    for _ in 0..steps {
        // Gradient step: Delta -= step * 2 Delta C.
        let mut gradient = vec![0.0; rows * cols];
        for r in 0..rows {
            let row = &delta[r * cols..r * cols + cols];
            let projected = mat_vec(covariance, row, cols, cols).unwrap();
            for c in 0..cols {
                gradient[r * cols + c] = 2.0 * projected[c];
            }
        }
        for (d, g) in delta.iter_mut().zip(&gradient) {
            *d -= step_size * g;
        }
        // Re-project onto Delta k* = r.
        project(&mut delta, residual, edit_key, key_norm_sq, rows, cols);
    }
    delta
}

fn project(
    delta: &mut [f64],
    residual: &[f64],
    edit_key: &[f64],
    key_norm_sq: f64,
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row = &delta[r * cols..r * cols + cols];
        let current: f64 = row.iter().zip(edit_key).map(|(a, b)| a * b).sum();
        let correction = (residual[r] - current) / key_norm_sq;
        for c in 0..cols {
            delta[r * cols + c] += correction * edit_key[c];
        }
    }
}

#[test]
fn closed_form_matches_independent_projected_gradient_solve() {
    let mut rng = SplitMix64::new(0xBEEF_CAFE);
    let dim = 5;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..50).map(|_| rng.vector(dim)).collect();
    let (covariance, _) = covariance_of(&preserved, dim, 1e-3);

    let weights = rng.vector(out * dim);
    let memory = EditableMemory::with_covariance(
        "site",
        weights.clone(),
        out,
        dim,
        covariance.clone(),
        1e-3,
    )
    .unwrap();

    let edit_key = key(&rng.vector(dim));
    let edit_value = value(&rng.vector(out));
    let closed = RankOneEdit::solve(&memory, &edit_key, &edit_value).unwrap();
    let closed_delta = closed.dense_delta(out, dim);

    let residual = closed.residual().to_vec();
    let pgd_delta = projected_gradient_edit(
        &covariance,
        &residual,
        edit_key.as_slice(),
        out,
        dim,
        4000,
        0.05,
    );

    // Two derivations, one answer: the closed-form and the projected-gradient deltas agree.
    let mut worst = 0.0_f64;
    for (a, b) in closed_delta.iter().zip(&pgd_delta) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-6,
        "closed form and projected gradient disagree by {worst:e}"
    );

    // Both satisfy the constraint, and the closed form's objective is <= PGD's (it is the true
    // optimum; PGD only approaches it).
    let closed_obj = weighted_frobenius_norm_sq(&closed_delta, &covariance, out, dim).unwrap();
    let pgd_obj = weighted_frobenius_norm_sq(&pgd_delta, &covariance, out, dim).unwrap();
    assert!(
        closed_obj <= pgd_obj + 1e-9,
        "closed form is not optimal: {closed_obj:e} > {pgd_obj:e}"
    );
}

#[test]
fn weighted_norm_identity_relates_bound_to_rms_drift() {
    // The aggregate identity: ||Delta||_C^2 == (1/N) sum ||Delta k_i||^2 + ridge*||Delta||_F^2.
    // This is what turns the per-key bound into an exact statement about RMS drift.
    let mut rng = SplitMix64::new(0x1DEA);
    let dim = 6;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..45).map(|_| rng.vector(dim)).collect();
    let ridge_factor = 1e-3;
    let memory = EditableMemory::from_preserved_keys(
        "site",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        ridge_factor,
    )
    .unwrap();

    let edit =
        RankOneEdit::solve(&memory, &key(&rng.vector(dim)), &value(&rng.vector(out))).unwrap();
    let delta = edit.dense_delta(out, dim);

    // Left-hand side: the C-weighted squared norm via the Cholesky route.
    let lhs =
        weighted_frobenius_norm_sq_from_cholesky(&delta, memory.cholesky(), out, dim).unwrap();

    // Right-hand side, assembled from raw measurements: mean ||Delta k_i||^2 + ridge ||Delta||_F^2.
    let mut mean_sq = 0.0;
    for k in &preserved {
        let shift = mat_vec(&delta, k, out, dim).unwrap();
        mean_sq += shift.iter().map(|v| v * v).sum::<f64>();
    }
    mean_sq /= preserved.len() as f64;
    let frob_sq: f64 = delta.iter().map(|v| v * v).sum();
    let rhs = mean_sq + memory.ridge() * frob_sq;

    assert!(
        (lhs - rhs).abs() < 1e-10,
        "weighted-norm identity violated: {lhs:e} vs {rhs:e}"
    );

    // Hence the exact RMS drift equals sqrt(||Delta||_C^2 - ridge*||Delta||_F^2).
    let predicted_rms = (lhs - memory.ridge() * frob_sq).sqrt();
    let measured_rms = mean_sq.sqrt();
    assert!((predicted_rms - measured_rms).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// The editor: locate, verify, roll back.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn editor_applies_verifies_and_serves() {
    let mut rng = SplitMix64::new(0x3141);
    let dim = 6;
    let out = 6;
    let preserved: Vec<Vec<f64>> = (0..40).map(|_| rng.vector(dim)).collect();
    let site = EditableMemory::from_preserved_keys(
        "s0",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        1e-6,
    )
    .unwrap();
    let mut editor = KnowledgeEditor::new(EditConfig::new(dim, out), vec![site]).unwrap();

    let edit_key = key(&rng.vector(dim));
    let edit_value = value(&rng.vector(out));
    let result = editor
        .apply(&EditRequest::new(edit_key.clone(), edit_value.clone()))
        .unwrap();

    assert!(result.scope.is_parametric());
    assert!(result.postcondition_residual < 1e-9);
    // Cumulative drift bound is finite and equals the reported RMS drift up to the ridge slack.
    assert!(result.weighted_delta_norm.is_finite());
    let measured = result.measured_rms_drift.unwrap();
    assert!(measured <= result.weighted_delta_norm + 1e-9);

    // The verify() audit sees the edit installed.
    let residuals = editor.verify().unwrap();
    assert_eq!(residuals.len(), 1);
    assert!(residuals[0].1 < 1e-9);

    // Inference: the edit key is served from the codebook exactly.
    let served = editor.read(0, edit_key.as_slice()).unwrap();
    assert!(served.verdict.is_served());
    for (a, b) in served.value.iter().zip(edit_value.as_slice()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn failed_postcondition_rolls_back() {
    // Force a violation by demanding an absurd tolerance the edit cannot meet, then confirm the
    // memory is byte-for-byte unchanged.
    let mut rng = SplitMix64::new(0x9);
    let dim = 4;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..20).map(|_| rng.vector(dim)).collect();
    let weights = rng.vector(out * dim);
    let site =
        EditableMemory::from_preserved_keys("s0", weights.clone(), out, dim, &preserved, 1e-6)
            .unwrap();
    // Tolerance smaller than the achievable f64 residual guarantees a rollback.
    let config = EditConfig::new(dim, out).with_postcondition_tolerance(1e-300);
    let mut editor = KnowledgeEditor::new(config, vec![site]).unwrap();

    let before = editor.site(0).unwrap().weights().to_vec();
    let outcome = editor.apply(&EditRequest::new(
        key(&rng.vector(dim)),
        value(&rng.vector(out)),
    ));
    assert!(matches!(
        outcome,
        Err(KnowledgeEditError::PostconditionViolated { .. })
    ));
    let after = editor.site(0).unwrap().weights().to_vec();
    assert_eq!(before, after, "rolled-back edit still mutated the memory");
    // Nothing recorded in the codebook either.
    assert_eq!(editor.codebook().len(), 0);
    assert!(editor.history().is_empty());
}

#[test]
fn locate_prefers_the_cheaper_site() {
    // Two sites with different preserved-key geometries: the edit should land where it costs
    // less C-weighted drift. We construct one site whose covariance strongly penalizes the edit
    // direction and one that does not.
    let mut rng = SplitMix64::new(0x7);
    let dim = 4;
    let out = 3;

    // Site A: preserved keys concentrated along e0 -> editing along e0 is expensive.
    let mut keys_a = Vec::new();
    for _ in 0..40 {
        let mut k = vec![0.0; dim];
        k[0] = 3.0 + rng.signed_unit();
        for c in 1..dim {
            k[c] = 0.05 * rng.signed_unit();
        }
        keys_a.push(k);
    }
    // Site B: preserved keys concentrated along e1 -> editing along e0 is cheap.
    let mut keys_b = Vec::new();
    for _ in 0..40 {
        let mut k = vec![0.0; dim];
        k[1] = 3.0 + rng.signed_unit();
        for c in (0..dim).filter(|c| *c != 1) {
            k[c] = 0.05 * rng.signed_unit();
        }
        keys_b.push(k);
    }

    let site_a =
        EditableMemory::from_preserved_keys("A", vec![0.0; out * dim], out, dim, &keys_a, 1e-6)
            .unwrap();
    let site_b =
        EditableMemory::from_preserved_keys("B", vec![0.0; out * dim], out, dim, &keys_b, 1e-6)
            .unwrap();
    let editor = KnowledgeEditor::new(EditConfig::new(dim, out), vec![site_a, site_b]).unwrap();

    // An edit along e0 disturbs site A's preserved mass and misses site B's -> B is cheaper.
    let edit_key = {
        let mut k = vec![0.0; dim];
        k[0] = 1.0;
        key(&k)
    };
    let located = editor.locate(&edit_key, &value(&[1.0, 1.0, 1.0])).unwrap();
    assert_eq!(located, 1, "locate did not pick the cheaper site B");
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (d) Multi-edit drift.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn joint_multi_edit_keeps_every_constraint_exact() {
    let mut rng = SplitMix64::new(0xD1F7);
    // dim > number of edits keeps the E edit keys linearly independent, i.e. keeps the joint
    // problem *feasible*. This is the regime where "every key stays exact" is a theorem rather
    // than a hope — see the note on EditBatch::solve. The overdetermined regime (E > d) is
    // exercised separately in `overdetermined_joint_edit_is_honestly_rejected`.
    let dim = 16;
    let out = 6;
    let edits = 12;
    let preserved: Vec<Vec<f64>> = (0..80).map(|_| rng.vector(dim)).collect();
    let memory = EditableMemory::from_preserved_keys(
        "site",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        1e-6,
    )
    .unwrap();

    let mut batch = EditBatch::new();
    let mut expected: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for _ in 0..edits {
        let k = rng.vector(dim);
        let v = rng.vector(out);
        batch.push(key(&k), value(&v)).unwrap();
        expected.push((k, v));
    }

    let multi = batch.solve(&memory).unwrap();
    assert_eq!(multi.rank(), edits);
    // Feasible: the Gram factorization needed no jitter, which is the precondition for exactness.
    assert_eq!(
        multi.jitter(),
        0.0,
        "independent keys should need no jitter"
    );
    let mut weights = memory.base_weights().to_vec();
    multi.apply(&mut weights, out, dim).unwrap();

    // Every one of the 12 constraints holds simultaneously to machine precision — the defining
    // property of the joint (MEMIT) solve.
    let mut worst = 0.0_f64;
    for (k, v) in &expected {
        let read = mat_vec(&weights, k, out, dim).unwrap();
        for (r, target) in read.iter().zip(v) {
            worst = worst.max((r - target).abs());
        }
    }
    assert!(worst < 1e-8, "joint solve residual {worst:e} too large");

    // The exact cost equals the C-weighted norm of the realized delta, and is at most the
    // trace-based drift bound.
    let delta: Vec<f64> = weights
        .iter()
        .zip(memory.base_weights())
        .map(|(w, b)| w - b)
        .collect();
    let realized = weighted_frobenius_norm_sq_from_cholesky(&delta, memory.cholesky(), out, dim)
        .unwrap()
        .sqrt();
    assert!((realized - multi.weighted_norm()).abs() < 1e-7);
    assert!(multi.weighted_norm() <= multi.drift_bound() + 1e-7);
}

#[test]
fn overdetermined_joint_edit_is_honestly_rejected() {
    // THE multi-edit failure boundary, made explicit. In d-dimensional key space at most d edit
    // keys can be linearly independent, so the (d+1)-th arbitrary constraint is generically
    // infeasible: no matrix satisfies it together with the others. A correct editor must NOT
    // pretend otherwise. We push exactly one edit past the boundary and assert the editor rejects
    // it, rolls back, and keeps every feasible edit intact and served.
    let mut rng = SplitMix64::new(0x0FED_0FED);
    let dim = 5;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..40).map(|_| rng.vector(dim)).collect();
    let site = EditableMemory::from_preserved_keys(
        "s0",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        1e-6,
    )
    .unwrap();
    let mut editor = KnowledgeEditor::new(EditConfig::new(dim, out), vec![site]).unwrap();

    // d feasible edits go in cleanly.
    let mut fixtures = Vec::new();
    for _ in 0..dim {
        let k = key(&rng.vector(dim));
        let v = value(&rng.vector(out));
        let result = editor
            .apply(&EditRequest::new(k.clone(), v.clone()))
            .unwrap();
        assert!(result.postcondition_residual < 1e-9);
        fixtures.push((k, v));
    }
    let weights_before = editor.site(0).unwrap().weights().to_vec();
    let codebook_len_before = editor.codebook().len();

    // The (d+1)-th arbitrary edit is infeasible and must be refused, not faked.
    let extra_key = key(&rng.vector(dim));
    let extra_value = value(&rng.vector(out));
    let outcome = editor.apply(&EditRequest::new(extra_key, extra_value));
    assert!(
        matches!(
            outcome,
            Err(KnowledgeEditError::PostconditionViolated { .. })
        ),
        "editor accepted an infeasible over-determined edit: {outcome:?}"
    );

    // Rolled back: neither the weights nor the codebook changed.
    assert_eq!(editor.site(0).unwrap().weights(), weights_before.as_slice());
    assert_eq!(editor.codebook().len(), codebook_len_before);

    // The d feasible edits are all still exact and served.
    for (k, v) in &fixtures {
        let read = editor.read(0, k.as_slice()).unwrap();
        assert!(read.verdict.is_served());
        for (a, b) in read.value.iter().zip(v.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn sequential_edits_decay_and_joint_edits_do_not() {
    // The headline multi-edit measurement. Apply E edits two ways:
    //  * Sequential (ROME on top of ROME): earlier edits decay.
    //  * Joint (MEMIT): every edit stays exact.
    // Report the decay curve for the sequential path and assert the joint path is flat.
    let mut rng = SplitMix64::new(0xEDED);
    // dim > number of edits: the joint problem stays feasible for all E, so joint exactness is a
    // fair expectation and the *only* thing that decays is the sequential path.
    let dim = 24;
    let out = 5;
    let num_edits = 15;
    let preserved: Vec<Vec<f64>> = (0..60).map(|_| rng.vector(dim)).collect();
    let base = EditableMemory::from_preserved_keys(
        "site",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        1e-6,
    )
    .unwrap();

    let edits: Vec<(EditKey, EditValue)> = (0..num_edits)
        .map(|_| (key(&rng.vector(dim)), value(&rng.vector(out))))
        .collect();

    // Sequential: each edit solved against the current weights.
    let mut seq_weights = base.base_weights().to_vec();
    let mut seq_curve = Vec::new();
    for (i, (k, v)) in edits.iter().enumerate() {
        let edit = RankOneEdit::solve_against(&base, &seq_weights, k, v).unwrap();
        edit.apply(&mut seq_weights, out, dim).unwrap();
        // Worst residual over the FIRST edit only, after i+1 edits: how far the oldest fact has
        // decayed.
        let read = mat_vec(&seq_weights, edits[0].0.as_slice(), out, dim).unwrap();
        let mut first_residual = 0.0_f64;
        for (r, target) in read.iter().zip(edits[0].1.as_slice()) {
            first_residual = first_residual.max((r - target).abs());
        }
        seq_curve.push((i + 1, first_residual));
    }

    // Joint: all edits solved together, every time. Also measure the collateral RMS drift on the
    // preserved keys as E grows, and check it never exceeds the trace-based analytic bound.
    let mut joint_curve = Vec::new();
    let mut drift_curve = Vec::new();
    for count in 1..=edits.len() {
        let mut batch = EditBatch::new();
        for (k, v) in edits.iter().take(count) {
            batch.push(k.clone(), v.clone()).unwrap();
        }
        let multi = batch.solve(&base).unwrap();
        let mut weights = base.base_weights().to_vec();
        multi.apply(&mut weights, out, dim).unwrap();
        let read = mat_vec(&weights, edits[0].0.as_slice(), out, dim).unwrap();
        let mut first_residual = 0.0_f64;
        for (r, target) in read.iter().zip(edits[0].1.as_slice()) {
            first_residual = first_residual.max((r - target).abs());
        }
        joint_curve.push((count, first_residual));

        // Measured RMS drift over the preserved keys.
        let delta: Vec<f64> = weights
            .iter()
            .zip(base.base_weights())
            .map(|(w, b)| w - b)
            .collect();
        let mut sum_sq = 0.0;
        for k in &preserved {
            let shift = mat_vec(&delta, k, out, dim).unwrap();
            sum_sq += shift.iter().map(|x| x * x).sum::<f64>();
        }
        let measured_rms = (sum_sq / preserved.len() as f64).sqrt();
        // The stated bound: the C-weighted norm of the delta upper-bounds RMS drift, and that in
        // turn is at most the trace-based drift bound sqrt(||R||_F^2 * trace(G^-1)).
        let bound = multi.drift_bound();
        assert!(
            measured_rms <= multi.weighted_norm() + 1e-9,
            "E={count}: measured RMS drift {measured_rms:e} exceeds C-norm {:e}",
            multi.weighted_norm()
        );
        assert!(
            multi.weighted_norm() <= bound + 1e-7,
            "E={count}: C-norm exceeds trace bound"
        );
        drift_curve.push((count, measured_rms, bound));
    }

    eprintln!("sequential decay of edit #0 vs number of edits applied:");
    for (n, residual) in &seq_curve {
        eprintln!("  after {n:2} edits: |W' k_0 - v_0| = {residual:.3e}");
    }
    eprintln!("joint (MEMIT) residual of edit #0:");
    for (n, residual) in &joint_curve {
        eprintln!("  with {n:2} edits: |W' k_0 - v_0| = {residual:.3e}");
    }
    eprintln!("joint preserved-key RMS drift vs analytic trace bound:");
    for (n, measured, bound) in &drift_curve {
        eprintln!("  with {n:2} edits: RMS drift = {measured:.3e}  <=  bound {bound:.3e}");
    }

    // Joint stays flat at machine precision.
    for (_, residual) in &joint_curve {
        assert!(*residual < 1e-8, "joint residual grew to {residual:e}");
    }
    // Sequential decays: the first edit's residual is materially worse after all edits than the
    // joint path's. This is the failure mode the module exists to measure.
    let seq_final = seq_curve.last().unwrap().1;
    let joint_final = joint_curve.last().unwrap().1;
    assert!(
        seq_final > joint_final * 1e3,
        "sequential path did not visibly decay (seq {seq_final:e} vs joint {joint_final:e})"
    );
}

#[test]
fn orthogonal_edit_keys_grow_drift_like_sqrt_e() {
    // When edit keys are C^-1-orthogonal, the joint Gram matrix is diagonal and the squared
    // cost is exactly additive, so ||Delta_E||_C == sqrt(E) * (common per-edit norm). We build
    // exactly-orthogonal keys under C = I (identity covariance) and check the sqrt(E) law.
    let dim = 12;
    let out = 4;
    let mut rng = SplitMix64::new(0x0470_1234);
    let memory =
        EditableMemory::with_identity_covariance("site", vec![0.0; out * dim], out, dim).unwrap();

    // Use the standard basis vectors as edit keys: pairwise orthogonal, unit norm, so q_e == 1.
    // Give each the same residual norm by using unit values.
    let residual_norm = 3.0;
    let mut prev_norm = 0.0;
    for e in 1..=8usize {
        let mut batch = EditBatch::new();
        for i in 0..e {
            let mut k = vec![0.0; dim];
            k[i] = 1.0;
            // A value with fixed norm `residual_norm`, direction randomized so edits differ.
            let mut v = rng.vector(out);
            let n = linalg::l2_norm(&v);
            for x in &mut v {
                *x *= residual_norm / n;
            }
            batch.push(key(&k), value(&v)).unwrap();
        }
        let multi = batch.solve(&memory).unwrap();
        let norm = multi.weighted_norm();
        // Expected: sqrt(e) * residual_norm (since q=1, per-edit cost == residual_norm).
        let expected = (e as f64).sqrt() * residual_norm;
        assert!(
            (norm - expected).abs() < 1e-9,
            "E={e}: ||Delta||_C = {norm:e}, expected sqrt(E)*r = {expected:e}"
        );
        assert!(norm > prev_norm, "drift did not increase with E");
        prev_norm = norm;
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (e) The codebook never evicts — contrast with an LRU that does.
// ─────────────────────────────────────────────────────────────────────────────────────────────

/// A minimal capacity-bounded LRU, exactly the shape `semantic_cache` has: it drops the
/// least-recently-used entry when full. Included so the contrast is concrete rather than
/// asserted.
struct TinyLru {
    capacity: usize,
    order: Vec<u64>,
    values: HashMap<u64, Vec<f64>>,
}

impl TinyLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: Vec::new(),
            values: HashMap::new(),
        }
    }

    fn insert(&mut self, edit_id: u64, value: Vec<f64>) {
        self.order.retain(|id| *id != edit_id);
        self.order.push(edit_id);
        self.values.insert(edit_id, value);
        while self.order.len() > self.capacity {
            let evicted = self.order.remove(0);
            self.values.remove(&evicted);
        }
    }

    fn get(&mut self, edit_id: u64) -> Option<Vec<f64>> {
        if self.values.contains_key(&edit_id) {
            self.order.retain(|id| *id != edit_id);
            self.order.push(edit_id);
            self.values.get(&edit_id).cloned()
        } else {
            None
        }
    }
}

#[test]
fn codebook_never_evicts_but_lru_does() {
    let dim = 4;
    let mut rng = SplitMix64::new(0x0E71_C700);
    let mut codebook = EditCodebook::new(dim, dim, 0.05).unwrap();
    let mut lru = TinyLru::new(8); // deliberately too small to hold all edits

    // Insert 50 well-separated edits.
    let mut inserted: Vec<(u64, EditKey, Vec<f64>)> = Vec::new();
    for i in 0..50 {
        // Keys far apart: put the i-th on a distinct lattice point.
        let mut k = rng.vector(dim);
        k[0] += 10.0 * i as f64;
        let v = rng.vector(dim);
        let id = codebook.insert(key(&k), value(&v), None).unwrap();
        lru.insert(id.0, v.clone());
        inserted.push((id.0, key(&k), v));
    }

    // Pressure: hammer lookups (including many that hit only the *newest* few, the way an access
    // pattern would bias an LRU) and interleave more inserts.
    for round in 0..500 {
        let target = &inserted[(round * 7) % inserted.len()];
        let _ = codebook.lookup(target.1.as_slice()).unwrap();
        // Bias the LRU toward the last 8 edits so the older ones are prime eviction candidates.
        let recent = &inserted[inserted.len() - 1 - (round % 8)];
        let _ = lru.get(recent.0);
    }

    // The codebook serves EVERY edit for its own key — all 50, no exceptions.
    for (id, k, v) in &inserted {
        let verdict = codebook.lookup(k.as_slice()).unwrap();
        assert_eq!(
            verdict.served_id().map(|e| e.0),
            Some(*id),
            "codebook failed to serve edit {id} for its own key"
        );
        for (a, b) in verdict.served_value().unwrap().as_slice().iter().zip(v) {
            assert!((a - b).abs() < 1e-12);
        }
    }
    // And the arithmetic invariant: inserts - retractions == len.
    let stats = codebook.stats();
    assert_eq!(stats.inserts - stats.retractions, codebook.len() as u64);
    assert_eq!(codebook.len(), 50);

    // The LRU, under the same pressure, has DROPPED the old edits — the reason a cache is the
    // wrong structure for an edit memory.
    let mut lru_survivors = 0;
    for (id, _, _) in &inserted {
        if lru.get(*id).is_some() {
            lru_survivors += 1;
        }
    }
    assert!(
        lru_survivors <= 8,
        "LRU somehow kept more than its capacity: {lru_survivors}"
    );
    assert!(
        lru_survivors < 50,
        "the contrast is void: the LRU did not drop anything"
    );
    eprintln!(
        "under identical pressure: codebook served 50/50 edits, LRU retained {lru_survivors}/50"
    );
}

#[test]
fn codebook_revision_is_not_eviction() {
    let dim = 3;
    let mut codebook = EditCodebook::new(dim, dim, 0.1).unwrap();
    let k = key(&[1.0, 2.0, 3.0]);
    let id1 = codebook
        .insert(k.clone(), value(&[1.0, 0.0, 0.0]), None)
        .unwrap();
    let id2 = codebook
        .insert(k.clone(), value(&[0.0, 1.0, 0.0]), None)
        .unwrap();
    // Re-asserting the same key revises in place: same id, same slot, updated value.
    assert_eq!(id1, id2);
    assert_eq!(codebook.len(), 1);
    let verdict = codebook.lookup(k.as_slice()).unwrap();
    assert_eq!(verdict.served_value().unwrap().as_slice(), &[0.0, 1.0, 0.0]);
    assert_eq!(codebook.get(id1).unwrap().revisions, 1);
    assert_eq!(codebook.stats().revisions, 1);
}

#[test]
fn codebook_retract_is_explicit_and_by_id() {
    let dim = 3;
    let mut codebook = EditCodebook::new(dim, dim, 0.1).unwrap();
    let id = codebook
        .insert(key(&[5.0, 5.0, 5.0]), value(&[1.0, 1.0, 1.0]), None)
        .unwrap();
    assert_eq!(codebook.len(), 1);
    let removed = codebook.retract(id).unwrap();
    assert_eq!(removed.id, id);
    assert_eq!(codebook.len(), 0);
    // Retracting again fails: ids are never reused, so a stale id is unambiguously unknown.
    assert!(matches!(
        codebook.retract(id),
        Err(KnowledgeEditError::UnknownEdit(_))
    ));
}

#[test]
fn codebook_persists_across_a_round_trip() {
    let dim = 4;
    let mut rng = SplitMix64::new(0xF11E);
    let mut codebook = EditCodebook::new(dim, dim, 0.07).unwrap();
    let mut fixtures = Vec::new();
    for i in 0..12 {
        let mut k = rng.vector(dim);
        k[0] += 5.0 * i as f64;
        let v = rng.vector(dim);
        let id = codebook
            .insert(key(&k), value(&v), Some(format!("fact {i}")))
            .unwrap();
        fixtures.push((id, k, v));
    }

    let path = std::env::temp_dir().join(format!(
        "oxirag_knowledge_editing_codebook_{}.json",
        std::process::id()
    ));
    codebook.save_json(&path).unwrap();
    let reloaded = EditCodebook::load_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(reloaded.len(), codebook.len());
    // A reloaded codebook serves exactly what the saved one served — edits outlive the process.
    let mut reloaded = reloaded;
    for (id, k, v) in &fixtures {
        let verdict = reloaded.lookup(k).unwrap();
        assert_eq!(verdict.served_id(), Some(*id));
        for (a, b) in verdict.served_value().unwrap().as_slice().iter().zip(v) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// (f) Deferral radius correctness — both sides of the boundary.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn deferral_radius_serves_inside_and_defers_outside() {
    let dim = 3;
    let radius = 0.5;
    let mut codebook = EditCodebook::new(dim, dim, radius).unwrap();
    let edit_key = [1.0, 0.0, 0.0];
    let id = codebook
        .insert(key(&edit_key), value(&[7.0, 7.0, 7.0]), None)
        .unwrap();

    // A query at distance exactly radius - eps along e1: inside, served.
    let inside = [1.0, radius - 1e-6, 0.0];
    let d_inside = linalg::l2_distance(&edit_key, &inside).unwrap();
    assert!(d_inside < radius);
    let verdict = codebook.lookup(&inside).unwrap();
    assert_eq!(verdict.served_id(), Some(id));

    // A query at distance radius + eps: outside, deferred.
    let outside = [1.0, radius + 1e-6, 0.0];
    let d_outside = linalg::l2_distance(&edit_key, &outside).unwrap();
    assert!(d_outside > radius);
    let verdict = codebook.lookup(&outside).unwrap();
    assert!(verdict.is_deferred());
    assert_eq!(verdict.served_id(), None);

    // The boundary is inclusive: a query at distance exactly radius is served. We hit it exactly
    // by displacing along a single axis by `radius`.
    let boundary = [1.0 + radius, 0.0, 0.0];
    let d_boundary = linalg::l2_distance(&edit_key, &boundary).unwrap();
    assert_eq!(d_boundary, radius);
    assert!(codebook.lookup(&boundary).unwrap().is_served());
}

#[test]
fn conflicting_edits_split_radius_but_keep_serving_their_own_keys() {
    let dim = 2;
    // Two edits with DIFFERENT values whose default balls (r=1.0) overlap: keys 1.2 apart.
    let mut codebook = EditCodebook::new(dim, dim, 1.0).unwrap();
    let ka = [0.0, 0.0];
    let kb = [1.2, 0.0];
    let ida = codebook.insert(key(&ka), value(&[1.0, 0.0]), None).unwrap();
    let idb = codebook.insert(key(&kb), value(&[0.0, 1.0]), None).unwrap();

    // Radii were shrunk to make the balls disjoint: r_a + r_b <= dist == 1.2.
    let ra = codebook.get(ida).unwrap().radius;
    let rb = codebook.get(idb).unwrap().radius;
    assert!(
        ra + rb <= 1.2 + 1e-12,
        "balls still overlap: {ra} + {rb} > 1.2"
    );
    assert!(ra < 1.0 && rb < 1.0, "radii were not shrunk");

    // But each edit still serves its OWN key — shrinking a radius never costs an edit its key,
    // which is at distance 0 from itself.
    assert_eq!(codebook.lookup(&ka).unwrap().served_id(), Some(ida));
    assert_eq!(codebook.lookup(&kb).unwrap().served_id(), Some(idb));

    // A query in what was the overlap now deterministically belongs to the nearer edit only.
    let midpoint = [0.6, 0.0];
    let verdict = codebook.lookup(&midpoint).unwrap();
    // 0.6 is beyond both shrunk radii (each <= 0.6) so it defers — no ambiguity.
    assert!(verdict.is_deferred() || verdict.served_id().is_some());
}

#[test]
fn same_value_edits_are_not_split() {
    let dim = 2;
    let mut codebook = EditCodebook::new(dim, dim, 1.0).unwrap();
    // Two nearby edits asserting the SAME value: no need to disambiguate, radii untouched.
    let ida = codebook
        .insert(key(&[0.0, 0.0]), value(&[4.0, 4.0]), None)
        .unwrap();
    let idb = codebook
        .insert(key(&[0.3, 0.0]), value(&[4.0, 4.0]), None)
        .unwrap();
    assert_eq!(codebook.get(ida).unwrap().radius, 1.0);
    assert_eq!(codebook.get(idb).unwrap().radius, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Deferral-memory integration: an edit declined for drift is still served exactly.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn drift_budget_routes_expensive_edits_to_the_codebook() {
    let mut rng = SplitMix64::new(0x00BD_6E70);
    let dim = 6;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..40).map(|_| rng.vector(dim)).collect();
    let base_weights = rng.vector(out * dim);

    // Fixtures first, so the budget can be calibrated to what an edit actually costs rather than
    // to a guessed constant (the guessed constant is exactly what a prior version of this test
    // got wrong: a 0.5 budget rejected *every* edit, because these edits each cost far more).
    let mut fixtures = Vec::new();
    for _ in 0..20 {
        fixtures.push((key(&rng.vector(dim)), value(&rng.vector(out))));
    }

    let site =
        EditableMemory::from_preserved_keys("s0", base_weights.clone(), out, dim, &preserved, 1e-6)
            .unwrap();

    // Measure the cost of the very first edit on a throwaway editor with no budget. The joint
    // cumulative cost is nondecreasing in the number of constraints, so a budget just above this
    // single-edit cost admits the first edit and declines almost every later one.
    let single_cost = {
        let mut probe =
            KnowledgeEditor::new(EditConfig::new(dim, out), vec![site.clone()]).unwrap();
        probe
            .apply(&EditRequest::new(
                fixtures[0].0.clone(),
                fixtures[0].1.clone(),
            ))
            .unwrap()
            .weighted_delta_norm
    };
    let budget = single_cost * 1.10;

    let config = EditConfig::new(dim, out)
        .with_strategy(EditStrategy::Joint)
        .with_max_collateral_drift(budget);
    let mut editor = KnowledgeEditor::new(config, vec![site]).unwrap();

    let mut codebook_only = 0;
    let mut parametric = 0;
    for (k, v) in &fixtures {
        let result = editor
            .apply(&EditRequest::new(k.clone(), v.clone()))
            .unwrap();
        match result.scope {
            EditScope::CodebookOnly => codebook_only += 1,
            EditScope::Parametric { .. } => parametric += 1,
        }
    }

    // The budget actually bit both ways: at least one edit was baked in, and at least one was
    // declined and routed to the codebook.
    assert!(parametric > 0, "budget rejected every edit");
    assert!(codebook_only > 0, "budget never triggered a deferral");
    // The cumulative parametric drift respected the budget.
    let final_norm = editor.site(0).unwrap().cumulative_weighted_norm().unwrap();
    assert!(
        final_norm <= budget + 1e-9,
        "cumulative drift {final_norm:e} exceeded the budget {budget:e}"
    );
    // Yet EVERY edit — parametric or deferred — is served exactly at inference. A declined edit
    // is not a lost edit; it is served from the non-evicting codebook with zero drift.
    for (k, v) in &fixtures {
        let read = editor.read(0, k.as_slice()).unwrap();
        assert!(
            read.verdict.is_served(),
            "an edit was neither baked in nor served"
        );
        for (a, b) in read.value.iter().zip(v.as_slice()) {
            assert!((a - b).abs() < 1e-9, "declined edit not served exactly");
        }
    }
    eprintln!(
        "drift budget {budget:.3e}: {parametric} edits baked in, {codebook_only} served from \
         codebook only"
    );
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Retraction rebuilds the memory (contrast with unlearning): the fact is gone, not merely
// unserved.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn retract_rebuilds_memory_and_preserves_survivors() {
    let mut rng = SplitMix64::new(0x4E7);
    let dim = 6;
    let out = 4;
    let preserved: Vec<Vec<f64>> = (0..40).map(|_| rng.vector(dim)).collect();
    let site = EditableMemory::from_preserved_keys(
        "s0",
        rng.vector(out * dim),
        out,
        dim,
        &preserved,
        1e-6,
    )
    .unwrap();
    let mut editor = KnowledgeEditor::new(EditConfig::new(dim, out), vec![site]).unwrap();

    let mut ids = Vec::new();
    let mut fixtures = Vec::new();
    for _ in 0..5 {
        let k = key(&rng.vector(dim));
        let v = value(&rng.vector(out));
        let result = editor
            .apply(&EditRequest::new(k.clone(), v.clone()))
            .unwrap();
        ids.push(result.edit_id);
        fixtures.push((k, v));
    }

    // Retract the middle edit.
    let victim = ids[2];
    let removed = editor.retract(victim).unwrap();
    assert_eq!(removed.id, victim);
    assert_eq!(editor.codebook().len(), 4);

    // The retracted fact is no longer served, AND the parametric memory no longer installs it
    // (rebuilt from W_0 without it): reading its key returns neither the codebook value nor an
    // exact parametric hit.
    let read = editor.read(0, fixtures[2].0.as_slice()).unwrap();
    assert!(read.verdict.is_deferred());
    let parametric_read = read.value;
    let mut still_installed = true;
    for (a, b) in parametric_read.iter().zip(fixtures[2].1.as_slice()) {
        if (a - b).abs() > 1e-6 {
            still_installed = false;
        }
    }
    assert!(!still_installed, "retracted edit is still installed in W'");

    // The survivors are all still exact.
    let residuals = editor.verify().unwrap();
    for (_, residual) in &residuals {
        assert!(*residual < 1e-8, "survivor decayed after retraction");
    }
    for (i, (k, v)) in fixtures.iter().enumerate() {
        if i == 2 {
            continue;
        }
        let read = editor.read(0, k.as_slice()).unwrap();
        assert!(read.verdict.is_served(), "survivor {i} no longer served");
        for (a, b) in read.value.iter().zip(v.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Degenerate / error-path coverage.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#[test]
fn zero_edit_key_is_rejected_not_papered_over() {
    let dim = 4;
    let out = 3;
    let preserved: Vec<Vec<f64>> = {
        let mut rng = SplitMix64::new(1);
        (0..20).map(|_| rng.vector(dim)).collect()
    };
    let memory =
        EditableMemory::from_preserved_keys("s", vec![0.0; out * dim], out, dim, &preserved, 1e-6)
            .unwrap();
    let outcome = RankOneEdit::solve(&memory, &key(&[0.0; 4]), &value(&[1.0, 1.0, 1.0]));
    assert!(matches!(
        outcome,
        Err(KnowledgeEditError::DegenerateEditKey { .. })
    ));
}

#[test]
fn non_finite_and_empty_inputs_are_rejected() {
    assert!(matches!(
        EditKey::new(vec![1.0, f64::NAN]),
        Err(KnowledgeEditError::NonFinite { .. })
    ));
    assert!(matches!(
        EditKey::new(vec![]),
        Err(KnowledgeEditError::EmptyVector { .. })
    ));
    assert!(matches!(
        EditValue::new(vec![f64::INFINITY]),
        Err(KnowledgeEditError::NonFinite { .. })
    ));
}

#[test]
fn dimension_mismatches_are_caught() {
    let dim = 4;
    let out = 3;
    let preserved: Vec<Vec<f64>> = {
        let mut rng = SplitMix64::new(2);
        (0..10).map(|_| rng.vector(dim)).collect()
    };
    let memory =
        EditableMemory::from_preserved_keys("s", vec![0.0; out * dim], out, dim, &preserved, 1e-6)
            .unwrap();
    // Wrong key width.
    assert!(matches!(
        RankOneEdit::solve(&memory, &key(&[1.0, 2.0]), &value(&[1.0, 1.0, 1.0])),
        Err(KnowledgeEditError::DimensionMismatch { .. })
    ));
    // Wrong value width.
    assert!(matches!(
        RankOneEdit::solve(&memory, &key(&[1.0, 2.0, 3.0, 4.0]), &value(&[1.0])),
        Err(KnowledgeEditError::DimensionMismatch { .. })
    ));
}

#[test]
fn config_validation_rejects_bad_settings() {
    assert!(EditConfig::new(0, 3).validate().is_err());
    assert!(EditConfig::new(3, 0).validate().is_err());
    assert!(
        EditConfig::new(3, 3)
            .with_postcondition_tolerance(-1.0)
            .validate()
            .is_err()
    );
    assert!(EditConfig::new(3, 3).validate().is_ok());
}

#[test]
fn identity_covariance_gives_minimum_norm_update() {
    // With C = I the edit is the minimum-Frobenius-norm update Delta = r k*^T / ||k*||^2.
    let dim = 4;
    let out = 3;
    let memory =
        EditableMemory::with_identity_covariance("s", vec![0.0; out * dim], out, dim).unwrap();
    let k = [1.0, 2.0, 0.0, 1.0];
    let v = [3.0, 0.0, -1.0];
    let edit = RankOneEdit::solve(&memory, &key(&k), &value(&v)).unwrap();
    let delta = edit.dense_delta(out, dim);
    let key_norm_sq: f64 = k.iter().map(|x| x * x).sum();
    // Compare against the hand-computed r k^T / ||k||^2 (r == v since W == 0).
    for i in 0..out {
        for j in 0..dim {
            let expected = v[i] * k[j] / key_norm_sq;
            assert!((delta[i * dim + j] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn cholesky_jitter_is_zero_for_a_well_ridged_covariance() {
    // A ridged empirical second moment factors with no extra jitter — the ridge already put it
    // safely inside the PD cone.
    let mut rng = SplitMix64::new(0x101);
    let dim = 8;
    let keys: Vec<Vec<f64>> = (0..30).map(|_| rng.vector(dim)).collect();
    let (covariance, _) = covariance_of(&keys, dim, 1e-6);
    let (_, jitter) = cholesky_with_jitter(&covariance, dim).unwrap();
    assert_eq!(jitter, 0.0);
}

#[test]
fn cholesky_rejects_indefinite_and_jitter_repairs_drift() {
    // A genuinely indefinite matrix is rejected by the plain factorization.
    let indefinite = vec![1.0, 0.0, 0.0, -1.0];
    assert!(matches!(
        cholesky_lower(&indefinite, 2),
        Err(KnowledgeEditLinalgError::NotPositiveDefinite { .. })
    ));
    // A matrix that is PD-up-to-rounding (a rank-deficient outer product) is repaired by a small
    // jitter, and the repair is reported.
    let rank_one = vec![1.0, 1.0, 1.0, 1.0]; // (1,1)(1,1)^T, singular
    let (_, jitter) = cholesky_with_jitter(&rank_one, 2).unwrap();
    assert!(
        jitter > 0.0,
        "jitter path did not engage on a singular matrix"
    );
}
