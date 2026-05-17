//! Pure-Rust scalar (fallback) implementations of similarity kernels.
//!
//! These are used when no SIMD intrinsics are available, and also serve as the
//! reference correctness baseline against which the SIMD paths are validated in
//! tests.

// ============================================================================
// Scalar / Fallback Implementations
// ============================================================================

/// Scalar dot product computation.
#[inline]
pub(super) fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar L2 norm computation.
#[inline]
pub(super) fn scalar_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Scalar Euclidean distance computation.
#[inline]
pub(super) fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Scalar cosine similarity computation.
#[inline]
pub(super) fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = scalar_dot_product(a, b);
    let norm_a = scalar_l2_norm(a);
    let norm_b = scalar_l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
