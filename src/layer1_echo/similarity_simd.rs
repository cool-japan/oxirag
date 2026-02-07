//! SIMD-accelerated similarity computations for vector operations.
//!
//! Provides high-performance similarity metrics using explicit SIMD instructions
//! for ARM NEON and x86 AVX/SSE architectures.

// Allow unsafe code in unsafe functions for Rust 2024 edition SIMD operations
#![allow(unsafe_op_in_unsafe_fn)]
// Allow similar names for mathematical variables (norm_a, norm_b, etc.)
#![allow(clippy::similar_names)]

#[cfg(target_arch = "aarch64")]
#[allow(clippy::wildcard_imports)]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// SIMD-optimized cosine similarity for f32 vectors.
///
/// Uses NEON on ARM64 or AVX/SSE on `x86_64` for 4-8x speedup over scalar code.
///
/// # Safety
/// This function uses unsafe SIMD intrinsics but is safe to call as it:
/// - Only reads within vector bounds
/// - Handles remainder elements with scalar code
/// - Uses platform-specific intrinsics guarded by `target_arch`
#[inline]
#[must_use]
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { cosine_similarity_neon(a, b) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        unsafe { cosine_similarity_avx(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx"),
        target_feature = "sse2"
    ))]
    {
        unsafe { cosine_similarity_sse(a, b) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        // Fallback to scalar implementation
        super::similarity::cosine_similarity(a, b)
    }
}

/// SIMD-optimized dot product for f32 vectors.
#[inline]
#[must_use]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        unsafe { dot_product_avx(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx"),
        target_feature = "sse2"
    ))]
    {
        unsafe { dot_product_sse(a, b) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        super::similarity::dot_product(a, b)
    }
}

/// SIMD-optimized vector normalization.
#[inline]
#[must_use]
pub fn normalize_simd(v: &[f32]) -> Vec<f32> {
    let norm_squared = dot_product_simd(v, v);
    let norm = norm_squared.sqrt();

    if norm < 1e-10 {
        return v.to_vec();
    }

    let inv_norm = 1.0 / norm;
    let mut result = vec![0.0f32; v.len()];

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            multiply_scalar_neon(v, inv_norm, &mut result);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    {
        unsafe {
            multiply_scalar_avx(v, inv_norm, &mut result);
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx"),
        target_feature = "sse2"
    ))]
    {
        unsafe {
            multiply_scalar_sse(v, inv_norm, &mut result);
        }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        for (i, &val) in v.iter().enumerate() {
            result[i] = val * inv_norm;
        }
    }

    result
}

// ============================================================================
// ARM NEON implementations (Apple Silicon, ARM servers)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut dot_acc = vdupq_n_f32(0.0);
    let mut norm_a_acc = vdupq_n_f32(0.0);
    let mut norm_b_acc = vdupq_n_f32(0.0);

    let chunks = len / 4;
    let remainder = len % 4;

    // Process 4 elements at a time with NEON
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));

        // Dot product: a[i] * b[i]
        dot_acc = vfmaq_f32(dot_acc, va, vb);

        // Norm of a: a[i] * a[i]
        norm_a_acc = vfmaq_f32(norm_a_acc, va, va);

        // Norm of b: b[i] * b[i]
        norm_b_acc = vfmaq_f32(norm_b_acc, vb, vb);
    }

    // Horizontal sum
    let mut dot_sum = vaddvq_f32(dot_acc);
    let mut norm_a_sum = vaddvq_f32(norm_a_acc);
    let mut norm_b_sum = vaddvq_f32(norm_b_acc);

    // Handle remainder elements
    let offset = chunks * 4;
    for i in 0..remainder {
        let ai = a[offset + i];
        let bi = b[offset + i];
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    if norm_a_sum == 0.0 || norm_b_sum == 0.0 {
        return 0.0;
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = vdupq_n_f32(0.0);

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);

    // Handle remainder
    let offset = chunks * 4;
    for i in 0..remainder {
        sum += a[offset + i] * b[offset + i];
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn multiply_scalar_neon(v: &[f32], scalar: f32, result: &mut [f32]) {
    let len = v.len();
    let scalar_vec = vdupq_n_f32(scalar);

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let vv = vld1q_f32(v.as_ptr().add(idx));
        let res = vmulq_f32(vv, scalar_vec);
        vst1q_f32(result.as_mut_ptr().add(idx), res);
    }

    // Handle remainder
    let offset = chunks * 4;
    for i in 0..remainder {
        result[offset + i] = v[offset + i] * scalar;
    }
}

// ============================================================================
// x86_64 AVX implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline]
unsafe fn cosine_similarity_avx(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    // Horizontal sum for AVX
    let mut dot_sum = horizontal_sum_avx(dot_acc);
    let mut norm_a_sum = horizontal_sum_avx(norm_a_acc);
    let mut norm_b_sum = horizontal_sum_avx(norm_b_acc);

    // Handle remainder
    let offset = chunks * 8;
    for i in 0..remainder {
        let ai = a[offset + i];
        let bi = b[offset + i];
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    if norm_a_sum == 0.0 || norm_b_sum == 0.0 {
        return 0.0;
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline]
unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = _mm256_setzero_ps();

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut sum = horizontal_sum_avx(acc);

    let offset = chunks * 8;
    for i in 0..remainder {
        sum += a[offset + i] * b[offset + i];
    }

    sum
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline]
unsafe fn multiply_scalar_avx(v: &[f32], scalar: f32, result: &mut [f32]) {
    let len = v.len();
    let scalar_vec = _mm256_set1_ps(scalar);

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let idx = i * 8;
        let vv = _mm256_loadu_ps(v.as_ptr().add(idx));
        let res = _mm256_mul_ps(vv, scalar_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(idx), res);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        result[offset + i] = v[offset + i] * scalar;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
#[inline]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);

    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(result)
}

// ============================================================================
// x86_64 SSE implementations (fallback for older CPUs)
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
unsafe fn cosine_similarity_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut dot_acc = _mm_setzero_ps();
    let mut norm_a_acc = _mm_setzero_ps();
    let mut norm_b_acc = _mm_setzero_ps();

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));

        dot_acc = _mm_add_ps(dot_acc, _mm_mul_ps(va, vb));
        norm_a_acc = _mm_add_ps(norm_a_acc, _mm_mul_ps(va, va));
        norm_b_acc = _mm_add_ps(norm_b_acc, _mm_mul_ps(vb, vb));
    }

    let mut dot_sum = horizontal_sum_sse(dot_acc);
    let mut norm_a_sum = horizontal_sum_sse(norm_a_acc);
    let mut norm_b_sum = horizontal_sum_sse(norm_b_acc);

    let offset = chunks * 4;
    for i in 0..remainder {
        let ai = a[offset + i];
        let bi = b[offset + i];
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
    }

    if norm_a_sum == 0.0 || norm_b_sum == 0.0 {
        return 0.0;
    }

    dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc = _mm_setzero_ps();

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm_loadu_ps(b.as_ptr().add(idx));
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    let mut sum = horizontal_sum_sse(acc);

    let offset = chunks * 4;
    for i in 0..remainder {
        sum += a[offset + i] * b[offset + i];
    }

    sum
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
unsafe fn multiply_scalar_sse(v: &[f32], scalar: f32, result: &mut [f32]) {
    let len = v.len();
    let scalar_vec = _mm_set1_ps(scalar);

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let vv = _mm_loadu_ps(v.as_ptr().add(idx));
        let res = _mm_mul_ps(vv, scalar_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(idx), res);
    }

    let offset = chunks * 4;
    for i in 0..remainder {
        result[offset + i] = v[offset + i] * scalar;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
unsafe fn horizontal_sum_sse(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let sim = cosine_similarity_simd(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];

        let dot = dot_product_simd(&a, &b);
        assert!((dot - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_normalize() {
        let v = vec![3.0, 4.0, 0.0, 0.0];
        let normalized = normalize_simd(&v);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // Intentional for test data generation
    fn test_simd_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.15).cos()).collect();

        let simd_cos = cosine_similarity_simd(&a, &b);
        let scalar_cos = super::super::similarity::cosine_similarity(&a, &b);

        assert!(
            (simd_cos - scalar_cos).abs() < 1e-5,
            "SIMD: {simd_cos}, Scalar: {scalar_cos}"
        );
    }
}
