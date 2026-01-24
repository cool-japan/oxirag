//! SIMD-optimized similarity computation for vector operations.
//!
//! This module provides high-performance similarity metrics using SIMD intrinsics
//! where available, with automatic fallback to scalar implementations.
//!
//! # Features
//!
//! - Architecture-specific optimizations (`x86_64` AVX2/SSE, aarch64 NEON)
//! - Automatic runtime detection and selection of best implementation
//! - Scalar fallback for non-SIMD targets
//! - Batch processing for improved throughput
//!
//! # Example
//!
//! ```rust
//! use oxirag::simd_similarity::{SimilarityEngine, SimdBackend};
//!
//! let engine = SimilarityEngine::new();
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![4.0, 3.0, 2.0, 1.0];
//!
//! let similarity = engine.cosine_similarity(&a, &b);
//! println!("Cosine similarity: {}", similarity);
//! println!("Backend used: {:?}", engine.backend());
//! ```

use std::fmt;

// ============================================================================
// SIMD Backend Detection and Selection
// ============================================================================

/// Available SIMD backends for similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// AVX2 (256-bit) on `x86_64`
    Avx2,
    /// SSE4.1 (128-bit) on `x86/x86_64`
    Sse4,
    /// NEON (128-bit) on aarch64
    Neon,
    /// Scalar fallback (no SIMD)
    Scalar,
}

impl fmt::Display for SimdBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Avx2 => write!(f, "AVX2"),
            Self::Sse4 => write!(f, "SSE4.1"),
            Self::Neon => write!(f, "NEON"),
            Self::Scalar => write!(f, "Scalar"),
        }
    }
}

/// Detects the best available SIMD backend at runtime.
#[must_use]
#[allow(unreachable_code)]
pub fn detect_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdBackend::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return SimdBackend::Sse4;
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return SimdBackend::Sse4;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdBackend::Neon;
    }

    // Fallback for architectures without SIMD support
    SimdBackend::Scalar
}

// ============================================================================
// Scalar Implementations (Fallback)
// ============================================================================

/// Scalar dot product computation.
#[inline]
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar L2 norm computation.
#[inline]
fn scalar_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Scalar Euclidean distance computation.
#[inline]
fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Scalar cosine similarity computation.
#[inline]
fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = scalar_dot_product(a, b);
    let norm_a = scalar_l2_norm(a);
    let norm_b = scalar_l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ============================================================================
// x86_64 AVX2 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    /// AVX2 dot product for aligned chunks of 8 floats.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }

            // Horizontal sum of 256-bit register
            let high = _mm256_extractf128_ps(sum, 1);
            let low = _mm256_castps256_ps128(sum);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 8 + i;
                result += a[idx] * b[idx];
            }

            result
        }
    }

    /// AVX2 L2 norm computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn l2_norm_avx2(v: &[f32]) -> f32 {
        unsafe {
            let len = v.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(v.as_ptr().add(offset));
                sum = _mm256_fmadd_ps(va, va, sum);
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(sum, 1);
            let low = _mm256_castps256_ps128(sum);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 8 + i;
                result += v[idx] * v[idx];
            }

            result.sqrt()
        }
    }

    /// AVX2 Euclidean distance computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(sum, 1);
            let low = _mm256_castps256_ps128(sum);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 8 + i;
                let diff = a[idx] - b[idx];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    /// AVX2 cosine similarity computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2", enable = "fma")]
    #[allow(clippy::similar_names)]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut dot_sum = _mm256_setzero_ps();
            let mut norm_a_sum = _mm256_setzero_ps();
            let mut norm_b_sum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));

                dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
                norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
                norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
            }

            // Horizontal sums
            let dot = horizontal_sum_avx2(dot_sum);
            let norm_a_sq = horizontal_sum_avx2(norm_a_sum);
            let norm_b_sq = horizontal_sum_avx2(norm_b_sum);

            // Handle remainder
            let mut dot_rem = 0.0f32;
            let mut norm_a_rem = 0.0f32;
            let mut norm_b_rem = 0.0f32;

            for i in 0..remainder {
                let idx = chunks * 8 + i;
                dot_rem += a[idx] * b[idx];
                norm_a_rem += a[idx] * a[idx];
                norm_b_rem += b[idx] * b[idx];
            }

            let dot_total = dot + dot_rem;
            let norm_a_total = (norm_a_sq + norm_a_rem).sqrt();
            let norm_b_total = (norm_b_sq + norm_b_rem).sqrt();

            if norm_a_total == 0.0 || norm_b_total == 0.0 {
                return 0.0;
            }

            dot_total / (norm_a_total * norm_b_total)
        }
    }

    /// Helper: horizontal sum of 256-bit register.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[allow(unused_unsafe)]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        // In Rust 2024+, target_feature functions may require explicit unsafe blocks
        // Allow unused_unsafe to handle both old and new behavior
        let high = _mm256_extractf128_ps(v, 1);
        let low = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// ============================================================================
// x86/x86_64 SSE4.1 Implementations
// ============================================================================

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse4 {
    #[cfg(target_arch = "x86")]
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    /// SSE4.1 dot product for aligned chunks of 4 floats.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSE4.1 is available.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn dot_product_sse4(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let prod = _mm_mul_ps(va, vb);
                sum = _mm_add_ps(sum, prod);
            }

            // Horizontal sum
            let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                result += a[idx] * b[idx];
            }

            result
        }
    }

    /// SSE4.1 L2 norm computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSE4.1 is available.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn l2_norm_sse4(v: &[f32]) -> f32 {
        unsafe {
            let len = v.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(v.as_ptr().add(offset));
                let sq = _mm_mul_ps(va, va);
                sum = _mm_add_ps(sum, sq);
            }

            // Horizontal sum
            let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                result += v[idx] * v[idx];
            }

            result.sqrt()
        }
    }

    /// SSE4.1 Euclidean distance computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSE4.1 is available.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn euclidean_distance_sse4(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let diff = _mm_sub_ps(va, vb);
                let sq = _mm_mul_ps(diff, diff);
                sum = _mm_add_ps(sum, sq);
            }

            // Horizontal sum
            let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                let diff = a[idx] - b[idx];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    /// SSE4.1 cosine similarity computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSE4.1 is available.
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::similar_names)]
    pub unsafe fn cosine_similarity_sse4(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut dot_sum = _mm_setzero_ps();
            let mut norm_a_sum = _mm_setzero_ps();
            let mut norm_b_sum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));

                dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
                norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
                norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
            }

            // Horizontal sums
            let dot = horizontal_sum_sse4(dot_sum);
            let norm_a_sq = horizontal_sum_sse4(norm_a_sum);
            let norm_b_sq = horizontal_sum_sse4(norm_b_sum);

            // Handle remainder
            let mut dot_rem = 0.0f32;
            let mut norm_a_rem = 0.0f32;
            let mut norm_b_rem = 0.0f32;

            for i in 0..remainder {
                let idx = chunks * 4 + i;
                dot_rem += a[idx] * b[idx];
                norm_a_rem += a[idx] * a[idx];
                norm_b_rem += b[idx] * b[idx];
            }

            let dot_total = dot + dot_rem;
            let norm_a_total = (norm_a_sq + norm_a_rem).sqrt();
            let norm_b_total = (norm_b_sq + norm_b_rem).sqrt();

            if norm_a_total == 0.0 || norm_b_total == 0.0 {
                return 0.0;
            }

            dot_total / (norm_a_total * norm_b_total)
        }
    }

    /// Helper: horizontal sum of 128-bit register.
    ///
    /// # Safety
    ///
    /// Caller must ensure SSE4.1 is available.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    #[allow(unused_unsafe)]
    unsafe fn horizontal_sum_sse4(v: __m128) -> f32 {
        // In Rust 2024+, target_feature functions may require explicit unsafe blocks
        // Allow unused_unsafe to handle both old and new behavior
        let sum64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}

// ============================================================================
// aarch64 NEON Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(clippy::wildcard_imports)]
mod neon {
    use std::arch::aarch64::*;

    /// NEON dot product for aligned chunks of 4 floats.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available (always true on aarch64).
    #[inline]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                sum = vfmaq_f32(sum, va, vb);
            }

            // Horizontal sum
            let mut result = vaddvq_f32(sum);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                result += a[idx] * b[idx];
            }

            result
        }
    }

    /// NEON L2 norm computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available (always true on aarch64).
    #[inline]
    pub unsafe fn l2_norm_neon(v: &[f32]) -> f32 {
        unsafe {
            let len = v.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(v.as_ptr().add(offset));
                sum = vfmaq_f32(sum, va, va);
            }

            // Horizontal sum
            let mut result = vaddvq_f32(sum);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                result += v[idx] * v[idx];
            }

            result.sqrt()
        }
    }

    /// NEON Euclidean distance computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available (always true on aarch64).
    #[inline]
    pub unsafe fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let diff = vsubq_f32(va, vb);
                sum = vfmaq_f32(sum, diff, diff);
            }

            // Horizontal sum
            let mut result = vaddvq_f32(sum);

            // Handle remainder
            for i in 0..remainder {
                let idx = chunks * 4 + i;
                let diff = a[idx] - b[idx];
                result += diff * diff;
            }

            result.sqrt()
        }
    }

    /// NEON cosine similarity computation.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available (always true on aarch64).
    #[inline]
    #[allow(clippy::similar_names)]
    pub unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let len = a.len();
            let chunks = len / 4;
            let remainder = len % 4;

            let mut dot_sum = vdupq_n_f32(0.0);
            let mut norm_a_sum = vdupq_n_f32(0.0);
            let mut norm_b_sum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));

                dot_sum = vfmaq_f32(dot_sum, va, vb);
                norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
                norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
            }

            // Horizontal sums
            let dot = vaddvq_f32(dot_sum);
            let norm_a_sq = vaddvq_f32(norm_a_sum);
            let norm_b_sq = vaddvq_f32(norm_b_sum);

            // Handle remainder
            let mut dot_rem = 0.0f32;
            let mut norm_a_rem = 0.0f32;
            let mut norm_b_rem = 0.0f32;

            for i in 0..remainder {
                let idx = chunks * 4 + i;
                dot_rem += a[idx] * b[idx];
                norm_a_rem += a[idx] * a[idx];
                norm_b_rem += b[idx] * b[idx];
            }

            let dot_total = dot + dot_rem;
            let norm_a_total = (norm_a_sq + norm_a_rem).sqrt();
            let norm_b_total = (norm_b_sq + norm_b_rem).sqrt();

            if norm_a_total == 0.0 || norm_b_total == 0.0 {
                return 0.0;
            }

            dot_total / (norm_a_total * norm_b_total)
        }
    }
}

// ============================================================================
// Public SIMD Functions
// ============================================================================

/// Computes the dot product of two vectors using SIMD when available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The dot product (a · b).
///
/// # Panics
///
/// Panics if the vectors have different lengths.
#[must_use]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We verified AVX2 and FMA are available
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::dot_product_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::dot_product_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        return unsafe { neon::dot_product_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar_dot_product(a, b)
}

/// Computes the L2 (Euclidean) norm of a vector using SIMD when available.
///
/// # Arguments
///
/// * `v` - The vector
///
/// # Returns
///
/// The L2 norm: sqrt(sum(v\[i\]^2)).
#[must_use]
pub fn simd_l2_norm(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We verified AVX2 and FMA are available
            return unsafe { avx2::l2_norm_avx2(v) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::l2_norm_sse4(v) };
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::l2_norm_sse4(v) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        return unsafe { neon::l2_norm_neon(v) };
    }

    #[allow(unreachable_code)]
    scalar_l2_norm(v)
}

/// Computes the Euclidean distance between two vectors using SIMD when available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The Euclidean distance: sqrt(sum((a\[i\] - b\[i\])^2)).
///
/// # Panics
///
/// Panics if the vectors have different lengths.
#[must_use]
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We verified AVX2 and FMA are available
            return unsafe { avx2::euclidean_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::euclidean_distance_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::euclidean_distance_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        return unsafe { neon::euclidean_distance_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar_euclidean_distance(a, b)
}

/// Computes the cosine similarity between two vectors using SIMD when available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The cosine similarity: (a · b) / (||a|| * ||b||).
/// Returns 0.0 if either vector has zero norm.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
#[must_use]
pub fn simd_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We verified AVX2 and FMA are available
            return unsafe { avx2::cosine_similarity_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::cosine_similarity_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: We verified SSE4.1 is available
            return unsafe { sse4::cosine_similarity_sse4(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        return unsafe { neon::cosine_similarity_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar_cosine_similarity(a, b)
}

/// Computes batch cosine similarities between a query and multiple vectors using SIMD.
///
/// # Arguments
///
/// * `query` - The query vector
/// * `vectors` - Slice of vectors to compare against
///
/// # Returns
///
/// A vector of cosine similarities, one for each input vector.
///
/// # Panics
///
/// Panics if any vector has a different length than the query.
#[must_use]
pub fn simd_batch_cosine(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| simd_cosine_similarity(query, v))
        .collect()
}

// ============================================================================
// Similarity Engine
// ============================================================================

/// A similarity computation engine that auto-selects the best SIMD implementation.
///
/// The engine detects available SIMD features at construction time and uses
/// the most efficient implementation for all subsequent operations.
#[derive(Debug, Clone)]
pub struct SimilarityEngine {
    backend: SimdBackend,
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityEngine {
    /// Creates a new `SimilarityEngine` with auto-detected backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: detect_backend(),
        }
    }

    /// Creates a new `SimilarityEngine` with a specific backend.
    ///
    /// Note: If the requested backend is not available on the current hardware,
    /// it will fall back to scalar implementation at runtime.
    #[must_use]
    pub fn with_backend(backend: SimdBackend) -> Self {
        Self { backend }
    }

    /// Returns the backend being used by this engine.
    #[must_use]
    pub fn backend(&self) -> SimdBackend {
        self.backend
    }

    /// Computes the dot product of two vectors.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    #[must_use]
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        match self.backend {
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: We verified AVX2 and FMA are available
                    return unsafe { avx2::dot_product_avx2(a, b) };
                }
                scalar_dot_product(a, b)
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Sse4 => {
                if is_x86_feature_detected!("sse4.1") {
                    // SAFETY: We verified SSE4.1 is available
                    return unsafe { sse4::dot_product_sse4(a, b) };
                }
                scalar_dot_product(a, b)
            }
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => {
                // SAFETY: NEON is always available on aarch64
                unsafe { neon::dot_product_neon(a, b) }
            }
            _ => scalar_dot_product(a, b),
        }
    }

    /// Computes the L2 norm of a vector.
    #[must_use]
    pub fn l2_norm(&self, v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }

        match self.backend {
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: We verified AVX2 and FMA are available
                    return unsafe { avx2::l2_norm_avx2(v) };
                }
                scalar_l2_norm(v)
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Sse4 => {
                if is_x86_feature_detected!("sse4.1") {
                    // SAFETY: We verified SSE4.1 is available
                    return unsafe { sse4::l2_norm_sse4(v) };
                }
                scalar_l2_norm(v)
            }
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => {
                // SAFETY: NEON is always available on aarch64
                unsafe { neon::l2_norm_neon(v) }
            }
            _ => scalar_l2_norm(v),
        }
    }

    /// Computes the Euclidean distance between two vectors.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    #[must_use]
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        match self.backend {
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: We verified AVX2 and FMA are available
                    return unsafe { avx2::euclidean_distance_avx2(a, b) };
                }
                scalar_euclidean_distance(a, b)
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Sse4 => {
                if is_x86_feature_detected!("sse4.1") {
                    // SAFETY: We verified SSE4.1 is available
                    return unsafe { sse4::euclidean_distance_sse4(a, b) };
                }
                scalar_euclidean_distance(a, b)
            }
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => {
                // SAFETY: NEON is always available on aarch64
                unsafe { neon::euclidean_distance_neon(a, b) }
            }
            _ => scalar_euclidean_distance(a, b),
        }
    }

    /// Computes the cosine similarity between two vectors.
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    #[must_use]
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        match self.backend {
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2 => {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: We verified AVX2 and FMA are available
                    return unsafe { avx2::cosine_similarity_avx2(a, b) };
                }
                scalar_cosine_similarity(a, b)
            }
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            SimdBackend::Sse4 => {
                if is_x86_feature_detected!("sse4.1") {
                    // SAFETY: We verified SSE4.1 is available
                    return unsafe { sse4::cosine_similarity_sse4(a, b) };
                }
                scalar_cosine_similarity(a, b)
            }
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => {
                // SAFETY: NEON is always available on aarch64
                unsafe { neon::cosine_similarity_neon(a, b) }
            }
            _ => scalar_cosine_similarity(a, b),
        }
    }

    /// Computes batch cosine similarities between a query and multiple vectors.
    ///
    /// # Panics
    ///
    /// Panics if any vector has a different length than the query.
    #[must_use]
    pub fn batch_cosine(&self, query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| self.cosine_similarity(query, v))
            .collect()
    }

    /// Converts Euclidean distance to a similarity score.
    ///
    /// Uses the formula: similarity = 1 / (1 + distance)
    #[must_use]
    pub fn euclidean_to_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let distance = self.euclidean_distance(a, b);
        1.0 / (1.0 + distance)
    }

    /// Normalizes a vector to unit length.
    #[must_use]
    pub fn normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm = self.l2_norm(v);

        if norm == 0.0 {
            return v.to_vec();
        }

        v.iter().map(|x| x / norm).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ========================================================================
    // Basic Functionality Tests
    // ========================================================================

    #[test]
    fn test_backend_detection() {
        let backend = detect_backend();
        println!("Detected SIMD backend: {backend}");
        // Just ensure detection doesn't panic
        assert!(matches!(
            backend,
            SimdBackend::Avx2 | SimdBackend::Sse4 | SimdBackend::Neon | SimdBackend::Scalar
        ));
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let result = simd_dot_product(&a, &b);
        // 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
        assert!(approx_eq(result, 20.0), "Expected 20.0, got {result}");
    }

    #[test]
    fn test_dot_product_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let result = simd_dot_product(&a, &a);
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        assert!(approx_eq(result, 14.0), "Expected 14.0, got {result}");
    }

    #[test]
    fn test_l2_norm_basic() {
        let v = vec![3.0, 4.0];
        let result = simd_l2_norm(&v);
        // sqrt(9 + 16) = sqrt(25) = 5
        assert!(approx_eq(result, 5.0), "Expected 5.0, got {result}");
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let v = vec![1.0, 0.0, 0.0];
        let result = simd_l2_norm(&v);
        assert!(approx_eq(result, 1.0), "Expected 1.0, got {result}");
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = simd_euclidean_distance(&a, &b);
        // sqrt((3-0)^2 + (4-0)^2) = sqrt(9 + 16) = 5
        assert!(approx_eq(result, 5.0), "Expected 5.0, got {result}");
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let result = simd_euclidean_distance(&a, &a);
        assert!(approx_eq(result, 0.0), "Expected 0.0, got {result}");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let result = simd_cosine_similarity(&a, &a);
        assert!(approx_eq(result, 1.0), "Expected 1.0, got {result}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = simd_cosine_similarity(&a, &b);
        assert!(approx_eq(result, 0.0), "Expected 0.0, got {result}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let result = simd_cosine_similarity(&a, &b);
        assert!(approx_eq(result, -1.0), "Expected -1.0, got {result}");
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_empty_vectors() {
        let empty: Vec<f32> = vec![];
        assert!(approx_eq(simd_dot_product(&empty, &empty), 0.0));
        assert!(approx_eq(simd_l2_norm(&empty), 0.0));
        assert!(approx_eq(simd_euclidean_distance(&empty, &empty), 0.0));
        assert!(approx_eq(simd_cosine_similarity(&empty, &empty), 0.0));
    }

    #[test]
    fn test_zero_vectors() {
        let zeros = vec![0.0; 8];
        let ones = vec![1.0; 8];

        assert!(approx_eq(simd_dot_product(&zeros, &ones), 0.0));
        assert!(approx_eq(simd_l2_norm(&zeros), 0.0));
        assert!(approx_eq(simd_euclidean_distance(&zeros, &zeros), 0.0));
        // Cosine similarity with zero vector should be 0
        assert!(approx_eq(simd_cosine_similarity(&zeros, &ones), 0.0));
    }

    #[test]
    fn test_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];

        assert!(approx_eq(simd_dot_product(&a, &b), 12.0));
        assert!(approx_eq(simd_l2_norm(&a), 3.0));
        assert!(approx_eq(simd_euclidean_distance(&a, &b), 1.0));
        assert!(approx_eq(simd_cosine_similarity(&a, &b), 1.0));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_non_aligned_lengths() {
        // Test with various non-SIMD-aligned lengths
        for len in [1, 2, 3, 5, 7, 9, 11, 13, 15, 17] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

            let dot = simd_dot_product(&a, &b);
            let norm_a = simd_l2_norm(&a);
            let dist = simd_euclidean_distance(&a, &b);
            let cos = simd_cosine_similarity(&a, &b);

            // Compare with scalar
            let scalar_dot = scalar_dot_product(&a, &b);
            let scalar_norm = scalar_l2_norm(&a);
            let scalar_dist = scalar_euclidean_distance(&a, &b);
            let scalar_cos = scalar_cosine_similarity(&a, &b);

            assert!(
                approx_eq(dot, scalar_dot),
                "dot mismatch for len {len}: {dot} vs {scalar_dot}"
            );
            assert!(
                approx_eq(norm_a, scalar_norm),
                "norm mismatch for len {len}: {norm_a} vs {scalar_norm}"
            );
            assert!(
                approx_eq(dist, scalar_dist),
                "dist mismatch for len {len}: {dist} vs {scalar_dist}"
            );
            assert!(
                approx_eq(cos, scalar_cos),
                "cos mismatch for len {len}: {cos} vs {scalar_cos}"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_large_vectors() {
        let len = 1024;
        let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..len).map(|i| ((len - i) as f32) * 0.001).collect();

        let dot = simd_dot_product(&a, &b);
        let norm_a = simd_l2_norm(&a);
        let dist = simd_euclidean_distance(&a, &b);
        let cos = simd_cosine_similarity(&a, &b);

        let scalar_dot = scalar_dot_product(&a, &b);
        let scalar_norm = scalar_l2_norm(&a);
        let scalar_dist = scalar_euclidean_distance(&a, &b);
        let scalar_cos = scalar_cosine_similarity(&a, &b);

        // Allow slightly larger epsilon for large vectors due to floating point accumulation
        let large_epsilon = 1e-3;
        assert!(
            (dot - scalar_dot).abs() < large_epsilon,
            "dot: {dot} vs {scalar_dot}"
        );
        assert!(
            (norm_a - scalar_norm).abs() < large_epsilon,
            "norm: {norm_a} vs {scalar_norm}"
        );
        assert!(
            (dist - scalar_dist).abs() < large_epsilon,
            "dist: {dist} vs {scalar_dist}"
        );
        assert!(
            (cos - scalar_cos).abs() < large_epsilon,
            "cos: {cos} vs {scalar_cos}"
        );
    }

    #[test]
    fn test_negative_values() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let dot = simd_dot_product(&a, &b);
        // -1*1 + -2*2 + -3*3 + -4*4 = -1 - 4 - 9 - 16 = -30
        assert!(approx_eq(dot, -30.0), "Expected -30.0, got {dot}");

        let cos = simd_cosine_similarity(&a, &b);
        // Opposite vectors
        assert!(approx_eq(cos, -1.0), "Expected -1.0, got {cos}");
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    #[test]
    fn test_batch_cosine() {
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![1.0, 0.0, 0.0, 0.0]; // identical
        let v2 = vec![0.0, 1.0, 0.0, 0.0]; // orthogonal
        let v3 = vec![-1.0, 0.0, 0.0, 0.0]; // opposite
        let v4 = vec![0.707, 0.707, 0.0, 0.0]; // 45 degrees

        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3, &v4];
        let results = simd_batch_cosine(&query, &vectors);

        assert_eq!(results.len(), 4);
        assert!(
            approx_eq(results[0], 1.0),
            "Expected 1.0, got {}",
            results[0]
        );
        assert!(
            approx_eq(results[1], 0.0),
            "Expected 0.0, got {}",
            results[1]
        );
        assert!(
            approx_eq(results[2], -1.0),
            "Expected -1.0, got {}",
            results[2]
        );
        assert!(
            (results[3] - 0.707).abs() < 0.01,
            "Expected ~0.707, got {}",
            results[3]
        );
    }

    // ========================================================================
    // Similarity Engine Tests
    // ========================================================================

    #[test]
    fn test_similarity_engine_new() {
        let engine = SimilarityEngine::new();
        let backend = engine.backend();
        println!("Engine backend: {backend}");
        assert!(matches!(
            backend,
            SimdBackend::Avx2 | SimdBackend::Sse4 | SimdBackend::Neon | SimdBackend::Scalar
        ));
    }

    #[test]
    fn test_similarity_engine_default() {
        let engine = SimilarityEngine::default();
        let backend = engine.backend();
        assert!(matches!(
            backend,
            SimdBackend::Avx2 | SimdBackend::Sse4 | SimdBackend::Neon | SimdBackend::Scalar
        ));
    }

    #[test]
    fn test_similarity_engine_with_backend() {
        let engine = SimilarityEngine::with_backend(SimdBackend::Scalar);
        assert_eq!(engine.backend(), SimdBackend::Scalar);
    }

    #[test]
    fn test_similarity_engine_operations() {
        let engine = SimilarityEngine::new();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // Compare engine methods with standalone functions
        assert!(approx_eq(
            engine.dot_product(&a, &b),
            simd_dot_product(&a, &b)
        ));
        assert!(approx_eq(engine.l2_norm(&a), simd_l2_norm(&a)));
        assert!(approx_eq(
            engine.euclidean_distance(&a, &b),
            simd_euclidean_distance(&a, &b)
        ));
        assert!(approx_eq(
            engine.cosine_similarity(&a, &b),
            simd_cosine_similarity(&a, &b)
        ));
    }

    #[test]
    fn test_similarity_engine_normalize() {
        let engine = SimilarityEngine::new();

        let v = vec![3.0, 4.0];
        let normalized = engine.normalize(&v);

        let norm = engine.l2_norm(&normalized);
        assert!(approx_eq(norm, 1.0), "Expected 1.0, got {norm}");

        // Check direction preserved
        let cos = engine.cosine_similarity(&v, &normalized);
        assert!(approx_eq(cos, 1.0), "Expected 1.0, got {cos}");
    }

    #[test]
    fn test_similarity_engine_normalize_zero() {
        let engine = SimilarityEngine::new();

        let v = vec![0.0, 0.0, 0.0];
        let normalized = engine.normalize(&v);

        assert_eq!(normalized, v);
    }

    #[test]
    fn test_similarity_engine_euclidean_to_similarity() {
        let engine = SimilarityEngine::new();

        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0]; // distance = 5

        let sim = engine.euclidean_to_similarity(&a, &b);
        // 1 / (1 + 5) = 1/6
        assert!(
            approx_eq(sim, 1.0 / 6.0),
            "Expected {}, got {sim}",
            1.0 / 6.0
        );
    }

    #[test]
    fn test_similarity_engine_batch_cosine() {
        let engine = SimilarityEngine::new();

        let query = vec![1.0, 0.0];
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];

        let vectors: Vec<&[f32]> = vec![&v1, &v2];
        let results = engine.batch_cosine(&query, &vectors);

        assert_eq!(results.len(), 2);
        assert!(approx_eq(results[0], 1.0));
        assert!(approx_eq(results[1], 0.0));
    }

    // ========================================================================
    // SIMD vs Scalar Correctness
    // ========================================================================

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_simd_scalar_equivalence() {
        // Test multiple vector sizes to ensure SIMD and scalar produce equivalent results
        let sizes = [4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024];

        for &size in &sizes {
            let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).cos()).collect();

            let simd_dot = simd_dot_product(&a, &b);
            let scalar_dot = scalar_dot_product(&a, &b);

            let simd_norm = simd_l2_norm(&a);
            let scalar_norm = scalar_l2_norm(&a);

            let simd_dist = simd_euclidean_distance(&a, &b);
            let scalar_dist = scalar_euclidean_distance(&a, &b);

            let simd_cos = simd_cosine_similarity(&a, &b);
            let scalar_cos = scalar_cosine_similarity(&a, &b);

            // Use relative tolerance for larger values
            let tol = 1e-4;
            assert!(
                (simd_dot - scalar_dot).abs() / scalar_dot.abs().max(1.0) < tol,
                "Size {size}: dot {simd_dot} vs {scalar_dot}"
            );
            assert!(
                (simd_norm - scalar_norm).abs() / scalar_norm.abs().max(1.0) < tol,
                "Size {size}: norm {simd_norm} vs {scalar_norm}"
            );
            assert!(
                (simd_dist - scalar_dist).abs() / scalar_dist.abs().max(1.0) < tol,
                "Size {size}: dist {simd_dist} vs {scalar_dist}"
            );
            assert!(
                (simd_cos - scalar_cos).abs() < tol,
                "Size {size}: cos {simd_cos} vs {scalar_cos}"
            );
        }
    }

    // ========================================================================
    // Panic Tests
    // ========================================================================

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_dot_product_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let _ = simd_dot_product(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_euclidean_distance_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let _ = simd_euclidean_distance(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let _ = simd_cosine_similarity(&a, &b);
    }

    // ========================================================================
    // Benchmark-style Tests (for CI verification, not actual benchmarking)
    // ========================================================================

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_benchmark_simulation() {
        // This test simulates a benchmark workload to ensure the implementation
        // is correct under repeated use
        let engine = SimilarityEngine::new();
        let dimension = 384; // Common embedding dimension
        let iterations = 100;

        let query: Vec<f32> = (0..dimension).map(|i| (i as f32 * 0.001).sin()).collect();
        let vectors: Vec<Vec<f32>> = (0..iterations)
            .map(|j| {
                (0..dimension)
                    .map(|i| ((i + j) as f32 * 0.001).cos())
                    .collect()
            })
            .collect();

        // Compute similarities
        let mut total_sim = 0.0f32;
        for v in &vectors {
            total_sim += engine.cosine_similarity(&query, v);
        }

        // Just verify we got some reasonable aggregate
        println!("Backend: {}", engine.backend());
        println!("Total similarity over {iterations} iterations: {total_sim}");

        // The sum should be a finite number
        assert!(total_sim.is_finite());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_embedding_dimensions() {
        // Test common embedding dimensions
        let dimensions = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096];

        for &dim in &dimensions {
            let a: Vec<f32> = (0..dim).map(|i| i as f32 / dim as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| 1.0 - i as f32 / dim as f32).collect();

            let cos = simd_cosine_similarity(&a, &b);
            let dot = simd_dot_product(&a, &b);
            let dist = simd_euclidean_distance(&a, &b);
            let norm = simd_l2_norm(&a);

            // All values should be finite
            assert!(cos.is_finite(), "dim {dim}: cosine not finite");
            assert!(dot.is_finite(), "dim {dim}: dot not finite");
            assert!(dist.is_finite(), "dim {dim}: distance not finite");
            assert!(norm.is_finite(), "dim {dim}: norm not finite");
        }
    }
}
