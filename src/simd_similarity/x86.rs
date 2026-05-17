//! `x86_64` AVX2/SSE4.1 intrinsic implementations of similarity kernels.
//!
//! All public functions in this module are conditionally compiled only on
//! `x86_64` (AVX2 submodule) or `x86`/`x86_64` (SSE4 submodule) targets.

// ============================================================================
// x86_64 AVX2 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
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
        // In Rust 2024+, target_feature functions may require explicit unsafe blocks.
        // Allow unused_unsafe to handle both old and new behavior.
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
pub(super) mod sse4 {
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
        // In Rust 2024+, target_feature functions may require explicit unsafe blocks.
        // Allow unused_unsafe to handle both old and new behavior.
        let sum64 = _mm_add_ps(v, _mm_movehl_ps(v, v));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        _mm_cvtss_f32(sum32)
    }
}
