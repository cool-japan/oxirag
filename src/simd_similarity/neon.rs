//! ARM NEON intrinsic implementations of similarity kernels.
//!
//! All items in this module are conditionally compiled only on `aarch64`.
//! NEON is mandatorily available on every aarch64 target, so no runtime
//! feature detection is required.

// ============================================================================
// aarch64 NEON Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[allow(clippy::wildcard_imports)]
pub(super) mod neon_impl {
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
