//! The scalar quantizer: calibration, int8 encoding/decoding, binary encoding,
//! asymmetric dot product and Hamming distance.

use super::types::{BinaryVector, QuantizedVector, SqConfig, SqError};

// ── ScalarQuantizer ───────────────────────────────────────────────────────────

/// Scalar quantizer that compresses `f32` vectors to int8 or binary codes.
///
/// # Workflow
///
/// 1. Create with [`ScalarQuantizer::new`] and a validated [`SqConfig`].
/// 2. Call [`calibrate`](Self::calibrate) with a representative set of vectors
///    to derive per-dimension quantization parameters.
/// 3. Use [`encode`](Self::encode) / [`decode`](Self::decode) for int8
///    round-trips, [`encode_binary`](Self::encode_binary) for 1-bit codes,
///    [`asymmetric_dot`](Self::asymmetric_dot) for efficient scoring, and
///    [`hamming_distance`](Self::hamming_distance) for binary similarity.
///
/// # Thread safety
///
/// [`ScalarQuantizer`] is `Send + Sync` once calibrated; all query-time
/// methods take `&self`.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    config: SqConfig,
    /// Per-dimension minimum value learned during calibration.
    /// Empty before the first successful [`calibrate`](Self::calibrate) call.
    dim_min: Vec<f32>,
    /// Per-dimension quantization step size:
    /// `dim_scale[d] = (dim_max[d] - dim_min[d]) / 255.0`.
    /// Empty before the first successful [`calibrate`](Self::calibrate) call.
    dim_scale: Vec<f32>,
    /// `true` once [`calibrate`](Self::calibrate) has completed successfully.
    calibrated: bool,
}

impl ScalarQuantizer {
    /// Create a new, uncalibrated quantizer for the given configuration.
    ///
    /// Configuration validation is deferred to [`calibrate`](Self::calibrate).
    /// This constructor never fails.
    #[must_use]
    pub fn new(config: SqConfig) -> Self {
        Self {
            config,
            dim_min: Vec::new(),
            dim_scale: Vec::new(),
            calibrated: false,
        }
    }

    /// Return `true` if the quantizer has been successfully calibrated.
    #[must_use]
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &SqConfig {
        &self.config
    }

    /// Borrow the per-dimension minimum values (empty before calibration).
    #[must_use]
    pub fn dim_min(&self) -> &[f32] {
        &self.dim_min
    }

    /// Borrow the per-dimension step sizes (empty before calibration).
    #[must_use]
    pub fn dim_scale(&self) -> &[f32] {
        &self.dim_scale
    }

    // ── Calibration ──────────────────────────────────────────────────────────

    /// Calibrate the quantizer from a set of representative vectors.
    ///
    /// Computes per-dimension `[min, max]` ranges across all `samples`, then
    /// derives:
    ///
    /// - `scale_d = (max_d − min_d) / 255` — the size of one quantisation step.
    ///   A tiny floor (`1e-9`) is applied to prevent division by zero in
    ///   zero-range dimensions.
    /// - `offset_d = min_d` — the value that maps to quantised code `−128`.
    ///
    /// Calling `calibrate` a second time replaces the existing parameters.
    ///
    /// # Errors
    ///
    /// - [`SqError::InvalidConfig`] when the configuration is invalid.
    /// - [`SqError::EmptyTrainingSet`] when `samples` is empty.
    /// - [`SqError::DimMismatch`] when any sample has a length different from
    ///   the configured `dim`.
    pub fn calibrate(&mut self, samples: &[Vec<f32>]) -> Result<(), SqError> {
        self.config.validate()?;
        if samples.is_empty() {
            return Err(SqError::EmptyTrainingSet);
        }
        for s in samples {
            if s.len() != self.config.dim {
                return Err(SqError::DimMismatch {
                    expected: self.config.dim,
                    got: s.len(),
                });
            }
        }

        let dim = self.config.dim;
        let mut dim_min = vec![f32::INFINITY; dim];
        let mut dim_max = vec![f32::NEG_INFINITY; dim];

        for sample in samples {
            for (d, &val) in sample.iter().enumerate() {
                if val < dim_min[d] {
                    dim_min[d] = val;
                }
                if val > dim_max[d] {
                    dim_max[d] = val;
                }
            }
        }

        // Derive per-dim scale with a small floor to guard zero-range dims.
        let dim_scale: Vec<f32> = dim_min
            .iter()
            .zip(dim_max.iter())
            .map(|(&lo, &hi)| {
                let range = hi - lo;
                if range < 1e-9_f32 {
                    1e-9_f32
                } else {
                    range / 255.0_f32
                }
            })
            .collect();

        self.dim_min = dim_min;
        self.dim_scale = dim_scale;
        self.calibrated = true;
        Ok(())
    }

    // ── Int8 encode / decode ─────────────────────────────────────────────────

    /// Encode a `f32` vector as int8 quantized codes.
    ///
    /// Each dimension is independently mapped to `[−128, 127]` using the
    /// calibrated per-dimension `min` and `scale`:
    ///
    /// ```text
    /// code[d] = round((v[d] - min_d) / scale_d).clamp(0, 255) - 128
    /// ```
    ///
    /// Values outside the calibration range are clamped to `−128` or `127`.
    ///
    /// The returned [`QuantizedVector`] carries convenience `scale` (maximum
    /// per-dim step) and `offset` (global minimum) fields.
    ///
    /// # Errors
    ///
    /// - [`SqError::NotCalibrated`] if the quantizer has not been calibrated.
    /// - [`SqError::DimMismatch`] if `v.len() != config.dim`.
    pub fn encode(&self, v: &[f32]) -> Result<QuantizedVector, SqError> {
        if !self.calibrated {
            return Err(SqError::NotCalibrated);
        }
        if v.len() != self.config.dim {
            return Err(SqError::DimMismatch {
                expected: self.config.dim,
                got: v.len(),
            });
        }

        let data: Vec<i8> = v
            .iter()
            .zip(self.dim_min.iter())
            .zip(self.dim_scale.iter())
            .map(|((&val, &lo), &scale)| encode_one(val, lo, scale))
            .collect();

        // Convenience global stats stored on the vector.
        let global_max_scale = self
            .dim_scale
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let global_min = self.dim_min.iter().copied().fold(f32::INFINITY, f32::min);

        Ok(QuantizedVector {
            data,
            scale: global_max_scale,
            offset: global_min,
        })
    }

    /// Decode an int8 [`QuantizedVector`] back to approximate `f32` values.
    ///
    /// Each coefficient is reconstructed as
    /// `(code + 128.0) * scale_d + offset_d`.
    ///
    /// Returns an empty `Vec` when the quantizer is not calibrated or the
    /// code length does not match `dim`.
    #[must_use]
    pub fn decode(&self, qv: &QuantizedVector) -> Vec<f32> {
        if !self.calibrated || qv.data.len() != self.config.dim {
            return Vec::new();
        }
        qv.data
            .iter()
            .zip(self.dim_min.iter())
            .zip(self.dim_scale.iter())
            .map(|((&code, &lo), &scale)| (f32::from(code) + 128.0_f32) * scale + lo)
            .collect()
    }

    // ── Binary encode ─────────────────────────────────────────────────────────

    /// Encode a `f32` vector as a packed 1-bit binary code.
    ///
    /// Each dimension is thresholded at `0.0`: values strictly greater than
    /// `0.0` become bit `1`; all others become bit `0`. Bits are packed
    /// MSB-first into bytes, so logical dimension `d` lives in byte `d / 8`
    /// at bit position `7 − (d % 8)`.
    ///
    /// The resulting [`BinaryVector`] has `data.len() == (v.len() + 7) / 8`.
    /// This method does **not** require prior calibration.
    #[must_use]
    pub fn encode_binary(&self, v: &[f32]) -> BinaryVector {
        let dim = v.len();
        let num_bytes = dim.div_ceil(8);
        let mut data = vec![0u8; num_bytes];
        for (d, &val) in v.iter().enumerate() {
            if val > 0.0_f32 {
                let byte_idx = d / 8;
                let bit_pos = 7 - (d % 8);
                data[byte_idx] |= 1u8 << bit_pos;
            }
        }
        BinaryVector { data, dim }
    }

    // ── Asymmetric dot ────────────────────────────────────────────────────────

    /// Compute the asymmetric dot product between a float query and an int8
    /// quantized database vector.
    ///
    /// Avoids full decoding by expanding element-wise:
    ///
    /// ```text
    /// dot(q, decoded) = Σ_d  q[d] · ((code[d] as f32 + 128.0) · scale_d + offset_d)
    /// ```
    ///
    /// This is more efficient than calling [`decode`](Self::decode) and then
    /// computing the dot product, because the decoded vector is never
    /// materialised.
    ///
    /// Returns `0.0` when the quantizer is not calibrated or the lengths do
    /// not agree (`query.len() != qv.data.len()`).
    #[must_use]
    pub fn asymmetric_dot(&self, query: &[f32], qv: &QuantizedVector) -> f32 {
        if !self.calibrated || query.len() != qv.data.len() {
            return 0.0_f32;
        }
        query
            .iter()
            .zip(qv.data.iter())
            .zip(self.dim_min.iter())
            .zip(self.dim_scale.iter())
            .map(|(((&q, &code), &lo), &scale)| {
                let decoded = (f32::from(code) + 128.0_f32) * scale + lo;
                q * decoded
            })
            .sum()
    }

    // ── Hamming distance ──────────────────────────────────────────────────────

    /// Compute the Hamming distance between two [`BinaryVector`]s.
    ///
    /// Returns the total number of differing bits (popcount of XOR). The
    /// common prefix of bytes is XOR-ed and popcount-ed; surplus bytes from
    /// the longer vector (if lengths differ) count as fully differing.
    ///
    /// Complexity: O(`a.data.len()` + `b.data.len()`).
    #[must_use]
    pub fn hamming_distance(&self, a: &BinaryVector, b: &BinaryVector) -> u32 {
        let common = a.data.len().min(b.data.len());
        let mut dist: u32 = a.data[..common]
            .iter()
            .zip(b.data[..common].iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum();
        // Surplus bytes of the longer vector are treated as all-different
        // (i.e., XOR against 0xFF would give full popcount).
        for &byte in &a.data[common..] {
            dist += byte.count_ones();
        }
        for &byte in &b.data[common..] {
            dist += byte.count_ones();
        }
        dist
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Encode a single `f32` scalar to an `i8` code using the given per-dimension
/// `lo` (minimum) and `scale` (step size).
///
/// The mapping is:
/// ```text
/// code = round((val - lo) / scale).clamp(0, 255) - 128
/// ```
fn encode_one(val: f32, lo: f32, scale: f32) -> i8 {
    // Map val into [0.0, 255.0].
    let normalized = ((val - lo) / scale).round().clamp(0.0_f32, 255.0_f32);
    // Shift to [-128, 127].  The value fits in i32 so no overflow occurs.
    #[allow(clippy::cast_possible_truncation)]
    let code = normalized as i32 - 128;
    // The clamp above guarantees code ∈ [-128, 127], so the cast is lossless.
    #[allow(clippy::cast_possible_truncation)]
    let result = code as i8;
    result
}
