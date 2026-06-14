//! Types for the `scalar_quantization` module.
//!
//! Defines the public configuration ([`SqConfig`]), the int8 code record
//! ([`QuantizedVector`]), the binary code record ([`BinaryVector`]) and the
//! error enumeration ([`SqError`]) used throughout scalar quantization.

use thiserror::Error;

// ── SqConfig ─────────────────────────────────────────────────────────────────

/// Configuration for a [`ScalarQuantizer`](super::quantizer::ScalarQuantizer).
///
/// Scalar quantization compresses `dim`-dimensional `f32` vectors to compact
/// integer codes (`bits = 8` → `i8`, `bits = 1` → 1-bit binary packed as
/// bytes), enabling memory-efficient storage and fast asymmetric distance
/// computation.
///
/// # Defaults
///
/// | Field | Default |
/// |-------|---------|
/// | [`dim`](Self::dim)   | `128` |
/// | [`bits`](Self::bits) | `8`   |
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "scalar-quantization")] {
/// use oxirag::scalar_quantization::SqConfig;
///
/// let cfg = SqConfig::new().with_dim(64).with_bits(8);
/// cfg.validate().unwrap();
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqConfig {
    /// Dimensionality of the input vectors.
    ///
    /// Defaults to `128`.
    pub dim: usize,
    /// Number of bits per quantized coefficient.
    ///
    /// Accepted values:
    /// * `8` — int8 quantization; each coefficient stored as `i8`.
    /// * `1` — binary quantization; each coefficient packed as one bit.
    ///
    /// Defaults to `8`.
    pub bits: u8,
}

impl Default for SqConfig {
    fn default() -> Self {
        Self { dim: 128, bits: 8 }
    }
}

impl SqConfig {
    /// Create a new configuration with default values (`dim = 128`, `bits = 8`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input vector dimensionality.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the number of bits per quantized coefficient.
    ///
    /// Accepted values are `1` (binary) and `8` (int8).
    #[must_use]
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`SqError::InvalidConfig`] when `dim` is `0`.
    /// - [`SqError::InvalidConfig`] when `bits` is not `1` or `8`.
    pub fn validate(&self) -> Result<(), SqError> {
        if self.dim == 0 {
            return Err(SqError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.bits != 1 && self.bits != 8 {
            return Err(SqError::InvalidConfig(
                "bits must be 1 (binary) or 8 (int8)".into(),
            ));
        }
        Ok(())
    }
}

// ── SqError ──────────────────────────────────────────────────────────────────

/// Errors returned by the `scalar_quantization` module.
#[derive(Debug, Error, PartialEq, Clone)]
pub enum SqError {
    /// The [`SqConfig`] is invalid (e.g., `dim == 0` or unsupported `bits`).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// The quantizer has not been calibrated yet.
    ///
    /// Call
    /// [`ScalarQuantizer::calibrate`](super::quantizer::ScalarQuantizer::calibrate)
    /// with a non-empty set of representative vectors before encoding.
    #[error("quantizer is not calibrated — call calibrate() first")]
    NotCalibrated,

    /// The supplied vector has a different length than the configured `dim`.
    #[error("vector length {got} does not match configured dim {expected}")]
    DimMismatch {
        /// The expected dimension from the configuration.
        expected: usize,
        /// The actual dimension of the supplied vector.
        got: usize,
    },

    /// The calibration training set is empty.
    #[error("calibration requires at least one sample vector")]
    EmptyTrainingSet,
}

// ── QuantizedVector ──────────────────────────────────────────────────────────

/// An int8-quantized vector.
///
/// Each coefficient in [`data`](Self::data) encodes one dimension of the
/// original `f32` vector as a signed 8-bit integer in `[-128, 127]`.
///
/// The original value is approximately recovered by the calibrated
/// [`ScalarQuantizer`](super::quantizer::ScalarQuantizer) using per-dimension
/// scale and offset:
///
/// ```text
/// v[d] ≈ (data[d] as f32 + 128.0) * scale_d + offset_d
/// ```
///
/// The `scale` and `offset` fields are a convenience copy of the *global*
/// calibration statistics (maximum per-dim step size and global minimum) that
/// allow fast asymmetric dot products without materialising the full decoded
/// vector.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    /// Quantized coefficients: one `i8` per dimension.
    pub data: Vec<i8>,
    /// Representative scale (maximum per-dim step size) for approximate
    /// decoding and metadata purposes.
    pub scale: f32,
    /// Representative offset (global minimum of the calibration set) for
    /// approximate decoding and metadata purposes.
    pub offset: f32,
}

// ── BinaryVector ─────────────────────────────────────────────────────────────

/// A 1-bit binary-quantized vector.
///
/// Each dimension of the source vector is thresholded at `0.0`: values
/// strictly greater than `0.0` become bit `1`; all others become bit `0`.
/// Bits are packed MSB-first into bytes, so logical bit `d` lives in byte
/// `d / 8` at bit position `7 − (d % 8)`.
///
/// Hamming distance between two [`BinaryVector`]s can be computed via
/// [`ScalarQuantizer::hamming_distance`](super::quantizer::ScalarQuantizer::hamming_distance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryVector {
    /// Packed bit array.  Byte length = `(dim + 7) / 8`.
    pub data: Vec<u8>,
    /// Number of logical dimensions (the original vector length before packing).
    pub dim: usize,
}
