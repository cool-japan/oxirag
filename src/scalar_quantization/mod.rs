//! Scalar quantization for memory-efficient vector storage and retrieval.
//!
//! This module compresses `f32` vectors to **int8** (`i8`) or **1-bit binary**
//! codes, enabling 4× (int8) or 32× (binary) memory savings over full
//! single-precision storage while retaining fast approximate distance
//! computation.
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`SqConfig`] | Dimensionality and bit-width settings |
//! | [`ScalarQuantizer`] | Calibration, encoding, decoding and scoring |
//! | [`QuantizedVector`] | An int8-quantized code with scale/offset metadata |
//! | [`BinaryVector`] | A 1-bit packed code for Hamming-distance retrieval |
//! | [`SqError`] | Configuration and runtime failure type |
//!
//! # Quantisation scheme
//!
//! The quantizer operates in two phases:
//!
//! 1. **Calibration** — pass a representative sample set to
//!    [`ScalarQuantizer::calibrate`]. The quantizer records per-dimension
//!    `[min, max]` ranges and derives a step size
//!    `scale_d = (max_d − min_d) / 255`.
//!
//! 2. **Encoding** — pass individual vectors to
//!    [`ScalarQuantizer::encode`]. Each coefficient is mapped to
//!    `[-128, 127]` via
//!    ```text
//!    code[d] = round((v[d] - min_d) / scale_d).clamp(0, 255) - 128
//!    ```
//!    and decoded approximately by reversing:
//!    ```text
//!    v̂[d] = (code[d] as f32 + 128.0) * scale_d + min_d
//!    ```
//!
//! # Asymmetric distance computation (ADC)
//!
//! [`ScalarQuantizer::asymmetric_dot`] scores a float query against an int8
//! database vector *without* fully decoding it:
//!
//! ```text
//! dot(q, v̂) = Σ_d q[d] · ((code[d] as f32 + 128.0) · scale_d + offset_d)
//! ```
//!
//! This avoids materialising the decoded vector and is suitable for large-scale
//! top-k retrieval.
//!
//! # Binary quantization
//!
//! [`ScalarQuantizer::encode_binary`] thresholds each coefficient at `0.0` and
//! packs the result into bytes (MSB-first). Similarity between binary codes is
//! measured with [`ScalarQuantizer::hamming_distance`] (XOR + popcount).
//! Binary encoding does not require prior calibration.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "scalar-quantization")] {
//! use oxirag::scalar_quantization::{ScalarQuantizer, SqConfig};
//!
//! // 1. Configure and calibrate.
//! let config = SqConfig::new().with_dim(4).with_bits(8);
//! let mut sq = ScalarQuantizer::new(config);
//! let samples = vec![
//!     vec![-1.0_f32, 0.0, 0.5, 1.0],
//!     vec![ 0.5_f32, 0.5, 0.0, -0.5],
//! ];
//! sq.calibrate(&samples).unwrap();
//!
//! // 2. Encode and decode — reconstruction within one step size.
//! let v = vec![0.0_f32, 0.25, 0.75, -0.5];
//! let qv = sq.encode(&v).unwrap();
//! let decoded = sq.decode(&qv);
//! assert_eq!(decoded.len(), 4);
//!
//! // 3. Asymmetric dot product (float query × int8 database).
//! let query = vec![1.0_f32, 0.0, 0.0, 0.0];
//! let dot = sq.asymmetric_dot(&query, &qv);
//! assert!((dot - query.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum::<f32>()).abs() < 1e-5);
//!
//! // 4. Binary encoding and Hamming distance.
//! let bv1 = sq.encode_binary(&[1.0, -1.0, 1.0, -1.0]);
//! let bv2 = sq.encode_binary(&[1.0,  1.0, 1.0, -1.0]);
//! let dist = sq.hamming_distance(&bv1, &bv2);
//! assert_eq!(dist, 1); // only bit 1 differs
//! # }
//! ```

pub mod quantizer;
#[cfg(test)]
mod tests;
pub mod types;

pub use quantizer::ScalarQuantizer;
pub use types::{BinaryVector, QuantizedVector, SqConfig, SqError};
