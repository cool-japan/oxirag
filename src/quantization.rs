//! Quantization support for vector embeddings.
//!
//! This module provides types and implementations for quantizing vector embeddings
//! to reduce memory usage and potentially speed up similarity computations.
//!
//! # Supported Quantization Types
//!
//! - `Float32`: No quantization (32-bit float)
//! - `Float16`: Half precision (16-bit float)
//! - `BFloat16`: Brain floating point (16-bit)
//! - `Int8`: Symmetric INT8 quantization
//! - `Int4`: Symmetric INT4 quantization (2 values per byte)
//! - `Binary`: Binary quantization (8 values per byte)

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::option_if_let_else)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::VectorStoreError;
use crate::types::{DocumentId, SearchResult};

/// The quantization type for vector storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantizationType {
    /// 32-bit floating point (no quantization).
    #[default]
    Float32,
    /// 16-bit floating point.
    Float16,
    /// Brain floating point (16-bit).
    BFloat16,
    /// 8-bit signed integer quantization.
    Int8,
    /// 4-bit signed integer quantization (2 values per byte).
    Int4,
    /// Binary quantization (8 values per byte).
    Binary,
}

impl QuantizationType {
    /// Get the number of bits per element for this quantization type.
    #[must_use]
    pub fn bits_per_element(&self) -> usize {
        match self {
            Self::Float32 => 32,
            Self::Float16 | Self::BFloat16 => 16,
            Self::Int8 => 8,
            Self::Int4 => 4,
            Self::Binary => 1,
        }
    }

    /// Calculate the number of bytes needed to store n elements.
    #[must_use]
    pub fn bytes_for_elements(&self, n: usize) -> usize {
        let bits = n * self.bits_per_element();
        (bits + 7) / 8
    }

    /// Check if this quantization type requires scale and zero_point.
    #[must_use]
    pub fn requires_scale(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int4)
    }
}

/// Configuration for quantization operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Target quantization type.
    pub target_type: QuantizationType,
    /// Number of calibration samples for determining scale/zero_point.
    pub calibration_samples: usize,
    /// Whether to use symmetric quantization (zero_point = 0).
    pub symmetric: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            target_type: QuantizationType::Int8,
            calibration_samples: 100,
            symmetric: true,
        }
    }
}

impl QuantizationConfig {
    /// Create a new quantization configuration.
    #[must_use]
    pub fn new(target_type: QuantizationType) -> Self {
        Self {
            target_type,
            ..Default::default()
        }
    }

    /// Set the number of calibration samples.
    #[must_use]
    pub fn with_calibration_samples(mut self, samples: usize) -> Self {
        self.calibration_samples = samples;
        self
    }

    /// Set whether to use symmetric quantization.
    #[must_use]
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
}

/// A quantized tensor representation.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// The quantized data stored as bytes.
    pub data: Vec<u8>,
    /// The original shape of the tensor.
    pub original_shape: Vec<usize>,
    /// The quantization type used.
    pub quantization_type: QuantizationType,
    /// Scale factor for dequantization (for INT8/INT4).
    pub scale: f32,
    /// Zero point for asymmetric quantization (for INT8/INT4).
    pub zero_point: i32,
}

impl QuantizedTensor {
    /// Create a new quantized tensor.
    #[must_use]
    pub fn new(
        data: Vec<u8>,
        original_shape: Vec<usize>,
        quantization_type: QuantizationType,
        scale: f32,
        zero_point: i32,
    ) -> Self {
        Self {
            data,
            original_shape,
            quantization_type,
            scale,
            zero_point,
        }
    }

    /// Get the number of elements in the original tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        if self.original_shape.is_empty() {
            0
        } else {
            self.original_shape.iter().product()
        }
    }

    /// Get the size in bytes of the quantized data.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Calculate the compression ratio compared to float32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.numel() * 4; // f32 = 4 bytes
        if self.data.is_empty() {
            1.0
        } else {
            original_bytes as f32 / self.data.len() as f32
        }
    }
}

/// Trait for quantizing and dequantizing tensors.
pub trait Quantizer: Send + Sync {
    /// Quantize a tensor of f32 values.
    fn quantize(&self, tensor: &[f32], config: &QuantizationConfig) -> QuantizedTensor;

    /// Dequantize a tensor back to f32 values.
    fn dequantize(&self, tensor: &QuantizedTensor) -> Vec<f32>;

    /// Get the quantization type this quantizer produces.
    fn quantization_type(&self) -> QuantizationType;
}

/// INT8 symmetric quantizer.
#[derive(Debug, Clone, Default)]
pub struct Int8Quantizer;

impl Int8Quantizer {
    /// Create a new INT8 quantizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute the scale factor for symmetric INT8 quantization.
    fn compute_scale(tensor: &[f32]) -> f32 {
        let max_abs = tensor
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, |a, b| a.max(b));
        if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 }
    }
}

impl Quantizer for Int8Quantizer {
    fn quantize(&self, tensor: &[f32], _config: &QuantizationConfig) -> QuantizedTensor {
        let scale = Self::compute_scale(tensor);
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let data: Vec<u8> = tensor
            .iter()
            .map(|&x| {
                let quantized = (x * inv_scale).round().clamp(-128.0, 127.0) as i8;
                quantized as u8
            })
            .collect();

        QuantizedTensor::new(
            data,
            vec![tensor.len()],
            QuantizationType::Int8,
            scale,
            0, // symmetric quantization
        )
    }

    fn dequantize(&self, tensor: &QuantizedTensor) -> Vec<f32> {
        tensor
            .data
            .iter()
            .map(|&b| {
                let quantized = b as i8;
                quantized as f32 * tensor.scale
            })
            .collect()
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Int8
    }
}

/// INT4 symmetric quantizer (2 values packed per byte).
#[derive(Debug, Clone, Default)]
pub struct Int4Quantizer;

impl Int4Quantizer {
    /// Create a new INT4 quantizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute the scale factor for symmetric INT4 quantization.
    fn compute_scale(tensor: &[f32]) -> f32 {
        let max_abs = tensor
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f32, |a, b| a.max(b));
        if max_abs == 0.0 {
            1.0
        } else {
            max_abs / 7.0 // INT4 range: -8 to 7, symmetric uses -7 to 7
        }
    }

    /// Pack two INT4 values into a single byte.
    /// High nibble: first value, Low nibble: second value.
    fn pack_int4(v1: i8, v2: i8) -> u8 {
        let high = ((v1 + 8) as u8) << 4;
        let low = (v2 + 8) as u8;
        high | low
    }

    /// Unpack two INT4 values from a single byte.
    fn unpack_int4(byte: u8) -> (i8, i8) {
        let high = ((byte >> 4) as i8) - 8;
        let low = ((byte & 0x0F) as i8) - 8;
        (high, low)
    }
}

impl Quantizer for Int4Quantizer {
    fn quantize(&self, tensor: &[f32], _config: &QuantizationConfig) -> QuantizedTensor {
        let scale = Self::compute_scale(tensor);
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Quantize values to INT4 range
        let quantized: Vec<i8> = tensor
            .iter()
            .map(|&x| (x * inv_scale).round().clamp(-7.0, 7.0) as i8)
            .collect();

        // Pack two values per byte
        let num_bytes = (quantized.len() + 1) / 2;
        let mut data = Vec::with_capacity(num_bytes);

        for chunk in quantized.chunks(2) {
            let v1 = chunk[0];
            let v2 = if chunk.len() > 1 { chunk[1] } else { 0 };
            data.push(Self::pack_int4(v1, v2));
        }

        QuantizedTensor::new(
            data,
            vec![tensor.len()],
            QuantizationType::Int4,
            scale,
            0, // symmetric quantization
        )
    }

    fn dequantize(&self, tensor: &QuantizedTensor) -> Vec<f32> {
        let numel = tensor.numel();
        let mut result = Vec::with_capacity(numel);

        for (i, &byte) in tensor.data.iter().enumerate() {
            let (v1, v2) = Self::unpack_int4(byte);
            result.push(v1 as f32 * tensor.scale);

            // Only add the second value if we haven't exceeded the original length
            if 2 * i + 1 < numel {
                result.push(v2 as f32 * tensor.scale);
            }
        }

        result
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Int4
    }
}

/// Binary quantizer (8 values packed per byte).
#[derive(Debug, Clone, Default)]
pub struct BinaryQuantizer;

impl BinaryQuantizer {
    /// Create a new binary quantizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Quantizer for BinaryQuantizer {
    fn quantize(&self, tensor: &[f32], _config: &QuantizationConfig) -> QuantizedTensor {
        let num_bytes = (tensor.len() + 7) / 8;
        let mut data = vec![0u8; num_bytes];

        for (i, &value) in tensor.iter().enumerate() {
            if value > 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        QuantizedTensor::new(
            data,
            vec![tensor.len()],
            QuantizationType::Binary,
            1.0, // Binary doesn't use scale
            0,
        )
    }

    fn dequantize(&self, tensor: &QuantizedTensor) -> Vec<f32> {
        let numel = tensor.numel();
        let mut result = Vec::with_capacity(numel);

        for i in 0..numel {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (tensor.data[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }

        result
    }

    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Binary
    }
}

/// Compute Hamming distance between two binary quantized vectors.
#[must_use]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

/// Compute dot product similarity for INT8 quantized vectors.
#[must_use]
pub fn int8_dot_product(a: &[u8], b: &[u8], scale_a: f32, scale_b: f32) -> f32 {
    let sum: i32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i8 as i32) * (y as i8 as i32))
        .sum();
    sum as f32 * scale_a * scale_b
}

/// Compute dot product similarity for INT4 quantized vectors.
#[must_use]
pub fn int4_dot_product(a: &[u8], b: &[u8], numel: usize, scale_a: f32, scale_b: f32) -> f32 {
    let mut sum: i32 = 0;
    let mut idx = 0;

    for (&byte_a, &byte_b) in a.iter().zip(b.iter()) {
        let (a1, a2) = Int4Quantizer::unpack_int4(byte_a);
        let (b1, b2) = Int4Quantizer::unpack_int4(byte_b);

        sum += (a1 as i32) * (b1 as i32);
        idx += 1;

        if idx < numel {
            sum += (a2 as i32) * (b2 as i32);
            idx += 1;
        }
    }

    sum as f32 * scale_a * scale_b
}

/// Quantized document storage entry.
#[derive(Debug, Clone)]
pub struct QuantizedDocument {
    /// Document ID.
    pub id: DocumentId,
    /// Quantized embedding data.
    pub embedding: QuantizedTensor,
    /// Original document content (optional, for retrieval).
    pub content: Option<String>,
    /// Document title (optional).
    pub title: Option<String>,
}

impl QuantizedDocument {
    /// Create a new quantized document.
    #[must_use]
    pub fn new(id: DocumentId, embedding: QuantizedTensor) -> Self {
        Self {
            id,
            embedding,
            content: None,
            title: None,
        }
    }

    /// Set the document content.
    #[must_use]
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set the document title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Trait for quantized vector storage with similarity search.
#[async_trait]
pub trait QuantizedVectorStore: Send + Sync {
    /// Insert a quantized document.
    async fn insert(&mut self, doc: QuantizedDocument) -> Result<(), VectorStoreError>;

    /// Get a document by ID.
    async fn get(&self, id: &DocumentId) -> Result<Option<QuantizedDocument>, VectorStoreError>;

    /// Delete a document by ID.
    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError>;

    /// Search for similar documents using quantized similarity.
    async fn search(
        &self,
        query: &QuantizedTensor,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Get the number of documents in the store.
    async fn count(&self) -> usize;

    /// Clear all documents from the store.
    async fn clear(&mut self) -> Result<(), VectorStoreError>;

    /// Get the quantization type used by this store.
    fn quantization_type(&self) -> QuantizationType;

    /// Get the expected embedding dimension.
    fn dimension(&self) -> usize;
}

/// Mock implementation of a quantized vector store for testing.
#[derive(Debug)]
pub struct MockQuantizedVectorStore {
    documents: HashMap<DocumentId, QuantizedDocument>,
    dimension: usize,
    quantization_type: QuantizationType,
}

impl MockQuantizedVectorStore {
    /// Create a new mock quantized vector store.
    #[must_use]
    pub fn new(dimension: usize, quantization_type: QuantizationType) -> Self {
        Self {
            documents: HashMap::new(),
            dimension,
            quantization_type,
        }
    }

    /// Compute similarity between two quantized tensors.
    fn compute_similarity(a: &QuantizedTensor, b: &QuantizedTensor) -> f32 {
        match a.quantization_type {
            QuantizationType::Binary => {
                let distance = hamming_distance(&a.data, &b.data);
                let max_distance = a.numel() as f32;
                1.0 - (distance as f32 / max_distance)
            }
            QuantizationType::Int8 => {
                let dot = int8_dot_product(&a.data, &b.data, a.scale, b.scale);
                // Normalize by vector magnitudes (approximate)
                let mag_a: f32 = a
                    .data
                    .iter()
                    .map(|&x| (x as i8 as f32).powi(2))
                    .sum::<f32>()
                    .sqrt()
                    * a.scale;
                let mag_b: f32 = b
                    .data
                    .iter()
                    .map(|&x| (x as i8 as f32).powi(2))
                    .sum::<f32>()
                    .sqrt()
                    * b.scale;
                if mag_a == 0.0 || mag_b == 0.0 {
                    0.0
                } else {
                    dot / (mag_a * mag_b)
                }
            }
            QuantizationType::Int4 => {
                let dot = int4_dot_product(&a.data, &b.data, a.numel(), a.scale, b.scale);
                // Simple normalization
                let norm = (a.numel() as f32).sqrt();
                dot / (norm * norm * a.scale * b.scale).max(1.0)
            }
            _ => {
                // For Float types, this shouldn't be called in practice
                0.0
            }
        }
    }
}

#[async_trait]
impl QuantizedVectorStore for MockQuantizedVectorStore {
    async fn insert(&mut self, doc: QuantizedDocument) -> Result<(), VectorStoreError> {
        if doc.embedding.quantization_type != self.quantization_type {
            return Err(VectorStoreError::Index(format!(
                "Quantization type mismatch: expected {:?}, got {:?}",
                self.quantization_type, doc.embedding.quantization_type
            )));
        }
        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<QuantizedDocument>, VectorStoreError> {
        Ok(self.documents.get(id).cloned())
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        Ok(self.documents.remove(id).is_some())
    }

    async fn search(
        &self,
        query: &QuantizedTensor,
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut results: Vec<(QuantizedDocument, f32)> = self
            .documents
            .values()
            .map(|doc| {
                let score = Self::compute_similarity(query, &doc.embedding);
                (doc.clone(), score)
            })
            .filter(|(_, score)| min_score.is_none_or(|min| *score >= min))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results
            .into_iter()
            .enumerate()
            .map(|(rank, (doc, score))| {
                let document =
                    crate::types::Document::new(doc.content.unwrap_or_default()).with_id(doc.id);
                let mut result = SearchResult::new(document, score, rank);
                if let Some(title) = doc.title {
                    result.document.title = Some(title);
                }
                result
            })
            .collect())
    }

    async fn count(&self) -> usize {
        self.documents.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.documents.clear();
        Ok(())
    }

    fn quantization_type(&self) -> QuantizationType {
        self.quantization_type
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Compute the mean squared error between original and dequantized values.
#[must_use]
pub fn compute_quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::MAX;
    }

    let mse: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    mse
}

/// Compute the signal-to-noise ratio in dB.
#[must_use]
pub fn compute_snr_db(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::NEG_INFINITY;
    }

    let signal_power: f32 =
        original.iter().map(|&x| x.powi(2)).sum::<f32>() / original.len() as f32;

    let noise_power: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;

    if noise_power == 0.0 {
        f32::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_bits_per_element() {
        assert_eq!(QuantizationType::Float32.bits_per_element(), 32);
        assert_eq!(QuantizationType::Float16.bits_per_element(), 16);
        assert_eq!(QuantizationType::BFloat16.bits_per_element(), 16);
        assert_eq!(QuantizationType::Int8.bits_per_element(), 8);
        assert_eq!(QuantizationType::Int4.bits_per_element(), 4);
        assert_eq!(QuantizationType::Binary.bits_per_element(), 1);
    }

    #[test]
    fn test_quantization_type_bytes_for_elements() {
        assert_eq!(QuantizationType::Float32.bytes_for_elements(10), 40);
        assert_eq!(QuantizationType::Int8.bytes_for_elements(10), 10);
        assert_eq!(QuantizationType::Int4.bytes_for_elements(10), 5);
        assert_eq!(QuantizationType::Int4.bytes_for_elements(9), 5); // Rounds up
        assert_eq!(QuantizationType::Binary.bytes_for_elements(8), 1);
        assert_eq!(QuantizationType::Binary.bytes_for_elements(9), 2); // Rounds up
    }

    #[test]
    fn test_quantization_config_builder() {
        let config = QuantizationConfig::new(QuantizationType::Int4)
            .with_calibration_samples(200)
            .with_symmetric(false);

        assert_eq!(config.target_type, QuantizationType::Int4);
        assert_eq!(config.calibration_samples, 200);
        assert!(!config.symmetric);
    }

    #[test]
    fn test_int8_quantization_roundtrip() {
        let quantizer = Int8Quantizer::new();
        let config = QuantizationConfig::default();

        let original = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0, 0.25];
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized.len(), original.len());
        assert_eq!(quantized.quantization_type, QuantizationType::Int8);
        assert_eq!(quantized.data.len(), original.len());

        // Check error is reasonably small
        let mse = compute_quantization_error(&original, &dequantized);
        assert!(mse < 0.01, "INT8 MSE too high: {mse}");
    }

    #[test]
    fn test_int4_quantization_roundtrip() {
        let quantizer = Int4Quantizer::new();
        let config = QuantizationConfig::default();

        let original = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0, 0.25];
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized.len(), original.len());
        assert_eq!(quantized.quantization_type, QuantizationType::Int4);
        // INT4 packs 2 values per byte
        assert_eq!(quantized.data.len(), (original.len() + 1) / 2);

        // Check compression ratio
        let ratio = quantized.compression_ratio();
        assert!(ratio > 1.5, "INT4 should compress ~4x, got {ratio}");
    }

    #[test]
    fn test_int4_odd_length() {
        let quantizer = Int4Quantizer::new();
        let config = QuantizationConfig::default();

        let original = vec![0.5, -0.3, 0.8, -0.1, 0.7]; // Odd length
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized.len(), original.len());
    }

    #[test]
    fn test_binary_quantization_roundtrip() {
        let quantizer = BinaryQuantizer::new();
        let config = QuantizationConfig::default();

        let original = vec![0.5, -0.3, 0.8, -0.1, 0.0, 1.0, -1.0, 0.25];
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized.len(), original.len());
        assert_eq!(quantized.quantization_type, QuantizationType::Binary);
        // Binary packs 8 values per byte
        assert_eq!(quantized.data.len(), 1);

        // Check that signs are preserved
        for (o, d) in original.iter().zip(dequantized.iter()) {
            if *o > 0.0 {
                assert!((*d - 1.0).abs() < f32::EPSILON);
            } else {
                assert!((*d + 1.0).abs() < f32::EPSILON);
            }
        }

        // Check compression ratio (should be ~32x for binary)
        let ratio = quantized.compression_ratio();
        assert!(ratio > 20.0, "Binary should compress ~32x, got {ratio}");
    }

    #[test]
    fn test_binary_quantization_longer_vector() {
        let quantizer = BinaryQuantizer::new();
        let config = QuantizationConfig::default();

        let original: Vec<f32> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized.len(), original.len());
        // 100 bits = 13 bytes (rounded up)
        assert_eq!(quantized.data.len(), 13);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b1111_0000_u8, 0b1010_1010_u8];
        let b = vec![0b1111_0000_u8, 0b0101_0101_u8];

        let distance = hamming_distance(&a, &b);
        // First byte: 0 differences, Second byte: 8 differences
        assert_eq!(distance, 8);
    }

    #[test]
    fn test_int8_dot_product() {
        let a = vec![127u8, 0u8, 64u8]; // As i8: 127, 0, 64
        let b = vec![127u8, 0u8, 64u8];

        let result = int8_dot_product(&a, &b, 1.0, 1.0);
        // 127*127 + 0*0 + 64*64 = 16129 + 0 + 4096 = 20225
        assert!((result - 20225.0).abs() < f32::EPSILON);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_quantized_tensor_properties() {
        let tensor =
            QuantizedTensor::new(vec![1, 2, 3, 4], vec![4], QuantizationType::Int8, 0.1, 0);

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.size_bytes(), 4);
        assert!((tensor.compression_ratio() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantized_document_builder() {
        let id = DocumentId::new();
        let tensor = QuantizedTensor::new(vec![1, 2], vec![2], QuantizationType::Int8, 0.1, 0);

        let doc = QuantizedDocument::new(id.clone(), tensor)
            .with_content("test content")
            .with_title("test title");

        assert_eq!(doc.id, id);
        assert_eq!(doc.content, Some("test content".to_string()));
        assert_eq!(doc.title, Some("test title".to_string()));
    }

    #[tokio::test]
    async fn test_mock_quantized_vector_store_basic() {
        let mut store = MockQuantizedVectorStore::new(4, QuantizationType::Int8);

        let quantizer = Int8Quantizer::new();
        let config = QuantizationConfig::default();

        let embedding = quantizer.quantize(&[0.5, 0.3, -0.2, 0.8], &config);
        let doc = QuantizedDocument::new(DocumentId::new(), embedding).with_content("test");

        store.insert(doc.clone()).await.unwrap();
        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&doc.id).await.unwrap();
        assert!(retrieved.is_some());

        let deleted = store.delete(&doc.id).await.unwrap();
        assert!(deleted);
        assert_eq!(store.count().await, 0);
    }

    #[tokio::test]
    async fn test_mock_quantized_vector_store_search() {
        let mut store = MockQuantizedVectorStore::new(4, QuantizationType::Int8);
        let quantizer = Int8Quantizer::new();
        let config = QuantizationConfig::default();

        // Insert some documents
        for i in 0..5 {
            let values: Vec<f32> = (0..4).map(|j| (i + j) as f32 / 10.0).collect();
            let embedding = quantizer.quantize(&values, &config);
            let doc = QuantizedDocument::new(DocumentId::new(), embedding)
                .with_content(format!("doc {i}"));
            store.insert(doc).await.expect("insert failed");
        }

        // Search
        let query = quantizer.quantize(&[0.1, 0.2, 0.3, 0.4], &config);
        let results = store.search(&query, 3, None).await.unwrap();

        assert_eq!(results.len(), 3);
        // Results should be sorted by score (descending)
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[tokio::test]
    async fn test_mock_quantized_vector_store_min_score() {
        let mut store = MockQuantizedVectorStore::new(4, QuantizationType::Binary);
        let quantizer = BinaryQuantizer::new();
        let config = QuantizationConfig::default();

        // Insert documents with different patterns
        let doc1 = QuantizedDocument::new(
            DocumentId::new(),
            quantizer.quantize(&[1.0, 1.0, 1.0, 1.0], &config),
        );
        let doc2 = QuantizedDocument::new(
            DocumentId::new(),
            quantizer.quantize(&[-1.0, -1.0, -1.0, -1.0], &config),
        );

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        // Search with query similar to doc1
        let query = quantizer.quantize(&[1.0, 1.0, 1.0, 1.0], &config);
        let results = store.search(&query, 10, Some(0.9)).await.unwrap();

        // Only doc1 should match with high similarity
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_compute_quantization_error() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let dequantized = vec![1.1, 1.9, 3.1, 3.9];

        let mse = compute_quantization_error(&original, &dequantized);
        // MSE = (0.01 + 0.01 + 0.01 + 0.01) / 4 = 0.01
        assert!((mse - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_compute_snr_db() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let dequantized = vec![1.0, 2.0, 3.0, 4.0]; // Perfect reconstruction

        let snr = compute_snr_db(&original, &dequantized);
        assert!(snr.is_infinite() && snr > 0.0);

        // With some noise
        let noisy = vec![1.1, 1.9, 3.1, 3.9];
        let snr_noisy = compute_snr_db(&original, &noisy);
        assert!(snr_noisy > 0.0);
        assert!(snr_noisy < 100.0);
    }

    #[test]
    fn test_int8_quantization_zero_vector() {
        let quantizer = Int8Quantizer::new();
        let config = QuantizationConfig::default();

        let original = vec![0.0, 0.0, 0.0, 0.0];
        let quantized = quantizer.quantize(&original, &config);
        let dequantized = quantizer.dequantize(&quantized);

        assert_eq!(dequantized, original);
    }

    #[test]
    fn test_int4_pack_unpack() {
        // Test packing and unpacking INT4 values
        for v1 in -7i8..=7 {
            for v2 in -7i8..=7 {
                let packed = Int4Quantizer::pack_int4(v1, v2);
                let (unpacked_v1, unpacked_v2) = Int4Quantizer::unpack_int4(packed);
                assert_eq!(v1, unpacked_v1, "v1 mismatch for ({v1}, {v2})");
                assert_eq!(v2, unpacked_v2, "v2 mismatch for ({v1}, {v2})");
            }
        }
    }

    #[test]
    fn test_similarity_preserved_after_quantization() {
        let quantizer = Int8Quantizer::new();
        let config = QuantizationConfig::default();

        // Two similar vectors
        let a = vec![0.9, 0.1, 0.2, 0.8];
        let b = vec![0.85, 0.15, 0.25, 0.75];

        // Two dissimilar vectors
        let c = vec![-0.9, -0.1, -0.2, -0.8];

        // Compute original cosine similarities
        let original_sim_ab = cosine_similarity(&a, &b);
        let original_sim_ac = cosine_similarity(&a, &c);

        // Quantize and compute quantized similarities
        let qa = quantizer.quantize(&a, &config);
        let qb = quantizer.quantize(&b, &config);
        let qc = quantizer.quantize(&c, &config);

        let quant_sim_ab = MockQuantizedVectorStore::compute_similarity(&qa, &qb);
        let quant_sim_ac = MockQuantizedVectorStore::compute_similarity(&qa, &qc);

        // Similar vectors should still be more similar than dissimilar vectors
        assert!(
            quant_sim_ab > quant_sim_ac,
            "Quantization broke similarity ordering"
        );

        // The relationship should be roughly preserved
        assert!(original_sim_ab > original_sim_ac);
    }

    // Helper function for testing
    #[allow(clippy::float_cmp)]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }
}
