//! Context fingerprinting for cache key generation.
//!
//! This module provides utilities for generating unique fingerprints
//! from text content, enabling efficient cache lookup and prefix matching.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::types::ContextFingerprint;

/// Maximum length of content summary in characters.
const MAX_SUMMARY_LENGTH: usize = 50;

/// Generator for context fingerprints.
///
/// The fingerprint generator creates unique identifiers for text content
/// that can be used as cache keys. It supports prefix-aware fingerprinting
/// for partial cache hits.
///
/// # Example
///
/// ```rust,ignore
/// use oxirag::prefix_cache::ContextFingerprintGenerator;
///
/// let generator = ContextFingerprintGenerator::new();
/// let fingerprint = generator.generate("This is some context text");
///
/// println!("Hash: {}", fingerprint.hash);
/// println!("Summary: {}", fingerprint.content_summary);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ContextFingerprintGenerator {
    /// Seed value for hash computation.
    seed: u64,
}

impl ContextFingerprintGenerator {
    /// Create a new fingerprint generator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new fingerprint generator with a custom seed.
    #[must_use]
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate a fingerprint from text content.
    ///
    /// The fingerprint includes:
    /// - A hash of the entire content
    /// - The content length (as prefix length)
    /// - A truncated summary for debugging
    #[must_use]
    pub fn generate(&self, content: &str) -> ContextFingerprint {
        let hash = self.compute_hash(content);
        let prefix_length = content.len();
        let content_summary = Self::create_summary(content);

        ContextFingerprint::new(hash, prefix_length, content_summary)
    }

    /// Generate a fingerprint from text content with explicit prefix length.
    ///
    /// This is useful when the prefix length represents tokens rather than
    /// characters, or when working with pre-tokenized content.
    #[must_use]
    pub fn generate_with_length(&self, content: &str, prefix_length: usize) -> ContextFingerprint {
        let hash = self.compute_hash(content);
        let content_summary = Self::create_summary(content);

        ContextFingerprint::new(hash, prefix_length, content_summary)
    }

    /// Generate a fingerprint for a prefix of the content.
    ///
    /// This creates a fingerprint for the first `prefix_len` characters of
    /// the content, enabling prefix-based cache lookups.
    #[must_use]
    pub fn generate_prefix(&self, content: &str, prefix_len: usize) -> ContextFingerprint {
        let prefix = if prefix_len >= content.len() {
            content
        } else {
            // Find a char boundary to avoid splitting UTF-8 sequences
            let mut end = prefix_len;
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            &content[..end]
        };

        let hash = self.compute_hash(prefix);
        let content_summary = Self::create_summary(prefix);

        ContextFingerprint::new(hash, prefix.len(), content_summary)
    }

    /// Generate multiple fingerprints for progressively longer prefixes.
    ///
    /// This is useful for finding the longest cached prefix by checking
    /// fingerprints from longest to shortest.
    ///
    /// # Arguments
    ///
    /// * `content` - The full content
    /// * `step` - Size increment between prefixes
    /// * `min_length` - Minimum prefix length to generate
    ///
    /// # Returns
    ///
    /// A vector of fingerprints, sorted from shortest to longest prefix.
    #[must_use]
    pub fn generate_prefix_hierarchy(
        &self,
        content: &str,
        step: usize,
        min_length: usize,
    ) -> Vec<ContextFingerprint> {
        let mut fingerprints = Vec::new();
        let mut current_len = min_length.min(content.len());

        while current_len <= content.len() {
            fingerprints.push(self.generate_prefix(content, current_len));
            if current_len == content.len() {
                break;
            }
            current_len = (current_len + step).min(content.len());
        }

        fingerprints
    }

    /// Compute a hash for the given content.
    fn compute_hash(&self, content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Create a summary of the content for debugging purposes.
    fn create_summary(content: &str) -> String {
        if content.len() <= MAX_SUMMARY_LENGTH {
            content.to_string()
        } else {
            let mut end = MAX_SUMMARY_LENGTH - 3;
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...", &content[..end])
        }
    }

    /// Check if two fingerprints represent the same content.
    #[must_use]
    pub fn matches(&self, fp1: &ContextFingerprint, fp2: &ContextFingerprint) -> bool {
        fp1.hash == fp2.hash && fp1.prefix_length == fp2.prefix_length
    }

    /// Check if fp1 could be a prefix of fp2.
    ///
    /// Note: This is a heuristic based on lengths. For exact prefix checking,
    /// the original content would need to be compared.
    #[must_use]
    pub fn could_be_prefix(&self, fp1: &ContextFingerprint, fp2: &ContextFingerprint) -> bool {
        fp1.prefix_length <= fp2.prefix_length
    }
}

/// Trait for types that can generate a context fingerprint.
pub trait Fingerprintable {
    /// Generate a fingerprint for this content.
    fn fingerprint(&self) -> ContextFingerprint;

    /// Generate a fingerprint using a custom generator.
    fn fingerprint_with(&self, generator: &ContextFingerprintGenerator) -> ContextFingerprint;
}

impl Fingerprintable for str {
    fn fingerprint(&self) -> ContextFingerprint {
        ContextFingerprintGenerator::new().generate(self)
    }

    fn fingerprint_with(&self, generator: &ContextFingerprintGenerator) -> ContextFingerprint {
        generator.generate(self)
    }
}

impl Fingerprintable for String {
    fn fingerprint(&self) -> ContextFingerprint {
        ContextFingerprintGenerator::new().generate(self)
    }

    fn fingerprint_with(&self, generator: &ContextFingerprintGenerator) -> ContextFingerprint {
        generator.generate(self)
    }
}

/// Compute a rolling hash for incremental fingerprinting.
///
/// This enables efficient computation of fingerprints for progressively
/// longer prefixes without rehashing the entire content.
#[derive(Debug, Clone)]
pub struct RollingHasher {
    /// Base for polynomial rolling hash.
    base: u64,
    /// Current hash value.
    current_hash: u64,
    /// Power of base for current length.
    base_power: u64,
    /// Current content length.
    length: usize,
}

impl RollingHasher {
    /// Create a new rolling hasher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: 31,
            current_hash: 0,
            base_power: 1,
            length: 0,
        }
    }

    /// Create a rolling hasher with a custom base.
    #[must_use]
    pub fn with_base(base: u64) -> Self {
        Self {
            base,
            current_hash: 0,
            base_power: 1,
            length: 0,
        }
    }

    /// Append a character to the hash.
    pub fn append(&mut self, c: char) {
        let char_value = c as u64;
        self.current_hash = self
            .current_hash
            .wrapping_mul(self.base)
            .wrapping_add(char_value);
        self.base_power = self.base_power.wrapping_mul(self.base);
        self.length += 1;
    }

    /// Append a string to the hash.
    pub fn append_str(&mut self, s: &str) {
        for c in s.chars() {
            self.append(c);
        }
    }

    /// Get the current hash value.
    #[must_use]
    pub fn hash(&self) -> u64 {
        self.current_hash
    }

    /// Get the current content length.
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Reset the hasher to initial state.
    pub fn reset(&mut self) {
        self.current_hash = 0;
        self.base_power = 1;
        self.length = 0;
    }

    /// Create a fingerprint from the current state.
    #[must_use]
    pub fn to_fingerprint(&self, content_summary: impl Into<String>) -> ContextFingerprint {
        ContextFingerprint::new(self.current_hash, self.length, content_summary)
    }
}

impl Default for RollingHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_generator_basic() {
        let generator = ContextFingerprintGenerator::new();
        let fp = generator.generate("Hello, world!");

        assert_eq!(fp.prefix_length, 13);
        assert_eq!(fp.content_summary, "Hello, world!");
    }

    #[test]
    fn test_fingerprint_generator_with_seed() {
        let generator1 = ContextFingerprintGenerator::with_seed(42);
        let generator2 = ContextFingerprintGenerator::with_seed(43);

        let fp1 = generator1.generate("test");
        let fp2 = generator2.generate("test");

        // Different seeds should produce different hashes
        assert_ne!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_fingerprint_generator_long_content() {
        let generator = ContextFingerprintGenerator::new();
        let content = "A".repeat(100);
        let fp = generator.generate(&content);

        assert_eq!(fp.prefix_length, 100);
        assert!(fp.content_summary.ends_with("..."));
        assert!(fp.content_summary.len() <= MAX_SUMMARY_LENGTH);
    }

    #[test]
    fn test_fingerprint_generator_deterministic() {
        let generator = ContextFingerprintGenerator::new();
        let fp1 = generator.generate("test content");
        let fp2 = generator.generate("test content");

        assert_eq!(fp1.hash, fp2.hash);
        assert_eq!(fp1.prefix_length, fp2.prefix_length);
    }

    #[test]
    fn test_fingerprint_generator_different_content() {
        let generator = ContextFingerprintGenerator::new();
        let fp1 = generator.generate("content A");
        let fp2 = generator.generate("content B");

        assert_ne!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_generate_with_length() {
        let generator = ContextFingerprintGenerator::new();
        let fp = generator.generate_with_length("test content", 50);

        assert_eq!(fp.prefix_length, 50);
        assert_eq!(fp.content_summary, "test content");
    }

    #[test]
    fn test_generate_prefix() {
        let generator = ContextFingerprintGenerator::new();
        let full = generator.generate("Hello, world!");
        let prefix = generator.generate_prefix("Hello, world!", 5);

        assert_ne!(full.hash, prefix.hash);
        assert_eq!(prefix.prefix_length, 5);
    }

    #[test]
    fn test_generate_prefix_utf8_safe() {
        let generator = ContextFingerprintGenerator::new();
        // Japanese text: each character is 3 bytes
        let content = "\u{3042}\u{3044}\u{3046}"; // "あいう"
        let prefix = generator.generate_prefix(content, 4); // 4 bytes cuts in middle of char

        // Should safely cut at character boundary
        assert!(prefix.prefix_length <= 4);
    }

    #[test]
    fn test_generate_prefix_hierarchy() {
        let generator = ContextFingerprintGenerator::new();
        let fps = generator.generate_prefix_hierarchy("Hello world!", 3, 3);

        assert!(!fps.is_empty());
        // Should be sorted by length
        for i in 1..fps.len() {
            assert!(fps[i].prefix_length >= fps[i - 1].prefix_length);
        }
        // Last should cover full content
        assert_eq!(fps.last().unwrap().prefix_length, 12);
    }

    #[test]
    fn test_matches() {
        let generator = ContextFingerprintGenerator::new();
        let fp1 = generator.generate("test");
        let fp2 = generator.generate("test");
        let fp3 = generator.generate("other");

        assert!(generator.matches(&fp1, &fp2));
        assert!(!generator.matches(&fp1, &fp3));
    }

    #[test]
    fn test_could_be_prefix() {
        let generator = ContextFingerprintGenerator::new();
        let short = generator.generate("short");
        let long = generator.generate("longer text");

        assert!(generator.could_be_prefix(&short, &long));
        assert!(!generator.could_be_prefix(&long, &short));
    }

    #[test]
    fn test_fingerprintable_str() {
        let fp = "test content".fingerprint();
        assert_eq!(fp.prefix_length, 12);
    }

    #[test]
    fn test_fingerprintable_string() {
        let s = String::from("test content");
        let fp = s.fingerprint();
        assert_eq!(fp.prefix_length, 12);
    }

    #[test]
    fn test_fingerprintable_with_generator() {
        let generator = ContextFingerprintGenerator::with_seed(42);
        let fp1 = "test".fingerprint_with(&generator);
        let fp2 = "test".fingerprint();

        // Should produce different hashes with different generators
        assert_ne!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_rolling_hasher_basic() {
        let mut hasher = RollingHasher::new();
        hasher.append('a');
        hasher.append('b');
        hasher.append('c');

        assert_eq!(hasher.length(), 3);
        assert_ne!(hasher.hash(), 0);
    }

    #[test]
    fn test_rolling_hasher_append_str() {
        let mut hasher1 = RollingHasher::new();
        hasher1.append_str("abc");

        let mut hasher2 = RollingHasher::new();
        hasher2.append('a');
        hasher2.append('b');
        hasher2.append('c');

        assert_eq!(hasher1.hash(), hasher2.hash());
        assert_eq!(hasher1.length(), hasher2.length());
    }

    #[test]
    fn test_rolling_hasher_reset() {
        let mut hasher = RollingHasher::new();
        hasher.append_str("test");

        assert_ne!(hasher.hash(), 0);
        assert_eq!(hasher.length(), 4);

        hasher.reset();

        assert_eq!(hasher.hash(), 0);
        assert_eq!(hasher.length(), 0);
    }

    #[test]
    fn test_rolling_hasher_to_fingerprint() {
        let mut hasher = RollingHasher::new();
        hasher.append_str("test");

        let fp = hasher.to_fingerprint("test summary");

        assert_eq!(fp.hash, hasher.hash());
        assert_eq!(fp.prefix_length, 4);
        assert_eq!(fp.content_summary, "test summary");
    }

    #[test]
    fn test_rolling_hasher_with_base() {
        let hasher1 = RollingHasher::new();
        let hasher2 = RollingHasher::with_base(37);

        let mut h1 = hasher1;
        let mut h2 = hasher2;

        h1.append_str("test");
        h2.append_str("test");

        // Different bases should produce different hashes
        assert_ne!(h1.hash(), h2.hash());
    }

    #[test]
    fn test_rolling_hasher_incremental() {
        let mut hasher = RollingHasher::new();

        hasher.append_str("Hello");
        let hash1 = hasher.hash();

        hasher.append_str(", world!");
        let hash2 = hasher.hash();

        // Hash should change when more content is added
        assert_ne!(hash1, hash2);
    }
}
