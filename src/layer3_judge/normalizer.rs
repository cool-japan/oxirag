//! Claim normalization and deduplication for improved verification quality.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::types::{ClaimStructure, LogicalClaim};

/// A trait for normalizing claims to canonical form.
pub trait ClaimNormalizer: Send + Sync {
    /// Normalize a claim to canonical form.
    fn normalize(&self, claim: &LogicalClaim) -> LogicalClaim;

    /// Normalize claim text.
    fn normalize_text(&self, text: &str) -> String;
}

/// Default claim normalizer using basic text normalization.
pub struct DefaultClaimNormalizer {
    /// Whether to lowercase text.
    lowercase: bool,
    /// Whether to remove extra whitespace.
    normalize_whitespace: bool,
    /// Whether to remove punctuation.
    remove_punctuation: bool,
}

impl Default for DefaultClaimNormalizer {
    fn default() -> Self {
        Self {
            lowercase: true,
            normalize_whitespace: true,
            remove_punctuation: false,
        }
    }
}

impl DefaultClaimNormalizer {
    /// Create a new default claim normalizer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to lowercase text.
    #[must_use]
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set whether to normalize whitespace.
    #[must_use]
    pub fn with_normalize_whitespace(mut self, normalize: bool) -> Self {
        self.normalize_whitespace = normalize;
        self
    }

    /// Set whether to remove punctuation.
    #[must_use]
    pub fn with_remove_punctuation(mut self, remove: bool) -> Self {
        self.remove_punctuation = remove;
        self
    }

    /// Normalize a claim structure recursively.
    fn normalize_structure(&self, structure: &ClaimStructure) -> ClaimStructure {
        match structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object,
            } => ClaimStructure::Predicate {
                subject: self.normalize_text(subject),
                predicate: self.normalize_text(predicate),
                object: object.as_ref().map(|o| self.normalize_text(o)),
            },
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => ClaimStructure::Comparison {
                left: self.normalize_text(left),
                operator: *operator,
                right: self.normalize_text(right),
            },
            ClaimStructure::And(claims) => {
                ClaimStructure::And(claims.iter().map(|c| self.normalize_structure(c)).collect())
            }
            ClaimStructure::Or(claims) => {
                ClaimStructure::Or(claims.iter().map(|c| self.normalize_structure(c)).collect())
            }
            ClaimStructure::Not(inner) => {
                ClaimStructure::Not(Box::new(self.normalize_structure(inner)))
            }
            ClaimStructure::Implies {
                premise,
                conclusion,
            } => ClaimStructure::Implies {
                premise: Box::new(self.normalize_structure(premise)),
                conclusion: Box::new(self.normalize_structure(conclusion)),
            },
            ClaimStructure::Quantified {
                quantifier,
                variable,
                domain,
                body,
            } => ClaimStructure::Quantified {
                quantifier: *quantifier,
                variable: self.normalize_text(variable),
                domain: self.normalize_text(domain),
                body: Box::new(self.normalize_structure(body)),
            },
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => ClaimStructure::Temporal {
                event: self.normalize_text(event),
                time_relation: time_relation.clone(),
                reference: self.normalize_text(reference),
            },
            ClaimStructure::Causal {
                cause,
                effect,
                strength,
            } => ClaimStructure::Causal {
                cause: Box::new(self.normalize_structure(cause)),
                effect: Box::new(self.normalize_structure(effect)),
                strength: strength.clone(),
            },
            ClaimStructure::Modal { claim, modality } => ClaimStructure::Modal {
                claim: Box::new(self.normalize_structure(claim)),
                modality: modality.clone(),
            },
            ClaimStructure::Raw(text) => ClaimStructure::Raw(self.normalize_text(text)),
        }
    }
}

impl ClaimNormalizer for DefaultClaimNormalizer {
    fn normalize(&self, claim: &LogicalClaim) -> LogicalClaim {
        LogicalClaim {
            id: claim.id.clone(),
            text: self.normalize_text(&claim.text),
            structure: self.normalize_structure(&claim.structure),
            confidence: claim.confidence,
            source_span: claim.source_span,
        }
    }

    fn normalize_text(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Trim
        result = result.trim().to_string();

        // Lowercase
        if self.lowercase {
            result = result.to_lowercase();
        }

        // Normalize whitespace
        if self.normalize_whitespace {
            result = result.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Remove punctuation (optional)
        if self.remove_punctuation {
            result = result
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect();
        }

        result
    }
}

/// A deduplicator that groups and merges similar claims.
pub struct ClaimDeduplicator {
    /// Normalizer used for comparison.
    normalizer: Box<dyn ClaimNormalizer>,
    /// Whether to merge confidence scores.
    merge_confidence: bool,
}

impl Default for ClaimDeduplicator {
    fn default() -> Self {
        Self::new(Box::new(DefaultClaimNormalizer::default()))
    }
}

impl ClaimDeduplicator {
    /// Create a new claim deduplicator with a normalizer.
    #[must_use]
    pub fn new(normalizer: Box<dyn ClaimNormalizer>) -> Self {
        Self {
            normalizer,
            merge_confidence: true,
        }
    }

    /// Set whether to merge confidence scores.
    #[must_use]
    pub fn with_merge_confidence(mut self, merge: bool) -> Self {
        self.merge_confidence = merge;
        self
    }

    /// Compute a hash for a normalized claim structure.
    fn hash_structure(structure: &ClaimStructure) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::hash_structure_recursive(structure, &mut hasher);
        hasher.finish()
    }

    /// Recursively hash a claim structure.
    fn hash_structure_recursive<H: Hasher>(structure: &ClaimStructure, hasher: &mut H) {
        match structure {
            ClaimStructure::Predicate {
                subject,
                predicate,
                object,
            } => {
                "predicate".hash(hasher);
                subject.hash(hasher);
                predicate.hash(hasher);
                object.hash(hasher);
            }
            ClaimStructure::Comparison {
                left,
                operator,
                right,
            } => {
                "comparison".hash(hasher);
                left.hash(hasher);
                format!("{operator:?}").hash(hasher);
                right.hash(hasher);
            }
            ClaimStructure::And(claims) => {
                "and".hash(hasher);
                for claim in claims {
                    Self::hash_structure_recursive(claim, hasher);
                }
            }
            ClaimStructure::Or(claims) => {
                "or".hash(hasher);
                for claim in claims {
                    Self::hash_structure_recursive(claim, hasher);
                }
            }
            ClaimStructure::Not(inner) => {
                "not".hash(hasher);
                Self::hash_structure_recursive(inner, hasher);
            }
            ClaimStructure::Implies {
                premise,
                conclusion,
            } => {
                "implies".hash(hasher);
                Self::hash_structure_recursive(premise, hasher);
                Self::hash_structure_recursive(conclusion, hasher);
            }
            ClaimStructure::Quantified {
                quantifier,
                variable,
                domain,
                body,
            } => {
                "quantified".hash(hasher);
                format!("{quantifier:?}").hash(hasher);
                variable.hash(hasher);
                domain.hash(hasher);
                Self::hash_structure_recursive(body, hasher);
            }
            ClaimStructure::Temporal {
                event,
                time_relation,
                reference,
            } => {
                "temporal".hash(hasher);
                event.hash(hasher);
                format!("{time_relation:?}").hash(hasher);
                reference.hash(hasher);
            }
            ClaimStructure::Causal {
                cause,
                effect,
                strength,
            } => {
                "causal".hash(hasher);
                Self::hash_structure_recursive(cause, hasher);
                Self::hash_structure_recursive(effect, hasher);
                format!("{strength:?}").hash(hasher);
            }
            ClaimStructure::Modal { claim, modality } => {
                "modal".hash(hasher);
                Self::hash_structure_recursive(claim, hasher);
                format!("{modality:?}").hash(hasher);
            }
            ClaimStructure::Raw(text) => {
                "raw".hash(hasher);
                text.hash(hasher);
            }
        }
    }

    /// Deduplicate a list of claims, merging duplicates.
    ///
    /// Returns a list of unique claims with merged confidence scores.
    #[must_use]
    pub fn deduplicate(&self, claims: Vec<LogicalClaim>) -> Vec<LogicalClaim> {
        let mut groups: HashMap<u64, Vec<LogicalClaim>> = HashMap::new();

        // Group claims by normalized structure hash
        for claim in claims {
            let normalized = self.normalizer.normalize(&claim);
            let hash = Self::hash_structure(&normalized.structure);
            groups.entry(hash).or_default().push(normalized);
        }

        // Merge each group into a single claim
        groups
            .into_values()
            .map(|group| self.merge_group(group))
            .collect()
    }

    /// Merge a group of similar claims into one.
    fn merge_group(&self, mut group: Vec<LogicalClaim>) -> LogicalClaim {
        if group.len() == 1 {
            return group.remove(0);
        }

        // Use the first claim as the base
        let mut merged = group.remove(0);

        if self.merge_confidence {
            // Aggregate confidence: use weighted average
            let total_confidence: f32 =
                merged.confidence + group.iter().map(|c| c.confidence).sum::<f32>();
            #[allow(clippy::cast_precision_loss)]
            let count = (group.len() + 1) as f32;
            merged.confidence = total_confidence / count;

            // Boost confidence slightly for claims that appear multiple times
            merged.confidence = (merged.confidence * 1.1).min(1.0);
        }

        merged
    }

    /// Find groups of similar claims without merging.
    ///
    /// Returns a map from hash to claims that share that hash.
    #[must_use]
    pub fn find_groups(&self, claims: &[LogicalClaim]) -> HashMap<u64, Vec<LogicalClaim>> {
        let mut groups: HashMap<u64, Vec<LogicalClaim>> = HashMap::new();

        for claim in claims {
            let normalized = self.normalizer.normalize(claim);
            let hash = Self::hash_structure(&normalized.structure);
            groups.entry(hash).or_default().push(claim.clone());
        }

        groups
    }
}

#[cfg(test)]
#[allow(clippy::similar_names)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text_lowercase() {
        let normalizer = DefaultClaimNormalizer::new();
        assert_eq!(normalizer.normalize_text("Hello World"), "hello world");
    }

    #[test]
    fn test_normalize_text_whitespace() {
        let normalizer = DefaultClaimNormalizer::new();
        assert_eq!(
            normalizer.normalize_text("  hello   world  "),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_text_combined() {
        let normalizer = DefaultClaimNormalizer::new()
            .with_lowercase(true)
            .with_normalize_whitespace(true);
        assert_eq!(
            normalizer.normalize_text("  Hello   WORLD  "),
            "hello world"
        );
    }

    #[test]
    fn test_normalize_text_no_lowercase() {
        let normalizer = DefaultClaimNormalizer::new().with_lowercase(false);
        assert_eq!(normalizer.normalize_text("Hello World"), "Hello World");
    }

    #[test]
    fn test_normalize_text_remove_punctuation() {
        let normalizer = DefaultClaimNormalizer::new().with_remove_punctuation(true);
        assert_eq!(normalizer.normalize_text("hello, world!"), "hello world");
    }

    #[test]
    fn test_normalize_claim_predicate() {
        let normalizer = DefaultClaimNormalizer::new();
        let claim = LogicalClaim::new(
            "  The SKY is BLUE  ",
            ClaimStructure::Predicate {
                subject: "  SKY  ".to_string(),
                predicate: "IS".to_string(),
                object: Some("BLUE".to_string()),
            },
        );

        let normalized = normalizer.normalize(&claim);
        assert_eq!(normalized.text, "the sky is blue");

        if let ClaimStructure::Predicate {
            subject,
            predicate,
            object,
        } = &normalized.structure
        {
            assert_eq!(subject, "sky");
            assert_eq!(predicate, "is");
            assert_eq!(object.as_deref(), Some("blue"));
        } else {
            panic!("Expected Predicate structure");
        }
    }

    #[test]
    fn test_normalize_claim_raw() {
        let normalizer = DefaultClaimNormalizer::new();
        let claim = LogicalClaim::new("TEST", ClaimStructure::Raw("  HELLO   WORLD  ".to_string()));

        let normalized = normalizer.normalize(&claim);

        if let ClaimStructure::Raw(text) = &normalized.structure {
            assert_eq!(text, "hello world");
        } else {
            panic!("Expected Raw structure");
        }
    }

    #[test]
    fn test_deduplicator_identical_claims() {
        let dedup = ClaimDeduplicator::default();

        let claims = vec![
            LogicalClaim::new(
                "sky is blue",
                ClaimStructure::Raw("sky is blue".to_string()),
            )
            .with_confidence(0.8),
            LogicalClaim::new(
                "SKY IS BLUE",
                ClaimStructure::Raw("SKY IS BLUE".to_string()),
            )
            .with_confidence(0.6),
        ];

        let deduped = dedup.deduplicate(claims);
        assert_eq!(deduped.len(), 1);

        // Confidence should be merged
        assert!(deduped[0].confidence > 0.7);
    }

    #[test]
    fn test_deduplicator_different_claims() {
        let dedup = ClaimDeduplicator::default();

        let claims = vec![
            LogicalClaim::new(
                "sky is blue",
                ClaimStructure::Raw("sky is blue".to_string()),
            ),
            LogicalClaim::new(
                "grass is green",
                ClaimStructure::Raw("grass is green".to_string()),
            ),
        ];

        let deduped = dedup.deduplicate(claims);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_deduplicator_no_merge_confidence() {
        let dedup = ClaimDeduplicator::default().with_merge_confidence(false);

        let claims = vec![
            LogicalClaim::new("test", ClaimStructure::Raw("test".to_string())).with_confidence(0.8),
            LogicalClaim::new("TEST", ClaimStructure::Raw("TEST".to_string())).with_confidence(0.6),
        ];

        let deduped = dedup.deduplicate(claims);
        assert_eq!(deduped.len(), 1);

        // Should use first claim's confidence without merging
        assert!((deduped[0].confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_find_groups() {
        let dedup = ClaimDeduplicator::default();

        let claims = vec![
            LogicalClaim::new("a", ClaimStructure::Raw("test".to_string())),
            LogicalClaim::new("b", ClaimStructure::Raw("TEST".to_string())),
            LogicalClaim::new("c", ClaimStructure::Raw("other".to_string())),
        ];

        let groups = dedup.find_groups(&claims);

        // Should have 2 groups: "test/TEST" and "other"
        assert_eq!(groups.len(), 2);

        // One group should have 2 claims
        assert!(groups.values().any(|g| g.len() == 2));
    }

    #[test]
    fn test_hash_structure_predicate() {
        let s1 = ClaimStructure::Predicate {
            subject: "sky".to_string(),
            predicate: "is".to_string(),
            object: Some("blue".to_string()),
        };
        let s2 = ClaimStructure::Predicate {
            subject: "sky".to_string(),
            predicate: "is".to_string(),
            object: Some("blue".to_string()),
        };
        let s3 = ClaimStructure::Predicate {
            subject: "grass".to_string(),
            predicate: "is".to_string(),
            object: Some("green".to_string()),
        };

        assert_eq!(
            ClaimDeduplicator::hash_structure(&s1),
            ClaimDeduplicator::hash_structure(&s2)
        );
        assert_ne!(
            ClaimDeduplicator::hash_structure(&s1),
            ClaimDeduplicator::hash_structure(&s3)
        );
    }

    #[test]
    fn test_normalize_nested_structure() {
        let normalizer = DefaultClaimNormalizer::new();

        let claim = LogicalClaim::new(
            "test",
            ClaimStructure::And(vec![
                ClaimStructure::Raw("  HELLO  ".to_string()),
                ClaimStructure::Raw("  WORLD  ".to_string()),
            ]),
        );

        let normalized = normalizer.normalize(&claim);

        if let ClaimStructure::And(parts) = &normalized.structure {
            if let ClaimStructure::Raw(text) = &parts[0] {
                assert_eq!(text, "hello");
            }
            if let ClaimStructure::Raw(text) = &parts[1] {
                assert_eq!(text, "world");
            }
        } else {
            panic!("Expected And structure");
        }
    }
}
