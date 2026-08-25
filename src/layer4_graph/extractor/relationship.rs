//! Relationship extraction implementations.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::GraphError;
use crate::layer4_graph::traits::RelationshipExtractor;
use crate::layer4_graph::types::{GraphEntity, GraphRelationship, RelationshipType};
use crate::text;

/// A mock relationship extractor that returns predefined relationships for testing.
#[derive(Debug, Default)]
pub struct MockRelationshipExtractor {
    relationships: Vec<GraphRelationship>,
}

impl MockRelationshipExtractor {
    /// Create a new mock extractor with no predefined relationships.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mock extractor with predefined relationships.
    #[must_use]
    pub fn with_relationships(relationships: Vec<GraphRelationship>) -> Self {
        Self { relationships }
    }

    /// Add a predefined relationship.
    pub fn add_relationship(&mut self, relationship: GraphRelationship) {
        self.relationships.push(relationship);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl RelationshipExtractor for MockRelationshipExtractor {
    async fn extract_relationships(
        &self,
        _text: &str,
        _entities: &[GraphEntity],
    ) -> Result<Vec<GraphRelationship>, GraphError> {
        Ok(self.relationships.clone())
    }

    fn supported_relationship_types(&self) -> Vec<RelationshipType> {
        vec![
            RelationshipType::IsA,
            RelationshipType::PartOf,
            RelationshipType::RelatedTo,
            RelationshipType::Uses,
            RelationshipType::CreatedBy,
        ]
    }
}

/// A pattern-based relationship extractor using keyword patterns.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct PatternRelationshipExtractor {
    /// Patterns that indicate `IS_A` relationships.
    is_a_patterns: Vec<String>,
    /// Patterns that indicate `PART_OF` relationships.
    part_of_patterns: Vec<String>,
    /// Patterns that indicate `USES` relationships.
    uses_patterns: Vec<String>,
    /// Patterns that indicate `CREATED_BY` relationships.
    created_by_patterns: Vec<String>,
    /// Patterns that indicate `LOCATED_IN` relationships.
    located_in_patterns: Vec<String>,
    /// Patterns that indicate `WORKS_FOR` relationships.
    works_for_patterns: Vec<String>,
    /// Patterns that indicate `DEPENDS_ON` relationships.
    depends_on_patterns: Vec<String>,
}

impl Default for PatternRelationshipExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternRelationshipExtractor {
    /// Create a new pattern-based relationship extractor with default patterns.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Both punctuation families in one list. The matcher looks for a pattern in the text
            // BETWEEN two entity mentions, and in Japanese that span is the particles and verb —
            // `東京 は 日本の首都 です` puts `は` between `東京` and `日本`, and the copula after.
            // Neither is a word in the English sense, which is why none of the English patterns
            // could ever fire on Japanese and why a Japanese corpus produced zero relationships.
            is_a_patterns: vec![
                "is a".to_string(),
                "is an".to_string(),
                "are".to_string(),
                "was a".to_string(),
                "were".to_string(),
                "type of".to_string(),
                "kind of".to_string(),
                "instance of".to_string(),
                "とは".to_string(),
                "という".to_string(),
            ],
            part_of_patterns: vec![
                "part of".to_string(),
                "belongs to".to_string(),
                "included in".to_string(),
                "component of".to_string(),
                "member of".to_string(),
                "within".to_string(),
                "の一部".to_string(),
                "に含まれ".to_string(),
                "に属す".to_string(),
            ],
            uses_patterns: vec![
                "uses".to_string(),
                "utilizes".to_string(),
                "employs".to_string(),
                "relies on".to_string(),
                "depends on".to_string(),
                "built with".to_string(),
                "implemented with".to_string(),
                "written in".to_string(),
                "を使".to_string(),
                "を利用".to_string(),
                "で作".to_string(),
                "を用い".to_string(),
            ],
            created_by_patterns: vec![
                "created by".to_string(),
                "developed by".to_string(),
                "built by".to_string(),
                "made by".to_string(),
                "authored by".to_string(),
                "designed by".to_string(),
                "invented by".to_string(),
                "が作".to_string(),
                "が開発".to_string(),
                "によって作".to_string(),
                "が生み出".to_string(),
            ],
            located_in_patterns: vec![
                "located in".to_string(),
                "based in".to_string(),
                "in".to_string(),
                "at".to_string(),
                "from".to_string(),
                "headquartered in".to_string(),
                "にあ".to_string(),
                "に位置".to_string(),
                "で生まれ".to_string(),
                "にまたが".to_string(),
            ],
            works_for_patterns: vec![
                "works for".to_string(),
                "employed by".to_string(),
                "works at".to_string(),
                "founder of".to_string(),
                "CEO of".to_string(),
                "leads".to_string(),
            ],
            depends_on_patterns: vec![
                "depends on".to_string(),
                "requires".to_string(),
                "needs".to_string(),
                "built on".to_string(),
                "based on".to_string(),
            ],
        }
    }

    /// Add custom `IS_A` patterns.
    pub fn add_is_a_patterns(&mut self, patterns: impl IntoIterator<Item = impl Into<String>>) {
        for p in patterns {
            self.is_a_patterns.push(p.into());
        }
    }

    /// Add custom USES patterns.
    pub fn add_uses_patterns(&mut self, patterns: impl IntoIterator<Item = impl Into<String>>) {
        for p in patterns {
            self.uses_patterns.push(p.into());
        }
    }

    /// Find the relationship type and direction between two entities based on text patterns.
    /// True when another entity's name sits between the two given positions in `sentence`.
    ///
    /// Used to require ADJACENCY on Japanese. `カレーはインドで生まれた料理です` mentions three
    /// entities, and the span between `カレー` and `料理` contains `で生まれ`, so a pattern search
    /// over that span reports `カレー LOCATED_IN 料理` — a relationship the sentence does not state.
    /// The verb belongs to the entity actually next to it. English does not need this: its patterns
    /// are words that sit directly between the pair, so an intervening entity does not smuggle one
    /// in, and requiring adjacency there would drop relationships that are really stated.
    fn has_entity_between(
        sentence_lower: &str,
        start: usize,
        end: usize,
        entities: &[GraphEntity],
        source: &GraphEntity,
        target: &GraphEntity,
    ) -> bool {
        if start >= end {
            return false;
        }
        let span = &sentence_lower[start..end];
        entities.iter().any(|other| {
            other.id != source.id
                && other.id != target.id
                && span.contains(&other.name.to_lowercase())
        })
    }

    #[allow(clippy::too_many_lines)]
    fn find_relationship_pattern(
        &self,
        text: &str,
        source: &GraphEntity,
        target: &GraphEntity,
        entities: &[GraphEntity],
    ) -> Option<(RelationshipType, bool)> {
        let lower_text = text.to_lowercase();
        let source_lower = source.name.to_lowercase();
        let target_lower = target.name.to_lowercase();

        // Try to find a sentence or clause containing both entities. Clause separators are added
        // to the sentence enders, in both punctuation families — `、` is the Japanese comma.
        let sentences: Vec<&str> = text::split_sentences(text)
            .into_iter()
            .flat_map(|sentence| sentence.split([';', ',', '；', '、']))
            .collect();

        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();

            // Check if both entities are mentioned in this sentence
            let source_pos = sentence_lower.find(&source_lower);
            let target_pos = sentence_lower.find(&target_lower);

            if let (Some(sp), Some(tp)) = (source_pos, target_pos) {
                // Get the text between the two entities
                let (start, end, forward) = if sp < tp {
                    (sp + source_lower.len(), tp, true)
                } else {
                    (tp + target_lower.len(), sp, false)
                };

                if start < end {
                    if text::has_cjk(sentence)
                        && Self::has_entity_between(
                            &sentence_lower,
                            start,
                            end,
                            entities,
                            source,
                            target,
                        )
                    {
                        continue;
                    }
                    // Japanese is verb-final: `OxiRAG は OxiZ を使います` puts the verb that names
                    // the relationship AFTER both entities, where a between-the-entities search
                    // cannot see it. English puts it in the middle (`OxiRAG uses OxiZ`), which is
                    // the only case the original span covers. So for text with CJK in it, search
                    // the span between the entities AND the tail after the second one.
                    let tail_start = end
                        + if forward {
                            target_lower.len()
                        } else {
                            source_lower.len()
                        };
                    let span = if text::has_cjk(sentence) && tail_start <= sentence_lower.len() {
                        format!(
                            "{}{}",
                            &sentence_lower[start..end],
                            &sentence_lower[tail_start..]
                        )
                    } else {
                        sentence_lower[start..end].to_string()
                    };
                    let between = span.as_str();

                    // Check each pattern type
                    for pattern in &self.is_a_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::IsA, forward));
                        }
                    }

                    for pattern in &self.part_of_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::PartOf, forward));
                        }
                    }

                    for pattern in &self.uses_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::Uses, forward));
                        }
                    }

                    for pattern in &self.created_by_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::CreatedBy, !forward));
                        }
                    }

                    for pattern in &self.located_in_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::LocatedIn, forward));
                        }
                    }

                    for pattern in &self.works_for_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::WorksFor, forward));
                        }
                    }

                    for pattern in &self.depends_on_patterns {
                        if between.contains(pattern) {
                            return Some((RelationshipType::DependsOn, forward));
                        }
                    }
                }

                // If entities are in the same sentence but no clear pattern, mark as related
                if lower_text.contains(&source_lower) && lower_text.contains(&target_lower) {
                    return Some((RelationshipType::RelatedTo, true));
                }
            }
        }

        None
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl RelationshipExtractor for PatternRelationshipExtractor {
    async fn extract_relationships(
        &self,
        text: &str,
        entities: &[GraphEntity],
    ) -> Result<Vec<GraphRelationship>, GraphError> {
        let mut relationships = Vec::new();
        let mut seen: HashMap<(String, String), bool> = HashMap::new();

        // Check all pairs of entities
        for (i, source) in entities.iter().enumerate() {
            for target in entities.iter().skip(i + 1) {
                // Skip self-relationships
                if source.id == target.id {
                    continue;
                }

                // Check if we've already processed this pair
                let pair_key = if source.id < target.id {
                    (source.id.clone(), target.id.clone())
                } else {
                    (target.id.clone(), source.id.clone())
                };

                if seen.contains_key(&pair_key) {
                    continue;
                }
                seen.insert(pair_key, true);

                // Try to find a relationship pattern
                if let Some((rel_type, forward)) =
                    self.find_relationship_pattern(text, source, target, entities)
                {
                    let (src_id, tgt_id) = if forward {
                        (source.id.clone(), target.id.clone())
                    } else {
                        (target.id.clone(), source.id.clone())
                    };

                    relationships.push(
                        GraphRelationship::new(src_id, tgt_id, rel_type).with_confidence(0.6), // Pattern-based has moderate confidence
                    );
                }
            }
        }

        Ok(relationships)
    }

    fn supported_relationship_types(&self) -> Vec<RelationshipType> {
        vec![
            RelationshipType::IsA,
            RelationshipType::PartOf,
            RelationshipType::Uses,
            RelationshipType::CreatedBy,
            RelationshipType::LocatedIn,
            RelationshipType::WorksFor,
            RelationshipType::DependsOn,
            RelationshipType::RelatedTo,
        ]
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::layer4_graph::types::EntityType;

    #[tokio::test]
    async fn test_mock_relationship_extractor() {
        let relationships = vec![GraphRelationship::new(
            "rust",
            "llvm",
            RelationshipType::Uses,
        )];
        let extractor = MockRelationshipExtractor::with_relationships(relationships);

        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm"),
        ];

        let result = extractor
            .extract_relationships("any text", &entities)
            .await
            .expect("test operation should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].relationship_type, RelationshipType::Uses);
    }

    #[tokio::test]
    async fn test_pattern_extractor_is_a() {
        let extractor = PatternRelationshipExtractor::new();
        let text = "Rust is a systems programming language.";

        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("systems programming language", EntityType::Concept)
                .with_id("systems_lang"),
        ];

        let result = extractor
            .extract_relationships(text, &entities)
            .await
            .expect("test operation should succeed");
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_extractor_uses() {
        let extractor = PatternRelationshipExtractor::new();
        let text = "Rust uses LLVM for code generation.";

        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("LLVM", EntityType::Technology).with_id("llvm"),
        ];

        let result = extractor
            .extract_relationships(text, &entities)
            .await
            .expect("test operation should succeed");
        assert!(!result.is_empty());

        // Should find USES relationship
        let uses_rel = result
            .iter()
            .find(|r| r.relationship_type == RelationshipType::Uses);
        assert!(uses_rel.is_some());
    }

    #[tokio::test]
    async fn test_pattern_extractor_created_by() {
        let extractor = PatternRelationshipExtractor::new();
        let text = "Rust was created by Mozilla.";

        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology).with_id("rust"),
            GraphEntity::new("Mozilla", EntityType::Organization).with_id("mozilla"),
        ];

        let result = extractor
            .extract_relationships(text, &entities)
            .await
            .expect("test operation should succeed");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_supported_relationship_types() {
        let extractor = PatternRelationshipExtractor::new();
        let types = extractor.supported_relationship_types();

        assert!(types.contains(&RelationshipType::IsA));
        assert!(types.contains(&RelationshipType::Uses));
        assert!(types.contains(&RelationshipType::CreatedBy));
    }
}
