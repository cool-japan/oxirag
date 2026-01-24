//! Entity extraction implementations.

use async_trait::async_trait;
use std::collections::HashSet;

use crate::error::GraphError;
use crate::layer4_graph::traits::EntityExtractor;
use crate::layer4_graph::types::{EntityType, GraphEntity};

/// A mock entity extractor that returns predefined entities for testing.
#[derive(Debug, Default)]
pub struct MockEntityExtractor {
    entities: Vec<GraphEntity>,
}

impl MockEntityExtractor {
    /// Create a new mock extractor with no predefined entities.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mock extractor with predefined entities.
    #[must_use]
    pub fn with_entities(entities: Vec<GraphEntity>) -> Self {
        Self { entities }
    }

    /// Add a predefined entity.
    pub fn add_entity(&mut self, entity: GraphEntity) {
        self.entities.push(entity);
    }
}

#[async_trait]
impl EntityExtractor for MockEntityExtractor {
    async fn extract_entities(&self, _text: &str) -> Result<Vec<GraphEntity>, GraphError> {
        Ok(self.entities.clone())
    }

    fn supported_entity_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Concept,
            EntityType::Technology,
        ]
    }
}

/// A pattern-based entity extractor that uses keywords and simple heuristics.
#[derive(Debug)]
pub struct PatternEntityExtractor {
    /// Keywords associated with person entities.
    person_keywords: HashSet<String>,
    /// Keywords associated with organization entities.
    org_keywords: HashSet<String>,
    /// Keywords associated with location entities.
    location_keywords: HashSet<String>,
    /// Keywords associated with technology entities.
    tech_keywords: HashSet<String>,
    /// Minimum word length to consider as entity.
    min_word_length: usize,
}

impl Default for PatternEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternEntityExtractor {
    /// Create a new pattern-based entity extractor with default patterns.
    #[must_use]
    pub fn new() -> Self {
        Self {
            person_keywords: Self::default_person_keywords(),
            org_keywords: Self::default_org_keywords(),
            location_keywords: Self::default_location_keywords(),
            tech_keywords: Self::default_tech_keywords(),
            min_word_length: 2,
        }
    }

    fn default_person_keywords() -> HashSet<String> {
        ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "CEO", "CTO", "CFO"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    fn default_org_keywords() -> HashSet<String> {
        [
            "Inc.",
            "Corp.",
            "Ltd.",
            "LLC",
            "Company",
            "Corporation",
            "Foundation",
            "Institute",
            "University",
            "College",
            "Organization",
            "Group",
            "Team",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    fn default_location_keywords() -> HashSet<String> {
        [
            "City",
            "State",
            "Country",
            "Street",
            "Avenue",
            "Road",
            "Boulevard",
            "Park",
            "River",
            "Mountain",
            "Lake",
            "Ocean",
            "Sea",
            "Island",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    fn default_tech_keywords() -> HashSet<String> {
        [
            "Rust",
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "Go",
            "C++",
            "C#",
            "Ruby",
            "Swift",
            "Kotlin",
            "Scala",
            "PHP",
            "SQL",
            "NoSQL",
            "API",
            "SDK",
            "Framework",
            "Library",
            "Database",
            "Server",
            "Cloud",
            "Docker",
            "Kubernetes",
            "Linux",
            "Windows",
            "macOS",
            "iOS",
            "Android",
            "React",
            "Vue",
            "Angular",
            "Node",
            "Django",
            "Flask",
            "Spring",
            "Rails",
            "AWS",
            "Azure",
            "GCP",
            "ML",
            "AI",
            "LLM",
            "GPU",
            "CPU",
            "RAM",
            "SSD",
            "HTTP",
            "HTTPS",
            "TCP",
            "UDP",
            "REST",
            "GraphQL",
            "gRPC",
            "WebSocket",
            "JSON",
            "XML",
            "YAML",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    /// Add custom person keywords.
    pub fn add_person_keywords(&mut self, keywords: impl IntoIterator<Item = impl Into<String>>) {
        for kw in keywords {
            self.person_keywords.insert(kw.into());
        }
    }

    /// Add custom organization keywords.
    pub fn add_org_keywords(&mut self, keywords: impl IntoIterator<Item = impl Into<String>>) {
        for kw in keywords {
            self.org_keywords.insert(kw.into());
        }
    }

    /// Add custom technology keywords.
    pub fn add_tech_keywords(&mut self, keywords: impl IntoIterator<Item = impl Into<String>>) {
        for kw in keywords {
            self.tech_keywords.insert(kw.into());
        }
    }

    /// Set minimum word length for entity consideration.
    pub fn set_min_word_length(&mut self, len: usize) {
        self.min_word_length = len;
    }

    /// Check if a word looks like a capitalized proper noun.
    fn is_capitalized_word(word: &str) -> bool {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.is_empty() {
            return false;
        }
        // Safe: we already checked trimmed.is_empty() above
        trimmed
            .chars()
            .next()
            .is_some_and(|first_char| first_char.is_uppercase() && trimmed.len() > 1)
    }

    /// Determine entity type based on context and keywords.
    fn classify_entity(&self, word: &str, context: &str) -> Option<EntityType> {
        let lower_word = word.to_lowercase();
        let lower_context = context.to_lowercase();

        // Check technology keywords (case-sensitive for most)
        if self.tech_keywords.contains(word) {
            return Some(EntityType::Technology);
        }

        // Check organization context
        for kw in &self.org_keywords {
            if lower_context.contains(&kw.to_lowercase()) {
                return Some(EntityType::Organization);
            }
        }

        // Check location context
        for kw in &self.location_keywords {
            if lower_context.contains(&kw.to_lowercase()) {
                return Some(EntityType::Location);
            }
        }

        // Check person indicators
        for kw in &self.person_keywords {
            if lower_context.contains(&kw.to_lowercase()) {
                return Some(EntityType::Person);
            }
        }

        // Default to Concept for capitalized words
        if Self::is_capitalized_word(word) && !lower_word.is_empty() {
            return Some(EntityType::Concept);
        }

        None
    }

    /// Extract potential entity names from text.
    fn extract_candidate_names(&self, text: &str) -> Vec<(String, String)> {
        let mut candidates = Vec::new();
        let sentences: Vec<&str> = text.split(['.', '!', '?']).collect();

        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let mut i = 0;

            while i < words.len() {
                let word = words[i];
                let cleaned =
                    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');

                if cleaned.len() >= self.min_word_length {
                    // Check for multi-word proper nouns (consecutive capitalized words)
                    if Self::is_capitalized_word(cleaned) {
                        let mut name_parts = vec![cleaned.to_string()];
                        let mut j = i + 1;

                        while j < words.len() {
                            let next_word = words[j];
                            let next_cleaned = next_word.trim_matches(|c: char| {
                                !c.is_alphanumeric() && c != '-' && c != '_'
                            });

                            if Self::is_capitalized_word(next_cleaned) {
                                name_parts.push(next_cleaned.to_string());
                                j += 1;
                            } else {
                                break;
                            }
                        }

                        let full_name = name_parts.join(" ");
                        candidates.push((full_name, sentence.to_string()));
                        i = j;
                        continue;
                    }

                    // Check for technology keywords
                    if self.tech_keywords.contains(cleaned) {
                        candidates.push((cleaned.to_string(), sentence.to_string()));
                    }
                }
                i += 1;
            }
        }

        candidates
    }
}

#[async_trait]
impl EntityExtractor for PatternEntityExtractor {
    async fn extract_entities(&self, text: &str) -> Result<Vec<GraphEntity>, GraphError> {
        let candidates = self.extract_candidate_names(text);
        let mut entities = Vec::new();
        let mut seen_names: HashSet<String> = HashSet::new();

        for (name, context) in candidates {
            let lower_name = name.to_lowercase();
            if seen_names.contains(&lower_name) {
                continue;
            }

            if let Some(entity_type) = self.classify_entity(&name, &context) {
                seen_names.insert(lower_name);
                entities.push(
                    GraphEntity::new(&name, entity_type).with_confidence(0.7), // Pattern-based extraction has moderate confidence
                );
            }
        }

        Ok(entities)
    }

    fn supported_entity_types(&self) -> Vec<EntityType> {
        vec![
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Technology,
            EntityType::Concept,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_entity_extractor() {
        let entities = vec![
            GraphEntity::new("Rust", EntityType::Technology),
            GraphEntity::new("Cargo", EntityType::Technology),
        ];
        let extractor = MockEntityExtractor::with_entities(entities);

        let result = extractor.extract_entities("any text").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "Rust");
    }

    #[tokio::test]
    async fn test_pattern_extractor_tech_keywords() {
        let extractor = PatternEntityExtractor::new();
        let text = "Rust is a systems programming language. It uses LLVM for compilation.";

        let entities = extractor.extract_entities(text).await.unwrap();

        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Rust"));
    }

    #[tokio::test]
    async fn test_pattern_extractor_capitalized_words() {
        let extractor = PatternEntityExtractor::new();
        let text = "The Mozilla Foundation created Firefox browser.";

        let entities = extractor.extract_entities(text).await.unwrap();

        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Mozilla Foundation") || names.contains(&"Firefox"));
    }

    #[tokio::test]
    async fn test_pattern_extractor_deduplication() {
        let extractor = PatternEntityExtractor::new();
        let text = "Rust is great. Rust is fast. RUST is memory-safe.";

        let entities = extractor.extract_entities(text).await.unwrap();

        let rust_count = entities
            .iter()
            .filter(|e| e.name.to_lowercase() == "rust")
            .count();
        assert_eq!(rust_count, 1);
    }

    #[test]
    fn test_supported_entity_types() {
        let extractor = PatternEntityExtractor::new();
        let types = extractor.supported_entity_types();

        assert!(types.contains(&EntityType::Person));
        assert!(types.contains(&EntityType::Technology));
        assert!(types.contains(&EntityType::Organization));
    }
}
