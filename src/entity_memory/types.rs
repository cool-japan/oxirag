//! Types for the `entity_memory` module.
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;
// ── EntityCategory ────────────────────────────────────────────────────────────
/// Coarse category of a tracked entity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum EntityCategory {
    /// A human or fictional person.
    Person,
    /// A company, institution, or group.
    Organization,
    /// A geographic or physical location.
    Location,
    /// An abstract concept or topic.
    #[default]
    Concept,
    /// A product, tool, or artifact.
    Product,
    /// Uncategorised entity.
    Other,
}
impl EntityCategory {
    /// Human-readable label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Concept => "concept",
            Self::Product => "product",
            Self::Other => "other",
        }
    }
}
// ── EntityKnowledge ───────────────────────────────────────────────────────────
/// Accumulated knowledge about a single entity across conversation turns.
#[derive(Debug, Clone)]
pub struct EntityKnowledge {
    /// Canonical entity name.
    pub name: String,
    /// Coarse category.
    pub category: EntityCategory,
    /// Running natural-language summary of what is known.
    pub summary: String,
    /// Individual fact strings extracted from turns.
    pub facts: Vec<String>,
    /// How many times this entity has been mentioned.
    pub mention_count: u32,
    /// When this entity was first seen.
    pub first_seen: DateTime<Utc>,
    /// When this entity was most recently mentioned.
    pub last_seen: DateTime<Utc>,
    /// Combined importance+recency salience score in [0.0, 1.0].
    pub salience: f32,
}
impl EntityKnowledge {
    /// Create with a name and category.
    #[must_use]
    pub fn new(name: impl Into<String>, category: EntityCategory) -> Self {
        let now = Utc::now();
        Self {
            name: name.into(),
            category,
            summary: String::new(),
            facts: Vec::new(),
            mention_count: 0,
            first_seen: now,
            last_seen: now,
            salience: 0.0,
        }
    }
}
// ── EntityMentionSpan ─────────────────────────────────────────────────────────
/// A detected entity mention span within a text.
#[derive(Debug, Clone)]
pub struct EntityMentionSpan {
    /// The surface form of the mention.
    pub text: String,
    /// Inferred category.
    pub category: EntityCategory,
    /// Byte offset of the start of the mention.
    pub start: usize,
    /// Byte offset just past the end of the mention.
    pub end: usize,
}
// ── EntityMentionExtractor ────────────────────────────────────────────────────
/// Synchronous trait for detecting entity mentions in text.
pub trait EntityMentionExtractor {
    /// Extract all entity mention spans from `text`.
    fn extract(&self, text: &str) -> Vec<EntityMentionSpan>;
}
// ── HeuristicEntityMentionExtractor ──────────────────────────────────────────
/// Capitalised n-gram heuristic entity extractor.
#[derive(Debug, Clone, Default)]
pub struct HeuristicEntityMentionExtractor {
    /// Minimum token length to consider as an entity. Defaults to `2`.
    pub min_len: usize,
}
impl HeuristicEntityMentionExtractor {
    /// Create a new extractor.
    #[must_use]
    pub fn new(min_len: usize) -> Self {
        Self { min_len }
    }
}
impl EntityMentionExtractor for HeuristicEntityMentionExtractor {
    fn extract(&self, text: &str) -> Vec<EntityMentionSpan> {
        let min = if self.min_len == 0 { 2 } else { self.min_len };
        let mut spans = Vec::new();
        for word in text.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if clean.len() >= min
                && clean.chars().next().is_some_and(char::is_uppercase)
                && let Some(start) = text.find(word)
            {
                spans.push(EntityMentionSpan {
                    text: clean.clone(),
                    category: EntityCategory::Other,
                    start,
                    end: start + word.len(),
                });
            }
        }
        spans
    }
}
// ── EntityMemoryConfig ────────────────────────────────────────────────────────
/// Configuration for [`EntityMemoryStore`].
#[derive(Debug, Clone)]
pub struct EntityMemoryConfig {
    /// Maximum number of entities to track. Defaults to `128`.
    pub max_entities: usize,
    /// Maximum facts retained per entity. Defaults to `8`.
    pub max_facts_per_entity: usize,
    /// Minimum entity surface-form length. Defaults to `2`.
    pub min_mention_len: usize,
    /// Exponential salience decay per turn without mention. Defaults to `0.9`.
    pub decay: f32,
}
impl Default for EntityMemoryConfig {
    fn default() -> Self {
        Self {
            max_entities: 128,
            max_facts_per_entity: 8,
            min_mention_len: 2,
            decay: 0.9,
        }
    }
}
impl EntityMemoryConfig {
    /// Set the maximum entity count.
    #[must_use]
    pub fn with_max_entities(mut self, v: usize) -> Self {
        self.max_entities = v;
        self
    }
    /// Set the max facts per entity.
    #[must_use]
    pub fn with_max_facts_per_entity(mut self, v: usize) -> Self {
        self.max_facts_per_entity = v;
        self
    }
    /// Set the minimum mention length.
    #[must_use]
    pub fn with_min_mention_len(mut self, v: usize) -> Self {
        self.min_mention_len = v;
        self
    }
    /// Set the salience decay rate.
    #[must_use]
    pub fn with_decay(mut self, v: f32) -> Self {
        self.decay = v;
        self
    }
}
// ── EntityMemoryStore ─────────────────────────────────────────────────────────
/// Per-entity knowledge store that accumulates facts across conversation turns.
#[derive(Debug, Clone)]
pub struct EntityMemoryStore {
    /// Entities indexed by lower-cased canonical name.
    pub entities: HashMap<String, EntityKnowledge>,
    /// Configuration for this store.
    pub config: EntityMemoryConfig,
}
impl EntityMemoryStore {
    /// Create a new, empty store.
    #[must_use]
    pub fn new(config: EntityMemoryConfig) -> Self {
        Self {
            entities: HashMap::new(),
            config,
        }
    }
    /// Observe `text` and upsert any detected entities.
    ///
    /// # Errors
    ///
    /// Returns [`EntityMemoryError::EmptyText`] if `text` is empty.
    pub fn observe(&mut self, text: &str) -> Result<Vec<String>, EntityMemoryError> {
        if text.trim().is_empty() {
            return Err(EntityMemoryError::EmptyText);
        }
        let extractor = HeuristicEntityMentionExtractor::new(self.config.min_mention_len);
        let spans = extractor.extract(text);
        let mut updated = Vec::new();
        for span in &spans {
            let key = span.text.to_lowercase();
            let entry = self
                .entities
                .entry(key.clone())
                .or_insert_with(|| EntityKnowledge::new(&span.text, span.category.clone()));
            entry.mention_count += 1;
            entry.last_seen = Utc::now();
            entry.salience = (entry.salience + 0.2).min(1.0);
            if entry.facts.len() < self.config.max_facts_per_entity {
                let fact = format!("Mentioned in: {}", &text[..text.len().min(80)]);
                if !entry.facts.contains(&fact) {
                    entry.facts.push(fact);
                }
            }
            updated.push(key);
        }
        Ok(updated)
    }
    /// Get knowledge for an entity by name.
    ///
    /// # Errors
    ///
    /// Returns [`EntityMemoryError::EntityNotFound`] if the entity is not tracked.
    pub fn get(&self, name: &str) -> Result<&EntityKnowledge, EntityMemoryError> {
        self.entities
            .get(&name.to_lowercase())
            .ok_or_else(|| EntityMemoryError::EntityNotFound(name.to_string()))
    }
    /// Return the top-`n` entities by salience.
    #[must_use]
    pub fn top_salient(&self, n: usize) -> Vec<&EntityKnowledge> {
        let mut v: Vec<&EntityKnowledge> = self.entities.values().collect();
        v.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v.into_iter().take(n).collect()
    }
    /// Generate a context string for `query` by finding relevant entities.
    #[must_use]
    pub fn context_for(&self, query: &str) -> String {
        let q_lower = query.to_lowercase();
        let relevant: Vec<&EntityKnowledge> = self
            .entities
            .values()
            .filter(|e| q_lower.contains(&e.name.to_lowercase()))
            .collect();
        if relevant.is_empty() {
            return String::new();
        }
        relevant
            .iter()
            .map(|e| {
                format!(
                    "{}: {}",
                    e.name,
                    e.facts.first().cloned().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}
impl Default for EntityMemoryStore {
    fn default() -> Self {
        Self::new(EntityMemoryConfig::default())
    }
}
// ── EntityMemoryError ─────────────────────────────────────────────────────────
/// Errors from the `entity_memory` module.
#[derive(Debug, Error)]
pub enum EntityMemoryError {
    /// The input text was empty.
    #[error("Input text must not be empty")]
    EmptyText,
    /// The requested entity was not found in the store.
    #[error("Entity not found: {0}")]
    EntityNotFound(String),
}
