//! Canonical entity catalog with a case-insensitive surface-form index.

use std::collections::HashMap;

use crate::entity_linking::types::CanonicalEntity;

// ── EntityCatalog ───────────────────────────────────────────────────────────

/// A collection of [`CanonicalEntity`] records with a surface-form lookup index.
///
/// Both primary names and aliases are indexed (lower-cased) so that a mention
/// can be resolved to candidate entities regardless of letter casing. A single
/// surface form may map to several entities, which is the source of ambiguity
/// that the linker resolves via context.
#[derive(Debug, Clone, Default)]
pub struct EntityCatalog {
    /// The canonical entities, in insertion order.
    entities: Vec<CanonicalEntity>,
    /// Lower-cased surface form → indices into `entities`.
    alias_index: HashMap<String, Vec<usize>>,
}

impl EntityCatalog {
    /// Create a new, empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            alias_index: HashMap::new(),
        }
    }

    /// Insert a canonical entity, indexing its name and every alias.
    ///
    /// The same index is never recorded twice for a given surface form, so an
    /// alias that duplicates the name does not create a duplicate candidate.
    pub fn add(&mut self, entity: CanonicalEntity) {
        let idx = self.entities.len();
        let mut keys: Vec<String> = Vec::with_capacity(1 + entity.aliases.len());
        keys.push(entity.name.to_lowercase());
        for alias in &entity.aliases {
            keys.push(alias.to_lowercase());
        }
        for key in keys {
            let bucket = self.alias_index.entry(key).or_default();
            if !bucket.contains(&idx) {
                bucket.push(idx);
            }
        }
        self.entities.push(entity);
    }

    /// Number of canonical entities held by the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Return `true` when the catalog holds no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Return the catalog indices whose name or any alias matches `mention`.
    ///
    /// Matching is case-insensitive and exact on the (trimmed) surface form;
    /// the returned indices are in ascending (insertion) order.
    #[must_use]
    pub fn candidates(&self, mention: &str) -> Vec<usize> {
        let key = mention.trim().to_lowercase();
        self.alias_index.get(&key).cloned().unwrap_or_default()
    }

    /// Borrow the canonical entity stored at `idx`, if any.
    #[must_use]
    pub fn entity(&self, idx: usize) -> Option<&CanonicalEntity> {
        self.entities.get(idx)
    }

    /// Borrow all canonical entities in insertion order.
    #[must_use]
    pub fn entities(&self) -> &[CanonicalEntity] {
        &self.entities
    }
}
