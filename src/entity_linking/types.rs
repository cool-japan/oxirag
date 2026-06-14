//! Types for the `entity_linking` module.

use thiserror::Error;

// ── CanonicalEntity ─────────────────────────────────────────────────────────

/// A canonical entry in an entity catalog.
///
/// Each canonical entity has a stable `id`, a primary display `name`, an
/// optional set of surface-form `aliases` that also resolve to it, and a free
/// text `description` used for context-based disambiguation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalEntity {
    /// Stable, catalog-unique identifier (e.g. a knowledge-base key).
    pub id: String,
    /// Primary, human-readable display name.
    pub name: String,
    /// Alternative surface forms that also resolve to this entity.
    pub aliases: Vec<String>,
    /// Free-text gloss describing the entity, used for disambiguation.
    pub description: String,
}

impl CanonicalEntity {
    /// Create a canonical entity from an `id` and a primary `name`.
    ///
    /// The alias list starts empty and the description starts blank; use
    /// [`CanonicalEntity::with_alias`], [`CanonicalEntity::with_aliases`], and
    /// [`CanonicalEntity::with_description`] to populate them.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            aliases: Vec::new(),
            description: String::new(),
        }
    }

    /// Add a single alias.
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Add several aliases at once.
    #[must_use]
    pub fn with_aliases<I, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.aliases.extend(aliases.into_iter().map(Into::into));
        self
    }

    /// Set the free-text description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

// ── EntityMention ───────────────────────────────────────────────────────────

/// A surface-form mention detected within a span of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityMention {
    /// The mention's surface text exactly as it appears in the source.
    pub text: String,
    /// Byte offset of the start of the mention within the source text.
    pub start: usize,
    /// Byte offset just past the end of the mention within the source text.
    pub end: usize,
}

impl EntityMention {
    /// Create a mention from its surface `text` and byte span.
    #[must_use]
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

// ── LinkedEntity ────────────────────────────────────────────────────────────

/// The outcome of linking a single mention to the catalog.
///
/// When `entity_id` is `None`, the mention was left unlinked (NIL) because no
/// candidate matched or because the best candidate's confidence fell below the
/// configured floor.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkedEntity {
    /// The surface form that was linked (or attempted).
    pub mention: String,
    /// The resolved canonical entity id, or `None` for an unlinked (NIL) result.
    pub entity_id: Option<String>,
    /// Confidence of the link in `[0.0, 1.0]`; `0.0` for a NIL result.
    pub confidence: f32,
}

impl LinkedEntity {
    /// Construct a successfully linked result.
    #[must_use]
    pub fn linked(
        mention: impl Into<String>,
        entity_id: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            mention: mention.into(),
            entity_id: Some(entity_id.into()),
            confidence,
        }
    }

    /// Construct an unlinked (NIL) result for `mention`.
    #[must_use]
    pub fn nil(mention: impl Into<String>) -> Self {
        Self {
            mention: mention.into(),
            entity_id: None,
            confidence: 0.0,
        }
    }

    /// Return `true` when the mention was resolved to a canonical entity.
    #[must_use]
    pub fn is_linked(&self) -> bool {
        self.entity_id.is_some()
    }
}

// ── EntityLinkConfig ────────────────────────────────────────────────────────

/// Configuration governing linking behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityLinkConfig {
    /// Minimum confidence required to accept a link; below this, the mention is
    /// left unlinked (NIL). Defaults to `0.0`.
    pub min_confidence: f32,
    /// Dimension of the lexical pseudo-embedding used for context scoring.
    /// Defaults to `128`.
    pub dim: usize,
}

impl Default for EntityLinkConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.0,
            dim: 128,
        }
    }
}

impl EntityLinkConfig {
    /// Create a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum confidence floor for accepting a link.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set the embedding dimension used for context scoring.
    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }
}

// ── EntityLinkError ─────────────────────────────────────────────────────────

/// Errors from the `entity_linking` module.
#[derive(Debug, Error)]
pub enum EntityLinkError {
    /// Linking was attempted against a catalog with no entities.
    #[error("catalog is empty")]
    EmptyCatalog,
}
