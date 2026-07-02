//! Types for the `lightrag` module.
//!
//! `LightRAG` (Guo et al. 2024, "`LightRAG`: Simple and Fast Retrieval-Augmented
//! Generation") indexes a corpus as a deduplicated **graph + vector hybrid**:
//! [`LightRagEntity`] nodes and [`LightRagRelation`] edges, each carrying an
//! accumulated description, a deterministic pseudo-embedding, and the ids of
//! every source [`LightRagChunk`] that mentioned it. Queries are answered by
//! extracting [`LightRagDualKeywords`] — separate low-level (specific
//! entity/concept) and high-level (thematic) keyword sets — and routing them
//! through one or both of two independent retrieval paths ([`LightRagMode`]).

use thiserror::Error;

// ── LightRagMode ─────────────────────────────────────────────────────────────

/// Which of `LightRAG`'s two retrieval paths a query should use.
///
/// `LightRAG`'s defining trait is that it keeps the low-level (entity) and
/// high-level (theme) retrieval paths **separate** rather than folding them
/// into a single community hierarchy: [`LightRagMode::Local`] walks the
/// entity graph, [`LightRagMode::Global`] walks the relation/theme graph, and
/// [`LightRagMode::Hybrid`] runs both and fuses the results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightRagMode {
    /// Low-level retrieval: match specific entity/concept keywords against
    /// indexed entities, then expand each match to its local neighborhood
    /// (incident relations and neighbor entities).
    Local,
    /// High-level retrieval: match thematic keywords against indexed
    /// relations (which carry relation-level keywords), then gather the
    /// entities those relations connect.
    Global,
    /// Run [`LightRagMode::Local`] and [`LightRagMode::Global`] independently
    /// and fuse the two result sets, deduplicating by key and combining
    /// scores.
    #[default]
    Hybrid,
}

impl LightRagMode {
    /// A short, lowercase label for the mode.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Global => "global",
            Self::Hybrid => "hybrid",
        }
    }

    /// Return `true` when the local (low-level) retrieval path runs under
    /// this mode.
    #[must_use]
    pub fn uses_local(self) -> bool {
        matches!(self, Self::Local | Self::Hybrid)
    }

    /// Return `true` when the global (high-level) retrieval path runs under
    /// this mode.
    #[must_use]
    pub fn uses_global(self) -> bool {
        matches!(self, Self::Global | Self::Hybrid)
    }
}

// ── LightRagEntityKind ───────────────────────────────────────────────────────

/// A deterministic, heuristic classification of an extracted entity.
///
/// Inferred from surface-form signals only (capitalization pattern, a small
/// gazetteer of organizational/geographic suffixes, and year-like tokens) —
/// no model is invoked. [`LightRagEntityKind::Concept`] and
/// [`LightRagEntityKind::Other`] are the fallback buckets when no stronger
/// signal fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightRagEntityKind {
    /// Two-capitalized-word span with no organization/location signal (e.g.
    /// "Marie Curie").
    Person,
    /// Span containing an organizational gazetteer term (e.g. "Acme Corp").
    Organization,
    /// Span containing a geographic gazetteer term, or immediately preceded
    /// by a locative preposition ("in", "at", "from").
    Location,
    /// A four-digit year, or a span containing a month name.
    Date,
    /// A single-token capitalized span (or all-caps acronym) with no other
    /// signal — treated as a named concept or topic.
    Concept,
    /// A multi-token span with no other signal.
    Other,
}

impl LightRagEntityKind {
    /// A short, lowercase label for the kind.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Date => "date",
            Self::Concept => "concept",
            Self::Other => "other",
        }
    }
}

// ── LightRagEntity ───────────────────────────────────────────────────────────

/// A node in the `LightRAG` entity graph.
///
/// Entities are deduplicated by [`LightRagEntity::key`] (a normalized form of
/// the canonical name): re-extracting the same entity from a different
/// [`LightRagChunk`] merges into the existing record rather than creating a
/// duplicate — `description` gains the new sentence context (if not already
/// present verbatim), `source_chunks` gains the new chunk id (if not already
/// present), and `occurrence_count` increments.
#[derive(Debug, Clone, PartialEq)]
pub struct LightRagEntity {
    /// Normalized, dedup-stable key (lowercased, punctuation-stripped,
    /// whitespace-collapsed form of the canonical name).
    pub key: String,
    /// Human-readable display name — the original casing of the first
    /// occurrence.
    pub name: String,
    /// Heuristically inferred entity kind.
    pub kind: LightRagEntityKind,
    /// Accumulated description: the distinct sentence contexts the entity
    /// was mentioned in, across every merged occurrence, joined by `" "`.
    pub description: String,
    /// Ids of every [`LightRagChunk`] that mentioned this entity, in
    /// first-seen order, deduplicated.
    pub source_chunks: Vec<String>,
    /// Number of chunk-insertions that mentioned (and thus merged into) this
    /// entity, including the one that created it.
    pub occurrence_count: usize,
}

impl LightRagEntity {
    /// Create a new entity from its first occurrence.
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        name: impl Into<String>,
        kind: LightRagEntityKind,
        description: impl Into<String>,
        source_chunk: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            name: name.into(),
            kind,
            description: description.into(),
            source_chunks: vec![source_chunk.into()],
            occurrence_count: 1,
        }
    }
}

// ── LightRagRelation ─────────────────────────────────────────────────────────

/// An edge in the `LightRAG` relation graph, connecting two
/// [`LightRagEntity`] records by their [`LightRagEntity::key`].
///
/// Relations are dedup-keyed by their **unordered** endpoint pair — a
/// relation re-extracted with `src`/`dst` swapped (e.g. found again in a
/// later chunk phrased the other way round) merges into the same record, so
/// the graph behaves as an undirected co-occurrence graph for neighborhood
/// traversal, while `src`/`dst` themselves retain the orientation of the
/// first occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct LightRagRelation {
    /// Key of the source entity, as oriented on first extraction.
    pub src: String,
    /// Key of the destination entity, as oriented on first extraction.
    pub dst: String,
    /// Accumulated description: the distinct sentence contexts this relation
    /// was mentioned in, across every merged occurrence, joined by `" "`.
    pub description: String,
    /// Relation-level keywords (thematic terms drawn from the sentences this
    /// relation was extracted from), deduplicated. Matched against
    /// high-level query keywords during [`LightRagMode::Global`] retrieval.
    pub keywords: Vec<String>,
    /// Ids of every [`LightRagChunk`] that mentioned this relation, in
    /// first-seen order, deduplicated.
    pub source_chunks: Vec<String>,
    /// Number of chunk-insertions that mentioned (and thus merged into) this
    /// relation, including the one that created it.
    pub occurrence_count: usize,
}

impl LightRagRelation {
    /// Create a new relation from its first occurrence.
    #[must_use]
    pub fn new(
        src: impl Into<String>,
        dst: impl Into<String>,
        description: impl Into<String>,
        keywords: Vec<String>,
        source_chunk: impl Into<String>,
    ) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            description: description.into(),
            keywords,
            source_chunks: vec![source_chunk.into()],
            occurrence_count: 1,
        }
    }

    /// The dedup key for this relation: its endpoint keys sorted so that
    /// orientation does not affect identity.
    #[must_use]
    pub fn dedup_key(&self) -> (String, String) {
        relation_dedup_key(&self.src, &self.dst)
    }

    /// Return the key of the endpoint that is not `from`, or `None` when
    /// `from` is neither endpoint.
    #[must_use]
    pub fn other_end(&self, from: &str) -> Option<&str> {
        if self.src == from {
            Some(self.dst.as_str())
        } else if self.dst == from {
            Some(self.src.as_str())
        } else {
            None
        }
    }
}

/// Compute the orientation-independent dedup key for a `(src, dst)` pair.
#[must_use]
pub fn relation_dedup_key(src: &str, dst: &str) -> (String, String) {
    if src <= dst {
        (src.to_string(), dst.to_string())
    } else {
        (dst.to_string(), src.to_string())
    }
}

// ── LightRagChunk ────────────────────────────────────────────────────────────

/// A single unit of source text supplied to [`crate::lightrag::LightRagIndex`]
/// for indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightRagChunk {
    /// Caller-assigned, unique chunk identifier.
    pub id: String,
    /// The chunk's raw text.
    pub text: String,
}

impl LightRagChunk {
    /// Create a new chunk.
    #[must_use]
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

// ── LightRagDualKeywords ─────────────────────────────────────────────────────

/// The result of extracting dual-level keywords from a query.
///
/// `LightRAG` splits a query's vocabulary into two disjoint sets: `low_level`
/// keywords name specific entities or concrete nouns (driving
/// [`LightRagMode::Local`] retrieval), while `high_level` keywords name
/// broad themes or concepts (driving [`LightRagMode::Global`] retrieval).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LightRagDualKeywords {
    /// Specific entity / concrete-noun keywords, lowercased.
    pub low_level: Vec<String>,
    /// Abstract theme / concept keywords, lowercased.
    pub high_level: Vec<String>,
}

impl LightRagDualKeywords {
    /// Create a new keyword set.
    #[must_use]
    pub fn new(low_level: Vec<String>, high_level: Vec<String>) -> Self {
        Self {
            low_level,
            high_level,
        }
    }

    /// Return `true` when both keyword lists are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.low_level.is_empty() && self.high_level.is_empty()
    }

    /// Total number of keywords across both levels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.low_level.len() + self.high_level.len()
    }
}

// ── LightRagResult ───────────────────────────────────────────────────────────

/// The complete output of a [`crate::lightrag::LightRagEngine::query`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct LightRagResult {
    /// The original query text.
    pub query: String,
    /// The retrieval mode used to produce this result.
    pub mode: LightRagMode,
    /// The dual-level keywords extracted from the query.
    pub keywords: LightRagDualKeywords,
    /// Retrieved entities, best-first.
    pub entities: Vec<LightRagEntity>,
    /// Retrieved relations, best-first.
    pub relations: Vec<LightRagRelation>,
    /// The union of source chunks referenced by every retrieved entity and
    /// relation, deduplicated.
    pub source_chunks: Vec<LightRagChunk>,
    /// A synthesized natural-language context string built from the
    /// retrieved entities and relations.
    pub context: String,
}

impl LightRagResult {
    /// Return `true` when nothing was retrieved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.relations.is_empty()
    }

    /// The keys of every retrieved entity, in result order.
    #[must_use]
    pub fn entity_keys(&self) -> Vec<&str> {
        self.entities.iter().map(|e| e.key.as_str()).collect()
    }
}

// ── LightRagIndexStats ───────────────────────────────────────────────────────

/// The delta produced by a single
/// [`crate::lightrag::LightRagIndex::insert_chunk`] call — how many entities
/// and relations were newly created versus merged into existing records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LightRagIndexStats {
    /// Number of entities created for the first time by this insertion.
    pub entities_added: usize,
    /// Number of entity mentions that merged into an already-existing entity.
    pub entities_merged: usize,
    /// Number of relations created for the first time by this insertion.
    pub relations_added: usize,
    /// Number of relation mentions that merged into an already-existing
    /// relation.
    pub relations_merged: usize,
}

impl LightRagIndexStats {
    /// Total number of entity mentions processed (added + merged).
    #[must_use]
    pub fn total_entity_mentions(&self) -> usize {
        self.entities_added + self.entities_merged
    }

    /// Total number of relation mentions processed (added + merged).
    #[must_use]
    pub fn total_relation_mentions(&self) -> usize {
        self.relations_added + self.relations_merged
    }
}

// ── LightRagConfig ───────────────────────────────────────────────────────────

/// Configuration for [`crate::lightrag::LightRagIndex`] and
/// [`crate::lightrag::LightRagEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct LightRagConfig {
    /// Number of top-scoring entities kept by the local (low-level)
    /// retrieval path before neighborhood expansion. Defaults to `5`.
    pub top_k_entities: usize,
    /// Number of top-scoring relations kept by the global (high-level)
    /// retrieval path. Defaults to `5`.
    pub top_k_relations: usize,
    /// The retrieval mode used by
    /// [`crate::lightrag::LightRagEngine::query_default`]. Defaults to
    /// [`LightRagMode::Hybrid`].
    pub default_mode: LightRagMode,
    /// Dimension of the deterministic lexical pseudo-embedding. Defaults to
    /// `128`.
    pub embedding_dim: usize,
    /// Whether incremental indexing deduplicates (merges) re-extracted
    /// entities/relations. When `false`, every occurrence becomes a distinct
    /// record. Defaults to `true`.
    pub dedup_enabled: bool,
    /// Number of hops expanded around each local-mode seed entity to build
    /// its neighborhood. Defaults to `1`.
    pub hop_depth: usize,
}

impl Default for LightRagConfig {
    fn default() -> Self {
        Self {
            top_k_entities: 5,
            top_k_relations: 5,
            default_mode: LightRagMode::Hybrid,
            embedding_dim: 128,
            dedup_enabled: true,
            hop_depth: 1,
        }
    }
}

impl LightRagConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of top-scoring entities kept by local retrieval.
    #[must_use]
    pub fn with_top_k_entities(mut self, top_k_entities: usize) -> Self {
        self.top_k_entities = top_k_entities;
        self
    }

    /// Set the number of top-scoring relations kept by global retrieval.
    #[must_use]
    pub fn with_top_k_relations(mut self, top_k_relations: usize) -> Self {
        self.top_k_relations = top_k_relations;
        self
    }

    /// Set the default retrieval mode.
    #[must_use]
    pub fn with_default_mode(mut self, default_mode: LightRagMode) -> Self {
        self.default_mode = default_mode;
        self
    }

    /// Set the pseudo-embedding dimension.
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set whether incremental indexing deduplicates.
    #[must_use]
    pub fn with_dedup_enabled(mut self, dedup_enabled: bool) -> Self {
        self.dedup_enabled = dedup_enabled;
        self
    }

    /// Set the neighborhood hop depth for local retrieval.
    #[must_use]
    pub fn with_hop_depth(mut self, hop_depth: usize) -> Self {
        self.hop_depth = hop_depth;
        self
    }
}

// ── LightRagError ────────────────────────────────────────────────────────────

/// Errors from the `lightrag` module.
#[derive(Debug, Error)]
pub enum LightRagError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// A chunk's text was empty or contained only whitespace.
    #[error("chunk text must not be empty")]
    EmptyChunkText,
    /// A chunk's id was empty or contained only whitespace.
    #[error("chunk id must not be empty")]
    EmptyChunkId,
    /// A chunk was inserted whose id already exists in the index.
    #[error("duplicate chunk id: {0}")]
    DuplicateChunkId(String),
    /// A query was attempted against an index with no entities.
    #[error("index is empty; insert chunks before querying")]
    EmptyIndex,
}
