//! Types, configuration, knowledge-graph container, and error enum for the
//! `think_on_graph` module.
//!
//! Think-on-Graph (`ToG`, Sun et al. 2023) reasons by *beam search over
//! knowledge-graph paths*. This file defines the plain-data building blocks the
//! [`TogEngine`](crate::think_on_graph::TogEngine) manipulates:
//!
//! - [`TogEntity`] / [`TogRelation`] / [`TogTriple`] — the head-relation-tail
//!   graph vocabulary.
//! - [`TogKnowledgeGraph`] — an entity + triple container with pre-built
//!   forward/backward adjacency indices, so the engine can enumerate a node's
//!   incident relations in one lookup.
//! - [`TogBeamPath`] — a single reasoning path: a topic entity followed by an
//!   ordered sequence of hops, carrying a cumulative relevance score.
//! - [`TogDecision`] / [`TogExploration`] — the per-depth reasoning /
//!   sufficiency verdict and its accompanying introspection record.
//! - [`TogResult`] — the synthesized outcome of a full run.
//! - [`TogConfig`] — beam width, depth bound, pruning fan-outs, and the
//!   sufficiency threshold.
//! - [`TogError`] — the fallible surface, surfaced as `Result<_, TogError>`.

use std::collections::HashMap;

use thiserror::Error;

// ── TogError ────────────────────────────────────────────────────────────────

/// Errors returned by the `think_on_graph` module.
#[derive(Debug, Error)]
pub enum TogError {
    /// The query was empty or contained only whitespace.
    #[error("query must not be empty")]
    EmptyQuery,
    /// The supplied [`TogKnowledgeGraph`] contained no entities, so no topic
    /// entity could ever be matched.
    #[error("knowledge graph is empty (no entities)")]
    EmptyGraph,
    /// No entity in the graph matched any term of the query, so beam search has
    /// no topic entity to seed from.
    #[error("no topic entity in the knowledge graph matched the query")]
    NoTopicEntity,
    /// A triple referenced an entity id that is absent from the graph.
    #[error("triple references unknown entity id `{id}`")]
    UnknownEntity {
        /// The offending entity id.
        id: String,
    },
    /// The configuration was invalid (a zero bound, or an out-of-range weight).
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// A human-readable description of what was invalid.
        reason: String,
    },
}

// ── TogConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`TogEngine`](crate::think_on_graph::TogEngine)'s bounded
/// beam search.
///
/// The two central knobs are [`beam_width`](Self::beam_width) `W` — how many
/// reasoning paths survive each pruning step — and
/// [`max_depth`](Self::max_depth) `D` — how many hops the search may follow.
/// The per-step fan-outs
/// [`relations_per_expansion`](Self::relations_per_expansion) and
/// [`entities_per_relation`](Self::entities_per_relation) implement the
/// relation- and entity-pruning stages, while
/// [`sufficiency_threshold`](Self::sufficiency_threshold) governs the
/// early-stop reasoning check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TogConfig {
    /// Beam width `W`: the maximum number of reasoning paths retained after
    /// each pruning step. Defaults to `3`.
    pub beam_width: usize,
    /// Maximum search depth `D`: the greatest number of hops any reasoning path
    /// may contain. Defaults to `3`.
    pub max_depth: usize,
    /// The maximum number of distinct relations kept per path during relation
    /// exploration (the relation-pruning fan-out). Defaults to `3`.
    pub relations_per_expansion: usize,
    /// The maximum number of neighbour entities followed per kept relation
    /// during entity exploration. Defaults to `3`.
    pub entities_per_relation: usize,
    /// Dimension of the deterministic FNV-1a lexical pseudo-embedding used for
    /// cosine relevance scoring. Defaults to `64`.
    pub embedding_dim: usize,
    /// The fraction (in `[0.0, 1.0]`) of the query's content terms that the
    /// beam paths' triples must collectively cover for the reasoning /
    /// sufficiency check to stop the search early. Defaults to `0.6`.
    pub sufficiency_threshold: f32,
    /// Blend weight (in `[0.0, 1.0]`) of the lexical-overlap component against
    /// the cosine component of every relevance score: `weight * lexical +
    /// (1 - weight) * cosine`. Defaults to `0.5`.
    pub lexical_weight: f32,
    /// The minimum relevance a relation must score to survive relation pruning;
    /// relations scoring strictly below this are dropped even if there is beam
    /// budget for them. Defaults to `0.0` (top-k pruning only).
    pub min_relation_score: f32,
}

impl Default for TogConfig {
    fn default() -> Self {
        Self {
            beam_width: 3,
            max_depth: 3,
            relations_per_expansion: 3,
            entities_per_relation: 3,
            embedding_dim: 64,
            sufficiency_threshold: 0.6,
            lexical_weight: 0.5,
            min_relation_score: 0.0,
        }
    }
}

impl TogConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the beam width `W`.
    #[must_use]
    pub fn with_beam_width(mut self, beam_width: usize) -> Self {
        self.beam_width = beam_width;
        self
    }

    /// Set the maximum search depth `D`.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the relation-pruning fan-out.
    #[must_use]
    pub fn with_relations_per_expansion(mut self, relations_per_expansion: usize) -> Self {
        self.relations_per_expansion = relations_per_expansion;
        self
    }

    /// Set the entity-pruning fan-out.
    #[must_use]
    pub fn with_entities_per_relation(mut self, entities_per_relation: usize) -> Self {
        self.entities_per_relation = entities_per_relation;
        self
    }

    /// Set the pseudo-embedding dimension.
    #[must_use]
    pub fn with_embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.embedding_dim = embedding_dim;
        self
    }

    /// Set the sufficiency threshold.
    #[must_use]
    pub fn with_sufficiency_threshold(mut self, sufficiency_threshold: f32) -> Self {
        self.sufficiency_threshold = sufficiency_threshold;
        self
    }

    /// Set the lexical-vs-cosine blend weight.
    #[must_use]
    pub fn with_lexical_weight(mut self, lexical_weight: f32) -> Self {
        self.lexical_weight = lexical_weight;
        self
    }

    /// Set the minimum relation-relevance pruning floor.
    #[must_use]
    pub fn with_min_relation_score(mut self, min_relation_score: f32) -> Self {
        self.min_relation_score = min_relation_score;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TogError::InvalidConfig`] when any bound is zero, when the
    /// embedding dimension is zero, or when a weight or threshold is outside
    /// its permitted range (or non-finite).
    pub fn validate(&self) -> Result<(), TogError> {
        if self.beam_width == 0 {
            return Err(TogError::InvalidConfig {
                reason: "beam_width must be >= 1".to_string(),
            });
        }
        if self.max_depth == 0 {
            return Err(TogError::InvalidConfig {
                reason: "max_depth must be >= 1".to_string(),
            });
        }
        if self.relations_per_expansion == 0 {
            return Err(TogError::InvalidConfig {
                reason: "relations_per_expansion must be >= 1".to_string(),
            });
        }
        if self.entities_per_relation == 0 {
            return Err(TogError::InvalidConfig {
                reason: "entities_per_relation must be >= 1".to_string(),
            });
        }
        if self.embedding_dim == 0 {
            return Err(TogError::InvalidConfig {
                reason: "embedding_dim must be >= 1".to_string(),
            });
        }
        if !self.sufficiency_threshold.is_finite() {
            return Err(TogError::InvalidConfig {
                reason: "sufficiency_threshold must be finite".to_string(),
            });
        }
        if !self.lexical_weight.is_finite() || !(0.0..=1.0).contains(&self.lexical_weight) {
            return Err(TogError::InvalidConfig {
                reason: "lexical_weight must be within [0.0, 1.0]".to_string(),
            });
        }
        if !self.min_relation_score.is_finite() {
            return Err(TogError::InvalidConfig {
                reason: "min_relation_score must be finite".to_string(),
            });
        }
        Ok(())
    }
}

// ── TogEntity ───────────────────────────────────────────────────────────────

/// A knowledge-graph entity: a stable `id` and a human-readable `name`.
///
/// The `id` is what [`TogTriple`]s reference and what the graph indices key on;
/// the `name` is the surface text matched against the query and used in
/// relevance scoring and answer synthesis. The two may coincide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TogEntity {
    /// Stable, unique identifier of the entity within its graph.
    pub id: String,
    /// Human-readable surface name of the entity.
    pub name: String,
}

impl TogEntity {
    /// Create a new entity from an `id` and a `name`.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
        }
    }
}

// ── TogRelation ─────────────────────────────────────────────────────────────

/// A typed relation label (the edge kind of a [`TogTriple`]).
///
/// Used during relation exploration as the scored unit: the engine ranks a
/// node's incident relation *labels* by query relevance and keeps only the top
/// few. [`TogExploration::scored_relations`] records these labels with their
/// scores for introspection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TogRelation {
    /// The relation label text (for example `"capital_of"`).
    pub label: String,
}

impl TogRelation {
    /// Create a new relation from its `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    /// Return the relation label as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.label
    }

    /// Return the relation label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

// ── TogTriple ───────────────────────────────────────────────────────────────

/// A directed knowledge-graph edge: `head -[relation]-> tail`.
///
/// `head` and `tail` are entity ids (they must reference entities present in
/// the [`TogKnowledgeGraph`]); `relation` is the edge's label.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TogTriple {
    /// Entity id of the head (subject) of the edge.
    pub head: String,
    /// Relation label of the edge.
    pub relation: String,
    /// Entity id of the tail (object) of the edge.
    pub tail: String,
}

impl TogTriple {
    /// Create a new triple `head -[relation]-> tail`.
    #[must_use]
    pub fn new(
        head: impl Into<String>,
        relation: impl Into<String>,
        tail: impl Into<String>,
    ) -> Self {
        Self {
            head: head.into(),
            relation: relation.into(),
            tail: tail.into(),
        }
    }
}

// ── TogKnowledgeGraph ───────────────────────────────────────────────────────

/// An entity + triple knowledge graph with pre-built adjacency indices.
///
/// The container maintains three indices alongside the raw entity and triple
/// vectors: an id → entity lookup, and forward (`head`) and backward (`tail`)
/// triple-index maps. Together they let
/// [`TogEngine`](crate::think_on_graph::TogEngine) enumerate every relation
/// incident to a node — in both directions — with a single hashed lookup, which
/// is what makes bounded beam expansion cheap.
///
/// Build a graph incrementally with [`TogKnowledgeGraph::add_entity`] and
/// [`TogKnowledgeGraph::add_triple`], or all at once with
/// [`TogKnowledgeGraph::from_parts`].
#[derive(Debug, Clone, Default)]
pub struct TogKnowledgeGraph {
    /// Entities in insertion order.
    entities: Vec<TogEntity>,
    /// Triples in insertion order.
    triples: Vec<TogTriple>,
    /// Map from entity id to its index in `entities`.
    entity_index: HashMap<String, usize>,
    /// Map from entity id to the indices of triples whose `head` is that id.
    outgoing: HashMap<String, Vec<usize>>,
    /// Map from entity id to the indices of triples whose `tail` is that id.
    incoming: HashMap<String, Vec<usize>>,
}

impl TogKnowledgeGraph {
    /// Create a new, empty knowledge graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a graph from a set of `entities` and `triples` in one call.
    ///
    /// # Errors
    ///
    /// Returns [`TogError::UnknownEntity`] when any triple references a `head`
    /// or `tail` id that is not among `entities`.
    pub fn from_parts(entities: Vec<TogEntity>, triples: Vec<TogTriple>) -> Result<Self, TogError> {
        let mut graph = Self::new();
        for entity in entities {
            graph.add_entity(entity);
        }
        for triple in triples {
            graph.add_triple(triple)?;
        }
        Ok(graph)
    }

    /// Insert an entity.
    ///
    /// If an entity with the same id already exists, its name is updated in
    /// place and no duplicate is created.
    pub fn add_entity(&mut self, entity: TogEntity) {
        if let Some(&idx) = self.entity_index.get(&entity.id) {
            if let Some(existing) = self.entities.get_mut(idx) {
                existing.name = entity.name;
            }
            return;
        }
        let idx = self.entities.len();
        self.entity_index.insert(entity.id.clone(), idx);
        self.entities.push(entity);
    }

    /// Insert a triple, updating the forward/backward adjacency indices.
    ///
    /// # Errors
    ///
    /// Returns [`TogError::UnknownEntity`] when the triple's `head` or `tail`
    /// id is not present in the graph. Add the entities first.
    pub fn add_triple(&mut self, triple: TogTriple) -> Result<(), TogError> {
        if !self.entity_index.contains_key(&triple.head) {
            return Err(TogError::UnknownEntity {
                id: triple.head.clone(),
            });
        }
        if !self.entity_index.contains_key(&triple.tail) {
            return Err(TogError::UnknownEntity {
                id: triple.tail.clone(),
            });
        }
        let idx = self.triples.len();
        self.outgoing
            .entry(triple.head.clone())
            .or_default()
            .push(idx);
        self.incoming
            .entry(triple.tail.clone())
            .or_default()
            .push(idx);
        self.triples.push(triple);
        Ok(())
    }

    /// All entities, in insertion order.
    #[must_use]
    pub fn entities(&self) -> &[TogEntity] {
        &self.entities
    }

    /// All triples, in insertion order.
    #[must_use]
    pub fn triples(&self) -> &[TogTriple] {
        &self.triples
    }

    /// Number of entities in the graph.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of triples in the graph.
    #[must_use]
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` when the graph holds no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Look up an entity by id.
    #[must_use]
    pub fn entity(&self, id: &str) -> Option<&TogEntity> {
        self.entity_index
            .get(id)
            .and_then(|&idx| self.entities.get(idx))
    }

    /// Look up an entity's name by id.
    #[must_use]
    pub fn entity_name(&self, id: &str) -> Option<&str> {
        self.entity(id).map(|e| e.name.as_str())
    }

    /// Return `true` when an entity with `id` exists.
    #[must_use]
    pub fn contains_entity(&self, id: &str) -> bool {
        self.entity_index.contains_key(id)
    }

    /// Triples whose `head` is `id` (out-edges from the entity).
    #[must_use]
    pub fn outgoing_triples(&self, id: &str) -> Vec<&TogTriple> {
        self.outgoing.get(id).map_or_else(Vec::new, |indices| {
            indices
                .iter()
                .filter_map(|&idx| self.triples.get(idx))
                .collect()
        })
    }

    /// Triples whose `tail` is `id` (in-edges to the entity).
    #[must_use]
    pub fn incoming_triples(&self, id: &str) -> Vec<&TogTriple> {
        self.incoming.get(id).map_or_else(Vec::new, |indices| {
            indices
                .iter()
                .filter_map(|&idx| self.triples.get(idx))
                .collect()
        })
    }
}

// ── TogBeamPath ─────────────────────────────────────────────────────────────

/// A single reasoning path explored by beam search.
///
/// A path begins as a bare topic entity (a *seed*, [`TogBeamPath::seed`]) and
/// grows one hop at a time via [`TogBeamPath::extended`]. `visited_ids` records
/// the entity ids on the path in order — the topic first, then each neighbour
/// stepped into — so cycles can be avoided and the path's frontier
/// ([`TogBeamPath::current_entity_id`]) is always the last visited id. `hops`
/// holds the actual graph [`TogTriple`]s traversed, and `score` is the
/// cumulative relevance of the whole path.
#[derive(Debug, Clone, PartialEq)]
pub struct TogBeamPath {
    /// The topic entity id this path was seeded from.
    pub topic_entity_id: String,
    /// The graph triples traversed, in hop order. Empty for a seed path.
    pub hops: Vec<TogTriple>,
    /// Entity ids visited along the path, topic entity first. Always contains
    /// at least the topic entity id.
    pub visited_ids: Vec<String>,
    /// Cumulative relevance score: the topic seed score plus the added score of
    /// every hop. Non-decreasing as the path grows.
    pub score: f32,
}

impl TogBeamPath {
    /// Create a seed path consisting of a single topic entity with the given
    /// seed relevance `score`.
    #[must_use]
    pub fn seed(topic_entity_id: impl Into<String>, score: f32) -> Self {
        let topic_entity_id = topic_entity_id.into();
        Self {
            visited_ids: vec![topic_entity_id.clone()],
            topic_entity_id,
            hops: Vec::new(),
            score,
        }
    }

    /// The entity id at the frontier of the path (the last visited id).
    #[must_use]
    pub fn current_entity_id(&self) -> &str {
        self.visited_ids
            .last()
            .map_or(self.topic_entity_id.as_str(), String::as_str)
    }

    /// The number of hops taken (the path's depth).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.hops.len()
    }

    /// Return `true` when this is a seed path (no hops yet).
    #[must_use]
    pub fn is_seed(&self) -> bool {
        self.hops.is_empty()
    }

    /// Return `true` when `id` already appears on the path.
    #[must_use]
    pub fn contains_entity(&self, id: &str) -> bool {
        self.visited_ids.iter().any(|v| v == id)
    }

    /// The entity ids visited along the path, topic first.
    #[must_use]
    pub fn entities(&self) -> &[String] {
        &self.visited_ids
    }

    /// The triples traversed along the path, in hop order.
    #[must_use]
    pub fn triples(&self) -> &[TogTriple] {
        &self.hops
    }

    /// Return a new path that extends `self` by one hop.
    ///
    /// `triple` is the graph edge traversed, `neighbour_id` is the entity id
    /// stepped into (which becomes the new frontier), and `added_score` is
    /// added to the cumulative score.
    #[must_use]
    pub fn extended(
        &self,
        triple: TogTriple,
        neighbour_id: impl Into<String>,
        added_score: f32,
    ) -> Self {
        let mut hops = self.hops.clone();
        hops.push(triple);
        let mut visited_ids = self.visited_ids.clone();
        visited_ids.push(neighbour_id.into());
        Self {
            topic_entity_id: self.topic_entity_id.clone(),
            hops,
            visited_ids,
            score: self.score + added_score,
        }
    }
}

// ── TogDecision ─────────────────────────────────────────────────────────────

/// The reasoning / sufficiency verdict reached at the end of a depth iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TogDecision {
    /// The beam paths collectively cover enough of the query: the search stops
    /// early and synthesizes an answer.
    Sufficient {
        /// The depth at which sufficiency was reached.
        depth: usize,
        /// The coverage ratio achieved (`>= sufficiency_threshold`).
        coverage: f32,
    },
    /// The evidence is not yet sufficient: the search continues to the next
    /// depth.
    Continue {
        /// The depth just completed.
        depth: usize,
        /// The coverage ratio achieved so far (`< sufficiency_threshold`).
        coverage: f32,
    },
    /// The depth bound `D` was reached without sufficiency: the search stops and
    /// synthesizes from the best paths found so far.
    MaxDepthReached {
        /// The (maximum) depth reached.
        depth: usize,
        /// The coverage ratio achieved.
        coverage: f32,
    },
    /// No candidate expansions were available (a dead end or a disconnected
    /// topic entity): the search stops with the beam as it stood.
    Exhausted {
        /// The depth at which expansion ran out.
        depth: usize,
        /// The coverage ratio achieved.
        coverage: f32,
    },
}

impl TogDecision {
    /// Return `true` for the [`TogDecision::Sufficient`] verdict.
    #[must_use]
    pub fn is_sufficient(&self) -> bool {
        matches!(self, Self::Sufficient { .. })
    }

    /// Return `true` for any terminal verdict — anything but
    /// [`TogDecision::Continue`].
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Continue { .. })
    }

    /// The coverage ratio recorded in the verdict.
    #[must_use]
    pub fn coverage(&self) -> f32 {
        match self {
            Self::Sufficient { coverage, .. }
            | Self::Continue { coverage, .. }
            | Self::MaxDepthReached { coverage, .. }
            | Self::Exhausted { coverage, .. } => *coverage,
        }
    }

    /// The depth recorded in the verdict.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Sufficient { depth, .. }
            | Self::Continue { depth, .. }
            | Self::MaxDepthReached { depth, .. }
            | Self::Exhausted { depth, .. } => *depth,
        }
    }

    /// A stable lowercase label for the verdict variant.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sufficient { .. } => "sufficient",
            Self::Continue { .. } => "continue",
            Self::MaxDepthReached { .. } => "max_depth_reached",
            Self::Exhausted { .. } => "exhausted",
        }
    }
}

// ── TogExploration ──────────────────────────────────────────────────────────

/// An introspection record of a single depth iteration of beam search.
///
/// Captures how many candidate paths were generated before pruning, the beam
/// that survived pruning, the relations that were scored during relation
/// exploration (with their query-relevance scores), and the reasoning verdict
/// that ended the iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct TogExploration {
    /// The 1-based depth of this iteration.
    pub depth: usize,
    /// The number of candidate extended paths generated before pruning.
    pub candidates_generated: usize,
    /// The beam retained after pruning to width `W`.
    pub beam: Vec<TogBeamPath>,
    /// The relations scored during relation exploration this depth, paired with
    /// their query-relevance scores.
    pub scored_relations: Vec<(TogRelation, f32)>,
    /// The reasoning / sufficiency verdict that ended this iteration.
    pub decision: TogDecision,
}

impl TogExploration {
    /// Create a new exploration record.
    #[must_use]
    pub fn new(
        depth: usize,
        candidates_generated: usize,
        beam: Vec<TogBeamPath>,
        scored_relations: Vec<(TogRelation, f32)>,
        decision: TogDecision,
    ) -> Self {
        Self {
            depth,
            candidates_generated,
            beam,
            scored_relations,
            decision,
        }
    }
}

// ── TogResult ───────────────────────────────────────────────────────────────

/// The synthesized outcome of a full [`TogEngine`](crate::think_on_graph::TogEngine)
/// run.
#[derive(Debug, Clone, PartialEq)]
pub struct TogResult {
    /// The original query.
    pub query: String,
    /// The topic entity ids matched from the query and seeded into the beam.
    pub topic_entity_ids: Vec<String>,
    /// The final beam of reasoning paths, ranked best (highest score) first.
    pub paths: Vec<TogBeamPath>,
    /// The evidence triples: the de-duplicated union of every final path's hops,
    /// in beam order.
    pub evidence: Vec<TogTriple>,
    /// The synthesized natural-language answer.
    pub answer: String,
    /// Whether the search stopped early because the sufficiency check passed.
    pub stopped_early: bool,
    /// The maximum hop depth actually reached by the final beam.
    pub depth_reached: usize,
    /// The final coverage ratio of the query's content terms by the evidence.
    pub coverage: f32,
    /// The per-depth exploration trace.
    pub explorations: Vec<TogExploration>,
}

impl TogResult {
    /// Return `true` when the synthesized answer is non-empty.
    #[must_use]
    pub fn has_answer(&self) -> bool {
        !self.answer.trim().is_empty()
    }

    /// The best (highest-scoring) reasoning path, if any.
    #[must_use]
    pub fn best_path(&self) -> Option<&TogBeamPath> {
        self.paths.first()
    }

    /// The number of distinct evidence triples.
    #[must_use]
    pub fn evidence_count(&self) -> usize {
        self.evidence.len()
    }

    /// The number of reasoning paths in the final beam.
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }
}
