//! The `HippoRagIndex`: entity graph construction and single-step retrieval.

use std::collections::HashMap;

use crate::hippo_rag::graph::{extract_entities, personalized_pagerank};
use crate::hippo_rag::types::{HippoConfig, HippoHit, HippoRagError};
use crate::types::Document;

// ── HippoRagIndex ─────────────────────────────────────────────────────────────

/// A neurobiologically-inspired single-step multi-hop retriever (`HippoRAG`).
///
/// The index extracts entities from every passage, links entities that co-occur
/// in the same passage (the *associative memory* graph), and records which
/// entities each passage contains. A query is answered in **one step**: query
/// entities seed a Personalized `PageRank` walk over the entity graph, and each
/// passage is then scored by the total PPR mass of the entities it holds.
/// Because rank flows across co-occurrence edges, passages connected to the
/// query only through a shared intermediate entity are retrieved without an
/// explicit hop loop — distinguishing this from the breadth-first
/// `multi_hop` traversal.
///
/// # Examples
///
/// ```rust,ignore
/// # #[cfg(feature = "hipporag")] {
/// use oxirag::hippo_rag::{HippoConfig, HippoRagIndex};
/// use oxirag::types::Document;
///
/// let mut index = HippoRagIndex::new(HippoConfig::default());
/// index
///     .build(&[
///         Document::new("Alice works with Bob at Acme."),
///         Document::new("Bob studied under Carol at Cambridge."),
///     ])
///     .unwrap();
/// // A query mentioning Alice can reach the Carol passage through Bob.
/// let hits = index.search("Alice", 5).unwrap();
/// assert!(!hits.is_empty());
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HippoRagIndex {
    /// Graph-walk and hashing configuration.
    config: HippoConfig,
    /// The indexed passages, in insertion order.
    passages: Vec<Document>,
    /// Distinct entity surface forms (lowercased), indexed by node id.
    entities: Vec<String>,
    /// Reverse lookup from entity surface form to its node id.
    entity_index: HashMap<String, usize>,
    /// Symmetric weighted adjacency: `adjacency[i]` holds `(neighbour, weight)`.
    adjacency: Vec<Vec<(usize, f32)>>,
    /// For each passage, the node ids of the entities it contains.
    passage_entities: Vec<Vec<usize>>,
    /// Whether [`HippoRagIndex::build`] has populated the index.
    built: bool,
}

impl HippoRagIndex {
    /// Create a new, empty index with the given configuration.
    #[must_use]
    pub fn new(config: HippoConfig) -> Self {
        Self {
            config,
            passages: Vec::new(),
            entities: Vec::new(),
            entity_index: HashMap::new(),
            adjacency: Vec::new(),
            passage_entities: Vec::new(),
            built: false,
        }
    }

    /// Build the entity graph from `corpus`.
    ///
    /// For every passage the entities are extracted, distinct entities are
    /// assigned stable node ids, every unordered pair of entities co-occurring
    /// in a passage has its edge weight incremented by one, and the passage's
    /// entity-id list is recorded for scoring. Calling `build` again replaces
    /// any previously built state.
    ///
    /// # Errors
    ///
    /// Returns [`HippoRagError::EmptyCorpus`] when `corpus` is empty.
    pub fn build(&mut self, corpus: &[Document]) -> Result<(), HippoRagError> {
        if corpus.is_empty() {
            return Err(HippoRagError::EmptyCorpus);
        }

        // Reset so a rebuild is idempotent with respect to prior contents.
        self.passages = corpus.to_vec();
        self.entities.clear();
        self.entity_index.clear();
        self.passage_entities.clear();

        // Accumulate undirected edge weights keyed on an ordered id pair.
        let mut edge_weights: HashMap<(usize, usize), f32> = HashMap::new();

        for doc in corpus {
            let surface_forms = extract_entities(&doc.content);
            let mut ids: Vec<usize> = Vec::with_capacity(surface_forms.len());
            for form in surface_forms {
                let id = self.intern_entity(form);
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }

            // Link every unordered pair of entities sharing this passage.
            for a in 0..ids.len() {
                for b in (a + 1)..ids.len() {
                    let (lo, hi) = if ids[a] < ids[b] {
                        (ids[a], ids[b])
                    } else {
                        (ids[b], ids[a])
                    };
                    *edge_weights.entry((lo, hi)).or_insert(0.0) += 1.0;
                }
            }

            self.passage_entities.push(ids);
        }

        // Materialise the symmetric adjacency list from accumulated weights.
        self.adjacency = vec![Vec::new(); self.entities.len()];
        // Deterministic edge order: sort the pair keys before insertion.
        let mut pairs: Vec<((usize, usize), f32)> = edge_weights.into_iter().collect();
        pairs.sort_by_key(|a| a.0);
        for ((lo, hi), w) in pairs {
            self.adjacency[lo].push((hi, w));
            self.adjacency[hi].push((lo, w));
        }

        self.built = true;
        Ok(())
    }

    /// Intern an entity surface form, returning its stable node id.
    fn intern_entity(&mut self, form: String) -> usize {
        if let Some(&id) = self.entity_index.get(&form) {
            return id;
        }
        let id = self.entities.len();
        self.entity_index.insert(form.clone(), id);
        self.entities.push(form);
        id
    }

    /// Number of indexed passages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.passages.len()
    }

    /// Return `true` when no passages have been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Number of distinct entity nodes in the graph.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Resolve an entity surface form to its graph node id, if present.
    ///
    /// The form is matched case-insensitively against the corpus entities, so
    /// `"Alice"`, `"ALICE"`, and `"alice"` all resolve to the same node.
    #[must_use]
    pub fn entity_id(&self, surface: &str) -> Option<usize> {
        self.entity_index.get(&surface.to_lowercase()).copied()
    }

    /// Compute Personalized `PageRank` over the entity graph seeded from `seeds`.
    ///
    /// `seeds` lists entity node ids to teleport toward; out-of-range ids are
    /// ignored. The returned vector has one entry per entity node and, whenever
    /// at least one seed is valid, sums to `≈ 1.0`. With no valid seed the
    /// all-zero vector is returned.
    #[must_use]
    pub fn personalized_pagerank(&self, seeds: &[usize]) -> Vec<f32> {
        personalized_pagerank(
            &self.adjacency,
            seeds,
            self.config.damping,
            self.config.ppr_iterations,
        )
    }

    /// Map query entity surface forms to their graph node ids.
    ///
    /// Unknown entities (absent from the corpus) are skipped.
    fn seed_ids(&self, query: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = Vec::new();
        for form in extract_entities(query) {
            if let Some(&id) = self.entity_index.get(&form)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }

    /// Retrieve the `top_k` passages most relevant to `query`.
    ///
    /// Query entities seed a Personalized `PageRank` walk over the entity graph;
    /// each passage is then scored by the summed PPR mass of the entities it
    /// contains. Passages reachable only through an intermediate shared entity
    /// are scored above zero, realising single-step multi-hop retrieval. Ties
    /// break on insertion order so results are deterministic.
    ///
    /// # Errors
    ///
    /// - [`HippoRagError::NotBuilt`] — [`HippoRagIndex::build`] has not run.
    /// - [`HippoRagError::EmptyQuery`] — `query` is blank.
    /// - [`HippoRagError::NoEntities`] — no query token is a known entity.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<HippoHit>, HippoRagError> {
        if !self.built {
            return Err(HippoRagError::NotBuilt);
        }
        if query.trim().is_empty() {
            return Err(HippoRagError::EmptyQuery);
        }

        let seeds = self.seed_ids(query);
        if seeds.is_empty() {
            return Err(HippoRagError::NoEntities);
        }

        let ranks = self.personalized_pagerank(&seeds);

        // Score every passage by the PPR mass of its constituent entities.
        let mut hits: Vec<(usize, f32)> = self
            .passage_entities
            .iter()
            .enumerate()
            .map(|(idx, entity_ids)| {
                let score: f32 = entity_ids.iter().map(|&e| ranks[e]).sum();
                (idx, score)
            })
            .collect();

        // Highest score first; ties resolved by insertion order (stable index).
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        hits.truncate(top_k);

        Ok(hits
            .into_iter()
            .map(|(idx, score)| HippoHit {
                document: self.passages[idx].clone(),
                score,
            })
            .collect())
    }
}
