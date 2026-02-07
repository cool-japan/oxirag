//! Approximate Nearest Neighbor (ANN) index using HNSW algorithm.
//!
//! This module provides a Hierarchical Navigable Small World (HNSW) graph
//! implementation for fast approximate nearest neighbor search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::VectorStoreError;
use crate::layer1_echo::filter::MetadataFilter;
use crate::layer1_echo::similarity::compute_similarity;
use crate::layer1_echo::traits::{IndexedDocument, SimilarityMetric, VectorStore};
use crate::types::{Document, DocumentId, SearchResult};

/// Configuration for the ANN index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnConfig {
    /// Max connections per node per layer (M parameter).
    pub m: usize,
    /// Max connections for layer 0.
    pub m_max: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    pub ef_search: usize,
    /// Level multiplier (typically 1/ln(M)).
    pub ml: f64,
    /// The distance metric to use.
    pub distance_metric: SimilarityMetric,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / 16.0_f64.ln(),
            distance_metric: SimilarityMetric::Cosine,
        }
    }
}

impl AnnConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the M parameter (max connections per node per layer).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set the `M_max` parameter (max connections for layer 0).
    #[must_use]
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Set the `ef_construction` parameter.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the `ef_search` parameter.
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_distance_metric(mut self, metric: SimilarityMetric) -> Self {
        self.distance_metric = metric;
        self
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// The document ID.
    pub id: DocumentId,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// The level of this node in the hierarchy.
    pub level: usize,
    /// Neighbors at each layer. neighbors\[layer\] = list of neighbor ids.
    pub neighbors: Vec<Vec<DocumentId>>,
}

impl HnswNode {
    /// Create a new HNSW node.
    fn new(id: DocumentId, vector: Vec<f32>, level: usize) -> Self {
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        Self {
            id,
            vector,
            level,
            neighbors,
        }
    }
}

/// Candidate for search priority queue.
#[derive(Debug, Clone)]
struct Candidate {
    id: DocumentId,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap candidate (for keeping farthest elements).
#[derive(Debug, Clone)]
struct MaxCandidate {
    id: DocumentId,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for MaxCandidate {}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// HNSW Index for approximate nearest neighbor search.
pub struct HnswIndex {
    config: AnnConfig,
    nodes: HashMap<DocumentId, HnswNode>,
    entry_point: Option<DocumentId>,
    max_level: usize,
    dimension: usize,
    rng_seed: u64,
}

impl HnswIndex {
    /// Create a new HNSW index.
    #[must_use]
    pub fn new(dimension: usize, config: AnnConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            dimension,
            rng_seed: 42,
        }
    }

    /// Insert a new vector into the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector dimension doesn't match the index dimension.
    ///
    /// # Panics
    ///
    /// This function will not panic under normal circumstances. The internal
    /// `.expect()` call is guarded by a check that the entry point exists.
    pub fn insert(&mut self, id: DocumentId, vector: Vec<f32>) -> Result<(), VectorStoreError> {
        if vector.len() != self.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }

        // Remove existing node with same ID if present
        self.remove(&id);

        let level = self.random_level();
        let mut node = HnswNode::new(id.clone(), vector, level);

        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(id.clone());
            self.max_level = level;
            self.nodes.insert(id, node);
            return Ok(());
        }

        let entry_point = self
            .entry_point
            .clone()
            .expect("entry point should exist when nodes are present");

        // Search from top level to node's level + 1, greedy search
        let mut current_ep = vec![entry_point];
        for lc in (level + 1..=self.max_level).rev() {
            let nearest = self.search_layer(&node.vector, current_ep, 1, lc);
            current_ep = nearest.into_iter().map(|c| c.id).collect();
        }

        // Search and connect from min(level, max_level) down to 0
        let top_level = level.min(self.max_level);
        for lc in (0..=top_level).rev() {
            let m = if lc == 0 {
                self.config.m_max
            } else {
                self.config.m
            };

            let candidates = self.search_layer(
                &node.vector,
                current_ep.clone(),
                self.config.ef_construction,
                lc,
            );

            // Select neighbors
            let neighbors = Self::select_neighbors(&candidates, m);

            // Add bidirectional connections
            node.neighbors[lc].clone_from(&neighbors);

            // Add reverse connections (connect neighbors to this node)
            for neighbor_id in &neighbors {
                // First check if we need to add this node as a neighbor
                let needs_connection = self.nodes.get(neighbor_id).is_some_and(|neighbor_node| {
                    lc < neighbor_node.neighbors.len() && !neighbor_node.neighbors[lc].contains(&id)
                });

                if needs_connection {
                    // Add the connection
                    if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id)
                        && lc < neighbor_node.neighbors.len()
                    {
                        neighbor_node.neighbors[lc].push(id.clone());
                    }

                    // Check if pruning is needed (separate borrow scope)
                    let pruning_data: Option<(Vec<f32>, Vec<DocumentId>)> = {
                        let max_conn = if lc == 0 {
                            self.config.m_max
                        } else {
                            self.config.m
                        };
                        self.nodes.get(neighbor_id).and_then(|neighbor_node| {
                            if lc < neighbor_node.neighbors.len()
                                && neighbor_node.neighbors[lc].len() > max_conn
                            {
                                Some((
                                    neighbor_node.vector.clone(),
                                    neighbor_node.neighbors[lc].clone(),
                                ))
                            } else {
                                None
                            }
                        })
                    };

                    // Perform pruning if needed
                    if let Some((neighbor_vec, current_neighbors)) = pruning_data {
                        let max_conn = if lc == 0 {
                            self.config.m_max
                        } else {
                            self.config.m
                        };
                        let neighbor_candidates: Vec<Candidate> = current_neighbors
                            .iter()
                            .filter_map(|nid| {
                                self.nodes.get(nid).map(|n| Candidate {
                                    id: nid.clone(),
                                    distance: self.distance(&neighbor_vec, &n.vector),
                                })
                            })
                            .collect();
                        let selected = Self::select_neighbors(&neighbor_candidates, max_conn);
                        if let Some(neighbor_node) = self.nodes.get_mut(neighbor_id)
                            && lc < neighbor_node.neighbors.len()
                        {
                            neighbor_node.neighbors[lc] = selected;
                        }
                    }
                }
            }

            current_ep = candidates.into_iter().map(|c| c.id).collect();
        }

        // Update entry point if this node has a higher level
        if level > self.max_level {
            self.entry_point = Some(id.clone());
            self.max_level = level;
        }

        self.nodes.insert(id, node);
        Ok(())
    }

    /// Remove a vector from the index.
    ///
    /// Returns the removed vector if found.
    pub fn remove(&mut self, id: &DocumentId) -> Option<Vec<f32>> {
        let node = self.nodes.remove(id)?;

        // Remove this node from all its neighbors
        for (level, neighbors) in node.neighbors.iter().enumerate() {
            for neighbor_id in neighbors {
                if let Some(neighbor) = self.nodes.get_mut(neighbor_id)
                    && level < neighbor.neighbors.len()
                {
                    neighbor.neighbors[level].retain(|nid| nid != id);
                }
            }
        }

        // Update entry point if needed
        if self.entry_point.as_ref() == Some(id) {
            self.entry_point = self.nodes.keys().next().cloned();
            self.max_level = self
                .entry_point
                .as_ref()
                .and_then(|ep| self.nodes.get(ep))
                .map_or(0, |n| n.level);

            // Find the node with maximum level to be new entry point
            for (node_id, n) in &self.nodes {
                if n.level > self.max_level {
                    self.max_level = n.level;
                    self.entry_point = Some(node_id.clone());
                }
            }
        }

        Some(node.vector)
    }

    /// Search for k nearest neighbors.
    ///
    /// Returns a vector of (`DocumentId`, `similarity_score`) pairs sorted by descending similarity.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(DocumentId, f32)> {
        self.search_with_threshold(query, k, f32::NEG_INFINITY)
    }

    /// Search for k nearest neighbors with a minimum score threshold.
    ///
    /// Returns a vector of (`DocumentId`, `similarity_score`) pairs sorted by descending similarity.
    #[must_use]
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        k: usize,
        min_score: f32,
    ) -> Vec<(DocumentId, f32)> {
        if self.nodes.is_empty() || query.len() != self.dimension {
            return Vec::new();
        }

        let entry_point = match &self.entry_point {
            Some(ep) => vec![ep.clone()],
            None => return Vec::new(),
        };

        // Greedy search from top to layer 1
        let mut current_ep = entry_point;
        for lc in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, current_ep, 1, lc);
            current_ep = nearest.into_iter().map(|c| c.id).collect();
        }

        // Search layer 0 with ef_search
        let candidates = self.search_layer(query, current_ep, self.config.ef_search.max(k), 0);

        // Convert distance to similarity and filter by threshold
        let mut results: Vec<(DocumentId, f32)> = candidates
            .into_iter()
            .map(|c| {
                let similarity = Self::distance_to_similarity(c.distance);
                (c.id, similarity)
            })
            .filter(|(_, score)| *score >= min_score)
            .collect();

        // Sort by descending similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Get the number of nodes in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.entry_point = None;
        self.max_level = 0;
    }

    /// Get statistics about the index.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> AnnStats {
        let num_nodes = self.nodes.len();
        let max_level = self.max_level;

        let total_connections: usize = self
            .nodes
            .values()
            .map(|n| n.neighbors.iter().map(Vec::len).sum::<usize>())
            .sum();

        let avg_connections = if num_nodes > 0 {
            total_connections as f32 / num_nodes as f32
        } else {
            0.0
        };

        // Estimate memory usage
        let memory_bytes = num_nodes
            * (std::mem::size_of::<HnswNode>()
                + self.dimension * std::mem::size_of::<f32>()
                + (max_level + 1) * self.config.m * std::mem::size_of::<DocumentId>());

        AnnStats {
            num_nodes,
            max_level,
            avg_connections,
            memory_bytes,
        }
    }

    /// Get a reference to a node by ID.
    #[must_use]
    pub fn get_node(&self, id: &DocumentId) -> Option<&HnswNode> {
        self.nodes.get(id)
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &AnnConfig {
        &self.config
    }

    /// Get the dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Generate a random level for a new node.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn random_level(&mut self) -> usize {
        // Simple LCG PRNG
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let r = (self.rng_seed >> 33) as f64 / (1u64 << 31) as f64;

        let level = (-r.ln() * self.config.ml).floor() as usize;
        level.min(32) // Cap at reasonable maximum
    }

    /// Search within a single layer.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<DocumentId>,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<DocumentId> = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();

        // Initialize with entry points
        for ep in entry_points {
            if visited.insert(ep.clone())
                && let Some(node) = self.nodes.get(&ep)
            {
                let dist = self.distance(query, &node.vector);
                candidates.push(Candidate {
                    id: ep.clone(),
                    distance: dist,
                });
                results.push(MaxCandidate {
                    id: ep,
                    distance: dist,
                });
            }
        }

        while let Some(current) = candidates.pop() {
            // Get the farthest result distance
            let farthest_dist = results.peek().map_or(f32::INFINITY, |r| r.distance);

            // If current is farther than the farthest result and we have enough results, stop
            if current.distance > farthest_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            if let Some(node) = self.nodes.get(&current.id)
                && layer < node.neighbors.len()
            {
                for neighbor_id in &node.neighbors[layer] {
                    if visited.insert(neighbor_id.clone())
                        && let Some(neighbor_node) = self.nodes.get(neighbor_id)
                    {
                        let dist = self.distance(query, &neighbor_node.vector);
                        let farthest_dist = results.peek().map_or(f32::INFINITY, |r| r.distance);

                        if dist < farthest_dist || results.len() < ef {
                            candidates.push(Candidate {
                                id: neighbor_id.clone(),
                                distance: dist,
                            });
                            results.push(MaxCandidate {
                                id: neighbor_id.clone(),
                                distance: dist,
                            });

                            // Keep only ef best results
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted vector
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|mc| Candidate {
                id: mc.id,
                distance: mc.distance,
            })
            .collect();

        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }

    /// Select the best neighbors from candidates.
    fn select_neighbors(candidates: &[Candidate], m: usize) -> Vec<DocumentId> {
        // Simple selection: take the closest m candidates
        let mut sorted: Vec<&Candidate> = candidates.iter().collect();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        sorted.into_iter().take(m).map(|c| c.id.clone()).collect()
    }

    /// Compute distance between two vectors (lower is more similar for internal use).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // We use negative similarity as distance (so lower = better)
        let similarity = compute_similarity(a, b, self.config.distance_metric);
        -similarity
    }

    /// Convert internal distance back to similarity score.
    fn distance_to_similarity(distance: f32) -> f32 {
        -distance
    }
}

/// ANN-enhanced vector store.
pub struct AnnVectorStore {
    index: HnswIndex,
    documents: HashMap<DocumentId, Document>,
    config: AnnConfig,
}

impl AnnVectorStore {
    /// Create a new ANN vector store.
    #[must_use]
    pub fn new(dimension: usize, config: AnnConfig) -> Self {
        Self {
            index: HnswIndex::new(dimension, config.clone()),
            documents: HashMap::new(),
            config,
        }
    }

    /// Create a new ANN vector store with default configuration.
    #[must_use]
    pub fn with_default_config(dimension: usize) -> Self {
        Self::new(dimension, AnnConfig::default())
    }

    /// Get the HNSW index statistics.
    #[must_use]
    pub fn stats(&self) -> AnnStats {
        self.index.stats()
    }

    /// Get a reference to the underlying HNSW index.
    #[must_use]
    pub fn index(&self) -> &HnswIndex {
        &self.index
    }
}

#[async_trait]
impl VectorStore for AnnVectorStore {
    async fn insert(&mut self, doc: IndexedDocument) -> Result<(), VectorStoreError> {
        if doc.embedding.len() != self.index.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension,
                actual: doc.embedding.len(),
            });
        }

        if self.documents.contains_key(&doc.document.id) {
            return Err(VectorStoreError::DuplicateId(doc.document.id.to_string()));
        }

        self.index.insert(doc.document.id.clone(), doc.embedding)?;
        self.documents.insert(doc.document.id.clone(), doc.document);
        Ok(())
    }

    async fn insert_batch(&mut self, docs: Vec<IndexedDocument>) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.insert(doc).await?;
        }
        Ok(())
    }

    async fn get(&self, id: &DocumentId) -> Result<Option<IndexedDocument>, VectorStoreError> {
        match (self.documents.get(id), self.index.get_node(id)) {
            (Some(doc), Some(node)) => {
                Ok(Some(IndexedDocument::new(doc.clone(), node.vector.clone())))
            }
            _ => Ok(None),
        }
    }

    async fn delete(&mut self, id: &DocumentId) -> Result<bool, VectorStoreError> {
        let removed = self.index.remove(id).is_some();
        self.documents.remove(id);
        Ok(removed)
    }

    async fn update(
        &mut self,
        id: &DocumentId,
        embedding: Vec<f32>,
    ) -> Result<bool, VectorStoreError> {
        if embedding.len() != self.index.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension,
                actual: embedding.len(),
            });
        }

        if !self.documents.contains_key(id) {
            return Err(VectorStoreError::NotFound(id.to_string()));
        }

        self.index.insert(id.clone(), embedding)?;
        Ok(true)
    }

    async fn upsert(&mut self, doc: IndexedDocument) -> Result<bool, VectorStoreError> {
        if doc.embedding.len() != self.index.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension,
                actual: doc.embedding.len(),
            });
        }

        let is_new = !self.documents.contains_key(&doc.document.id);
        self.index.insert(doc.document.id.clone(), doc.embedding)?;
        self.documents.insert(doc.document.id.clone(), doc.document);
        Ok(is_new)
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.index.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension,
                actual: query_embedding.len(),
            });
        }

        let min_score = min_score.unwrap_or(f32::NEG_INFINITY);
        let results = self
            .index
            .search_with_threshold(query_embedding, top_k, min_score);

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .enumerate()
            .filter_map(|(rank, (id, score))| {
                self.documents
                    .get(&id)
                    .map(|doc| SearchResult::new(doc.clone(), score, rank))
            })
            .collect();

        Ok(search_results)
    }

    async fn search_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_score: Option<f32>,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        if query_embedding.len() != self.index.dimension {
            return Err(VectorStoreError::DimensionMismatch {
                expected: self.index.dimension,
                actual: query_embedding.len(),
            });
        }

        // For filtered search, we need to search more candidates and then filter
        // This is a simple approach; more sophisticated methods exist
        let search_multiplier = 10; // Search more to account for filtered out results
        let extended_k = top_k * search_multiplier;

        let min_score = min_score.unwrap_or(f32::NEG_INFINITY);
        let results = self
            .index
            .search_with_threshold(query_embedding, extended_k, min_score);

        let search_results: Vec<SearchResult> = results
            .into_iter()
            .filter_map(|(id, score)| {
                self.documents.get(&id).and_then(|doc| {
                    // Apply filter if provided
                    match filter {
                        Some(f) if !f.matches(&doc.metadata) => None,
                        _ => Some((doc.clone(), score)),
                    }
                })
            })
            .take(top_k)
            .enumerate()
            .map(|(rank, (doc, score))| SearchResult::new(doc, score, rank))
            .collect();

        Ok(search_results)
    }

    async fn count(&self) -> usize {
        self.documents.len()
    }

    async fn clear(&mut self) -> Result<(), VectorStoreError> {
        self.index.clear();
        self.documents.clear();
        Ok(())
    }

    fn dimension(&self) -> usize {
        self.index.dimension
    }

    fn similarity_metric(&self) -> SimilarityMetric {
        self.config.distance_metric
    }
}

/// Statistics for the ANN index.
#[derive(Debug, Clone, Default)]
pub struct AnnStats {
    /// Number of nodes in the index.
    pub num_nodes: usize,
    /// Maximum level in the hierarchy.
    pub max_level: usize,
    /// Average number of connections per node.
    pub avg_connections: f32,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_random_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                #[allow(clippy::cast_precision_loss)]
                let val = ((rng >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0;
                val
            })
            .collect()
    }

    fn normalize_vector(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    // Test 1: Index creation and configuration
    #[test]
    fn test_index_creation_default_config() {
        let config = AnnConfig::default();
        let index = HnswIndex::new(64, config);

        assert_eq!(index.dimension(), 64);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    // Test 2: Custom configuration
    #[test]
    fn test_index_custom_config() {
        let config = AnnConfig::new()
            .with_m(32)
            .with_m_max(64)
            .with_ef_construction(100)
            .with_ef_search(25)
            .with_distance_metric(SimilarityMetric::DotProduct);

        assert_eq!(config.m, 32);
        assert_eq!(config.m_max, 64);
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef_search, 25);
        assert_eq!(config.distance_metric, SimilarityMetric::DotProduct);
    }

    // Test 3: Insert single vector
    #[test]
    fn test_insert_single_vector() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id = DocumentId::new();
        let vector = vec![1.0, 0.0, 0.0, 0.0];

        index.insert(id.clone(), vector.clone()).unwrap();

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());

        let node = index.get_node(&id).unwrap();
        assert_eq!(node.vector, vector);
    }

    // Test 4: Insert multiple vectors
    #[test]
    fn test_insert_multiple_vectors() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        for i in 0..10 {
            let id = DocumentId::from_string(format!("doc_{i}"));
            let vector = create_random_vector(4, i);
            index.insert(id, vector).unwrap();
        }

        assert_eq!(index.len(), 10);
    }

    // Test 5: Dimension mismatch error
    #[test]
    fn test_insert_dimension_mismatch() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id = DocumentId::new();
        let vector = vec![1.0, 0.0, 0.0]; // Wrong dimension

        let result = index.insert(id, vector);
        assert!(matches!(
            result,
            Err(VectorStoreError::DimensionMismatch { .. })
        ));
    }

    // Test 6: Search returns correct results
    #[test]
    fn test_search_basic() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut index = HnswIndex::new(4, config);

        // Insert vectors
        let target = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
        let other1 = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);
        let other2 = normalize_vector(&[0.0, 0.0, 1.0, 0.0]);
        let similar = normalize_vector(&[0.9, 0.1, 0.0, 0.0]);

        index
            .insert(DocumentId::from_string("target"), target.clone())
            .unwrap();
        index
            .insert(DocumentId::from_string("other1"), other1)
            .unwrap();
        index
            .insert(DocumentId::from_string("other2"), other2)
            .unwrap();
        index
            .insert(DocumentId::from_string("similar"), similar)
            .unwrap();

        let results = index.search(&target, 2);

        assert_eq!(results.len(), 2);
        // The target itself should be the most similar
        assert_eq!(results[0].0, DocumentId::from_string("target"));
        // Similar should be second
        assert_eq!(results[1].0, DocumentId::from_string("similar"));
    }

    // Test 7: Search accuracy comparison with brute force
    #[test]
    fn test_search_accuracy_vs_brute_force() {
        // Use higher M for better connectivity and higher ef for better search
        let config = AnnConfig::default()
            .with_m(32)
            .with_m_max(64)
            .with_ef_search(100)
            .with_ef_construction(200);
        let mut index = HnswIndex::new(32, config.clone());

        let num_vectors = 100;
        let mut vectors: Vec<(DocumentId, Vec<f32>)> = Vec::new();

        for i in 0..num_vectors {
            let id = DocumentId::from_string(format!("doc_{i}"));
            let vector = normalize_vector(&create_random_vector(32, i));
            vectors.push((id.clone(), vector.clone()));
            index.insert(id, vector).unwrap();
        }

        // Search with a query
        let query = normalize_vector(&create_random_vector(32, 999));
        let k = 10;

        // HNSW search
        let hnsw_results = index.search(&query, k);

        // Brute force search
        let mut brute_force: Vec<(DocumentId, f32)> = vectors
            .iter()
            .map(|(id, v)| {
                let similarity = compute_similarity(&query, v, config.distance_metric);
                (id.clone(), similarity)
            })
            .collect();
        brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        brute_force.truncate(k);

        // Check recall (how many of top-k brute force results are in HNSW results)
        let hnsw_ids: HashSet<_> = hnsw_results.iter().map(|(id, _)| id.clone()).collect();
        let bf_ids: HashSet<_> = brute_force.iter().map(|(id, _)| id.clone()).collect();

        #[allow(clippy::cast_precision_loss)]
        let recall = hnsw_ids.intersection(&bf_ids).count() as f32 / k as f32;
        // ANN is approximate - with good parameters we should get at least 30% recall
        // This test validates the algorithm works, not that it achieves perfect recall
        assert!(recall >= 0.3, "Recall should be at least 30%, got {recall}");
    }

    // Test 8: Remove vector
    #[test]
    fn test_remove_vector() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id1 = DocumentId::from_string("doc1");
        let id2 = DocumentId::from_string("doc2");

        index.insert(id1.clone(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(id2.clone(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);

        let removed = index.remove(&id1);
        assert!(removed.is_some());
        assert_eq!(index.len(), 1);
        assert!(index.get_node(&id1).is_none());
        assert!(index.get_node(&id2).is_some());
    }

    // Test 9: Remove non-existent vector
    #[test]
    fn test_remove_nonexistent() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id = DocumentId::from_string("nonexistent");
        let removed = index.remove(&id);
        assert!(removed.is_none());
    }

    // Test 10: Search with different similarity metrics
    #[test]
    fn test_different_similarity_metrics() {
        // Cosine similarity
        let cosine_config = AnnConfig::default().with_distance_metric(SimilarityMetric::Cosine);
        let mut cosine_index = HnswIndex::new(4, cosine_config);

        // Dot product
        let dot_config = AnnConfig::default().with_distance_metric(SimilarityMetric::DotProduct);
        let mut dot_index = HnswIndex::new(4, dot_config);

        // Euclidean
        let euc_config = AnnConfig::default().with_distance_metric(SimilarityMetric::Euclidean);
        let mut euc_index = HnswIndex::new(4, euc_config);

        let id = DocumentId::from_string("test");
        let vector = vec![1.0, 0.0, 0.0, 0.0];

        cosine_index.insert(id.clone(), vector.clone()).unwrap();
        dot_index.insert(id.clone(), vector.clone()).unwrap();
        euc_index.insert(id, vector.clone()).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];

        let cosine_results = cosine_index.search(&query, 1);
        let dot_results = dot_index.search(&query, 1);
        let euc_results = euc_index.search(&query, 1);

        // All should find the same vector
        assert_eq!(cosine_results.len(), 1);
        assert_eq!(dot_results.len(), 1);
        assert_eq!(euc_results.len(), 1);

        // Scores should be 1.0 for identical vectors
        assert!((cosine_results[0].1 - 1.0).abs() < 0.01);
        assert!((dot_results[0].1 - 1.0).abs() < 0.01);
        assert!((euc_results[0].1 - 1.0).abs() < 0.01);
    }

    // Test 11: Search with threshold
    #[test]
    fn test_search_with_threshold() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut index = HnswIndex::new(4, config);

        let similar = normalize_vector(&[1.0, 0.1, 0.0, 0.0]);
        let dissimilar = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);

        index
            .insert(DocumentId::from_string("similar"), similar)
            .unwrap();
        index
            .insert(DocumentId::from_string("dissimilar"), dissimilar)
            .unwrap();

        let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);

        // Search with high threshold
        let results = index.search_with_threshold(&query, 10, 0.9);

        // Only the similar vector should pass the threshold
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, DocumentId::from_string("similar"));
    }

    // Test 12: Empty index search
    #[test]
    fn test_search_empty_index() {
        let config = AnnConfig::default();
        let index = HnswIndex::new(4, config);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10);

        assert!(results.is_empty());
    }

    // Test 13: Single element index
    #[test]
    fn test_single_element_index() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id = DocumentId::from_string("only");
        let vector = vec![1.0, 0.0, 0.0, 0.0];
        index.insert(id.clone(), vector.clone()).unwrap();

        let results = index.search(&vector, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }

    // Test 14: Clear index
    #[test]
    fn test_clear_index() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        for i in 0..10 {
            let id = DocumentId::from_string(format!("doc_{i}"));
            index.insert(id, create_random_vector(4, i)).unwrap();
        }

        assert_eq!(index.len(), 10);

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    // Test 15: Statistics
    #[test]
    fn test_index_stats() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        for i in 0..10 {
            let id = DocumentId::from_string(format!("doc_{i}"));
            index.insert(id, create_random_vector(4, i)).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_nodes, 10);
        assert!(stats.memory_bytes > 0);
    }

    // Test 16: Duplicate ID replacement
    #[test]
    fn test_duplicate_id_replacement() {
        let config = AnnConfig::default();
        let mut index = HnswIndex::new(4, config);

        let id = DocumentId::from_string("doc");
        let vector1 = vec![1.0, 0.0, 0.0, 0.0];
        let vector2 = vec![0.0, 1.0, 0.0, 0.0];

        index.insert(id.clone(), vector1).unwrap();
        index.insert(id.clone(), vector2.clone()).unwrap();

        assert_eq!(index.len(), 1);

        let node = index.get_node(&id).unwrap();
        assert_eq!(node.vector, vector2);
    }

    // Test 17: AnnVectorStore basic operations
    #[tokio::test]
    async fn test_ann_vector_store_basic() {
        let config = AnnConfig::default();
        let mut store = AnnVectorStore::new(4, config);

        let doc = IndexedDocument::new(Document::new("test content"), vec![1.0, 0.0, 0.0, 0.0]);

        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().document.content, "test content");
    }

    // Test 18: AnnVectorStore search
    #[tokio::test]
    async fn test_ann_vector_store_search() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut store = AnnVectorStore::new(4, config);

        let doc1 = IndexedDocument::new(
            Document::new("similar"),
            normalize_vector(&[1.0, 0.1, 0.0, 0.0]),
        );

        let doc2 = IndexedDocument::new(
            Document::new("dissimilar"),
            normalize_vector(&[0.0, 1.0, 0.0, 0.0]),
        );

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
        let results = store.search(&query, 2, None).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.content, "similar");
        assert_eq!(results[0].rank, 0);
    }

    // Test 19: AnnVectorStore with filter
    #[tokio::test]
    async fn test_ann_vector_store_with_filter() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut store = AnnVectorStore::new(4, config);

        let doc1 = IndexedDocument::new(
            Document::new("science doc").with_metadata("category", "science"),
            normalize_vector(&[1.0, 0.1, 0.0, 0.0]),
        );

        let doc2 = IndexedDocument::new(
            Document::new("tech doc").with_metadata("category", "technology"),
            normalize_vector(&[0.9, 0.2, 0.0, 0.0]),
        );

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);
        let filter = MetadataFilter::eq("category", "science");

        let results = store
            .search_with_filter(&query, 10, None, Some(&filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.content, "science doc");
    }

    // Test 20: AnnVectorStore upsert
    #[tokio::test]
    async fn test_ann_vector_store_upsert() {
        let config = AnnConfig::default();
        let mut store = AnnVectorStore::new(4, config);

        let doc = IndexedDocument::new(Document::new("original"), vec![1.0, 0.0, 0.0, 0.0]);

        let id = doc.document.id.clone();

        // First upsert (insert)
        let is_new = store.upsert(doc).await.unwrap();
        assert!(is_new);
        assert_eq!(store.count().await, 1);

        // Second upsert (update)
        let updated_doc = IndexedDocument::new(
            Document::new("updated").with_id(id.clone()),
            vec![0.0, 1.0, 0.0, 0.0],
        );

        let is_new = store.upsert(updated_doc).await.unwrap();
        assert!(!is_new);
        assert_eq!(store.count().await, 1);

        let retrieved = store.get(&id).await.unwrap().unwrap();
        assert_eq!(retrieved.document.content, "updated");
    }

    // Test 21: AnnVectorStore delete
    #[tokio::test]
    async fn test_ann_vector_store_delete() {
        let config = AnnConfig::default();
        let mut store = AnnVectorStore::new(4, config);

        let doc = IndexedDocument::new(Document::new("test"), vec![1.0, 0.0, 0.0, 0.0]);

        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();
        assert_eq!(store.count().await, 1);

        let deleted = store.delete(&id).await.unwrap();
        assert!(deleted);
        assert_eq!(store.count().await, 0);
    }

    // Test 22: AnnVectorStore update
    #[tokio::test]
    async fn test_ann_vector_store_update() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut store = AnnVectorStore::new(4, config);

        let doc = IndexedDocument::new(
            Document::new("test"),
            normalize_vector(&[1.0, 0.0, 0.0, 0.0]),
        );

        let id = doc.document.id.clone();

        store.insert(doc).await.unwrap();

        // Update embedding
        let new_embedding = normalize_vector(&[0.0, 1.0, 0.0, 0.0]);
        let updated = store.update(&id, new_embedding.clone()).await.unwrap();
        assert!(updated);

        // Verify by searching
        let results = store.search(&new_embedding, 1, None).await.unwrap();
        assert_eq!(results[0].document.id, id);
    }

    // Test 23: AnnVectorStore clear
    #[tokio::test]
    async fn test_ann_vector_store_clear() {
        let config = AnnConfig::default();
        let mut store = AnnVectorStore::new(4, config);

        for i in 0..5 {
            let doc =
                IndexedDocument::new(Document::new(format!("doc{i}")), create_random_vector(4, i));
            store.insert(doc).await.unwrap();
        }

        assert_eq!(store.count().await, 5);

        store.clear().await.unwrap();

        assert_eq!(store.count().await, 0);
    }

    // Test 24: Search results order (descending similarity)
    #[test]
    fn test_search_results_order() {
        let config = AnnConfig::default().with_ef_search(100);
        let mut index = HnswIndex::new(4, config);

        // Insert vectors with known similarities to query [1, 0, 0, 0]
        let query = normalize_vector(&[1.0, 0.0, 0.0, 0.0]);

        let v1 = normalize_vector(&[1.0, 0.0, 0.0, 0.0]); // sim ~ 1.0
        let v2 = normalize_vector(&[0.9, 0.3, 0.0, 0.0]); // sim ~ 0.95
        let v3 = normalize_vector(&[0.7, 0.7, 0.0, 0.0]); // sim ~ 0.71
        let v4 = normalize_vector(&[0.0, 1.0, 0.0, 0.0]); // sim ~ 0.0

        index.insert(DocumentId::from_string("v4"), v4).unwrap();
        index.insert(DocumentId::from_string("v2"), v2).unwrap();
        index.insert(DocumentId::from_string("v3"), v3).unwrap();
        index.insert(DocumentId::from_string("v1"), v1).unwrap();

        let results = index.search(&query, 4);

        // Should be sorted by descending similarity
        assert_eq!(results[0].0, DocumentId::from_string("v1"));
        assert_eq!(results[1].0, DocumentId::from_string("v2"));
        assert_eq!(results[2].0, DocumentId::from_string("v3"));
        assert_eq!(results[3].0, DocumentId::from_string("v4"));

        // Verify scores are in descending order
        for i in 0..results.len() - 1 {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    // Test 25: Benchmark-style test comparing search times
    #[test]
    fn test_search_performance_scales() {
        let config = AnnConfig::default().with_ef_search(50);
        let mut index = HnswIndex::new(64, config);

        // Insert vectors
        for i in 0..500 {
            let id = DocumentId::from_string(format!("doc_{i}"));
            let vector = normalize_vector(&create_random_vector(64, i));
            index.insert(id, vector).expect("Failed to insert vector");
        }

        let query = normalize_vector(&create_random_vector(64, 999));

        // Measure search time (basic timing)
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = index.search(&query, 10);
        }
        let duration = start.elapsed();

        // Should complete 100 searches in reasonable time
        // Use different thresholds for debug vs release builds
        #[cfg(debug_assertions)]
        let max_duration_secs = 5; // Debug builds are much slower
        #[cfg(not(debug_assertions))]
        let max_duration_secs = 1;

        assert!(
            duration.as_secs() < max_duration_secs,
            "Search took too long: {duration:?} (max: {max_duration_secs}s)"
        );
    }
}
