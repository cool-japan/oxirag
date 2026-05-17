//! HNSW graph nodes and index implementation.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::VectorStoreError;
use crate::layer1_echo::similarity::compute_similarity;
use crate::types::DocumentId;

use super::config::AnnConfig;
use super::stats::AnnStats;

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
    pub(crate) fn new(id: DocumentId, vector: Vec<f32>, level: usize) -> Self {
        let neighbors = (0..=level).map(|_| Vec::new()).collect();
        Self {
            id,
            vector,
            level,
            neighbors,
        }
    }
}

/// Candidate for search priority queue (min-heap by distance).
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) id: DocumentId,
    pub(crate) distance: f32,
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
    pub(crate) fn search_layer(
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
