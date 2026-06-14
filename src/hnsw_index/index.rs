//! Core HNSW index: node storage, insertion, and beam-search retrieval.
//!
//! Implements the Hierarchical Navigable Small World (HNSW) graph as described
//! in Malkov & Yashunin (2018). Each node lives in a randomly assigned set of
//! layers (level 0 is always present); within each layer the node is connected
//! to at most `M` nearest neighbors via bidirectional edges. Search descends
//! greedily through upper layers and runs an `ef_search`-wide beam scan at
//! layer 0. All vectors are assumed to be pre-normalised to unit length so
//! cosine similarity reduces to a dot product.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use super::types::{HnswConfig, HnswError, HnswHit};

// ── Internal node representation ──────────────────────────────────────────────

/// A single node stored inside the graph.
#[derive(Debug, Clone)]
struct Node {
    id: String,
    vector: Vec<f32>,
    /// `layers[l]` holds the neighbor indices at graph layer `l`.
    layers: Vec<Vec<usize>>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// FNV-1a hash of a `u64` value (used for deterministic level assignment).
#[inline]
fn fnv_u64(x: u64) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in x.to_le_bytes() {
        h = h.wrapping_mul(1_099_511_628_211) ^ u64::from(b);
    }
    h
}

/// Assign a graph level to the node at position `node_idx`.
///
/// The distribution mirrors the probabilistic layer assignment from the original
/// HNSW paper: level `l` is chosen with probability proportional to
/// `(1/m)^l`.  Here we approximate that by hashing the index deterministically
/// and solving `-ln(U) / ln(m)` where `U` is derived from the hash.
fn level_for(node_idx: usize, m: usize, max_layers: usize) -> usize {
    let hash = fnv_u64(node_idx as u64);
    let u = (hash % 1024) as f64 / 1024.0; // uniform in [0, 1)
    // Avoid ln(0) — if u == 0 return 0 directly
    if u == 0.0 {
        return 0;
    }
    let raw = (-u.ln() / (m as f64).ln()).floor() as usize;
    raw.min(max_layers - 1)
}

/// Cosine similarity assuming both vectors are pre-normalised (dot product).
///
/// The result is clamped to `[-1.0, 1.0]`.
#[inline]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// Convert a cosine value in `[-1, 1]` to a score in `[0, 1]`.
///
/// Uses `f32::midpoint` to avoid the overflow lint.
#[inline]
fn score_of(cos: f32) -> f32 {
    f32::midpoint(cos, 1.0)
}

// ── HnswIndex ─────────────────────────────────────────────────────────────────

/// Hierarchical Navigable Small World approximate-nearest-neighbour index.
///
/// Build the index by calling [`insert`](HnswIndex::insert) for each vector,
/// then query with [`search`](HnswIndex::search).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "hnsw")] {
/// use oxirag::hnsw_index::{HnswConfig, HnswIndex};
///
/// fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
///     let mut v = vec![0.0f32; dim];
///     let bytes = text.as_bytes();
///     for (i, slot) in v.iter_mut().enumerate() {
///         let mut h: u64 = 14_695_981_039_346_656_037;
///         for &b in bytes { h = h.wrapping_mul(1_099_511_628_211) ^ b as u64; }
///         h = h.wrapping_mul(1_099_511_628_211) ^ i as u64;
///         *slot = ((h >> 32) as f32) / u32::MAX as f32 * 2.0 - 1.0;
///     }
///     let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
///     v.iter_mut().for_each(|x| *x /= norm);
///     v
/// }
///
/// let cfg = HnswConfig::new().with_dim(16).with_m(4).with_ef_construction(20).with_ef_search(10);
/// let mut index = HnswIndex::new(cfg);
/// index.insert("doc_a", fnv_embed("alpha", 16)).unwrap();
/// index.insert("doc_b", fnv_embed("beta",  16)).unwrap();
///
/// let hits = index.search(&fnv_embed("alpha", 16), 1).unwrap();
/// assert_eq!(hits[0].id, "doc_a");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct HnswIndex {
    config: HnswConfig,
    nodes: Vec<Node>,
    /// Index of the current entry point (node with the highest layer).
    entry_point: Option<usize>,
    /// Maximum layer currently occupied by any node.
    max_layer: usize,
}

impl HnswIndex {
    /// Create a new, empty HNSW index with the given configuration.
    ///
    /// # Panics
    ///
    /// Does not panic; invalid configurations surface as errors on the first
    /// [`insert`](HnswIndex::insert).
    #[must_use]
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Return the number of vectors stored in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` if the index contains no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return `true` if the index contains a node with the given `id`.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.nodes.iter().any(|n| n.id == id)
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /// Insert a vector into the index with the given string identifier.
    ///
    /// The vector must have exactly `config.dim` components. Vectors need not
    /// be pre-normalised but cosine similarity works best on unit vectors; the
    /// caller is responsible for normalisation.
    ///
    /// # Errors
    ///
    /// * [`HnswError::InvalidConfig`] — if the configuration is invalid.
    /// * [`HnswError::DimMismatch`] — if `vector.len() != config.dim`.
    pub fn insert(&mut self, id: impl Into<String>, vector: Vec<f32>) -> Result<(), HnswError> {
        self.config.validate()?;

        if vector.len() != self.config.dim {
            return Err(HnswError::DimMismatch {
                expected: self.config.dim,
                got: vector.len(),
            });
        }

        let node_idx = self.nodes.len();
        let level = level_for(node_idx, self.config.m, self.config.max_layers);

        // Allocate neighbor lists for each layer this node occupies.
        let layers: Vec<Vec<usize>> = (0..=level).map(|_| Vec::new()).collect();
        self.nodes.push(Node {
            id: id.into(),
            vector,
            layers,
        });

        let Some(ep) = self.entry_point else {
            // First node — it becomes the entry point.
            self.entry_point = Some(node_idx);
            self.max_layer = level;
            return Ok(());
        };

        // Greedy descent from the current top layer down to `level + 1`.
        let mut current_ep = ep;
        let query = self.nodes[node_idx].vector.clone();

        let top = self.max_layer;
        if top > level {
            for lc in (level + 1..=top).rev() {
                current_ep = self.greedy_closest(current_ep, &query, lc);
            }
        }

        // For each layer from `min(level, max_layer)` down to 0, run a beam
        // search with `ef_construction` candidates and wire bidirectional edges.
        let start_layer = level.min(self.max_layer);
        let ef = self.config.ef_construction;
        let m = self.config.m;

        for lc in (0..=start_layer).rev() {
            let candidates = self.beam_search(current_ep, &query, ef, lc);

            // Pick the M nearest as neighbors for the new node.
            let neighbors: Vec<usize> = candidates.iter().take(m).map(|&(_, idx)| idx).collect();

            // Connect: new node → neighbors
            self.nodes[node_idx].layers[lc].clone_from(&neighbors);

            // Connect: neighbors → new node (bidirectional), pruning to M if needed.
            for &nb in &neighbors {
                self.nodes[nb].layers[lc].push(node_idx);
                if self.nodes[nb].layers[lc].len() > m {
                    self.prune_neighbors(nb, lc, m);
                }
            }

            // The best candidate becomes the entry point for the next lower layer.
            if let Some(&(_, best)) = candidates.first() {
                current_ep = best;
            }
        }

        // Update global entry point if the new node occupies a higher layer.
        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(node_idx);
        }

        Ok(())
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Returns hits sorted by descending score (`1.0` = perfect cosine match).
    ///
    /// # Panics
    ///
    /// Will not panic in practice — the internal `expect` is guarded by the
    /// `EmptyIndex` check immediately above it.
    ///
    /// # Errors
    ///
    /// * [`HnswError::EmptyIndex`] — if the index contains no vectors.
    /// * [`HnswError::InvalidK`] — if `k == 0`.
    /// * [`HnswError::DimMismatch`] — if `query.len() != config.dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<HnswHit>, HnswError> {
        if self.is_empty() {
            return Err(HnswError::EmptyIndex);
        }
        if k == 0 {
            return Err(HnswError::InvalidK);
        }
        if query.len() != self.config.dim {
            return Err(HnswError::DimMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }

        let ep = self.entry_point.expect("entry_point set when non-empty");
        let mut current_ep = ep;

        // Greedy descent from the top layer down to layer 1.
        for lc in (1..=self.max_layer).rev() {
            current_ep = self.greedy_closest(current_ep, query, lc);
        }

        // At layer 0, run a full ef_search beam search.
        let ef = self.config.ef_search.max(k);
        let candidates = self.beam_search(current_ep, query, ef, 0);

        let hits: Vec<HnswHit> = candidates
            .into_iter()
            .take(k)
            .map(|(cos, idx)| HnswHit::new(self.nodes[idx].id.clone(), score_of(cos)))
            .collect();

        Ok(hits)
    }

    // ── Internal graph traversal ───────────────────────────────────────────────

    /// Greedy single-step descent: starting from `ep`, find the neighbor at
    /// `layer` with the highest cosine similarity to `query`.
    fn greedy_closest(&self, ep: usize, query: &[f32], layer: usize) -> usize {
        let mut best_idx = ep;
        let mut best_sim = cosine(query, &self.nodes[ep].vector);

        loop {
            let mut improved = false;
            // Only iterate over neighbors that exist at this layer.
            let neighbors = if layer < self.nodes[best_idx].layers.len() {
                self.nodes[best_idx].layers[layer].clone()
            } else {
                Vec::new()
            };

            for nb in neighbors {
                let sim = cosine(query, &self.nodes[nb].vector);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = nb;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
        best_idx
    }

    /// Beam search at `layer` starting from `ep` with beam width `ef`.
    ///
    /// Returns a list of `(cosine, node_idx)` pairs sorted by **descending**
    /// cosine (best first).
    fn beam_search(&self, ep: usize, query: &[f32], ef: usize, layer: usize) -> Vec<(f32, usize)> {
        let mut visited = std::collections::HashSet::new();
        visited.insert(ep);

        let ep_cos = cosine(query, &self.nodes[ep].vector);

        // frontier: candidates to expand (highest cosine first).
        let mut frontier: Vec<(f32, usize)> = vec![(ep_cos, ep)];
        // result: the ef nearest found so far.
        let mut result: Vec<(f32, usize)> = vec![(ep_cos, ep)];

        while !frontier.is_empty() {
            // Pop the candidate with the highest similarity from the frontier.
            let front_pos = frontier
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let (front_sim, front_idx) = frontier.remove(front_pos);

            // Worst score in the result set so far.
            let worst_result = result.iter().map(|&(c, _)| c).fold(f32::INFINITY, f32::min);

            // If the best candidate is worse than the worst in our result, stop.
            if front_sim < worst_result && result.len() >= ef {
                break;
            }

            // Expand neighbors at this layer.
            let neighbors = if layer < self.nodes[front_idx].layers.len() {
                self.nodes[front_idx].layers[layer].clone()
            } else {
                Vec::new()
            };

            for nb in neighbors {
                if !visited.insert(nb) {
                    continue;
                }
                let nb_sim = cosine(query, &self.nodes[nb].vector);

                let worst = result.iter().map(|&(c, _)| c).fold(f32::INFINITY, f32::min);

                if result.len() < ef || nb_sim > worst {
                    frontier.push((nb_sim, nb));
                    result.push((nb_sim, nb));

                    // Trim result to ef.
                    if result.len() > ef {
                        let worst_pos = result
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap())
                            .map(|(i, _)| i)
                            .unwrap();
                        result.remove(worst_pos);
                    }
                }
            }
        }

        // Sort result by descending cosine.
        result.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        result
    }

    /// Prune the neighbor list of node `idx` at `layer` to at most `m` entries,
    /// retaining the `m` neighbors with the highest cosine similarity to the node.
    fn prune_neighbors(&mut self, idx: usize, layer: usize, m: usize) {
        if layer >= self.nodes[idx].layers.len() {
            return;
        }
        let v = self.nodes[idx].vector.clone();
        let neighbors = self.nodes[idx].layers[layer].clone();

        let mut scored: Vec<(f32, usize)> = neighbors
            .into_iter()
            .map(|nb| (cosine(&v, &self.nodes[nb].vector), nb))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(m);

        self.nodes[idx].layers[layer] = scored.into_iter().map(|(_, nb)| nb).collect();
    }
}
