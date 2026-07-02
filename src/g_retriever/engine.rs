//! [`GRetrieverEngine`] — G-Retriever subgraph retrieval over a textual
//! knowledge graph.
//!
//! The engine turns a natural-language query plus a set of
//! [`GRetrieverEntity`] nodes and [`GRetrieverRelation`] edges into a
//! Prize-Collecting Steiner Tree instance — nodes carry query-relevance
//! *prizes*, edges carry *costs* — solves it with the
//! [`PcstSolver`](super::pcst::PcstSolver), and maps the resulting
//! [`PcstForest`](super::types::PcstForest) back into a
//! [`GRetrieverSubgraph`].
//!
//! Relevance prizes are computed from deterministic, dependency-free FNV-1a
//! character-trigram pseudo-embeddings (the same style used by the crate's
//! `semantic_router` and `eigenscore` modules), so a retrieval is fully
//! reproducible: identical inputs always yield an identical subgraph.

use std::collections::{HashMap, HashSet};

use super::pcst::PcstSolver;
use super::types::{
    GRetrieverConfig, GRetrieverEntity, GRetrieverError, GRetrieverRelation, GRetrieverResult,
    GRetrieverRootMode, GRetrieverSubgraph, PcstEdge, PcstNode, UnionFind,
};

/// Numerical tolerance for treating a prize/cost quantity as zero.
const EPS: f64 = 1e-9;

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;
/// Length (in `char`s) of the n-grams hashed by [`embed`].
const NGRAM_N: usize = 3;

// ── FNV-1a character-trigram pseudo-embeddings ────────────────────────────────

/// Deterministic `FNV-1a` 64-bit hash of a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Derive a deterministic L2-normalised pseudo-embedding for `text`.
///
/// The lowercased, whitespace-normalised text is split into overlapping
/// character trigrams (shorter texts fall back to a single gram over the whole
/// string); each gram is FNV-1a-hashed into one of `dim` buckets, and the
/// resulting histogram is L2-normalised. Texts with similar character content
/// hash into similar histograms, giving a cheap stand-in for a real sentence
/// embedding. Deterministic: identical input always produces an identical
/// vector.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let normalized: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let chars: Vec<char> = normalized.chars().collect();

    let mut buckets = vec![0.0_f64; dim];
    #[allow(clippy::cast_possible_truncation)]
    let bucket_of = |gram: &str| -> usize { (fnv1a(gram.as_bytes()) % dim as u64) as usize };

    if chars.is_empty() {
        return vec![0.0; dim];
    }
    if chars.len() < NGRAM_N {
        let gram: String = chars.iter().collect();
        buckets[bucket_of(&gram)] += 1.0;
    } else {
        for window in chars.windows(NGRAM_N) {
            let gram: String = window.iter().collect();
            buckets[bucket_of(&gram)] += 1.0;
        }
    }

    let norm: f64 = buckets
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm < 1e-10 {
        return vec![0.0; dim];
    }
    #[allow(clippy::cast_possible_truncation)]
    buckets.iter().map(|&value| (value / norm) as f32).collect()
}

/// Cosine similarity between two equal-length L2-normalised vectors, clamped to
/// `[0, 1]` (the histograms are nonnegative, so genuine similarities never fall
/// below zero; clamping only absorbs floating-point rounding).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

// ── GRetrieverEngine ──────────────────────────────────────────────────────────

/// The G-Retriever subgraph-retrieval engine.
///
/// The engine is **pure compute** — no I/O, no async, fully deterministic. It
/// scores entity relevance with FNV-1a pseudo-embeddings, frames subgraph
/// selection as a Prize-Collecting Steiner Tree, and returns the
/// prize-maximising [`GRetrieverSubgraph`].
#[derive(Debug, Clone, Default)]
pub struct GRetrieverEngine {
    /// The configuration for this engine.
    pub config: GRetrieverConfig,
}

impl GRetrieverEngine {
    /// Create a new engine with the supplied configuration.
    #[must_use]
    pub fn new(config: GRetrieverConfig) -> Self {
        Self { config }
    }

    /// Score one text's relevance to the query embedding, applying the prize
    /// floor and scale from the configuration.
    fn prize_of(&self, query_vec: &[f32], text: &str) -> f64 {
        let vector = embed(text, self.config.embed_dim);
        let similarity = f64::from(cosine(query_vec, &vector));
        if similarity < self.config.prize_floor {
            0.0
        } else {
            similarity * self.config.prize_scale
        }
    }

    /// Retrieve the query-relevant subgraph of the knowledge graph given by
    /// `entities` and `relations`.
    ///
    /// Each entity is assigned a prize (its query relevance), each relation a
    /// cost; the connected subgraph maximising prize minus cost is found by the
    /// PCST solver and returned.
    ///
    /// # Errors
    ///
    /// - [`GRetrieverError::InvalidConfig`] when the configuration is invalid.
    /// - [`GRetrieverError::EmptyQuery`] when `query` is blank.
    /// - [`GRetrieverError::EmptyGraph`] when `entities` is empty.
    /// - [`GRetrieverError::DanglingRelation`] when a relation references an
    ///   unknown entity id.
    /// - [`GRetrieverError::InvalidGraph`] when a relation carries a
    ///   non-positive weight.
    /// - [`GRetrieverError::RootNotFound`] when a rooted retrieval names an
    ///   absent entity.
    /// - [`GRetrieverError::NoRelevantNode`] when, in unrooted mode, nothing
    ///   scored above the prize floor.
    #[allow(clippy::too_many_lines)]
    pub fn retrieve(
        &self,
        query: &str,
        entities: &[GRetrieverEntity],
        relations: &[GRetrieverRelation],
    ) -> GRetrieverResult<GRetrieverSubgraph> {
        self.config.validate()?;
        if query.trim().is_empty() {
            return Err(GRetrieverError::EmptyQuery);
        }
        if entities.is_empty() {
            return Err(GRetrieverError::EmptyGraph);
        }

        let query_vec = embed(query, self.config.embed_dim);
        let n_real = entities.len();

        // Node prizes for the real entities.
        let node_prize: Vec<f64> = entities
            .iter()
            .map(|entity| self.prize_of(&query_vec, &entity.text))
            .collect();

        // Entity id -> index.
        let mut id_to_index: HashMap<&str, usize> = HashMap::with_capacity(n_real);
        for (index, entity) in entities.iter().enumerate() {
            id_to_index.insert(entity.id.as_str(), index);
        }

        // Build the abstract PCST view.
        let mut solver_nodes: Vec<PcstNode> = (0..n_real)
            .map(|index| PcstNode::new(index, node_prize[index]))
            .collect();
        let mut solver_edges: Vec<PcstEdge> = Vec::new();
        let mut relation_endpoints: Vec<(usize, usize)> = Vec::with_capacity(relations.len());
        let mut relation_cost: Vec<f64> = Vec::with_capacity(relations.len());
        let mut relation_prize: Vec<f64> = Vec::with_capacity(relations.len());
        let mut relation_solver_edges: Vec<Vec<usize>> = Vec::with_capacity(relations.len());

        for (index, relation) in relations.iter().enumerate() {
            let source = *id_to_index
                .get(relation.source_id.as_str())
                .ok_or_else(|| GRetrieverError::DanglingRelation {
                    index,
                    endpoint: relation.source_id.clone(),
                })?;
            let target = *id_to_index
                .get(relation.target_id.as_str())
                .ok_or_else(|| GRetrieverError::DanglingRelation {
                    index,
                    endpoint: relation.target_id.clone(),
                })?;
            let weight = relation.weight.unwrap_or(1.0);
            if !weight.is_finite() || weight <= 0.0 {
                return Err(GRetrieverError::InvalidGraph {
                    reason: format!("relation {index} has a non-positive weight {weight}"),
                });
            }
            let cost = self.config.edge_cost * self.config.edge_cost_scale * weight;
            let edge_prize = if self.config.edge_prizes {
                self.prize_of(&query_vec, &relation.label)
            } else {
                0.0
            };

            relation_endpoints.push((source, target));
            relation_cost.push(cost);
            relation_prize.push(edge_prize);

            if self.config.edge_prizes && edge_prize > EPS {
                // Reduce an edge prize to a node prize via a virtual midpoint.
                let virtual_index = solver_nodes.len();
                solver_nodes.push(PcstNode::new(virtual_index, edge_prize));
                let half = cost / 2.0;
                let first = solver_edges.len();
                solver_edges.push(PcstEdge::new(source, virtual_index, half));
                let second = solver_edges.len();
                solver_edges.push(PcstEdge::new(virtual_index, target, half));
                relation_solver_edges.push(vec![first, second]);
            } else {
                let edge_index = solver_edges.len();
                solver_edges.push(PcstEdge::new(source, target, cost));
                relation_solver_edges.push(vec![edge_index]);
            }
        }

        // Resolve the root (if any).
        let root = match &self.config.root_mode {
            GRetrieverRootMode::Unrooted => None,
            GRetrieverRootMode::Rooted { entity_id } => {
                let index = id_to_index
                    .get(entity_id.as_str())
                    .copied()
                    .ok_or_else(|| GRetrieverError::RootNotFound {
                        entity_id: entity_id.clone(),
                    })?;
                Some(index)
            }
        };

        // Relevance guard (unrooted only): something must be relevant.
        if root.is_none() {
            let any_node = node_prize.iter().any(|&prize| prize > EPS);
            let any_edge = relation_prize.iter().any(|&prize| prize > EPS);
            if !any_node && !any_edge {
                return Err(GRetrieverError::NoRelevantNode);
            }
        }

        // Solve.
        let solver =
            PcstSolver::new(self.config.prune, self.config.single_component).with_root(root);
        let forest = solver.solve(&solver_nodes, &solver_edges)?;

        // ── Map the forest back onto real entities and relations. ─────────
        let kept_edges: HashSet<usize> = forest.edge_indices.iter().copied().collect();

        let mut included_relation: Vec<bool> = vec![false; relations.len()];
        for (index, edges) in relation_solver_edges.iter().enumerate() {
            if !edges.is_empty() && edges.iter().all(|edge| kept_edges.contains(edge)) {
                included_relation[index] = true;
            }
        }

        let mut kept_real: Vec<usize> = forest
            .node_indices
            .iter()
            .copied()
            .filter(|&index| index < n_real)
            .collect();
        for (index, &included) in included_relation.iter().enumerate() {
            if included {
                let (source, target) = relation_endpoints[index];
                kept_real.push(source);
                kept_real.push(target);
            }
        }
        kept_real.sort_unstable();
        kept_real.dedup();

        // ── Optional maximum-size cap by trimming lowest-prize leaves. ────
        if let Some(cap) = self.config.max_subgraph_size {
            apply_size_cap(
                cap,
                &mut kept_real,
                &mut included_relation,
                &relation_endpoints,
                &node_prize,
            );
        }

        // ── Summaries. ────────────────────────────────────────────────────
        let kept_set: HashSet<usize> = kept_real.iter().copied().collect();
        let total_node_prize: f64 = kept_real.iter().map(|&index| node_prize[index]).sum();
        let mut total_edge_prize = 0.0_f64;
        let mut total_cost = 0.0_f64;
        let mut edges_for_components: Vec<(usize, usize)> = Vec::new();
        for index in 0..included_relation.len() {
            if !included_relation[index] {
                continue;
            }
            let (source, target) = relation_endpoints[index];
            if kept_set.contains(&source) && kept_set.contains(&target) {
                total_edge_prize += relation_prize[index];
                total_cost += relation_cost[index];
                edges_for_components.push((source, target));
            } else {
                included_relation[index] = false;
            }
        }
        let num_components = count_components(&kept_real, &edges_for_components);

        let entities_out: Vec<GRetrieverEntity> = kept_real
            .iter()
            .map(|&index| entities[index].clone())
            .collect();
        let node_prizes: Vec<f64> = kept_real.iter().map(|&index| node_prize[index]).collect();
        let relations_out: Vec<GRetrieverRelation> = included_relation
            .iter()
            .enumerate()
            .filter(|&(_, &included)| included)
            .map(|(index, _)| relations[index].clone())
            .collect();

        Ok(GRetrieverSubgraph {
            entities: entities_out,
            node_prizes,
            relations: relations_out,
            total_prize: total_node_prize + total_edge_prize,
            total_cost,
            num_components,
        })
    }
}

// ── mapping helpers ───────────────────────────────────────────────────────────

/// Trim the retrieved subgraph down to at most `cap` entities by repeatedly
/// removing the lowest-prize leaf (a node of degree at most one), which
/// preserves the connectivity of what remains.
fn apply_size_cap(
    cap: usize,
    kept_real: &mut Vec<usize>,
    included_relation: &mut [bool],
    relation_endpoints: &[(usize, usize)],
    node_prize: &[f64],
) {
    let mut alive_nodes: HashSet<usize> = kept_real.iter().copied().collect();
    let mut alive_edges: Vec<usize> = included_relation
        .iter()
        .enumerate()
        .filter_map(|(index, &included)| included.then_some(index))
        .filter(|&index| {
            let (source, target) = relation_endpoints[index];
            alive_nodes.contains(&source) && alive_nodes.contains(&target)
        })
        .collect();

    while alive_nodes.len() > cap {
        // Degree of each alive node over the alive edges.
        let mut degree: HashMap<usize, usize> =
            alive_nodes.iter().map(|&node| (node, 0usize)).collect();
        for &index in &alive_edges {
            let (source, target) = relation_endpoints[index];
            if let Some(count) = degree.get_mut(&source) {
                *count += 1;
            }
            if let Some(count) = degree.get_mut(&target) {
                *count += 1;
            }
        }

        // The removable leaf with the smallest prize (smallest id breaks ties).
        let mut victim: Option<(f64, usize)> = None;
        for &node in &alive_nodes {
            if degree.get(&node).copied().unwrap_or(0) > 1 {
                continue;
            }
            let prize = node_prize.get(node).copied().unwrap_or(0.0);
            let replace = match victim {
                None => true,
                Some((best_prize, best_id)) => {
                    prize < best_prize - EPS
                        || ((prize - best_prize).abs() <= EPS && node < best_id)
                }
            };
            if replace {
                victim = Some((prize, node));
            }
        }

        let Some((_, node)) = victim else {
            break;
        };
        alive_nodes.remove(&node);
        alive_edges.retain(|&index| {
            let (source, target) = relation_endpoints[index];
            source != node && target != node
        });
    }

    let alive_edge_set: HashSet<usize> = alive_edges.into_iter().collect();
    for (index, included) in included_relation.iter_mut().enumerate() {
        *included = *included && alive_edge_set.contains(&index);
    }
    let mut remaining: Vec<usize> = alive_nodes.into_iter().collect();
    remaining.sort_unstable();
    *kept_real = remaining;
}

/// Count the connected components induced by `edges` over the node set `kept`.
fn count_components(kept: &[usize], edges: &[(usize, usize)]) -> usize {
    if kept.is_empty() {
        return 0;
    }
    let position: HashMap<usize, usize> = kept
        .iter()
        .enumerate()
        .map(|(pos, &node)| (node, pos))
        .collect();
    let mut union = UnionFind::new(kept.len());
    for &(source, target) in edges {
        if let (Some(&ps), Some(&pt)) = (position.get(&source), position.get(&target)) {
            union.union(ps, pt);
        }
    }
    let mut roots: HashSet<usize> = HashSet::new();
    for pos in 0..kept.len() {
        roots.insert(union.find(pos));
    }
    roots.len()
}
