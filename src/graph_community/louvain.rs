//! Greedy Louvain community detection.
//!
//! Implements a simplified Louvain algorithm using single-pass greedy
//! modularity optimization without random shuffling (deterministic output,
//! suitable for deterministic tests).

use std::collections::HashMap;

#[cfg(feature = "graphrag")]
use crate::layer4_graph::types::{EntityId, GraphEntity, GraphRelationship};

#[cfg(feature = "graphrag")]
use super::types::{
    Community, CommunityDetector, CommunityGraph, CommunityId, GraphCommunityConfig,
    GraphCommunityError,
};

// ── adjacency helpers ─────────────────────────────────────────────────────────

/// Build a symmetric adjacency map: `node_idx → Vec<(neighbor_idx, weight)>`.
#[cfg(feature = "graphrag")]
fn build_adjacency(
    n: usize,
    entity_index: &HashMap<&str, usize>,
    relationships: &[GraphRelationship],
) -> Vec<Vec<(usize, f64)>> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for rel in relationships {
        let src: &str = &rel.source_id;
        let tgt: &str = &rel.target_id;
        if let (Some(&si), Some(&ti)) = (entity_index.get(src), entity_index.get(tgt))
            && si != ti
        {
            let w = f64::from(rel.confidence);
            adj[si].push((ti, w));
            adj[ti].push((si, w));
        }
    }
    adj
}

/// Total edge weight (sum of all relationship weights / 2 for undirected).
#[cfg(feature = "graphrag")]
fn total_weight(adj: &[Vec<(usize, f64)>]) -> f64 {
    adj.iter()
        .flat_map(|nbrs| nbrs.iter().map(|(_, w)| w))
        .sum::<f64>()
        / 2.0
}

/// Node degree (sum of incident edge weights).
#[cfg(feature = "graphrag")]
fn degree(adj: &[Vec<(usize, f64)>], node: usize) -> f64 {
    adj[node].iter().map(|(_, w)| w).sum()
}

/// Compute graph modularity Q given community assignments.
#[cfg(feature = "graphrag")]
fn modularity(adj: &[Vec<(usize, f64)>], assignments: &[usize], m: f64, resolution: f64) -> f64 {
    if m == 0.0 {
        return 0.0;
    }
    let n = adj.len();
    let mut q = 0.0f64;
    for i in 0..n {
        let ki = degree(adj, i);
        for (j, w) in &adj[i] {
            let kj = degree(adj, *j);
            if assignments[i] == assignments[*j] {
                q += w - resolution * ki * kj / (2.0 * m);
            }
        }
    }
    q / (2.0 * m)
}

// ── LouvainDetector ───────────────────────────────────────────────────────────

/// Greedy Louvain community detector.
///
/// The algorithm is a deterministic single-pass greedy optimization:
/// each node is moved to the neighbour community that maximises the modularity
/// gain, iterating until no improvement is found (or `max_levels` are reached).
#[cfg(feature = "graphrag")]
#[derive(Debug, Clone, Default)]
pub struct LouvainDetector;

#[cfg(feature = "graphrag")]
impl LouvainDetector {
    /// Create a new [`LouvainDetector`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "graphrag")]
impl CommunityDetector for LouvainDetector {
    fn detect(
        &self,
        entities: &[GraphEntity],
        relationships: &[GraphRelationship],
        config: &GraphCommunityConfig,
    ) -> Result<CommunityGraph, GraphCommunityError> {
        if entities.is_empty() {
            return Err(GraphCommunityError::EmptyGraph);
        }

        let n = entities.len();

        // Build index: entity_id → position
        let entity_index: HashMap<&str, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id.as_str(), i))
            .collect();
        let adj = build_adjacency(n, &entity_index, relationships);
        let m = total_weight(&adj);

        // Initial assignment: each node in its own community
        let mut assignments: Vec<usize> = (0..n).collect();

        // Greedy single-pass Louvain (up to max_levels sweeps)
        let mut level = 0;
        loop {
            let mut improved = false;

            for i in 0..n {
                let current_comm = assignments[i];

                // Temporarily remove i from its community and try each neighbour's community
                let mut best_comm = current_comm;
                let mut best_gain = 0.0f64;

                let ki = degree(&adj, i);

                // Community weight sums
                let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                for (j, w) in &adj[i] {
                    *comm_weights.entry(assignments[*j]).or_insert(0.0) += w;
                }

                for (&cand_comm, &kc_in) in &comm_weights {
                    if cand_comm == current_comm {
                        continue;
                    }
                    // Sum of degrees in candidate community
                    let sigma_c: f64 = assignments
                        .iter()
                        .enumerate()
                        .filter(|&(j, c)| *c == cand_comm && j != i)
                        .map(|(j, _)| degree(&adj, j))
                        .sum();

                    if m > 0.0 {
                        let gain = kc_in / m - config.resolution * ki * sigma_c / (2.0 * m * m);
                        if gain > best_gain {
                            best_gain = gain;
                            best_comm = cand_comm;
                        }
                    }
                }

                if best_comm != current_comm {
                    assignments[i] = best_comm;
                    improved = true;
                }
            }

            level += 1;
            if !improved || level >= config.max_levels {
                break;
            }
        }

        // Normalise community ids to be dense [0..K)
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for &c in &assignments {
            let next = remap.len();
            remap.entry(c).or_insert(next);
        }
        let assignments: Vec<usize> = assignments.iter().map(|c| remap[c]).collect();

        // Build Community objects
        let num_communities = remap.len();
        let mut member_lists: Vec<Vec<EntityId>> = vec![Vec::new(); num_communities];
        for (i, &comm) in assignments.iter().enumerate() {
            member_lists[comm].push(entities[i].id.clone());
        }

        let communities: Vec<Community> = member_lists
            .into_iter()
            .enumerate()
            .filter(|(_, members)| !members.is_empty())
            .map(|(id, members)| Community {
                id: CommunityId::new(id),
                members,
                level: 0,
            })
            .collect();

        let q = if m > 0.0 {
            modularity(&adj, &assignments, m, config.resolution)
        } else {
            0.0
        };

        Ok(CommunityGraph {
            communities,
            modularity: q,
        })
    }
}
