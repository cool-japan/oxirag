//! Centroid tree construction via recursive KMeans-lite clustering.

use super::types::cosine;

// ── TreeNode ──────────────────────────────────────────────────────────────────

/// A single node of the semantic centroid tree.
///
/// Internal nodes carry child node ids; leaf nodes carry the indices of the
/// documents that fall beneath them. Every node stores the L2-normalised mean
/// embedding (centroid) of all documents in its subtree, used to steer beam
/// traversal during retrieval.
#[derive(Debug, Clone)]
pub(crate) struct TreeNode {
    /// L2-normalised mean embedding of every document in this subtree.
    pub centroid: Vec<f32>,
    /// Child node ids (empty for leaves).
    pub children: Vec<usize>,
    /// Document indices held by this leaf (empty for internal nodes).
    pub doc_indices: Vec<usize>,
}

impl TreeNode {
    /// Return `true` when this node holds documents directly (a leaf).
    pub(crate) fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ── Centroid helper ───────────────────────────────────────────────────────────

/// Compute the L2-normalised mean of the embeddings selected by `indices`.
fn mean_centroid(embeddings: &[Vec<f32>], indices: &[usize], dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; dim];
    if indices.is_empty() {
        return acc;
    }
    for &i in indices {
        for (d, value) in embeddings[i].iter().enumerate() {
            acc[d] += value;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = indices.len() as f32;
    for x in &mut acc {
        *x /= count;
    }
    let norm: f32 = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut acc {
            *x /= norm;
        }
    }
    acc
}

// ── KMeans-lite clustering ────────────────────────────────────────────────────

/// Partition `members` into at most `k` clusters with deterministic KMeans-lite.
///
/// Initial centroids are evenly spaced members; assignment uses cosine
/// similarity; centroids are recomputed as the normalised member mean for at
/// most ten iterations. Returns groups of *positions into `members`* (not raw
/// document indices), each sorted, ordered by their smallest member position so
/// the resulting cluster indices are stable across runs.
fn kmeans_lite(
    embeddings: &[Vec<f32>],
    members: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<usize>> {
    let n = members.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.max(1).min(n);
    if k == 1 {
        return vec![(0..n).collect()];
    }

    // Initial centroids: evenly-spaced members.
    let step = n / k;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|ci| embeddings[members[(ci * step).min(n - 1)]].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..10 {
        let mut changed = false;
        for (pos, &member) in members.iter().enumerate() {
            let emb = &embeddings[member];
            let mut best_ci = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (ci, centroid) in centroids.iter().enumerate() {
                let sim = cosine(emb, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_ci = ci;
                }
            }
            if assignments[pos] != best_ci {
                assignments[pos] = best_ci;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Recompute centroids as the normalised mean of assigned members.
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (pos, &ci) in assignments.iter().enumerate() {
            let member = members[pos];
            for (d, value) in embeddings[member].iter().enumerate() {
                new_centroids[ci][d] += value;
            }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let cnt = counts[ci] as f32;
                for x in &mut new_centroids[ci] {
                    *x /= cnt;
                }
                let norm: f32 = new_centroids[ci].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut new_centroids[ci] {
                        *x /= norm;
                    }
                }
            } else {
                // Keep the previous centroid for an emptied cluster.
                centroids[ci].clone_into(&mut new_centroids[ci]);
            }
        }
        centroids = new_centroids;
    }

    // Collect non-empty groups in assignment order, then order them by their
    // smallest member position for deterministic cluster indices.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (pos, &ci) in assignments.iter().enumerate() {
        groups[ci].push(pos);
    }
    let mut result: Vec<Vec<usize>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    result.sort_by_key(|g| g[0]);
    result
}

// ── Recursive builder ─────────────────────────────────────────────────────────

/// State threaded through the recursive tree build.
pub(crate) struct BuildOutput {
    /// All tree nodes; the root is always node `0` when non-empty.
    pub nodes: Vec<TreeNode>,
    /// Per-document cluster-index path (parallel to the input documents).
    pub paths: Vec<Vec<usize>>,
}

/// Build the centroid tree from precomputed document `embeddings`.
///
/// Recursively clusters the documents into at most `branching` groups per level
/// until `max_depth` is reached or a node can no longer be subdivided, assigning
/// each document the path of cluster indices from root to its leaf. The root
/// always performs at least one clustering level, so every document receives a
/// non-empty path whenever `max_depth >= 1`.
pub(crate) fn build_tree(
    embeddings: &[Vec<f32>],
    branching: usize,
    max_depth: usize,
    dim: usize,
) -> BuildOutput {
    let doc_count = embeddings.len();
    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut paths: Vec<Vec<usize>> = vec![Vec::new(); doc_count];
    if doc_count == 0 {
        return BuildOutput { nodes, paths };
    }
    let all: Vec<usize> = (0..doc_count).collect();
    build_node(
        embeddings,
        &all,
        0,
        branching.max(1),
        max_depth,
        dim,
        &mut nodes,
        &mut paths,
    );
    BuildOutput { nodes, paths }
}

/// Recursively build a node over `members`, returning its node id.
#[allow(clippy::too_many_arguments)]
fn build_node(
    embeddings: &[Vec<f32>],
    members: &[usize],
    depth: usize,
    branching: usize,
    max_depth: usize,
    dim: usize,
    nodes: &mut Vec<TreeNode>,
    paths: &mut [Vec<usize>],
) -> usize {
    let centroid = mean_centroid(embeddings, members, dim);

    // Reserve this node's slot up front so children receive larger ids.
    let node_id = nodes.len();
    nodes.push(TreeNode {
        centroid,
        children: Vec::new(),
        doc_indices: Vec::new(),
    });

    // Stop subdividing at the depth limit.
    if depth >= max_depth {
        nodes[node_id].doc_indices = members.to_vec();
        return node_id;
    }

    let groups = kmeans_lite(embeddings, members, branching, dim);

    // Past the root, a single all-encompassing group means no further structure
    // can be extracted, so terminate here as a leaf.
    if groups.len() <= 1 && depth > 0 {
        nodes[node_id].doc_indices = members.to_vec();
        return node_id;
    }

    let mut children = Vec::with_capacity(groups.len());
    for (cluster_index, group) in groups.iter().enumerate() {
        let group_members: Vec<usize> = group.iter().map(|&pos| members[pos]).collect();
        for &member in &group_members {
            paths[member].push(cluster_index);
        }
        let child_id = build_node(
            embeddings,
            &group_members,
            depth + 1,
            branching,
            max_depth,
            dim,
            nodes,
            paths,
        );
        children.push(child_id);
    }
    nodes[node_id].children = children;
    node_id
}
