//! Lexical pseudo-embeddings and clustering for RAPTOR.

use std::collections::HashMap;

use super::types::ClusterStrategy;

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise → hash each token to a bucket index → accumulate
/// per-bucket count → L2-normalise to the requested `dim`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    // L2 normalise
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // Both are L2-normalised, so dot ≈ cosine similarity
    dot
}

// ── Agglomerative clustering ──────────────────────────────────────────────────

/// Single-linkage agglomerative clustering that groups items into clusters of
/// roughly `target_size` by greedily merging the closest pair.
///
/// Returns `Vec<Vec<usize>>` — each inner vec is the indices of one cluster.
#[must_use]
pub fn agglomerative(embeddings: &[Vec<f32>], target_size: usize) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let effective_target = target_size.max(1);
    // Start with each item in its own cluster
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > 1 {
        let k = clusters.len();
        let target_k = n.div_ceil(effective_target);
        if clusters.len() <= target_k {
            break;
        }

        // Find most-similar pair of clusters (representative = first element)
        let mut best_sim = -1.0f32;
        let mut best_i = 0;
        let mut best_j = 1;

        for i in 0..k {
            let rep_i = &embeddings[clusters[i][0]];
            for j in (i + 1)..k {
                let rep_j = &embeddings[clusters[j][0]];
                let sim = cosine(rep_i, rep_j);
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge j into i
        let merged_j = clusters.remove(best_j);
        clusters[best_i].extend(merged_j);
    }

    clusters
}

// ── KMeansLite clustering ─────────────────────────────────────────────────────

/// Lightweight k-means clustering.
///
/// k = `ceil(n / target_size)`; initial centroids = evenly spaced items.
/// Runs at most 10 iterations.
#[must_use]
pub fn kmeans_lite(
    embeddings: &[Vec<f32>],
    target_size: usize,
    strategy: ClusterStrategy,
) -> Vec<Vec<usize>> {
    if let ClusterStrategy::Agglomerative = strategy {
        return agglomerative(embeddings, target_size);
    }

    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let effective_target = target_size.max(1);
    let k = n.div_ceil(effective_target).max(1).min(n);
    let dim = embeddings[0].len();

    // Initial centroids: evenly-spaced items
    let step = n / k;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|ci| embeddings[(ci * step).min(n - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..10 {
        // Assign
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let best = (0..k)
                .map(|ci| (cosine(emb, &centroids[ci]), ci))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(_, ci)| ci);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &ci) in assignments.iter().enumerate() {
            for d in 0..dim {
                new_centroids[ci][d] += embeddings[i][d];
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
                // L2 normalise centroid
                let norm: f32 = new_centroids[ci].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut new_centroids[ci] {
                        *x /= norm;
                    }
                }
            } else {
                // Empty centroid: reassign to the original
                centroids[ci].clone_into(&mut new_centroids[ci]);
            }
        }
        centroids = new_centroids;
    }

    // Build cluster groups
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &ci) in assignments.iter().enumerate() {
        groups.entry(ci).or_default().push(i);
    }
    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    result.sort_by_key(|g| g[0]);
    result
}

/// Dispatch clustering based on `strategy`.
#[must_use]
pub fn cluster(
    embeddings: &[Vec<f32>],
    target_size: usize,
    strategy: ClusterStrategy,
) -> Vec<Vec<usize>> {
    match strategy {
        ClusterStrategy::Agglomerative => agglomerative(embeddings, target_size),
        ClusterStrategy::KMeansLite => kmeans_lite(embeddings, target_size, strategy),
    }
}
