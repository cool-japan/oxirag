//! Lexical pseudo-embeddings and deterministic clustering for speculative drafting.
//!
//! Documents are embedded with a deterministic FNV-1a hashing trick and then
//! partitioned into at most `k` diverse groups by a lightweight k-means variant
//! seeded by spreading the initial centroids evenly across the corpus.

// ── Lexical pseudo-embedding ──────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise (non-alphanumeric split, length `>= 2`, lowercased) →
/// hash each token to a bucket with FNV-1a → accumulate per-bucket counts →
/// L2-normalise into a vector of length `dim`. Returns an empty vector when
/// `dim == 0`.
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
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity of two equal-length vectors (dot product for L2-normalised
/// inputs). Returns `0.0` for mismatched or empty vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── KMeans-lite clustering ────────────────────────────────────────────────────

/// Partition `embeddings` into at most `k` clusters with deterministic k-means.
///
/// Returns `Vec<Vec<usize>>` where each inner vector holds the original indices
/// of one cluster. The number of clusters is `min(k, n)`; the result contains
/// only non-empty clusters and is sorted by each cluster's smallest member
/// index, so the output is a deterministic partition of `0..n`.
///
/// # Determinism
///
/// Initial centroids are seeded by *spreading* — picking items at evenly spaced
/// offsets `i * (n / k)` across the corpus — and assignment ties always resolve
/// to the lowest centroid index, so the partition depends only on the inputs.
#[must_use]
pub fn kmeans_lite(embeddings: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.max(1).min(n);
    let dim = embeddings[0].len();

    // Initial centroids: evenly-spaced ("spread") items for deterministic seeding.
    let step = (n / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|ci| embeddings[(ci * step).min(n - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..16 {
        let mut changed = false;
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best_ci = 0;
            let mut best_sim = f32::NEG_INFINITY;
            for (ci, centroid) in centroids.iter().enumerate() {
                let sim = cosine(emb, centroid);
                // Strictly-greater keeps the lowest index on ties: deterministic.
                if sim > best_sim {
                    best_sim = sim;
                    best_ci = ci;
                }
            }
            if assignments[i] != best_ci {
                assignments[i] = best_ci;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Recompute centroids as the L2-normalised mean of each cluster.
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
                let norm: f32 = new_centroids[ci].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in &mut new_centroids[ci] {
                        *x /= norm;
                    }
                }
            } else {
                // Empty centroid: retain the previous one to stay deterministic.
                centroids[ci].clone_into(&mut new_centroids[ci]);
            }
        }
        centroids = new_centroids;
    }

    // Build cluster groups, dropping empties and sorting by smallest index.
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &ci) in assignments.iter().enumerate() {
        groups[ci].push(i);
    }
    let mut result: Vec<Vec<usize>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    result.sort_by_key(|g| g[0]);
    result
}
