//! Deterministic balanced k-means, boundary-closure replication and the
//! RNG-rule pruning that together implement SPANN's clustering stage.
//!
//! # Pipeline
//!
//! 1. [`balanced_kmeans`] partitions every indexed vector into
//!    `num_postings` centroids using deterministic Lloyd k-means, then
//!    recursively re-clusters (into two) any cluster whose primary
//!    membership exceeds `posting_limit`, so no balanced-clustering
//!    assignment is oversized.
//! 2. [`boundary_replicas`] decides, independently for every point, which
//!    centroids (beyond its nearest one) it should also be replicated into
//!    — every centroid within `(1 + epsilon)` of the nearest distance,
//!    capped at `replica_count` — and then removes redundant replicas using
//!    the relative-neighbourhood-graph (RNG) rule: a candidate centroid is
//!    dropped when an already-accepted, closer centroid lies "between" the
//!    point and the candidate.
//!
//! No `rand` crate is used anywhere: k-means seeding is derived from
//! deterministic FNV-1a hashing of each vector's own bit pattern combined
//! with its position, so [`SpannIndex::build`](crate::spann::SpannIndex::build)
//! is perfectly reproducible.

use std::cmp::Ordering;
use std::collections::VecDeque;

use super::types::SpannMetric;

// ── FNV-1a seeding ────────────────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a 64-bit hash of raw bytes.
///
/// Used only to derive a deterministic pseudo-random ordering for k-means
/// seed selection — never for cryptographic purposes. Because the same
/// input bytes always hash to the same value, no external `rand`
/// dependency is required to make index construction reproducible.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministic FNV-seeded ordering key for the vector at global index
/// `gi`.
///
/// Combines the vector's own bit pattern with its index so that seeding is
/// stable across repeated `build` calls on identical input, while still
/// varying with vector content (unlike a plain "evenly spaced by input
/// order" seed).
fn fnv_seed_key(vector: &[f32], gi: usize) -> u64 {
    let mut bytes = Vec::with_capacity(8 + vector.len() * 4);
    bytes.extend_from_slice(&(gi as u64).to_le_bytes());
    for x in vector {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    fnv1a(&bytes)
}

// ── Metric primitives ─────────────────────────────────────────────────────────

/// Cosine similarity in `[-1.0, 1.0]`. Returns `0.0` when either vector has
/// (near-)zero norm, avoiding division by zero.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a <= 1e-12 || norm_b <= 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Euclidean (L2) distance between two equal-length vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Raw dot product between two equal-length vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Distance between `a` and `b` under `metric`, where a **smaller** value
/// always means "closer". This shared convention lets clustering, boundary
/// closure and RNG pruning use one implementation across every metric.
pub(super) fn metric_distance(a: &[f32], b: &[f32], metric: SpannMetric) -> f32 {
    match metric {
        SpannMetric::L2 => l2_distance(a, b),
        SpannMetric::Cosine => 1.0 - cosine_similarity(a, b),
        SpannMetric::Dot => -dot_product(a, b),
    }
}

/// Similarity score between `a` and `b` under `metric`, where a **larger**
/// value always means "more similar". Used only for the final ranked
/// [`SpannHit`](super::types::SpannHit) scores reported by search.
pub(super) fn metric_score(a: &[f32], b: &[f32], metric: SpannMetric) -> f32 {
    match metric {
        SpannMetric::Cosine => cosine_similarity(a, b),
        SpannMetric::L2 => 1.0 / (1.0 + l2_distance(a, b)),
        SpannMetric::Dot => dot_product(a, b),
    }
}

/// Index of the centroid nearest to `vector` by [`metric_distance`]. Ties
/// are broken by the lower centroid index for determinism.
fn nearest_index(vector: &[f32], centroids: &[Vec<f32>], metric: SpannMetric) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (ci, c) in centroids.iter().enumerate() {
        let d = metric_distance(vector, c, metric);
        if d < best_dist {
            best_dist = d;
            best = ci;
        }
    }
    best
}

// ── Deterministic k-means over a subset of global indices ────────────────────

/// Pick `k` FNV-seeded, deterministically spread seed global-indices out of
/// `indices`.
///
/// Every candidate index is keyed by [`fnv_seed_key`] (content + position),
/// the candidates are sorted by that key, and `k` seeds are then chosen at
/// evenly spaced positions across the *hash-sorted* order. This differs
/// from a plain "evenly spaced across input order" seed (as used elsewhere
/// in this crate for coarse IVF quantizers) by first re-ordering candidates
/// through a deterministic hash, giving a reproducible pseudo-random-like
/// spread without any `rand` dependency.
fn seed_global_indices(vectors: &[Vec<f32>], indices: &[usize], k: usize) -> Vec<usize> {
    let n = indices.len();
    let mut keyed: Vec<(u64, usize)> = indices
        .iter()
        .map(|&gi| (fnv_seed_key(&vectors[gi], gi), gi))
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    (0..k)
        .map(|ci| {
            let pos = if k <= 1 { 0 } else { (ci * (n - 1)) / (k - 1) };
            keyed[pos.min(n - 1)].1
        })
        .collect()
}

/// Run deterministic Lloyd k-means restricted to the vectors named by
/// `indices` (global indices into `vectors`), producing up to `k` centroids.
///
/// Returns `(centroids, groups)` where `groups[ci]` holds the global
/// indices assigned to `centroids[ci]`. `k` is clamped to
/// `[1, indices.len()]`; empty clusters retain their previous centroid so
/// that repeated runs on identical input are always identical.
fn kmeans_subset(
    vectors: &[Vec<f32>],
    indices: &[usize],
    k: usize,
    iters: usize,
    metric: SpannMetric,
) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let n = indices.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let dim = vectors[indices[0]].len();
    let k = k.max(1).min(n);

    let seeds = seed_global_indices(vectors, indices, k);
    let mut centroids: Vec<Vec<f32>> = seeds.iter().map(|&gi| vectors[gi].clone()).collect();

    let mut assignment = vec![0usize; n];
    for _ in 0..iters.max(1) {
        let mut changed = false;
        for (local_i, &gi) in indices.iter().enumerate() {
            let best = nearest_index(&vectors[gi], &centroids, metric);
            if assignment[local_i] != best {
                assignment[local_i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (local_i, &gi) in indices.iter().enumerate() {
            let ci = assignment[local_i];
            for d in 0..dim {
                sums[ci][d] += vectors[gi][d];
            }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                #[allow(clippy::cast_precision_loss)]
                let cnt = counts[ci] as f32;
                for x in &mut sums[ci] {
                    *x /= cnt;
                }
                centroids[ci].clone_from(&sums[ci]);
            }
            // Empty clusters keep their previous centroid (deterministic).
        }
    }

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (local_i, &gi) in indices.iter().enumerate() {
        groups[assignment[local_i]].push(gi);
    }

    (centroids, groups)
}

// ── Balanced clustering (posting-limit enforcement) ──────────────────────────

/// The result of [`balanced_kmeans`]: final centroids and, in parallel, the
/// global indices assigned as *primary* members of each centroid.
pub(super) struct BalancedClusters {
    /// Final centroid vectors, one per posting list.
    pub centroids: Vec<Vec<f32>>,
    /// `members[ci]` holds the global indices primarily assigned to
    /// `centroids[ci]`, i.e. before boundary-closure replication.
    ///
    /// Retained on the result for balance introspection and tests even
    /// though [`SpannIndex::build`](crate::spann::SpannIndex::build) only
    /// consumes `centroids` (it re-derives full boundary-closure membership
    /// independently via [`boundary_replicas`]).
    #[allow(dead_code)]
    pub members: Vec<Vec<usize>>,
}

/// Deterministic balanced k-means: cluster `vectors` into up to
/// `num_postings` centroids, then recursively re-cluster (into two) any
/// cluster whose primary membership exceeds `posting_limit`, so that no
/// balanced-clustering assignment is oversized.
///
/// A cluster is left oversized only when it is *impossible* to split
/// further (e.g. every remaining member is an identical duplicate vector,
/// so k-means collapses both children onto one side) — this avoids an
/// infinite splitting loop while never fabricating an artificial split.
pub(super) fn balanced_kmeans(
    vectors: &[Vec<f32>],
    num_postings: usize,
    posting_limit: usize,
    max_iters: usize,
    metric: SpannMetric,
) -> BalancedClusters {
    let n = vectors.len();
    if n == 0 {
        return BalancedClusters {
            centroids: Vec::new(),
            members: Vec::new(),
        };
    }

    let k = num_postings.max(1).min(n);
    let all_indices: Vec<usize> = (0..n).collect();
    let (mut centroids, mut members) = kmeans_subset(vectors, &all_indices, k, max_iters, metric);

    let limit = posting_limit.max(1);
    let mut queue: VecDeque<usize> = (0..centroids.len()).collect();
    while let Some(ci) = queue.pop_front() {
        if members[ci].len() <= limit {
            continue;
        }

        let subset = members[ci].clone();
        let (sub_centroids, sub_members) = kmeans_subset(vectors, &subset, 2, max_iters, metric);

        let progressed = sub_members.len() == 2
            && !sub_members[0].is_empty()
            && !sub_members[1].is_empty()
            && sub_members[0].len() < subset.len()
            && sub_members[1].len() < subset.len();
        if !progressed {
            // Cannot balance this cluster further (e.g. all members are
            // identical); leave it as-is rather than looping forever.
            continue;
        }

        centroids[ci].clone_from(&sub_centroids[0]);
        members[ci].clone_from(&sub_members[0]);
        centroids.push(sub_centroids[1].clone());
        members.push(sub_members[1].clone());
        let new_ci = centroids.len() - 1;

        queue.push_back(ci);
        queue.push_back(new_ci);
    }

    BalancedClusters { centroids, members }
}

// ── Boundary-closure replication + RNG-rule pruning ──────────────────────────

/// For `point`, compute the set of centroid indices it should be replicated
/// into: every centroid whose distance to `point` is at most
/// `nearest_dist + epsilon * |nearest_dist|` (the nearest centroid's own
/// distance, loosened by a magnitude-scaled `epsilon`), capped at
/// `replica_count`, then pruned by the relative-neighbourhood-graph (RNG)
/// rule.
///
/// The RNG rule walks candidates in ascending distance-to-`point` order and
/// greedily accepts them, dropping a candidate `c2` whenever an
/// already-accepted, closer centroid `c1` satisfies
/// `dist(c1, c2) < dist(point, c2)` — i.e. `c1` lies inside the "lune"
/// between `point` and `c2`, making the `c2` replica redundant. The nearest
/// centroid is always accepted (nothing can dominate the first candidate).
pub(super) fn boundary_replicas(
    point: &[f32],
    centroids: &[Vec<f32>],
    epsilon: f32,
    replica_count: usize,
    metric: SpannMetric,
) -> Vec<usize> {
    if centroids.is_empty() {
        return Vec::new();
    }

    let mut dists: Vec<(f32, usize)> = centroids
        .iter()
        .enumerate()
        .map(|(ci, c)| (metric_distance(point, c, metric), ci))
        .collect();
    dists.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let nearest_dist = dists[0].0;
    // Scale the slack by the *magnitude* of the nearest distance rather than
    // multiplying the (possibly negative) distance directly. `SpannMetric::Dot`
    // yields negative "distances" (it is a negated dot product), for which a
    // plain `nearest_dist * (1.0 + epsilon)` would make the threshold *more*
    // negative — i.e. stricter, inverting the intended slack. Adding
    // `epsilon * |nearest_dist|` always loosens the threshold by the same
    // proportional amount regardless of the distance's sign, and is
    // equivalent to the plain multiplicative form whenever `nearest_dist` is
    // non-negative (as it always is for `L2` and `Cosine`).
    let threshold = nearest_dist + epsilon.max(0.0) * nearest_dist.abs();

    let candidates: Vec<(f32, usize)> = dists
        .into_iter()
        .filter(|&(d, _)| d <= threshold)
        .take(replica_count.max(1))
        .collect();

    let mut accepted: Vec<usize> = Vec::new();
    for &(dist_point_c2, c2) in &candidates {
        let dominated = accepted
            .iter()
            .any(|&c1| metric_distance(&centroids[c1], &centroids[c2], metric) < dist_point_c2);
        if !dominated {
            accepted.push(c2);
        }
    }
    accepted
}
