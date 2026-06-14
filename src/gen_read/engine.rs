//! [`GenReadEngine`] — generate diverse contextual documents, cluster them,
//! and assemble one representative per cluster into a reading context.

use crate::gen_read::types::{
    ContextGenerator, GenReadConfig, GenReadError, GenReadOutput, GeneratedDoc,
};

// ── GenReadEngine ────────────────────────────────────────────────────────────────

/// Generate-then-read engine (Yu et al., 2023).
///
/// Rather than retrieving passages, the engine prompts a [`ContextGenerator`]
/// for `num_docs` *contextual documents*, embeds them with a deterministic
/// FNV-1a pseudo-embedding, clusters them into at most `num_clusters` diverse
/// groups, picks the most central document of each cluster as its
/// representative, and joins the representatives into a reading context.
///
/// This is distinct from `HyDE`: `HyDE` generates a *single* hypothetical
/// document whose *embedding* drives retrieval, whereas `GenRead` generates
/// *several* documents and uses the documents themselves — clustered for
/// diversity — as the context.
#[derive(Debug, Clone)]
pub struct GenReadEngine {
    /// Configuration for this engine.
    pub config: GenReadConfig,
}

impl GenReadEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: GenReadConfig) -> Self {
        Self { config }
    }

    /// Run generate-then-read end to end.
    ///
    /// Steps: generate `num_docs` contextual documents via the generator
    /// (indices `0..num_docs`) → embed each with a deterministic FNV-1a
    /// pseudo-embedding of dimension `config.dim` → cluster into at most
    /// `config.num_clusters` non-empty groups with deterministic k-means →
    /// select each cluster's most-central document as its representative → join
    /// the representatives in cluster order with `config.join_separator` to form
    /// the reading context.
    ///
    /// Each returned [`GeneratedDoc`] carries the index of the cluster it was
    /// assigned to; that index is always less than the number of clusters, which
    /// in turn never exceeds `config.num_clusters`.
    ///
    /// # Errors
    ///
    /// Returns [`GenReadError::EmptyQuery`] when `query` is empty after trimming.
    pub fn run<G: ContextGenerator>(
        &self,
        query: &str,
        generator: &G,
    ) -> Result<GenReadOutput, GenReadError> {
        if query.trim().is_empty() {
            return Err(GenReadError::EmptyQuery);
        }

        let num_docs = self.config.num_docs.max(1);

        // Generate the contextual documents (deterministic diversity by index).
        let contents: Vec<String> = (0..num_docs)
            .map(|index| generator.generate(query, index))
            .collect();

        // Embed and cluster.
        let embeddings: Vec<Vec<f32>> =
            contents.iter().map(|c| embed(c, self.config.dim)).collect();
        let clusters = kmeans_lite(&embeddings, self.config.num_clusters.max(1));

        // Tag every document with its cluster id (position in the cluster list).
        let mut documents: Vec<GeneratedDoc> = contents
            .iter()
            .map(|content| GeneratedDoc {
                content: content.clone(),
                cluster_id: 0,
            })
            .collect();
        for (cluster_id, members) in clusters.iter().enumerate() {
            for &doc_idx in members {
                documents[doc_idx].cluster_id = cluster_id;
            }
        }

        // Pick the most-central representative of each cluster and assemble.
        let representatives: Vec<&str> = clusters
            .iter()
            .map(|members| {
                let rep = representative(members, &embeddings);
                contents[rep].as_str()
            })
            .collect();
        let context = representatives.join(&self.config.join_separator);

        Ok(GenReadOutput {
            documents,
            clusters: clusters.len(),
            context,
        })
    }
}

// ── Lexical pseudo-embedding ─────────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise (non-alphanumeric split, length `>= 2`, lowercased) →
/// hash each token to a bucket with FNV-1a → accumulate per-bucket counts →
/// L2-normalise into a vector of length `dim`. Returns an empty vector when
/// `dim == 0`.
#[must_use]
fn embed(text: &str, dim: usize) -> Vec<f32> {
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

// ── Cosine similarity ────────────────────────────────────────────────────────────

/// Cosine similarity of two equal-length vectors (dot product for L2-normalised
/// inputs). Returns `0.0` for mismatched or empty vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── KMeans-lite clustering ───────────────────────────────────────────────────────

/// Partition `embeddings` into at most `k` clusters with deterministic k-means.
///
/// Returns a vector of clusters, each holding the original indices of its
/// members. The number of clusters is `min(k, n)`; only non-empty clusters are
/// returned, sorted by each cluster's smallest member index, so the output is a
/// deterministic partition of `0..n`.
///
/// Initial centroids are seeded by *spreading* — picking items at evenly spaced
/// offsets across the corpus — and assignment ties always resolve to the lowest
/// centroid index, so the partition depends only on the inputs.
fn kmeans_lite(embeddings: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
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

// ── Representative selection ─────────────────────────────────────────────────────

/// Select the most *central* member of a cluster as its representative.
///
/// Centrality is the sum of cosine similarities from a member to every other
/// member of the cluster; the member maximising this sum is the most
/// representative. Ties resolve to the lowest original index (deterministic),
/// and a singleton cluster trivially returns its sole member. Panics never
/// occur because clusters returned by [`kmeans_lite`] are always non-empty.
fn representative(members: &[usize], embeddings: &[Vec<f32>]) -> usize {
    debug_assert!(!members.is_empty(), "clusters are always non-empty");
    let mut best = members[0];
    let mut best_centrality = f32::NEG_INFINITY;
    for &i in members {
        let mut centrality = 0.0f32;
        for &j in members {
            if i != j {
                centrality += cosine(&embeddings[i], &embeddings[j]);
            }
        }
        // Strictly-greater keeps the lowest index on ties: deterministic.
        if centrality > best_centrality {
            best_centrality = centrality;
            best = i;
        }
    }
    best
}
