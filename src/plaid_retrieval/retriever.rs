//! PLAID retriever: centroid-accelerated `ColBERTv2` late interaction.

use std::collections::BTreeSet;

use crate::plaid_retrieval::types::{PlaidConfig, PlaidError, PlaidHit};
use crate::types::Document;

// ── Tokenisation ──────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens of length `>= 2`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Per-token embedding ─────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of `bytes` seeded with `seed`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// Deterministic per-token embedding.
///
/// Unlike a bag-of-tokens document vector, this assigns each *individual* token
/// its own dense vector so that late interaction can match query tokens against
/// document tokens. The token's FNV-1a hash deterministically activates a small
/// fixed number of dimensions with signed magnitudes; the result is
/// L2-normalised. Identical tokens map to identical vectors (cosine `= 1`) while
/// unrelated tokens are near-orthogonal.
fn embed_token(token: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    if dim == 0 {
        return vec;
    }
    // Activate a handful of dimensions deterministically from independent hashes.
    let spread = 8usize.min(dim);
    let bytes = token.as_bytes();
    for k in 0..spread {
        let h = fnv1a(bytes, (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        // Sign and magnitude drawn from other bits of the hash, kept deterministic.
        let sign = if (h >> 33) & 1 == 0 { 1.0 } else { -1.0 };
        #[allow(clippy::cast_precision_loss)]
        let mag = 1.0 + ((h >> 7) % 7) as f32; // 1.0 ..= 7.0
        vec[idx] += sign * mag;
    }
    l2_normalise(&mut vec);
    vec
}

/// L2-normalise `vec` in place (no-op for a zero vector).
fn l2_normalise(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Inputs are L2-normalised on construction, so this is their dot product; it
/// still divides by the norms defensively to stay correct for centroids.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ── Centroid training (spread k-means) ──────────────────────────────────────────

/// Train `k` centroids over `points` with deterministic spread initialisation.
///
/// Initial centroids are evenly spaced picks from `points`; assignment uses
/// cosine similarity (highest wins); update averages assigned members and
/// re-normalises. Empty centroids retain their previous position. Runs at most
/// `iters` iterations or until assignments stabilise.
fn train_centroids(points: &[Vec<f32>], k: usize, dim: usize, iters: usize) -> Vec<Vec<f32>> {
    let n = points.len();
    if n == 0 || k == 0 || dim == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let step = (n / k).max(1);
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|ci| points[(ci * step).min(n - 1)].clone())
        .collect();

    let mut assignments = vec![0usize; n];
    for _ in 0..iters.max(1) {
        let mut changed = false;
        for (i, point) in points.iter().enumerate() {
            let best = nearest_centroid(point, &centroids);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, &ci) in assignments.iter().enumerate() {
            for d in 0..dim {
                sums[ci][d] += points[i][d];
            }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let cnt = counts[ci] as f32;
            for x in &mut sums[ci] {
                *x /= cnt;
            }
            l2_normalise(&mut sums[ci]);
            centroids[ci] = std::mem::take(&mut sums[ci]);
        }
    }
    centroids
}

/// Index of the centroid most similar to `point` (cosine; ties → lowest index).
fn nearest_centroid(point: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_sim = f32::NEG_INFINITY;
    for (ci, centroid) in centroids.iter().enumerate() {
        let sim = cosine(point, centroid);
        if sim > best_sim {
            best_sim = sim;
            best = ci;
        }
    }
    best
}

/// Indices of the `nprobe` centroids most similar to `point` (descending cosine).
fn nearest_centroids(point: &[f32], centroids: &[Vec<f32>], nprobe: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = centroids
        .iter()
        .enumerate()
        .map(|(ci, c)| (cosine(point, c), ci))
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .take(nprobe.max(1))
        .map(|(_, ci)| ci)
        .collect()
}

// ── PlaidRetriever ──────────────────────────────────────────────────────────────

/// Centroid-accelerated late-interaction retriever (PLAID / `ColBERTv2`).
///
/// Each document is represented as a *set* of per-token embeddings. Building the
/// retriever trains centroids over every token vector in the corpus and records,
/// for each document, the set of centroids its tokens are assigned to. A search
/// embeds the query's tokens, probes each token's `nprobe` nearest centroids to
/// gather candidate documents, and re-scores those candidates with full `MaxSim`.
#[derive(Debug, Clone)]
pub struct PlaidRetriever {
    /// Retriever configuration.
    config: PlaidConfig,
    /// Trained centroids over all corpus token embeddings.
    centroids: Vec<Vec<f32>>,
    /// Documents paired with their per-token embeddings.
    docs: Vec<(Document, Vec<Vec<f32>>)>,
    /// For each document, the sorted set of centroid ids its tokens hit.
    doc_centroids: Vec<BTreeSet<usize>>,
    /// Whether [`build`](Self::build) has populated the retriever.
    built: bool,
}

impl PlaidRetriever {
    /// Create a new, empty retriever with the given configuration.
    #[must_use]
    pub fn new(config: PlaidConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            docs: Vec::new(),
            doc_centroids: Vec::new(),
            built: false,
        }
    }

    /// Build the retriever over `docs`.
    ///
    /// Tokenises and embeds every document's tokens, trains centroids over all
    /// token vectors at once, then records which centroids each document's tokens
    /// are assigned to for fast candidate generation.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::EmptyCorpus`] when `docs` is empty.
    pub fn build(&mut self, docs: &[Document]) -> Result<(), PlaidError> {
        if docs.is_empty() {
            return Err(PlaidError::EmptyCorpus);
        }
        let dim = self.config.dim;
        let mut stored: Vec<(Document, Vec<Vec<f32>>)> = Vec::with_capacity(docs.len());
        let mut all_tokens: Vec<Vec<f32>> = Vec::new();
        for doc in docs {
            let token_vecs: Vec<Vec<f32>> = tokenize(&doc.content)
                .iter()
                .map(|tok| embed_token(tok, dim))
                .collect();
            all_tokens.extend(token_vecs.iter().cloned());
            stored.push((doc.clone(), token_vecs));
        }

        self.centroids = train_centroids(
            &all_tokens,
            self.config.num_centroids,
            dim,
            self.config.kmeans_iters,
        );

        self.doc_centroids = stored
            .iter()
            .map(|(_, toks)| {
                let mut set = BTreeSet::new();
                for tok in toks {
                    if !self.centroids.is_empty() {
                        set.insert(nearest_centroid(tok, &self.centroids));
                    }
                }
                set
            })
            .collect();

        self.docs = stored;
        self.built = true;
        Ok(())
    }

    /// Number of documents held by the retriever.
    #[must_use]
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Return `true` when the retriever holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Number of trained centroids.
    #[must_use]
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// `MaxSim` late-interaction score between query and document token sets.
    ///
    /// `score = Σ_{t ∈ q} max_{s ∈ d} cosine(t, s)`. Returns `0.0` if either set
    /// is empty.
    #[must_use]
    pub fn maxsim(q_tokens: &[Vec<f32>], d_tokens: &[Vec<f32>]) -> f32 {
        if q_tokens.is_empty() || d_tokens.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for q in q_tokens {
            let mut best = f32::NEG_INFINITY;
            for d in d_tokens {
                let sim = cosine(q, d);
                if sim > best {
                    best = sim;
                }
            }
            if best > f32::NEG_INFINITY {
                total += best;
            }
        }
        total
    }

    /// Search for the `top_k` documents best matching `query`.
    ///
    /// Embeds the query tokens, gathers candidate documents that share any of a
    /// query token's `nprobe` nearest centroids, and scores those candidates with
    /// [`maxsim`](Self::maxsim). When centroid probing surfaces no candidates the
    /// search falls back to scoring the whole corpus. Results are sorted by
    /// descending score (ties broken by document id) and truncated to `top_k`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaidError::NotBuilt`] when [`build`](Self::build) has not run,
    /// [`PlaidError::EmptyCorpus`] when the corpus is empty, and
    /// [`PlaidError::EmptyQuery`] when `query` has no usable tokens.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<PlaidHit>, PlaidError> {
        if !self.built {
            return Err(PlaidError::NotBuilt);
        }
        if self.docs.is_empty() {
            return Err(PlaidError::EmptyCorpus);
        }
        let q_tokens: Vec<Vec<f32>> = tokenize(query)
            .iter()
            .map(|tok| embed_token(tok, self.config.dim))
            .collect();
        if q_tokens.is_empty() {
            return Err(PlaidError::EmptyQuery);
        }

        // Candidate generation: union of docs hit by each query token's nprobe
        // nearest centroids.
        let mut candidates: BTreeSet<usize> = BTreeSet::new();
        if !self.centroids.is_empty() {
            let mut probed: BTreeSet<usize> = BTreeSet::new();
            for q in &q_tokens {
                for ci in nearest_centroids(q, &self.centroids, self.config.nprobe) {
                    probed.insert(ci);
                }
            }
            for (doc_idx, hit) in self.doc_centroids.iter().enumerate() {
                if hit.iter().any(|c| probed.contains(c)) {
                    candidates.insert(doc_idx);
                }
            }
        }

        // Fall back to the full corpus when probing surfaced nothing.
        let candidate_indices: Vec<usize> = if candidates.is_empty() {
            (0..self.docs.len()).collect()
        } else {
            candidates.into_iter().collect()
        };

        let mut hits: Vec<PlaidHit> = candidate_indices
            .into_iter()
            .map(|idx| {
                let (doc, toks) = &self.docs[idx];
                PlaidHit {
                    document: doc.clone(),
                    score: Self::maxsim(&q_tokens, toks),
                }
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document.id.as_str().cmp(b.document.id.as_str()))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}
