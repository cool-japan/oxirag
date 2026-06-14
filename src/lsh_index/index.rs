//! Random-hyperplane LSH index for dense float vectors (cosine similarity).
//!
//! Implements the classic sign-random-projection LSH of Charikar (2002).
//! Each indexed vector is projected onto `num_planes` deterministic random
//! unit hyperplanes; the sign of each projection becomes one bit of a
//! `u64`-packed bit-signature. Vectors with identical signatures land in the
//! same bucket and are treated as approximate nearest-neighbour candidates.
//! At query time candidates are re-ranked by exact cosine similarity and the
//! top-k are returned.
//!
//! Hyperplanes are generated **deterministically** via FNV-1a hashing over
//! `(plane_idx, dim_idx)` pairs, so no external RNG is needed.

use std::collections::HashMap;

use super::types::{LshConfig, LshError, LshHit};

// ── internals ─────────────────────────────────────────────────────────────────

/// Compute a single FNV-1a hash component `∈ [-1, 1]` for `(plane, dim)`.
///
/// Produces a deterministic pseudo-random float that stands as one element of
/// the `plane`-th hyperplane's normal vector.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn fnv_hyperplane_elem(plane: usize, dim: usize) -> f32 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut h: u64 = OFFSET;
    // Mix plane index bytes
    for byte in plane.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(PRIME);
    }
    // Mix dim index bytes
    for byte in dim.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(PRIME);
    }
    // Map upper 32 bits to [-1, 1]
    let upper = (h >> 32) as u32;
    (upper as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

/// Generate `num_planes` hyperplanes of dimensionality `dim`, each
/// normalised to unit length.
fn generate_hyperplanes(num_planes: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..num_planes)
        .map(|p| {
            let mut hp: Vec<f32> = (0..dim).map(|d| fnv_hyperplane_elem(p, d)).collect();
            let norm: f32 = hp.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for x in &mut hp {
                *x /= norm;
            }
            hp
        })
        .collect()
}

/// Compute the bit-signature for a vector given the set of hyperplanes.
///
/// Each bit `i` is `1` when `dot(hyperplane[i], vector) >= 0`.
/// The result is packed into a `Vec<u64>` (one `u64` per 64 planes).
fn bit_signature(hyperplanes: &[Vec<f32>], vector: &[f32]) -> Vec<u64> {
    let num_words = hyperplanes.len().div_ceil(64);
    let mut sig = vec![0u64; num_words];
    for (i, hp) in hyperplanes.iter().enumerate() {
        let dot: f32 = hp.iter().zip(vector.iter()).map(|(h, v)| h * v).sum();
        if dot >= 0.0 {
            sig[i / 64] |= 1u64 << (i % 64);
        }
    }
    sig
}

/// Exact cosine similarity between two already-normalised vectors.
#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

// ── LshIndex ──────────────────────────────────────────────────────────────────

/// A stored entry in the LSH index.
struct Entry {
    id: String,
    vector: Vec<f32>,
}

/// Random-hyperplane LSH index for dense float vectors.
///
/// Vectors are hashed using sign-random-projection into buckets defined by
/// their bit-signature. At search time, all vectors in the same bucket as the
/// query are retrieved as candidates and re-ranked by exact cosine similarity.
///
/// # Construction
///
/// Hyperplanes are generated deterministically from the configured
/// [`LshConfig::dim`] and [`LshConfig::num_planes`] via FNV-1a hashing,
/// so the same configuration always produces the same hyperplanes.
///
/// # Notes
///
/// - Input vectors need not be normalised; the index normalises them
///   internally before projecting and storing.
/// - Cosine scores in `LshHit` are mapped to `[0, 1]` via
///   `(raw_cosine + 1.0) / 2.0`.
pub struct LshIndex {
    config: LshConfig,
    hyperplanes: Vec<Vec<f32>>,
    // signature → list of entry indices
    buckets: HashMap<Vec<u64>, Vec<usize>>,
    entries: Vec<Entry>,
}

impl LshIndex {
    /// Create a new, empty index for the given configuration.
    ///
    /// Hyperplanes are generated immediately during construction.
    ///
    /// # Errors
    ///
    /// Returns [`LshError::InvalidConfig`] when the configuration is invalid.
    pub fn new(config: LshConfig) -> Result<Self, LshError> {
        config.validate()?;
        let hyperplanes = generate_hyperplanes(config.num_planes, config.dim);
        Ok(Self {
            config,
            hyperplanes,
            buckets: HashMap::new(),
            entries: Vec::new(),
        })
    }

    /// Number of vectors currently indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no vectors have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &LshConfig {
        &self.config
    }

    /// Insert a dense float vector into the index.
    ///
    /// The vector is normalised to unit length before being stored and hashed.
    ///
    /// # Errors
    ///
    /// Returns [`LshError::DimMismatch`] when `vector.len()` does not equal
    /// [`LshConfig::dim`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn insert(&mut self, id: impl Into<String>, vector: Vec<f32>) -> Result<(), LshError> {
        if vector.len() != self.config.dim {
            return Err(LshError::DimMismatch {
                expected: self.config.dim,
                got: vector.len(),
            });
        }
        // Normalise to unit length so cosine_sim is just a dot product.
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normed: Vec<f32> = vector.iter().map(|x| x / norm).collect();

        let sig = bit_signature(&self.hyperplanes, &normed);
        let idx = self.entries.len();
        self.entries.push(Entry {
            id: id.into(),
            vector: normed,
        });
        self.buckets.entry(sig).or_default().push(idx);
        Ok(())
    }

    /// Search for the `k` most similar vectors to `query`.
    ///
    /// The query is normalised, its bit-signature is computed, and all vectors
    /// sharing that signature are re-ranked by exact cosine similarity.
    /// If fewer than `k` candidates exist in the matching bucket, additional
    /// candidates are drawn from buckets with the highest Hamming-bit overlap
    /// (i.e., nearest signatures by XOR popcount) until `k` candidates are
    /// gathered or all buckets are exhausted.
    ///
    /// Returned [`LshHit`] scores are in `[0, 1]` (cosine mapped from
    /// `[-1, 1]` by `(cos + 1) / 2`), ordered descending.
    ///
    /// # Errors
    ///
    /// - [`LshError::EmptyIndex`] when no vectors have been inserted.
    /// - [`LshError::InvalidK`] when `k == 0`.
    /// - [`LshError::DimMismatch`] when `query.len() != config.dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<LshHit>, LshError> {
        if self.entries.is_empty() {
            return Err(LshError::EmptyIndex);
        }
        if k == 0 {
            return Err(LshError::InvalidK);
        }
        if query.len() != self.config.dim {
            return Err(LshError::DimMismatch {
                expected: self.config.dim,
                got: query.len(),
            });
        }

        // Normalise query
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let q_normed: Vec<f32> = query.iter().map(|x| x / norm).collect();
        let q_sig = bit_signature(&self.hyperplanes, &q_normed);

        // Collect candidate indices, starting from exact-signature bucket
        // and expanding to neighbouring buckets ranked by Hamming distance.
        let candidate_indices = self.gather_candidates(&q_sig, k);

        // Re-rank candidates by exact cosine similarity
        let mut scored: Vec<(f32, &str)> = candidate_indices
            .iter()
            .map(|&idx| {
                let entry = &self.entries[idx];
                let cos = cosine_sim(&q_normed, &entry.vector);
                (cos, entry.id.as_str())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.dedup_by(|a, b| a.1 == b.1);
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(cos, id)| {
                let score = f32::midpoint(cos, 1.0);
                LshHit::new(id, score.clamp(0.0, 1.0))
            })
            .collect())
    }

    /// Gather up to `needed` candidate entry indices by scanning buckets in
    /// order of ascending Hamming distance from `q_sig`.
    fn gather_candidates(&self, q_sig: &[u64], needed: usize) -> Vec<usize> {
        // Collect all (hamming_dist, bucket_key, entry_indices) tuples
        let mut bucket_scores: Vec<(u32, &Vec<usize>)> = self
            .buckets
            .iter()
            .map(|(sig, indices)| {
                let dist = hamming_distance(q_sig, sig);
                (dist, indices)
            })
            .collect();

        // Sort by ascending Hamming distance (exact-match bucket first)
        bucket_scores.sort_by_key(|(d, _)| *d);

        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::with_capacity(needed.min(self.entries.len()));
        // Cap for non-exact buckets; the exact bucket is always fully included so
        // that an indexed vector is never excluded because of the cap.
        let neighbour_cap = (needed * 4).min(self.entries.len());

        for (dist, indices) in bucket_scores {
            for &idx in indices {
                if seen.insert(idx) {
                    result.push(idx);
                }
            }
            // After exhausting the exact-match bucket, apply the cap to avoid
            // scanning the entire corpus for neighbour buckets.
            if dist > 0 && result.len() >= neighbour_cap {
                return result;
            }
        }
        result
    }
}

/// Hamming distance between two signature words (popcount of XOR).
fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// ── MinHashIndex ──────────────────────────────────────────────────────────────

/// A stored set entry for the `MinHash` index.
struct SetEntry {
    id: String,
    /// Original set stored for exact Jaccard re-ranking.
    set: Vec<u64>,
}

/// `MinHash` LSH index for sparse set-based similarity (Jaccard).
///
/// Uses `num_bands × rows_per_band` independent min-hash functions. Two sets
/// are considered candidates when they collide in at least one band (i.e.,
/// their `rows_per_band` consecutive min-hash values are all identical within
/// that band). Candidates are re-ranked by exact Jaccard similarity.
///
/// # Hash functions
///
/// Hash function `k` maps each element `e` to
/// `FNV-1a(k || e) mod 2^64`, and the minimum over all elements in the set is
/// used. The constants `(k * large_prime)` distinguish the functions without
/// using an RNG.
pub struct MinHashIndex {
    config: LshConfig,
    /// Multipliers that differentiate the `total_minhash_funcs` hash functions.
    hash_seeds: Vec<u64>,
    /// `band_id` → list of (`band_hash` → `entry_indices`)
    band_tables: Vec<HashMap<u64, Vec<usize>>>,
    entries: Vec<SetEntry>,
}

impl MinHashIndex {
    /// Create a new empty `MinHash` index.
    ///
    /// # Errors
    ///
    /// Returns [`LshError::InvalidConfig`] for invalid configurations.
    pub fn new(config: LshConfig) -> Result<Self, LshError> {
        config.validate()?;
        let total = config.total_minhash_funcs();
        let hash_seeds = generate_hash_seeds(total);
        let band_tables = (0..config.num_bands).map(|_| HashMap::new()).collect();
        Ok(Self {
            config,
            hash_seeds,
            band_tables,
            entries: Vec::new(),
        })
    }

    /// Number of sets currently indexed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no sets have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a set of element hashes (e.g. shingle hashes) into the index.
    ///
    /// # Errors
    ///
    /// Returns [`LshError::EmptyIndex`] when `elements` is empty.
    pub fn insert(&mut self, id: impl Into<String>, elements: Vec<u64>) -> Result<(), LshError> {
        if elements.is_empty() {
            return Err(LshError::EmptyIndex);
        }
        let signature = compute_minhash_signature(&elements, &self.hash_seeds);
        let idx = self.entries.len();

        // File into band buckets
        for (band, table) in self.band_tables.iter_mut().enumerate() {
            let band_hash = band_hash_key(&signature, band, self.config.rows_per_band);
            table.entry(band_hash).or_default().push(idx);
        }

        self.entries.push(SetEntry {
            id: id.into(),
            set: elements,
        });
        Ok(())
    }

    /// Search for the `k` most Jaccard-similar sets to `query_elements`.
    ///
    /// # Errors
    ///
    /// - [`LshError::EmptyIndex`] when no sets are indexed.
    /// - [`LshError::InvalidK`] when `k == 0`.
    pub fn search(&self, query_elements: &[u64], k: usize) -> Result<Vec<LshHit>, LshError> {
        if self.entries.is_empty() {
            return Err(LshError::EmptyIndex);
        }
        if k == 0 {
            return Err(LshError::InvalidK);
        }
        let q_sig = compute_minhash_signature(query_elements, &self.hash_seeds);

        // Gather candidates from all band collisions
        let mut candidate_set = std::collections::HashSet::new();
        for (band, table) in self.band_tables.iter().enumerate() {
            let key = band_hash_key(&q_sig, band, self.config.rows_per_band);
            if let Some(indices) = table.get(&key) {
                for &idx in indices {
                    candidate_set.insert(idx);
                }
            }
        }

        // Re-rank by exact Jaccard
        let mut scored: Vec<(f32, &str)> = candidate_set
            .iter()
            .map(|&idx| {
                let entry = &self.entries[idx];
                let j = jaccard_similarity(query_elements, &entry.set);
                (j, entry.id.as_str())
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(score, id)| LshHit::new(id, score))
            .collect())
    }
}

// ── MinHash helpers ───────────────────────────────────────────────────────────

/// Generate `n` distinct large odd seeds for the min-hash functions.
fn generate_hash_seeds(n: usize) -> Vec<u64> {
    // Use FNV-derived primes so the seeds are deterministic
    const BASE: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    (0..n)
        .map(|i| {
            let mut h = BASE;
            for byte in i.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(PRIME);
            }
            // Ensure odd so it is coprime to 2^64
            h | 1
        })
        .collect()
}

/// Compute the min-hash signature for a set of element hashes.
///
/// For each hash function `k` (represented by seed `hash_seeds[k]`), the
/// min-hash is `min over e in elements of FNV(seed_k XOR e)`.
fn compute_minhash_signature(elements: &[u64], hash_seeds: &[u64]) -> Vec<u64> {
    hash_seeds
        .iter()
        .map(|&seed| {
            elements
                .iter()
                .map(|&e| fnv_mix(seed ^ e))
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

/// A single FNV-1a mixing step for `u64`.
#[inline]
fn fnv_mix(mut v: u64) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    // Mix all 8 bytes
    for i in 0..8u64 {
        v ^= (v >> (i * 7 + 3)) & 0xFF;
        v = v.wrapping_mul(PRIME);
    }
    v
}

/// Produce a single `u64` band-key for band `b` from a signature.
fn band_hash_key(signature: &[u64], band: usize, rows_per_band: usize) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    let start = band * rows_per_band;
    let end = (start + rows_per_band).min(signature.len());
    let mut h: u64 = 14_695_981_039_346_656_037;
    // Include band index so hashes across bands don't accidentally collide
    h ^= band as u64;
    h = h.wrapping_mul(PRIME);
    for &val in &signature[start..end] {
        for byte in val.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(PRIME);
        }
    }
    h
}

/// Exact Jaccard similarity between two sorted-or-unsorted element sets.
#[allow(clippy::cast_precision_loss)]
fn jaccard_similarity(a: &[u64], b: &[u64]) -> f32 {
    let mut set_a: Vec<u64> = a.to_vec();
    let mut set_b: Vec<u64> = b.to_vec();
    set_a.sort_unstable();
    set_a.dedup();
    set_b.sort_unstable();
    set_b.dedup();

    let mut intersection = 0usize;
    let mut i = 0;
    let mut j = 0;
    while i < set_a.len() && j < set_b.len() {
        match set_a[i].cmp(&set_b[j]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let union = set_a.len() + set_b.len() - intersection;
    if union == 0 {
        return 1.0;
    }
    intersection as f32 / union as f32
}
