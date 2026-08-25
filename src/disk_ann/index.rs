//! Core `DiskANN` index: medoid computation, two-pass Vamana construction, and
//! `GreedySearch`-based retrieval.
//!
//! Implements the Vamana graph algorithm as described in Subramanya et al.,
//! *"`DiskANN`: Fast Accurate Billion-point Nearest Neighbor Search on a Single
//! Node"* (`NeurIPS` 2019). This is an **in-memory** approximation: the full
//! `DiskANN` system additionally supports SSD-resident PQ-compressed vectors
//! with beam-search-driven disk reads, which is out of scope here — this
//! module keeps every vector in memory and focuses on the graph algorithm
//! itself (medoid entry point, random init, `GreedySearch`, `RobustPrune`,
//! two-pass build).

use super::graph::{VamanaGraph, robust_prune};
use super::types::{DiskAnnConfig, DiskAnnError, DiskAnnHit, DiskAnnMetric};

/// Corpora at or below this size use the exact `O(n^2)` medoid computation
/// (sum of pairwise distances). Larger corpora fall back to a
/// centroid-nearest approximation so `build` stays sub-quadratic — an
/// explicit, honest approximation rather than a silently-degraded exact
/// computation.
const MEDOID_EXACT_THRESHOLD: usize = 256;

/// Compute (or approximate) the medoid of `vectors`: the point minimizing
/// the sum of distances to every other point.
///
/// For corpora of at most [`MEDOID_EXACT_THRESHOLD`] points this is computed
/// exactly in `O(n^2)`. Larger corpora use the point nearest to the
/// centroid (mean vector) as a deterministic, allocation-light
/// approximation — this never requires randomness.
fn compute_medoid(vectors: &[Vec<f32>], metric: DiskAnnMetric) -> usize {
    let n = vectors.len();
    if n <= MEDOID_EXACT_THRESHOLD {
        let mut best_idx = 0usize;
        let mut best_sum = f32::INFINITY;
        for i in 0..n {
            let mut sum = 0.0f32;
            for (j, other) in vectors.iter().enumerate() {
                if i != j {
                    sum += metric.distance(&vectors[i], other);
                }
            }
            if sum < best_sum {
                best_sum = sum;
                best_idx = i;
            }
        }
        best_idx
    } else {
        let dim = vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for v in vectors {
            for (c, x) in centroid.iter_mut().zip(v.iter()) {
                *c += x;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let count = n as f32;
        for c in &mut centroid {
            *c /= count;
        }

        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, v) in vectors.iter().enumerate() {
            let dist = metric.distance(&centroid, v);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }
}

// ── DiskAnnIndex ──────────────────────────────────────────────────────────────

/// A `DiskANN` / Vamana approximate-nearest-neighbour index.
///
/// Build the index once with [`DiskAnnIndex::build`] (Vamana graphs are not
/// designed for incremental single-point inserts the way HNSW is), then
/// query repeatedly with [`DiskAnnIndex::search`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "disk-ann")] {
/// use oxirag::disk_ann::{DiskAnnConfig, DiskAnnIndex};
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
///     v
/// }
///
/// let items = vec![
///     ("doc_a".to_string(), fnv_embed("alpha", 16)),
///     ("doc_b".to_string(), fnv_embed("beta", 16)),
/// ];
/// let index = DiskAnnIndex::build(items, DiskAnnConfig::new().with_max_degree(4)).unwrap();
///
/// let hits = index.search(&fnv_embed("alpha", 16), 1).unwrap();
/// assert_eq!(hits[0].id, "doc_a");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DiskAnnIndex {
    config: DiskAnnConfig,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    graph: VamanaGraph,
    dim: usize,
}

impl DiskAnnIndex {
    /// Build a Vamana graph index over `items`.
    ///
    /// Runs the full construction algorithm:
    ///
    /// 1. Compute (or approximate) the **medoid** — the fixed entry point.
    /// 2. Initialize a random `R`-regular graph via a deterministic FNV-1a
    ///    permutation (no `rand` crate).
    /// 3. Run two `RobustPrune` build passes over the points in index order:
    ///    the first with `alpha = 1.0`, the second with the configured
    ///    `alpha`. Each pass runs `GreedySearch(medoid, p, L)` for every
    ///    point `p`, sets `p`'s out-edges via `RobustPrune`, then adds and
    ///    (if necessary) re-prunes back-edges at every new neighbor.
    ///
    /// # Errors
    ///
    /// * [`DiskAnnError::InvalidConfig`] — if `config` fails
    ///   [`DiskAnnConfig::validate`].
    /// * [`DiskAnnError::EmptyIndex`] — if `items` is empty.
    /// * [`DiskAnnError::DimensionMismatch`] — if any vector's length differs
    ///   from the first vector's length (which fixes the index dimension).
    pub fn build(
        items: Vec<(String, Vec<f32>)>,
        config: DiskAnnConfig,
    ) -> Result<Self, DiskAnnError> {
        config.validate()?;

        if items.is_empty() {
            return Err(DiskAnnError::EmptyIndex);
        }

        let dim = items[0].1.len();
        if dim == 0 {
            return Err(DiskAnnError::InvalidConfig(
                "vector dimension must be greater than zero".into(),
            ));
        }

        let mut ids: Vec<String> = Vec::with_capacity(items.len());
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(items.len());
        for (id, vector) in items {
            if vector.len() != dim {
                return Err(DiskAnnError::DimensionMismatch {
                    expected: dim,
                    got: vector.len(),
                });
            }
            ids.push(id);
            vectors.push(vector);
        }

        let n = vectors.len();
        let metric = config.metric;
        let r = config.max_degree;
        let l = config.search_list_size;

        let medoid = compute_medoid(&vectors, metric);
        let mut graph = VamanaGraph::random_init(n, r, medoid);

        // Two build passes: alpha = 1.0 first, then the configured alpha.
        for pass_alpha in [1.0_f32, config.alpha] {
            for p in 0..n {
                let (_, visited) = graph.greedy_search(&vectors, metric, medoid, &vectors[p], l);
                let pruned = robust_prune(&vectors, metric, p, visited, pass_alpha, r);
                graph.set_neighbors(p, pruned.clone());

                for &neighbor in &pruned {
                    graph.add_back_edge(neighbor, p);
                    if graph.neighbors_of(neighbor).len() > r {
                        let neighbor_edges = graph.neighbors_of(neighbor).to_vec();
                        let neighbor_pruned =
                            robust_prune(&vectors, metric, neighbor, neighbor_edges, pass_alpha, r);
                        graph.set_neighbors(neighbor, neighbor_pruned);
                    }
                }
            }
        }

        Ok(Self {
            config,
            ids,
            vectors,
            graph,
            dim,
        })
    }

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Runs `GreedySearch(medoid, query, max(L, k))` and returns the top `k`
    /// hits, sorted by descending [`DiskAnnHit::score`] (best match first).
    /// If `k == 0` an empty vector is returned (this is not treated as an
    /// error, since there is no dedicated error variant for it).
    ///
    /// # Errors
    ///
    /// * [`DiskAnnError::EmptyIndex`] — if the index holds no vectors.
    /// * [`DiskAnnError::EmptyQuery`] — if `query` has zero length.
    /// * [`DiskAnnError::DimensionMismatch`] — if `query.len() != self.dim()`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<DiskAnnHit>, DiskAnnError> {
        if self.is_empty() {
            return Err(DiskAnnError::EmptyIndex);
        }
        if query.is_empty() {
            return Err(DiskAnnError::EmptyQuery);
        }
        if query.len() != self.dim {
            return Err(DiskAnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let l = self.config.search_list_size.max(k);
        let (candidates, _visited) = self.graph.greedy_search(
            &self.vectors,
            self.config.metric,
            self.graph.medoid(),
            query,
            l,
        );

        let hits: Vec<DiskAnnHit> = candidates
            .into_iter()
            .take(k)
            .map(|(_dist, idx)| {
                let score = self.config.metric.score(query, &self.vectors[idx]);
                DiskAnnHit::new(self.ids[idx].clone(), score)
            })
            .collect();

        Ok(hits)
    }

    /// Number of vectors stored in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// `true` if the index contains no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The dimensionality of the indexed vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `true` if the index contains a vector with the given `id`.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.ids.iter().any(|existing| existing == id)
    }

    /// Borrow the configuration this index was built with.
    #[must_use]
    pub fn config(&self) -> &DiskAnnConfig {
        &self.config
    }

    /// Borrow the underlying [`VamanaGraph`].
    #[must_use]
    pub fn graph(&self) -> &VamanaGraph {
        &self.graph
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
impl DiskAnnIndex {
    /// Test-only constructor for a vector-less index.
    ///
    /// [`DiskAnnIndex::build`] always requires at least one item, so a real
    /// empty index can never be observed through the public API. This
    /// constructor exists solely so the test suite can exercise the
    /// `EmptyIndex` branch of [`DiskAnnIndex::search`] directly.
    pub(crate) fn empty_for_test(config: DiskAnnConfig, dim: usize) -> Self {
        Self {
            config,
            ids: Vec::new(),
            vectors: Vec::new(),
            graph: VamanaGraph::new(Vec::new(), 0),
            dim,
        }
    }
}
