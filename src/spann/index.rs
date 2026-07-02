//! The SPANN index itself: build (balanced clustering + boundary-closure
//! replication) and search (nprobe centroid scan + dedup + rank).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::cluster::{balanced_kmeans, boundary_replicas, metric_distance, metric_score};
use super::types::{Posting, SpannConfig, SpannError, SpannHit, SpannStats};

/// SPANN: a memory-disk hybrid approximate-nearest-neighbour index (Chen et
/// al., `NeurIPS` 2021).
///
/// The index keeps a small "head" of centroids together with per-centroid
/// posting lists (the "tail"). Unlike a plain inverted-file index
/// ([`crate::ivf_index::IvfIndex`]), SPANN additionally:
///
/// 1. **Balances** clustering so no posting list's *primary* membership
///    exceeds [`SpannConfig::posting_limit`] (oversized clusters are
///    recursively re-clustered into two).
/// 2. **Replicates boundary points** into every centroid within
///    `(1 + boundary_epsilon)` of their nearest centroid, up to
///    [`SpannConfig::replica_count`], so points near a Voronoi boundary are
///    still found when the query lands in a neighbouring cell.
/// 3. **Prunes redundant replicas** with the relative-neighbourhood-graph
///    (RNG) rule, so replication raises recall without needlessly
///    duplicating every boundary point into every nearby list.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "spann")] {
/// use oxirag::spann::{SpannConfig, SpannIndex};
///
/// let items = vec![
///     ("a".to_string(), vec![1.0, 0.0, 0.0]),
///     ("b".to_string(), vec![0.9, 0.1, 0.0]),
///     ("c".to_string(), vec![0.0, 1.0, 0.0]),
///     ("d".to_string(), vec![0.0, 0.0, 1.0]),
/// ];
/// let config = SpannConfig::new().with_num_postings(2).with_nprobe(2);
/// let index = SpannIndex::build(items, config).unwrap();
///
/// let hits = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
/// assert_eq!(hits[0].id, "a");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SpannIndex {
    config: SpannConfig,
    dim: usize,
    postings: Vec<Posting>,
    vectors: HashMap<String, Vec<f32>>,
}

impl SpannIndex {
    /// Build a SPANN index over `items` (identifier, vector pairs).
    ///
    /// Runs balanced k-means clustering (splitting any oversized cluster),
    /// then replicates every point into every centroid within
    /// `(1 + config.boundary_epsilon)` of its nearest centroid (capped at
    /// `config.replica_count` and pruned by the RNG rule).
    ///
    /// If `items` contains duplicate identifiers, the vector stored for
    /// lookup at search time is the one from the *last* occurrence, but
    /// every occurrence still participates independently in clustering and
    /// replication.
    ///
    /// # Errors
    ///
    /// Returns [`SpannError::InvalidConfig`] when `config` fails
    /// [`SpannConfig::validate`], [`SpannError::EmptyIndex`] when `items` is
    /// empty, or [`SpannError::DimensionMismatch`] when the vectors in
    /// `items` do not all share the same length.
    pub fn build(items: Vec<(String, Vec<f32>)>, config: SpannConfig) -> Result<Self, SpannError> {
        config.validate()?;
        if items.is_empty() {
            return Err(SpannError::EmptyIndex);
        }

        let dim = items[0].1.len();
        for (_, v) in &items {
            if v.len() != dim {
                return Err(SpannError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let ids: Vec<String> = items.iter().map(|(id, _)| id.clone()).collect();
        let vecs: Vec<Vec<f32>> = items.into_iter().map(|(_, v)| v).collect();

        let clusters = balanced_kmeans(
            &vecs,
            config.num_postings,
            config.posting_limit,
            config.max_iters,
            config.metric,
        );

        let mut postings: Vec<Posting> = clusters
            .centroids
            .iter()
            .map(|c| Posting {
                centroid: c.clone(),
                member_ids: Vec::new(),
            })
            .collect();

        for (gi, v) in vecs.iter().enumerate() {
            let accepted = boundary_replicas(
                v,
                &clusters.centroids,
                config.boundary_epsilon,
                config.replica_count,
                config.metric,
            );
            for ci in accepted {
                postings[ci].member_ids.push(ids[gi].clone());
            }
        }

        let vectors: HashMap<String, Vec<f32>> = ids.into_iter().zip(vecs).collect();

        Ok(Self {
            config,
            dim,
            postings,
            vectors,
        })
    }

    /// Search for the `k` nearest neighbours of `query`.
    ///
    /// Locates the `nprobe` nearest centroids, scans only their posting
    /// lists, de-duplicates points that were replicated into more than one
    /// probed list, scores each surviving candidate against `query`, and
    /// returns the `k` highest-scoring hits in descending-score order (ties
    /// broken by ascending identifier for determinism).
    ///
    /// # Errors
    ///
    /// Returns [`SpannError::EmptyIndex`] if the index holds no postings,
    /// [`SpannError::EmptyQuery`] if `query` is empty while the index
    /// expects a non-zero dimensionality, or
    /// [`SpannError::DimensionMismatch`] when `query.len() != self.dim()`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SpannHit>, SpannError> {
        if self.postings.is_empty() {
            return Err(SpannError::EmptyIndex);
        }
        if query.is_empty() && self.dim != 0 {
            return Err(SpannError::EmptyQuery);
        }
        if query.len() != self.dim {
            return Err(SpannError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let probe = self.nearest_postings(query);

        let mut seen: HashSet<&str> = HashSet::new();
        let mut candidate_ids: Vec<&str> = Vec::new();
        for ci in probe {
            for id in &self.postings[ci].member_ids {
                if seen.insert(id.as_str()) {
                    candidate_ids.push(id.as_str());
                }
            }
        }

        let mut hits: Vec<SpannHit> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                self.vectors.get(id).map(|v| SpannHit {
                    id: id.to_string(),
                    score: metric_score(query, v, self.config.metric),
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Return the indices of the `nprobe` nearest posting lists to `query`,
    /// nearest first. `nprobe` is clamped to the number of available
    /// postings.
    fn nearest_postings(&self, query: &[f32]) -> Vec<usize> {
        let nprobe = self.config.nprobe.max(1).min(self.postings.len());
        let mut scored: Vec<(f32, usize)> = self
            .postings
            .iter()
            .enumerate()
            .map(|(ci, p)| (metric_distance(query, &p.centroid, self.config.metric), ci))
            .collect();
        scored.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored.into_iter().take(nprobe).map(|(_, ci)| ci).collect()
    }

    /// Borrow the configuration backing this index.
    #[must_use]
    pub fn config(&self) -> &SpannConfig {
        &self.config
    }

    /// Borrow the posting lists (centroid "head" plus member "tail" ids).
    #[must_use]
    pub fn postings(&self) -> &[Posting] {
        &self.postings
    }

    /// Number of distinct indexed identifiers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index holds no indexed points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Dimensionality the index was built with.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of posting lists (centroids) in the index.
    #[must_use]
    pub fn num_postings(&self) -> usize {
        self.postings.len()
    }

    /// Compute summary statistics over the current posting lists.
    #[must_use]
    pub fn stats(&self) -> SpannStats {
        let num_postings = self.postings.len();
        let lens: Vec<usize> = self.postings.iter().map(|p| p.member_ids.len()).collect();
        let total_replicas: usize = lens.iter().sum();
        #[allow(clippy::cast_precision_loss)]
        let avg_posting_len = if num_postings == 0 {
            0.0
        } else {
            total_replicas as f32 / num_postings as f32
        };
        let max_posting_len = lens.iter().copied().max().unwrap_or(0);
        let min_posting_len = lens.iter().copied().min().unwrap_or(0);

        SpannStats {
            num_postings,
            total_replicas,
            avg_posting_len,
            max_posting_len,
            min_posting_len,
        }
    }
}
