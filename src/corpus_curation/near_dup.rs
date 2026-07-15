//! Pillar 4 — near-duplicate clustering, composed on top of
//! [`crate::lsh_index::MinHashIndex`] rather than reimplemented.
//!
//! `MinHash`-banded near-duplicate detection already exists three times in
//! this crate ([`crate::lsh_index::MinHashIndex`], the O(n²) union-find
//! clustering in `crate::semantic_dedup`, and the hand-rolled `MinHash` in
//! `crate::knowledge_unlearning::dedup`). `corpus_curation`'s `lsh` feature
//! dependency exists specifically so this pillar can **compose**
//! [`crate::lsh_index::MinHashIndex`] instead of writing a fourth one: every
//! document's word-shingle set is inserted into one `MinHashIndex`, and each
//! document then queries that same index for its near-duplicate neighbours.
//! Only the shingling (turning text into the `Vec<u64>` element set the index
//! expects) is written here — the min-hash functions, banding, and candidate
//! gathering all live in [`crate::lsh_index`].
//!
//! Because candidates come from the index's banded buckets rather than an
//! all-pairs scan, this stays a banding-driven lookup (governed by
//! [`NearDupConfig::num_bands`]/[`NearDupConfig::rows_per_band`]) even though
//! each per-document query asks for up to the whole indexed set back — unlike
//! `crate::semantic_dedup`'s deliberately-exhaustive O(n²) comparison.
//!
//! # Distinct from `semantic_dedup`
//!
//! `crate::semantic_dedup` near-duplicate-clusters a **retrieved result
//! set** at query time. This pillar clusters the **corpus itself at ingest
//! time**, deciding what is admitted to the index in the first place.

use std::collections::HashMap;

use crate::lsh_index::{LshConfig, MinHashIndex};
use crate::types::Document;

use super::rng::fnv1a;
use super::types::CurationError;

// ── shingling ────────────────────────────────────────────────────────────────

/// Split `text` into lowercase alphanumeric tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Reduce `text` to a set of `shingle_size`-word shingle hashes, suitable for
/// [`MinHashIndex::insert`]/[`MinHashIndex::search`]. A document shorter than
/// `shingle_size` words becomes a single shingle over all of its words. A
/// document with no words at all yields an empty set.
fn shingle_hashes(text: &str, shingle_size: usize) -> Vec<u64> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return Vec::new();
    }
    let size = shingle_size.max(1);
    if tokens.len() < size {
        return vec![fnv1a(tokens.join(" ").as_bytes())];
    }
    tokens
        .windows(size)
        .map(|w| fnv1a(w.join(" ").as_bytes()))
        .collect()
}

// ── union-find ───────────────────────────────────────────────────────────────

/// Disjoint-set forest with path compression and union by size, used to turn
/// pairwise near-duplicate edges (from [`MinHashIndex`] queries) into
/// clusters.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        let (large, small) = if self.size[root_a] >= self.size[root_b] {
            (root_a, root_b)
        } else {
            (root_b, root_a)
        };
        self.parent[small] = large;
        self.size[large] += self.size[small];
    }
}

// ── NearDupConfig ────────────────────────────────────────────────────────────

/// Configuration for [`find_near_duplicate_clusters`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearDupConfig {
    /// Whether the near-duplicate pillar runs at all. Defaults to `true`.
    pub enabled: bool,
    /// Word-shingle window size. Defaults to `4`.
    pub shingle_size: usize,
    /// Number of `MinHash` banding groups (see [`LshConfig::num_bands`]).
    /// Defaults to `16`.
    pub num_bands: usize,
    /// Number of `MinHash` rows per band (see [`LshConfig::rows_per_band`]).
    /// Defaults to `4`. With the defaults above (`16 x 4 = 64` hash
    /// functions), the banding collision curve crosses 50% at a Jaccard
    /// similarity of `(1/16)^(1/4) ≈ 0.5`, sharply favouring recall for
    /// near-identical documents and precision against unrelated ones.
    pub rows_per_band: usize,
    /// Minimum estimated `Jaccard` similarity for two documents to be placed
    /// in the same cluster. Defaults to `0.5`.
    pub jaccard_threshold: f64,
}

impl Default for NearDupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            shingle_size: 4,
            num_bands: 16,
            rows_per_band: 4,
            jaccard_threshold: 0.5,
        }
    }
}

impl NearDupConfig {
    /// Construct a configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable the near-duplicate pillar.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the word-shingle window size.
    #[must_use]
    pub fn with_shingle_size(mut self, shingle_size: usize) -> Self {
        self.shingle_size = shingle_size;
        self
    }

    /// Set the number of `MinHash` banding groups.
    #[must_use]
    pub fn with_num_bands(mut self, num_bands: usize) -> Self {
        self.num_bands = num_bands;
        self
    }

    /// Set the number of `MinHash` rows per band.
    #[must_use]
    pub fn with_rows_per_band(mut self, rows_per_band: usize) -> Self {
        self.rows_per_band = rows_per_band;
        self
    }

    /// Set the minimum `Jaccard` similarity for clustering.
    #[must_use]
    pub fn with_jaccard_threshold(mut self, jaccard_threshold: f64) -> Self {
        self.jaccard_threshold = jaccard_threshold;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`CurationError::InvalidConfig`] when `shingle_size`,
    /// `num_bands`, or `rows_per_band` is zero, or when `jaccard_threshold` is
    /// non-finite or outside `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), CurationError> {
        if self.shingle_size == 0 {
            return Err(CurationError::InvalidConfig(
                "shingle_size must be > 0".into(),
            ));
        }
        if self.num_bands == 0 {
            return Err(CurationError::InvalidConfig("num_bands must be > 0".into()));
        }
        if self.rows_per_band == 0 {
            return Err(CurationError::InvalidConfig(
                "rows_per_band must be > 0".into(),
            ));
        }
        if !self.jaccard_threshold.is_finite() || !(0.0..=1.0).contains(&self.jaccard_threshold) {
            return Err(CurationError::InvalidConfig(format!(
                "jaccard_threshold must be finite and within [0, 1], got {}",
                self.jaccard_threshold
            )));
        }
        Ok(())
    }
}

// ── NearDuplicateCluster ─────────────────────────────────────────────────────

/// One cluster of mutually near-duplicate corpus documents.
#[derive(Debug, Clone, PartialEq)]
pub struct NearDuplicateCluster {
    /// The id of the cluster's kept representative (the lowest-index member
    /// in the input slice).
    pub representative_id: String,
    /// Every document id in the cluster, including the representative,
    /// ordered by ascending input index.
    pub member_ids: Vec<String>,
}

impl NearDuplicateCluster {
    /// The members other than [`Self::representative_id`], in the order they
    /// appear in [`Self::member_ids`].
    #[must_use]
    pub fn duplicates(&self) -> Vec<&str> {
        self.member_ids
            .iter()
            .map(String::as_str)
            .filter(|id| *id != self.representative_id)
            .collect()
    }
}

// ── clustering ───────────────────────────────────────────────────────────────

/// Cluster `docs` into groups of mutual near-duplicates, by composing
/// [`MinHashIndex`] (see the [module documentation](self)).
///
/// Every document is word-shingled and inserted into one `MinHashIndex`; each
/// document then queries that index for its own near-duplicate neighbours
/// (candidates gathered via the index's banding, re-ranked by exact
/// `Jaccard`), and pairs scoring at or above
/// [`NearDupConfig::jaccard_threshold`] are unioned into a cluster. Documents
/// with no extractable shingles (empty content) cannot be compared to
/// anything and never appear in a cluster. Only clusters with two or more
/// members are returned (singletons are not "duplicates" of anything), sorted
/// by ascending `representative_id`.
///
/// # Errors
///
/// Returns [`CurationError::NearDupIndexFailed`] when the underlying
/// [`MinHashIndex`] construction, insertion, or search fails.
pub fn find_near_duplicate_clusters(
    docs: &[Document],
    config: &NearDupConfig,
) -> Result<Vec<NearDuplicateCluster>, CurationError> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }

    let lsh_config = LshConfig::new()
        .with_num_bands(config.num_bands.max(1))
        .with_rows_per_band(config.rows_per_band.max(1));
    let mut index = MinHashIndex::new(lsh_config)
        .map_err(|e| CurationError::NearDupIndexFailed(e.to_string()))?;

    let shingles: Vec<Vec<u64>> = docs
        .iter()
        .map(|d| shingle_hashes(&d.content, config.shingle_size))
        .collect();

    let mut insertable: Vec<usize> = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if shingles[i].is_empty() {
            // No extractable shingles (empty content): cannot be judged
            // similar to anything, so it is simply never inserted.
            continue;
        }
        index
            .insert(doc.id.as_str(), shingles[i].clone())
            .map_err(|e| CurationError::NearDupIndexFailed(e.to_string()))?;
        insertable.push(i);
    }
    if index.is_empty() {
        return Ok(Vec::new());
    }

    let id_to_index: HashMap<&str, usize> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.as_str(), i))
        .collect();

    let mut forest = UnionFind::new(docs.len());
    for &i in &insertable {
        let hits = index
            .search(&shingles[i], index.len())
            .map_err(|e| CurationError::NearDupIndexFailed(e.to_string()))?;
        for hit in hits {
            if f64::from(hit.score) < config.jaccard_threshold {
                continue;
            }
            if let Some(&j) = id_to_index.get(hit.id.as_str())
                && j != i
            {
                forest.union(i, j);
            }
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in &insertable {
        let root = forest.find(i);
        groups.entry(root).or_default().push(i);
    }

    let mut clusters: Vec<NearDuplicateCluster> = groups
        .into_values()
        .filter(|members| members.len() >= 2)
        .map(|mut members| {
            members.sort_unstable();
            let representative_id = docs[members[0]].id.as_str().to_string();
            let member_ids = members
                .iter()
                .map(|&i| docs[i].id.as_str().to_string())
                .collect();
            NearDuplicateCluster {
                representative_id,
                member_ids,
            }
        })
        .collect();
    clusters.sort_by(|a, b| a.representative_id.cmp(&b.representative_id));

    Ok(clusters)
}
