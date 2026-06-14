//! The [`SemanticDeduplicator`] orchestrator.
//!
//! Combines `SimHash` and `MinHash` detectors with single-linkage (union-find)
//! clustering to remove near-duplicate passages from a retrieved set while
//! preserving the relative order of the items that are kept.

use crate::types::SearchResult;

use super::minhash::{jaccard, minhash_signature};
use super::simhash::{hamming, simhash};
use super::types::{DedupMethod, KeepPolicy, SemanticDedupConfig, SemanticDedupError};

// ── union-find ────────────────────────────────────────────────────────────────

/// Disjoint-set forest with path compression and union by size.
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
        // Path compression.
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (large, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = large;
        self.size[large] += self.size[small];
    }
}

// ── SemanticDeduplicator ──────────────────────────────────────────────────────

/// Detects and removes near-duplicate passages via locality-sensitive hashing.
#[derive(Debug, Clone)]
pub struct SemanticDeduplicator {
    /// Configuration governing method, thresholds, and keep policy.
    pub config: SemanticDedupConfig,
}

impl SemanticDeduplicator {
    /// Create a new deduplicator with the given configuration.
    #[must_use]
    pub fn new(config: SemanticDedupConfig) -> Self {
        Self { config }
    }

    /// Compute the 64-bit `SimHash` fingerprint of `text`.
    #[must_use]
    pub fn simhash(&self, text: &str) -> u64 {
        simhash(text)
    }

    /// Compute the `MinHash` signature of `text` using the configured parameters.
    #[must_use]
    pub fn minhash_signature(&self, text: &str) -> Vec<u64> {
        minhash_signature(text, self.config.shingle_size, self.config.num_perm)
    }

    /// Estimate the `Jaccard` similarity between two `MinHash` signatures.
    #[must_use]
    pub fn jaccard(&self, a: &[u64], b: &[u64]) -> f32 {
        jaccard(a, b)
    }

    /// `Hamming` distance between two `SimHash` fingerprints.
    #[must_use]
    pub fn hamming(a: u64, b: u64) -> u32 {
        hamming(a, b)
    }

    /// Return `true` when `a` and `b` are near-duplicates under the active method.
    #[must_use]
    pub fn is_duplicate(&self, a: &str, b: &str) -> bool {
        match self.config.method {
            DedupMethod::SimHash => {
                let ha = self.simhash(a);
                let hb = self.simhash(b);
                hamming(ha, hb) <= self.config.simhash_max_hamming
            }
            DedupMethod::MinHash => {
                let sa = self.minhash_signature(a);
                let sb = self.minhash_signature(b);
                jaccard(&sa, &sb) >= self.config.minhash_min_jaccard
            }
        }
    }

    /// Cluster `texts` by single-linkage near-duplicate detection.
    ///
    /// Each inner vector holds the indices (into `texts`) of one cluster, sorted
    /// ascending. Clusters themselves are ordered by their smallest member index,
    /// so singletons and duplicate groups appear in stable input order.
    #[must_use]
    pub fn cluster(&self, texts: &[&str]) -> Vec<Vec<usize>> {
        let n = texts.len();
        if n == 0 {
            return Vec::new();
        }

        // Pre-compute fingerprints / signatures once per item.
        let simhashes: Vec<u64> = match self.config.method {
            DedupMethod::SimHash => texts.iter().map(|t| simhash(t)).collect(),
            DedupMethod::MinHash => Vec::new(),
        };
        let signatures: Vec<Vec<u64>> = match self.config.method {
            DedupMethod::MinHash => texts.iter().map(|t| self.minhash_signature(t)).collect(),
            DedupMethod::SimHash => Vec::new(),
        };

        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let dup = match self.config.method {
                    DedupMethod::SimHash => {
                        hamming(simhashes[i], simhashes[j]) <= self.config.simhash_max_hamming
                    }
                    DedupMethod::MinHash => {
                        jaccard(&signatures[i], &signatures[j]) >= self.config.minhash_min_jaccard
                    }
                };
                if dup {
                    uf.union(i, j);
                }
            }
        }

        // Bucket indices by representative root, preserving input order.
        let mut roots: Vec<usize> = Vec::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let root = uf.find(i);
            if let Some(pos) = roots.iter().position(|&r| r == root) {
                clusters[pos].push(i);
            } else {
                roots.push(root);
                clusters.push(vec![i]);
            }
        }
        clusters
    }

    /// Pick the surviving index from a cluster according to [`KeepPolicy`].
    fn representative(&self, cluster: &[usize], results: &[SearchResult]) -> usize {
        match self.config.keep {
            KeepPolicy::First => cluster.iter().copied().min().unwrap_or(0),
            KeepPolicy::HighestScore => cluster
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let sa = results[a].score;
                    let sb = results[b].score;
                    sa.partial_cmp(&sb)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Lower index wins ties.
                        .then(b.cmp(&a))
                })
                .unwrap_or(0),
            KeepPolicy::Longest => cluster
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let la = results[a].document.content.chars().count();
                    let lb = results[b].document.content.chars().count();
                    // Lower index wins ties.
                    la.cmp(&lb).then(b.cmp(&a))
                })
                .unwrap_or(0),
        }
    }

    /// Remove near-duplicates from `results`, keeping one item per cluster.
    ///
    /// The kept items preserve their relative input order and are re-ranked from
    /// `0`. An empty input returns an empty vector.
    #[must_use]
    pub fn deduplicate(&self, results: &[SearchResult]) -> Vec<SearchResult> {
        if results.is_empty() {
            return Vec::new();
        }

        let texts: Vec<&str> = results
            .iter()
            .map(|r| r.document.content.as_str())
            .collect();
        let clusters = self.cluster(&texts);

        // Mark the surviving index of each cluster.
        let mut keep = vec![false; results.len()];
        for cluster in &clusters {
            let rep = self.representative(cluster, results);
            keep[rep] = true;
        }

        let mut kept: Vec<SearchResult> = Vec::new();
        for (i, result) in results.iter().enumerate() {
            if keep[i] {
                let mut r = result.clone();
                r.rank = kept.len();
                kept.push(r);
            }
        }
        kept
    }

    /// Like [`deduplicate`] but returns an error for empty input.
    ///
    /// [`deduplicate`]: Self::deduplicate
    ///
    /// # Errors
    ///
    /// Returns [`SemanticDedupError::EmptyInput`] when `results` is empty.
    pub fn deduplicate_checked(
        &self,
        results: &[SearchResult],
    ) -> Result<Vec<SearchResult>, SemanticDedupError> {
        if results.is_empty() {
            return Err(SemanticDedupError::EmptyInput);
        }
        Ok(self.deduplicate(results))
    }
}
