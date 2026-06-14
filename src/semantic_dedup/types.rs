//! Types for the `semantic_dedup` module.

use thiserror::Error;

// ── DedupMethod ───────────────────────────────────────────────────────────────

/// Locality-sensitive hashing strategy used to detect near-duplicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DedupMethod {
    /// `SimHash` with `Hamming`-distance thresholding.
    ///
    /// Robust to small token-level edits; cheap single-`u64` fingerprints.
    #[default]
    SimHash,
    /// `MinHash` signatures with `Jaccard`-estimate thresholding.
    ///
    /// Estimates set overlap over `shingle_size`-word shingles.
    MinHash,
}

// ── KeepPolicy ────────────────────────────────────────────────────────────────

/// Policy that selects which member of a near-duplicate cluster to retain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepPolicy {
    /// Keep the earliest item (lowest original index) in the cluster.
    #[default]
    First,
    /// Keep the item with the highest [`crate::types::SearchResult::score`].
    ///
    /// Ties are broken by the lowest original index.
    HighestScore,
    /// Keep the item with the longest content (most characters).
    ///
    /// Ties are broken by the lowest original index.
    Longest,
}

// ── SemanticDedupConfig ───────────────────────────────────────────────────────

/// Configuration for [`SemanticDeduplicator`].
///
/// [`SemanticDeduplicator`]: crate::semantic_dedup::SemanticDeduplicator
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticDedupConfig {
    /// Which LSH method drives duplicate detection.
    pub method: DedupMethod,
    /// Number of words per shingle for `MinHash`. Defaults to `2`.
    pub shingle_size: usize,
    /// Number of permutations (signature length) for `MinHash`. Defaults to `64`.
    pub num_perm: usize,
    /// Maximum `Hamming` distance for two `SimHash` fingerprints to be duplicates.
    ///
    /// Defaults to `3`.
    pub simhash_max_hamming: u32,
    /// Minimum `Jaccard` estimate for two `MinHash` signatures to be duplicates.
    ///
    /// Defaults to `0.8`.
    pub minhash_min_jaccard: f32,
    /// Policy selecting the representative kept from each cluster.
    pub keep: KeepPolicy,
}

impl Default for SemanticDedupConfig {
    fn default() -> Self {
        Self {
            method: DedupMethod::SimHash,
            shingle_size: 2,
            num_perm: 64,
            simhash_max_hamming: 3,
            minhash_min_jaccard: 0.8,
            keep: KeepPolicy::First,
        }
    }
}

impl SemanticDedupConfig {
    /// Create a new config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the LSH method.
    #[must_use]
    pub fn with_method(mut self, method: DedupMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the `MinHash` shingle size (in words). Values below `1` are clamped to `1`.
    #[must_use]
    pub fn with_shingle_size(mut self, shingle_size: usize) -> Self {
        self.shingle_size = shingle_size.max(1);
        self
    }

    /// Set the `MinHash` signature length. Values below `1` are clamped to `1`.
    #[must_use]
    pub fn with_num_perm(mut self, num_perm: usize) -> Self {
        self.num_perm = num_perm.max(1);
        self
    }

    /// Set the maximum `SimHash` `Hamming` distance for duplicates.
    #[must_use]
    pub fn with_simhash_max_hamming(mut self, simhash_max_hamming: u32) -> Self {
        self.simhash_max_hamming = simhash_max_hamming;
        self
    }

    /// Set the minimum `MinHash` `Jaccard` estimate for duplicates.
    #[must_use]
    pub fn with_minhash_min_jaccard(mut self, minhash_min_jaccard: f32) -> Self {
        self.minhash_min_jaccard = minhash_min_jaccard;
        self
    }

    /// Set the cluster keep policy.
    #[must_use]
    pub fn with_keep(mut self, keep: KeepPolicy) -> Self {
        self.keep = keep;
        self
    }
}

impl Default for crate::semantic_dedup::SemanticDeduplicator {
    fn default() -> Self {
        Self::new(SemanticDedupConfig::default())
    }
}

// ── SemanticDedupError ────────────────────────────────────────────────────────

/// Errors from the `semantic_dedup` module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SemanticDedupError {
    /// The supplied result set was empty.
    #[error("empty input")]
    EmptyInput,
}
