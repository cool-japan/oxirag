//! RAG-Fusion: multi-query retrieval with Reciprocal Rank Fusion (RRF).
//!
//! RAG-Fusion expands a single user query into `num_queries` structural
//! variants, issues a separate retrieval request for each variant, and then
//! merges the per-variant ranked lists using the Reciprocal Rank Fusion
//! formula:
//!
//! ```text
//! rrf_score(doc) = Σ_i  1 / (k + rank_i(doc))
//! ```
//!
//! where `k` is a smoothing constant (typically 60), `rank_i(doc)` is the
//! 1-based rank of `doc` in the `i`-th result list, and the sum is over all
//! variant queries that returned `doc`.  Documents are deduplicated by their
//! [`DocumentId`] string; the final list is sorted by descending RRF score and
//! truncated to `top_k`.
//!
//! [`DocumentId`]: crate::types::DocumentId

use std::collections::HashMap;

use crate::layer1_echo::traits::Echo;
use crate::types::SearchResult;

use super::types::AdvancedRetrievalError;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the [`RagFusion`] retriever.
#[derive(Debug, Clone)]
pub struct RagFusionConfig {
    /// Number of query variants to generate (including the original).
    ///
    /// Defaults to `4`.
    pub num_queries: usize,

    /// RRF smoothing constant `k`.
    ///
    /// Larger values reduce the influence of very high-ranked documents.
    /// The original RRF paper recommends `60.0`.
    pub rrf_k: f32,

    /// Number of results to return after fusion.
    ///
    /// Defaults to `10`.
    pub top_k: usize,
}

impl Default for RagFusionConfig {
    fn default() -> Self {
        Self {
            num_queries: 4,
            rrf_k: 60.0,
            top_k: 10,
        }
    }
}

impl RagFusionConfig {
    /// Set the number of query variants.
    #[must_use]
    pub fn with_num_queries(mut self, num_queries: usize) -> Self {
        self.num_queries = num_queries;
        self
    }

    /// Set the RRF smoothing constant.
    #[must_use]
    pub fn with_rrf_k(mut self, rrf_k: f32) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    /// Set the number of results to return after fusion.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
}

// ── RagFusion ─────────────────────────────────────────────────────────────────

/// Retriever that implements RAG-Fusion.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "advanced-retrieval")]
/// # {
/// use oxirag::advanced_retrieval::{RagFusion, RagFusionConfig};
///
/// let fusion = RagFusion::new(RagFusionConfig::default());
/// let variants = fusion.generate_variants("What is Rust?");
/// assert!(!variants.is_empty());
/// assert_eq!(variants[0], "What is Rust?");
/// # }
/// ```
pub struct RagFusion {
    config: RagFusionConfig,
}

impl RagFusion {
    /// Create a new [`RagFusion`] retriever with the given configuration.
    #[must_use]
    pub fn new(config: RagFusionConfig) -> Self {
        Self { config }
    }

    /// Generate `num_queries` structural variants of `query`.
    ///
    /// The first element of the returned vector is always the original query.
    /// Subsequent elements are deterministic transformations that preserve the
    /// semantic intent while varying lexical form:
    ///
    /// 1. Original query (unchanged).
    /// 2. Reversed word order.
    /// 3. First half of tokens only (prefix).
    /// 4. Last half of tokens only (suffix).
    ///
    /// 5+ Question-word rotation / term permutations for additional variants.
    ///
    /// All generated variants are guaranteed to be non-empty.
    #[must_use]
    pub fn generate_variants(&self, query: &str) -> Vec<String> {
        let query = query.trim();
        let mut variants: Vec<String> = Vec::with_capacity(self.config.num_queries);

        // Variant 0: original
        variants.push(query.to_string());

        if self.config.num_queries <= 1 {
            return variants;
        }

        let words: Vec<&str> = query.split_whitespace().collect();

        // Variant 1: reversed word order
        let reversed = words.iter().rev().copied().collect::<Vec<_>>().join(" ");
        if reversed != query && !reversed.is_empty() {
            variants.push(reversed);
        } else {
            variants.push(format!("Tell me about: {query}"));
        }

        if variants.len() >= self.config.num_queries {
            return variants;
        }

        // Variant 2: prefix (first half)
        let mid = (words.len() / 2).max(1);
        let prefix = words[..mid].join(" ");
        if !prefix.is_empty() && !variants.contains(&prefix) {
            variants.push(prefix);
        } else {
            variants.push(format!("Explain {query}"));
        }

        if variants.len() >= self.config.num_queries {
            return variants;
        }

        // Variant 3: suffix (last half)
        let suffix_start = words.len().saturating_sub(mid);
        let suffix = words[suffix_start..].join(" ");
        if !suffix.is_empty() && !variants.contains(&suffix) {
            variants.push(suffix);
        } else {
            variants.push(format!("Describe {query}"));
        }

        if variants.len() >= self.config.num_queries {
            return variants;
        }

        // Variant 4+: question-word rotations and synonym-style expansions
        let question_prefixes = [
            "What is",
            "How does",
            "Why is",
            "When did",
            "Where can",
            "Who uses",
        ];
        let core = strip_question_word(query);
        for prefix in &question_prefixes {
            if variants.len() >= self.config.num_queries {
                break;
            }
            let candidate = format!("{prefix} {core}");
            if !variants.contains(&candidate) {
                variants.push(candidate);
            }
        }

        // If still not enough, generate numeric permutations of word pairs
        let mut i = 0usize;
        while variants.len() < self.config.num_queries {
            let candidate = format!("{query} [variant {i}]");
            if !variants.contains(&candidate) {
                variants.push(candidate);
            }
            i = i.wrapping_add(1);
            // safety exit: prevent infinite loop if num_queries is huge
            if i > self.config.num_queries * 4 {
                break;
            }
        }

        variants
    }

    /// Run RAG-Fusion: retrieve results for each generated query variant, then
    /// merge and rerank using Reciprocal Rank Fusion.
    ///
    /// # Errors
    ///
    /// Returns [`AdvancedRetrievalError::Search`] if any underlying retrieval
    /// call fails.
    pub async fn retrieve<E>(
        &self,
        query: &str,
        echo: &E,
    ) -> Result<Vec<SearchResult>, AdvancedRetrievalError>
    where
        E: Echo + ?Sized,
    {
        let variants = self.generate_variants(query);

        // Retrieve for each variant, collecting per-variant ranked lists.
        let mut all_lists: Vec<Vec<SearchResult>> = Vec::with_capacity(variants.len());
        for variant in &variants {
            let results = echo
                .search(variant, self.config.top_k, None)
                .await
                .map_err(|e| AdvancedRetrievalError::Search(e.to_string()))?;
            all_lists.push(results);
        }

        let fused = reciprocal_rank_fusion(&all_lists, self.config.rrf_k);

        Ok(fused.into_iter().take(self.config.top_k).collect())
    }
}

// ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

/// Merge multiple ranked result lists using Reciprocal Rank Fusion.
///
/// For each document `d` appearing in one or more of `result_lists`,
/// its fused score is:
///
/// ```text
/// rrf_score(d) = Σ_i  1 / (k + rank_i(d))
/// ```
///
/// where `rank_i(d)` is the 1-based position of `d` in list `i` (documents
/// not present in list `i` contribute 0).  The returned vector is sorted by
/// descending fused score; ties are broken by document ID string order.
///
/// Document identity is determined by [`DocumentId`]'s string representation.
///
/// [`DocumentId`]: crate::types::DocumentId
#[must_use]
pub fn reciprocal_rank_fusion(
    result_lists: &[Vec<SearchResult>],
    k: f32,
) -> Vec<SearchResult> {
    if result_lists.is_empty() {
        return Vec::new();
    }

    // Map doc_id_string → (cumulative_rrf_score, last_seen_SearchResult)
    let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

    for list in result_lists {
        for (zero_based_rank, sr) in list.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let rank_1based = (zero_based_rank + 1) as f32;
            let rrf_contribution = 1.0_f32 / (k + rank_1based);

            let doc_key = sr.document.id.as_str().to_string();
            scores
                .entry(doc_key)
                .and_modify(|(score, _)| *score += rrf_contribution)
                .or_insert((rrf_contribution, sr.clone()));
        }
    }

    // Collect, sort by descending score, assign new ranks.
    let mut fused: Vec<(f32, SearchResult)> = scores.into_values().collect();
    fused.sort_by(|(a_score, a_sr), (b_score, b_sr)| {
        b_score
            .partial_cmp(a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_sr.document.id.as_str().cmp(b_sr.document.id.as_str()))
    });

    fused
        .into_iter()
        .enumerate()
        .map(|(rank, (score, mut sr))| {
            sr.score = score;
            sr.rank = rank;
            sr
        })
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Remove common question-word prefixes from a query string for variant
/// generation purposes.
fn strip_question_word(query: &str) -> &str {
    let lower = query.to_lowercase();
    let prefixes = [
        "what is ", "what are ", "how does ", "how do ", "why is ", "why are ",
        "when did ", "when does ", "where can ", "where is ", "who is ", "who uses ",
        "tell me about ", "explain ", "describe ",
    ];
    for prefix in &prefixes {
        if lower.starts_with(prefix) {
            return &query[prefix.len()..];
        }
    }
    query
}
