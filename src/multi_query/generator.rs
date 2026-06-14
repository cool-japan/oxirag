//! [`MultiQueryGenerator`] — multi-query retrieval with RRF fusion.
//!
//! The generator follows a three-stage pipeline:
//!
//! 1. **Expansion** — a [`QueryVariantGenerator`] produces `num_variants`
//!    paraphrase variants from the original query.
//! 2. **Per-variant retrieval** — each variant is scored against the document
//!    corpus using a BM25-lite (token-overlap) heuristic and the top `top_k`
//!    documents are collected.
//! 3. **RRF fusion** — all per-variant ranked lists are merged with Reciprocal
//!    Rank Fusion: `rrf_score(doc) = Σ 1 / (k + rank)` summed over every
//!    variant list that includes `doc`.  The final result is sorted by
//!    descending RRF score.

use std::collections::{HashMap, HashSet};

use crate::multi_query::types::{
    GeneratedQuery, MultiQueryConfig, MultiQueryError, MultiQueryHit, MultiQueryResult,
    QueryVariantGenerator,
};

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokenize `text` into lowercase alphanumeric tokens of length ≥ 2.
///
/// The exact split predicate is `!c.is_alphanumeric()`, matching the canonical
/// tokenizer specified for the `multi_query` module.
#[inline]
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── BM25-lite scoring ─────────────────────────────────────────────────────────

/// Compute a lightweight token-overlap score between a query and a document.
///
/// ```text
/// score = |unique_query_tokens ∩ doc_token_set| / |unique_query_tokens|
/// ```
///
/// Returns `0.0` when the query token set is empty.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn bm25_lite_score(query_tokens: &[String], doc_token_set: &HashSet<String>) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let unique_query: HashSet<&str> = query_tokens.iter().map(String::as_str).collect();
    let matching = unique_query
        .iter()
        .filter(|&&t| doc_token_set.contains(t))
        .count();
    matching as f64 / unique_query.len() as f64
}

// ── MultiQueryGenerator ───────────────────────────────────────────────────────

/// Multi-Query RAG retriever with Reciprocal Rank Fusion.
///
/// Expands a single query into `num_variants` variants using a
/// [`QueryVariantGenerator`], scores every document in the corpus against
/// each variant with BM25-lite (token overlap), and fuses the per-variant
/// ranked lists with RRF.
///
/// The generator is constructed once and reused across many [`retrieve`] calls.
/// The underlying [`QueryVariantGenerator`] is held as a trait object so that
/// callers can swap strategies without changing call sites.
///
/// [`retrieve`]: MultiQueryGenerator::retrieve
pub struct MultiQueryGenerator {
    /// Retrieval and fusion configuration.
    pub config: MultiQueryConfig,
    variant_gen: Box<dyn QueryVariantGenerator + Send + Sync>,
}

impl std::fmt::Debug for MultiQueryGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiQueryGenerator")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl MultiQueryGenerator {
    /// Create a new generator with the given `config` and `variant_gen`.
    #[must_use]
    pub fn new(
        config: MultiQueryConfig,
        variant_gen: impl QueryVariantGenerator + Send + Sync + 'static,
    ) -> Self {
        Self {
            config,
            variant_gen: Box::new(variant_gen),
        }
    }

    /// Retrieve documents from `docs` for `query` using multi-query RRF fusion.
    ///
    /// **Pipeline:**
    ///
    /// 1. Generate `config.num_variants` variant queries via `variant_gen`.
    /// 2. For each variant, score all `docs` by BM25-lite and keep the `top_k`
    ///    highest-scoring documents.
    /// 3. Apply RRF: each document accumulates
    ///    `1.0 / (rrf_k + rank_1based)` from every variant list that contains
    ///    it.
    /// 4. Sort the fused list by descending RRF score (ties broken by
    ///    ascending document id) and truncate to `top_k` results.
    ///
    /// `docs` is a slice of `(id, content)` pairs.  Document ids must be
    /// unique within the slice; if duplicates exist, only the content
    /// associated with the *first* occurrence is retained in the output.
    ///
    /// # Errors
    ///
    /// - [`MultiQueryError::EmptyCorpus`] when `docs` is empty.
    /// - [`MultiQueryError::GenerationFailed`] when the variant generator
    ///   cannot produce variants (e.g. empty query).
    pub fn retrieve(
        &self,
        query: &str,
        docs: &[(String, String)],
    ) -> Result<MultiQueryResult, MultiQueryError> {
        if docs.is_empty() {
            return Err(MultiQueryError::EmptyCorpus);
        }

        // ── Stage 1: expand ───────────────────────────────────────────────────
        let variant_texts = self
            .variant_gen
            .generate_variants(query, self.config.num_variants)?;

        let generated_queries: Vec<GeneratedQuery> = variant_texts
            .iter()
            .enumerate()
            .map(|(idx, text)| GeneratedQuery::new(text.clone(), idx))
            .collect();

        // Bail out early when num_variants is 0.
        if variant_texts.is_empty() || self.config.top_k == 0 {
            return Ok(MultiQueryResult {
                original_query: query.to_string(),
                generated_queries,
                hits: Vec::new(),
            });
        }

        // ── Pre-tokenize corpus ───────────────────────────────────────────────
        let doc_token_sets: Vec<HashSet<String>> = docs
            .iter()
            .map(|(_, content)| tokenize(content).into_iter().collect())
            .collect();

        // Content lookup (first occurrence wins for duplicate ids).
        let mut doc_content_map: HashMap<&str, &str> = HashMap::with_capacity(docs.len());
        for (id, content) in docs {
            doc_content_map
                .entry(id.as_str())
                .or_insert(content.as_str());
        }

        // ── Stage 2 & 3: per-variant retrieval + RRF accumulation ─────────────
        let mut rrf_scores: HashMap<&str, f64> = HashMap::new();

        for variant_text in &variant_texts {
            let q_tokens = tokenize(variant_text);

            // Score every document.
            let mut scored: Vec<(usize, f64)> = docs
                .iter()
                .enumerate()
                .map(|(doc_idx, _)| {
                    let score = bm25_lite_score(&q_tokens, &doc_token_sets[doc_idx]);
                    (doc_idx, score)
                })
                .collect();

            // Sort descending by score; break ties by ascending document id.
            scored.sort_by(|&(ia, sa), &(ib, sb)| {
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| docs[ia].0.as_str().cmp(docs[ib].0.as_str()))
            });

            // Accumulate RRF for each top-k slot.
            #[allow(clippy::cast_precision_loss)]
            for (rank, (doc_idx, _score)) in scored.iter().take(self.config.top_k).enumerate() {
                let rank_1based = (rank + 1) as f64;
                let contribution = 1.0 / (self.config.rrf_k as f64 + rank_1based);
                let doc_id: &str = docs[*doc_idx].0.as_str();
                *rrf_scores.entry(doc_id).or_insert(0.0) += contribution;
            }
        }

        // ── Stage 4: build, sort, and truncate ───────────────────────────────
        let mut hits: Vec<MultiQueryHit> = rrf_scores
            .into_iter()
            .map(|(id, rrf_score)| MultiQueryHit {
                id: id.to_string(),
                content: doc_content_map
                    .get(id)
                    .copied()
                    .unwrap_or_default()
                    .to_string(),
                rrf_score,
            })
            .collect();

        hits.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        hits.truncate(self.config.top_k);

        Ok(MultiQueryResult {
            original_query: query.to_string(),
            generated_queries,
            hits,
        })
    }
}
