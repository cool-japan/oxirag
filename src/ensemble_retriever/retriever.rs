//! The [`SubRetriever`] trait, the [`LexicalSubRetriever`] strategy, and the
//! [`EnsembleRetriever`] orchestrator.
//!
//! [`EnsembleRetriever`] runs several pluggable [`SubRetriever`] strategies —
//! each registered with a weight — and fuses their candidate lists into a
//! single ranking using the [`EnsembleFusion`] strategy selected in its
//! [`EnsembleConfig`]. A document surfaced by several retrievers accumulates a
//! contribution from each, so cross-retriever agreement lifts its rank.
//!
//! This differs from `crate::rank_fusion`, which fuses *given* ranked lists:
//! here the ensemble owns the retrievers and *produces* the lists before fusing.
//!
//! [`EnsembleFusion`]: crate::ensemble_retriever::EnsembleFusion
//! [`EnsembleConfig`]: crate::ensemble_retriever::EnsembleConfig

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::types::{Document, DocumentId};

use super::types::{EnsembleConfig, EnsembleError, EnsembleFusion};

/// A pluggable retrieval strategy that the ensemble can orchestrate.
///
/// Implementations return scored candidate documents for a query. The ensemble
/// treats every implementation uniformly, so dense vector search, sparse lexical
/// search, metadata filters, or any custom strategy can all participate behind
/// this single trait.
pub trait SubRetriever {
    /// Human-readable retriever name.
    fn name(&self) -> &str;

    /// Return (doc id, score) candidates for the query.
    ///
    /// At most `top_k` candidates should be returned, ordered best-first. Scores
    /// are arbitrary per-retriever magnitudes; the ensemble normalizes or ranks
    /// them as dictated by its fusion strategy, so the scales need not align
    /// across retrievers.
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<(DocumentId, f32)>;
}

/// A [`SubRetriever`] that scores documents by lexical token overlap.
///
/// Each document is scored by the size of the intersection between the query's
/// token set and the document's token set (the count of distinct shared tokens),
/// so documents that mention more of the query's terms rank higher. Tokens are
/// lowercased alphanumeric runs of length two or more; scoring is deterministic
/// and requires no model, randomness, or external state.
#[derive(Debug, Clone, Default)]
pub struct LexicalSubRetriever {
    /// The retriever's human-readable name.
    name: String,
    /// The corpus this retriever searches over.
    docs: Vec<Document>,
}

impl LexicalSubRetriever {
    /// Create a new lexical sub-retriever over `docs` with the given `name`.
    #[must_use]
    pub fn new(name: impl Into<String>, docs: Vec<Document>) -> Self {
        Self {
            name: name.into(),
            docs,
        }
    }

    /// Number of documents in this retriever's corpus.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.docs.len()
    }
}

impl SubRetriever for LexicalSubRetriever {
    fn name(&self) -> &str {
        &self.name
    }

    fn retrieve(&self, query: &str, top_k: usize) -> Vec<(DocumentId, f32)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(DocumentId, f32)> = self
            .docs
            .iter()
            .filter_map(|doc| {
                let doc_tokens = tokenize(&doc.content);
                let overlap = query_tokens.intersection(&doc_tokens).count();
                if overlap == 0 {
                    None
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    Some((doc.id.clone(), overlap as f32))
                }
            })
            .collect();

        // Sort best-first with a deterministic id tie-break.
        scored.sort_by(|(a_id, a_score), (b_id, b_score)| {
            b_score
                .partial_cmp(a_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a_id.as_str().cmp(b_id.as_str()))
        });

        if top_k > 0 && scored.len() > top_k {
            scored.truncate(top_k);
        }
        scored
    }
}

/// Orchestrates multiple weighted [`SubRetriever`] strategies and fuses their
/// candidate lists into a single ranking.
///
/// Register retrievers with [`EnsembleRetriever::add_retriever`], then call
/// [`EnsembleRetriever::retrieve`] to run every retriever and fuse the results
/// according to the configured [`EnsembleFusion`] strategy.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "ensemble-retriever")]
/// # {
/// use oxirag::ensemble_retriever::{
///     EnsembleConfig, EnsembleRetriever, LexicalSubRetriever, SubRetriever,
/// };
/// use oxirag::types::Document;
///
/// let docs = vec![
///     Document::new("the quick brown fox").with_id("a"),
///     Document::new("a lazy brown dog").with_id("b"),
/// ];
/// let lexical = LexicalSubRetriever::new("lexical", docs);
///
/// let mut ensemble = EnsembleRetriever::new(EnsembleConfig::new());
/// ensemble.add_retriever(Box::new(lexical), 1.0);
/// assert_eq!(ensemble.retriever_count(), 1);
///
/// let ranked = ensemble.retrieve("brown fox", 10).unwrap();
/// assert_eq!(ranked[0].0.as_str(), "a");
/// # }
/// ```
pub struct EnsembleRetriever {
    /// The configuration controlling the fusion strategy and result limit.
    config: EnsembleConfig,
    /// The registered sub-retrievers paired with their fusion weights.
    retrievers: Vec<(Box<dyn SubRetriever>, f32)>,
}

impl EnsembleRetriever {
    /// Create a new ensemble retriever with the given configuration and no
    /// registered sub-retrievers.
    #[must_use]
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            retrievers: Vec::new(),
        }
    }

    /// Borrow the configuration backing this ensemble.
    #[must_use]
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Register a sub-retriever with the given fusion `weight`.
    ///
    /// Heavier weights amplify the retriever's contribution to every document's
    /// fused score. Retrievers can be added incrementally; ordering only affects
    /// tie-breaking among equal-scoring documents indirectly via their ids.
    pub fn add_retriever(&mut self, retriever: Box<dyn SubRetriever>, weight: f32) {
        self.retrievers.push((retriever, weight));
    }

    /// Number of registered sub-retrievers.
    #[must_use]
    pub fn retriever_count(&self) -> usize {
        self.retrievers.len()
    }

    /// Run every sub-retriever for `query` and fuse their candidates into a
    /// single ranking of `(DocumentId, fused_score)` pairs.
    ///
    /// Each retriever is asked for up to `top_k` candidates. Their lists are then
    /// fused according to [`EnsembleConfig::fusion`]:
    ///
    /// * [`EnsembleFusion::WeightedScore`] sums `weight * min_max_normalized_score`
    ///   over the retrievers that surfaced the document.
    /// * [`EnsembleFusion::WeightedRrf`] sums `weight * 1 / (rrf_k + rank + 1)`
    ///   over the retrievers that surfaced the document, using the 0-based rank.
    ///
    /// The fused pairs are sorted by descending score with a deterministic
    /// [`DocumentId`]-string tie-break (ascending) and truncated to
    /// [`EnsembleConfig::top_n`] (`0` = all).
    ///
    /// # Errors
    ///
    /// * [`EnsembleError::NoRetrievers`] if no sub-retrievers are registered.
    /// * [`EnsembleError::EmptyQuery`] if `query` is empty or whitespace-only.
    ///
    /// [`DocumentId`]: crate::types::DocumentId
    pub fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(DocumentId, f32)>, EnsembleError> {
        if self.retrievers.is_empty() {
            return Err(EnsembleError::NoRetrievers);
        }
        if query.trim().is_empty() {
            return Err(EnsembleError::EmptyQuery);
        }

        let lists: Vec<(Vec<(DocumentId, f32)>, f32)> = self
            .retrievers
            .iter()
            .map(|(retriever, weight)| (retriever.retrieve(query, top_k), *weight))
            .collect();

        let fused = match self.config.fusion {
            EnsembleFusion::WeightedScore => Self::fuse_weighted_score(&lists),
            EnsembleFusion::WeightedRrf => self.fuse_weighted_rrf(&lists),
        };

        Ok(self.sort_and_truncate(fused))
    }

    // ── Internal fusion helpers ─────────────────────────────────────────────────

    /// Fuse via weighted min-max-normalized score summation.
    fn fuse_weighted_score(lists: &[(Vec<(DocumentId, f32)>, f32)]) -> HashMap<DocumentId, f32> {
        let mut acc: HashMap<DocumentId, f32> = HashMap::new();
        for (list, weight) in lists {
            for (id, score) in min_max_normalize(list) {
                *acc.entry(id).or_insert(0.0) += weight * score;
            }
        }
        acc
    }

    /// Fuse via weighted reciprocal rank summation.
    fn fuse_weighted_rrf(
        &self,
        lists: &[(Vec<(DocumentId, f32)>, f32)],
    ) -> HashMap<DocumentId, f32> {
        let mut acc: HashMap<DocumentId, f32> = HashMap::new();
        for (list, weight) in lists {
            for (rank, (id, _score)) in list.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let rank_1based = (rank + 1) as f32;
                let contribution = weight * (1.0 / (self.config.rrf_k + rank_1based));
                *acc.entry(id.clone()).or_insert(0.0) += contribution;
            }
        }
        acc
    }

    /// Sort fused scores descending (deterministic id tie-break) and truncate to
    /// `top_n`.
    fn sort_and_truncate(&self, fused: HashMap<DocumentId, f32>) -> Vec<(DocumentId, f32)> {
        let mut ranked: Vec<(DocumentId, f32)> = fused.into_iter().collect();
        ranked.sort_by(|(a_id, a_score), (b_id, b_score)| {
            b_score
                .partial_cmp(a_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a_id.as_str().cmp(b_id.as_str()))
        });
        if self.config.top_n > 0 && ranked.len() > self.config.top_n {
            ranked.truncate(self.config.top_n);
        }
        ranked
    }
}

// ── Free functions ─────────────────────────────────────────────────────────────

/// Tokenize text into a set of lowercased alphanumeric tokens of length >= 2.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Min-max normalize a single retriever's scores onto `[0, 1]`.
///
/// Document ids are preserved verbatim; only scores are transformed. A list with
/// a single distinct score (zero range) maps every score to `1.0`, and an empty
/// list yields an empty result.
fn min_max_normalize(list: &[(DocumentId, f32)]) -> Vec<(DocumentId, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (_, s) in list {
        if *s < min {
            min = *s;
        }
        if *s > max {
            max = *s;
        }
    }
    let range = max - min;
    list.iter()
        .map(|(id, s)| {
            let scaled = if range == 0.0 { 1.0 } else { (s - min) / range };
            (id.clone(), scaled)
        })
        .collect()
}
