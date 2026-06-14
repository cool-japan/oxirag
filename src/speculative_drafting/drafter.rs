//! [`SpeculativeDrafter`] — cluster the corpus, draft per cluster in parallel,
//! verify, then combine with self-consistency.

use crate::speculative_drafting::cluster::{embed, kmeans_lite};
use crate::speculative_drafting::types::{
    DraftCandidate, DraftVerifier, Drafter, SpecDraftConfig, SpecDraftError, SpeculativeOutput,
    token_jaccard,
};
use crate::types::Document;

// ── SpeculativeDrafter ───────────────────────────────────────────────────────────

/// Speculative RAG drafter (Wang et al., 2024).
///
/// The retrieved corpus is partitioned into diverse document clusters; one
/// draft answer is generated per non-empty cluster (each draft sees a distinct
/// perspective). Drafts are then **verified** (a support score) and combined
/// with **self-consistency** (mean agreement with the other drafts); the
/// highest-scoring draft wins.
///
/// This is distinct from self-consistency decoding, which samples reasoning
/// paths from a single query — here the diversity comes from *disjoint document
/// clusters* rather than from stochastic decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeDrafter {
    /// Configuration for this drafter.
    pub config: SpecDraftConfig,
}

impl SpeculativeDrafter {
    /// Create a new drafter with the given configuration.
    #[must_use]
    pub fn new(config: SpecDraftConfig) -> Self {
        Self { config }
    }

    /// Partition `docs` into at most `config.num_clusters` non-overlapping groups.
    ///
    /// Each document is embedded with a deterministic FNV-1a pseudo-embedding of
    /// dimension `config.dim`, then grouped by a k-means-lite pass with
    /// deterministic centroid spreading. The returned vector contains only
    /// non-empty clusters; together they form a partition of `0..docs.len()`,
    /// each inner vector holding indices *into* `docs`.
    #[must_use]
    pub fn cluster_docs(&self, docs: &[Document]) -> Vec<Vec<usize>> {
        if docs.is_empty() {
            return Vec::new();
        }
        let embeddings: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| embed(&combined_text(d), self.config.dim))
            .collect();
        kmeans_lite(&embeddings, self.config.num_clusters.max(1))
    }

    /// Run speculative drafting end to end.
    ///
    /// Steps: cluster the corpus → generate one draft per non-empty cluster (in
    /// parallel) → verify each draft for support → compute each draft's
    /// self-consistency as the mean token-Jaccard agreement with the *other*
    /// drafts → blend `total = verify_weight * support + consistency_weight *
    /// self_consistency` → select the draft with the highest total. Confidence
    /// is the winning draft's total score.
    ///
    /// A lone draft has no peers to agree with, so its self-consistency is
    /// defined as `0.0` and its total reduces to the support contribution.
    ///
    /// # Errors
    ///
    /// - [`SpecDraftError::EmptyQuery`] when `query` is empty after trimming.
    /// - [`SpecDraftError::EmptyCorpus`] when `docs` is empty.
    pub fn run<D: Drafter, V: DraftVerifier>(
        &self,
        query: &str,
        docs: &[Document],
        drafter: &D,
        verifier: &V,
    ) -> Result<SpeculativeOutput, SpecDraftError> {
        if query.trim().is_empty() {
            return Err(SpecDraftError::EmptyQuery);
        }
        if docs.is_empty() {
            return Err(SpecDraftError::EmptyCorpus);
        }

        let clusters = self.cluster_docs(docs);

        // One draft + support score per cluster, produced in parallel. The
        // `Sync` bounds on the traits make sharing `drafter`/`verifier` across
        // threads sound; results are collected back in cluster order.
        let cluster_drafts: Vec<(String, f32)> = std::thread::scope(|scope| {
            let handles: Vec<_> = clusters
                .iter()
                .map(|members| {
                    scope.spawn(move || {
                        let subset: Vec<&Document> = members.iter().map(|&i| &docs[i]).collect();
                        let content = drafter.draft(query, &subset);
                        let support = clamp01(verifier.verify(query, &content, &subset));
                        (content, support)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| (String::new(), 0.0)))
                .collect()
        });

        // Self-consistency: mean token-Jaccard agreement with the other drafts.
        let contents: Vec<&str> = cluster_drafts.iter().map(|(c, _)| c.as_str()).collect();
        let mut candidates: Vec<DraftCandidate> = cluster_drafts
            .iter()
            .enumerate()
            .map(|(idx, (content, support))| {
                let self_consistency = mean_agreement(idx, &contents);
                let total = self.config.verify_weight * support
                    + self.config.consistency_weight * self_consistency;
                DraftCandidate {
                    content: content.clone(),
                    cluster_id: idx,
                    support_score: *support,
                    self_consistency,
                    total_score: total,
                }
            })
            .collect();

        // Deterministic ranking: total desc, then support desc, then cluster asc.
        candidates.sort_by(candidate_order);

        let (best, confidence) = match candidates.first() {
            Some(winner) => (winner.content.clone(), winner.total_score),
            None => (String::new(), 0.0),
        };

        Ok(SpeculativeOutput {
            best,
            drafts: candidates,
            confidence,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────────

/// Concatenate a document's title (if any) and content for embedding.
fn combined_text(doc: &Document) -> String {
    match &doc.title {
        Some(title) => format!("{title} {}", doc.content),
        None => doc.content.clone(),
    }
}

/// Mean token-Jaccard agreement between draft `idx` and every *other* draft.
///
/// Returns `0.0` when there are fewer than two drafts (a lone draft has no
/// peers to agree with).
fn mean_agreement(idx: usize, contents: &[&str]) -> f32 {
    if contents.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (other, text) in contents.iter().enumerate() {
        if other == idx {
            continue;
        }
        sum += token_jaccard(contents[idx], text);
    }
    #[allow(clippy::cast_precision_loss)]
    let denom = (contents.len() - 1) as f32;
    sum / denom
}

/// Clamp a score into `[0, 1]`.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Total order over candidates: total score desc, support desc, cluster id asc.
fn candidate_order(a: &DraftCandidate, b: &DraftCandidate) -> std::cmp::Ordering {
    b.total_score
        .partial_cmp(&a.total_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| {
            b.support_score
                .partial_cmp(&a.support_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.cluster_id.cmp(&b.cluster_id))
}
