//! [`SpeculativeDrafter`] — cluster the corpus, draft per cluster (or
//! paper-faithful subset) in parallel, verify, then combine with
//! self-consistency and optional self-reflection.

use crate::speculative_drafting::cluster::{embed, kmeans_lite};
use crate::speculative_drafting::types::{
    DraftCandidate, DraftSubsetStrategy, DraftVerifier, Drafter, SpecDraftConfig, SpecDraftError,
    SpeculativeOutput, token_jaccard,
};
use crate::types::Document;

// ── SpeculativeDrafter ───────────────────────────────────────────────────────────

/// `Speculative RAG` drafter (Wang et al., 2024).
///
/// The retrieved corpus is partitioned into diverse document clusters, then
/// drafts are formed per [`SpecDraftConfig::subset_strategy`]: either one
/// draft per whole cluster (the default,
/// [`DraftSubsetStrategy::PerCluster`]), or the paper's own sampling scheme —
/// `M` subsets, each with one representative document from *every* cluster
/// ([`DraftSubsetStrategy::OneRepresentativePerCluster`]). Drafts are then
/// **verified** (a support score) and combined with **self-consistency**
/// (mean agreement with the other drafts); when a draft also carries a
/// rationale (see [`Drafter::draft_with_rationale`]), that blend is further
/// multiplied by a **self-reflection** confidence `ρ_SR`, matching the
/// paper's `ρ_SC * ρ_SR` verifier score. The highest-scoring draft wins.
///
/// This is distinct from self-consistency decoding, which samples reasoning
/// paths from a single query — here the diversity comes from *disjoint document
/// clusters* rather than from stochastic decoding. It is also distinct from
/// [`layer2_speculator`](crate::layer2_speculator), which performs
/// **token-level** draft-and-verify speculative decoding (a small model
/// proposes tokens, a large model verifies them) — this module performs
/// **document-level** draft-and-verify retrieval-augmented generation.
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

    /// Form the per-draft document groups from a cluster partition (typically
    /// [`Self::cluster_docs`]'s output), according to
    /// [`SpecDraftConfig::subset_strategy`].
    ///
    /// Under [`DraftSubsetStrategy::PerCluster`] the groups are exactly
    /// `clusters`, unchanged — this is the path every caller took before the
    /// strategy existed, and remains the default. Under
    /// [`DraftSubsetStrategy::OneRepresentativePerCluster`] the groups are
    /// `M = config.num_subsets` subsets, each holding one representative
    /// index sampled from every cluster (deterministically, by cycling each
    /// cluster's members). Exposed alongside [`Self::cluster_docs`] so
    /// callers (and tests) can inspect exactly which documents each draft
    /// will see, under either strategy, without running the full
    /// [`Self::run`] pipeline.
    #[must_use]
    pub fn draft_groups(&self, clusters: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        match self.config.subset_strategy {
            DraftSubsetStrategy::PerCluster => clusters,
            DraftSubsetStrategy::OneRepresentativePerCluster => {
                form_representative_subsets(&clusters, self.config.num_subsets)
            }
        }
    }

    /// Run speculative drafting end to end.
    ///
    /// Steps: cluster the corpus → form draft groups per
    /// [`SpecDraftConfig::subset_strategy`] (one group per cluster by
    /// default, or `M` cross-cluster representative subsets under
    /// [`DraftSubsetStrategy::OneRepresentativePerCluster`]) → generate one
    /// draft per group (in parallel), optionally with a rationale (see
    /// [`Drafter::draft_with_rationale`]) → verify each draft for support →
    /// compute each draft's self-consistency as the mean token-Jaccard
    /// agreement with the *other* drafts → blend `blend = verify_weight *
    /// support + consistency_weight * self_consistency` → when the draft
    /// carries a rationale, score self-reflection `ρ_SR` (see
    /// [`DraftVerifier::reflect`]) and set `total = blend * ρ_SR`; otherwise
    /// `total = blend`, identical to the formula used before rationales
    /// existed → select the draft with the highest total. Confidence is the
    /// winning draft's total score.
    ///
    /// A lone draft has no peers to agree with, so its self-consistency is
    /// defined as `0.0` and its total reduces to the support contribution
    /// (times `ρ_SR` if a rationale is present).
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
        let groups = self.draft_groups(clusters);

        // One draft (+ optional rationale) and support score (+ optional
        // self-reflection score) per group, produced in parallel. The `Sync`
        // bounds on the traits make sharing `drafter`/`verifier` across
        // threads sound; results are collected back in group order.
        let draft_results: Vec<DraftResult> = std::thread::scope(|scope| {
            let handles: Vec<_> = groups
                .iter()
                .map(|members| {
                    scope.spawn(move || {
                        let subset: Vec<&Document> = members.iter().map(|&i| &docs[i]).collect();
                        let (content, rationale) = drafter.draft_with_rationale(query, &subset);
                        let support = clamp01(verifier.verify(query, &content, &subset));
                        let self_reflection = rationale.as_deref().map(|rationale_text| {
                            clamp01(verifier.reflect(query, &content, rationale_text, &subset))
                        });
                        DraftResult {
                            content,
                            rationale,
                            support,
                            self_reflection,
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| DraftResult::default()))
                .collect()
        });

        // Self-consistency: mean token-Jaccard agreement with the other drafts.
        let contents: Vec<&str> = draft_results.iter().map(|d| d.content.as_str()).collect();
        let mut candidates: Vec<DraftCandidate> = draft_results
            .iter()
            .enumerate()
            .map(|(idx, d)| {
                let self_consistency = mean_agreement(idx, &contents);
                let blend = self.config.verify_weight * d.support
                    + self.config.consistency_weight * self_consistency;
                // Additive: with no rationale this is `blend`, bit-for-bit the
                // same expression today's callers have always gotten. Only a
                // present rationale brings in the `ρ_SR` factor.
                let total_score = match d.self_reflection {
                    Some(rho_sr) => blend * rho_sr,
                    None => blend,
                };
                DraftCandidate {
                    content: d.content.clone(),
                    cluster_id: idx,
                    support_score: d.support,
                    self_consistency,
                    rationale: d.rationale.clone(),
                    self_reflection: d.self_reflection,
                    total_score,
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

/// One group's raw draft output, before self-consistency (which needs to see
/// every group's content) and total scoring are computed.
#[derive(Debug, Clone, Default)]
struct DraftResult {
    /// The drafted answer text.
    content: String,
    /// The rationale that justified `content`, if the [`Drafter`] supplied one.
    rationale: Option<String>,
    /// Verifier support score for `content`, in `[0, 1]`.
    support: f32,
    /// Self-reflection confidence `ρ_SR` for `content`, computed only when
    /// `rationale` is `Some`.
    self_reflection: Option<f32>,
}

/// Concatenate a document's title (if any) and content for embedding.
fn combined_text(doc: &Document) -> String {
    match &doc.title {
        Some(title) => format!("{title} {}", doc.content),
        None => doc.content.clone(),
    }
}

/// Form `num_subsets` subsets for
/// [`DraftSubsetStrategy::OneRepresentativePerCluster`].
///
/// Each subset `m` (`0..num_subsets`) contains exactly one representative
/// index from *every* non-empty cluster — `clusters[c][m % clusters[c].len()]`
/// — so a subset spans all topics rather than being confined to one,
/// matching the sampling scheme of `Speculative RAG` (Wang et al., 2024).
/// Representative selection is a pure function of `m` and the cluster
/// contents (no randomness), so identical inputs always yield identical
/// subsets, and cycling the modulus means a cluster with more than one member
/// contributes a different representative to different subsets. Returns an
/// empty vector when `clusters` is empty; `num_subsets` is floored to `1`.
fn form_representative_subsets(clusters: &[Vec<usize>], num_subsets: usize) -> Vec<Vec<usize>> {
    if clusters.is_empty() {
        return Vec::new();
    }
    let num_subsets = num_subsets.max(1);
    (0..num_subsets)
        .map(|m| {
            clusters
                .iter()
                .filter(|cluster| !cluster.is_empty())
                .map(|cluster| cluster[m % cluster.len()])
                .collect::<Vec<usize>>()
        })
        .collect()
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
