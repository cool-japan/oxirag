//! [`MoaSynthesisAggregator`] — the crate's real, default
//! [`super::types::MoaAggregator`] implementation — together with the small
//! lexical toolkit it (and [`super::engine`]'s [`MoaLayerStats`](super::types::MoaLayerStats)
//! computation) are built on.
//!
//! ## The synthesis algorithm
//!
//! Given `N` proposals, [`MoaSynthesisAggregator::aggregate`] does **not**
//! pick a winner. It:
//!
//! 1. **Splits every proposal into sentences** (`split_sentences`) — the
//!    unit of "one claim" this module works at.
//! 2. **Deduplicates by lexical overlap**: each sentence is reduced to its
//!    content-term set (`content_terms` — lowercase alphanumeric tokens of
//!    at least three characters, stopwords excluded), and sentences whose
//!    Jaccard overlap (`jaccard`) with an existing cluster clears
//!    [`MoaSynthesisAggregator::similarity_threshold`] are folded into that
//!    cluster rather than kept as separate near-duplicates (see
//!    `cluster_sentences`). A sentence with **no** content terms at all
//!    (pure filler, e.g. "Indeed." or "Thus,") is dropped outright — it
//!    carries no claim to preserve.
//! 3. **Tracks agreement per cluster**: every cluster remembers the
//!    *distinct* set of proposers (by [`MoaResponse::proposer_id`]) whose
//!    text contributed a sentence to it — a proposer repeating itself within
//!    one proposal only counts once.
//! 4. **Orders clusters by agreement, then by first appearance**: clusters
//!    supported by more proposers are ranked ahead of clusters supported by
//!    fewer, so content multiple proposals agree on reads first; ties are
//!    broken by the order the underlying claim first appeared across the
//!    proposals, keeping the result deterministic and roughly narrative.
//! 5. **Never drops a substantive minority claim**: every cluster survives
//!    into the merged output regardless of how few proposers support it —
//!    only content-free filler is ever excluded (step 2). A claim raised by
//!    exactly one proposer therefore still appears, just later in the
//!    ranking than claims multiple proposers converged on. This is what
//!    makes the result a genuine *synthesis* rather than a *selection*.
//!
//! The output is the ranked clusters' representative sentences, joined with
//! a single space. No sentence's *wording* is invented — every sentence in
//! the output is verbatim (module the cluster's chosen representative — the
//! longest, most content-rich phrasing among near-duplicates) text some
//! proposal actually wrote.
//!
//! All scoring is pure floating-point arithmetic over the supplied text —
//! no randomness (`rand`/`rand_distr`) and no array/tensor machinery
//! (`ndarray`/`SciRS2-Core`) is needed anywhere in this module, mirroring
//! `crate::multi_agent_debate::engine`'s "deliberately self-contained
//! lexical toolkit" convention. This module owns its own copy of the
//! tokenizer/Jaccard helpers rather than importing `multi_agent_debate`'s.

use std::collections::BTreeSet;

use super::types::{MoaAggregator, MoaError, MoaResponse};

// ── lexical helpers ──────────────────────────────────────────────────────────

/// Stopwords excluded from the content-term vocabulary used throughout this
/// module's synthesis algorithm.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "did", "its", "let", "put", "say", "she", "too", "use", "that",
    "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then", "than",
    "into", "some", "such", "only", "also", "been", "more", "very", "will", "would", "there",
    "their", "which", "about", "could", "these", "those", "does", "still", "even",
];

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep
/// non-empty fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` when `token` is a content token: at least three characters
/// and not a stopword.
fn is_content_token(token: &str) -> bool {
    token.chars().count() >= 3 && !STOPWORDS.contains(&token)
}

/// The distinct, sorted content-term vocabulary of `text`. A [`BTreeSet`]
/// (rather than a hash-based set) so iterating and comparing the result is
/// itself deterministic across process runs.
pub(crate) fn content_terms(text: &str) -> BTreeSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| is_content_token(t))
        .collect()
}

/// Jaccard similarity (`|intersection| / |union|`) between two content-term
/// sets. `0.0` when both sets are empty.
pub(crate) fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f32 / union as f32
    }
}

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`
/// boundaries. Any trailing fragment without closing punctuation is kept as
/// a final sentence, so unpunctuated input still yields one sentence rather
/// than none.
pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }
    sentences
}

// ── sentence clustering ──────────────────────────────────────────────────────

/// The default lexical-overlap threshold, in `[0.0, 1.0]`, above which two
/// sentences are treated as the same underlying claim by [`cluster_sentences`].
///
/// Used both by [`MoaSynthesisAggregator::default`] and by
/// [`super::engine`]'s [`MoaLayerStats`](super::types::MoaLayerStats)
/// coverage computation, so "how many distinct claims survived into the
/// aggregate" is measured against the same notion of "distinct claim" the
/// default aggregator itself uses.
pub(crate) const DEFAULT_CLAIM_SIMILARITY_THRESHOLD: f32 = 0.6;

/// One cluster of near-duplicate sentences — i.e. one distinct claim —
/// gathered across a layer's proposals by [`cluster_sentences`].
#[derive(Debug, Clone)]
pub(crate) struct SentenceCluster {
    /// The cluster's representative sentence text: the longest (most
    /// content-rich) phrasing seen among the sentences folded into it.
    pub(crate) text: String,
    /// The representative sentence's content-term set.
    pub(crate) terms: BTreeSet<String>,
    /// The distinct proposers ([`MoaResponse::proposer_id`]) whose text
    /// contributed at least one sentence to this cluster.
    pub(crate) proposer_ids: BTreeSet<usize>,
    /// The position, in overall (proposal-then-sentence) scan order, at
    /// which this cluster was first created. Used as the ranking
    /// tie-breaker among equally-agreed-upon clusters.
    pub(crate) first_seen_order: usize,
}

/// Group every content-bearing sentence across `proposals` into
/// [`SentenceCluster`]s, folding sentences whose content-term Jaccard
/// overlap clears `similarity_threshold` into the same cluster. Sentences
/// with no content terms at all are dropped (see the module-level
/// documentation, step 2).
///
/// Proposals are scanned in slice order, and each proposal's sentences are
/// scanned in the order [`split_sentences`] returns them, so
/// [`SentenceCluster::first_seen_order`] reflects the proposals' own
/// narrative order.
pub(crate) fn cluster_sentences(
    proposals: &[MoaResponse],
    similarity_threshold: f32,
) -> Vec<SentenceCluster> {
    let mut clusters: Vec<SentenceCluster> = Vec::new();
    let mut order = 0_usize;
    for proposal in proposals {
        for sentence in split_sentences(&proposal.text) {
            let terms = content_terms(&sentence);
            if terms.is_empty() {
                continue;
            }
            if let Some(cluster) = clusters
                .iter_mut()
                .find(|cluster| jaccard(&cluster.terms, &terms) >= similarity_threshold)
            {
                cluster.proposer_ids.insert(proposal.proposer_id);
                // Prefer the more content-rich phrasing as the cluster's
                // representative, so a terse near-duplicate never displaces
                // a fuller one.
                if terms.len() > cluster.terms.len() {
                    cluster.text = sentence;
                    cluster.terms = terms;
                }
            } else {
                let mut proposer_ids = BTreeSet::new();
                proposer_ids.insert(proposal.proposer_id);
                clusters.push(SentenceCluster {
                    text: sentence,
                    terms,
                    proposer_ids,
                    first_seen_order: order,
                });
            }
            order += 1;
        }
    }
    clusters
}

// ── MoaSynthesisAggregator ───────────────────────────────────────────────────

/// The crate's real, default [`MoaAggregator`] implementation: deterministic
/// sentence-level extraction, lexical-overlap dedup, agreement-weighted
/// ordering, and coverage-preserving merge. See the module-level
/// documentation for the full algorithm.
///
/// No live model is involved — every output is fully determined by the
/// input text — so this is also suitable as one of the "deterministic
/// in-module test implementations" [`super::engine::MoaEngine`] can be
/// exercised against with no real LLM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoaSynthesisAggregator {
    /// The lexical-overlap threshold (`[0.0, 1.0]`) above which two
    /// sentences are folded into the same claim cluster. Higher values keep
    /// more near-duplicates as separate claims; lower values merge more
    /// aggressively. Defaults to `0.6`.
    pub similarity_threshold: f32,
}

impl Default for MoaSynthesisAggregator {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_CLAIM_SIMILARITY_THRESHOLD,
        }
    }
}

impl MoaSynthesisAggregator {
    /// Create a new aggregator with a custom claim-clustering similarity
    /// threshold.
    #[must_use]
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
        }
    }
}

impl MoaAggregator for MoaSynthesisAggregator {
    fn aggregate(&self, query: &str, proposals: &[MoaResponse]) -> Result<String, MoaError> {
        if query.trim().is_empty() {
            return Err(MoaError::EmptyQuery);
        }
        if proposals.is_empty() {
            return Err(MoaError::EmptyProposals);
        }

        let mut clusters = cluster_sentences(proposals, self.similarity_threshold);

        // Agreement-weighted ordering: content asserted by more proposers
        // ranks first; ties broken by first-seen order, which keeps the
        // result deterministic and roughly follows the proposals' own
        // narrative order. Every cluster survives this sort — none are
        // truncated or discarded here (coverage-preserving merge).
        clusters.sort_by(|a, b| {
            b.proposer_ids
                .len()
                .cmp(&a.proposer_ids.len())
                .then_with(|| a.first_seen_order.cmp(&b.first_seen_order))
        });

        let merged = clusters
            .into_iter()
            .map(|cluster| cluster.text)
            .collect::<Vec<_>>()
            .join(" ");
        Ok(merged)
    }
}
