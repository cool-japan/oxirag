//! Profile-driven reranking of retrieval results.
//!
//! The [`PersonalizedReranker`] re-scores a list of [`SearchResult`]s by blending
//! each document's original relevance with a *personal affinity* derived from a
//! [`UserProfile`]. Affinity has two deterministic ingredients:
//!
//! 1. **Topic-interest match** — how strongly the document's tokens overlap the
//!    user's explicitly weighted topic keywords.
//! 2. **History similarity** — the cosine similarity between the document's
//!    lexical embedding and the centroid of the user's interaction history.
//!
//! Both ingredients are pure functions of their inputs (FNV-1a hashing, no
//! randomness), so the reranker is fully reproducible.

use crate::personalized_rag::types::{PersonalizedConfig, PersonalizedError, UserProfile};
use crate::types::{Document, SearchResult};

// ── Lexical helpers ────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens of length >= 2.
///
/// Splitting occurs on every non-alphanumeric character; tokens shorter than two
/// characters are dropped to suppress noise from stray punctuation and initials.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise, hash each token into a bucket with FNV-1a, accumulate
/// per-bucket counts, then L2-normalise. A `dim` of `0` yields an empty vector.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in tokenize(text) {
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    l2_normalize(&mut buckets);
    buckets
}

/// L2-normalise `vector` in place; vectors with negligible norm are left as-is.
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two equal-length, L2-normalised vectors.
///
/// Returns `0.0` for empty or mismatched inputs; the result is clamped to
/// `[0, 1]` because affinity is defined to be non-negative.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(0.0, 1.0)
}

// ── PersonalizedReranker ───────────────────────────────────────────────────────

/// Reranks retrieval results against a [`UserProfile`].
///
/// Construct one with [`PersonalizedReranker::new`] and reuse it across queries;
/// it is cheap to clone and holds only its [`PersonalizedConfig`].
#[derive(Debug, Clone)]
pub struct PersonalizedReranker {
    /// Reranking configuration (blend factor and embedding dimensionality).
    config: PersonalizedConfig,
}

impl PersonalizedReranker {
    /// Create a reranker with the given configuration.
    #[must_use]
    pub fn new(config: PersonalizedConfig) -> Self {
        Self { config }
    }

    /// Borrow the active configuration.
    #[must_use]
    pub fn config(&self) -> &PersonalizedConfig {
        &self.config
    }

    /// Compute the centroid of the user's interaction history.
    ///
    /// Each history snippet is embedded, the embeddings are summed, and the sum
    /// is L2-normalised. An empty history yields the zero vector.
    fn history_centroid(&self, profile: &UserProfile) -> Vec<f32> {
        let mut centroid = vec![0.0f32; self.config.dim];
        if self.config.dim == 0 {
            return centroid;
        }
        for snippet in profile.history() {
            let emb = embed(snippet, self.config.dim);
            for (slot, value) in centroid.iter_mut().zip(emb.iter()) {
                *slot += value;
            }
        }
        l2_normalize(&mut centroid);
        centroid
    }

    /// Score how strongly `doc` matches the user's explicit topic interests.
    ///
    /// For every interest topic present among the document tokens, its weight is
    /// accumulated; the total is divided by the sum of all interest weights so
    /// the result lands in `[0, 1]`. A document matching every interested topic
    /// scores `1.0`; one matching none scores `0.0`.
    fn topic_match(profile: &UserProfile, doc: &Document) -> f32 {
        let topics = profile.topics();
        if topics.is_empty() {
            return 0.0;
        }
        let total_weight: f32 = topics.values().sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let haystack = format!("{} {}", doc.title.as_deref().unwrap_or(""), doc.content);
        let tokens: std::collections::HashSet<String> = tokenize(&haystack).into_iter().collect();
        let mut matched = 0.0f32;
        for (topic, weight) in topics {
            if tokens.contains(topic) {
                matched += *weight;
            }
        }
        (matched / total_weight).clamp(0.0, 1.0)
    }

    /// Compute the personal affinity of `doc` for the given `profile`.
    ///
    /// Affinity blends two signals, each in `[0, 1]`:
    ///
    /// * the topic-interest match against the document tokens, and
    /// * the cosine similarity between the document embedding and the
    ///   interaction-history centroid.
    ///
    /// When the profile only carries one of the two signals (interests but no
    /// history, or vice versa), the available signal is returned directly;
    /// when it carries both, their mean is returned. An empty profile — or one
    /// whose signals are all zero — yields `0.0`. The result is always in
    /// `[0, 1]`.
    #[must_use]
    pub fn profile_affinity(&self, profile: &UserProfile, doc: &Document) -> f32 {
        if profile.is_empty() {
            return 0.0;
        }
        let has_topics = !profile.topics().is_empty();
        let has_history = !profile.history().is_empty() && self.config.dim > 0;

        let topic_score = if has_topics {
            Self::topic_match(profile, doc)
        } else {
            0.0
        };
        let history_score = if has_history {
            let centroid = self.history_centroid(profile);
            let doc_emb = embed(&doc.content, self.config.dim);
            cosine(&doc_emb, &centroid)
        } else {
            0.0
        };

        let affinity = match (has_topics, has_history) {
            (true, true) => f32::midpoint(topic_score, history_score),
            (true, false) => topic_score,
            (false, true) => history_score,
            (false, false) => 0.0,
        };
        affinity.clamp(0.0, 1.0)
    }

    /// Blend `relevance` and `affinity` using the configured weight.
    ///
    /// Returns `(1 - personalization_weight) * relevance + personalization_weight * affinity`,
    /// clamped to `[0, 1]`.
    fn blend(&self, relevance: f32, affinity: f32) -> f32 {
        let w = self.config.personalization_weight.clamp(0.0, 1.0);
        ((1.0 - w) * relevance + w * affinity).clamp(0.0, 1.0)
    }

    /// Rerank `results` against `profile`, returning a freshly ordered list.
    ///
    /// Each result's score becomes the blend of its original relevance and its
    /// personal affinity. Results are then sorted by the blended score in
    /// descending order, ties are broken deterministically by ascending document
    /// id, and ranks are renumbered from `0`. An empty input yields an empty
    /// output. The original `results` slice is left untouched.
    #[must_use]
    pub fn rerank(&self, profile: &UserProfile, results: &[SearchResult]) -> Vec<SearchResult> {
        let mut reranked: Vec<SearchResult> = results
            .iter()
            .map(|result| {
                let affinity = self.profile_affinity(profile, &result.document);
                let score = self.blend(result.score, affinity);
                SearchResult::new(result.document.clone(), score, result.rank)
            })
            .collect();

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.document.id.as_str().cmp(b.document.id.as_str()))
        });

        for (rank, result) in reranked.iter_mut().enumerate() {
            result.rank = rank;
        }
        reranked
    }

    /// Rerank `results`, erroring when the input is empty.
    ///
    /// Behaves exactly like [`PersonalizedReranker::rerank`] for non-empty
    /// input.
    ///
    /// # Errors
    ///
    /// Returns [`PersonalizedError::EmptyResults`] when `results` is empty.
    pub fn rerank_checked(
        &self,
        profile: &UserProfile,
        results: &[SearchResult],
    ) -> Result<Vec<SearchResult>, PersonalizedError> {
        if results.is_empty() {
            return Err(PersonalizedError::EmptyResults);
        }
        Ok(self.rerank(profile, results))
    }
}
