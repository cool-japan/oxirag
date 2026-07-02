//! Cross-task exemplar pool ([`UpriseIndex`]) and outcome-weighted retrieval
//! ([`UpriseRetriever`]).

use super::types::{PromptExemplar, UpriseConfig, UpriseError, UpriseHit, is_valid_unit_interval};

// ── FNV-1a pseudo-embedding (house pattern; mirrors semantic_router /
//    prompt_optimization / retrieval_diversity, kept private here since it is
//    an implementation detail of similarity scoring, not part of this
//    module's public API) ────────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries, keep tokens of length
/// `>= 2`, hash each lower-cased token with FNV-1a into a `dim`-sized bucket
/// histogram, then L2-normalise. Returns an all-zero vector of length `dim`
/// when `text` has no tokens, and an empty vector when `dim == 0`.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0_f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lowercased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two equal-length vectors, clamped to
/// `[-1.0, 1.0]`. Returns `0.0` for mismatched or empty lengths and for
/// zero-magnitude inputs.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ── UpriseIndex ───────────────────────────────────────────────────────────────

/// A pool of [`PromptExemplar`]s spanning (potentially) many different tasks.
///
/// This is deliberately *not* partitioned by task: the whole point of UPRISE
/// is that a query for one task can be served by an exemplar authored for a
/// completely different task, discovered purely through embedding similarity
/// of the query against `prompt_text` (see [`UpriseRetriever`]).
#[derive(Debug, Clone, Default)]
pub struct UpriseIndex {
    /// All exemplars currently in the pool, across all task labels.
    pub exemplars: Vec<PromptExemplar>,
}

impl UpriseIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single exemplar to the pool.
    pub fn add(&mut self, exemplar: PromptExemplar) {
        self.exemplars.push(exemplar);
    }

    /// Add several exemplars to the pool at once.
    pub fn add_many(&mut self, exemplars: impl IntoIterator<Item = PromptExemplar>) {
        self.exemplars.extend(exemplars);
    }

    /// Number of exemplars in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Return `true` when the pool holds no exemplars.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// The distinct task labels present in the pool, sorted and deduplicated.
    #[must_use]
    pub fn task_labels(&self) -> Vec<&str> {
        let mut labels: Vec<&str> = self
            .exemplars
            .iter()
            .map(|e| e.task_label.as_str())
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    /// Update the outcome quality of the exemplar at `index` via an
    /// **exponential moving average (EMA)**:
    ///
    /// ```text
    /// new_quality = alpha * new_signal + (1 - alpha) * old_quality
    /// ```
    ///
    /// An EMA (rather than a simple overwrite) is used deliberately: a single
    /// noisy usage (one unlucky generation scored by a flaky judge, one
    /// atypical query) should nudge the running estimate rather than erase a
    /// long history of otherwise-consistent performance. Repeated feedback of
    /// the same signal converges geometrically towards that signal.
    ///
    /// # Errors
    ///
    /// - [`UpriseError::InvalidOutcomeQuality`] if `new_signal` is not finite
    ///   or lies outside `[0.0, 1.0]`.
    /// - [`UpriseError::InvalidEmaAlpha`] if `alpha` is not finite or lies
    ///   outside `(0.0, 1.0]`.
    /// - [`UpriseError::IndexOutOfBounds`] if `index >= self.len()`.
    pub fn update_outcome(
        &mut self,
        index: usize,
        new_signal: f64,
        alpha: f64,
    ) -> Result<(), UpriseError> {
        if !is_valid_unit_interval(new_signal) {
            return Err(UpriseError::InvalidOutcomeQuality(new_signal));
        }
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err(UpriseError::InvalidEmaAlpha(alpha));
        }
        let exemplar = self
            .exemplars
            .get_mut(index)
            .ok_or(UpriseError::IndexOutOfBounds(index))?;
        exemplar.outcome_quality =
            alpha.mul_add(new_signal, (1.0 - alpha) * exemplar.outcome_quality);
        Ok(())
    }

    /// Convenience wrapper around [`update_outcome`](Self::update_outcome)
    /// that locates the exemplar by an exact `(task_label, prompt_text)`
    /// match (the first one found in pool order) instead of a numeric index.
    ///
    /// # Errors
    ///
    /// Returns [`UpriseError::IndexOutOfBounds`] (carrying the pool length,
    /// since no real index applies) when no exemplar matches; otherwise
    /// propagates the errors documented on
    /// [`update_outcome`](Self::update_outcome).
    pub fn update_outcome_for(
        &mut self,
        task_label: &str,
        prompt_text: &str,
        new_signal: f64,
        alpha: f64,
    ) -> Result<(), UpriseError> {
        let idx = self
            .exemplars
            .iter()
            .position(|e| e.task_label == task_label && e.prompt_text == prompt_text)
            .ok_or(UpriseError::IndexOutOfBounds(self.exemplars.len()))?;
        self.update_outcome(idx, new_signal, alpha)
    }
}

// ── UpriseRetriever ───────────────────────────────────────────────────────────

/// Universal, cross-task prompt-exemplar retriever.
///
/// Given a new query — which may describe a task unrelated to any exemplar's
/// [`task_label`](PromptExemplar::task_label) in the pool — [`retrieve`
/// ](Self::retrieve) computes each eligible exemplar's embedding similarity
/// to the query, combines it with the exemplar's historical
/// [`outcome_quality`](PromptExemplar::outcome_quality) using the weighted
/// sum in [`UpriseConfig::combined_score`], and returns the top-k exemplars
/// ranked by that combined score. This outcome-weighted reranking is the
/// centerpiece of UPRISE: a moderately-similar exemplar with a strong track
/// record can outrank a more-similar exemplar that has historically
/// underperformed.
#[derive(Debug, Clone, Default)]
pub struct UpriseRetriever {
    /// Retrieval configuration (top-k, weighting, embedding dimension).
    pub config: UpriseConfig,
}

impl UpriseRetriever {
    /// Create a retriever with the given configuration.
    #[must_use]
    pub fn new(config: UpriseConfig) -> Self {
        Self { config }
    }

    /// Retrieve the top-[`UpriseConfig::top_k`] exemplars for `query`, ranked
    /// by the outcome-weighted [`combined_score`
    /// ](UpriseConfig::combined_score). Every exemplar in `index` is
    /// eligible, regardless of `task_label` — this is the "universal"
    /// cross-task retrieval path.
    ///
    /// # Errors
    ///
    /// - [`UpriseError::EmptyQuery`] if `query` is empty or only whitespace.
    /// - [`UpriseError::EmptyIndex`] if `index` holds no exemplars.
    pub fn retrieve(
        &self,
        query: &str,
        index: &UpriseIndex,
    ) -> Result<Vec<UpriseHit>, UpriseError> {
        self.retrieve_filtered(query, index, |_| true)
    }

    /// As [`retrieve`](Self::retrieve), but every exemplar whose
    /// [`task_label`](PromptExemplar::task_label) equals `exclude_task_label`
    /// is excluded before scoring.
    ///
    /// This lets a caller solving a task it has *not* seen before measure
    /// pure cross-task transfer: excluding same-task exemplars shows how well
    /// exemplars authored for entirely different tasks alone can serve the
    /// query.
    ///
    /// # Errors
    ///
    /// - [`UpriseError::EmptyQuery`] if `query` is empty or only whitespace.
    /// - [`UpriseError::EmptyIndex`] if `index` holds no exemplars at all.
    /// - [`UpriseError::NoEligibleExemplars`] if `index` holds exemplars but
    ///   every one of them carries `exclude_task_label`.
    pub fn retrieve_excluding_task(
        &self,
        query: &str,
        index: &UpriseIndex,
        exclude_task_label: &str,
    ) -> Result<Vec<UpriseHit>, UpriseError> {
        self.retrieve_filtered(query, index, |e| e.task_label != exclude_task_label)
    }

    /// As [`retrieve`](Self::retrieve), but only exemplars for which
    /// `predicate` returns `true` are eligible for scoring/ranking.
    ///
    /// [`retrieve`](Self::retrieve) and [`retrieve_excluding_task`
    /// ](Self::retrieve_excluding_task) are both thin wrappers around this
    /// method.
    ///
    /// # Errors
    ///
    /// - [`UpriseError::EmptyQuery`] if `query` is empty or only whitespace.
    /// - [`UpriseError::EmptyIndex`] if `index` holds no exemplars at all.
    /// - [`UpriseError::NoEligibleExemplars`] if `index` holds exemplars but
    ///   none satisfy `predicate`.
    pub fn retrieve_filtered(
        &self,
        query: &str,
        index: &UpriseIndex,
        predicate: impl Fn(&PromptExemplar) -> bool,
    ) -> Result<Vec<UpriseHit>, UpriseError> {
        let mut hits = self.score_eligible(query, index, &predicate)?;
        hits.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(self.config.top_k);
        Ok(hits)
    }

    /// Rank exemplars by **pure embedding similarity** to `query`, ignoring
    /// historical outcome quality entirely.
    ///
    /// Every returned [`UpriseHit`] still carries its `combined_score` field
    /// (computed the same way as in [`retrieve`](Self::retrieve)) so callers
    /// — and this module's tests — can directly contrast the pure-similarity
    /// order against the outcome-weighted order and observe the reranking
    /// effect.
    ///
    /// # Errors
    ///
    /// Same as [`retrieve`](Self::retrieve).
    pub fn rank_by_similarity(
        &self,
        query: &str,
        index: &UpriseIndex,
    ) -> Result<Vec<UpriseHit>, UpriseError> {
        let mut hits = self.score_eligible(query, index, &|_| true)?;
        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(self.config.top_k);
        Ok(hits)
    }

    /// Score every exemplar in `index` matching `predicate` against `query`,
    /// without sorting or truncating to `top_k`. Shared by every public
    /// retrieval entry point above.
    fn score_eligible(
        &self,
        query: &str,
        index: &UpriseIndex,
        predicate: &dyn Fn(&PromptExemplar) -> bool,
    ) -> Result<Vec<UpriseHit>, UpriseError> {
        if query.trim().is_empty() {
            return Err(UpriseError::EmptyQuery);
        }
        if index.is_empty() {
            return Err(UpriseError::EmptyIndex);
        }

        let query_embedding = embed(query, self.config.dim);
        let hits: Vec<UpriseHit> = index
            .exemplars
            .iter()
            .filter(|e| predicate(e))
            .map(|e| {
                let exemplar_embedding = if e.embedding.len() == self.config.dim {
                    e.embedding.clone()
                } else {
                    embed(&e.prompt_text, self.config.dim)
                };
                let similarity = f64::from(cosine(&query_embedding, &exemplar_embedding));
                let combined_score = self.config.combined_score(similarity, e.outcome_quality);
                UpriseHit {
                    task_label: e.task_label.clone(),
                    prompt_text: e.prompt_text.clone(),
                    outcome_quality: e.outcome_quality,
                    similarity,
                    combined_score,
                }
            })
            .collect();

        if hits.is_empty() {
            return Err(UpriseError::NoEligibleExemplars);
        }
        Ok(hits)
    }
}
