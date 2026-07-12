//! [`SelfKnowledgePool`] (accumulated self-knowledge exemplars, with
//! oldest-first eviction) and [`SkrGate`] (the memory-based, k-nearest-
//! neighbor retrieve-or-not gate).

use super::types::{
    SkrConfig, SkrDecision, SkrError, SkrExemplar, SkrNeighbor, SkrResult, SkrRetrievalChoice,
};

// ── FNV-1a pseudo-embeddings (house pattern; mirrors semantic_router /
//    uprise_retrieval / prompt_optimization, kept private here since it is an
//    implementation detail of similarity scoring, not part of this module's
//    public API) ────────────────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries, keep tokens of length
/// `>= 2`, hash each lower-cased token with FNV-1a into a `dim`-sized bucket
/// histogram, then L2-normalise. Returns an all-zero vector of length `dim`
/// when `text` has no qualifying tokens, and an empty vector when `dim == 0`.
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

/// Cosine similarity between two vectors, clamped to `[-1.0, 1.0]`.
///
/// Returns `0.0` for mismatched-length or empty inputs and for
/// zero-magnitude vectors, rather than dividing by zero.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

// ── vote aggregation ─────────────────────────────────────────────────────────

/// Combine `neighbors`' self-knowledge labels into a single "known" score in
/// `[0.0, 1.0]`: the fraction of vote weight cast by exemplars labelled
/// `answerable_without_retrieval = true`.
///
/// When `weight_by_similarity` is `true`, each neighbour's vote is weighted by
/// its similarity (clamped to be non-negative first, so a neighbour that is
/// dissimilar in the "opposite direction" cannot cast a negative vote), so
/// closer exemplars count for more than distant ones. If every neighbour's
/// weight is (near) zero, dividing by that near-zero total would be
/// meaningless, so this degenerate case instead falls back to an unweighted
/// majority vote. When `weight_by_similarity` is `false`, every neighbour
/// always casts an equal vote (a plain majority), regardless of similarity.
///
/// Returns `0.0` for an empty neighbour list.
fn weighted_known_score(neighbors: &[SkrNeighbor], weight_by_similarity: bool) -> f32 {
    if neighbors.is_empty() {
        return 0.0;
    }

    if weight_by_similarity {
        let mut weight_sum = 0.0_f32;
        let mut positive_weight = 0.0_f32;
        for neighbor in neighbors {
            let weight = neighbor.similarity.max(0.0);
            weight_sum += weight;
            if neighbor.answerable_without_retrieval {
                positive_weight += weight;
            }
        }
        if weight_sum > 1e-6 {
            return (positive_weight / weight_sum).clamp(0.0, 1.0);
        }
    }

    majority_vote(neighbors)
}

/// Unweighted majority vote: the fraction of `neighbors` labelled
/// `answerable_without_retrieval = true`. Returns `0.0` for an empty slice.
fn majority_vote(neighbors: &[SkrNeighbor]) -> f32 {
    if neighbors.is_empty() {
        return 0.0;
    }
    let positive = neighbors
        .iter()
        .filter(|n| n.answerable_without_retrieval)
        .count();
    #[allow(clippy::cast_precision_loss)]
    let score = positive as f32 / neighbors.len() as f32;
    score
}

// ── SelfKnowledgePool ─────────────────────────────────────────────────────────

/// An accumulated pool of [`SkrExemplar`]s: the model's self-knowledge memory
/// of which questions it could previously answer without retrieval, and which
/// it could not.
///
/// Unlike a fixed, hand-curated calibration set, a [`SelfKnowledgePool`]
/// starts empty (or however it is seeded) and grows over time as
/// [`SkrGate::observe_outcome`] feeds back fresh labelled datapoints from real
/// usage, so a gate's retrieve-or-not decisions get sharper the more
/// questions it has seen. Once [`SkrConfig::pool_cap`] is reached, the oldest
/// exemplar is evicted before a new one is admitted (FIFO), keeping the pool
/// biased toward the model's *current* self-knowledge rather than
/// unboundedly accumulating stale signal.
#[derive(Debug, Clone, Default)]
pub struct SelfKnowledgePool {
    /// All exemplars currently in the pool, oldest first.
    pub exemplars: Vec<SkrExemplar>,
}

impl SelfKnowledgePool {
    /// Create an empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of exemplars currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Return `true` when the pool holds no exemplars.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// Number of "positive" exemplars (`answerable_without_retrieval = true`),
    /// i.e. questions the model knows it can answer directly.
    #[must_use]
    pub fn known_count(&self) -> usize {
        self.exemplars
            .iter()
            .filter(|e| e.answerable_without_retrieval)
            .count()
    }

    /// Number of "negative" exemplars (`answerable_without_retrieval =
    /// false`), i.e. questions the model knows it needs retrieval for.
    #[must_use]
    pub fn unknown_count(&self) -> usize {
        self.exemplars
            .iter()
            .filter(|e| !e.answerable_without_retrieval)
            .count()
    }

    /// Add a single exemplar to the pool.
    ///
    /// Computes and stores `exemplar`'s pseudo-embedding first whenever its
    /// current length does not match `config.embedding_dim` (so callers who
    /// pre-populate [`SkrExemplar::embedding`] via
    /// [`SkrExemplar::with_embedding`] skip the recomputation). If the pool
    /// then exceeds [`SkrConfig::pool_cap`] (and `pool_cap` is not `0`,
    /// meaning unbounded), the oldest exemplars are evicted first (FIFO)
    /// until the pool is back at capacity.
    pub fn add_exemplar(&mut self, mut exemplar: SkrExemplar, config: &SkrConfig) {
        if exemplar.embedding.len() != config.embedding_dim {
            exemplar.embedding = embed(&exemplar.question, config.embedding_dim);
        }
        self.exemplars.push(exemplar);
        self.evict_to_cap(config.pool_cap);
    }

    /// Convenience wrapper around [`SelfKnowledgePool::add_exemplar`] that
    /// builds the [`SkrExemplar`] from a raw question/label pair.
    pub fn add_labeled(
        &mut self,
        question: impl Into<String>,
        answerable_without_retrieval: bool,
        config: &SkrConfig,
    ) {
        self.add_exemplar(
            SkrExemplar::new(question, answerable_without_retrieval),
            config,
        );
    }

    /// Append a new self-knowledge datapoint from feedback.
    ///
    /// After actually attempting to answer `question`, call this with
    /// whether it turned out to be answerable without retrieval. This is the
    /// write side of the pool's online-learning loop: repeated calls let a
    /// [`SkrGate`]'s future decisions track the model's *current*
    /// self-knowledge rather than a fixed calibration snapshot. Implemented
    /// as a direct call to [`SelfKnowledgePool::add_labeled`]; kept as a
    /// distinct method so call sites can name the feedback-driven intent
    /// explicitly.
    pub fn observe_outcome(
        &mut self,
        question: impl Into<String>,
        was_answerable: bool,
        config: &SkrConfig,
    ) {
        self.add_labeled(question, was_answerable, config);
    }

    /// Evict the oldest exemplars (FIFO, from the front) until the pool size
    /// is at most `cap`. A `cap` of `0` means unbounded: no eviction occurs.
    fn evict_to_cap(&mut self, cap: usize) {
        if cap == 0 {
            return;
        }
        while self.exemplars.len() > cap {
            self.exemplars.remove(0);
        }
    }
}

// ── SkrGate ───────────────────────────────────────────────────────────────────

/// The SKR memory-based retrieve-or-not gate.
///
/// Owns both the tunable [`SkrConfig`] and the accumulated
/// [`SelfKnowledgePool`], mirroring the caller-owns-both-state pattern used
/// elsewhere in this crate (e.g. `buffer_of_thoughts::BotEngine` owning its
/// `ThoughtBuffer`). A single long-lived [`SkrGate`] is expected to answer
/// many [`SkrGate::decide`] calls and periodically grow via
/// [`SkrGate::observe_outcome`].
///
/// # Example
///
/// ```
/// use oxirag::skr::{SkrConfig, SkrGate, SkrRetrievalChoice};
///
/// let mut gate = SkrGate::new(SkrConfig::new().with_k(1)).expect("valid config");
///
/// // Teach the gate that questions about the capital of France are
/// // "known" (answerable without retrieval)...
/// gate.add_labeled("What is the capital of France?", true).unwrap();
/// // ...and that questions about a fictitious internal report are "unknown".
/// gate.add_labeled("What were Q3 sales for Acme Corp internal report?", false)
///     .unwrap();
///
/// let known = gate.decide("What is the capital of France?").unwrap();
/// assert_eq!(known.choice, SkrRetrievalChoice::Skip);
///
/// let unknown = gate
///     .decide("What were Q3 sales for Acme Corp internal report?")
///     .unwrap();
/// assert_eq!(unknown.choice, SkrRetrievalChoice::Retrieve);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SkrGate {
    /// The gate's configuration.
    pub config: SkrConfig,
    /// The accumulated self-knowledge pool consulted by [`SkrGate::decide`].
    pub pool: SelfKnowledgePool,
}

impl SkrGate {
    /// Create a new gate with an empty pool.
    ///
    /// # Errors
    ///
    /// Returns an error from [`SkrConfig::validate`] if `config` is invalid.
    pub fn new(config: SkrConfig) -> SkrResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            pool: SelfKnowledgePool::new(),
        })
    }

    /// Create a new gate seeded with an existing pool (e.g. restored from a
    /// prior session).
    ///
    /// # Errors
    ///
    /// Returns an error from [`SkrConfig::validate`] if `config` is invalid.
    pub fn with_pool(config: SkrConfig, pool: SelfKnowledgePool) -> SkrResult<Self> {
        config.validate()?;
        Ok(Self { config, pool })
    }

    /// Find the [`SkrConfig::k`] nearest pool exemplars to `question` by
    /// cosine similarity of their pseudo-embeddings, best match first.
    ///
    /// Returns an empty vector (not an error) when the pool is empty.
    ///
    /// # Errors
    ///
    /// - [`SkrError::EmptyQuestion`] if `question` is empty or only
    ///   whitespace.
    /// - Any error from [`SkrConfig::validate`] if [`SkrGate::config`] is
    ///   invalid.
    pub fn k_nearest(&self, question: &str) -> SkrResult<Vec<SkrNeighbor>> {
        if question.trim().is_empty() {
            return Err(SkrError::EmptyQuestion);
        }
        self.config.validate()?;

        if self.pool.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = embed(question, self.config.embedding_dim);
        let mut neighbors: Vec<SkrNeighbor> = self
            .pool
            .exemplars
            .iter()
            .map(|exemplar| SkrNeighbor {
                question: exemplar.question.clone(),
                answerable_without_retrieval: exemplar.answerable_without_retrieval,
                similarity: cosine(&query_embedding, &exemplar.embedding),
            })
            .collect();
        // Stable sort: exemplars that tie on similarity keep their original
        // pool (insertion) order, so results stay fully deterministic.
        neighbors.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        neighbors.truncate(self.config.k);
        Ok(neighbors)
    }

    /// Decide whether `question` needs retrieval augmentation.
    ///
    /// Embeds `question`, finds its [`SkrConfig::k`] nearest exemplars in the
    /// self-knowledge pool (see [`SkrGate::k_nearest`]), and takes a
    /// similarity-weighted (or, with [`SkrConfig::weight_by_similarity`]
    /// disabled, plain-majority) vote of their
    /// [`SkrExemplar::answerable_without_retrieval`] labels — see
    /// `weighted_known_score`. When the resulting "known" score meets
    /// [`SkrConfig::decision_threshold`], the decision is
    /// [`SkrRetrievalChoice::Skip`]; otherwise
    /// [`SkrRetrievalChoice::Retrieve`].
    ///
    /// An empty pool has no evidence to vote with, so it falls back to
    /// [`SkrConfig::default_choice`] with a known score of `0.0` and no
    /// neighbours — check [`SkrDecision::is_fallback`] to distinguish this
    /// case from a genuine vote.
    ///
    /// # Errors
    ///
    /// - [`SkrError::EmptyQuestion`] if `question` is empty or only
    ///   whitespace.
    /// - Any error from [`SkrConfig::validate`] if [`SkrGate::config`] is
    ///   invalid.
    pub fn decide(&self, question: &str) -> SkrResult<SkrDecision> {
        if question.trim().is_empty() {
            return Err(SkrError::EmptyQuestion);
        }
        self.config.validate()?;

        if self.pool.is_empty() {
            return Ok(SkrDecision {
                choice: self.config.default_choice,
                known_score: 0.0,
                neighbors: Vec::new(),
                reason: format!(
                    "self-knowledge pool is empty; falling back to default choice '{}'",
                    self.config.default_choice
                ),
            });
        }

        let neighbors = self.k_nearest(question)?;
        let known_score = weighted_known_score(&neighbors, self.config.weight_by_similarity);
        let meets_threshold = known_score >= self.config.decision_threshold;
        let choice = if meets_threshold {
            SkrRetrievalChoice::Skip
        } else {
            SkrRetrievalChoice::Retrieve
        };

        let reason = format!(
            "known-score {known_score:.3} from {} neighbor(s) ({} threshold {:.3}) => {choice}",
            neighbors.len(),
            if meets_threshold { ">=" } else { "<" },
            self.config.decision_threshold,
        );

        Ok(SkrDecision {
            choice,
            known_score,
            neighbors,
            reason,
        })
    }

    /// Add a single exemplar to the pool (see
    /// [`SelfKnowledgePool::add_exemplar`]).
    pub fn add_exemplar(&mut self, exemplar: SkrExemplar) {
        self.pool.add_exemplar(exemplar, &self.config);
    }

    /// Label a question and add it to the pool (see
    /// [`SelfKnowledgePool::add_labeled`]).
    ///
    /// # Errors
    ///
    /// Returns [`SkrError::EmptyQuestion`] if `question` is empty or only
    /// whitespace.
    pub fn add_labeled(
        &mut self,
        question: impl Into<String>,
        answerable_without_retrieval: bool,
    ) -> SkrResult<()> {
        let question = question.into();
        if question.trim().is_empty() {
            return Err(SkrError::EmptyQuestion);
        }
        self.pool
            .add_labeled(question, answerable_without_retrieval, &self.config);
        Ok(())
    }

    /// Feed back a real-world outcome for `question` into the pool (see
    /// [`SelfKnowledgePool::observe_outcome`]), so future [`SkrGate::decide`]
    /// calls reflect it.
    ///
    /// # Errors
    ///
    /// Returns [`SkrError::EmptyQuestion`] if `question` is empty or only
    /// whitespace.
    pub fn observe_outcome(
        &mut self,
        question: impl Into<String>,
        was_answerable: bool,
    ) -> SkrResult<()> {
        let question = question.into();
        if question.trim().is_empty() {
            return Err(SkrError::EmptyQuestion);
        }
        self.pool
            .observe_outcome(question, was_answerable, &self.config);
        Ok(())
    }
}
