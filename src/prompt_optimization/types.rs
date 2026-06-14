//! Types for the `prompt_optimization` module.
use thiserror::Error;
// ── DemoSelectionStrategy ─────────────────────────────────────────────────────
/// Strategy for selecting few-shot demonstrations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum DemoSelectionStrategy {
    /// Nearest-neighbour selection by FNV-1a cosine similarity.
    #[default]
    Knn,
    /// Maximum Marginal Relevance: balance relevance and diversity.
    MmrDiverse,
    /// Deterministic selection by index (no randomness — use for testing).
    Deterministic,
    /// Select demonstrations where expected output is least similar (hardest examples).
    Hardest,
}
// ── Demonstration ─────────────────────────────────────────────────────────────
/// A single few-shot demonstration (input → output example).
#[derive(Debug, Clone)]
pub struct Demonstration {
    /// The example input text.
    pub input: String,
    /// The expected output text.
    pub output: String,
    /// Pre-computed FNV-1a embedding of the input.
    pub embedding: Vec<f32>,
    /// Quality score in [0.0, 1.0]. Used by `Hardest` strategy.
    pub quality: f32,
}
impl Demonstration {
    /// Create a demonstration with an embedding dimension of 0 (will be computed on demand).
    #[must_use]
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            embedding: Vec::new(),
            quality: 1.0,
        }
    }
    /// Attach a quality score.
    #[must_use]
    pub fn with_quality(mut self, q: f32) -> Self {
        self.quality = q;
        self
    }
}
// ── DemoPool ──────────────────────────────────────────────────────────────────
/// A collection of demonstrations available for selection.
#[derive(Debug, Clone, Default)]
pub struct DemoPool {
    /// All demonstrations in the pool.
    pub demos: Vec<Demonstration>,
}
impl DemoPool {
    /// Create a pool from a list of demonstrations.
    #[must_use]
    pub fn new(demos: Vec<Demonstration>) -> Self {
        Self { demos }
    }
    /// Add a demonstration to the pool.
    pub fn add(&mut self, demo: Demonstration) {
        self.demos.push(demo);
    }
    /// Return the number of demonstrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.demos.len()
    }
    /// Return true if the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.demos.is_empty()
    }
}
// ── DemoSelector ─────────────────────────────────────────────────────────────
/// Selects few-shot demonstrations from a pool for a given query.
#[derive(Debug, Clone)]
pub struct DemoSelector {
    /// Selection strategy.
    pub strategy: DemoSelectionStrategy,
    /// MMR diversity weight λ (only used by `MmrDiverse`). Defaults to `0.5`.
    pub mmr_lambda: f32,
    /// Embedding dimension for FNV-1a embeddings. Defaults to `128`.
    pub dim: usize,
}
impl Default for DemoSelector {
    fn default() -> Self {
        Self {
            strategy: DemoSelectionStrategy::Knn,
            mmr_lambda: 0.5,
            dim: 128,
        }
    }
}
impl DemoSelector {
    /// Create a new selector.
    #[must_use]
    pub fn new(strategy: DemoSelectionStrategy, dim: usize) -> Self {
        Self {
            strategy,
            mmr_lambda: 0.5,
            dim,
        }
    }
    /// Set the MMR lambda.
    #[must_use]
    pub fn with_mmr_lambda(mut self, v: f32) -> Self {
        self.mmr_lambda = v;
        self
    }
    /// Select `k` demonstrations from `pool` for `query`.
    ///
    /// # Errors
    ///
    /// Returns [`PromptOptimizationError::EmptyPool`] if the pool is empty.
    /// Returns [`PromptOptimizationError::EmptyQuery`] if the query is empty.
    ///
    /// # Panics
    ///
    /// Does not panic in practice; internal `unwrap` calls are on non-empty iterators
    /// that are guarded by the `!remaining.is_empty()` loop condition.
    #[allow(clippy::too_many_lines)]
    pub fn select<'a>(
        &self,
        query: &str,
        pool: &'a DemoPool,
        k: usize,
    ) -> Result<Vec<&'a Demonstration>, PromptOptimizationError> {
        if pool.is_empty() {
            return Err(PromptOptimizationError::EmptyPool);
        }
        if query.trim().is_empty() {
            return Err(PromptOptimizationError::EmptyQuery);
        }
        let take = k.min(pool.demos.len());
        if take == 0 {
            return Ok(Vec::new());
        }
        match self.strategy {
            DemoSelectionStrategy::Deterministic => Ok(pool.demos.iter().take(take).collect()),
            DemoSelectionStrategy::Hardest => {
                let mut indexed: Vec<(usize, &Demonstration)> =
                    pool.demos.iter().enumerate().collect();
                indexed.sort_by(|(_, a), (_, b)| {
                    a.quality
                        .partial_cmp(&b.quality)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(indexed.into_iter().take(take).map(|(_, d)| d).collect())
            }
            DemoSelectionStrategy::Knn => {
                let q_emb = fnv_embed(query, self.dim);
                let mut scored: Vec<(usize, f32)> = pool
                    .demos
                    .iter()
                    .enumerate()
                    .map(|(i, d)| {
                        let d_emb = if d.embedding.len() == self.dim {
                            d.embedding.clone()
                        } else {
                            fnv_embed(&d.input, self.dim)
                        };
                        (i, cosine_sim(&q_emb, &d_emb))
                    })
                    .collect();
                scored.sort_by(|(_, a), (_, b)| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(scored
                    .into_iter()
                    .take(take)
                    .map(|(i, _)| &pool.demos[i])
                    .collect())
            }
            DemoSelectionStrategy::MmrDiverse => {
                let q_emb = fnv_embed(query, self.dim);
                let embeddings: Vec<Vec<f32>> = pool
                    .demos
                    .iter()
                    .map(|d| {
                        if d.embedding.len() == self.dim {
                            d.embedding.clone()
                        } else {
                            fnv_embed(&d.input, self.dim)
                        }
                    })
                    .collect();
                let mut selected: Vec<usize> = Vec::with_capacity(take);
                let mut remaining: Vec<usize> = (0..pool.demos.len()).collect();
                while selected.len() < take && !remaining.is_empty() {
                    let best_idx = if selected.is_empty() {
                        // First: pick the most relevant to the query
                        remaining
                            .iter()
                            .copied()
                            .max_by(|&a, &b| {
                                let sa = cosine_sim(&q_emb, &embeddings[a]);
                                let sb = cosine_sim(&q_emb, &embeddings[b]);
                                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap()
                    } else {
                        // MMR: balance relevance and diversity
                        let lambda = self.mmr_lambda;
                        remaining
                            .iter()
                            .copied()
                            .max_by(|&a, &b| {
                                let rel_a = cosine_sim(&q_emb, &embeddings[a]);
                                let max_sim_a = selected
                                    .iter()
                                    .map(|&s| cosine_sim(&embeddings[a], &embeddings[s]))
                                    .fold(f32::NEG_INFINITY, f32::max);
                                let mmr_a = lambda * rel_a - (1.0 - lambda) * max_sim_a;
                                let rel_b = cosine_sim(&q_emb, &embeddings[b]);
                                let max_sim_b = selected
                                    .iter()
                                    .map(|&s| cosine_sim(&embeddings[b], &embeddings[s]))
                                    .fold(f32::NEG_INFINITY, f32::max);
                                let mmr_b = lambda * rel_b - (1.0 - lambda) * max_sim_b;
                                mmr_a
                                    .partial_cmp(&mmr_b)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap()
                    };
                    selected.push(best_idx);
                    remaining.retain(|&x| x != best_idx);
                }
                Ok(selected.into_iter().map(|i| &pool.demos[i]).collect())
            }
        }
    }
}

// ── FNV-1a embedding helpers ──────────────────────────────────────────────────

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Deterministic FNV-1a bag-of-words embedding into `dim`-dimensional space.
fn fnv_embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let mut buckets = vec![0.0_f32; dim];
    for tok in &tokens {
        let mut h = FNV_OFFSET;
        for b in tok.as_bytes() {
            h = h.wrapping_mul(FNV_PRIME) ^ u64::from(*b);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h % dim as u64) as usize;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for b in &mut buckets {
            *b /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two equal-length vectors, clamped to [-1, 1].
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}
// ── PromptVariant ─────────────────────────────────────────────────────────────
/// A prompt variant to evaluate.
#[derive(Debug, Clone)]
pub struct PromptVariant {
    /// Unique name for this variant.
    pub name: String,
    /// Template string (may contain `{input}` placeholder).
    pub template: String,
    /// Optional instruction prefix.
    pub instruction: Option<String>,
}
impl PromptVariant {
    /// Create a new variant.
    #[must_use]
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            instruction: None,
        }
    }
    /// Attach an instruction.
    #[must_use]
    pub fn with_instruction(mut self, v: impl Into<String>) -> Self {
        self.instruction = Some(v.into());
        self
    }
    /// Render the template by substituting `{input}`.
    #[must_use]
    pub fn render(&self, input: &str) -> String {
        self.template.replace("{input}", input)
    }
}
// ── DevExample ────────────────────────────────────────────────────────────────
/// A development-set example for prompt variant evaluation.
#[derive(Debug, Clone)]
pub struct DevExample {
    /// The input text.
    pub input: String,
    /// The expected output.
    pub expected: String,
}
impl DevExample {
    /// Create a dev example.
    #[must_use]
    pub fn new(input: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            expected: expected.into(),
        }
    }
}
// ── OutputScorer ─────────────────────────────────────────────────────────────
/// Synchronous scorer comparing produced vs expected output.
pub trait OutputScorer {
    /// Score `produced` against `expected` in [0.0, 1.0].
    fn score(&self, produced: &str, expected: &str) -> f32;
}
// ── VariantScore ──────────────────────────────────────────────────────────────
/// Score of a prompt variant over the dev set.
#[derive(Debug, Clone)]
pub struct VariantScore {
    /// Name of the variant.
    pub variant_name: String,
    /// Mean score across all dev examples.
    pub score: f32,
    /// Per-example scores.
    pub per_example: Vec<f32>,
}
// ── PromptOptimizationConfig ──────────────────────────────────────────────────
/// Configuration for `PromptOptimizer`.
#[derive(Debug, Clone)]
pub struct PromptOptimizationConfig {
    /// Number of few-shot demonstrations to include. Defaults to `4`.
    pub num_demos: usize,
    /// MMR lambda. Defaults to `0.5`.
    pub mmr_lambda: f32,
    /// Embedding dimension. Defaults to `128`.
    pub dim: usize,
    /// Demo selection strategy. Defaults to [`DemoSelectionStrategy::Knn`].
    pub strategy: DemoSelectionStrategy,
}
impl Default for PromptOptimizationConfig {
    fn default() -> Self {
        Self {
            num_demos: 4,
            mmr_lambda: 0.5,
            dim: 128,
            strategy: DemoSelectionStrategy::Knn,
        }
    }
}
impl PromptOptimizationConfig {
    /// Set `num_demos`.
    #[must_use]
    pub fn with_num_demos(mut self, v: usize) -> Self {
        self.num_demos = v;
        self
    }
    /// Set `mmr_lambda`.
    #[must_use]
    pub fn with_mmr_lambda(mut self, v: f32) -> Self {
        self.mmr_lambda = v;
        self
    }
    /// Set dim.
    #[must_use]
    pub fn with_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }
    /// Set strategy.
    #[must_use]
    pub fn with_strategy(mut self, v: DemoSelectionStrategy) -> Self {
        self.strategy = v;
        self
    }
}
// ── PromptOptimizationError ───────────────────────────────────────────────────
/// Errors from the `prompt_optimization` module.
#[derive(Debug, Error)]
pub enum PromptOptimizationError {
    /// The demonstration pool was empty.
    #[error("Demonstration pool must not be empty")]
    EmptyPool,
    /// The dev set was empty.
    #[error("Dev set must not be empty")]
    EmptyDevSet,
    /// The query was empty.
    #[error("Query must not be empty")]
    EmptyQuery,
}
