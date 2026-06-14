//! Late chunking implementation (Jina AI 2024).
//!
//! Naive chunking embeds each chunk independently, so a chunk loses whole-document
//! context. Late chunking instead embeds the *whole* document into token-level
//! contextual vectors first, then pools the token vectors of each chunk's span —
//! so every chunk embedding carries document context.
//!
//! This module provides a pure-Rust simulation of that pipeline using
//! deterministic FNV-1a pseudo-embeddings (no external models, no randomness).
use crate::late_chunking::types::{LateChunk, LateChunkConfig, LateChunkError, LatePooling};

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokenise `text` into lower-cased alphanumeric tokens of length `>= 2`,
/// preserving document order.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

// ── Base token embedding ──────────────────────────────────────────────────────

/// Compute a deterministic FNV-1a base embedding for a single token.
///
/// The token is hashed into one bucket of a `dim`-wide vector (a one-hot-ish
/// accumulation), then L2-normalised so its norm is `1.0`.
fn base_token_embedding(token: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    if dim == 0 {
        return vec;
    }
    // Deterministic hash: FNV-1a.
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in token.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(1_099_511_628_211);
    }
    #[allow(clippy::cast_possible_truncation)]
    let idx = (h as usize) % dim;
    vec[idx] = 1.0;
    vec
}

// ── L2 normalisation ──────────────────────────────────────────────────────────

/// L2-normalise `vec` in place. A zero vector is left unchanged.
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

// ── Span tiling ───────────────────────────────────────────────────────────────

/// Compute the `[start, end)` token spans tiling `n_tokens` tokens using windows
/// of `chunk_size` with `overlap` overlapping tokens.
///
/// `overlap` is clamped to `chunk_size - 1` to guarantee forward progress, so a
/// non-empty document always yields at least one span.
fn token_spans(n_tokens: usize, chunk_size: usize, overlap: usize) -> Vec<(usize, usize)> {
    if n_tokens == 0 {
        return Vec::new();
    }
    let window = chunk_size.max(1);
    // Clamp overlap so the stride is at least 1 (forward progress guaranteed).
    let effective_overlap = overlap.min(window - 1);
    let stride = window - effective_overlap;

    let mut spans = Vec::new();
    let mut start = 0usize;
    while start < n_tokens {
        let end = (start + window).min(n_tokens);
        spans.push((start, end));
        if end == n_tokens {
            break;
        }
        start += stride;
    }
    spans
}

// ── Pooling ───────────────────────────────────────────────────────────────────

/// Pool the token vectors `tokens[start..end]` (each of length `dim`) into a
/// single `dim`-wide vector using `pooling`, then L2-normalise.
fn pool_span(
    tokens: &[Vec<f32>],
    start: usize,
    end: usize,
    dim: usize,
    pooling: LatePooling,
) -> Vec<f32> {
    let mut pooled = match pooling {
        LatePooling::Mean => vec![0.0f32; dim],
        LatePooling::Max => vec![f32::NEG_INFINITY; dim],
    };
    let span = &tokens[start..end];
    match pooling {
        LatePooling::Mean => {
            for tok in span {
                for (acc, &v) in pooled.iter_mut().zip(tok.iter()) {
                    *acc += v;
                }
            }
            if !span.is_empty() {
                #[allow(clippy::cast_precision_loss)]
                let count = span.len() as f32;
                for acc in &mut pooled {
                    *acc /= count;
                }
            }
        }
        LatePooling::Max => {
            for tok in span {
                for (acc, &v) in pooled.iter_mut().zip(tok.iter()) {
                    if v > *acc {
                        *acc = v;
                    }
                }
            }
            // Guard against an empty span leaving sentinel -inf values.
            if span.is_empty() {
                pooled.fill(0.0);
            }
        }
    }
    l2_normalize(&mut pooled);
    pooled
}

// ── LateChunker ───────────────────────────────────────────────────────────────

/// Late chunker that embeds a document into contextual token vectors before
/// pooling per-chunk embeddings.
#[derive(Debug, Clone)]
pub struct LateChunker {
    /// The active configuration.
    pub config: LateChunkConfig,
}

impl LateChunker {
    /// Create a new late chunker with the given configuration.
    #[must_use]
    pub fn new(config: LateChunkConfig) -> Self {
        Self { config }
    }

    /// Cosine similarity of two equal-length vectors.
    ///
    /// Returns `0.0` if the lengths differ, either vector is empty, or either
    /// has zero magnitude.
    #[must_use]
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    /// Build the per-token base embeddings for `tokens`.
    fn base_embeddings(&self, tokens: &[String]) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .map(|t| base_token_embedding(t, self.config.dim))
            .collect()
    }

    /// Contextualise the base token embeddings by blending in the document mean.
    ///
    /// `contextual[i] = l2_normalize((1 - w) * base[i] + w * doc_mean)`.
    fn contextualize(&self, base: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let dim = self.config.dim;
        if base.is_empty() || dim == 0 {
            return base.to_vec();
        }
        // Document mean of the base token embeddings.
        let mut doc_mean = vec![0.0f32; dim];
        for tok in base {
            for (acc, &v) in doc_mean.iter_mut().zip(tok.iter()) {
                *acc += v;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let count = base.len() as f32;
        for acc in &mut doc_mean {
            *acc /= count;
        }

        let w = self.config.context_weight;
        base.iter()
            .map(|tok| {
                let mut ctx: Vec<f32> = tok
                    .iter()
                    .zip(doc_mean.iter())
                    .map(|(&b, &m)| (1.0 - w) * b + w * m)
                    .collect();
                l2_normalize(&mut ctx);
                ctx
            })
            .collect()
    }

    /// Reconstruct the chunk text for the token span `[start, end)`.
    fn span_text(tokens: &[String], start: usize, end: usize) -> String {
        tokens[start..end].join(" ")
    }

    /// Encode a document into late chunks.
    ///
    /// The whole document is embedded into contextual token vectors first, then
    /// each chunk's span is mean/max-pooled so every chunk embedding carries
    /// document context.
    ///
    /// # Errors
    ///
    /// Returns [`LateChunkError::EmptyDocument`] when `text` yields no tokens.
    pub fn encode_document(&self, text: &str) -> Result<Vec<LateChunk>, LateChunkError> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Err(LateChunkError::EmptyDocument);
        }
        let base = self.base_embeddings(&tokens);
        let contextual = self.contextualize(&base);
        let spans = token_spans(
            tokens.len(),
            self.config.chunk_size_tokens,
            self.config.overlap_tokens,
        );
        Ok(spans
            .into_iter()
            .map(|(start, end)| LateChunk {
                text: Self::span_text(&tokens, start, end),
                token_start: start,
                token_end: end,
                embedding: pool_span(
                    &contextual,
                    start,
                    end,
                    self.config.dim,
                    self.config.pooling,
                ),
            })
            .collect())
    }

    /// Encode a query as a context-free base embedding (no document mean),
    /// L2-normalised.
    #[must_use]
    pub fn encode_query(&self, q: &str) -> Vec<f32> {
        let tokens = tokenize(q);
        let dim = self.config.dim;
        let mut vec = vec![0.0f32; dim];
        if dim == 0 {
            return vec;
        }
        for tok in &tokens {
            let emb = base_token_embedding(tok, dim);
            for (acc, &v) in vec.iter_mut().zip(emb.iter()) {
                *acc += v;
            }
        }
        l2_normalize(&mut vec);
        vec
    }

    /// Chunk a document the naive way: embed each chunk's tokens independently
    /// using only base embeddings (no document context), then pool.
    ///
    /// Used as a baseline so callers can observe the difference late chunking
    /// makes. When `context_weight == 0.0`, late chunks equal naive chunks.
    ///
    /// # Errors
    ///
    /// Returns [`LateChunkError::EmptyDocument`] when `text` yields no tokens.
    pub fn naive_chunks(&self, text: &str) -> Result<Vec<LateChunk>, LateChunkError> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Err(LateChunkError::EmptyDocument);
        }
        let spans = token_spans(
            tokens.len(),
            self.config.chunk_size_tokens,
            self.config.overlap_tokens,
        );
        Ok(spans
            .into_iter()
            .map(|(start, end)| {
                // Embed only this chunk's tokens with the base embedder.
                let base: Vec<Vec<f32>> = tokens[start..end]
                    .iter()
                    .map(|t| base_token_embedding(t, self.config.dim))
                    .collect();
                let embedding =
                    pool_span(&base, 0, base.len(), self.config.dim, self.config.pooling);
                LateChunk {
                    text: Self::span_text(&tokens, start, end),
                    token_start: start,
                    token_end: end,
                    embedding,
                }
            })
            .collect())
    }
}
