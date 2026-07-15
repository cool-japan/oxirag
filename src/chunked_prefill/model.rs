//! Where a chunk's keys, values and queries come from.
//!
//! Chunked prefill is a statement about *scheduling and bookkeeping*, not about
//! transformer weights: it says that if you feed the attention kernel the right
//! keys, values, queries and positions, chunk by chunk, you get the one-shot
//! answer back. [`PrefillModel`] is the seam at which those tensors arrive, and
//! it is deliberately narrow — a projection stage on the other side of it can be
//! anything at all.
//!
//! [`PrefillTensorModel`] is the honest in-memory implementation: you hand it
//! the `K`, `V` and `Q` your forward pass produced, and it hands them back. It
//! computes nothing and pretends nothing.

use crate::kv_cache_compression::KvCacheTensor;

use super::types::{ChunkedPrefillError, ChunkedPrefillResult, PrefillGeometry};

// ── PrefillModel ─────────────────────────────────────────────────────────────

/// The source of one prompt's keys, values and queries.
///
/// # Layouts
///
/// | Method | Layout | Length |
/// |---|---|---|
/// | [`Self::token_keys`], [`Self::token_values`] | `[layer][head][dim]` | [`PrefillGeometry::token_stride`] |
/// | [`Self::token_query`] | `[head][dim]` | [`PrefillGeometry::layer_stride`] |
///
/// The key/value layout is not a matter of taste: it is exactly what
/// [`KvCacheTensor::append_token`] consumes, so a chunk's keys can be appended
/// without a transpose. The queries are per-layer because the attention kernel
/// is per-layer.
pub trait PrefillModel {
    /// The `[num_layers, num_heads, head_dim]` shape of this model's tensors.
    fn geometry(&self) -> PrefillGeometry;

    /// How many tokens the prompt holds.
    fn prompt_tokens(&self) -> usize;

    /// Keys of prompt token `token`, laid out `[layer][head][dim]`.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::TokenOutOfRange`] if the token does not exist.
    fn token_keys(&self, token: usize) -> ChunkedPrefillResult<&[f32]>;

    /// Values of prompt token `token`, laid out `[layer][head][dim]`.
    ///
    /// # Errors
    ///
    /// [`ChunkedPrefillError::TokenOutOfRange`] if the token does not exist.
    fn token_values(&self, token: usize) -> ChunkedPrefillResult<&[f32]>;

    /// The query vector of prompt token `token` at layer `layer`, laid out
    /// `[head][dim]`.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::TokenOutOfRange`] if the token does not exist.
    /// - [`ChunkedPrefillError::LayerOutOfRange`] if the layer does not exist.
    fn token_query(&self, layer: usize, token: usize) -> ChunkedPrefillResult<&[f32]>;
}

// ── PrefillTensorModel ───────────────────────────────────────────────────────

/// A prompt whose keys, values and queries are already computed and held in
/// memory.
///
/// Every buffer is token-major — `[token][layer][head][dim]` — so one token's
/// key block is a contiguous slice of exactly the length
/// [`KvCacheTensor::append_token`] wants, and appending it copies rather than
/// gathers.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillTensorModel {
    geometry: PrefillGeometry,
    prompt_tokens: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    queries: Vec<f32>,
}

impl PrefillTensorModel {
    /// Take ownership of a prompt's `K`, `V` and `Q`.
    ///
    /// All three buffers are `[token][layer][head][dim]` and must each hold
    /// exactly `prompt_tokens * geometry.token_stride()` elements.
    ///
    /// # Errors
    ///
    /// - [`ChunkedPrefillError::EmptyPrompt`] if `prompt_tokens` is zero.
    /// - [`ChunkedPrefillError::ShapeMismatch`] if a buffer has the wrong length.
    /// - [`ChunkedPrefillError::NonFiniteInput`] if a buffer holds a `NaN` or an
    ///   infinity. Finiteness is enforced here, at the boundary, because it is
    ///   the precondition the attention kernel's finiteness proof rests on.
    pub fn new(
        geometry: PrefillGeometry,
        prompt_tokens: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
        queries: Vec<f32>,
    ) -> ChunkedPrefillResult<Self> {
        if prompt_tokens == 0 {
            return Err(ChunkedPrefillError::EmptyPrompt);
        }
        let expected = prompt_tokens * geometry.token_stride();
        check_len("key buffer", &keys, expected)?;
        check_len("value buffer", &values, expected)?;
        check_len("query buffer", &queries, expected)?;
        check_finite("key buffer", &keys)?;
        check_finite("value buffer", &values)?;
        check_finite("query buffer", &queries)?;
        Ok(Self {
            geometry,
            prompt_tokens,
            keys,
            values,
            queries,
        })
    }

    /// The whole key buffer, `[token][layer][head][dim]`.
    #[must_use]
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    /// The whole value buffer, `[token][layer][head][dim]`.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The whole query buffer, `[token][layer][head][dim]`.
    #[must_use]
    pub fn queries(&self) -> &[f32] {
        &self.queries
    }

    /// The `[token][layer][head][dim]` slice of `buffer` belonging to one token.
    fn token_slice<'a>(&self, buffer: &'a [f32], token: usize) -> ChunkedPrefillResult<&'a [f32]> {
        if token >= self.prompt_tokens {
            return Err(ChunkedPrefillError::TokenOutOfRange {
                token,
                prompt_tokens: self.prompt_tokens,
            });
        }
        let stride = self.geometry.token_stride();
        let start = token * stride;
        buffer
            .get(start..start + stride)
            .ok_or(ChunkedPrefillError::ShapeMismatch {
                what: "token slice",
                expected: start + stride,
                actual: buffer.len(),
            })
    }
}

impl PrefillModel for PrefillTensorModel {
    fn geometry(&self) -> PrefillGeometry {
        self.geometry
    }

    fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    fn token_keys(&self, token: usize) -> ChunkedPrefillResult<&[f32]> {
        self.token_slice(&self.keys, token)
    }

    fn token_values(&self, token: usize) -> ChunkedPrefillResult<&[f32]> {
        self.token_slice(&self.values, token)
    }

    fn token_query(&self, layer: usize, token: usize) -> ChunkedPrefillResult<&[f32]> {
        if layer >= self.geometry.num_layers() {
            return Err(ChunkedPrefillError::LayerOutOfRange {
                layer,
                num_layers: self.geometry.num_layers(),
            });
        }
        let token_block = self.token_slice(&self.queries, token)?;
        let stride = self.geometry.layer_stride();
        let start = layer * stride;
        token_block
            .get(start..start + stride)
            .ok_or(ChunkedPrefillError::ShapeMismatch {
                what: "layer query block",
                expected: start + stride,
                actual: token_block.len(),
            })
    }
}

// ── Geometry agreement ───────────────────────────────────────────────────────

/// Confirm that a cache can hold what a model produces.
///
/// # Errors
///
/// [`ChunkedPrefillError::GeometryMismatch`] if any of the three dimensions
/// disagree.
pub fn check_cache_geometry(
    geometry: PrefillGeometry,
    cache: &KvCacheTensor,
) -> ChunkedPrefillResult<()> {
    if geometry.num_layers() != cache.num_layers()
        || geometry.num_heads() != cache.num_heads()
        || geometry.head_dim() != cache.head_dim()
    {
        return Err(ChunkedPrefillError::GeometryMismatch {
            model_layers: geometry.num_layers(),
            model_heads: geometry.num_heads(),
            model_head_dim: geometry.head_dim(),
            cache_layers: cache.num_layers(),
            cache_heads: cache.num_heads(),
            cache_head_dim: cache.head_dim(),
        });
    }
    Ok(())
}

fn check_len(what: &'static str, buffer: &[f32], expected: usize) -> ChunkedPrefillResult<()> {
    if buffer.len() == expected {
        return Ok(());
    }
    Err(ChunkedPrefillError::ShapeMismatch {
        what,
        expected,
        actual: buffer.len(),
    })
}

fn check_finite(what: &'static str, buffer: &[f32]) -> ChunkedPrefillResult<()> {
    for (index, value) in buffer.iter().enumerate() {
        if !value.is_finite() {
            return Err(ChunkedPrefillError::NonFiniteInput { what, index });
        }
    }
    Ok(())
}
