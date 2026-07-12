//! Indexing and blended structural/textual search for `code_retrieval`.
//!
//! [`CodeRetrievalEngine`] parses a corpus into [`CodeRetrievalUnit`]s (via the
//! [`parser`](crate::code_retrieval::parser)), derives per-unit structural
//! feature sets and a deterministic FNV-1a pseudo-embedding, and stores them in
//! a [`CodeRetrievalIndex`]. A search blends two similarities into one score:
//!
//! * **Structural** — the mean of the enabled Jaccard overlaps between the
//!   query's and a unit's identifier sets, import sets, and call-site sets.
//! * **Textual** — the cosine similarity of the query's and the unit's FNV-1a
//!   bag-of-tokens pseudo-embeddings (raw text, comments included).
//!
//! The blend is `blend_weight * structural + (1 - blend_weight) * text`, so a
//! `blend_weight` of `1.0` is structural-only and `0.0` is text-only. The
//! parse/embed parameters are stamped into the index at build time, so the same
//! index can be searched under different blend weights without re-indexing.

use std::cmp::Ordering;
use std::collections::HashSet;

use crate::code_retrieval::parser::{ident_tokens, parse_source};
use crate::code_retrieval::types::{
    CodeRetrievalConfig, CodeRetrievalError, CodeRetrievalHit, CodeRetrievalResult,
    CodeRetrievalUnit,
};

// ── Pseudo-embedding primitives ──────────────────────────────────────────────

/// FNV-1a domain-separation seed for the text pseudo-embedding space. This is a
/// well-known 64-bit mixing constant, reused here purely as a fixed, arbitrary
/// seed (nothing cryptographic is implied).
const TEXT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// FNV-1a 64-bit hash of `bytes`, seeded with `seed` (offset basis
/// `0xcbf29ce484222325`, prime `1099511628211`).
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// L2-normalise `vector` in place. A zero (or near-zero) vector is left
/// unchanged rather than divided by (approximately) zero.
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// Deterministic FNV-1a bag-of-tokens pseudo-embedding of `text`.
///
/// Every identifier token of length two or more (comments included, lowercased)
/// increments one hashed bucket; the resulting histogram is L2-normalised, so a
/// cosine of two such vectors is their dot product. A `dim` of `0` yields an
/// empty vector.
fn text_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut buckets = vec![0.0f32; dim];
    if dim == 0 {
        return buckets;
    }
    for token in ident_tokens(text) {
        let lower = token.to_lowercase();
        if lower.chars().count() < 2 {
            continue;
        }
        let hash = fnv1a(lower.as_bytes(), TEXT_SEED);
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash as usize) % dim;
        buckets[idx] += 1.0;
    }
    l2_normalize(&mut buckets);
    buckets
}

/// Cosine similarity of two equal-length, L2-normalised vectors (their dot
/// product), clamped to `[0.0, 1.0]`. Mismatched-length or empty inputs score
/// `0.0`. Because every bucket count is non-negative, the raw dot never drops
/// below zero.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

/// Jaccard overlap `|A ∩ B| / |A ∪ B|` of two identifier/import/call sets. Two
/// empty sets score `0.0` (there is no structural evidence to share).
#[allow(clippy::cast_precision_loss)]
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Blend the three structural sub-signals into a single structural score: the
/// mean of the enabled Jaccard overlaps. With no signal enabled the structural
/// score is `0.0`.
#[allow(clippy::cast_precision_loss)]
fn structural_blend(
    identifier_overlap: f32,
    import_overlap: f32,
    call_site_overlap: f32,
    config: &CodeRetrievalConfig,
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    if config.use_identifiers {
        sum += identifier_overlap;
        count += 1;
    }
    if config.use_imports {
        sum += import_overlap;
        count += 1;
    }
    if config.use_call_sites {
        sum += call_site_overlap;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

// ── Per-unit indexed features ────────────────────────────────────────────────

/// The precomputed features of one indexed unit: its pseudo-embedding and its
/// three structural sets, kept parallel to [`CodeRetrievalIndex::units`].
#[derive(Debug, Clone, PartialEq)]
struct IndexedFeatures {
    /// The FNV-1a pseudo-embedding of the unit's raw text.
    embedding: Vec<f32>,
    /// The unit's distinct identifier names.
    identifiers: HashSet<String>,
    /// The unit's distinct imported module/paths.
    imports: HashSet<String>,
    /// The unit's distinct call-site names.
    calls: HashSet<String>,
}

/// The query counterpart of [`IndexedFeatures`]: the aggregated feature sets and
/// embedding of a parsed query.
struct QueryFeatures {
    /// The query's pseudo-embedding.
    embedding: Vec<f32>,
    /// The query's distinct identifier names.
    identifiers: HashSet<String>,
    /// The query's distinct imported module/paths.
    imports: HashSet<String>,
    /// The query's distinct call-site names.
    calls: HashSet<String>,
}

// ── CodeRetrievalIndex ───────────────────────────────────────────────────────

/// An immutable, searchable index of a parsed code corpus.
///
/// Holds every extracted [`CodeRetrievalUnit`] alongside its precomputed
/// structural feature sets and pseudo-embedding, plus a copy of the
/// [`CodeRetrievalConfig`] the corpus was parsed and embedded with (so queries
/// are tokenised and embedded into the same space).
#[derive(Debug, Clone, PartialEq)]
pub struct CodeRetrievalIndex {
    /// The extracted units, in corpus order.
    units: Vec<CodeRetrievalUnit>,
    /// Per-unit features, parallel to `units`.
    features: Vec<IndexedFeatures>,
    /// The configuration used to build this index.
    config: CodeRetrievalConfig,
}

impl CodeRetrievalIndex {
    /// The indexed units, in corpus order.
    #[must_use]
    pub fn units(&self) -> &[CodeRetrievalUnit] {
        &self.units
    }

    /// The number of indexed units.
    #[must_use]
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Return `true` when the index holds no units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// The pseudo-embedding dimensionality this index was built with.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// The configuration this index was built with.
    #[must_use]
    pub fn config(&self) -> &CodeRetrievalConfig {
        &self.config
    }
}

// ── CodeRetrievalEngine ──────────────────────────────────────────────────────

/// Builds [`CodeRetrievalIndex`]es and runs blended structural/textual search.
///
/// The engine is a thin holder of a [`CodeRetrievalConfig`]: parse/embed
/// parameters are used when building an index; the blend weight, `top_k`, and
/// signal toggles are read at search time. A single index can therefore be
/// searched by several engines carrying different score-time configurations, as
/// long as their `embedding_dim` matches the index's.
#[derive(Debug, Clone, Default)]
pub struct CodeRetrievalEngine {
    /// This engine's configuration.
    pub config: CodeRetrievalConfig,
}

impl CodeRetrievalEngine {
    /// Create an engine with the given configuration.
    #[must_use]
    pub fn new(config: CodeRetrievalConfig) -> Self {
        Self { config }
    }

    /// Parse a single source item into structural units, using this engine's
    /// configuration. A thin convenience wrapper over
    /// [`parse_source`].
    #[must_use]
    pub fn parse(&self, source_id: &str, text: &str) -> Vec<CodeRetrievalUnit> {
        parse_source(source_id, text, &self.config)
    }

    /// Build a searchable index from `items`, each a `(source_id, source_text)`
    /// pair.
    ///
    /// Every item is parsed into one or more units, and each unit is given a
    /// pseudo-embedding and structural feature sets.
    ///
    /// # Errors
    ///
    /// * [`CodeRetrievalError::ZeroEmbeddingDim`] when the configured embedding
    ///   dimension is zero.
    /// * [`CodeRetrievalError::EmptyCorpus`] when `items` yields nothing.
    pub fn index<I, S, T>(&self, items: I) -> Result<CodeRetrievalIndex, CodeRetrievalError>
    where
        I: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
        T: AsRef<str>,
    {
        if self.config.embedding_dim == 0 {
            return Err(CodeRetrievalError::ZeroEmbeddingDim);
        }

        let mut units: Vec<CodeRetrievalUnit> = Vec::new();
        let mut item_count = 0usize;
        for (id, text) in items {
            item_count += 1;
            units.extend(parse_source(id.as_ref(), text.as_ref(), &self.config));
        }

        if item_count == 0 {
            return Err(CodeRetrievalError::EmptyCorpus);
        }

        let features = units
            .iter()
            .map(|unit| IndexedFeatures {
                embedding: text_embed(&unit.raw_text, self.config.embedding_dim),
                identifiers: unit.identifier_set(),
                imports: unit.imports.iter().cloned().collect(),
                calls: unit.call_sites.iter().cloned().collect(),
            })
            .collect();

        Ok(CodeRetrievalIndex {
            units,
            features,
            config: self.config.clone(),
        })
    }

    /// Extract the aggregate feature sets and pseudo-embedding of a query,
    /// parsed in the index's own space (so its tokens and embedding are
    /// comparable to the corpus).
    fn query_features(index: &CodeRetrievalIndex, query: &str) -> QueryFeatures {
        let query_units = parse_source("<query>", query, index.config());
        let mut identifiers = HashSet::new();
        let mut imports = HashSet::new();
        let mut calls = HashSet::new();
        for unit in &query_units {
            for symbol in &unit.symbols {
                identifiers.insert(symbol.name.clone());
            }
            for import in &unit.imports {
                imports.insert(import.clone());
            }
            for call in &unit.call_sites {
                calls.insert(call.clone());
            }
        }
        QueryFeatures {
            embedding: text_embed(query, index.embedding_dim()),
            identifiers,
            imports,
            calls,
        }
    }

    /// Search `index` for the units most relevant to `query`, returning up to
    /// [`CodeRetrievalConfig::top_k`] hits ranked best-first.
    ///
    /// `query` may be a plain-text description or another code snippet; either
    /// way it is parsed with the same structural tokenizer and embedded into the
    /// same space as the corpus, then scored by the configured structural/text
    /// blend. Ties are broken deterministically by `source_id` then
    /// `unit_index`.
    ///
    /// # Errors
    ///
    /// * [`CodeRetrievalError::ZeroEmbeddingDim`] or
    ///   [`CodeRetrievalError::InvalidBlendWeight`] when this engine's
    ///   configuration is invalid.
    /// * [`CodeRetrievalError::DimensionMismatch`] when this engine's embedding
    ///   dimension differs from the index's.
    /// * [`CodeRetrievalError::EmptyQuery`] when `query` is empty or only
    ///   whitespace.
    pub fn search(
        &self,
        index: &CodeRetrievalIndex,
        query: &str,
    ) -> Result<CodeRetrievalResult, CodeRetrievalError> {
        self.config.validate()?;
        if index.embedding_dim() != self.config.embedding_dim {
            return Err(CodeRetrievalError::DimensionMismatch {
                index_dim: index.embedding_dim(),
                config_dim: self.config.embedding_dim,
            });
        }
        if query.trim().is_empty() {
            return Err(CodeRetrievalError::EmptyQuery);
        }

        let query_features = Self::query_features(index, query);

        let mut hits: Vec<CodeRetrievalHit> = Vec::with_capacity(index.len());
        for (unit, feature) in index.units.iter().zip(index.features.iter()) {
            let identifier_overlap = jaccard(&query_features.identifiers, &feature.identifiers);
            let import_overlap = jaccard(&query_features.imports, &feature.imports);
            let call_site_overlap = jaccard(&query_features.calls, &feature.calls);
            let structural_score = structural_blend(
                identifier_overlap,
                import_overlap,
                call_site_overlap,
                &self.config,
            );
            let text_score = cosine(&query_features.embedding, &feature.embedding);
            let weight = self.config.blend_weight;
            let score = weight.mul_add(structural_score, (1.0 - weight) * text_score);

            hits.push(CodeRetrievalHit {
                unit: unit.clone(),
                score,
                structural_score,
                text_score,
                identifier_overlap,
                import_overlap,
                call_site_overlap,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.unit.source_id.cmp(&b.unit.source_id))
                .then_with(|| a.unit.unit_index.cmp(&b.unit.unit_index))
        });

        let units_searched = index.len();
        hits.truncate(self.config.top_k);

        Ok(CodeRetrievalResult {
            query: query.to_string(),
            hits,
            units_searched,
        })
    }
}
