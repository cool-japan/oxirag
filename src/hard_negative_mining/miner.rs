//! The single-round mining procedure — the algorithmic core.
//!
//! This file owns the deterministic pseudo-embedding function (an
//! `FNV-1a`/`splitmix64` construction, self-contained rather than shared with
//! other modules), the per-query corpus ranking, and the hard-negative
//! selection that together make up mining one [`MiningRound`] under a single
//! [`EmbeddingVersion`].
//!
//! The embedding of a text under a version blends two components:
//!
//! 1. a shared, **version-independent content histogram** (an `FNV-1a`
//!    bag-of-tokens over `embedding_dim` buckets, L2-normalised) — this gives
//!    lexically-similar texts similar vectors, so that "hard" negatives are
//!    genuinely the plausible-looking wrong documents; and
//! 2. a **version-specific pseudo-random component** (a `splitmix64` stream
//!    seeded from the text *and* the version seed, L2-normalised) whose weight
//!    is [`EmbeddingVersion::effective_drift`].
//!
//! At drift `0.0` the embedding is the pure content histogram (identical for
//! every seed — the canonical model); as drift grows the version-specific
//! component increasingly reorders rankings. That is exactly how a re-trained
//! model is simulated: a small drift barely perturbs the ranking (mined
//! negatives stay fresh), a large drift scrambles it (they go stale).

use std::collections::HashSet;

use crate::types::DocumentId;

use super::types::{
    EmbeddingVersion, HardNegativeConfig, HardNegativeDocument, HardNegativeError,
    HardNegativePositivePair, HardNegativeQueryResult, HardNegativeResult, HardNegativeSample,
    MiningRound,
};

// ── Hashing primitives ────────────────────────────────────────────────────────

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;
/// Fixed, arbitrary seed domain-separating the content histogram's token
/// hashing from the version-specific noise stream.
const TOKEN_SEED: u64 = 0x51ed_2701_9b3c_a7f5;
/// The `splitmix64` increment (the golden-ratio odd constant, Vigna, public
/// domain).
const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// `FNV-1a` 64-bit hash of `bytes`, seeded with `seed`.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// One `splitmix64` mixing step (Vigna, public domain): advances `state` and
/// returns a well-mixed pseudo-random `u64`.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map a `u64` to a signed unit-ish value in `[-1.0, 1.0)` using its top 24
/// bits.
#[allow(clippy::cast_precision_loss)]
fn signed_unit(bits: u64) -> f32 {
    // Top 24 bits give an exactly-representable integer in `[0, 2^24)`.
    #[allow(clippy::cast_possible_truncation)]
    let top24 = (bits >> 40) as u32;
    let unit = top24 as f32 / 16_777_216.0; // 2^24, so `unit` is in [0.0, 1.0)
    unit.mul_add(2.0, -1.0) // [-1.0, 1.0)
}

// ── Tokenisation ──────────────────────────────────────────────────────────────

/// Tokenise `text`: split on non-alphanumeric boundaries, lowercase, drop empty
/// fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The lowercased, whitespace-collapsed normalisation of `text`, used to seed
/// the version-specific noise stream.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

// ── Embedding ─────────────────────────────────────────────────────────────────

/// L2-normalise `vector` in place. A (near-)zero vector is left untouched
/// rather than divided by (approximately) zero.
fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for v in vector.iter_mut() {
            *v /= norm;
        }
    }
}

/// The shared, version-independent content histogram: an `FNV-1a`
/// bag-of-tokens over `dim` buckets, L2-normalised. A `dim` of `0` yields an
/// empty vector.
fn content_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut buckets = vec![0.0f32; dim];
    if dim == 0 {
        return buckets;
    }
    for token in tokenize(text) {
        let hash = fnv1a(token.as_bytes(), TOKEN_SEED);
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash % dim as u64) as usize;
        buckets[idx] += 1.0;
    }
    l2_normalize(&mut buckets);
    buckets
}

/// The version-specific pseudo-random component: a `splitmix64` stream seeded
/// from `text` *and* `seed`, one step per dimension, L2-normalised. Different
/// seeds point in unrelated directions; identical `(text, seed)` reproduce the
/// same vector.
fn version_noise(text: &str, seed: u64, dim: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dim];
    if dim == 0 {
        return vector;
    }
    let text_hash = fnv1a(normalize(text).as_bytes(), seed);
    for (i, slot) in vector.iter_mut().enumerate() {
        let dim_mix = (i as u64).wrapping_mul(SPLITMIX_INCREMENT);
        let mut state = text_hash ^ dim_mix;
        *slot = signed_unit(splitmix64_next(&mut state));
    }
    l2_normalize(&mut vector);
    vector
}

/// Embed `text` under `version` into a `dim`-dimensional unit vector.
///
/// Returns `content` at drift `0.0`, a drift-weighted blend of `content` and
/// [`version_noise`] otherwise, always re-normalised. Deterministic in
/// `(text, version, dim)`.
pub(crate) fn embed(text: &str, version: EmbeddingVersion, dim: usize) -> Vec<f32> {
    let mut content = content_embedding(text, dim);
    let drift = version.effective_drift();
    if drift <= 0.0 || dim == 0 {
        return content;
    }
    let noise = version_noise(text, version.seed, dim);
    let keep = 1.0 - drift;
    for (c, n) in content.iter_mut().zip(noise.iter()) {
        *c = keep * *c + drift * *n;
    }
    l2_normalize(&mut content);
    content
}

/// Cosine similarity of two equal-length vectors. Since callers pass
/// L2-normalised vectors this is their dot product; a zero vector yields `0.0`.
/// Mismatched lengths yield `0.0`.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Ranking ───────────────────────────────────────────────────────────────────

/// One ranked corpus entry for a query: `(doc_id, 1-based rank, similarity)`.
type RankedEntry = (DocumentId, usize, f32);

/// Rank every document in `corpus_vectors` against `query_vector`, most similar
/// first.
///
/// Ties in similarity are broken by ascending document id so the ranking is
/// fully deterministic. Returns entries carrying their 1-based rank.
fn rank_corpus(
    query_vector: &[f32],
    corpus: &[HardNegativeDocument],
    corpus_vectors: &[Vec<f32>],
) -> Vec<RankedEntry> {
    let mut scored: Vec<(DocumentId, f32)> = corpus
        .iter()
        .zip(corpus_vectors.iter())
        .map(|(doc, vector)| (doc.id.clone(), cosine(query_vector, vector)))
        .collect();

    scored.sort_by(|a, b| {
        // Descending similarity (total order via `total_cmp`), then ascending
        // id as a deterministic tie-break. `DocumentId` is not `Ord`, so
        // compare its string form.
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.as_str().cmp(b.0.as_str()))
    });

    scored
        .into_iter()
        .enumerate()
        .map(|(index, (id, similarity))| (id, index + 1, similarity))
        .collect()
}

/// Select the hard negatives for one query from its ranking.
///
/// Walks the ranking best-first, skips the labelled positive when
/// [`HardNegativeConfig::exclude_positive`] is set, stops once the rank exceeds
/// [`HardNegativeConfig::hard_rank_cutoff`] (everything beyond is an *easy*
/// negative), and stops once [`HardNegativeConfig::negatives_per_query`] have
/// been collected. Also returns the labelled positive's own 1-based rank.
fn select_negatives(
    ranking: &[RankedEntry],
    positive_id: &DocumentId,
    config: &HardNegativeConfig,
) -> (usize, Vec<HardNegativeSample>) {
    // The positive is guaranteed present in the corpus (validated up front); if
    // it were somehow absent, fall back to the worst possible rank rather than
    // panicking.
    let positive_rank = ranking
        .iter()
        .find(|(id, _, _)| id == positive_id)
        .map_or(ranking.len(), |&(_, rank, _)| rank);

    let mut negatives = Vec::new();
    for (doc_id, rank, similarity) in ranking {
        if *rank > config.hard_rank_cutoff {
            // Every remaining candidate ranks below the cutoff, i.e. is an easy
            // negative: no further hard negatives exist.
            break;
        }
        if config.exclude_positive && doc_id == positive_id {
            continue;
        }
        negatives.push(HardNegativeSample {
            doc_id: doc_id.clone(),
            rank: *rank,
            similarity: *similarity,
        });
        if negatives.len() >= config.negatives_per_query {
            break;
        }
    }

    (positive_rank, negatives)
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate the config, corpus, and positives once, up front.
///
/// # Errors
///
/// See [`HardNegativeError`]: empty corpus/positives, duplicate corpus ids,
/// empty queries, unknown positive ids, and the zero-valued config guards.
pub(crate) fn validate_inputs(
    config: &HardNegativeConfig,
    positives: &[HardNegativePositivePair],
    corpus: &[HardNegativeDocument],
) -> HardNegativeResult<()> {
    config.validate()?;

    if corpus.is_empty() {
        return Err(HardNegativeError::EmptyCorpus);
    }
    if positives.is_empty() {
        return Err(HardNegativeError::EmptyPositives);
    }

    let mut seen: HashSet<&DocumentId> = HashSet::with_capacity(corpus.len());
    for doc in corpus {
        if !seen.insert(&doc.id) {
            return Err(HardNegativeError::DuplicateDocumentId(doc.id.clone()));
        }
    }

    for pair in positives {
        if pair.query.trim().is_empty() {
            return Err(HardNegativeError::EmptyQuery);
        }
        if !seen.contains(&pair.positive_id) {
            return Err(HardNegativeError::UnknownPositiveDocument(
                pair.positive_id.clone(),
            ));
        }
    }

    Ok(())
}

// ── Single-round mining ───────────────────────────────────────────────────────

/// Mine one [`MiningRound`] under `version`.
///
/// Embeds the whole corpus once under `version`, then for every positive pair
/// embeds the query, ranks the corpus, records the positive's rank, and selects
/// the top hard negatives (see [`select_negatives`]). Finally computes the mean
/// positive rank across queries.
///
/// # Errors
///
/// See [`validate_inputs`].
pub(crate) fn mine_round(
    config: &HardNegativeConfig,
    positives: &[HardNegativePositivePair],
    corpus: &[HardNegativeDocument],
    version: EmbeddingVersion,
    round_index: usize,
) -> HardNegativeResult<MiningRound> {
    validate_inputs(config, positives, corpus)?;

    let dim = config.embedding_dim;
    let corpus_vectors: Vec<Vec<f32>> = corpus
        .iter()
        .map(|doc| embed(&doc.text, version, dim))
        .collect();

    let mut results = Vec::with_capacity(positives.len());
    let mut rank_sum: usize = 0;

    for pair in positives {
        let query_vector = embed(&pair.query, version, dim);
        let ranking = rank_corpus(&query_vector, corpus, &corpus_vectors);
        let (positive_rank, negatives) = select_negatives(&ranking, &pair.positive_id, config);
        rank_sum = rank_sum.saturating_add(positive_rank);
        results.push(HardNegativeQueryResult {
            query: pair.query.clone(),
            positive_id: pair.positive_id.clone(),
            positive_rank,
            negatives,
        });
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let mean_positive_rank = if results.is_empty() {
        0.0
    } else {
        (rank_sum as f64 / results.len() as f64) as f32
    };

    Ok(MiningRound {
        round_index,
        version,
        results,
        mean_positive_rank,
    })
}
