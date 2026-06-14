//! Diversity and coverage metric functions.
//!
//! Each function is free-standing and deterministic. Embeddings are the FNV-1a
//! lexical pseudo-embeddings produced by [`embed`], mirroring the scheme used
//! elsewhere in `OxiRAG` so cosine similarities are comparable across modules.

use std::collections::BTreeSet;

// ── Deterministic lexical pseudo-embedding ────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries → hash each token of
/// length >= 2 to a bucket with FNV-1a → accumulate per-bucket counts →
/// L2-normalise to the requested `dim`. Returns an empty vector when `dim` is
/// `0`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
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

// ── Cosine similarity ─────────────────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors.
///
/// Returns `0.0` for mismatched or empty lengths and for zero-magnitude inputs.
/// The result is clamped to `[-1.0, 1.0]`.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

// ── Intra-List Diversity (ILD) ────────────────────────────────────────────────

/// Intra-List Diversity over the first `k` of `embeddings`.
///
/// ILD is the mean pairwise dissimilarity (`1 - cosine`) over all unordered
/// pairs of the top-`k` embeddings. A list of fewer than two items has no pairs
/// and scores `0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn intra_list_diversity(embeddings: &[Vec<f32>], k: usize) -> f32 {
    let cut = k.min(embeddings.len());
    if cut < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut pairs = 0u32;
    for (a_pos, a) in embeddings[..cut].iter().enumerate() {
        for b in &embeddings[a_pos + 1..cut] {
            total += 1.0 - cosine(a, b);
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        total / pairs as f32
    }
}

// ── Subtopic Recall (S-recall@k) ──────────────────────────────────────────────

/// Subtopic recall at cutoff `k`.
///
/// Returns the fraction of `total_subtopics` distinct subtopics that appear in
/// the union of the first `k` results' subtopic labels. Returns `0.0` when
/// `total_subtopics` is `0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn subtopic_recall(result_subtopics: &[Vec<usize>], total_subtopics: usize, k: usize) -> f32 {
    if total_subtopics == 0 {
        return 0.0;
    }
    let cut = k.min(result_subtopics.len());
    let mut covered: BTreeSet<usize> = BTreeSet::new();
    for subtopics in &result_subtopics[..cut] {
        for &s in subtopics {
            if s < total_subtopics {
                covered.insert(s);
            }
        }
    }
    covered.len() as f32 / total_subtopics as f32
}

// ── α-nDCG@k ──────────────────────────────────────────────────────────────────

/// Novelty-discounted cumulative gain over an ordering of subtopic-label lists.
///
/// At each rank, a result earns `(1 - alpha)^c` for every subtopic it covers,
/// where `c` is the number of times that subtopic has already been seen at
/// strictly higher ranks. The per-subtopic seen counts are then updated. The
/// result is summed over the first `k` positions with the standard
/// `1 / log2(rank + 2)` positional discount.
#[allow(clippy::cast_precision_loss)]
fn novelty_dcg(order: &[&Vec<usize>], alpha: f32, k: usize, total_subtopics: usize) -> f32 {
    let mut seen = vec![0u32; total_subtopics];
    let mut dcg = 0.0f32;
    for (rank, subtopics) in order.iter().take(k).enumerate() {
        let mut gain = 0.0f32;
        for &s in *subtopics {
            if s < total_subtopics {
                gain += (1.0 - alpha).powi(seen[s].cast_signed());
            }
        }
        let discount = ((rank as f32) + 2.0).log2();
        dcg += gain / discount;
        for &s in *subtopics {
            if s < total_subtopics {
                seen[s] += 1;
            }
        }
    }
    dcg
}

/// Greedily order `result_subtopics` to maximise novelty-discounted gain.
///
/// At each step the remaining result whose immediate novelty gain (each covered
/// subtopic weighted by `(1 - alpha)^seen`) is largest is appended; ties are
/// broken by the original index so the ordering stays deterministic. This is the
/// standard ideal ordering used to normalise α-nDCG.
#[allow(clippy::cast_precision_loss)]
fn ideal_order(
    result_subtopics: &[Vec<usize>],
    alpha: f32,
    total_subtopics: usize,
) -> Vec<&Vec<usize>> {
    let n = result_subtopics.len();
    let mut taken = vec![false; n];
    let mut seen = vec![0u32; total_subtopics];
    let mut order: Vec<&Vec<usize>> = Vec::with_capacity(n);

    for _ in 0..n {
        let mut best_idx: Option<usize> = None;
        let mut best_gain = f32::NEG_INFINITY;
        for (i, subtopics) in result_subtopics.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let mut gain = 0.0f32;
            for &s in subtopics {
                if s < total_subtopics {
                    gain += (1.0 - alpha).powi(seen[s].cast_signed());
                }
            }
            // Strict `>` keeps the lowest index on ties (increasing-index scan).
            if best_idx.is_none() || gain > best_gain {
                best_gain = gain;
                best_idx = Some(i);
            }
        }
        let Some(idx) = best_idx else {
            break;
        };
        for &s in &result_subtopics[idx] {
            if s < total_subtopics {
                seen[s] += 1;
            }
        }
        taken[idx] = true;
        order.push(&result_subtopics[idx]);
    }
    order
}

/// α-nDCG at cutoff `k`.
///
/// Computes the novelty-discounted DCG of `result_subtopics` in its given order,
/// then normalises by the ideal greedy ordering's novelty-discounted DCG. The
/// number of distinct subtopics is inferred as one plus the maximum label seen.
/// Returns `0.0` when there are no subtopics or the ideal DCG is `0.0`, and the
/// result is clamped to `[0.0, 1.0]`.
#[must_use]
pub fn alpha_ndcg(result_subtopics: &[Vec<usize>], alpha: f32, k: usize) -> f32 {
    let total_subtopics = result_subtopics
        .iter()
        .flat_map(|s| s.iter().copied())
        .max()
        .map_or(0, |m| m + 1);
    if total_subtopics == 0 {
        return 0.0;
    }

    let actual_order: Vec<&Vec<usize>> = result_subtopics.iter().collect();
    let actual = novelty_dcg(&actual_order, alpha, k, total_subtopics);
    if actual == 0.0 {
        return 0.0;
    }

    let ideal = ideal_order(result_subtopics, alpha, total_subtopics);
    let ideal_dcg = novelty_dcg(&ideal, alpha, k, total_subtopics);
    if ideal_dcg == 0.0 {
        0.0
    } else {
        (actual / ideal_dcg).clamp(0.0, 1.0)
    }
}
