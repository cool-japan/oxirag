//! IR ranking metric functions.
use crate::retrieval_eval::types::Qrels;
use crate::types::SearchResult;

// ── Precision@k ───────────────────────────────────────────────────────────────

/// Precision at cutoff k.
///
/// Returns 0.0 when `k` is 0.
#[must_use]
pub fn precision_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, threshold: f32) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let top = results.iter().take(k);
    let rel = top
        .filter(|r| qrels.is_relevant(r.document.id.as_str(), threshold))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        rel as f32 / k as f32
    }
}

// ── Recall@k ─────────────────────────────────────────────────────────────────

/// Recall at cutoff k.
///
/// Returns 0.0 when there are no relevant documents in qrels.
#[must_use]
pub fn recall_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, threshold: f32) -> f32 {
    let total_rel = qrels
        .judgments
        .values()
        .filter(|&&g| g >= threshold)
        .count();
    if total_rel == 0 {
        return 0.0;
    }
    let found = results
        .iter()
        .take(k)
        .filter(|r| qrels.is_relevant(r.document.id.as_str(), threshold))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        found as f32 / total_rel as f32
    }
}

// ── F1@k ──────────────────────────────────────────────────────────────────────

/// F1 at cutoff k.
#[must_use]
pub fn f1_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, threshold: f32) -> f32 {
    let p = precision_at_k(results, qrels, k, threshold);
    let r = recall_at_k(results, qrels, k, threshold);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

// ── Hit Rate@k ────────────────────────────────────────────────────────────────

/// Hit rate at cutoff k: 1.0 if at least one relevant result in top-k, else 0.0.
#[must_use]
pub fn hit_rate_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, threshold: f32) -> f32 {
    if results
        .iter()
        .take(k)
        .any(|r| qrels.is_relevant(r.document.id.as_str(), threshold))
    {
        1.0
    } else {
        0.0
    }
}

// ── Reciprocal Rank ───────────────────────────────────────────────────────────

/// Reciprocal rank of the first relevant result.
///
/// Returns 0.0 if no relevant result is found.
#[must_use]
pub fn reciprocal_rank(results: &[SearchResult], qrels: &Qrels, threshold: f32) -> f32 {
    for (i, r) in results.iter().enumerate() {
        if qrels.is_relevant(r.document.id.as_str(), threshold) {
            #[allow(clippy::cast_precision_loss)]
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

// ── MRR ───────────────────────────────────────────────────────────────────────

/// Mean Reciprocal Rank over a list of (results, qrels) pairs.
///
/// Returns 0.0 when `query_results` is empty.
#[must_use]
pub fn mrr(query_results: &[(&[SearchResult], &Qrels)], threshold: f32) -> f32 {
    if query_results.is_empty() {
        return 0.0;
    }
    let sum: f32 = query_results
        .iter()
        .map(|(r, q)| reciprocal_rank(r, q, threshold))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    {
        sum / query_results.len() as f32
    }
}

// ── Average Precision ─────────────────────────────────────────────────────────

/// Average precision (AP) for a single query.
///
/// Returns 0.0 when there are no relevant documents.
#[must_use]
pub fn average_precision(results: &[SearchResult], qrels: &Qrels, threshold: f32) -> f32 {
    let total_rel = qrels
        .judgments
        .values()
        .filter(|&&g| g >= threshold)
        .count();
    if total_rel == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    let mut found = 0usize;
    for (i, r) in results.iter().enumerate() {
        if qrels.is_relevant(r.document.id.as_str(), threshold) {
            found += 1;
            #[allow(clippy::cast_precision_loss)]
            {
                sum += found as f32 / (i + 1) as f32;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    {
        sum / total_rel as f32
    }
}

// ── DCG@k ─────────────────────────────────────────────────────────────────────

/// Discounted Cumulative Gain at cutoff k (graded).
#[must_use]
pub fn dcg_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, log_base: f32) -> f32 {
    if k == 0 || log_base <= 1.0 {
        return 0.0;
    }
    let mut dcg = 0.0_f32;
    for (i, r) in results.iter().take(k).enumerate() {
        let gain = qrels.gain(r.document.id.as_str());
        if gain > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            {
                dcg += (2.0_f32.powf(gain) - 1.0) / (i as f32 + 2.0).log(log_base);
            }
        }
    }
    dcg
}

// ── nDCG@k ────────────────────────────────────────────────────────────────────

/// Normalised DCG at cutoff k.
///
/// Returns 0.0 when ideal DCG is 0.
#[must_use]
pub fn ndcg_at_k(results: &[SearchResult], qrels: &Qrels, k: usize, log_base: f32) -> f32 {
    let actual = dcg_at_k(results, qrels, k, log_base);
    if actual == 0.0 {
        return 0.0;
    }
    // Build ideal ranking: sort all relevant docs by gain descending
    let mut ideal_gains: Vec<f32> = qrels
        .judgments
        .values()
        .filter(|&&g| g > 0.0)
        .copied()
        .collect();
    ideal_gains.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Compute ideal DCG as if results were sorted by gain
    let ideal_results: Vec<SearchResult> = ideal_gains
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &g)| {
            let doc_id = format!("ideal_{i}");
            let doc = crate::types::Document::new("ideal")
                .with_id(crate::types::DocumentId::from_string(&doc_id));
            SearchResult::new(doc, g, i)
        })
        .collect();
    // Build a Qrels for the ideal set
    let ideal_qrels = crate::retrieval_eval::types::Qrels {
        judgments: ideal_gains
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &g)| (format!("ideal_{i}"), g))
            .collect(),
    };
    let ideal_dcg = dcg_at_k(&ideal_results, &ideal_qrels, k, log_base);
    if ideal_dcg == 0.0 {
        0.0
    } else {
        (actual / ideal_dcg).min(1.0)
    }
}
