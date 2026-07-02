//! [`CrudRagHarness`] — dispatches each [`CrudCase`] to its operation's
//! lexical metric(s) and aggregates the results into a [`CrudRagReport`].
//!
//! Every metric here is deterministic, dependency-free lexical computation
//! (tokenization, longest-common-subsequence, n-gram counting, bag-of-words
//! and set overlap) — no LLM, randomness, or ML model is involved.

use std::collections::{HashMap, HashSet};

use super::types::{
    CrudCase, CrudMetric, CrudOperation, CrudRagConfig, CrudRagError, CrudRagReport, CrudRagResult,
    CrudScore,
};

// ── tokenization ─────────────────────────────────────────────────────────────

/// Tokenize `text` into whitespace-separated tokens, applying
/// [`CrudRagConfig::lowercase`] and [`CrudRagConfig::strip_punctuation`]
/// normalization. When punctuation stripping is enabled, every
/// non-alphanumeric character (punctuation *and* existing whitespace) is
/// replaced with a single space before splitting, so adjoining words never
/// fuse together.
fn tokenize(text: &str, cfg: &CrudRagConfig) -> Vec<String> {
    let cased = if cfg.lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let normalized = if cfg.strip_punctuation {
        cased
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect::<String>()
    } else {
        cased
    };
    normalized.split_whitespace().map(str::to_string).collect()
}

/// Minimum token length (in characters) for a token to count as a *content*
/// token in [`content_token_set`] — long enough to exclude common short
/// function words ("is", "the", "of", "a") while keeping ordinary content
/// words and four-digit numbers/years.
const CONTENT_TOKEN_MIN_LEN: usize = 4;

/// The distinct content-token vocabulary of `text`: normalized tokens (see
/// [`tokenize`]) of at least [`CONTENT_TOKEN_MIN_LEN`] characters. Used by
/// [`CrudOperation::Delete`]'s coverage and redundancy computations.
fn content_token_set(text: &str, cfg: &CrudRagConfig) -> HashSet<String> {
    tokenize(text, cfg)
        .into_iter()
        .filter(|t| t.chars().count() >= CONTENT_TOKEN_MIN_LEN)
        .collect()
}

/// `true` when the normalized `needle` appears as a contiguous token
/// sequence inside normalized `haystack`. A `needle` that normalizes to
/// nothing (blank, or only stripped punctuation) never matches.
fn normalized_contains(haystack: &str, needle: &str, cfg: &CrudRagConfig) -> bool {
    let needle_norm = tokenize(needle, cfg).join(" ");
    if needle_norm.is_empty() {
        return false;
    }
    let haystack_norm = tokenize(haystack, cfg).join(" ");
    haystack_norm.contains(&needle_norm)
}

// ── Create: ROUGE-L (LCS-based F-beta) ───────────────────────────────────────

/// Length of the longest common subsequence between `candidate` and
/// `reference`, via the standard `O(|candidate| * |reference|)`
/// dynamic-programming table.
fn lcs_len(candidate: &[String], reference: &[String]) -> usize {
    let candidate_len = candidate.len();
    let reference_len = reference.len();
    if candidate_len == 0 || reference_len == 0 {
        return 0;
    }
    let mut dp = vec![vec![0usize; reference_len + 1]; candidate_len + 1];
    for row in 1..=candidate_len {
        for col in 1..=reference_len {
            dp[row][col] = if candidate[row - 1] == reference[col - 1] {
                dp[row - 1][col - 1] + 1
            } else {
                dp[row - 1][col].max(dp[row][col - 1])
            };
        }
    }
    dp[candidate_len][reference_len]
}

/// ROUGE-L: the F-beta combination of LCS-based recall and precision between
/// `candidate` and `reference`.
///
/// `recall = lcs / reference.len()`, `precision = lcs / candidate.len()`, and
/// the result is `((1 + beta^2) * recall * precision) / (recall + beta^2 *
/// precision)`, falling back to `0.0` when the denominator is not positive
/// (including whenever either sequence is empty).
#[allow(clippy::cast_precision_loss)]
fn rouge_l_f(candidate: &[String], reference: &[String], beta: f32) -> f32 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let lcs = lcs_len(candidate, reference) as f32;
    let recall = lcs / reference.len() as f32;
    let precision = lcs / candidate.len() as f32;
    let beta_sq = beta * beta;
    let denom = recall + beta_sq * precision;
    if denom <= 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * recall * precision / denom
    }
}

// ── Create: BLEU (clipped n-gram precision + brevity penalty) ───────────────

/// Count `n`-gram occurrences in `tokens` (an empty map when `n == 0` or
/// `tokens.len() < n`).
fn ngram_counts(tokens: &[String], n: usize) -> HashMap<Vec<&str>, usize> {
    let mut counts: HashMap<Vec<&str>, usize> = HashMap::new();
    if n == 0 || tokens.len() < n {
        return counts;
    }
    for window in tokens.windows(n) {
        let key: Vec<&str> = window.iter().map(String::as_str).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// The BLEU modified (clipped) `n`-gram precision of `candidate` against
/// `reference`: for every distinct `n`-gram in `candidate`, its count is
/// clipped to at most its count in `reference` before summing, then divided
/// by the total number of `n`-grams in `candidate`. `None` when `candidate`
/// has no `n`-grams of this order (`candidate.len() < n`).
fn modified_precision(candidate: &[String], reference: &[String], n: usize) -> Option<f64> {
    let candidate_counts = ngram_counts(candidate, n);
    if candidate_counts.is_empty() {
        return None;
    }
    let reference_counts = ngram_counts(reference, n);
    let mut clipped = 0usize;
    let mut total = 0usize;
    for (gram, count) in &candidate_counts {
        total += count;
        let reference_count = reference_counts.get(gram).copied().unwrap_or(0);
        clipped += (*count).min(reference_count);
    }
    if total == 0 {
        Some(0.0)
    } else {
        #[allow(clippy::cast_precision_loss)]
        let ratio = clipped as f64 / total as f64;
        Some(ratio)
    }
}

/// The BLEU brevity penalty: `1.0` when `candidate_len >= reference_len`,
/// otherwise `exp(1 - reference_len / candidate_len)`. Returns `0.0` when
/// `candidate_len == 0` (an empty hypothesis is maximally penalized).
#[allow(clippy::cast_precision_loss)]
fn brevity_penalty(candidate_len: usize, reference_len: usize) -> f64 {
    if candidate_len == 0 {
        return 0.0;
    }
    if candidate_len >= reference_len {
        1.0
    } else {
        (1.0 - reference_len as f64 / candidate_len as f64).exp()
    }
}

/// BLEU-style score: the geometric mean of modified `n`-gram precisions for
/// orders `1..=effective_order`, scaled by the brevity penalty.
///
/// `effective_order = max_order.min(candidate.len()).max(1)`: capping the
/// order at the candidate's own token count means a short candidate is
/// scored over the orders it can actually produce, rather than collapsing to
/// `0.0` simply because it is shorter than `max_order` tokens. As in
/// classic BLEU, a `0.0` modified precision at any evaluated order collapses
/// the whole geometric mean to `0.0`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn bleu_score(candidate: &[String], reference: &[String], max_order: usize) -> f32 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let effective_order = max_order.min(candidate.len()).max(1);
    let mut log_sum = 0.0f64;
    let mut orders_used = 0usize;
    for n in 1..=effective_order {
        let Some(precision) = modified_precision(candidate, reference, n) else {
            continue;
        };
        if precision <= 0.0 {
            return 0.0;
        }
        log_sum += precision.ln();
        orders_used += 1;
    }
    if orders_used == 0 {
        return 0.0;
    }
    let geometric_mean = (log_sum / orders_used as f64).exp();
    let bp = brevity_penalty(candidate.len(), reference.len());
    (bp * geometric_mean) as f32
}

// ── Read: exact match + token F1 ─────────────────────────────────────────────

/// Bag-of-words overlap: `sum(min(count_a[t], count_b[t]))` over every
/// distinct token `t`, respecting multiplicities (a token repeated twice in
/// both `a` and `b` contributes `2`, not `1`).
fn overlap_count(candidate: &[String], reference: &[String]) -> usize {
    let mut available: HashMap<&str, usize> = HashMap::new();
    for token in candidate {
        *available.entry(token.as_str()).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for token in reference {
        if let Some(remaining) = available.get_mut(token.as_str())
            && *remaining > 0
        {
            *remaining -= 1;
            overlap += 1;
        }
    }
    overlap
}

/// Token-level F1: the harmonic mean of precision (`overlap /
/// candidate.len()`) and recall (`overlap / reference.len()`), where
/// `overlap` is the bag-of-words intersection size. Returns `0.0` when
/// either token slice is empty, or when precision and recall are both zero.
#[allow(clippy::cast_precision_loss)]
fn token_f1(candidate: &[String], reference: &[String]) -> f32 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    let overlap = overlap_count(candidate, reference) as f32;
    let precision = overlap / candidate.len() as f32;
    let recall = overlap / reference.len() as f32;
    if precision + recall <= 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

// ── Delete: coverage + redundancy ────────────────────────────────────────────

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`. When
/// no sentence-terminating punctuation is present, the whole trimmed text is
/// returned as a single sentence (blank input yields no sentences).
fn split_sentences(text: &str) -> Vec<&str> {
    let parts: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            Vec::new()
        } else {
            vec![trimmed]
        }
    } else {
        parts
    }
}

/// Jaccard similarity `|set_a ∩ set_b| / |set_a ∪ set_b|` between two token
/// sets. Returns `0.0` when both sets are empty (rather than dividing `0` by
/// `0`).
#[allow(clippy::cast_precision_loss)]
fn jaccard(set_a: &HashSet<String>, set_b: &HashSet<String>) -> f32 {
    let union = set_a.union(set_b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = set_a.intersection(set_b).count();
    intersection as f32 / union as f32
}

/// A repeated-content ratio for `summary`: the mean pairwise Jaccard
/// similarity (over [`content_token_set`]s) across every unordered pair of
/// its sentences. `0.0` when `summary` has fewer than two sentences (nothing
/// to be redundant against).
#[allow(clippy::cast_precision_loss)]
fn summary_redundancy(summary: &str, cfg: &CrudRagConfig) -> f32 {
    let sentences = split_sentences(summary);
    if sentences.len() < 2 {
        return 0.0;
    }
    let sets: Vec<HashSet<String>> = sentences
        .iter()
        .map(|s| content_token_set(s, cfg))
        .collect();

    let mut sum = 0.0f32;
    let mut pairs = 0usize;
    for (i, set_a) in sets.iter().enumerate() {
        for set_b in &sets[i + 1..] {
            sum += jaccard(set_a, set_b);
            pairs += 1;
        }
    }
    if pairs == 0 { 0.0 } else { sum / pairs as f32 }
}

// ── best-of-reference selection ──────────────────────────────────────────────

/// The index of the first element of `items` achieving the maximum score
/// under `score`, breaking ties toward the earliest element. `None` only
/// when `items` is empty. Used to pick the "best matching reference" out of
/// a [`CrudCase`]'s possibly-multiple `references`.
fn best_by_key<T, F>(items: &[T], mut score: F) -> Option<usize>
where
    F: FnMut(&T) -> f32,
{
    let mut best: Option<(usize, f32)> = None;
    for (idx, item) in items.iter().enumerate() {
        let value = score(item);
        let is_better = match best {
            None => true,
            Some((_, best_value)) => value > best_value,
        };
        if is_better {
            best = Some((idx, value));
        }
    }
    best.map(|(idx, _)| idx)
}

// ── CrudRagHarness ────────────────────────────────────────────────────────────

/// Evaluates [`CrudCase`]s against the CRUD-operation taxonomy: dispatches
/// each case to its operation's lexical metric(s) and aggregates the results
/// into a [`CrudRagReport`].
///
/// See the [module docs](crate::crud_rag) for the per-operation metric
/// mapping. The harness is stateless beyond its [`CrudRagConfig`] and fully
/// deterministic.
#[derive(Debug, Clone, Default)]
pub struct CrudRagHarness {
    /// Configuration controlling normalization, n-gram order, and which
    /// operations are enabled.
    pub config: CrudRagConfig,
}

impl CrudRagHarness {
    /// Create a new harness with the given config.
    #[must_use]
    pub fn new(config: CrudRagConfig) -> Self {
        Self { config }
    }

    /// Score a single `case`, returning every [`CrudScore`] its operation
    /// produces (including [`CrudMetric::Combined`]).
    ///
    /// # Errors
    ///
    /// - [`CrudRagError::OperationDisabled`] when `case.operation` is not in
    ///   [`CrudRagConfig::enabled_operations`].
    /// - [`CrudRagError::EmptyOutput`] when `case.output` is blank.
    /// - [`CrudRagError::MissingReference`] when
    ///   [`CrudOperation::Create`]/[`CrudOperation::Read`]/[`CrudOperation::Update`]
    ///   is given no `references`.
    /// - [`CrudRagError::MissingKeyPoints`] when [`CrudOperation::Delete`]
    ///   is given no `key_points`.
    /// - [`CrudRagError::MissingErrorSpan`] / [`CrudRagError::MissingCorrectSpan`]
    ///   when [`CrudOperation::Update`] is missing either span.
    pub fn evaluate_case(&self, case: &CrudCase) -> CrudRagResult<Vec<CrudScore>> {
        if !self.config.is_enabled(case.operation) {
            return Err(CrudRagError::OperationDisabled {
                case_id: case.id.clone(),
                operation: case.operation,
            });
        }
        if case.output.trim().is_empty() {
            return Err(CrudRagError::EmptyOutput {
                case_id: case.id.clone(),
            });
        }
        match case.operation {
            CrudOperation::Create => self.score_create(case),
            CrudOperation::Read => self.score_read(case),
            CrudOperation::Update => self.score_update(case),
            CrudOperation::Delete => self.score_delete(case),
        }
    }

    /// Score every case in `cases` and aggregate the results into a
    /// [`CrudRagReport`].
    ///
    /// # Errors
    ///
    /// [`CrudRagError::EmptyCases`] when `cases` is empty, or any error
    /// [`CrudRagHarness::evaluate_case`] can return for the first invalid
    /// case encountered (cases are processed in order; evaluation stops at
    /// the first failure).
    pub fn evaluate(&self, cases: &[CrudCase]) -> CrudRagResult<CrudRagReport> {
        if cases.is_empty() {
            return Err(CrudRagError::EmptyCases);
        }
        let mut scores = Vec::new();
        for case in cases {
            scores.extend(self.evaluate_case(case)?);
        }
        Ok(aggregate_report(scores))
    }

    /// [`CrudOperation::Create`]: ROUGE-L + BLEU against the best-matching
    /// reference.
    fn score_create(&self, case: &CrudCase) -> CrudRagResult<Vec<CrudScore>> {
        if case.references.is_empty() {
            return Err(CrudRagError::MissingReference {
                case_id: case.id.clone(),
                operation: CrudOperation::Create,
            });
        }
        let output_tokens = tokenize(&case.output, &self.config);
        let per_reference: Vec<(f32, f32, f32)> = case
            .references
            .iter()
            .map(|reference| {
                let reference_tokens = tokenize(reference, &self.config);
                let rouge = rouge_l_f(&output_tokens, &reference_tokens, self.config.rouge_beta);
                let bleu = bleu_score(
                    &output_tokens,
                    &reference_tokens,
                    self.config.bleu_ngram_order,
                );
                (rouge, bleu, f32::midpoint(rouge, bleu))
            })
            .collect();
        let best_idx = best_by_key(&per_reference, |&(_, _, combined)| combined).unwrap_or(0);
        let (rouge, bleu, combined) = per_reference[best_idx];

        let case_id = case.id.clone();
        Ok(vec![
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Create,
                metric: CrudMetric::RougeL,
                value: rouge,
            },
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Create,
                metric: CrudMetric::Bleu,
                value: bleu,
            },
            CrudScore {
                case_id,
                operation: CrudOperation::Create,
                metric: CrudMetric::Combined,
                value: combined,
            },
        ])
    }

    /// [`CrudOperation::Read`]: exact match + token F1 against the
    /// best-matching reference.
    fn score_read(&self, case: &CrudCase) -> CrudRagResult<Vec<CrudScore>> {
        if case.references.is_empty() {
            return Err(CrudRagError::MissingReference {
                case_id: case.id.clone(),
                operation: CrudOperation::Read,
            });
        }
        let output_tokens = tokenize(&case.output, &self.config);
        let per_reference: Vec<(f32, f32, f32)> = case
            .references
            .iter()
            .map(|reference| {
                let reference_tokens = tokenize(reference, &self.config);
                let exact = if output_tokens == reference_tokens {
                    1.0
                } else {
                    0.0
                };
                let f1 = token_f1(&output_tokens, &reference_tokens);
                (exact, f1, 0.5 * exact + 0.5 * f1)
            })
            .collect();
        let best_idx = best_by_key(&per_reference, |&(_, _, combined)| combined).unwrap_or(0);
        let (exact, f1, combined) = per_reference[best_idx];

        let case_id = case.id.clone();
        Ok(vec![
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Read,
                metric: CrudMetric::ExactMatch,
                value: exact,
            },
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Read,
                metric: CrudMetric::TokenF1,
                value: f1,
            },
            CrudScore {
                case_id,
                operation: CrudOperation::Read,
                metric: CrudMetric::Combined,
                value: combined,
            },
        ])
    }

    /// [`CrudOperation::Update`]: correction similarity + error-removal +
    /// correct-span-introduced detection.
    fn score_update(&self, case: &CrudCase) -> CrudRagResult<Vec<CrudScore>> {
        if case.references.is_empty() {
            return Err(CrudRagError::MissingReference {
                case_id: case.id.clone(),
                operation: CrudOperation::Update,
            });
        }
        let Some(error_span) = case.error_span.as_deref() else {
            return Err(CrudRagError::MissingErrorSpan {
                case_id: case.id.clone(),
            });
        };
        let Some(correct_span) = case.correct_span.as_deref() else {
            return Err(CrudRagError::MissingCorrectSpan {
                case_id: case.id.clone(),
            });
        };

        let output_tokens = tokenize(&case.output, &self.config);
        let similarities: Vec<f32> = case
            .references
            .iter()
            .map(|reference| {
                let reference_tokens = tokenize(reference, &self.config);
                token_f1(&output_tokens, &reference_tokens)
            })
            .collect();
        let best_idx = best_by_key(&similarities, |&s| s).unwrap_or(0);
        let similarity = similarities[best_idx];

        let error_removed = !normalized_contains(&case.output, error_span, &self.config);
        let correct_introduced = normalized_contains(&case.output, correct_span, &self.config);
        let error_removed_val = if error_removed { 1.0 } else { 0.0 };
        let correct_introduced_val = if correct_introduced { 1.0 } else { 0.0 };
        let combined = 0.5 * similarity + 0.25 * error_removed_val + 0.25 * correct_introduced_val;

        let case_id = case.id.clone();
        Ok(vec![
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Update,
                metric: CrudMetric::CorrectionSimilarity,
                value: similarity,
            },
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Update,
                metric: CrudMetric::ErrorRemoval,
                value: error_removed_val,
            },
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Update,
                metric: CrudMetric::CorrectSpanIntroduced,
                value: correct_introduced_val,
            },
            CrudScore {
                case_id,
                operation: CrudOperation::Update,
                metric: CrudMetric::Combined,
                value: combined,
            },
        ])
    }

    /// [`CrudOperation::Delete`]: key-point coverage minus a weighted
    /// redundancy penalty.
    fn score_delete(&self, case: &CrudCase) -> CrudRagResult<Vec<CrudScore>> {
        if case.key_points.is_empty() {
            return Err(CrudRagError::MissingKeyPoints {
                case_id: case.id.clone(),
            });
        }
        let summary_content = content_token_set(&case.output, &self.config);
        let covered = case
            .key_points
            .iter()
            .filter(|kp| {
                let kp_content = content_token_set(kp, &self.config);
                !kp_content.is_empty() && kp_content.is_subset(&summary_content)
            })
            .count();
        #[allow(clippy::cast_precision_loss)]
        let coverage = covered as f32 / case.key_points.len() as f32;

        let redundancy = summary_redundancy(&case.output, &self.config);
        let combined =
            (coverage - self.config.redundancy_penalty_weight * redundancy).clamp(0.0, 1.0);

        let case_id = case.id.clone();
        Ok(vec![
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Delete,
                metric: CrudMetric::Coverage,
                value: coverage,
            },
            CrudScore {
                case_id: case_id.clone(),
                operation: CrudOperation::Delete,
                metric: CrudMetric::Redundancy,
                value: redundancy,
            },
            CrudScore {
                case_id,
                operation: CrudOperation::Delete,
                metric: CrudMetric::Combined,
                value: combined,
            },
        ])
    }
}

// ── aggregation ───────────────────────────────────────────────────────────────

/// `sum / n`, or `0.0` when `n == 0`.
#[allow(clippy::cast_precision_loss)]
fn mean(sum: f32, n: usize) -> f32 {
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Aggregate every case's [`CrudScore`]s into a [`CrudRagReport`]: a
/// per-operation mean of [`CrudMetric::Combined`] values, and an overall
/// mean over only the operations that were actually present.
#[allow(clippy::cast_precision_loss)]
fn aggregate_report(scores: Vec<CrudScore>) -> CrudRagReport {
    let mut create_sum = 0.0f32;
    let mut create_n = 0usize;
    let mut read_sum = 0.0f32;
    let mut read_n = 0usize;
    let mut update_sum = 0.0f32;
    let mut update_n = 0usize;
    let mut delete_sum = 0.0f32;
    let mut delete_n = 0usize;

    for score in &scores {
        if score.metric != CrudMetric::Combined {
            continue;
        }
        match score.operation {
            CrudOperation::Create => {
                create_sum += score.value;
                create_n += 1;
            }
            CrudOperation::Read => {
                read_sum += score.value;
                read_n += 1;
            }
            CrudOperation::Update => {
                update_sum += score.value;
                update_n += 1;
            }
            CrudOperation::Delete => {
                delete_sum += score.value;
                delete_n += 1;
            }
        }
    }

    let create_mean = mean(create_sum, create_n);
    let read_mean = mean(read_sum, read_n);
    let update_mean = mean(update_sum, update_n);
    let delete_mean = mean(delete_sum, delete_n);

    let present: Vec<f32> = [
        (create_n, create_mean),
        (read_n, read_mean),
        (update_n, update_mean),
        (delete_n, delete_mean),
    ]
    .into_iter()
    .filter_map(|(n, m)| (n > 0).then_some(m))
    .collect();
    let overall = if present.is_empty() {
        0.0
    } else {
        present.iter().sum::<f32>() / present.len() as f32
    };

    CrudRagReport {
        scores,
        create_mean,
        read_mean,
        update_mean,
        delete_mean,
        overall,
    }
}
