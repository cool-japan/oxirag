//! Core consistency-checking logic: sentence splitting, number/year/negation extraction.

use super::types::{
    ConflictType, ConsistencyConfig, ConsistencyError, ConsistencyReport, Inconsistency,
};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Split `text` into sentences on `". "`, `"! "`, `"? "`, and `".\n"`.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];
        current.push(c);

        if matches!(c, '.' | '!' | '?') {
            let at_newline = c == '.' && i + 1 < n && chars[i + 1] == '\n';
            let at_space = i + 1 < n && chars[i + 1] == ' ';
            if at_newline || at_space {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                i += 2; // skip the '\n' or ' '
                continue;
            }
        }
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Extract all numeric values from `text` (handles `$`, `%`, `,` prefixes/suffixes).
fn extract_numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| {
            // Strip leading/trailing punctuation that is not part of the number
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Extract 4-digit year tokens in the range `1000..=2100`.
fn extract_years(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|token| {
            // Strip non-digit characters at boundaries
            let cleaned: String = token.chars().filter(char::is_ascii_digit).collect();
            if cleaned.len() == 4 {
                cleaned
                    .parse::<u32>()
                    .ok()
                    .filter(|&y| (1000..=2100).contains(&y))
            } else {
                None
            }
        })
        .collect()
}

/// Return `true` when `text` contains an explicit negation marker.
fn contains_negation(text: &str) -> bool {
    let lower = text.to_lowercase();
    let negation_patterns: &[&str] = &[
        "not ", "never ", "no ", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't",
        "won't", "can't", "couldn't",
    ];
    negation_patterns.iter().any(|p| lower.contains(p))
}

/// Stop-word list for shared-noun filtering.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had", "and", "or", "but", "it",
    "in", "of", "to", "for", "at", "by", "be", "do", "so", "if", "as",
];

/// Tokenize both strings, remove stop-words, and return the intersection.
fn shared_noun_tokens(a: &str, b: &str) -> Vec<String> {
    let tokenize_filtered = |text: &str| -> std::collections::HashSet<String> {
        text.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|t| !t.is_empty() && t.len() > 3 && !STOP_WORDS.contains(&t.as_str()))
            .collect()
    };

    let a_set = tokenize_filtered(a);
    let b_set = tokenize_filtered(b);
    a_set.intersection(&b_set).cloned().collect()
}

// ── ConsistencyChecker ────────────────────────────────────────────────────────

/// Heuristic cross-claim consistency detector.
///
/// Checks pairwise sentence relationships for negation, numerical, and
/// temporal conflicts using pure-Rust lexical analysis.
#[derive(Debug, Clone)]
pub struct ConsistencyChecker {
    /// Configuration controlling confidence thresholds.
    pub config: ConsistencyConfig,
}

impl ConsistencyChecker {
    /// Construct a checker with the given configuration.
    #[must_use]
    pub fn new(config: ConsistencyConfig) -> Self {
        Self { config }
    }

    /// Check `text` for internal inconsistencies.
    ///
    /// # Errors
    ///
    /// Returns [`ConsistencyError::EmptyInput`] when `text` is blank.
    /// Returns [`ConsistencyError::TooFewSentences`] when fewer than two sentences are found.
    pub fn check(&self, text: &str) -> Result<ConsistencyReport, ConsistencyError> {
        if text.trim().is_empty() {
            return Err(ConsistencyError::EmptyInput);
        }

        let sentences = split_sentences(text);
        if sentences.len() < 2 {
            return Err(ConsistencyError::TooFewSentences);
        }

        let min_confidence = self.config.min_confidence;
        let mut inconsistencies: Vec<Inconsistency> = Vec::new();

        let n = sentences.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let si = &sentences[i];
                let sj = &sentences[j];

                let shared = shared_noun_tokens(si, sj);

                // ── Negation conflict ──────────────────────────────────────
                // Shared ≥2 nouns AND exactly one sentence has negation
                if shared.len() >= 2 {
                    let neg_i = contains_negation(si);
                    let neg_j = contains_negation(sj);
                    if neg_i ^ neg_j {
                        inconsistencies.push(Inconsistency {
                            claim_a: si.clone(),
                            claim_b: sj.clone(),
                            conflict_type: ConflictType::Negation,
                            confidence: 0.7,
                        });
                        continue; // one conflict per pair is sufficient
                    }
                }

                // ── Temporal conflict ──────────────────────────────────────
                // Both have years AND shared ≥1 noun AND years differ
                if !shared.is_empty() {
                    let years_i = extract_years(si);
                    let years_j = extract_years(sj);
                    if !years_i.is_empty() && !years_j.is_empty() {
                        let years_differ = years_i.iter().any(|yi| !years_j.contains(yi));
                        if years_differ {
                            inconsistencies.push(Inconsistency {
                                claim_a: si.clone(),
                                claim_b: sj.clone(),
                                conflict_type: ConflictType::Temporal,
                                confidence: 0.8,
                            });
                            continue;
                        }
                    }
                }

                // ── Numerical conflict ────────────────────────────────────
                // Both have numbers AND shared ≥1 noun AND numbers differ
                if !shared.is_empty() {
                    let nums_i = extract_numbers(si);
                    let nums_j = extract_numbers(sj);
                    if !nums_i.is_empty() && !nums_j.is_empty() {
                        // Find the maximum absolute difference across all pairs
                        let max_diff = nums_i
                            .iter()
                            .flat_map(|a| nums_j.iter().map(move |b| (a - b).abs()))
                            .fold(0.0_f64, f64::max);
                        // Any difference means a conflict; confidence scales with magnitude
                        if max_diff > 0.0 {
                            #[allow(clippy::cast_precision_loss)]
                            let confidence = if max_diff > 10.0 { 0.6_f32 } else { 0.4_f32 };
                            inconsistencies.push(Inconsistency {
                                claim_a: si.clone(),
                                claim_b: sj.clone(),
                                conflict_type: ConflictType::Numerical,
                                confidence,
                            });
                        }
                    }
                }
            }
        }

        // Filter by minimum confidence
        let filtered: Vec<Inconsistency> = inconsistencies
            .into_iter()
            .filter(|inc| inc.confidence >= min_confidence)
            .collect();

        Ok(ConsistencyReport::new(filtered))
    }
}

impl Default for ConsistencyChecker {
    fn default() -> Self {
        Self::new(ConsistencyConfig::default())
    }
}
