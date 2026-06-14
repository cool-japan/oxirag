//! Chain-of-Density engine: iterative entity-dense summarization (Adams et al. 2023).

use std::collections::HashSet;

use super::types::{ChainOfDensityOutput, CodConfig, CodError, DensityStep};

// ── Tokeniser & sentence splitter ─────────────────────────────────────────────

/// Split `text` into trimmed, non-empty sentences on `.`, `!`, and `?`.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Split `text` into raw alphanumeric tokens (preserving original case),
/// discarding empty fragments produced by runs of separators.
pub(crate) fn raw_tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Returns `true` when `token` qualifies as a salient entity.
///
/// An entity is either a capitalized multi-character token (first character is
/// an uppercase letter and the token is at least two characters long) or a token
/// consisting entirely of digits (a number).
pub(crate) fn is_entity(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if first.is_numeric() {
        return token.chars().all(char::is_numeric);
    }
    first.is_uppercase() && token.chars().count() >= 2
}

/// Extract the ordered, de-duplicated list of salient source entities.
///
/// Entities are returned in first-appearance order; duplicates (compared
/// case-insensitively) are dropped so that each surface form appears once.
pub(crate) fn extract_entities(source: &str) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut entities: Vec<String> = Vec::new();
    for token in raw_tokens(source) {
        if is_entity(token) {
            let key = token.to_lowercase();
            if seen.insert(key) {
                entities.push(token.to_string());
            }
        }
    }
    entities
}

/// Count whitespace-delimited words in `text`.
pub(crate) fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Returns `true` when `entity` is present in `summary` (case-insensitive,
/// whole-token match against the summary's alphanumeric tokens).
fn entity_present(summary: &str, entity: &str) -> bool {
    let needle = entity.to_lowercase();
    raw_tokens(summary)
        .iter()
        .any(|t| t.to_lowercase() == needle)
}

/// Count the distinct source `entities` present in `summary`.
fn distinct_entities_present(summary: &str, entities: &[String]) -> usize {
    entities
        .iter()
        .filter(|e| entity_present(summary, e))
        .count()
}

// ── Filler vocabulary ─────────────────────────────────────────────────────────

/// Low-information filler words that may be trimmed to honour the word budget.
///
/// These carry little entity signal, so dropping them compresses the summary
/// without sacrificing salient content (the core Chain-of-Density move).
const FILLER_WORDS: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "of", "to", "in", "on", "at", "for", "and", "or", "but", "with", "as", "by", "from",
    "very", "really", "quite", "just", "also", "then", "thus", "which", "it", "its", "their",
    "there", "here", "some", "such", "into", "about", "over", "than", "so", "well", "much",
];

/// Returns `true` when `word` (ignoring trailing punctuation, case-insensitive)
/// is a trimmable filler word.
pub(crate) fn is_filler(word: &str) -> bool {
    let cleaned: String = word
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    !cleaned.is_empty() && FILLER_WORDS.contains(&cleaned.as_str())
}

/// Trim trailing filler words from `summary` until it fits within
/// `target_words`, never removing the final word (to keep a complete clause).
///
/// Scans from the end, removing the right-most filler word repeatedly. Stops as
/// soon as the budget is met or no further trailing filler can be removed.
pub(crate) fn trim_filler(summary: &str, target_words: usize) -> String {
    let mut words: Vec<String> = summary.split_whitespace().map(str::to_string).collect();
    while words.len() > target_words && words.len() > 1 {
        // Find the right-most filler word (excluding the very last token).
        let mut removed = false;
        for idx in (0..words.len() - 1).rev() {
            if is_filler(&words[idx]) {
                words.remove(idx);
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
    words.join(" ")
}

/// Build the initial *sparse* summary from the lead sentence(s).
///
/// Faithful to Chain-of-Density, the initial summary is intentionally sparse:
/// it seeds with the first sentence only, extending to a second lead sentence
/// solely when the lead is very short (under a third of the budget). This keeps
/// most source entities available for the subsequent densification passes even
/// when the whole source would fit the budget. The result is always trimmed to
/// stay within `target_words`.
pub(crate) fn initial_summary(sentences: &[String], target_words: usize) -> String {
    if sentences.is_empty() {
        return String::new();
    }

    let mut acc: Vec<String> = vec![sentences[0].clone()];
    let mut count = word_count(&sentences[0]);

    // Optionally absorb a second lead sentence when the first is very short,
    // keeping the initial summary recognisable without exhausting the source.
    if sentences.len() > 1 && target_words > 0 && count < target_words / 3 {
        let second = word_count(&sentences[1]);
        if count + second <= target_words {
            acc.push(sentences[1].clone());
            count += second;
        }
    }

    let joined = acc.join(" ");

    // Trim to the word budget if the lead alone overruns it.
    let bounded = if target_words > 0 && count > target_words {
        let trimmed: Vec<&str> = joined.split_whitespace().take(target_words).collect();
        trimmed.join(" ")
    } else {
        joined
    };

    ensure_period(&bounded)
}

/// Append a single trailing period to `text` when it lacks terminal
/// punctuation, leaving non-empty already-punctuated text unchanged.
fn ensure_period(text: &str) -> String {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
        trimmed.to_string()
    } else {
        format!("{trimmed}.")
    }
}

/// Compose a compact clause that mentions `entities`.
///
/// The clause is intentionally terse ("It covers X, Y.") so that incorporating
/// entities adds the least possible filler, keeping the summary entity-dense.
fn compose_clause(entities: &[String]) -> String {
    if entities.is_empty() {
        return String::new();
    }
    let joined = entities.join(", ");
    format!("Covers {joined}.")
}

// ── ChainOfDensityEngine ──────────────────────────────────────────────────────

/// Engine that produces increasingly entity-dense summaries via the
/// Chain-of-Density procedure (Adams et al. 2023).
///
/// Starting from a sparse lead-sentence summary, each iteration identifies
/// salient source entities missing from the current summary, folds in
/// [`entities_per_step`](CodConfig::entities_per_step) of them through a compact
/// clause, and trims trailing filler so the summary stays close to
/// [`target_words`](CodConfig::target_words). The output records one
/// [`DensityStep`] per iteration.
#[derive(Debug, Clone)]
pub struct ChainOfDensityEngine {
    config: CodConfig,
}

impl ChainOfDensityEngine {
    /// Create a new [`ChainOfDensityEngine`] with the given configuration.
    #[must_use]
    pub fn new(config: CodConfig) -> Self {
        Self { config }
    }

    /// Borrow the engine's configuration.
    #[must_use]
    pub fn config(&self) -> &CodConfig {
        &self.config
    }

    /// Produce a Chain-of-Density summary of `source`.
    ///
    /// Salient source entities are extracted in order, an initial sparse summary
    /// is built from the lead sentence(s), and then
    /// [`iterations`](CodConfig::iterations) densification passes fold in fresh
    /// entities while trimming filler to respect the word budget. Each pass is
    /// recorded as a [`DensityStep`]; the densest result is returned as
    /// [`ChainOfDensityOutput::final_summary`].
    ///
    /// # Errors
    ///
    /// Returns [`CodError::EmptySource`] when `source` is empty or whitespace.
    pub fn summarize(&self, source: &str) -> Result<ChainOfDensityOutput, CodError> {
        if source.trim().is_empty() {
            return Err(CodError::EmptySource);
        }

        let entities = extract_entities(source);
        let sentences = split_sentences(source);

        let target = self.config.target_words;
        let mut current = initial_summary(&sentences, target);

        let mut steps: Vec<DensityStep> = Vec::with_capacity(self.config.iterations);

        for _ in 0..self.config.iterations {
            // Select the next batch of source entities not yet present.
            let mut batch: Vec<String> = Vec::with_capacity(self.config.entities_per_step);
            for entity in &entities {
                if batch.len() >= self.config.entities_per_step {
                    break;
                }
                if !entity_present(&current, entity) && !batch.contains(entity) {
                    batch.push(entity.clone());
                }
            }

            // Rewrite the summary, folding in the selected entities.
            if !batch.is_empty() {
                let clause = compose_clause(&batch);
                current = ensure_period(&format!("{current} {clause}"));
            }

            // Compress by dropping trailing filler if over budget.
            if word_count(&current) > target {
                current = trim_filler(&current, target);
            }

            let entity_count = distinct_entities_present(&current, &entities);
            steps.push(DensityStep {
                summary: current.clone(),
                added_entities: batch,
                word_count: word_count(&current),
                entity_count,
            });
        }

        let final_summary = steps
            .last()
            .map_or_else(|| current.clone(), |s| s.summary.clone());

        Ok(ChainOfDensityOutput {
            steps,
            final_summary,
        })
    }
}

impl Default for ChainOfDensityEngine {
    fn default() -> Self {
        Self::new(CodConfig::default())
    }
}
