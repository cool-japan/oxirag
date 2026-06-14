//! Question templating for synthetic evaluation-set generation.
//!
//! The [`HeuristicTemplater`] turns a salient sentence inside a document into a
//! set of `(question, answer, type)` tuples using purely lexical heuristics:
//! term-frequency salience, capitalized-token entity detection, and a small
//! stopword list.  No external models or randomness are involved, so output is
//! fully deterministic.

use crate::synthetic_eval::types::QuestionType;
use crate::types::Document;
use std::collections::HashMap;

// ── Tokenization & salience helpers ───────────────────────────────────────────

/// A small, fixed stopword list used to discount non-salient terms.
///
/// Kept intentionally short and lower-case; matching is performed on the
/// lower-cased token form produced by [`tokenize`].
pub(crate) const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "have", "has",
    "had", "not", "but", "they", "their", "them", "its", "his", "her", "she", "him", "you", "your",
    "our", "ours", "who", "what", "which", "when", "where", "why", "how", "into", "over", "under",
    "than", "then", "there", "here", "such", "some", "any", "all", "can", "will", "would", "could",
    "should", "may", "might", "must", "been", "being", "also", "about", "between", "of", "in",
    "on", "at", "to", "by", "as", "is", "an", "a", "it", "be", "or", "if", "so", "we", "do",
];

/// Tokenize text into lower-case alphanumeric tokens of length `>= 2`.
///
/// This matches the project-wide tokenizer convention used by the lexical
/// cross-encoder and evaluation modules.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` if the lower-cased token is in the stopword list.
#[must_use]
pub(crate) fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// Split text into trimmed, non-empty sentences on `.`, `!`, and `?`.
#[must_use]
pub fn sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Compute term-frequency salience over `tokens`, discounting stopwords.
///
/// The returned map weights each non-stopword token by its term frequency.
/// Stopwords are assigned a salience of `0.0` so they are never chosen as the
/// most salient term.
#[must_use]
pub(crate) fn salience(tokens: &[String]) -> HashMap<String, f32> {
    let mut tf: HashMap<String, f32> = HashMap::new();
    for tok in tokens {
        if is_stopword(tok) {
            continue;
        }
        *tf.entry(tok.clone()).or_insert(0.0) += 1.0;
    }
    tf
}

/// Pick the most salient token in a sentence, breaking ties by first appearance.
///
/// Returns the original-cased surface form (as it appears in the sentence) of the
/// winning token, or `None` if the sentence has no non-stopword token.
#[must_use]
pub(crate) fn most_salient_term(sentence: &str) -> Option<String> {
    let tokens = tokenize(sentence);
    let sal = salience(&tokens);
    if sal.is_empty() {
        return None;
    }
    // Determine the best lower-cased token deterministically: highest score, then
    // earliest first occurrence in the token stream. Salience values are integer
    // term frequencies, so they are compared via total ordering on their bits.
    let mut best: Option<&String> = None;
    let mut best_bits = u32::MIN;
    for tok in &tokens {
        let score = sal.get(tok).copied().unwrap_or(0.0);
        if score <= 0.0 {
            continue;
        }
        let bits = score.to_bits();
        if best.is_none() || bits > best_bits {
            best_bits = bits;
            best = Some(tok);
        }
    }
    let best = best?;
    // Recover the original surface form from the sentence for nicer blanks.
    surface_form(sentence, best).or_else(|| Some(best.clone()))
}

/// Find the first surface (original-cased) occurrence of `lower_token` in `text`.
#[must_use]
pub(crate) fn surface_form(text: &str, lower_token: &str) -> Option<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .find(|t| t.to_lowercase() == lower_token)
        .map(str::to_string)
}

/// Extract capitalized multi-character tokens (length `>= 2`) as entity terms.
///
/// Consecutive capitalized tokens are joined into a single multi-word entity
/// (e.g. "New York"). Order of first appearance is preserved and duplicates are
/// removed.
#[must_use]
pub(crate) fn entities(sentence: &str) -> Vec<String> {
    let raw: Vec<&str> = sentence
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let mut result: Vec<String> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for tok in raw {
        if is_capitalized(tok) {
            current.push(tok);
        } else {
            flush_entity(&mut current, &mut result);
        }
    }
    flush_entity(&mut current, &mut result);
    result
}

/// Push the accumulated capitalized run (if any) as one entity, then clear it.
fn flush_entity(current: &mut Vec<&str>, result: &mut Vec<String>) {
    if !current.is_empty() {
        let joined = current.join(" ");
        if !result.contains(&joined) {
            result.push(joined);
        }
        current.clear();
    }
}

/// Return `true` if the first character of the token is upper-case.
#[must_use]
fn is_capitalized(token: &str) -> bool {
    token.chars().next().is_some_and(char::is_uppercase)
}

/// The leading subject of a sentence: its first capitalized entity run.
#[must_use]
pub(crate) fn leading_subject(sentence: &str) -> Option<String> {
    entities(sentence).into_iter().next()
}

// ── QuestionTemplater trait ───────────────────────────────────────────────────

/// Produces `(question, answer, type)` tuples from a sentence within a document.
pub trait QuestionTemplater {
    /// Generate `(question, answer, type)` tuples from a sentence within a document.
    ///
    /// Only the question types present in `allowed` should be produced. The order
    /// of returned tuples must be deterministic for a given input.
    fn generate(
        &self,
        sentence: &str,
        doc: &Document,
        allowed: &[QuestionType],
    ) -> Vec<(String, String, QuestionType)>;
}

// ── HeuristicTemplater ────────────────────────────────────────────────────────

/// A purely lexical, deterministic [`QuestionTemplater`].
///
/// Generates factoid, definitional, cloze, and relational questions using
/// term-frequency salience and capitalized-token entity detection.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicTemplater;

impl HeuristicTemplater {
    /// Create a new heuristic templater.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Build a factoid question from a leading subject and the remaining predicate.
    fn factoid(sentence: &str) -> Option<(String, String, QuestionType)> {
        let subject = leading_subject(sentence)?;
        // The remainder after the subject is the predicate/object (the answer).
        let idx = sentence.find(subject.as_str())?;
        let remainder = sentence[idx + subject.len()..].trim();
        if remainder.is_empty() || tokenize(remainder).is_empty() {
            return None;
        }
        let question = format!("What did {subject} do?");
        Some((question, remainder.to_string(), QuestionType::Factoid))
    }

    /// Build a definitional question for the most salient entity in the sentence.
    fn definitional(sentence: &str) -> Option<(String, String, QuestionType)> {
        let entity = entities(sentence).into_iter().next()?;
        let question = format!("What is {entity}?");
        Some((question, sentence.to_string(), QuestionType::Definitional))
    }

    /// Build a cloze question by masking the most salient term in the sentence.
    fn cloze(sentence: &str) -> Option<(String, String, QuestionType)> {
        let term = most_salient_term(sentence)?;
        let blanked = blank_first(sentence, &term)?;
        Some((blanked, term, QuestionType::Cloze))
    }

    /// Build a relational question from the first two entities in the sentence.
    fn relational(sentence: &str) -> Option<(String, String, QuestionType)> {
        let ents = entities(sentence);
        if ents.len() < 2 {
            return None;
        }
        let question = format!(
            "What is the relationship between {} and {}?",
            ents[0], ents[1]
        );
        Some((question, sentence.to_string(), QuestionType::Relational))
    }
}

/// Replace the first surface occurrence of `term` in `sentence` with `___`.
///
/// Matching is case-insensitive on the alphanumeric token boundary so that the
/// blank lands on the exact word, not on a substring of a larger word.
#[must_use]
fn blank_first(sentence: &str, term: &str) -> Option<String> {
    let lower_term = term.to_lowercase();
    let bytes = sentence.as_bytes();
    let mut start = 0usize;
    while start < sentence.len() {
        // Advance over non-alphanumeric separators.
        while start < sentence.len() && !is_word_byte(bytes, start) {
            start += 1;
        }
        if start >= sentence.len() {
            break;
        }
        let mut end = start;
        while end < sentence.len() && is_word_byte(bytes, end) {
            end += 1;
        }
        let word = &sentence[start..end];
        if word.to_lowercase() == lower_term {
            let mut out = String::with_capacity(sentence.len());
            out.push_str(&sentence[..start]);
            out.push_str("___");
            out.push_str(&sentence[end..]);
            return Some(out);
        }
        start = end;
    }
    None
}

/// Return `true` if the byte at `index` is part of an alphanumeric word.
#[must_use]
fn is_word_byte(bytes: &[u8], index: usize) -> bool {
    let b = bytes[index];
    b.is_ascii_alphanumeric() || b >= 0x80
}

impl QuestionTemplater for HeuristicTemplater {
    fn generate(
        &self,
        sentence: &str,
        _doc: &Document,
        allowed: &[QuestionType],
    ) -> Vec<(String, String, QuestionType)> {
        let mut out: Vec<(String, String, QuestionType)> = Vec::new();
        // Emit in canonical template order, filtered by `allowed`.
        for kind in QuestionType::all() {
            if !allowed.contains(&kind) {
                continue;
            }
            let produced = match kind {
                QuestionType::Factoid => Self::factoid(sentence),
                QuestionType::Definitional => Self::definitional(sentence),
                QuestionType::Cloze => Self::cloze(sentence),
                QuestionType::Relational => Self::relational(sentence),
            };
            if let Some(tuple) = produced {
                out.push(tuple);
            }
        }
        out
    }
}
