//! Query reformulation for multi-turn conversational RAG.
//!
//! This module provides the machinery to transform a raw user query into a
//! context-aware form that can be passed directly to the retrieval pipeline.
//!
//! # Strategy overview
//!
//! | Strategy | Result |
//! |---|---|
//! | [`Concatenation`] | `"{context}\n\nFollow-up: {query}"` |
//! | [`ContextInjection`] | `"[Context: {last_assistant}] {query}"` |
//! | [`FollowUpResolution`] | Pronouns in `query` replaced with entity references |
//! | [`Standalone`] | Query returned unchanged |
//!
//! [`Concatenation`]: ReformulationStrategy::Concatenation
//! [`ContextInjection`]: ReformulationStrategy::ContextInjection
//! [`FollowUpResolution`]: ReformulationStrategy::FollowUpResolution
//! [`Standalone`]: ReformulationStrategy::Standalone

use serde::{Deserialize, Serialize};

use super::types::{ConversationError, ConversationHistory};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Pronouns that suggest a follow-up query rather than a standalone question.
const FOLLOW_UP_PRONOUNS: &[&str] = &[
    "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those", "he", "him",
    "his", "she", "her", "hers",
];

/// Comparative or additive qualifiers that suggest the query continues a thread.
const FOLLOW_UP_QUALIFIERS: &[&str] = &["also", "too", "as well", "either", "likewise"];

/// Maximum query length (in bytes) below which a short query is considered a
/// likely follow-up even without explicit pronoun markers.
const SHORT_QUERY_THRESHOLD: usize = 50;

// ── ReformulationStrategy ─────────────────────────────────────────────────────

/// How to reformulate a user query given the conversation context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReformulationStrategy {
    /// Prepend the full context string followed by the query.
    ///
    /// Produces: `"{context}\n\nFollow-up: {query}"`
    Concatenation,

    /// Inject a summary of the last assistant turn before the query.
    ///
    /// Produces: `"[Context: {last_assistant_summary}] {query}"`
    ContextInjection,

    /// Detect and resolve pronouns in the query using entity references
    /// extracted from recent assistant turns.
    #[default]
    FollowUpResolution,

    /// Return the query unchanged — treat it as completely self-contained.
    Standalone,
}

// ── QueryReformulator ─────────────────────────────────────────────────────────

/// Transforms a raw user query into a context-enriched form.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "conversational")]
/// # {
/// use oxirag::conversation::reformulator::{QueryReformulator, ReformulationStrategy};
///
/// let reformulator = QueryReformulator::new(ReformulationStrategy::Concatenation);
/// let result = reformulator.reformulate("What is its purpose?", "Rust is a systems language.").unwrap();
/// assert!(result.contains("Rust is a systems language."));
/// assert!(result.contains("What is its purpose?"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QueryReformulator {
    /// The reformulation strategy to apply.
    pub strategy: ReformulationStrategy,
}

impl QueryReformulator {
    /// Create a new reformulator with the given strategy.
    #[must_use]
    pub fn new(strategy: ReformulationStrategy) -> Self {
        Self { strategy }
    }

    /// Reformulate `query` using `context` and the configured strategy.
    ///
    /// # Errors
    ///
    /// - [`ConversationError::EmptyQuery`] if `query` is blank after trimming.
    /// - [`ConversationError::ReformulationFailed`] if the strategy cannot
    ///   produce a usable result.
    pub fn reformulate(&self, query: &str, context: &str) -> Result<String, ConversationError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(ConversationError::EmptyQuery);
        }

        match &self.strategy {
            ReformulationStrategy::Concatenation => {
                if context.is_empty() {
                    Ok(trimmed.to_string())
                } else {
                    Ok(format!("{context}\n\nFollow-up: {trimmed}"))
                }
            }

            ReformulationStrategy::ContextInjection => {
                let last_assistant = extract_last_assistant_summary(context);
                if last_assistant.is_empty() {
                    Ok(trimmed.to_string())
                } else {
                    Ok(format!("[Context: {last_assistant}] {trimmed}"))
                }
            }

            ReformulationStrategy::FollowUpResolution => {
                // Build a synthetic history view from the context string so the
                // detector can parse it.  We extract references directly from
                // context lines that start with "Assistant:".
                let references = extract_references_from_context(context);
                let resolved = FollowUpDetector::resolve_pronouns(trimmed, &references);
                Ok(resolved)
            }

            ReformulationStrategy::Standalone => Ok(trimmed.to_string()),
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract the last assistant turn from a formatted context string.
///
/// The context is expected to contain lines of the form `"Assistant: ..."`.
/// Returns an empty string if no assistant turn is found.
fn extract_last_assistant_summary(context: &str) -> String {
    let prefix = "Assistant: ";
    context
        .lines()
        .rfind(|line| line.starts_with(prefix))
        .map(|line| line[prefix.len()..].trim().to_string())
        .unwrap_or_default()
}

/// Extract capitalised word references from lines that start with "Assistant:"
/// in a formatted context string. Also extracts quoted phrases.
fn extract_references_from_context(context: &str) -> Vec<String> {
    let prefix = "Assistant: ";
    let mut references: Vec<String> = Vec::new();

    for line in context.lines() {
        if !line.starts_with(prefix) {
            continue;
        }
        let text = &line[prefix.len()..];

        // Collect quoted phrases first.
        let mut in_quote = false;
        let mut quote_start = 0usize;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            match chars[i] {
                '"' | '\'' => {
                    if in_quote {
                        let phrase: String = chars[quote_start..i].iter().collect();
                        let phrase = phrase.trim().to_string();
                        if phrase.len() > 3 {
                            references.push(phrase);
                        }
                        in_quote = false;
                    } else {
                        in_quote = true;
                        quote_start = i + 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }

        // Collect capitalised words longer than 3 chars, skipping sentence-
        // start words (preceded by '. ' or at position 0).
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        for (idx, word) in words.iter().enumerate() {
            // Strip punctuation.
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-')
                .collect();
            if clean.len() <= 3 {
                continue;
            }
            let first_char = clean.chars().next().unwrap_or(' ');
            if !first_char.is_uppercase() {
                continue;
            }
            // Skip if it is the very first word of the text or follows a period.
            if idx == 0 {
                continue;
            }
            if idx > 0 {
                let prev = words[idx - 1];
                if prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?') {
                    continue;
                }
            }
            // Avoid duplicates.
            if !references.contains(&clean) && idx < word_count {
                references.push(clean);
            }
        }
    }

    references
}

// ── FollowUpDetector ──────────────────────────────────────────────────────────

/// Heuristic detector for follow-up queries and pronoun resolver.
///
/// This is a purely textual (no-ML) component that uses word-boundary matching
/// to detect pronouns and short queries that likely refer to context established
/// in prior turns.
pub struct FollowUpDetector;

impl FollowUpDetector {
    /// Return `true` when `query` is likely a follow-up to a previous turn.
    ///
    /// The heuristic fires when any of the following is true:
    /// 1. The query is shorter than `SHORT_QUERY_THRESHOLD` bytes.
    /// 2. The query contains one of the `FOLLOW_UP_PRONOUNS` at a word boundary.
    /// 3. The query contains a `FOLLOW_UP_QUALIFIERS` phrase.
    #[must_use]
    pub fn is_follow_up(query: &str) -> bool {
        let lower = query.to_lowercase();

        // Heuristic 1: very short query.
        if query.trim().len() < SHORT_QUERY_THRESHOLD {
            return true;
        }

        // Heuristic 2: pronoun at a word boundary.
        for pronoun in FOLLOW_UP_PRONOUNS {
            if contains_word_boundary(&lower, pronoun) {
                return true;
            }
        }

        // Heuristic 3: comparative / additive qualifiers.
        for qualifier in FOLLOW_UP_QUALIFIERS {
            if lower.contains(qualifier) {
                return true;
            }
        }

        false
    }

    /// Extract noun/entity references from the last 3 assistant turns in
    /// `history`.
    ///
    /// Uses the same heuristic as `extract_references_from_context` but
    /// operates directly on the [`ConversationHistory`].
    #[must_use]
    pub fn extract_references(history: &ConversationHistory) -> Vec<String> {
        use super::types::TurnRole;

        let mut references: Vec<String> = Vec::new();

        // Walk turns in reverse, collecting up to 3 assistant turns.
        let mut assistant_turns_seen = 0usize;
        for turn in history.iter().rev() {
            if turn.role != TurnRole::Assistant {
                continue;
            }
            if assistant_turns_seen >= 3 {
                break;
            }
            assistant_turns_seen += 1;

            // Collect quoted phrases.
            let text = &turn.text;
            let chars: Vec<char> = text.chars().collect();
            let mut in_quote = false;
            let mut quote_start = 0usize;
            let mut i = 0;
            while i < chars.len() {
                match chars[i] {
                    '"' | '\'' => {
                        if in_quote {
                            let phrase: String = chars[quote_start..i].iter().collect();
                            let phrase = phrase.trim().to_string();
                            if phrase.len() > 3 && !references.contains(&phrase) {
                                references.push(phrase);
                            }
                            in_quote = false;
                        } else {
                            in_quote = true;
                            quote_start = i + 1;
                        }
                    }
                    _ => {}
                }
                i += 1;
            }

            // Collect capitalised words.
            let words: Vec<&str> = text.split_whitespace().collect();
            for (idx, word) in words.iter().enumerate() {
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '-')
                    .collect();
                if clean.len() <= 3 {
                    continue;
                }
                let first_char = clean.chars().next().unwrap_or(' ');
                if !first_char.is_uppercase() {
                    continue;
                }
                if idx == 0 {
                    continue;
                }
                if idx > 0 {
                    let prev = words[idx - 1];
                    if prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?') {
                        continue;
                    }
                }
                if !references.contains(&clean) {
                    references.push(clean);
                }
            }
        }

        references
    }

    /// Resolve pronouns in `query` by prepending a reference to the most
    /// salient entity extracted from recent context.
    ///
    /// If `references` is non-empty and the query starts with a known pronoun,
    /// the result is `"Regarding {references[0]}: {query}"`.  Otherwise the
    /// query is returned unchanged.
    #[must_use]
    pub fn resolve_pronouns(query: &str, references: &[String]) -> String {
        if references.is_empty() {
            return query.to_string();
        }

        let lower = query.to_lowercase();
        let trimmed = query.trim();

        // Check whether the query starts with a follow-up pronoun.
        let starts_with_pronoun = FOLLOW_UP_PRONOUNS.iter().any(|pronoun| {
            lower.starts_with(pronoun)
                && lower[pronoun.len()..]
                    .chars()
                    .next()
                    .is_none_or(|c| !c.is_alphanumeric())
        });

        if starts_with_pronoun {
            format!("Regarding {}: {trimmed}", references[0])
        } else {
            trimmed.to_string()
        }
    }
}

// ── Word-boundary search ───────────────────────────────────────────────────────

/// Return `true` if `needle` appears in `haystack` at a word boundary.
fn contains_word_boundary(haystack: &str, needle: &str) -> bool {
    let haystack_bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let needle_len = needle_bytes.len();
    let hay_len = haystack_bytes.len();

    if needle_len > hay_len {
        return false;
    }

    let mut start = 0usize;
    while start + needle_len <= hay_len {
        // Find the next occurrence.
        match haystack[start..].find(needle) {
            None => return false,
            Some(rel_pos) => {
                let pos = start + rel_pos;
                let end = pos + needle_len;

                let left_ok = pos == 0 || !haystack_bytes[pos - 1].is_ascii_alphanumeric();
                let right_ok = end == hay_len || !haystack_bytes[end].is_ascii_alphanumeric();

                if left_ok && right_ok {
                    return true;
                }
                start = pos + 1;
            }
        }
    }
    false
}

// ── ConversationAwareQuery ─────────────────────────────────────────────────────

/// The output of the reformulation step — a bundle that carries both the
/// original and reformulated forms of a query alongside provenance information.
#[derive(Debug, Clone)]
pub struct ConversationAwareQuery {
    /// The exact string the user typed.
    pub original_query: String,
    /// The context-enriched form suitable for retrieval.
    pub reformulated_query: String,
    /// The context string that was used during reformulation.
    pub context_used: String,
    /// Whether the detector classified this as a follow-up query.
    pub is_follow_up: bool,
}

impl ConversationAwareQuery {
    /// Construct a new `ConversationAwareQuery`.
    #[must_use]
    pub fn new(
        original: impl Into<String>,
        reformulated: impl Into<String>,
        context: impl Into<String>,
        is_follow_up: bool,
    ) -> Self {
        Self {
            original_query: original.into(),
            reformulated_query: reformulated.into(),
            context_used: context.into(),
            is_follow_up,
        }
    }

    /// The text that should be fed into the retrieval system.
    #[must_use]
    pub fn as_search_text(&self) -> &str {
        &self.reformulated_query
    }
}
