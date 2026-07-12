//! The pluggable [`CoaWorker`] / [`CoaManager`] traits, plus deterministic
//! lexical default implementations ([`CoaLexicalWorker`],
//! [`CoaLexicalManager`]) so the chain can be exercised — and this module's
//! tests can prove the sequential-forward-flow property — without a live
//! model.
//!
//! ## Lexical helpers
//!
//! Deliberately self-contained (not imported from `multi_agent_debate` or
//! any other module): each module in this crate owns its own small lexical
//! toolkit rather than sharing private helpers across module boundaries.
//! Content tokens are lowercase alphanumeric runs of at least three
//! characters, excluding a small stopword list; term matching additionally
//! tolerates simple suffix variation (e.g. `"launch"` matching
//! `"launched"`) via a cheap prefix-based stemming heuristic, so evidence
//! scoring is not defeated by trivial morphology. No randomness
//! (`rand`/`rand_distr`) and no array/tensor machinery (`ndarray`/
//! `SciRS2-Core`) is needed anywhere in this module — every input and
//! output is plain text.

use std::collections::{BTreeMap, BTreeSet};

use super::engine::split_sentences;
use super::types::{CoaError, CoaEvidence};
use super::unit::CoaCommunicationUnit;

// ── CoaWorker ────────────────────────────────────────────────────────────────

/// A pluggable participant in a chain-of-agents run.
///
/// Worker `i` is handed the query, the `i`-th chunk, and the communication
/// unit produced by worker `i - 1` (or a fresh, empty unit for worker `0` —
/// see [`CoaCommunicationUnit::new`]), and must return an *updated*
/// communication unit that folds in whatever this chunk contributed. This is
/// the sole channel through which information can flow forward along the
/// chain: a worker never sees any chunk but its own, and only ever learns
/// about earlier chunks through what a prior worker chose to carry forward
/// in `incoming`.
///
/// [`CoaLexicalWorker`] provides a deterministic implementation for tests
/// and examples.
pub trait CoaWorker {
    /// Process one chunk, threading the communication unit forward.
    ///
    /// # Errors
    ///
    /// Implementations may return [`CoaError::WorkerFailed`] (or any other
    /// [`CoaError`] variant that genuinely applies) when they cannot
    /// process the chunk.
    fn process(
        &self,
        query: &str,
        chunk: &str,
        incoming: &CoaCommunicationUnit,
    ) -> Result<CoaCommunicationUnit, CoaError>;
}

// ── CoaManager ───────────────────────────────────────────────────────────────

/// The pluggable final-synthesis step of a chain-of-agents run.
///
/// Structurally distinct from [`CoaWorker`]: `synthesize` takes only the
/// query and the *final* communication unit — there is no parameter through
/// which a raw chunk could reach it. A [`CoaManager`] implementation
/// therefore cannot see anything the chain's workers did not choose to
/// carry all the way to the end, by construction rather than by convention.
///
/// [`CoaLexicalManager`] provides a deterministic implementation for tests
/// and examples.
pub trait CoaManager {
    /// Produce the final answer from `final_unit` alone.
    ///
    /// # Errors
    ///
    /// Implementations may return [`CoaError::ManagerFailed`] (or any other
    /// [`CoaError`] variant that genuinely applies) when they cannot
    /// synthesize an answer.
    fn synthesize(
        &self,
        query: &str,
        final_unit: &CoaCommunicationUnit,
    ) -> Result<String, CoaError>;
}

// ── lexical helpers ──────────────────────────────────────────────────────────

/// Stopwords excluded from the content-term vocabulary.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "did", "its", "let", "put", "say", "she", "too", "use", "that",
    "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then", "than",
    "into", "some", "such", "only", "also", "been", "more", "very", "will", "would", "there",
    "their", "which", "about", "could", "these", "those", "does", "still", "even", "where", "why",
    "does", "being",
];

/// Sentence-initial capitalized words that are never proper-noun subjects
/// (articles, pronouns, question words, auxiliaries), excluded from
/// [`capitalized_tokens`] regardless of case.
const CAPITALIZED_STOPWORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "i", "when", "who", "what", "where", "why", "how", "did", "was", "were", "is", "are", "does",
    "do", "will", "would", "can", "could",
];

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep
/// non-empty fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` when `token` is a content token: at least three characters
/// and not a stopword.
fn is_content_token(token: &str) -> bool {
    token.chars().count() >= 3 && !STOPWORDS.contains(&token)
}

/// The distinct, sorted content-token vocabulary of `text`. A [`BTreeSet`]
/// (rather than a hash-based set) so iteration order is itself deterministic
/// across process runs.
fn content_terms(text: &str) -> BTreeSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| is_content_token(t))
        .collect()
}

/// Cheap prefix-based stemming heuristic: two content tokens "match" when
/// they are equal, or when the shorter is at least 4 characters and is a
/// prefix of the longer (e.g. `"launch"` / `"launched"`, `"probe"` /
/// `"probes"`). Deliberately simple rather than a real stemmer (Porter,
/// Snowball, ...) — no such dependency exists in this crate and adding one
/// is out of scope for a deterministic, dependency-free default
/// implementation.
fn stem_match(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    shorter.len() >= 4 && longer.starts_with(shorter)
}

/// Recall-oriented relevance score: the fraction of `query_terms` that have
/// a [`stem_match`] somewhere in `sentence_terms`, in `[0.0, 1.0]`. `0.0`
/// when `query_terms` is empty.
fn term_overlap_score(query_terms: &BTreeSet<String>, sentence_terms: &BTreeSet<String>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let matched = query_terms
        .iter()
        .filter(|q| sentence_terms.iter().any(|s| stem_match(q, s)))
        .count();
    #[allow(clippy::cast_precision_loss)]
    {
        matched as f32 / query_terms.len() as f32
    }
}

/// The subset of `query_terms` that have a [`stem_match`] somewhere across
/// `evidence`'s text.
fn covered_terms(query_terms: &BTreeSet<String>, evidence: &[CoaEvidence]) -> BTreeSet<String> {
    let evidence_terms: BTreeSet<String> = evidence
        .iter()
        .flat_map(|e| content_terms(&e.text))
        .collect();
    query_terms
        .iter()
        .filter(|q| evidence_terms.iter().any(|t| stem_match(q, t)))
        .cloned()
        .collect()
}

/// Capitalized (proper-noun-like) tokens of `text`, in order of appearance,
/// excluding [`CAPITALIZED_STOPWORDS`] (case-insensitively) and any token
/// that isn't purely alphabetic (so years like `"2031"` are never mistaken
/// for a name).
fn capitalized_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| s.chars().next().is_some_and(char::is_uppercase))
        .filter(|s| s.chars().all(char::is_alphabetic))
        .filter(|s| !CAPITALIZED_STOPWORDS.contains(&s.to_lowercase().as_str()))
        .map(str::to_string)
        .collect()
}

/// The first 4-digit run of ASCII digits in `text`, parsed as a year.
fn first_year_token(text: &str) -> Option<u32> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|s| s.len() == 4)
        .find_map(|s| s.parse::<u32>().ok())
}

/// Markers that introduce a rename / alias relation, e.g. `"X was renamed
/// Y"`. Checked case-insensitively; ordered longest-first so a longer marker
/// is preferred over a shorter one it contains.
const ALIAS_MARKERS: &[&str] = &[
    " was renamed to ",
    " was renamed ",
    " renamed to ",
    " is now known as ",
    " is now called ",
    " now known as ",
    " now called ",
    " also known as ",
];

/// Attempt to extract a rename / alias relation (`original name`, `new
/// name`) from `sentence`, e.g. `"The Kepler probe was renamed Artemis."` →
/// `("Kepler", "Artemis")`. A simple, deterministic pattern match — not a
/// general coreference resolver — over a fixed set of [`ALIAS_MARKERS`].
fn extract_alias_relation(sentence: &str) -> Option<(String, String)> {
    let lower = sentence.to_lowercase();
    for marker in ALIAS_MARKERS {
        if let Some(pos) = lower.find(marker) {
            let before = &sentence[..pos];
            let after = &sentence[pos + marker.len()..];
            let original = capitalized_tokens(before).into_iter().next_back()?;
            let renamed = capitalized_tokens(after).into_iter().next()?;
            return Some((original, renamed));
        }
    }
    None
}

/// Attempt to extract a "launched in `<year>`" fact (`subject`, `year`) from
/// `sentence`, e.g. `"Artemis launched in 2031."` → `("Artemis", 2031)`.
fn extract_launch_fact(sentence: &str) -> Option<(String, u32)> {
    let lower = sentence.to_lowercase();
    let marker_pos = lower.find("launch")?;
    let before = &sentence[..marker_pos];
    let after = &sentence[marker_pos..];
    let subject = capitalized_tokens(before).into_iter().next_back()?;
    let year = first_year_token(after)?;
    Some((subject, year))
}

/// Follow `aliases` from `name` (case-insensitively keyed) up to `max_hops`
/// steps, returning the last display-cased name reached. Guards against
/// alias cycles by refusing to revisit an already-seen key.
fn resolve_alias(aliases: &BTreeMap<String, String>, name: &str, max_hops: usize) -> String {
    let mut current = name.to_string();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for _ in 0..max_hops {
        if !seen.insert(current.to_lowercase()) {
            break;
        }
        match aliases.get(&current.to_lowercase()) {
            Some(next) => current.clone_from(next),
            None => break,
        }
    }
    current
}

// ── CoaLexicalWorker ─────────────────────────────────────────────────────────

/// Deterministic [`CoaWorker`] for tests and examples.
///
/// No live model is involved: [`CoaLexicalWorker::process`] splits `chunk`
/// into sentences, scores each against the query by `term_overlap_score`,
/// and [`CoaCommunicationUnit::merge_evidence`]s every sentence that scores
/// above `0.0` into the incoming unit (which performs the actual bounded
/// eviction — this worker never grows `evidence` unboundedly itself).
/// `completeness` is recomputed from scratch each call as the fraction of
/// query content terms `covered_terms` by the *post-merge* evidence set;
/// `open_questions` tracks the query terms not yet covered, resolving
/// entries once they are; `partial_answer` mirrors the current top-ranked
/// evidence entry (or stays empty if there is none yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoaLexicalWorker;

impl CoaWorker for CoaLexicalWorker {
    fn process(
        &self,
        query: &str,
        chunk: &str,
        incoming: &CoaCommunicationUnit,
    ) -> Result<CoaCommunicationUnit, CoaError> {
        if query.trim().is_empty() {
            return Err(CoaError::EmptyQuery);
        }

        let chunk_index = incoming.chunks_seen;
        let query_terms = content_terms(query);
        let mut unit = incoming.clone();

        let candidates: Vec<CoaEvidence> = split_sentences(chunk)
            .into_iter()
            .filter_map(|sentence| {
                let sentence_terms = content_terms(sentence);
                let score = term_overlap_score(&query_terms, &sentence_terms);
                (score > 0.0).then(|| CoaEvidence::new(sentence, chunk_index, score))
            })
            .collect();
        unit.merge_evidence(candidates);

        let covered = covered_terms(&query_terms, &unit.evidence);
        unit.resolve_open_questions(|question| {
            covered.iter().any(|term| question.contains(term.as_str()))
        });
        let new_gaps: Vec<String> = query_terms
            .difference(&covered)
            .map(|term| format!("unresolved aspect of the query: \"{term}\""))
            .filter(|question| !unit.open_questions.contains(question))
            .collect();
        unit.merge_open_questions(new_gaps);

        unit.completeness = if query_terms.is_empty() {
            1.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                covered.len() as f32 / query_terms.len() as f32
            }
        };

        unit.partial_answer = unit
            .evidence
            .first()
            .map(|e| e.text.clone())
            .unwrap_or_default();

        unit.chunks_seen = incoming.chunks_seen + 1;
        Ok(unit)
    }
}

// ── CoaLexicalManager ────────────────────────────────────────────────────────

/// Deterministic [`CoaManager`] for tests and examples.
///
/// No live model is involved and — like every [`CoaManager`] — it never
/// sees a raw chunk, only `final_unit`. [`CoaLexicalManager::synthesize`]
/// scans `final_unit.evidence` (and *only* `final_unit.evidence` — nothing
/// evicted from the unit earlier in the chain is visible here) for two
/// simple relation patterns via `extract_alias_relation` and
/// `extract_launch_fact`, builds an alias map and a fact map from
/// whatever it finds, then tries to answer a "when did `<subject>` launch"
/// style query by resolving the query's subject through the alias chain
/// (via `resolve_alias`) before looking it up in the fact map — this is
/// what allows a rename learned from one chunk to be combined with a launch
/// date learned from a much later chunk, entirely through what both chunks'
/// workers chose to carry into `final_unit`. When no structured fact
/// resolves, it falls back to the top-ranked evidence entry, then to
/// `partial_answer`, then to an honest "insufficient evidence" message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoaLexicalManager;

/// Maximum alias hops [`resolve_alias`] follows before giving up, used by
/// [`CoaLexicalManager`].
const MAX_ALIAS_HOPS: usize = 8;

impl CoaManager for CoaLexicalManager {
    fn synthesize(
        &self,
        query: &str,
        final_unit: &CoaCommunicationUnit,
    ) -> Result<String, CoaError> {
        if query.trim().is_empty() {
            return Err(CoaError::EmptyQuery);
        }

        let mut aliases: BTreeMap<String, String> = BTreeMap::new();
        let mut facts: BTreeMap<String, (String, u32)> = BTreeMap::new();
        for evidence in &final_unit.evidence {
            if let Some((from, to)) = extract_alias_relation(&evidence.text) {
                aliases.insert(from.to_lowercase(), to);
            }
            if let Some((subject, year)) = extract_launch_fact(&evidence.text) {
                facts.insert(subject.to_lowercase(), (subject, year));
            }
        }

        let query_subject = capitalized_tokens(query).into_iter().next();
        if let Some(subject) = query_subject {
            let resolved = resolve_alias(&aliases, &subject, MAX_ALIAS_HOPS);
            if let Some((display, year)) = facts.get(&resolved.to_lowercase()) {
                return Ok(if resolved.eq_ignore_ascii_case(&subject) {
                    format!("{subject} launched in {year}.")
                } else {
                    format!("{subject} launched in {year} (as {display}).")
                });
            }
        }

        if let Some(top) = final_unit.evidence.first() {
            return Ok(format!("Based on the available evidence: {}", top.text));
        }
        if !final_unit.partial_answer.trim().is_empty() {
            return Ok(final_unit.partial_answer.clone());
        }
        Ok(format!(
            "insufficient evidence to answer \"{query}\" (completeness={:.2})",
            final_unit.completeness
        ))
    }
}
