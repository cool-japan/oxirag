//! The [`QueryRefiner`] trait, its deterministic [`MockRefiner`]
//! implementation, and the [`QueryRefinementEngine`] orchestrator.
//!
//! ## Heuristic classification algorithm ([`MockRefiner`])
//!
//! 1. **Decompose** — split the query on compound-clause separators (`" and
//!    "`, `" or "`, `"; "`, `" as well as "`, ...) or on multiple `?`
//!    characters. If the split yields at least
//!    [`RqRagConfig::min_decompose_parts`] non-empty parts, classify as
//!    [`RefinementAction::Decompose`].
//! 2. **Disambiguate** — otherwise, if any configured
//!    [`RqRagConfig::ambiguity_markers`] entry is present at a word boundary,
//!    classify as [`RefinementAction::Disambiguate`].
//! 3. **Rewrite** — otherwise, if any configured
//!    [`RqRagConfig::colloquial_markers`] entry is present at a word boundary,
//!    classify as [`RefinementAction::Rewrite`].
//! 4. **Respond** — otherwise, the query is already clear and atomic.
//!
//! All matching is deterministic, ASCII-case-insensitive, and word-boundary
//! aware (so e.g. `"or"` never matches inside `"horror"`).

use super::types::{
    AMBIGUITY_SENSE_TABLE, RefinementAction, RefinementPlan, RqRagConfig, RqRagError,
};

// ── Word-boundary / substring helpers ────────────────────────────────────────
//
// All helpers operate byte-wise using `str::find`, whose match positions are
// always valid UTF-8 char boundaries, so these functions never panic on
// arbitrary (including multi-byte) input.

/// Returns the byte range `(start, end)` of the first word-boundary match of
/// `marker` inside `text`, or `None` if no such match exists.
///
/// A match is only accepted when the byte immediately before `start` (if any)
/// and the byte immediately at `end` (if any) are not ASCII alphanumeric —
/// this prevents `"or"` from matching inside `"horror"` or `"um"` from
/// matching inside `"forum"`.
fn find_boundary_match(text: &str, marker: &str) -> Option<(usize, usize)> {
    let marker_len = marker.len();
    let text_len = text.len();
    if marker_len == 0 || marker_len > text_len {
        return None;
    }

    let mut search_start = 0usize;
    while search_start + marker_len <= text_len {
        let rel = text[search_start..].find(marker)?;
        let pos = search_start + rel;
        let bytes = text.as_bytes();
        let before_ok = pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
        let end = pos + marker_len;
        let after_ok = end >= text_len || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some((pos, end));
        }
        search_start = pos + 1;
    }
    None
}

/// Returns `true` if `marker` is present in `haystack` at a word boundary
/// (ASCII case-insensitive).
fn marker_present(haystack: &str, marker: &str) -> bool {
    if marker.trim().is_empty() {
        return false;
    }
    let haystack_lower = haystack.to_ascii_lowercase();
    let marker_lower = marker.to_ascii_lowercase();
    find_boundary_match(&haystack_lower, &marker_lower).is_some()
}

/// Removes every word-boundary occurrence of `marker_lower` from `text`
/// (which must already be ASCII-lowercased), replacing each with a single
/// space.
fn remove_all_marker_occurrences(text: &str, marker_lower: &str) -> String {
    if marker_lower.trim().is_empty() {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        if let Some((start, end)) = find_boundary_match(rest, marker_lower) {
            result.push_str(&rest[..start]);
            result.push(' ');
            rest = &rest[end..];
        } else {
            result.push_str(rest);
            break;
        }
    }
    result
}

/// Collapses runs of whitespace to a single space, trims the result, and
/// removes stray spaces immediately before common terminal punctuation.
fn collapse_whitespace(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut collapsed = joined;
    for punct in ["?", ".", ",", "!"] {
        let spaced = format!(" {punct}");
        collapsed = collapsed.replace(&spaced, punct);
    }
    collapsed.trim().to_string()
}

/// Strips leftover leading punctuation (commas/semicolons/colons) that can
/// result from removing a marker at the very start of the query, then
/// re-trims whitespace.
fn strip_leading_punctuation(text: &str) -> String {
    text.trim_start_matches([',', ';', ':']).trim().to_string()
}

/// Capitalises the first alphabetic character of `text`, leaving everything
/// before and after it unchanged.
fn capitalize_first(text: &str) -> String {
    match text.find(char::is_alphabetic) {
        Some(idx) => {
            let (prefix, rest) = text.split_at(idx);
            let mut chars = rest.chars();
            match chars.next() {
                Some(first) => {
                    format!("{prefix}{}{}", first.to_uppercase(), chars.as_str())
                }
                None => text.to_string(),
            }
        }
        None => text.to_string(),
    }
}

// ── Compound-clause splitting ─────────────────────────────────────────────────

/// Separators that indicate a compound (multi-part) query.
const COMPOUND_SEPARATORS: &[&str] = &[
    ", and ",
    ", or ",
    " as well as ",
    " along with ",
    " together with ",
    " and ",
    " or ",
    "; ",
];

/// Splits `text` (ASCII case-insensitively) on the first occurrence of `sep`
/// found anywhere, returning every occurrence-delimited segment.
///
/// Splitting is performed by locating `sep` in an ASCII-lowercased copy of
/// `text` (byte-length-preserving) and slicing the *original* `text` at the
/// same offsets, so the returned segments retain their original casing.
fn split_case_insensitive<'a>(text: &'a str, sep: &str) -> Vec<&'a str> {
    let lower = text.to_ascii_lowercase();
    let mut parts = Vec::new();
    let mut start = 0usize;
    while let Some(pos) = lower[start..].find(sep) {
        let abs = start + pos;
        parts.push(&text[start..abs]);
        start = abs + sep.len();
    }
    parts.push(&text[start..]);
    parts
}

/// Splits a compound query into its constituent clauses.
///
/// Tries, in order: (1) splitting on `?` when more than one is present, then
/// (2) each entry of [`COMPOUND_SEPARATORS`]. Returns a single-element vector
/// containing the trimmed whole query if no split point is found.
fn split_into_parts(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let question_marks = trimmed.matches('?').count();
    if question_marks > 1 {
        let parts: Vec<String> = trimmed
            .split('?')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!("{s}?"))
            .collect();
        if parts.len() >= 2 {
            return parts;
        }
    }

    let lower = trimmed.to_ascii_lowercase();
    for sep in COMPOUND_SEPARATORS {
        if lower.contains(sep) {
            let parts: Vec<String> = split_case_insensitive(trimmed, sep)
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                return parts;
            }
        }
    }

    vec![trimmed.to_string()]
}

// ── QueryRefiner ──────────────────────────────────────────────────────────────

/// Strategy interface for classifying and refining a query under the RQ-RAG
/// action taxonomy.
///
/// Implementations classify a query into a [`RefinementAction`] and, for
/// three of the four actions, produce the concrete refined query text(s).
/// [`RefinementAction::Respond`] needs no refiner-produced text: the engine
/// passes the original query straight through.
pub trait QueryRefiner: Send + Sync {
    /// Classifies `query` into a [`RefinementAction`].
    fn classify(&self, query: &str) -> RefinementAction;

    /// Produces a single rewritten query for [`RefinementAction::Rewrite`].
    fn rewrite(&self, query: &str) -> String;

    /// Produces one sub-query per part for [`RefinementAction::Decompose`].
    fn decompose(&self, query: &str) -> Vec<String>;

    /// Produces one variant per plausible reading for
    /// [`RefinementAction::Disambiguate`].
    fn disambiguate(&self, query: &str) -> Vec<String>;
}

// ── MockRefiner ───────────────────────────────────────────────────────────────

/// Deterministic, signal-based default implementation of [`QueryRefiner`].
///
/// `MockRefiner` requires no LLM or external service: it classifies and
/// refines queries purely from the marker word tables carried in its
/// [`RqRagConfig`]. See the [module-level docs](self) for the exact
/// classification algorithm.
#[derive(Debug, Clone)]
pub struct MockRefiner {
    config: RqRagConfig,
}

impl MockRefiner {
    /// Creates a new [`MockRefiner`] driven by `config`.
    #[must_use]
    pub fn new(config: RqRagConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration this refiner was constructed with.
    #[must_use]
    pub fn config(&self) -> &RqRagConfig {
        &self.config
    }
}

impl Default for MockRefiner {
    fn default() -> Self {
        Self::new(RqRagConfig::default())
    }
}

impl QueryRefiner for MockRefiner {
    fn classify(&self, query: &str) -> RefinementAction {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return RefinementAction::Respond;
        }

        let min_parts = self.config.min_decompose_parts.max(2);
        if split_into_parts(trimmed).len() >= min_parts {
            return RefinementAction::Decompose;
        }

        if self
            .config
            .ambiguity_markers
            .iter()
            .any(|marker| marker_present(trimmed, marker))
        {
            return RefinementAction::Disambiguate;
        }

        if self
            .config
            .colloquial_markers
            .iter()
            .any(|marker| marker_present(trimmed, marker))
        {
            return RefinementAction::Rewrite;
        }

        RefinementAction::Respond
    }

    fn rewrite(&self, query: &str) -> String {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        let mut working = trimmed.to_ascii_lowercase();
        for marker in &self.config.colloquial_markers {
            let marker_lower = marker.to_ascii_lowercase();
            working = remove_all_marker_occurrences(&working, &marker_lower);
        }

        let collapsed = collapse_whitespace(&working);
        let cleaned = strip_leading_punctuation(&collapsed);
        capitalize_first(&cleaned)
    }

    fn decompose(&self, query: &str) -> Vec<String> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        let parts = split_into_parts(trimmed);
        if parts.is_empty() {
            vec![trimmed.to_string()]
        } else {
            parts
        }
    }

    fn disambiguate(&self, query: &str) -> Vec<String> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        for marker in &self.config.ambiguity_markers {
            if marker_present(trimmed, marker) {
                let marker_lower = marker.to_ascii_lowercase();
                if let Some((_, senses)) = AMBIGUITY_SENSE_TABLE
                    .iter()
                    .find(|(term, _)| term.eq_ignore_ascii_case(&marker_lower))
                {
                    return senses
                        .iter()
                        .map(|sense| format!("{trimmed} (sense: {sense})"))
                        .collect();
                }
                // A user-supplied marker with no built-in sense entry: emit
                // two generic reading variants so the action still produces
                // *multiple* candidates, as RQ-RAG's Disambiguate requires.
                return vec![
                    format!("{trimmed} (interpreting \"{marker}\" as a proper noun)"),
                    format!("{trimmed} (interpreting \"{marker}\" as a common noun)"),
                ];
            }
        }

        vec![trimmed.to_string()]
    }
}

// ── QueryRefinementEngine ─────────────────────────────────────────────────────

/// Orchestrates classify-then-refine query processing under a
/// [`QueryRefiner`] strategy.
///
/// # Type parameter
///
/// `R` is any [`QueryRefiner`] implementation. Use
/// [`crate::rq_rag::MockRefiner`] for the deterministic built-in heuristic.
///
/// # Example
///
/// ```
/// use oxirag::rq_rag::{MockRefiner, QueryRefinementEngine, RefinementAction, RqRagConfig};
///
/// let engine = QueryRefinementEngine::new(RqRagConfig::default(), MockRefiner::default());
///
/// let plan = engine.run("What is Rust and what is Python?").unwrap();
/// assert_eq!(plan.action, RefinementAction::Decompose);
/// assert!(plan.refined_queries.len() >= 2);
/// ```
pub struct QueryRefinementEngine<R: QueryRefiner> {
    /// Configuration governing decompose thresholds and validity bounds.
    config: RqRagConfig,
    /// The underlying classify-and-refine strategy.
    refiner: R,
}

impl<R: QueryRefiner> QueryRefinementEngine<R> {
    /// Creates a new [`QueryRefinementEngine`] with the given configuration
    /// and refiner.
    #[must_use]
    pub fn new(config: RqRagConfig, refiner: R) -> Self {
        Self { config, refiner }
    }

    /// Replaces the configuration, returning `self` for chaining.
    #[must_use]
    pub fn with_config(mut self, config: RqRagConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &RqRagConfig {
        &self.config
    }

    /// Returns a reference to the underlying refiner.
    #[must_use]
    pub fn refiner(&self) -> &R {
        &self.refiner
    }

    /// Classifies `query` and dispatches to the matching refiner strategy,
    /// returning a complete [`RefinementPlan`].
    ///
    /// # Errors
    ///
    /// - [`RqRagError::InvalidConfig`] if the engine's [`RqRagConfig`] fails
    ///   [`RqRagConfig::validate`].
    /// - [`RqRagError::EmptyQuery`] if `query` trims to empty.
    pub fn run(&self, query: &str) -> Result<RefinementPlan, RqRagError> {
        self.config.validate()?;

        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(RqRagError::EmptyQuery);
        }
        let original_query = trimmed.to_string();

        let action = self.refiner.classify(trimmed);
        let refined_queries = match action {
            RefinementAction::Respond => vec![original_query.clone()],
            RefinementAction::Rewrite => vec![self.refiner.rewrite(trimmed)],
            RefinementAction::Decompose => {
                let parts = self.refiner.decompose(trimmed);
                if parts.is_empty() {
                    vec![original_query.clone()]
                } else {
                    parts
                }
            }
            RefinementAction::Disambiguate => {
                let variants = self.refiner.disambiguate(trimmed);
                if variants.is_empty() {
                    vec![original_query.clone()]
                } else {
                    variants
                }
            }
        };

        Ok(RefinementPlan {
            original_query,
            action,
            refined_queries,
        })
    }
}
