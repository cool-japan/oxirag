//! [`DraginEngine`] — the DRAGIN dynamic retrieval loop (RIND + QFS).

use crate::dragin::types::{
    DraginConfig, DraginError, DraginTrace, RetrievalTrigger, Retriever, TokenInfo,
    UncertaintyGenerator,
};

/// Small stopword set used by RIND (to reject low-confidence function words)
/// and by QFS (to drop non-salient tokens). Kept deliberately compact and
/// lowercase; membership is tested case-insensitively.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "that", "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then",
    "than", "into", "some", "such", "only", "also", "been", "more", "very", "will", "would",
    "there", "their", "which", "about", "could", "these", "those",
];

// ── DraginEngine ──────────────────────────────────────────────────────────────

/// Drives the DRAGIN loop: an [`UncertaintyGenerator`] emits token segments with
/// per-token confidence; **RIND** scans each segment for the first uncertain
/// **content** token; on a trigger, **QFS** forms a query from the salient
/// recent tokens, a [`Retriever`] fetches passages, the passages are folded back
/// into the context, and generation resumes.
///
/// The generator and retriever are supplied *per call* through the generic
/// [`DraginEngine::run`] method, mirroring the caller-supplies-executor pattern
/// used across the crate.
#[derive(Debug, Clone, Default)]
pub struct DraginEngine {
    /// Configuration for this engine.
    pub config: DraginConfig,
}

impl DraginEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: DraginConfig) -> Self {
        Self { config }
    }

    /// Tokenise `text` on non-alphanumeric boundaries, discarding empty
    /// fragments. This is the canonical tokeniser used throughout DRAGIN.
    #[must_use]
    pub fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Decide whether `tok` is a **content** token.
    ///
    /// A content token carries information: it is at least three characters long
    /// and is not a member of the small stopword set. RIND only triggers on
    /// uncertain content tokens, and QFS only keeps content tokens.
    #[must_use]
    pub fn is_content_token(&self, tok: &str) -> bool {
        let trimmed = tok.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.chars().count() < 3 {
            return false;
        }
        let lower = trimmed.to_lowercase();
        !STOPWORDS.contains(&lower.as_str())
    }

    /// QFS — Query Formulation by Self-attention (content-token proxy).
    ///
    /// Takes the last `query_window` tokens of `context_tokens`, keeps the
    /// salient (content) tokens, and joins them with single spaces. The
    /// attention signal is approximated by content-token salience: stopwords and
    /// short function words are dropped. If the window contains no content
    /// tokens, the surviving window tokens are joined verbatim so the formed
    /// query is never empty when input was non-empty.
    #[must_use]
    pub fn formulate_query(&self, context_tokens: &[&str]) -> String {
        let window = self.config.query_window.max(1);
        let start = context_tokens.len().saturating_sub(window);
        let recent = &context_tokens[start..];

        let salient: Vec<&str> = recent
            .iter()
            .copied()
            .filter(|t| self.is_content_token(t))
            .collect();

        if salient.is_empty() {
            return recent.join(" ");
        }
        salient.join(" ")
    }

    /// Scan a generated `segment` for the first RIND trigger.
    ///
    /// Returns the `(position, token)` of the first token that is both uncertain
    /// (`confidence < uncertainty_threshold`) and a content token, or `None` if
    /// the segment contains no such token.
    fn first_trigger(&self, segment: &[TokenInfo]) -> Option<(usize, String)> {
        segment.iter().enumerate().find_map(|(idx, info)| {
            let uncertain = info.confidence < self.config.uncertainty_threshold;
            if uncertain && self.is_content_token(&info.token) {
                Some((idx, info.token.clone()))
            } else {
                None
            }
        })
    }

    /// Run the DRAGIN loop for `query`.
    ///
    /// Repeatedly: the generator produces a token segment from the current
    /// context; RIND scans it for the first uncertain content token. When a
    /// trigger fires and the [`max_retrievals`](DraginConfig::max_retrievals)
    /// cap has not been reached, QFS forms a query from the tokens generated so
    /// far (up to and including the trigger), the retriever is consulted, the
    /// trigger is recorded, and the retrieved passages are appended to the
    /// context before the next segment. The loop stops when a segment produces
    /// no trigger, when the generator returns an empty segment, or when the cap
    /// is hit. Generated text is accumulated across segments.
    ///
    /// # Errors
    ///
    /// Returns [`DraginError::EmptyQuery`] if `query` is empty after trimming.
    pub fn run<G, R>(
        &self,
        query: &str,
        generator: &G,
        retriever: &R,
    ) -> Result<DraginTrace, DraginError>
    where
        G: UncertaintyGenerator + ?Sized,
        R: Retriever + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(DraginError::EmptyQuery);
        }

        let mut triggers: Vec<RetrievalTrigger> = Vec::new();
        let mut generated = String::new();
        // The context the generator sees: the query, plus generated text, plus
        // any retrieved passages folded in.
        let mut context = query.to_string();

        loop {
            let segment = generator.generate(&context);
            if segment.is_empty() {
                break;
            }

            // Accumulate this segment's surface text into `generated` and the
            // working context, regardless of whether it triggers.
            let segment_text = join_tokens(&segment);
            append_with_space(&mut generated, &segment_text);
            append_with_space(&mut context, &segment_text);

            match self.first_trigger(&segment) {
                Some((position, trigger_token)) if triggers.len() < self.config.max_retrievals => {
                    // QFS over everything generated so far (content-token proxy
                    // for self-attention).
                    let generated_tokens = Self::tokenize(&generated);
                    let refs: Vec<&str> = generated_tokens.iter().map(String::as_str).collect();
                    let formed_query = self.formulate_query(&refs);

                    let passages = retriever.retrieve(&formed_query);

                    // Fold retrieved passages into the context for subsequent
                    // generation.
                    for passage in &passages {
                        append_with_space(&mut context, passage);
                    }

                    triggers.push(RetrievalTrigger {
                        position,
                        trigger_token,
                        formed_query,
                        retrieved: passages,
                    });
                }
                // Either no trigger, or the retrieval cap is reached: stop.
                _ => break,
            }
        }

        let num_retrievals = triggers.len();
        Ok(DraginTrace {
            triggers,
            generated,
            num_retrievals,
        })
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Join token surfaces with single spaces.
fn join_tokens(segment: &[TokenInfo]) -> String {
    segment
        .iter()
        .map(|t| t.token.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Append `addition` to `buffer`, inserting a single separating space when both
/// the buffer and the addition are non-empty.
fn append_with_space(buffer: &mut String, addition: &str) {
    if addition.is_empty() {
        return;
    }
    if !buffer.is_empty() {
        buffer.push(' ');
    }
    buffer.push_str(addition);
}
