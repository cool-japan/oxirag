//! Types and traits for the `constitutional_critique` module.
//!
//! Constitutional AI (Bai et al. 2022) walks a **fixed, named** list of
//! [`ConstitutionalPrinciple`]s over a draft, one at a time: for each
//! principle, a [`ConstitutionalReviser`] first critiques the *current*
//! draft against that single principle, producing a
//! [`ConstitutionalCritique`]; when the critique flags a violation, the
//! reviser revises the draft to address it, producing a
//! [`ConstitutionalRevision`] that becomes the current draft for every
//! later principle in the list.
//!
//! Each [`ConstitutionalPrinciple`] carries its own set of
//! [`ConstitutionalTrigger`]s — deterministic heuristics the principle
//! watches for: literal keyword/phrase matches, absolute-language phrases
//! paired with a hedged replacement, or hand-rolled scanning for
//! PII-shaped tokens (email/phone/SSN patterns — no regex crate involved).
//! A critique's [`matches`](ConstitutionalCritique::matches) record not
//! just *what* triggered but *what it should become*, so that
//! [`ConstitutionalReviser::revise`] — which sees only the draft and the
//! critique, never the original principle — has everything it needs to
//! apply a targeted, principle-specific fix.
//!
//! [`ConstitutionalPrinciple::default_constitution`] provides a small
//! built-in constitution; [`MockConstitutionalReviser`] provides a fully
//! deterministic reference implementation of [`ConstitutionalReviser`] for
//! use without a live model.

use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalTrigger
// ═══════════════════════════════════════════════════════════════════════

/// A deterministic heuristic a [`ConstitutionalPrinciple`] watches for
/// within a draft.
///
/// Each variant pairs a detection rule with the fix it implies, so a
/// [`ConstitutionalReviser`] can compute both "did this fire" and "what
/// should replace it" in a single pass over the draft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstitutionalTrigger {
    /// A case-insensitive keyword or phrase substring match.
    ///
    /// On violation, every matched occurrence is replaced with a neutral
    /// `[redacted]` marker.
    Keyword {
        /// The keyword or phrase to match, case-insensitively, as a
        /// substring of the draft.
        word: String,
    },
    /// A case-insensitive absolute or overconfident phrase, paired with a
    /// softened replacement.
    ///
    /// On violation, every matched occurrence of `phrase` is replaced with
    /// `hedge`.
    Absolute {
        /// The absolute/overconfident phrase to match, case-insensitively.
        phrase: String,
        /// The hedged phrase substituted in place of every match.
        hedge: String,
    },
    /// Hand-rolled structural scanning for PII-shaped tokens: email-,
    /// phone-, and SSN-shaped character runs. No external regex crate is
    /// used — matching is a char-class state machine, mirroring the
    /// `guardrails` module's approach but owned independently by this
    /// module.
    ///
    /// On violation, every matched span is redacted with a
    /// `[REDACTED-<KIND>]` marker.
    PiiShaped,
}

impl ConstitutionalTrigger {
    /// Construct a [`ConstitutionalTrigger::Keyword`] trigger.
    #[must_use]
    pub fn keyword(word: impl Into<String>) -> Self {
        Self::Keyword { word: word.into() }
    }

    /// Construct a [`ConstitutionalTrigger::Absolute`] trigger.
    #[must_use]
    pub fn absolute(phrase: impl Into<String>, hedge: impl Into<String>) -> Self {
        Self::Absolute {
            phrase: phrase.into(),
            hedge: hedge.into(),
        }
    }

    /// Construct a [`ConstitutionalTrigger::PiiShaped`] trigger.
    #[must_use]
    pub fn pii_shaped() -> Self {
        Self::PiiShaped
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalMatch
// ═══════════════════════════════════════════════════════════════════════

/// A single triggered span within a draft, produced during critique.
///
/// Carries not only *what* matched but *what it should become*, so that
/// [`ConstitutionalReviser::revise`] — which only sees the draft and the
/// critique, not the original [`ConstitutionalPrinciple`] — has everything
/// it needs to apply a targeted fix without re-deriving the principle's
/// trigger definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalMatch {
    /// Human-readable description of which trigger fired, e.g.
    /// `"keyword: kill"`, `"absolute: definitely"`, or
    /// `"pii-shaped: email"`.
    pub trigger: String,
    /// Byte offset of the match start within the draft that was critiqued.
    pub start: usize,
    /// Byte offset of the match end (exclusive) within the draft that was
    /// critiqued.
    pub end: usize,
    /// The exact substring of the draft that matched.
    pub matched_text: String,
    /// The text that should replace `matched_text` to address this
    /// trigger.
    pub replacement: String,
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalPrinciple
// ═══════════════════════════════════════════════════════════════════════

/// A single named principle in a constitution: an id, a human-readable
/// statement, and the [`ConstitutionalTrigger`]s it watches for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalPrinciple {
    /// Short, stable identifier for this principle (e.g. `"avoid_harm"`).
    pub id: String,
    /// Human-readable statement of the principle.
    pub statement: String,
    /// The deterministic heuristics that detect a violation of this
    /// principle.
    pub triggers: Vec<ConstitutionalTrigger>,
}

/// Harm-related keyword triggers for [`ConstitutionalPrinciple::avoid_harm`].
static HARM_KEYWORDS: &[&str] = &[
    "kill",
    "murder",
    "weapon",
    "poison",
    "bomb",
    "torture",
    "self-harm",
    "suicide",
];

/// Insult/derogatory keyword triggers for
/// [`ConstitutionalPrinciple::be_respectful`].
static INSULT_KEYWORDS: &[&str] = &[
    "idiot",
    "stupid",
    "moron",
    "worthless",
    "pathetic",
    "dumbass",
];

/// Illegal-activity phrase triggers for
/// [`ConstitutionalPrinciple::avoid_illegal_activity`].
static ILLEGAL_ACTIVITY_PHRASES: &[&str] = &[
    "how to hack into",
    "how to steal",
    "launder money",
    "evade taxes",
    "counterfeit currency",
    "buy illegal drugs",
];

/// `(absolute phrase, hedged replacement)` pairs for
/// [`ConstitutionalPrinciple::be_truthful`].
static ABSOLUTE_LANGUAGE: &[(&str, &str)] = &[
    ("definitely", "likely"),
    ("always", "often"),
    ("never", "rarely"),
    ("guaranteed", "likely"),
    ("proven", "suggested"),
    ("certainly", "probably"),
    ("undeniably", "arguably"),
    ("100% effective", "highly effective"),
];

impl ConstitutionalPrinciple {
    /// Create a new principle with no triggers.
    #[must_use]
    pub fn new(id: impl Into<String>, statement: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            statement: statement.into(),
            triggers: Vec::new(),
        }
    }

    /// Append a single trigger.
    #[must_use]
    pub fn with_trigger(mut self, trigger: ConstitutionalTrigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Append every trigger in `triggers`.
    #[must_use]
    pub fn with_triggers(
        mut self,
        triggers: impl IntoIterator<Item = ConstitutionalTrigger>,
    ) -> Self {
        self.triggers.extend(triggers);
        self
    }

    /// The built-in default constitution: five principles spanning every
    /// [`ConstitutionalTrigger`] heuristic, in order —
    /// [`ConstitutionalPrinciple::avoid_harm`],
    /// [`ConstitutionalPrinciple::be_truthful`],
    /// [`ConstitutionalPrinciple::respect_privacy`],
    /// [`ConstitutionalPrinciple::be_respectful`], and
    /// [`ConstitutionalPrinciple::avoid_illegal_activity`].
    ///
    /// Callers are free to ignore this and build a fully custom
    /// constitution from scratch with [`ConstitutionalPrinciple::new`].
    #[must_use]
    pub fn default_constitution() -> Vec<Self> {
        vec![
            Self::avoid_harm(),
            Self::be_truthful(),
            Self::respect_privacy(),
            Self::be_respectful(),
            Self::avoid_illegal_activity(),
        ]
    }

    /// "Avoid harm" — flags a configurable list of harm-related keywords
    /// (e.g. `"kill"`, `"weapon"`, `"suicide"`).
    #[must_use]
    pub fn avoid_harm() -> Self {
        Self::new(
            "avoid_harm",
            "Avoid content that could cause physical, psychological, or societal harm.",
        )
        .with_triggers(
            HARM_KEYWORDS
                .iter()
                .map(|word| ConstitutionalTrigger::keyword(*word)),
        )
    }

    /// "Be truthful" — flags absolute or overconfident language (e.g.
    /// `"definitely"`, `"always"`, `"guaranteed"`) and softens it to a
    /// calibrated hedge on revision.
    #[must_use]
    pub fn be_truthful() -> Self {
        Self::new(
            "be_truthful",
            "Avoid absolute or overconfident claims; prefer calibrated, hedged language.",
        )
        .with_triggers(
            ABSOLUTE_LANGUAGE
                .iter()
                .map(|(phrase, hedge)| ConstitutionalTrigger::absolute(*phrase, *hedge)),
        )
    }

    /// "Respect privacy" — flags hand-rolled PII-shaped tokens (email-,
    /// phone-, and SSN-shaped character runs).
    #[must_use]
    pub fn respect_privacy() -> Self {
        Self::new(
            "respect_privacy",
            "Do not include personally identifiable information such as email addresses, \
             phone numbers, or government ID numbers.",
        )
        .with_trigger(ConstitutionalTrigger::pii_shaped())
    }

    /// "Be respectful" — flags a configurable list of insulting or
    /// derogatory keywords.
    #[must_use]
    pub fn be_respectful() -> Self {
        Self::new(
            "be_respectful",
            "Avoid insulting, demeaning, or derogatory language toward individuals or groups.",
        )
        .with_triggers(
            INSULT_KEYWORDS
                .iter()
                .map(|word| ConstitutionalTrigger::keyword(*word)),
        )
    }

    /// "Avoid illegal activity" — flags a configurable list of
    /// illegal-activity phrases.
    #[must_use]
    pub fn avoid_illegal_activity() -> Self {
        Self::new(
            "avoid_illegal_activity",
            "Do not provide instructions or encouragement for illegal activity.",
        )
        .with_triggers(
            ILLEGAL_ACTIVITY_PHRASES
                .iter()
                .map(|word| ConstitutionalTrigger::keyword(*word)),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalCritique
// ═══════════════════════════════════════════════════════════════════════

/// The result of critiquing a draft against a single
/// [`ConstitutionalPrinciple`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalCritique {
    /// The id of the [`ConstitutionalPrinciple`] this critique was
    /// produced against.
    pub principle_id: String,
    /// Whether the principle was violated.
    pub violated: bool,
    /// Human-readable explanation of the critique's verdict.
    pub explanation: String,
    /// The specific triggered spans that drove the `violated` decision,
    /// sorted left-to-right with overlaps removed. Empty when `violated`
    /// is `false`.
    pub matches: Vec<ConstitutionalMatch>,
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalRevision
// ═══════════════════════════════════════════════════════════════════════

/// A targeted revision produced from a single [`ConstitutionalCritique`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalRevision {
    /// The id of the [`ConstitutionalPrinciple`] this revision addresses.
    pub principle_id: String,
    /// The draft text before this revision was applied.
    pub before: String,
    /// The draft text after this revision was applied.
    pub after: String,
    /// Human-readable summary of what changed and why.
    pub notes: String,
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalReviser
// ═══════════════════════════════════════════════════════════════════════

/// Critiques a draft against a single [`ConstitutionalPrinciple`] and, when
/// violated, revises the draft to address that critique.
///
/// Implementations are **pure sync** — no I/O, no async — mirroring the
/// `Refiner`/`Retriever`/`Generator` executor traits used elsewhere in this
/// crate. The caller supplies a concrete reviser (e.g. wrapping an LLM
/// prompted with the principle's statement); [`MockConstitutionalReviser`]
/// provides a fully deterministic, heuristic implementation for use
/// without a live model.
pub trait ConstitutionalReviser {
    /// Critique `draft` against a single `principle`.
    ///
    /// Returns a [`ConstitutionalCritique`] whose
    /// [`violated`](ConstitutionalCritique::violated) flag records whether
    /// `principle` was violated, together with a human-readable
    /// [`explanation`](ConstitutionalCritique::explanation) and the
    /// specific [`matches`](ConstitutionalCritique::matches) that drove
    /// the decision.
    ///
    /// # Errors
    ///
    /// Implementations should return an error when they cannot produce a
    /// meaningful critique, e.g. [`ConstitutionalError::EmptyPrincipleId`]
    /// when `principle.id` is empty after trimming.
    fn critique(
        &self,
        draft: &str,
        principle: &ConstitutionalPrinciple,
    ) -> Result<ConstitutionalCritique, ConstitutionalError>;

    /// Revise `draft` to address `critique`.
    ///
    /// Callers should only invoke this when
    /// [`critique.violated`](ConstitutionalCritique::violated) is `true`;
    /// implementations should treat a non-violating critique as a caller
    /// error rather than silently fabricating a no-op revision.
    ///
    /// # Errors
    ///
    /// Implementations should return
    /// [`ConstitutionalError::NoViolationToRevise`] when `critique` did
    /// not flag a violation, and
    /// [`ConstitutionalError::InconsistentCritique`] when `critique`
    /// claims a violation but carries no usable matches.
    fn revise(
        &self,
        draft: &str,
        critique: &ConstitutionalCritique,
    ) -> Result<ConstitutionalRevision, ConstitutionalError>;
}

// ═══════════════════════════════════════════════════════════════════════
// MockConstitutionalReviser
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic [`ConstitutionalReviser`] for use without a live model.
///
/// Unlike a scripted mock, every method here performs genuine text
/// analysis: [`critique`](ConstitutionalReviser::critique) scans the draft
/// against the principle's own [`ConstitutionalTrigger`] definitions using
/// hand-rolled char-class matching (no regex), and
/// [`revise`](ConstitutionalReviser::revise) mechanically applies the
/// per-match `replacement` recorded on each [`ConstitutionalMatch`] —
/// stripping/redacting a keyword, substituting a hedge phrase, or
/// redacting a PII-shaped span, depending on which trigger fired.
#[derive(Debug, Clone, Default)]
pub struct MockConstitutionalReviser;

impl MockConstitutionalReviser {
    /// Create a new [`MockConstitutionalReviser`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConstitutionalReviser for MockConstitutionalReviser {
    fn critique(
        &self,
        draft: &str,
        principle: &ConstitutionalPrinciple,
    ) -> Result<ConstitutionalCritique, ConstitutionalError> {
        if principle.id.trim().is_empty() {
            return Err(ConstitutionalError::EmptyPrincipleId);
        }

        let matches = scan_principle(draft, principle);
        let violated = !matches.is_empty();
        let explanation = if violated {
            let labels: Vec<&str> = matches.iter().map(|item| item.trigger.as_str()).collect();
            format!(
                "principle '{}' ({}) flagged {} issue(s): {}",
                principle.id,
                principle.statement,
                matches.len(),
                labels.join("; ")
            )
        } else {
            format!(
                "draft complies with principle '{}' ({})",
                principle.id, principle.statement
            )
        };

        Ok(ConstitutionalCritique {
            principle_id: principle.id.clone(),
            violated,
            explanation,
            matches,
        })
    }

    fn revise(
        &self,
        draft: &str,
        critique: &ConstitutionalCritique,
    ) -> Result<ConstitutionalRevision, ConstitutionalError> {
        if !critique.violated {
            return Err(ConstitutionalError::NoViolationToRevise {
                principle_id: critique.principle_id.clone(),
            });
        }
        if critique.matches.is_empty() {
            return Err(ConstitutionalError::InconsistentCritique {
                principle_id: critique.principle_id.clone(),
            });
        }

        let mut sorted_matches = critique.matches.clone();
        sorted_matches.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));

        let mut after = String::with_capacity(draft.len());
        let mut cursor = 0usize;
        let mut applied = 0usize;
        for applied_match in &sorted_matches {
            if applied_match.start < cursor
                || applied_match.end > draft.len()
                || applied_match.start > applied_match.end
            {
                // Defensively skip overlapping or out-of-range matches
                // rather than panicking on a malformed critique.
                continue;
            }
            let Some(prefix) = draft.get(cursor..applied_match.start) else {
                continue;
            };
            after.push_str(prefix);
            after.push_str(&applied_match.replacement);
            cursor = applied_match.end;
            applied += 1;
        }

        let Some(suffix) = draft.get(cursor..) else {
            return Err(ConstitutionalError::InconsistentCritique {
                principle_id: critique.principle_id.clone(),
            });
        };
        after.push_str(suffix);

        if applied == 0 {
            return Err(ConstitutionalError::InconsistentCritique {
                principle_id: critique.principle_id.clone(),
            });
        }

        let notes = format!(
            "applied {applied} fix(es) for principle '{}': {}",
            critique.principle_id,
            sorted_matches
                .iter()
                .map(|item| item.trigger.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(ConstitutionalRevision {
            principle_id: critique.principle_id.clone(),
            before: draft.to_string(),
            after,
            notes,
        })
    }
}

// ── heuristic scanning (private) ─────────────────────────────────────────
//
// Hand-rolled char-class scanning, deliberately self-contained (not
// imported from `guardrails`, which is a separate optional feature this
// module does not depend on): this module owns its own small lexical
// toolkit rather than sharing private helpers across module/feature
// boundaries.

/// Neutral marker substituted for every keyword-trigger match.
const KEYWORD_REDACTION: &str = "[redacted]";

/// Build a `(chars, byte_offsets)` pair for `text`: `byte_offsets[i]` is
/// the byte offset of `chars[i]`, with a sentinel final entry equal to
/// `text.len()`.
fn char_index(text: &str) -> (Vec<char>, Vec<usize>) {
    let chars: Vec<char> = text.chars().collect();
    let mut byte_offsets = Vec::with_capacity(chars.len() + 1);
    let mut offset = 0usize;
    for ch in &chars {
        byte_offsets.push(offset);
        offset += ch.len_utf8();
    }
    byte_offsets.push(offset);
    (chars, byte_offsets)
}

/// Case-insensitive char equality: ASCII fast path, Unicode-aware fallback.
fn chars_eq_ignore_case(left: char, right: char) -> bool {
    if left.is_ascii() && right.is_ascii() {
        left.eq_ignore_ascii_case(&right)
    } else {
        left.to_lowercase().eq(right.to_lowercase())
    }
}

/// Find every non-overlapping, case-insensitive occurrence of `needle`
/// within `chars`, returning `(start_char_idx, end_char_idx)` pairs in
/// ascending order.
fn find_all_char_spans(chars: &[char], needle: &str) -> Vec<(usize, usize)> {
    let needle_chars: Vec<char> = needle.chars().collect();
    let needle_len = needle_chars.len();
    let hay_len = chars.len();
    if needle_len == 0 || needle_len > hay_len {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut i = 0usize;
    while i + needle_len <= hay_len {
        let is_match = (0..needle_len)
            .all(|offset| chars_eq_ignore_case(chars[i + offset], needle_chars[offset]));
        if is_match {
            spans.push((i, i + needle_len));
            i += needle_len;
        } else {
            i += 1;
        }
    }
    spans
}

/// Scan for every occurrence of a [`ConstitutionalTrigger::Keyword`].
fn scan_keyword(chars: &[char], byte_offsets: &[usize], word: &str) -> Vec<ConstitutionalMatch> {
    find_all_char_spans(chars, word)
        .into_iter()
        .map(|(start_idx, end_idx)| ConstitutionalMatch {
            trigger: format!("keyword: {word}"),
            start: byte_offsets[start_idx],
            end: byte_offsets[end_idx],
            matched_text: chars[start_idx..end_idx].iter().collect(),
            replacement: KEYWORD_REDACTION.to_string(),
        })
        .collect()
}

/// Scan for every occurrence of a [`ConstitutionalTrigger::Absolute`]
/// phrase.
fn scan_absolute(
    chars: &[char],
    byte_offsets: &[usize],
    phrase: &str,
    hedge: &str,
) -> Vec<ConstitutionalMatch> {
    find_all_char_spans(chars, phrase)
        .into_iter()
        .map(|(start_idx, end_idx)| ConstitutionalMatch {
            trigger: format!("absolute: {phrase}"),
            start: byte_offsets[start_idx],
            end: byte_offsets[end_idx],
            matched_text: chars[start_idx..end_idx].iter().collect(),
            replacement: hedge.to_string(),
        })
        .collect()
}

/// Return `true` when `ch` is a valid email local-part/domain character.
fn is_email_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | '+')
}

/// Scan for email-shaped tokens: `local@domain.tld`.
fn scan_pii_email(chars: &[char], byte_offsets: &[usize]) -> Vec<ConstitutionalMatch> {
    let len = chars.len();
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos < len {
        let mut local_end = pos;
        while local_end < len && is_email_char(chars[local_end]) {
            local_end += 1;
        }
        if local_end > pos && local_end < len && chars[local_end] == '@' {
            let domain_start = local_end + 1;
            let mut domain_end = domain_start;
            while domain_end < len && is_email_char(chars[domain_end]) {
                domain_end += 1;
            }
            let domain: String = chars[domain_start..domain_end].iter().collect();
            if domain.contains('.') && domain_end > domain_start + 2 {
                matches.push(ConstitutionalMatch {
                    trigger: "pii-shaped: email".to_string(),
                    start: byte_offsets[pos],
                    end: byte_offsets[domain_end],
                    matched_text: chars[pos..domain_end].iter().collect(),
                    replacement: "[REDACTED-EMAIL]".to_string(),
                });
                pos = domain_end;
                continue;
            }
        }
        pos += 1;
    }
    matches
}

/// Scan for phone-shaped tokens: 10-15 digits with `-`, `.`, ` `, `(`, `)`,
/// or a leading `+` as separators.
fn scan_pii_phone(chars: &[char], byte_offsets: &[usize]) -> Vec<ConstitutionalMatch> {
    let len = chars.len();
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos < len {
        if !chars[pos].is_ascii_digit() && chars[pos] != '+' {
            pos += 1;
            continue;
        }
        let start = pos;
        let mut digit_count = 0usize;
        let mut cursor = pos;
        while cursor < len {
            let current = chars[cursor];
            if current.is_ascii_digit() {
                digit_count += 1;
                cursor += 1;
            } else if (matches!(current, '-' | '.' | ' ' | '(' | ')') && digit_count > 0)
                || (current == '+' && cursor == start)
            {
                cursor += 1;
            } else {
                break;
            }
        }
        if (10..=15).contains(&digit_count) {
            matches.push(ConstitutionalMatch {
                trigger: "pii-shaped: phone".to_string(),
                start: byte_offsets[start],
                end: byte_offsets[cursor],
                matched_text: chars[start..cursor].iter().collect(),
                replacement: "[REDACTED-PHONE]".to_string(),
            });
            pos = cursor;
        } else {
            pos += 1;
        }
    }
    matches
}

/// Scan for U.S. SSN-shaped tokens: `XXX-XX-XXXX` or `XXX XX XXXX`.
fn scan_pii_ssn(chars: &[char], byte_offsets: &[usize]) -> Vec<ConstitutionalMatch> {
    let len = chars.len();
    let mut matches = Vec::new();
    let mut pos = 0usize;
    while pos + 11 <= len {
        let sep = chars[pos + 3];
        let looks_like_ssn = (sep == '-' || sep == ' ')
            && chars[pos + 6] == sep
            && chars[pos..pos + 3].iter().all(char::is_ascii_digit)
            && chars[pos + 4..pos + 6].iter().all(char::is_ascii_digit)
            && chars[pos + 7..pos + 11].iter().all(char::is_ascii_digit);
        if looks_like_ssn {
            matches.push(ConstitutionalMatch {
                trigger: "pii-shaped: ssn".to_string(),
                start: byte_offsets[pos],
                end: byte_offsets[pos + 11],
                matched_text: chars[pos..pos + 11].iter().collect(),
                replacement: "[REDACTED-SSN]".to_string(),
            });
            pos += 11;
        } else {
            pos += 1;
        }
    }
    matches
}

/// Scan for every [`ConstitutionalTrigger::PiiShaped`] token: email,
/// phone, and SSN shapes combined.
fn scan_pii_shaped(chars: &[char], byte_offsets: &[usize]) -> Vec<ConstitutionalMatch> {
    let mut matches = scan_pii_email(chars, byte_offsets);
    matches.extend(scan_pii_phone(chars, byte_offsets));
    matches.extend(scan_pii_ssn(chars, byte_offsets));
    matches
}

/// Drop any match whose span overlaps an earlier (lower-`start`) match
/// already kept. `matches` must already be sorted by `(start, end)`.
fn dedup_overlaps(matches: Vec<ConstitutionalMatch>) -> Vec<ConstitutionalMatch> {
    let mut result = Vec::with_capacity(matches.len());
    let mut cursor = 0usize;
    for candidate in matches {
        if candidate.start >= cursor {
            cursor = candidate.end;
            result.push(candidate);
        }
    }
    result
}

/// Scan `draft` against every trigger declared on `principle`, returning
/// all matches sorted left-to-right with overlaps removed.
fn scan_principle(draft: &str, principle: &ConstitutionalPrinciple) -> Vec<ConstitutionalMatch> {
    let (chars, byte_offsets) = char_index(draft);
    let mut matches = Vec::new();
    for trigger in &principle.triggers {
        match trigger {
            ConstitutionalTrigger::Keyword { word } => {
                matches.extend(scan_keyword(&chars, &byte_offsets, word));
            }
            ConstitutionalTrigger::Absolute { phrase, hedge } => {
                matches.extend(scan_absolute(&chars, &byte_offsets, phrase, hedge));
            }
            ConstitutionalTrigger::PiiShaped => {
                matches.extend(scan_pii_shaped(&chars, &byte_offsets));
            }
        }
    }
    matches.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
    dedup_overlaps(matches)
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalConfig
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for [`crate::constitutional_critique::ConstitutionalEngine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalConfig {
    /// If `Some`, only principles whose id is contained in this list are
    /// applied; every other principle passed to
    /// [`ConstitutionalEngine::run`](crate::constitutional_critique::ConstitutionalEngine::run)
    /// is skipped entirely (no critique is run, no trace entry is
    /// produced). `None` (the default) applies every principle passed in.
    pub enabled_principle_ids: Option<Vec<String>>,
    /// Maximum number of revisions accepted across the whole run.
    ///
    /// Once this many revisions have been applied, later principles are
    /// still critiqued — so the trace stays complete — but a violation is
    /// no longer revised; the draft stays frozen at the last accepted
    /// revision. Defaults to `10`.
    pub max_revisions: usize,
    /// If `Some(n)` with `n > 0`, stop processing further principles as
    /// soon as `n` consecutive principles (in list order) find no
    /// violation. A capped-but-unrevised violation resets the streak, the
    /// same as an actual violation. `None` (the default), or `Some(0)`,
    /// disables early stopping.
    pub stop_after_consecutive_clean: Option<usize>,
}

impl Default for ConstitutionalConfig {
    fn default() -> Self {
        Self {
            enabled_principle_ids: None,
            max_revisions: 10,
            stop_after_consecutive_clean: None,
        }
    }
}

impl ConstitutionalConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict which principles are applied, by id. Passing an empty
    /// iterator disables every principle.
    #[must_use]
    pub fn with_enabled_principle_ids<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enabled_principle_ids = Some(ids.into_iter().map(Into::into).collect());
        self
    }

    /// Set the maximum number of revisions accepted across the whole run.
    #[must_use]
    pub fn with_max_revisions(mut self, max_revisions: usize) -> Self {
        self.max_revisions = max_revisions;
        self
    }

    /// Set the number of consecutive clean principles that triggers an
    /// early stop.
    #[must_use]
    pub fn with_stop_after_consecutive_clean(mut self, threshold: usize) -> Self {
        self.stop_after_consecutive_clean = Some(threshold);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalTraceEntry / ConstitutionalResult
// ═══════════════════════════════════════════════════════════════════════

/// One entry in a constitutional-critique trace: the principle applied,
/// the critique produced against the draft as it stood at that point in
/// the run, and the resulting revision (`None` when no violation was
/// found, or when the violation was found but not revised because
/// [`ConstitutionalConfig::max_revisions`] had already been reached).
pub type ConstitutionalTraceEntry = (
    ConstitutionalPrinciple,
    ConstitutionalCritique,
    Option<ConstitutionalRevision>,
);

/// The complete output of a
/// [`crate::constitutional_critique::ConstitutionalEngine`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstitutionalResult {
    /// The draft as it was passed in, before any principle was applied.
    pub original_draft: String,
    /// The draft after every applicable principle has been critiqued and
    /// (where violated) revised.
    pub final_draft: String,
    /// The full per-principle trace, in the order principles were
    /// applied.
    pub trace: Vec<ConstitutionalTraceEntry>,
    /// Total number of revisions actually applied across the run.
    pub revisions_applied: usize,
    /// `true` when the run ended early because
    /// [`ConstitutionalConfig::stop_after_consecutive_clean`] was
    /// reached, rather than because every principle was processed.
    pub stopped_early: bool,
}

impl ConstitutionalResult {
    /// Return `true` when no revision changed the draft at all.
    #[must_use]
    pub fn is_unchanged(&self) -> bool {
        self.original_draft == self.final_draft
    }

    /// Return the ids of every principle whose critique flagged a
    /// violation, in trace order.
    #[must_use]
    pub fn violated_principle_ids(&self) -> Vec<String> {
        self.trace
            .iter()
            .filter(|(_, critique, _)| critique.violated)
            .map(|(principle, _, _)| principle.id.clone())
            .collect()
    }

    /// Return the ids of every principle that actually produced a
    /// revision, in trace order.
    #[must_use]
    pub fn revised_principle_ids(&self) -> Vec<String> {
        self.trace
            .iter()
            .filter(|(_, _, revision)| revision.is_some())
            .map(|(principle, _, _)| principle.id.clone())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstitutionalError
// ═══════════════════════════════════════════════════════════════════════

/// Errors from the `constitutional_critique` module.
#[derive(Debug, Error)]
pub enum ConstitutionalError {
    /// The draft was empty or contained only whitespace.
    #[error("draft must not be empty")]
    EmptyDraft,
    /// A principle's id was empty or contained only whitespace.
    #[error("principle id must not be empty")]
    EmptyPrincipleId,
    /// [`ConstitutionalReviser::revise`] was called with a critique that
    /// did not flag a violation.
    #[error("no violation to revise for principle '{principle_id}'")]
    NoViolationToRevise {
        /// The id of the principle whose (non-violating) critique was
        /// passed to `revise`.
        principle_id: String,
    },
    /// A critique claimed a violation but carried no usable matches to
    /// revise, or its matches could not be applied to the draft.
    #[error(
        "critique for principle '{principle_id}' is inconsistent: claims a violation but has no usable matches"
    )]
    InconsistentCritique {
        /// The id of the principle whose critique was inconsistent.
        principle_id: String,
    },
}
