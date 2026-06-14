//! Natural-language → structured-query parser.
//!
//! [`SelfQueryParser`] turns a free-form query such as
//! `"papers about transformers after 2020 by Vaswani rated above 4"` into a
//! [`ParsedFilter`] plus a residual semantic query (`"papers about
//! transformers"`).
//!
//! The parser is fully deterministic and uses only `std`: it tokenises on
//! non-alphanumeric boundaries, scans for a small set of trigger words, and
//! maps each recognised pattern to a [`FilterCondition`]. Matched word spans
//! are recorded so they can be removed from the semantic query when
//! [`SelfQueryConfig::strip_filter_phrases`] is enabled.

use super::types::{
    FieldType, FilterCondition, FilterOp, FilterSchema, ParsedFilter, SelfQueryConfig,
    SelfQueryError, StructuredQuery,
};
use crate::types::Document;

// ── Trigger word tables ───────────────────────────────────────────────────────

/// Triggers that introduce a year and map to a *greater-than* date comparison.
const AFTER_TRIGGERS: &[&str] = &["after", "since"];

/// Triggers that introduce a year and map to a *less-than* date comparison.
const BEFORE_TRIGGERS: &[&str] = &["before", "until"];

/// Triggers that introduce a year and map to an *equality* date comparison.
const IN_TRIGGERS: &[&str] = &["in", "during"];

/// Triggers that introduce an author/person name (equality on a text field).
const BY_TRIGGERS: &[&str] = &["by", "authored", "author"];

/// Triggers that introduce a number with *greater-than* semantics.
const ABOVE_TRIGGERS: &[&str] = &["above", "over", "rated", "rating"];

/// Triggers that introduce a number with *less-than* semantics.
const BELOW_TRIGGERS: &[&str] = &["below", "under"];

/// Triggers that introduce a tag/category value.
const TAG_TRIGGERS: &[&str] = &["tagged", "tag", "category", "categorized"];

// ── Token ─────────────────────────────────────────────────────────────────────

/// A single token with its byte span in the original (untrimmed) query.
#[derive(Debug, Clone)]
struct Token {
    /// Lower-cased text of the token.
    lower: String,
    /// Original-cased text of the token.
    raw: String,
    /// Byte offset of the token start in the source string.
    start: usize,
    /// Byte offset of the token end (exclusive) in the source string.
    end: usize,
}

/// Tokenise `text` on non-alphanumeric boundaries, preserving spans.
fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            let raw = &text[s..idx];
            tokens.push(Token {
                lower: raw.to_lowercase(),
                raw: raw.to_string(),
                start: s,
                end: idx,
            });
        }
    }
    if let Some(s) = start.take() {
        let raw = &text[s..];
        tokens.push(Token {
            lower: raw.to_lowercase(),
            raw: raw.to_string(),
            start: s,
            end: text.len(),
        });
    }
    tokens
}

/// Return `true` if `token` looks like a four-digit year.
fn is_year(token: &str) -> bool {
    token.len() == 4 && token.bytes().all(|b| b.is_ascii_digit())
}

/// Return `true` if `token` parses as an `f64` number.
fn is_number(token: &str) -> bool {
    token.parse::<f64>().is_ok()
}

// ── SelfQueryParser ───────────────────────────────────────────────────────────

/// Parses natural-language queries into [`StructuredQuery`] values using a
/// configurable [`FilterSchema`].
#[derive(Debug, Clone)]
pub struct SelfQueryParser {
    /// Known metadata fields.
    pub schema: FilterSchema,
    /// Parser behaviour configuration.
    pub config: SelfQueryConfig,
}

impl SelfQueryParser {
    /// Create a parser with the given schema and default configuration.
    #[must_use]
    pub fn new(schema: FilterSchema) -> Self {
        Self {
            schema,
            config: SelfQueryConfig::default(),
        }
    }

    /// Create a parser with an explicit configuration (builder).
    #[must_use]
    pub fn with_config(mut self, config: SelfQueryConfig) -> Self {
        self.config = config;
        self
    }

    /// Resolve the target field name for a date pattern: the field explicitly
    /// named by `field_token` if it is a known [`FieldType::Date`] field,
    /// otherwise the first date field in the schema.
    fn date_field(&self) -> Option<String> {
        self.schema
            .first_of_type(FieldType::Date)
            .map(|spec| spec.name.clone())
    }

    /// Resolve the target field name for a number pattern: first
    /// [`FieldType::Number`] field in the schema.
    fn number_field(&self) -> Option<String> {
        self.schema
            .first_of_type(FieldType::Number)
            .map(|spec| spec.name.clone())
    }

    /// Resolve the target field name for an author pattern: first
    /// [`FieldType::Text`] field, falling back to a field literally named or
    /// aliased `author`.
    fn author_field(&self) -> Option<String> {
        if let Some(spec) = self.schema.field_for("author") {
            return Some(spec.name.clone());
        }
        self.schema
            .first_of_type(FieldType::Text)
            .map(|spec| spec.name.clone())
    }

    /// Resolve the target field name for a tag pattern: first
    /// [`FieldType::Tag`] field, falling back to one named/aliased `tag` or
    /// `category`.
    fn tag_field(&self) -> Option<String> {
        if let Some(spec) = self.schema.first_of_type(FieldType::Tag) {
            return Some(spec.name.clone());
        }
        for token in ["tag", "category", "tags"] {
            if let Some(spec) = self.schema.field_for(token) {
                return Some(spec.name.clone());
            }
        }
        None
    }

    /// Parse `nl_query` into a [`StructuredQuery`].
    ///
    /// # Errors
    ///
    /// Returns [`SelfQueryError::EmptyQuery`] when `nl_query` is empty or only
    /// whitespace.
    pub fn parse(&self, nl_query: &str) -> Result<StructuredQuery, SelfQueryError> {
        if nl_query.trim().is_empty() {
            return Err(SelfQueryError::EmptyQuery);
        }

        let tokens = tokenize(nl_query);
        let mut conditions: Vec<FilterCondition> = Vec::new();
        // Byte ranges (start, end) of tokens consumed by a matched pattern.
        let mut consumed: Vec<(usize, usize)> = Vec::new();

        let mut idx = 0_usize;
        while idx < tokens.len() {
            if let Some(matched) = self.match_at(&tokens, idx) {
                conditions.push(matched.condition);
                consumed.push((tokens[idx].start, matched.span_end));
                idx = matched.next_idx;
            } else {
                idx += 1;
            }
        }

        let semantic_query = if self.config.strip_filter_phrases {
            strip_spans(nl_query, &consumed)
        } else {
            nl_query.trim().to_string()
        };

        Ok(StructuredQuery::new(
            semantic_query,
            ParsedFilter { conditions },
        ))
    }

    /// Attempt to match a single filter pattern anchored at `tokens[idx]`.
    ///
    /// Trigger families are tried in a fixed order so parsing is deterministic.
    fn match_at(&self, tokens: &[Token], idx: usize) -> Option<Match> {
        let trigger = tokens[idx].lower.as_str();

        if AFTER_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_date_value(tokens, idx, FilterOp::Gt)
        {
            return Some(m);
        }
        if BEFORE_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_date_value(tokens, idx, FilterOp::Lt)
        {
            return Some(m);
        }
        if IN_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_date_value(tokens, idx, FilterOp::Eq)
        {
            return Some(m);
        }
        if ABOVE_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_number_value(tokens, idx, FilterOp::Gt)
        {
            return Some(m);
        }
        if BELOW_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_number_value(tokens, idx, FilterOp::Lt)
        {
            return Some(m);
        }
        if BY_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_author_value(tokens, idx)
        {
            return Some(m);
        }
        if TAG_TRIGGERS.contains(&trigger)
            && let Some(m) = self.match_tag_value(tokens, idx)
        {
            return Some(m);
        }
        None
    }

    /// Match `<trigger> <year>` for a date field with the given operator.
    fn match_date_value(&self, tokens: &[Token], idx: usize, op: FilterOp) -> Option<Match> {
        let value_token = tokens.get(idx + 1)?;
        if !is_year(&value_token.lower) {
            return None;
        }
        let field = self.date_field()?;
        Some(Match {
            condition: FilterCondition::new(field, op, value_token.raw.clone()),
            span_end: value_token.end,
            next_idx: idx + 2,
        })
    }

    /// Match `<trigger> [<connective>] <number>` for a number field.
    ///
    /// `default_op` is used unless an intervening directional connective
    /// (`above`/`over` → `Gt`, `below`/`under` → `Lt`) overrides it, so
    /// `"rated above 4"` and `"rated below 2"` resolve correctly while the bare
    /// `"rated 4"` keeps `default_op`.
    fn match_number_value(
        &self,
        tokens: &[Token],
        idx: usize,
        default_op: FilterOp,
    ) -> Option<Match> {
        let next = tokens.get(idx + 1)?;
        let (value_token, op, next_idx) = if is_number(&next.lower) {
            (next, default_op, idx + 2)
        } else if ABOVE_TRIGGERS.contains(&next.lower.as_str()) {
            let value = tokens.get(idx + 2).filter(|t| is_number(&t.lower))?;
            (value, FilterOp::Gt, idx + 3)
        } else if BELOW_TRIGGERS.contains(&next.lower.as_str()) {
            let value = tokens.get(idx + 2).filter(|t| is_number(&t.lower))?;
            (value, FilterOp::Lt, idx + 3)
        } else {
            return None;
        };
        let field = self.number_field()?;
        Some(Match {
            condition: FilterCondition::new(field, op, value_token.raw.clone()),
            span_end: value_token.end,
            next_idx,
        })
    }

    /// Match `by <Name…>` and build an author equality condition.
    ///
    /// The name spans one or more consecutive capitalised tokens following the
    /// trigger; if none are capitalised, the single next token is used.
    fn match_author_value(&self, tokens: &[Token], idx: usize) -> Option<Match> {
        let first = tokens.get(idx + 1)?;
        let field = self.author_field()?;

        let mut parts: Vec<String> = Vec::new();
        let mut end = first.end;
        let mut cursor = idx + 1;
        while cursor < tokens.len() && is_name_token(&tokens[cursor].raw) {
            parts.push(tokens[cursor].raw.clone());
            end = tokens[cursor].end;
            cursor += 1;
        }
        if parts.is_empty() {
            // Fall back to a single (lower-case) token.
            parts.push(first.raw.clone());
            end = first.end;
            cursor = idx + 2;
        }

        Some(Match {
            condition: FilterCondition::new(field, FilterOp::Eq, parts.join(" ")),
            span_end: end,
            next_idx: cursor,
        })
    }

    /// Match `tagged/category [as|with] <X>` and build a tag containment
    /// condition.
    fn match_tag_value(&self, tokens: &[Token], idx: usize) -> Option<Match> {
        let next = tokens.get(idx + 1)?;
        let (value_token, next_idx) = if matches!(next.lower.as_str(), "as" | "with") {
            let real = tokens.get(idx + 2)?;
            (real, idx + 3)
        } else {
            (next, idx + 2)
        };
        let field = self.tag_field()?;
        Some(Match {
            condition: FilterCondition::new(field, FilterOp::Contains, value_token.raw.clone()),
            span_end: value_token.end,
            next_idx,
        })
    }
}

/// A successfully matched filter pattern.
struct Match {
    /// The extracted condition.
    condition: FilterCondition,
    /// Byte offset (exclusive) where the matched span ends in the source.
    span_end: usize,
    /// Token index at which scanning should resume.
    next_idx: usize,
}

// ── SelfQueryRetriever ────────────────────────────────────────────────────────

/// A self-querying retriever.
///
/// Wraps a [`SelfQueryParser`]: it parses a natural-language query into a
/// [`StructuredQuery`], then returns every document whose metadata satisfies
/// the extracted [`ParsedFilter`]. The semantic portion of the query is
/// returned to the caller for use with a downstream semantic retriever.
#[derive(Debug, Clone)]
pub struct SelfQueryRetriever {
    /// The underlying parser.
    pub parser: SelfQueryParser,
}

impl SelfQueryRetriever {
    /// Create a retriever from a metadata [`FilterSchema`].
    #[must_use]
    pub fn new(schema: FilterSchema) -> Self {
        Self {
            parser: SelfQueryParser::new(schema),
        }
    }

    /// Create a retriever from an existing parser.
    #[must_use]
    pub fn from_parser(parser: SelfQueryParser) -> Self {
        Self { parser }
    }

    /// Parse `nl_query` and return the structured query alongside references to
    /// every document in `docs` whose metadata matches the parsed filter.
    ///
    /// When the parsed filter is empty, all documents are returned (an empty
    /// filter matches everything).
    ///
    /// # Errors
    ///
    /// Returns [`SelfQueryError::EmptyQuery`] when `nl_query` is empty or only
    /// whitespace.
    pub fn retrieve<'a>(
        &self,
        nl_query: &str,
        docs: &'a [Document],
    ) -> Result<(StructuredQuery, Vec<&'a Document>), SelfQueryError> {
        let structured = self.parser.parse(nl_query)?;
        let matched: Vec<&'a Document> = docs
            .iter()
            .filter(|doc| structured.filter.matches(&doc.metadata))
            .collect();
        Ok((structured, matched))
    }
}

/// Return `true` if `raw` starts with an ASCII uppercase letter (a name-like
/// token).
fn is_name_token(raw: &str) -> bool {
    raw.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

/// Remove the given byte spans from `text` and collapse the resulting
/// whitespace, returning a trimmed string.
fn strip_spans(text: &str, spans: &[(usize, usize)]) -> String {
    if spans.is_empty() {
        return collapse_whitespace(text);
    }

    // Build a sorted, non-overlapping copy of the spans.
    let mut sorted = spans.to_vec();
    sorted.sort_unstable();

    let mut result = String::with_capacity(text.len());
    let mut cursor = 0_usize;
    for (start, end) in sorted {
        let start = start.min(text.len());
        let end = end.min(text.len());
        if start < cursor {
            // Overlapping span; extend the cut if needed.
            if end > cursor {
                cursor = end;
            }
            continue;
        }
        result.push_str(&text[cursor..start]);
        result.push(' ');
        cursor = end;
    }
    if cursor < text.len() {
        result.push_str(&text[cursor..]);
    }

    collapse_whitespace(&result)
}

/// Collapse runs of whitespace into single spaces and trim the result.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
