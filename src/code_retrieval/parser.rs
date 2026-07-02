//! The lightweight, deterministic **structural tokenizer** for
//! `code_retrieval`.
//!
//! This is a heuristic brace/keyword/indentation scanner — *not* a parser for
//! any real language, and it never executes anything. Given a chunk of source
//! text it extracts, in one deterministic pass:
//!
//! * **Function/method definitions** — a name, a top-level parameter count, and
//!   a rough body span (via brace-depth matching for brace-delimited syntax, or
//!   an indentation-block scan for Python-like syntax), each yielded as a
//!   [`CodeRetrievalUnit`].
//! * **Imports/uses** — lines matching import-like keywords (`use`, `import`,
//!   `from ... import`, `#include`, `require(...)`), reduced to the imported
//!   module/path.
//! * **Identifiers** — distinct identifier-like tokens (letter/digit/underscore
//!   runs) that are not language keywords, shorter than the configured minimum
//!   length, or purely numeric, recorded as [`CodeRetrievalSymbol`]s with
//!   occurrence counts.
//! * **Call-sites** — an identifier immediately followed by `(` inside a unit's
//!   body (a heuristic function-call signal), with control- and
//!   definition-keyword heads (`if`, `while`, `fn`, …) filtered out.
//!
//! Comments (`//`, `/* */`, and Python `#`, while preserving C preprocessor
//! directives and Rust `#[...]`/`#![...]` attributes) are stripped before
//! identifier/call/body scanning, so that prose in comments never masquerades
//! as code structure — while the *textual* similarity computed elsewhere still
//! sees the raw text, comments included.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::code_retrieval::types::{
    CodeRetrievalConfig, CodeRetrievalLanguageHint, CodeRetrievalSymbol, CodeRetrievalUnit,
    CodeRetrievalUnitKind,
};

// ── Keyword tables (parser-internal) ─────────────────────────────────────────
//
// These are distinct from [`CodeRetrievalConfig::keywords`] (which governs
// identifier-set exclusion). `DEF_KEYWORDS` anchor function-definition
// detection; `CONTROL_KEYWORDS` prevent control-flow constructs from being
// misread as definitions or calls; `PREPROCESSOR_DIRECTIVES` keep C
// preprocessor lines intact through comment stripping.

/// Keyword heads that introduce a function/method definition.
const DEF_KEYWORDS: &[&str] = &[
    "fn", "func", "function", "def", "sub", "fun", "method", "proc",
];

/// Control-flow / operator keyword heads that may be followed by `(` but never
/// denote a definition or a call.
const CONTROL_KEYWORDS: &[&str] = &[
    "if", "for", "while", "switch", "catch", "match", "else", "elif", "do", "loop", "with",
    "return", "foreach", "when", "try", "except", "finally", "case", "await", "yield", "in", "and",
    "or", "not", "new", "delete", "typeof", "sizeof", "defer", "go", "select", "throw", "raise",
    "assert", "del", "print",
];

/// C-family preprocessor directive words that follow a leading `#`.
const PREPROCESSOR_DIRECTIVES: &[&str] = &[
    "include", "define", "undef", "ifdef", "ifndef", "endif", "pragma", "if", "elif", "else",
    "error", "line", "import", "warning",
];

// ── Character / token helpers ────────────────────────────────────────────────

/// Return `true` when `c` may appear inside an identifier token.
#[must_use]
pub(crate) fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Split `text` into identifier-like tokens: maximal runs of identifier
/// characters, in source case, with empties dropped.
///
/// This is the shared lexical primitive: the parser applies it to
/// comment-stripped code for identifier/symbol extraction, and the engine
/// applies it to raw text for the pseudo-embedding.
#[must_use]
pub(crate) fn ident_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !is_ident_char(c))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Return `true` when the lowercase form of `token` is a definition keyword.
fn is_def_keyword(token: &str) -> bool {
    let lower = token.to_lowercase();
    DEF_KEYWORDS.contains(&lower.as_str())
}

/// Return `true` when the lowercase form of `token` is a control keyword.
fn is_control_keyword(token: &str) -> bool {
    let lower = token.to_lowercase();
    CONTROL_KEYWORDS.contains(&lower.as_str())
}

/// Return `true` when `token` is a plain identifier: non-empty, beginning with
/// a letter or underscore, and composed solely of identifier characters.
fn is_plain_identifier(token: &str) -> bool {
    let mut chars = token.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => chars.all(is_ident_char),
        _ => false,
    }
}

/// The width of the leading whitespace of `line`, counted in characters (a tab
/// counts as one).
fn leading_ws_width(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).count()
}

// ── Comment stripping ────────────────────────────────────────────────────────

/// Return `true` when the `#` at `pos` in `chars` begins a C-family
/// preprocessor directive (skipping any spaces after the `#`).
fn is_preprocessor_directive(chars: &[char], pos: usize) -> bool {
    let mut i = pos + 1;
    while i < chars.len() && chars[i] == ' ' {
        i += 1;
    }
    let start = i;
    while i < chars.len() && chars[i].is_alphabetic() {
        i += 1;
    }
    if start == i {
        return false;
    }
    let word: String = chars[start..i].iter().collect::<String>().to_lowercase();
    PREPROCESSOR_DIRECTIVES.contains(&word.as_str())
}

/// Remove comments from `text` while preserving every newline (so line indices
/// are unchanged) and preserving C preprocessor directives and Rust
/// `#[...]` / `#![...]` attributes.
///
/// Handles `//` line comments, `/* ... */` block comments, and Python-style
/// `#` line comments. String-literal contents are *not* tracked (a deliberate
/// heuristic limitation): a `//` or `#` inside a string literal is treated as a
/// comment.
fn strip_comments(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    // Whether only whitespace has been seen since the last newline.
    let mut at_line_start = true;

    while i < n {
        let c = chars[i];

        // `//` line comment.
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // `/* ... */` block comment (may span lines).
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(chars[i] == '*' && chars[i + 1] == '/') {
                if chars[i] == '\n' {
                    out.push('\n');
                    at_line_start = true;
                }
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }

        // `#` — a Python comment, unless it is a Rust attribute (`#[`, `#!`) or
        // a C preprocessor directive at the line start.
        if c == '#' {
            let next = chars.get(i + 1).copied();
            let is_attribute = matches!(next, Some('[' | '!'));
            let is_directive = at_line_start && is_preprocessor_directive(&chars, i);
            if !is_attribute && !is_directive {
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
        }

        if c == '\n' {
            at_line_start = true;
        } else if !c.is_whitespace() {
            at_line_start = false;
        }
        out.push(c);
        i += 1;
    }

    out
}

// ── Import extraction ────────────────────────────────────────────────────────

/// If `s` is wrapped in a matching pair of single quotes, double quotes, or
/// backticks (length at least two), return its unquoted interior.
fn strip_quotes(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'\'' || first == b'"' || first == b'`') && first == last {
            return Some(s[1..s.len() - 1].to_string());
        }
    }
    None
}

/// Extract the interior between the first `open` and the next `close` after it.
fn extract_delimited(s: &str, open: char, close: char) -> Option<String> {
    let start = s.find(open)?;
    let rest = &s[start + open.len_utf8()..];
    let end = rest.find(close)?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        None
    } else {
        Some(inner.to_string())
    }
}

/// Extract the imported module/path from a single source `line`, if it looks
/// like an import/use/include directive.
fn extract_import_from_line(line: &str) -> Option<String> {
    let t = line.trim();

    // C/C++: `#include <path>` or `#include "path"`.
    if let Some(rest) = t.strip_prefix("#include") {
        let rest = rest.trim();
        return extract_delimited(rest, '<', '>').or_else(|| extract_delimited(rest, '"', '"'));
    }

    // Python: `from module import ...`.
    if let Some(rest) = t.strip_prefix("from ") {
        if let Some(idx) = rest.find(" import ") {
            let module = rest[..idx].trim();
            if !module.is_empty() {
                return Some(module.to_string());
            }
        }
        return None;
    }

    // Python / JavaScript / Java: `import ...`.
    if let Some(rest) = t.strip_prefix("import ") {
        // JavaScript: `import { a } from 'mod'`.
        if let Some(fidx) = rest.rfind(" from ") {
            let after = rest[fidx + " from ".len()..]
                .trim()
                .trim_end_matches(';')
                .trim();
            if let Some(module) = strip_quotes(after) {
                return Some(module);
            }
        }
        // Otherwise take the first path token, dropping `as`, `,`, or `;` tails.
        let head = rest.split([';', ',']).next().unwrap_or(rest).trim();
        let head = head.split_whitespace().next().unwrap_or(head);
        let cleaned = strip_quotes(head).unwrap_or_else(|| head.to_string());
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
        return None;
    }

    // JavaScript / Node: `require('mod')` anywhere on the line.
    if let Some(idx) = t.find("require(") {
        let after = &t[idx + "require(".len()..];
        if let Some(close) = after.find(')') {
            let inner = after[..close].trim();
            if let Some(module) = strip_quotes(inner) {
                return Some(module);
            }
        }
    }

    // Rust: `use path::to::thing;` (also `pub use ...`).
    let use_rest = t
        .strip_prefix("use ")
        .or_else(|| t.strip_prefix("pub use "));
    if let Some(rest) = use_rest {
        let path = rest.trim().trim_end_matches(';').trim();
        let base = path
            .split('{')
            .next()
            .unwrap_or(path)
            .trim()
            .trim_end_matches("::")
            .trim();
        if !base.is_empty() {
            return Some(base.to_string());
        }
    }

    None
}

/// Extract every distinct imported module/path in `text`, sorted ascending.
fn extract_imports(text: &str) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for line in text.lines() {
        if let Some(import) = extract_import_from_line(line) {
            set.insert(import);
        }
    }
    set.into_iter().collect()
}

// ── Source scan buffer ───────────────────────────────────────────────────────

/// A flattened, comment-stripped view of a source item, holding both the raw
/// lines (for `raw_text` and imports) and the comment-stripped code (for
/// structural scanning), plus a char-index-to-line map for cross-line brace and
/// parenthesis matching.
struct Scan {
    /// The item's raw lines, in order.
    raw_lines: Vec<String>,
    /// The item's comment-stripped lines, aligned index-for-index with
    /// `raw_lines`.
    code_lines: Vec<String>,
    /// The comment-stripped code as a flat character buffer.
    code: Vec<char>,
    /// For each char in `code`, the zero-based line it belongs to.
    line_of: Vec<usize>,
    /// For each line, the char index in `code` at which it begins.
    line_start: Vec<usize>,
    /// The number of lines (equal to `raw_lines.len()`).
    line_count: usize,
}

impl Scan {
    /// Build a scan buffer from raw source `text`.
    fn new(text: &str) -> Self {
        let raw_lines: Vec<String> = text.lines().map(str::to_string).collect();
        let code_str = strip_comments(text);
        let code_lines: Vec<String> = code_str.lines().map(str::to_string).collect();
        let code: Vec<char> = code_str.chars().collect();

        let mut line_of = Vec::with_capacity(code.len());
        let mut line_start = vec![0usize];
        let mut line = 0usize;
        for (idx, &c) in code.iter().enumerate() {
            line_of.push(line);
            if c == '\n' {
                line += 1;
                line_start.push(idx + 1);
            }
        }

        let line_count = raw_lines.len();
        Self {
            raw_lines,
            code_lines,
            code,
            line_of,
            line_start,
            line_count,
        }
    }

    /// The comment-stripped text of line `i`, or `""` when out of range.
    fn code_line(&self, i: usize) -> &str {
        self.code_lines.get(i).map_or("", String::as_str)
    }

    /// The raw text of line `i`, or `""` when out of range.
    fn raw_line(&self, i: usize) -> &str {
        self.raw_lines.get(i).map_or("", String::as_str)
    }

    /// The flat char index of column `col` on line `line_idx`.
    fn flat_pos(&self, line_idx: usize, col: usize) -> usize {
        self.line_start.get(line_idx).copied().unwrap_or(0) + col
    }

    /// The flat char index just past the end of line `line_idx`.
    fn line_end_flat(&self, line_idx: usize) -> usize {
        self.line_start
            .get(line_idx + 1)
            .copied()
            .unwrap_or(self.code.len())
    }

    /// The flat code slice `[lo, hi)` as an owned string (bounds-clamped).
    fn slice_flat(&self, lo: usize, hi: usize) -> String {
        let lo = lo.min(self.code.len());
        let hi = hi.min(self.code.len());
        if lo >= hi {
            String::new()
        } else {
            self.code[lo..hi].iter().collect()
        }
    }

    /// The comment-stripped lines `[start, end]` joined by newlines.
    fn join_code(&self, start: usize, end: usize) -> String {
        if start > end {
            return String::new();
        }
        (start..=end)
            .filter_map(|k| self.code_lines.get(k))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The raw lines `[start, end]` joined by newlines.
    fn join_raw(&self, start: usize, end: usize) -> String {
        if start > end {
            return String::new();
        }
        (start..=end)
            .filter_map(|k| self.raw_lines.get(k))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ── Cross-line matching over the flat code buffer ────────────────────────────

/// Given `open_pos` pointing at a `(`, return the flat index of the matching
/// `)`, or `None` if unbalanced.
fn match_paren_end(code: &[char], open_pos: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = open_pos;
    while i < code.len() {
        match code[i] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Given `open_pos` pointing at a `{`, return the line of the matching `}`; if
/// unbalanced, the last line.
fn match_brace_end(scan: &Scan, open_pos: usize) -> usize {
    let mut depth = 0i32;
    let mut i = open_pos;
    while i < scan.code.len() {
        match scan.code[i] {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return scan.line_of.get(i).copied().unwrap_or(0);
                }
            }
            _ => {}
        }
        i += 1;
    }
    scan.line_count.saturating_sub(1)
}

/// From `from_pos`, scan forward for the `{` that opens a function body,
/// returning its flat index. Returns `None` on hitting a `;` first (a
/// declaration, not a definition) or on reaching the end.
fn find_body_open(code: &[char], from_pos: usize) -> Option<usize> {
    let mut i = from_pos;
    while i < code.len() {
        match code[i] {
            '{' => return Some(i),
            ';' => return None,
            _ => {}
        }
        i += 1;
    }
    None
}

/// The identifier ending immediately before `pos` (skipping intervening
/// whitespace), returned with its start index, or `None` if there is no
/// letter/underscore-initial identifier there.
fn ident_ending_before(chars: &[char], pos: usize) -> Option<(String, usize)> {
    let mut end = pos;
    while end > 0 && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && is_ident_char(chars[start - 1]) {
        start -= 1;
    }
    if start == end {
        return None;
    }
    let name: String = chars[start..end].iter().collect();
    match name.chars().next() {
        Some(c) if c.is_ascii_digit() => None,
        _ => Some((name, start)),
    }
}

/// The token immediately before `name_start` (skipping whitespace): an
/// identifier run if one is there, otherwise the single symbol character.
fn token_before(chars: &[char], name_start: usize) -> Option<String> {
    let mut end = name_start;
    while end > 0 && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    if is_ident_char(chars[end - 1]) {
        let mut start = end;
        while start > 0 && is_ident_char(chars[start - 1]) {
            start -= 1;
        }
        Some(chars[start..end].iter().collect())
    } else {
        Some(chars[end - 1].to_string())
    }
}

/// Count the top-level (comma-separated) parameters in the char slice between a
/// signature's parentheses. Commas nested inside `()`, `[]`, `{}`, or `<>` are
/// ignored, and a single trailing comma is not counted.
fn count_params(inner: &[char]) -> usize {
    let mut start = 0;
    let mut end = inner.len();
    while start < end && inner[start].is_whitespace() {
        start += 1;
    }
    while end > start && inner[end - 1].is_whitespace() {
        end -= 1;
    }
    if start == end {
        return 0;
    }
    let slice = &inner[start..end];
    let mut depth = 0i32;
    let mut count = 1usize;
    for &c in slice {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            ',' if depth == 0 => count += 1,
            _ => {}
        }
    }
    if slice.last() == Some(&',') {
        count = count.saturating_sub(1);
    }
    count
}

// ── Brace-language function detection ────────────────────────────────────────

/// A detected brace-language function signature.
struct BraceSignature {
    /// The function name (source case).
    name: String,
    /// The top-level parameter count.
    param_count: usize,
    /// The flat index of the `{` opening the body.
    body_open: usize,
}

/// Attempt to detect a function definition whose signature begins on line
/// `line_idx`. Multi-line signatures and bodies are handled via the flat code
/// buffer.
fn detect_brace_signature(scan: &Scan, line_idx: usize) -> Option<BraceSignature> {
    let line_chars: Vec<char> = scan.code_line(line_idx).chars().collect();
    let mut col = 0;
    while col < line_chars.len() {
        if line_chars[col] == '('
            && let Some((name, name_start)) = ident_ending_before(&line_chars, col)
            && !is_control_keyword(&name)
        {
            let preceding = token_before(&line_chars, name_start);
            let is_def = preceding.as_deref().is_some_and(is_def_keyword);
            let is_typed = preceding.as_deref().is_some_and(|p| {
                is_plain_identifier(p) && !is_control_keyword(p) && !is_def_keyword(p)
            });
            if is_def || is_typed {
                let paren_flat = scan.flat_pos(line_idx, col);
                if let Some(close_flat) = match_paren_end(&scan.code, paren_flat) {
                    let param_count = count_params(&scan.code[paren_flat + 1..close_flat]);
                    if let Some(body_open) = find_body_open(&scan.code, close_flat + 1) {
                        return Some(BraceSignature {
                            name,
                            param_count,
                            body_open,
                        });
                    }
                }
            }
        }
        col += 1;
    }
    None
}

/// Extend a definition's span upward to absorb the contiguous run of
/// pure-comment lines immediately preceding it (its doc comment), stopping at
/// the first blank or code-bearing line, or at `floor` (the line after the
/// previous unit's end). A pure-comment line is one whose raw text is
/// non-blank but whose comment-stripped text is blank.
fn extend_leading_comments(scan: &Scan, sig_line: usize, floor: usize) -> usize {
    let mut start = sig_line;
    let mut k = sig_line;
    while k > floor {
        let prev = k - 1;
        let raw_nonblank = !scan.raw_line(prev).trim().is_empty();
        let code_blank = scan.code_line(prev).trim().is_empty();
        if raw_nonblank && code_blank {
            start = prev;
            k = prev;
        } else {
            break;
        }
    }
    start
}

/// Scan a brace-delimited item into function units. Nested definitions are
/// absorbed into their enclosing unit (the scan resumes past each emitted
/// body). A definition's span also absorbs the contiguous doc-comment lines
/// directly above its signature.
fn scan_brace_units(
    scan: &Scan,
    source_id: &str,
    config: &CodeRetrievalConfig,
    keyword_set: &HashSet<String>,
    imports: &[String],
    language: CodeRetrievalLanguageHint,
) -> Vec<CodeRetrievalUnit> {
    let mut units = Vec::new();
    let n = scan.line_count;
    let mut i = 0;
    let mut floor = 0;
    while i < n {
        if let Some(sig) = detect_brace_signature(scan, i) {
            let end_line = match_brace_end(scan, sig.body_open);
            let start_line = extend_leading_comments(scan, i, floor);
            let call_region = scan.slice_flat(sig.body_open + 1, scan.line_end_flat(end_line));
            let unit = build_unit(
                scan,
                source_id,
                units.len(),
                &sig.name,
                CodeRetrievalUnitKind::Function,
                sig.param_count,
                start_line,
                end_line,
                &call_region,
                config,
                keyword_set,
                imports,
                language,
            );
            units.push(unit);
            floor = end_line + 1;
            i = end_line + 1;
        } else {
            i += 1;
        }
    }
    units
}

// ── Indentation-language function detection ──────────────────────────────────

/// Detect a Python-like `def NAME(params):` (or `async def ...`) header on a
/// comment-stripped line, returning the name and top-level parameter count.
fn detect_indent_def(code_line: &str) -> Option<(String, usize)> {
    let trimmed = code_line.trim();
    let after_def = trimmed
        .strip_prefix("def ")
        .or_else(|| trimmed.strip_prefix("async def "))?
        .trim_start();

    let name = leading_identifier(after_def)?;
    let rest = after_def[name.len()..].trim_start();
    let rest_chars: Vec<char> = rest.chars().collect();
    if rest_chars.first() != Some(&'(') {
        return None;
    }
    let close = match_paren_end(&rest_chars, 0)?;
    let param_count = count_params(&rest_chars[1..close]);

    let tail: String = rest_chars[close + 1..].iter().collect();
    if !tail.contains(':') {
        return None;
    }
    Some((name.to_string(), param_count))
}

/// The leading identifier of `s` (letter/underscore-initial), or `None`.
fn leading_identifier(s: &str) -> Option<&str> {
    let mut end = 0;
    for (idx, c) in s.char_indices() {
        if idx == 0 {
            if !(c.is_alphabetic() || c == '_') {
                return None;
            }
        } else if !is_ident_char(c) {
            break;
        }
        end = idx + c.len_utf8();
    }
    if end == 0 { None } else { Some(&s[..end]) }
}

/// Scan an indentation-delimited item into function units. A function body is
/// the run of blank or more-indented lines following its `def` header; nested
/// `def`s are absorbed into their enclosing unit.
fn scan_indent_units(
    scan: &Scan,
    source_id: &str,
    config: &CodeRetrievalConfig,
    keyword_set: &HashSet<String>,
    imports: &[String],
    language: CodeRetrievalLanguageHint,
) -> Vec<CodeRetrievalUnit> {
    let mut units = Vec::new();
    let n = scan.line_count;
    let mut i = 0;
    let mut floor = 0;
    while i < n {
        if let Some((name, param_count)) = detect_indent_def(scan.code_line(i)) {
            let def_indent = leading_ws_width(scan.raw_line(i));
            let mut end = i;
            let mut j = i + 1;
            while j < n {
                if scan.code_line(j).trim().is_empty() {
                    j += 1;
                    continue;
                }
                if leading_ws_width(scan.raw_line(j)) > def_indent {
                    end = j;
                    j += 1;
                } else {
                    break;
                }
            }
            let start_line = extend_leading_comments(scan, i, floor);
            let call_region = if i < end {
                scan.join_code(i + 1, end)
            } else {
                String::new()
            };
            let unit = build_unit(
                scan,
                source_id,
                units.len(),
                &name,
                CodeRetrievalUnitKind::Function,
                param_count,
                start_line,
                end,
                &call_region,
                config,
                keyword_set,
                imports,
                language,
            );
            units.push(unit);
            floor = end + 1;
            i = end + 1;
        } else {
            i += 1;
        }
    }
    units
}

// ── Symbol / call-site extraction ────────────────────────────────────────────

/// Extract the distinct identifier symbols (with occurrence counts) from
/// comment-stripped `code`, excluding configured keywords, tokens shorter than
/// `config.min_identifier_len`, and non-letter-initial tokens. Names are
/// lowercased and returned sorted ascending.
fn extract_symbols(
    code: &str,
    config: &CodeRetrievalConfig,
    keyword_set: &HashSet<String>,
) -> Vec<CodeRetrievalSymbol> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for token in ident_tokens(code) {
        let lower = token.to_lowercase();
        if lower.chars().count() < config.min_identifier_len {
            continue;
        }
        if keyword_set.contains(&lower) {
            continue;
        }
        let letter_initial = lower
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        if !letter_initial {
            continue;
        }
        *counts.entry(lower).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(name, occurrences)| CodeRetrievalSymbol::new(name, occurrences))
        .collect()
}

/// Extract the distinct call-site names from comment-stripped `code`: each
/// identifier immediately followed by `(`, excluding control- and
/// definition-keyword heads. Names are lowercased and returned sorted
/// ascending.
fn extract_calls(code: &str) -> Vec<String> {
    let chars: Vec<char> = code.chars().collect();
    let mut set: BTreeSet<String> = BTreeSet::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '('
            && let Some((name, _)) = ident_ending_before(&chars, i)
        {
            let lower = name.to_lowercase();
            let letter_initial = lower
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
            if letter_initial && !is_control_keyword(&lower) && !is_def_keyword(&lower) {
                set.insert(lower);
            }
        }
        i += 1;
    }
    set.into_iter().collect()
}

// ── Unit construction ────────────────────────────────────────────────────────

/// Assemble a [`CodeRetrievalUnit`] spanning lines `[start_line, end_line]`,
/// with identifiers drawn from the whole span and call-sites from
/// `call_region` (the body only, so a function's own signature name is not
/// counted as a call).
#[allow(clippy::too_many_arguments)]
fn build_unit(
    scan: &Scan,
    source_id: &str,
    unit_index: usize,
    name: &str,
    kind: CodeRetrievalUnitKind,
    param_count: usize,
    start_line: usize,
    end_line: usize,
    call_region: &str,
    config: &CodeRetrievalConfig,
    keyword_set: &HashSet<String>,
    imports: &[String],
    language: CodeRetrievalLanguageHint,
) -> CodeRetrievalUnit {
    let raw_text = scan.join_raw(start_line, end_line);
    let unit_code = scan.join_code(start_line, end_line);
    let symbols = extract_symbols(&unit_code, config, keyword_set);
    let call_sites = extract_calls(call_region);

    CodeRetrievalUnit {
        source_id: source_id.to_string(),
        unit_index,
        name: name.to_string(),
        kind,
        param_count,
        body_start_line: start_line,
        body_end_line: end_line,
        language,
        symbols,
        imports: imports.to_vec(),
        call_sites,
        raw_text,
    }
}

// ── Language resolution ──────────────────────────────────────────────────────

/// Return `true` when `code` contains a Python-like `def ...:` header.
fn has_indent_def(code: &str) -> bool {
    code.lines().any(|line| {
        let t = line.trim();
        (t.starts_with("def ") || t.starts_with("async def "))
            && t.contains('(')
            && t.trim_end().ends_with(':')
    })
}

/// Resolve a possibly-[`CodeRetrievalLanguageHint::Auto`] hint against the
/// item's content into a concrete brace/indent family.
fn resolve_language(text: &str, hint: CodeRetrievalLanguageHint) -> CodeRetrievalLanguageHint {
    match hint {
        CodeRetrievalLanguageHint::Auto => {
            let code = strip_comments(text);
            if code.contains('{') {
                CodeRetrievalLanguageHint::BraceDelimited
            } else if has_indent_def(&code) {
                CodeRetrievalLanguageHint::IndentDelimited
            } else {
                CodeRetrievalLanguageHint::BraceDelimited
            }
        }
        other => other,
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Parse one source item into structural [`CodeRetrievalUnit`]s.
///
/// Each detected function/method becomes a unit; if none is detected, the whole
/// item becomes a single [`CodeRetrievalUnitKind::WholeFile`] fallback unit.
/// Every unit from the item shares the item's file-level import set. The
/// returned units carry no pseudo-embedding — that is derived from each unit's
/// `raw_text` by the engine at index time.
///
/// This function never executes the source; it is a purely structural,
/// deterministic scan.
#[must_use]
pub fn parse_source(
    source_id: &str,
    text: &str,
    config: &CodeRetrievalConfig,
) -> Vec<CodeRetrievalUnit> {
    let language = resolve_language(text, config.language_hint);
    let imports = extract_imports(text);
    let scan = Scan::new(text);
    let keyword_set: HashSet<String> = config.keywords.iter().map(|k| k.to_lowercase()).collect();

    let mut units = if matches!(language, CodeRetrievalLanguageHint::IndentDelimited) {
        scan_indent_units(&scan, source_id, config, &keyword_set, &imports, language)
    } else {
        scan_brace_units(&scan, source_id, config, &keyword_set, &imports, language)
    };

    if units.is_empty() {
        let end_line = scan.line_count.saturating_sub(1);
        let whole_code = scan.join_code(0, end_line);
        units.push(build_unit(
            &scan,
            source_id,
            0,
            source_id,
            CodeRetrievalUnitKind::WholeFile,
            0,
            0,
            end_line,
            &whole_code,
            config,
            &keyword_set,
            &imports,
            language,
        ));
    }

    units
}
