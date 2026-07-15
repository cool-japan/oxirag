//! Text-level primitives the bootstrapping loop depends on: answer equivalence
//! (the correctness oracle) and cheat-rationale detection (the correctness
//! *guard* on backward rationalization).
//!
//! Both are deliberately hand-rolled here rather than reused from
//! `self_consistency` or `semantic_entropy`, so the `self-taught-reasoner`
//! feature stays independent of theirs. The answer-equivalence check mirrors the
//! greedy token-Jaccard those modules use; the cheat detector is specific to
//! `STaR` and has no analogue elsewhere in the crate.

use std::collections::HashSet;

// ── Tokenization ───────────────────────────────────────────────────────────────

/// Tokenize `text` into lowercase alphanumeric tokens.
///
/// Splits on every non-alphanumeric character and drops empty fragments, so
/// `"7 + 5 = 12"` becomes `["7", "5", "12"]` and `"[A] x=7"` becomes
/// `["a", "x", "7"]`.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Tokenize into a set (membership is all the callers here need).
fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

// ── Answer equivalence ─────────────────────────────────────────────────────────

/// Canonicalize an answer string for equivalence comparison.
///
/// Lowercases; if the trimmed content is a signed integer, returns its canonical
/// decimal form (`"07"` ⇒ `"7"`, `"+3."` ⇒ `"3"`, `"-0"` ⇒ `"0"`); otherwise
/// strips surrounding punctuation/whitespace and collapses internal whitespace
/// runs to a single space.
#[must_use]
pub fn normalize_answer(answer: &str) -> String {
    let lowered = answer.to_lowercase();

    // Integer fast path (sign-aware): keep a leading '+'/'-' that belongs to the
    // number, trim everything else off the ends, then canonicalize.
    let head_trimmed =
        lowered.trim_start_matches(|c: char| !(c.is_alphanumeric() || c == '+' || c == '-'));
    let int_candidate = head_trimmed.trim_end_matches(|c: char| !c.is_alphanumeric());
    if let Some(canon) = canonicalize_integer(int_candidate) {
        return canon;
    }

    // General path: strip surrounding punctuation, collapse internal whitespace.
    let trimmed = lowered.trim_matches(|c: char| !c.is_alphanumeric());
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_ws = false;
    for c in trimmed.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

/// If `s` is a signed integer (optional `+`/`-` then ASCII digits), return its
/// canonical decimal form; otherwise `None`.
fn canonicalize_integer(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (negative, digits) = match bytes[0] {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: i128 = digits.parse().ok()?;
    let signed = if negative { -value } else { value };
    Some(signed.to_string())
}

/// Token-set Jaccard similarity between two answer strings, in `[0, 1]`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn answer_jaccard(a: &str, b: &str) -> f32 {
    let set_a = token_set(a);
    let set_b = token_set(b);
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    intersection as f32 / union as f32
}

/// Two answers are equivalent when their normalized forms are identical, or when
/// their token-set Jaccard similarity is at least `threshold`.
///
/// This is the loop's **correctness oracle**: a forward generation is kept only
/// if its answer is equivalent to the gold answer, and a backward
/// rationalization is kept only if it *reaches* the gold answer under this same
/// check. The normalized-equality fast path makes numeric answers robust to
/// trailing punctuation and leading zeros; the Jaccard fallback handles
/// multi-token answers that differ only in wording.
#[must_use]
pub fn answers_equivalent(a: &str, b: &str, threshold: f32) -> bool {
    if normalize_answer(a) == normalize_answer(b) {
        return true;
    }
    answer_jaccard(a, b) >= threshold
}

// ── Cheat-rationale detection ───────────────────────────────────────────────────

/// Filler tokens that carry no derivation content. A rationale built only from
/// these plus a restatement of the answer has shown no independent work.
const STOPWORDS: &[&str] = &[
    "the",
    "is",
    "a",
    "an",
    "because",
    "so",
    "therefore",
    "thus",
    "then",
    "and",
    "of",
    "to",
    "that",
    "this",
    "it",
    "answer",
    "must",
    "be",
    "we",
    "get",
    "have",
    "hence",
    "since",
    "as",
    "final",
    "result",
    "i",
    "its",
    "are",
    "was",
    "will",
    "by",
    "for",
    "with",
    "or",
    "no",
    "not",
    "but",
    "if",
    "which",
    "what",
    "in",
    "on",
    "at",
    "our",
    "here",
    "just",
    "simply",
    "obviously",
    "clearly",
];

/// Detect whether a backward-rationalized `rationale` merely restates the
/// provided `answer` without independent, problem-grounded derivation.
///
/// This is the single most load-bearing correctness check in the module. `STaR`'s
/// backward pass is handed the gold answer as a hint, and a rationale can "reach"
/// that answer by *cheating* — echoing the hint ("the answer is `X` because the
/// answer is `X`") with no reasoning. Such a rationale is worthless as training
/// signal: it teaches the model nothing except to parrot back whatever answer it
/// is shown. `STaR`'s entire validity rests on excluding these, so this function
/// rejects them and the engine drops any rationalization it flags.
///
/// # The rule
///
/// Tokenize all three inputs (lowercase alphanumeric). From the rationale's
/// tokens, remove every token that is part of the answer and every stopword,
/// leaving the **derivation residual** — the substantive content the rationale
/// contributes beyond restating the answer. A genuine derivation must exhibit
/// *both* of:
///
/// - **grounding** — at least one residual token that also appears in the
///   *problem*, i.e. the rationale actually engages with what was asked; and
/// - **an intermediate** — at least one residual token that appears in *neither*
///   the problem nor the answer, i.e. the rationale computes something new on the
///   way from the question to the answer.
///
/// A rationale that lacks either is a **cheat** and returns `true`. "The answer
/// is `X` because the answer is `X`" has an empty residual (fails both). "`X`
/// because `x` is `X`" (restating the problem's own value as the answer) has
/// grounding but no intermediate — no work was done — and is also rejected. A
/// real derivation such as "`7 + 8 = 15`, `15 mod 13 = 2`, so the answer is `2`"
/// keeps grounding (`7`) and intermediates (`8`, `15`, `mod`, `13`) even though
/// it legitimately mentions the answer, and is accepted.
///
/// The check is deliberately conservative and structural: it never inspects the
/// arithmetic (the module's trait is over free-text rationales, not numbers), so
/// it generalizes to any task whose rationales name the problem and show their
/// working.
#[must_use]
pub fn is_cheating_rationale(rationale: &str, problem: &str, answer: &str) -> bool {
    let rationale_tokens = token_set(rationale);
    let answer_tokens = token_set(answer);
    let problem_tokens = token_set(problem);
    let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();

    // Residual: rationale content that is neither an echo of the answer nor filler.
    let residual: Vec<&String> = rationale_tokens
        .iter()
        .filter(|t| !answer_tokens.contains(*t) && !stopwords.contains(t.as_str()))
        .collect();

    // Does the rationale engage the problem at all?
    let has_grounding = residual.iter().any(|t| problem_tokens.contains(*t));
    // Does the rationale compute anything the problem and answer did not already
    // state — an intermediate step distinct from both?
    let has_intermediate = residual.iter().any(|t| !problem_tokens.contains(*t));

    !(has_grounding && has_intermediate)
}
