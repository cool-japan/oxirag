//! Script-aware text primitives shared by the four pipeline layers.
//!
//! # Why this module exists
//!
//! Every heuristic in Layers 2, 3 and 4 was written against English and encodes two assumptions
//! that are both false for Japanese:
//!
//! 1. **Sentences end with `.`, `!` or `?`.** Japanese ends them with `。`, `！`, `？`. A Japanese
//!    paragraph therefore arrived at the claim extractor as ONE sentence.
//! 2. **Words are separated by whitespace.** Japanese has none. `split_whitespace()` on a Japanese
//!    sentence returns a single token, and every extractor in the crate bails out at
//!    `tokens.len() < 2`.
//!
//! The measured consequence, on a seven-document Japanese corpus: Layer 3 extracted **0 claims**
//! and Layer 4 extracted **0 entities**, while Layer 1 ranked perfectly (it works on character
//! bigrams). The pipeline reported `status: Unknown, summary: "No verifiable claims found"` —
//! which is not a wrong answer so much as no answer at all.
//!
//! This module is the shared substrate for fixing that. It does NOT attempt morphological
//! analysis: `MeCrab` does that properly and is a separate crate. What is here is the smaller,
//! honest thing — script detection, sentence segmentation that knows both punctuation families,
//! and particle-boundary segmentation, which is enough for the subject/predicate/object shapes the
//! extractors look for.

use std::collections::BTreeSet;

/// Sentence-final punctuation in both punctuation families, plus the newline.
///
/// `\n` is included because a draft is assembled by joining passages, and a passage that ends
/// without punctuation would otherwise run into the next one — that is exactly how
/// `MeCrab uses the IPADIC dicti OxiZ is a Pure Rust SMT solver` reached a verdict table.
const SENTENCE_ENDINGS: [char; 7] = ['.', '!', '?', '。', '！', '？', '\n'];

/// Japanese case particles that end a phrase — the segmentation boundaries this module uses in
/// place of whitespace.
const PARTICLES: [&str; 12] = [
    "は", "が", "を", "に", "へ", "と", "で", "や", "から", "まで", "より", "の",
];

/// Copula endings that mark a predicate-nominal sentence (`A は B です`).
const COPULAS: [&str; 8] = [
    "です",
    "だ",
    "である",
    "でした",
    "だった",
    "でございます",
    "であった",
    "なのです",
];

/// Endings that negate the sentence they close.
const NEGATIONS: [&str; 10] = [
    "ではありません",
    "ではない",
    "じゃない",
    "ありません",
    "ません",
    "ませんでした",
    "なかった",
    "ない",
    "ぬ",
    "いません",
];

/// True when `text` contains any character in a CJK script — Han, Hiragana or Katakana.
///
/// Used to pick between the whitespace path and the particle path. Deliberately a character-class
/// test rather than a language detector: a sentence with one Japanese clause in it still needs the
/// Japanese path, and a language detector would need a model this crate does not ship.
#[must_use]
pub fn has_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

/// True for a single Han, Hiragana or Katakana character (including the長音符 `ー`).
#[must_use]
pub fn is_cjk_char(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{309F}'   // hiragana
        | '\u{30A0}'..='\u{30FF}' // katakana, incl. ー
        | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
        | '\u{3400}'..='\u{4DBF}' // extension A
        | '\u{F900}'..='\u{FAFF}' // compatibility ideographs
    )
}

/// True for a Han character.
#[must_use]
pub fn is_han_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}'
    )
}

/// True for a Katakana character or the長音符.
#[must_use]
pub fn is_katakana_char(c: char) -> bool {
    matches!(c, '\u{30A1}'..='\u{30FA}' | '\u{30FC}')
}

/// True for a Hiragana character.
#[must_use]
pub fn is_hiragana_char(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{309F}')
}

/// Splits `text` into sentences, understanding both punctuation families.
///
/// Empty pieces are dropped, so a run of dots (`" ... "`, which the old draft assembler emitted)
/// contributes nothing rather than a stream of blank sentences. Each piece is trimmed of
/// whitespace and of the ideographic space `　`, which `str::trim` does handle but which is easy to
/// lose track of when reading this.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<&str> {
    text.split(SENTENCE_ENDINGS)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Normalises a sentence for equality comparison: lowercased, with every whitespace run collapsed
/// and all punctuation dropped.
///
/// This is what makes de-duplication work across the two ways the same passage reaches a draft —
/// once verbatim and once as a truncated context summary.
#[must_use]
pub fn normalize_for_compare(sentence: &str) -> String {
    let mut out = String::with_capacity(sentence.len());
    let mut pending_space = false;
    for c in sentence.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
        } else if c.is_alphanumeric() || is_cjk_char(c) {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.extend(c.to_lowercase());
        }
    }
    out
}

/// Splits a sentence into comparable tokens, choosing the strategy by script.
///
/// For text with no CJK this is the historical behaviour — whitespace split, non-alphanumerics
/// dropped, lowercased — so English extraction is bit-for-bit unchanged. For text with CJK it
/// segments on particle boundaries and script changes, which yields the content words
/// (`ペンギン`, `空`, `飛びません`) rather than one 30-character token.
#[must_use]
pub fn tokenize(sentence: &str) -> Vec<String> {
    if has_cjk(sentence) {
        segment_japanese(sentence)
    } else {
        sentence
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }
}

/// Segments Japanese text on particle boundaries and script transitions.
///
/// Not a morphological analyser and not pretending to be one. It works in three passes:
///
/// 1. Cut at every script transition and at every punctuation mark.
/// 2. Glue a hiragana run back onto the content word before it. Script transitions alone would cut
///    `飛びません` into `飛` + `びません`, splitting a verb from its own negation — and the
///    negation is the half that changes the meaning.
/// 3. Strip the grammatical tail that gluing just re-attached: a trailing particle or copula. So
///    `東京` + `は` becomes `東京`, while `飛` + `びません` stays `飛びません`.
///
/// `ペンギンは空を飛びません` becomes `["ペンギン", "空", "飛びません"]`. Good enough for "does this
/// sentence have a subject and something said about it", which is all the callers ask.
#[must_use]
pub fn segment_japanese(sentence: &str) -> Vec<String> {
    let mut runs: Vec<(CharClass, String)> = Vec::new();
    for c in sentence.chars() {
        let class = CharClass::of(c);
        if class == CharClass::Other {
            runs.push((CharClass::Other, String::new()));
            continue;
        }
        match runs.last_mut() {
            Some((last_class, text)) if *last_class == class => text.push(c),
            _ => runs.push((class, c.to_string())),
        }
    }

    let mut glued: Vec<String> = Vec::new();
    let mut previous_was_content = false;
    for (class, text) in runs {
        match class {
            CharClass::Other => previous_was_content = false,
            CharClass::Hiragana if previous_was_content => {
                if let Some(last) = glued.last_mut() {
                    last.push_str(&text);
                }
            }
            CharClass::Han | CharClass::Katakana => {
                glued.push(text);
                previous_was_content = true;
            }
            CharClass::Hiragana | CharClass::Latin => {
                glued.push(text);
                previous_was_content = false;
            }
        }
    }

    glued
        .into_iter()
        .map(|token| strip_grammatical_tail(&token).to_lowercase())
        .filter(|t| !t.is_empty() && !is_particle(t))
        .collect()
}

/// Removes one trailing particle or copula from a glued token.
///
/// Only one, and only from the end: `日本の` is `日本`, but `炊いたもの` is not `炊いた` — the `の`
/// there is inside a word, not closing the phrase.
fn strip_grammatical_tail(token: &str) -> &str {
    let (without_copula, had_copula) = strip_copula(token);
    if had_copula && !without_copula.is_empty() {
        return without_copula;
    }
    for particle in PARTICLES {
        if token.ends_with(particle) && token.len() > particle.len() {
            return &token[..token.len() - particle.len()];
        }
    }
    token
}

/// The script classes segmentation cuts between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharClass {
    /// Han ideographs.
    Han,
    /// Hiragana — mostly grammar, so it separates content words.
    Hiragana,
    /// Katakana, which in practice marks loanwords and is usually one whole term.
    Katakana,
    /// Latin letters and digits.
    Latin,
    /// Punctuation and whitespace: a hard cut, kept in no token.
    Other,
}

impl CharClass {
    fn of(c: char) -> Self {
        if is_han_char(c) {
            Self::Han
        } else if is_katakana_char(c) {
            Self::Katakana
        } else if is_hiragana_char(c) {
            Self::Hiragana
        } else if c.is_alphanumeric() {
            Self::Latin
        } else {
            Self::Other
        }
    }
}

/// True when the whole token is one of the case particles.
#[must_use]
pub fn is_particle(token: &str) -> bool {
    PARTICLES.contains(&token)
}

/// Splits `sentence` at its topic or subject particle (`は` / `が`), returning `(subject, rest)`.
///
/// Returns `None` when there is no such particle, when it is at the very start (no subject), or
/// when nothing follows it (nothing said about the subject). The FIRST particle wins: in
/// `富士山は日本で一番高い山です` that is the `は` after `富士山`, which is the topic — later `は`
/// occurrences belong to subordinate clauses and taking them would cut the sentence in the wrong
/// place.
#[must_use]
pub fn split_topic(sentence: &str) -> Option<(&str, &str)> {
    let mut best: Option<usize> = None;
    for marker in ["は", "が"] {
        if let Some(index) = sentence.find(marker)
            && index > 0
            && best.is_none_or(|b| index < b)
        {
            best = Some(index);
        }
    }
    let index = best?;
    // `は` is 3 bytes in UTF-8, and so is `が`; slice by the char's own length rather than a
    // literal 3 so this stays correct if the marker list ever grows.
    let marker_len = sentence[index..].chars().next().map_or(0, char::len_utf8);
    let subject = sentence[..index].trim();
    let rest = sentence[index + marker_len..].trim();
    if subject.is_empty() || rest.is_empty() {
        return None;
    }
    Some((subject, rest))
}

/// Strips a trailing copula from a predicate phrase, returning `(phrase, true)` when one was
/// found.
///
/// `日本の首都です` → `("日本の首都", true)`. `空を飛びます` → `("空を飛びます", false)`.
#[must_use]
pub fn strip_copula(phrase: &str) -> (&str, bool) {
    let trimmed = phrase.trim_end_matches(['。', '．', '.', '、']);
    // Longest match first, so `である` is not mistaken for a bare `だ` plus noise.
    let mut best: Option<&str> = None;
    for copula in COPULAS {
        if trimmed.ends_with(copula)
            && trimmed.len() > copula.len()
            && best.is_none_or(|b| copula.len() > b.len())
        {
            best = Some(copula);
        }
    }
    match best {
        Some(copula) => (trimmed[..trimmed.len() - copula.len()].trim_end(), true),
        None => (trimmed, false),
    }
}

/// True when the phrase ends in a negation.
///
/// Checked on the phrase rather than on tokens because Japanese negation is a suffix, not a word:
/// `飛びません` has no separable `not` to find in a token list.
#[must_use]
pub fn is_negated(phrase: &str) -> bool {
    let trimmed = phrase.trim_end_matches(['。', '．', '.', '、']);
    NEGATIONS.iter().any(|n| trimmed.ends_with(n))
}

/// Extracts the noun-phrase candidates from Japanese text: maximal Han runs and maximal Katakana
/// runs of at least two characters.
///
/// This is the Japanese counterpart of "consecutive capitalised words", which is how the English
/// entity extractor finds proper nouns and which finds nothing at all here — Japanese has no case.
/// Runs shorter than two characters are dropped: single kanji are overwhelmingly function-ish
/// (`人`, `事`, `方`) and flood the graph with nodes nobody asked about.
///
/// Returned in first-appearance order, de-duplicated.
#[must_use]
pub fn japanese_noun_candidates(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_class: Option<CharClass> = None;

    let mut flush = |current: &mut String, class: Option<CharClass>| {
        let word = std::mem::take(current);
        if !matches!(class, Some(CharClass::Han | CharClass::Katakana)) {
            return;
        }
        let word = strip_common_prefix(&word).to_string();
        if word.chars().count() < 2 || is_common_word(&word) {
            return;
        }
        if seen.insert(word.clone()) {
            out.push(word);
        }
    };

    for c in text.chars() {
        let class = CharClass::of(c);
        if current_class != Some(class) {
            flush(&mut current, current_class);
            current_class = Some(class);
        }
        if matches!(class, CharClass::Han | CharClass::Katakana) {
            current.push(c);
        }
    }
    flush(&mut current, current_class);

    out
}

/// Words that pass the "two or more kanji" test but name nothing.
///
/// Kept deliberately short. A long stoplist is a dictionary in disguise, and a wrong entry silently
/// removes a real entity — the failure this list exists to avoid is a graph full of `一番` and
/// `場合`, not a graph that is missing a noun.
const COMMON_WORDS: [&str; 24] = [
    "一番", "場合", "今日", "明日", "昨日", "今回", "自分", "全部", "一部", "以上", "以下", "以外",
    "程度", "多く", "沢山", "本当", "普通", "最近", "現在", "将来", "内容", "関係", "必要", "可能",
];

/// True for a token in [`COMMON_WORDS`].
#[must_use]
pub fn is_common_word(word: &str) -> bool {
    COMMON_WORDS.contains(&word)
}

/// Removes a leading [`COMMON_WORDS`] entry from a Han run.
///
/// Japanese runs kanji together across a word boundary, so `日本で一番高い山` yields the run
/// `一番高` — `一番` (in the stoplist) glued to the stem of `高い`. Dropping the prefix leaves one
/// character, which is below the run-length floor, so the candidate disappears. Without this the
/// knowledge graph gains a node called `一番高`.
fn strip_common_prefix(word: &str) -> &str {
    for common in COMMON_WORDS {
        if let Some(rest) = word.strip_prefix(common) {
            return rest;
        }
    }
    word
}

/// Scores how well `sentence` answers a query whose tokens are `query_tokens`.
///
/// Returns the fraction of distinct query tokens the sentence contains, in `0.0..=1.0`. A token
/// counts as present when it appears as a token of the sentence OR as a substring of it — the
/// substring arm is what makes `東京` match `東京都`, which token equality alone would miss because
/// the segmenter has no dictionary and cannot know `東京都` decomposes.
///
/// This is the whole of the relevance judgement used to build an extractive answer. It is a weak
/// signal by design: it is checkable by a reader, which a learned scorer would not be, and this
/// crate's promise is that every step of the pipeline can be inspected.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn overlap_score(query_tokens: &[String], sentence: &str) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let sentence_lower = sentence.to_lowercase();
    let sentence_tokens = tokenize(sentence);
    let matched = query_tokens
        .iter()
        .filter(|token| {
            sentence_tokens.iter().any(|t| t == *token) || sentence_lower.contains(token.as_str())
        })
        .count();
    matched as f32 / query_tokens.len() as f32
}

/// The distinct, meaningful tokens of a query, in first-appearance order.
///
/// Drops one-character Latin tokens and the handful of English function words that would otherwise
/// match every sentence in the corpus and flatten [`overlap_score`] to a constant.
#[must_use]
pub fn query_tokens(query: &str) -> Vec<String> {
    const STOP_WORDS: [&str; 20] = [
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "and",
        "or", "it", "its", "this", "that", "with",
    ];
    let mut seen = BTreeSet::new();
    tokenize(query)
        .into_iter()
        .filter(|token| {
            let long_enough = token.chars().count() > 1 || has_cjk(token);
            long_enough && !STOP_WORDS.contains(&token.as_str())
        })
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

/// Joins sentences back into a paragraph, restoring the terminator each one lost when it was split.
///
/// [`split_sentences`] drops the delimiter, so a naive `join(" ")` produces text that splits back
/// into ONE sentence. That is not cosmetic: an extractive draft is assembled this way, and a draft
/// that is one 200-character "sentence" reaches the claim extractor as one claim and the revision
/// step's "is this sentence already present?" check as one opaque string. Both symptoms were
/// visible on the page — a verdict table whose first row was the entire answer, and an answer with
/// every sentence in it twice.
///
/// The terminator follows the script of the sentence it closes: `。` with no space after it for
/// Japanese, `.` plus a space for everything else.
#[must_use]
pub fn join_sentences<S: AsRef<str>>(sentences: &[S]) -> String {
    let mut out = String::new();
    for sentence in sentences {
        let sentence = sentence.as_ref().trim();
        if sentence.is_empty() {
            continue;
        }
        if has_cjk(sentence) {
            out.push_str(sentence);
            out.push('。');
        } else {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(sentence);
            out.push('.');
        }
    }
    out.trim_end().to_string()
}

/// True when `sentence` is already present in `text`, compared normalised.
///
/// Substring rather than sentence equality on purpose: the caller cannot rely on `text` being
/// split-able into the same sentences, and the question being asked — "would adding this repeat
/// something already said?" — is answered correctly either way.
#[must_use]
pub fn contains_sentence(text: &str, sentence: &str) -> bool {
    let needle = normalize_for_compare(sentence);
    !needle.is_empty() && normalize_for_compare(text).contains(&needle)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn splits_on_both_punctuation_families() {
        assert_eq!(
            split_sentences("東京は日本の首都です。人口は約1400万人です。"),
            vec!["東京は日本の首都です", "人口は約1400万人です"]
        );
        assert_eq!(split_sentences("A is B. C is D!"), vec!["A is B", "C is D"]);
    }

    #[test]
    fn a_newline_ends_a_sentence() {
        // The concrete regression: joined passages used to run together into one "sentence".
        assert_eq!(
            split_sentences("MeCrab uses the IPADIC dicti\n\nOxiZ is a solver"),
            vec!["MeCrab uses the IPADIC dicti", "OxiZ is a solver"]
        );
    }

    #[test]
    fn runs_of_dots_contribute_nothing() {
        assert_eq!(split_sentences("one ... two"), vec!["one", "two"]);
    }

    #[test]
    fn english_tokenization_is_unchanged() {
        assert_eq!(
            tokenize("The capital, of France!"),
            vec!["the", "capital", "of", "france"]
        );
    }

    #[test]
    fn japanese_segments_into_content_words() {
        assert_eq!(
            segment_japanese("ペンギンは空を飛びません"),
            vec!["ペンギン", "空", "飛びません"]
        );
    }

    #[test]
    fn particles_are_dropped_but_content_hiragana_is_not() {
        let tokens = segment_japanese("ごはんはお米を炊いたものです");
        assert!(tokens.iter().all(|t| t != "は" && t != "を"));
        assert!(tokens.iter().any(|t| t == "米"));
    }

    #[test]
    fn topic_splits_at_the_first_marker() {
        assert_eq!(
            split_topic("富士山は日本で一番高い山です"),
            Some(("富士山", "日本で一番高い山です"))
        );
        assert_eq!(split_topic("はじめに"), None);
        assert_eq!(split_topic("東京は"), None);
        assert_eq!(split_topic("no particle here"), None);
    }

    #[test]
    fn copulas_are_stripped_longest_first() {
        assert_eq!(strip_copula("日本の首都です"), ("日本の首都", true));
        assert_eq!(strip_copula("鳥である"), ("鳥", true));
        assert_eq!(strip_copula("空を飛びます"), ("空を飛びます", false));
    }

    #[test]
    fn negation_is_a_suffix_not_a_word() {
        assert!(is_negated("空を飛びません"));
        assert!(is_negated("エンジンがありません。"));
        assert!(!is_negated("空を飛びます"));
    }

    #[test]
    fn noun_candidates_are_han_and_katakana_runs() {
        let found = japanese_noun_candidates("ペンギンは鳥です。富士山は静岡県にあります。");
        assert!(found.contains(&"ペンギン".to_string()));
        assert!(found.contains(&"富士山".to_string()));
        assert!(found.contains(&"静岡県".to_string()));
        // `鳥` is one character — below the run-length floor.
        assert!(!found.contains(&"鳥".to_string()));
    }

    #[test]
    fn noun_candidates_are_deduplicated_in_order() {
        assert_eq!(
            japanese_noun_candidates("東京は日本の首都。東京の人口。"),
            vec!["東京", "日本", "首都", "人口"]
        );
    }

    #[test]
    fn normalization_makes_the_same_passage_compare_equal() {
        assert_eq!(
            normalize_for_compare("OxiZ is a  solver."),
            normalize_for_compare("oxiz is a solver")
        );
    }

    #[test]
    fn overlap_matches_japanese_by_substring_too() {
        let tokens = query_tokens("東京 人口");
        assert!(overlap_score(&tokens, "東京都の人口は約1400万人です") > 0.99);
        assert!(overlap_score(&tokens, "ペンギンは鳥です") < f32::EPSILON);
    }

    #[test]
    fn query_tokens_drop_function_words() {
        assert_eq!(
            query_tokens("what is the capital of France"),
            vec!["what", "capital", "france"]
        );
    }

    #[test]
    fn joining_sentences_keeps_them_separable() {
        let joined = join_sentences(&["富士山は日本で一番高い山です", "高さは3776メートルです"]);
        assert_eq!(
            joined,
            "富士山は日本で一番高い山です。高さは3776メートルです。"
        );
        assert_eq!(split_sentences(&joined).len(), 2);

        let english = join_sentences(&["OxiZ is a solver", "It runs in the browser"]);
        assert_eq!(english, "OxiZ is a solver. It runs in the browser.");
        assert_eq!(split_sentences(&english).len(), 2);
    }

    #[test]
    fn presence_is_checked_normalised() {
        let text = "富士山は日本で一番高い山です。高さは3776メートルです。";
        assert!(contains_sentence(text, "高さは3776メートルです"));
        assert!(!contains_sentence(text, "富士山は静岡県にあります"));
    }

    #[test]
    fn has_cjk_distinguishes_the_two_paths() {
        assert!(has_cjk("東京"));
        assert!(has_cjk("Tokyo と Osaka"));
        assert!(!has_cjk("Tokyo and Osaka"));
    }
}
