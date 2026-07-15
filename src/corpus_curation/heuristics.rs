//! Pillar 1 — heuristic quality signals in the style of `DeepMind`'s `Gopher`
//! (Rae et al., 2021) and Google's `C4` (Raffel et al., 2019) corpus filters.
//!
//! Each [`QualityRule`] is a cheap, deterministic statistic computed directly
//! over a document's text — word count, mean word length, symbol density,
//! stop-word density, line-level repetition, bullet/ellipsis density, and
//! alphabetic-character density — compared against a configurable threshold.
//! [`evaluate_quality_rules`] runs the whole configured rule set over one
//! document and returns a [`QualityVerdict`]: a pass/fail per rule *with a
//! human-readable reason*, plus an overall `passed` that is the logical AND of
//! every rule (mirroring the source papers, where heuristic filters are a hard
//! gate applied before any learned scoring).

use std::collections::HashSet;

// ── stop words ───────────────────────────────────────────────────────────────

/// A small, hand-rolled English stop-word list, shared by the stop-word-ratio
/// heuristic here and by the n-gram window filter in
/// [`super::contamination`]. Not exhaustive by design — it exists to bias
/// statistics, not to build a general NLP stop-word filter.
pub(crate) const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "if", "in", "into", "is", "it", "its",
    "may", "might", "no", "not", "of", "on", "onto", "or", "over", "should", "so", "than", "that",
    "the", "their", "then", "there", "these", "this", "those", "to", "under", "was", "were",
    "will", "with", "would",
];

/// `true` when `word` (already lowercased) is in [`STOPWORDS`].
pub(crate) fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

// ── DocumentStats ────────────────────────────────────────────────────────────

/// The raw statistics [`QualityRule`] variants are evaluated against, computed
/// once per document by [`DocumentStats::compute`] and shared across every
/// rule in the configured set.
#[derive(Debug, Clone, PartialEq)]
struct DocumentStats {
    /// Number of whitespace-separated tokens.
    word_count: usize,
    /// Mean character length of a whitespace-separated token.
    mean_word_length: f64,
    /// `#`/ellipsis "symbol" occurrences per word.
    symbol_to_word_ratio: f64,
    /// Fraction of tokens that are common English stop words.
    stopword_ratio: f64,
    /// Fraction of non-empty lines that are exact duplicates of an earlier
    /// non-empty line.
    duplicate_line_ratio: f64,
    /// Fraction of non-empty lines that open with a bullet marker.
    bullet_line_ratio: f64,
    /// Fraction of non-empty lines that close with an ellipsis.
    ellipsis_line_ratio: f64,
    /// Fraction of non-whitespace characters that are alphabetic.
    alphabetic_char_ratio: f64,
}

/// Count "symbol" occurrences in `text`: every `#` character, every maximal
/// run of three or more `.` characters (an ASCII ellipsis), and every Unicode
/// ellipsis character (`…`).
fn count_symbols(text: &str) -> usize {
    let hash_count = text.chars().filter(|&c| c == '#').count();
    let unicode_ellipsis_count = text.chars().filter(|&c| c == '…').count();

    let mut ascii_ellipsis_count = 0usize;
    let mut run = 0usize;
    for c in text.chars() {
        if c == '.' {
            run += 1;
        } else {
            if run >= 3 {
                ascii_ellipsis_count += 1;
            }
            run = 0;
        }
    }
    if run >= 3 {
        ascii_ellipsis_count += 1;
    }

    hash_count + unicode_ellipsis_count + ascii_ellipsis_count
}

/// `true` when trimmed line `line` opens with a bullet marker: a literal
/// bullet glyph, a dash/asterisk bullet, or a numbered-list prefix
/// (`"1. "`, `"12) "`, ...).
fn is_bullet_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed
        .strip_prefix('-')
        .or_else(|| trimmed.strip_prefix('*'))
        .or_else(|| trimmed.strip_prefix('•'))
        .or_else(|| trimmed.strip_prefix('·'))
        .or_else(|| trimmed.strip_prefix('‣'))
    {
        return rest.is_empty() || rest.starts_with(char::is_whitespace);
    }
    // Numbered-list prefix: one or more digits followed by '.' or ')' then
    // whitespace (or end of line).
    let digits_end = trimmed
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map_or(trimmed.len(), |(i, _)| i);
    if digits_end == 0 {
        return false;
    }
    let rest = &trimmed[digits_end..];
    rest.starts_with(". ") || rest.starts_with(") ") || rest == "." || rest == ")"
}

/// `true` when trimmed line `line` closes with an ellipsis (`"..."` or `…`).
fn is_ellipsis_line(line: &str) -> bool {
    line.ends_with("...") || line.ends_with('…')
}

impl DocumentStats {
    /// Compute every statistic over `content` in one pass.
    #[allow(clippy::cast_precision_loss)]
    fn compute(content: &str) -> Self {
        let words: Vec<&str> = content.split_whitespace().collect();
        let word_count = words.len();

        let mean_word_length = if word_count == 0 {
            0.0
        } else {
            words.iter().map(|w| w.chars().count()).sum::<usize>() as f64 / word_count as f64
        };

        let symbol_to_word_ratio = if word_count == 0 {
            0.0
        } else {
            count_symbols(content) as f64 / word_count as f64
        };

        let stopword_ratio = if word_count == 0 {
            0.0
        } else {
            let hits = words
                .iter()
                .filter(|w| is_stopword(&w.to_lowercase()))
                .count();
            hits as f64 / word_count as f64
        };

        let lines: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        let total_lines = lines.len();

        let duplicate_line_ratio = if total_lines == 0 {
            0.0
        } else {
            let unique: HashSet<&String> = lines.iter().collect();
            (total_lines - unique.len()) as f64 / total_lines as f64
        };

        let bullet_line_ratio = if total_lines == 0 {
            0.0
        } else {
            lines.iter().filter(|l| is_bullet_line(l)).count() as f64 / total_lines as f64
        };

        let ellipsis_line_ratio = if total_lines == 0 {
            0.0
        } else {
            lines.iter().filter(|l| is_ellipsis_line(l)).count() as f64 / total_lines as f64
        };

        let non_ws_chars: Vec<char> = content.chars().filter(|c| !c.is_whitespace()).collect();
        let alphabetic_char_ratio = if non_ws_chars.is_empty() {
            0.0
        } else {
            non_ws_chars.iter().filter(|c| c.is_alphabetic()).count() as f64
                / non_ws_chars.len() as f64
        };

        Self {
            word_count,
            mean_word_length,
            symbol_to_word_ratio,
            stopword_ratio,
            duplicate_line_ratio,
            bullet_line_ratio,
            ellipsis_line_ratio,
            alphabetic_char_ratio,
        }
    }
}

// ── QualityRule ──────────────────────────────────────────────────────────────

/// One configurable heuristic quality rule (`Gopher`/`C4`-style), carrying its
/// own threshold(s).
///
/// Every variant's `evaluate` compares one pre-computed document statistic
/// against its threshold(s); bounds are **inclusive** (a value exactly at the
/// threshold passes).
#[derive(Debug, Clone, PartialEq)]
pub enum QualityRule {
    /// Word count must fall in `[min, max]`. Catches near-empty stubs and
    /// runaway scrapes.
    WordCount {
        /// Minimum word count (inclusive).
        min: usize,
        /// Maximum word count (inclusive).
        max: usize,
    },
    /// Mean word length (characters) must fall in `[min, max]`. Catches
    /// character-soup and single-character-token spam.
    MeanWordLength {
        /// Minimum mean word length (inclusive).
        min: f64,
        /// Maximum mean word length (inclusive).
        max: f64,
    },
    /// `#`/ellipsis symbol occurrences per word must not exceed `max`.
    SymbolToWordRatio {
        /// Maximum symbol-to-word ratio (inclusive).
        max: f64,
    },
    /// Fraction of words that are stop words must be at least `min`. Natural
    /// prose contains common function words; text lacking them is often
    /// keyword-stuffed spam or boilerplate.
    StopWordRatio {
        /// Minimum stop-word ratio (inclusive).
        min: f64,
    },
    /// Fraction of non-empty lines that repeat an earlier line must not
    /// exceed `max`. Catches navigation menus and boilerplate repeats.
    DuplicateLineRatio {
        /// Maximum duplicate-line ratio (inclusive).
        max: f64,
    },
    /// Fraction of non-empty lines opening with a bullet marker must not
    /// exceed `max`. Catches link-farm / list-dump pages.
    BulletLineRatio {
        /// Maximum bullet-line ratio (inclusive).
        max: f64,
    },
    /// Fraction of non-empty lines closing with an ellipsis must not exceed
    /// `max`. Catches truncated listicle/preview scrapes.
    EllipsisLineRatio {
        /// Maximum ellipsis-line ratio (inclusive).
        max: f64,
    },
    /// Fraction of non-whitespace characters that are alphabetic must be at
    /// least `min`. Catches numeric/symbol-dominated junk.
    AlphabeticCharRatio {
        /// Minimum alphabetic-character ratio (inclusive).
        min: f64,
    },
}

impl QualityRule {
    /// A stable, human-readable name for this rule kind, used in
    /// [`QualitySignal::name`].
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::WordCount { .. } => "word_count",
            Self::MeanWordLength { .. } => "mean_word_length",
            Self::SymbolToWordRatio { .. } => "symbol_to_word_ratio",
            Self::StopWordRatio { .. } => "stopword_ratio",
            Self::DuplicateLineRatio { .. } => "duplicate_line_ratio",
            Self::BulletLineRatio { .. } => "bullet_line_ratio",
            Self::EllipsisLineRatio { .. } => "ellipsis_line_ratio",
            Self::AlphabeticCharRatio { .. } => "alphabetic_char_ratio",
        }
    }

    /// The standard `Gopher`/`C4`-inspired rule set, with defaults tuned for
    /// general prose:
    ///
    /// | Rule | Threshold |
    /// |------|-----------|
    /// | `word_count` | `[50, 100_000]` |
    /// | `mean_word_length` | `[3.0, 10.0]` |
    /// | `symbol_to_word_ratio` | `<= 0.1` |
    /// | `stopword_ratio` | `>= 0.06` |
    /// | `duplicate_line_ratio` | `<= 0.3` |
    /// | `bullet_line_ratio` | `<= 0.9` |
    /// | `ellipsis_line_ratio` | `<= 0.3` |
    /// | `alphabetic_char_ratio` | `>= 0.6` |
    #[must_use]
    pub fn default_rule_set() -> Vec<Self> {
        vec![
            Self::WordCount {
                min: 50,
                max: 100_000,
            },
            Self::MeanWordLength {
                min: 3.0,
                max: 10.0,
            },
            Self::SymbolToWordRatio { max: 0.1 },
            Self::StopWordRatio { min: 0.06 },
            Self::DuplicateLineRatio { max: 0.3 },
            Self::BulletLineRatio { max: 0.9 },
            Self::EllipsisLineRatio { max: 0.3 },
            Self::AlphabeticCharRatio { min: 0.6 },
        ]
    }

    /// Evaluate this rule against pre-computed `stats`, producing a
    /// [`QualitySignal`] with the observed value, the pass/fail verdict, and a
    /// human-readable reason.
    #[allow(clippy::cast_precision_loss)]
    fn evaluate(&self, stats: &DocumentStats) -> QualitySignal {
        match *self {
            Self::WordCount { min, max } => {
                let value = stats.word_count as f64;
                let passed = stats.word_count >= min && stats.word_count <= max;
                let reason = format!(
                    "word count {} {} bounds [{min}, {max}]",
                    stats.word_count,
                    if passed { "within" } else { "outside" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::MeanWordLength { min, max } => {
                let value = stats.mean_word_length;
                let passed = value >= min && value <= max;
                let reason = format!(
                    "mean word length {value:.3} {} bounds [{min}, {max}]",
                    if passed { "within" } else { "outside" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::SymbolToWordRatio { max } => {
                let value = stats.symbol_to_word_ratio;
                let passed = value <= max;
                let reason = format!(
                    "symbol-to-word ratio {value:.3} {} maximum {max}",
                    if passed { "at or below" } else { "above" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::StopWordRatio { min } => {
                let value = stats.stopword_ratio;
                let passed = value >= min;
                let reason = format!(
                    "stop-word ratio {value:.3} {} minimum {min}",
                    if passed { "at or above" } else { "below" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::DuplicateLineRatio { max } => {
                let value = stats.duplicate_line_ratio;
                let passed = value <= max;
                let reason = format!(
                    "duplicate-line ratio {value:.3} {} maximum {max}",
                    if passed { "at or below" } else { "above" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::BulletLineRatio { max } => {
                let value = stats.bullet_line_ratio;
                let passed = value <= max;
                let reason = format!(
                    "bullet-line ratio {value:.3} {} maximum {max}",
                    if passed { "at or below" } else { "above" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::EllipsisLineRatio { max } => {
                let value = stats.ellipsis_line_ratio;
                let passed = value <= max;
                let reason = format!(
                    "ellipsis-line ratio {value:.3} {} maximum {max}",
                    if passed { "at or below" } else { "above" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
            Self::AlphabeticCharRatio { min } => {
                let value = stats.alphabetic_char_ratio;
                let passed = value >= min;
                let reason = format!(
                    "alphabetic-character ratio {value:.3} {} minimum {min}",
                    if passed { "at or above" } else { "below" }
                );
                QualitySignal::new(self.name(), value, passed, reason)
            }
        }
    }
}

// ── QualitySignal / QualityVerdict ──────────────────────────────────────────

/// The outcome of evaluating one [`QualityRule`] against one document.
#[derive(Debug, Clone, PartialEq)]
pub struct QualitySignal {
    /// The rule's stable name (see [`QualityRule::name`]).
    pub name: &'static str,
    /// The observed statistic the rule compared against its threshold.
    pub value: f64,
    /// `true` when the observed value satisfies the rule's threshold(s).
    pub passed: bool,
    /// A human-readable explanation of the verdict, including the observed
    /// value and the configured threshold.
    pub reason: String,
}

impl QualitySignal {
    /// Construct a signal.
    #[must_use]
    fn new(name: &'static str, value: f64, passed: bool, reason: String) -> Self {
        Self {
            name,
            value,
            passed,
            reason,
        }
    }
}

/// The aggregate outcome of running a whole [`QualityRule`] set against one
/// document.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityVerdict {
    /// One [`QualitySignal`] per configured rule, in rule-set order.
    pub signals: Vec<QualitySignal>,
    /// `true` only when every signal passed (the rules are a hard, `Gopher`/
    /// `C4`-style AND-gate, not a blended score).
    pub passed: bool,
}

impl QualityVerdict {
    /// Iterate the signals that failed, in rule-set order.
    pub fn failed_signals(&self) -> impl Iterator<Item = &QualitySignal> {
        self.signals.iter().filter(|s| !s.passed)
    }

    /// The fraction of rules that passed, in `[0, 1]`. `1.0` for an empty rule
    /// set (vacuously, nothing failed).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pass_rate(&self) -> f64 {
        if self.signals.is_empty() {
            return 1.0;
        }
        let passed = self.signals.iter().filter(|s| s.passed).count();
        passed as f64 / self.signals.len() as f64
    }
}

/// Evaluate `rules` against `content`, returning a [`QualityVerdict`].
///
/// Statistics are computed once and shared across every rule. An empty `rules`
/// slice yields a vacuous pass (`passed: true`, no signals).
#[must_use]
pub fn evaluate_quality_rules(content: &str, rules: &[QualityRule]) -> QualityVerdict {
    let stats = DocumentStats::compute(content);
    let signals: Vec<QualitySignal> = rules.iter().map(|r| r.evaluate(&stats)).collect();
    let passed = signals.iter().all(|s| s.passed);
    QualityVerdict { signals, passed }
}
