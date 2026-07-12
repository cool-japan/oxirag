//! The surrogate language model that drives compression, plus the shared
//! text-processing helpers (sentence splitting, surface tokenisation, token
//! normalisation).
//!
//! # Why a self-fit n-gram model?
//!
//! The real `LLMLingua` (Jiang et al., 2023) scores each token's *surprisal*
//! (its perplexity contribution) with a small pretrained causal LM and prunes
//! the low-surprisal — highly predictable, hence low-information — tokens. This
//! crate has no pretrained-model access, so instead of faking that signal with
//! a stopword list or a word-length rule, [`PerplexityModel`] builds a genuine
//! smoothed n-gram language model **fit on the input context itself** (an
//! optional larger reference corpus can be prepended, see
//! [`PerplexityModel::fit_with_reference`]) and evaluates real token surprisal
//! under it.
//!
//! # Smoothing: Jelinek-Mercer interpolation with an add-k unigram base
//!
//! We use **linear (Jelinek-Mercer) interpolation** with recursive backoff and
//! an **add-k** smoothed unigram base, chosen because — unlike Stupid/Katz
//! backoff, which produce an *unnormalised* score — interpolation yields a
//! *proper conditional probability distribution* that sums to one over the
//! vocabulary (verified in this module's tests), which is exactly what a
//! faithful surprisal estimate `-log2 P(token | history)` requires.
//!
//! For an order-`o` context `c = (w_{i-o+1}, ..., w_{i-1})`:
//!
//! ```text
//! P_o(w | c) = lambda * P_ML(w | c) + (1 - lambda) * P_{o-1}(w | c[1..])   if c was seen
//!            = P_{o-1}(w | c[1..])                                          otherwise
//! P_1(w)     = (count(w) + k) / (N + k * (V + 1))          [add-k, includes <unk> mass]
//! ```
//!
//! where `P_ML(w | c) = count(c, w) / count(c)` uses the **context total**
//! (the sum of higher-order gram counts sharing prefix `c`), so the maximum-
//! likelihood term integrates to exactly one and the interpolation therefore
//! preserves normalisation at every level. The add-k base gives every token —
//! including any never seen (`<unk>`) — strictly positive probability, so no
//! surprisal is ever infinite.

use std::collections::HashMap;

/// Separator byte used to build compound n-gram / context keys. `\u{1F}` (unit
/// separator) never appears inside a normalised token, so keys are unambiguous.
const KEY_SEP: char = '\u{1F}';

/// Normalised form assigned to a surface token that has no alphanumeric
/// content (pure punctuation), keeping the model's token stream aligned 1:1
/// with the surface stream.
const PUNCT_SYMBOL: &str = "<punct>";

// ── text helpers ───────────────────────────────────────────────────────────────

/// Splits `text` into sentence-like segments.
///
/// A boundary is a `.`, `!`, or `?` immediately followed by whitespace or the
/// end of input; newlines also terminate a segment. Each returned segment is
/// trimmed and non-empty. When `text` has no such boundary the whole trimmed
/// input is returned as a single segment.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let c = chars[i];

        if c == '\n' {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                segments.push(trimmed.to_string());
            }
            current.clear();
            i += 1;
            continue;
        }

        current.push(c);

        if matches!(c, '.' | '!' | '?') {
            let at_end = i + 1 >= n;
            let at_break = i + 1 < n && (chars[i + 1] == ' ' || chars[i + 1] == '\n');
            if at_end || at_break {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    segments.push(trimmed.to_string());
                }
                current.clear();
                i += 1;
                continue;
            }
        }
        i += 1;
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }

    segments
}

/// Splits a segment into surface tokens on ASCII/Unicode whitespace, preserving
/// each token's exact original characters (including attached punctuation) so
/// the compressed text can be reconstructed faithfully.
#[must_use]
pub fn surface_tokens(segment: &str) -> Vec<String> {
    segment.split_whitespace().map(str::to_string).collect()
}

/// Normalises a surface token into the symbol used by the language model:
/// lower-cased, with leading/trailing non-alphanumeric characters stripped.
///
/// A token with no alphanumeric content collapses to `PUNCT_SYMBOL` so the
/// normalised stream stays aligned one-to-one with the surface stream.
#[must_use]
pub fn normalize_token(surface: &str) -> String {
    let trimmed: String = surface
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    if trimmed.is_empty() {
        PUNCT_SYMBOL.to_string()
    } else {
        trimmed
    }
}

// ── PerplexityModel ────────────────────────────────────────────────────────────

/// One order's count tables: full-gram counts and their shared context totals.
#[derive(Debug, Clone, Default)]
struct OrderTable {
    /// Map from an order-`o` gram key (`ctx ++ token`) to its occurrence count.
    grams: HashMap<String, u64>,
    /// Map from an order-`(o-1)` context key to the total count of all grams
    /// sharing that context — the maximum-likelihood denominator.
    context_totals: HashMap<String, u64>,
}

/// A smoothed n-gram surrogate language model used to estimate token surprisal.
///
/// Fit with [`PerplexityModel::fit`] (or [`PerplexityModel::fit_with_reference`]
/// to blend in a larger corpus), then query [`PerplexityModel::probability`]
/// or [`PerplexityModel::surprisal`]. The model is immutable after fitting and
/// safe to share across threads.
///
/// See the [module documentation](self) for the smoothing scheme.
#[derive(Debug, Clone)]
pub struct PerplexityModel {
    /// Model order (`1` = unigram, `2` = bigram, ...).
    order: usize,
    /// Add-k unigram smoothing constant.
    add_k: f64,
    /// Jelinek-Mercer higher-order interpolation weight.
    lambda: f64,
    /// Unigram counts keyed by normalised token.
    unigram_counts: HashMap<String, u64>,
    /// Total number of tokens fitted (unigram denominator `N`).
    total_tokens: usize,
    /// Count tables for orders `2..=order`, indexed by order (slots `0` and `1`
    /// are unused placeholders).
    higher: Vec<OrderTable>,
}

impl PerplexityModel {
    /// Fits the model on a single normalised token stream.
    ///
    /// `order` is clamped to at least `1`; `add_k` and `lambda` are the
    /// smoothing constants documented on
    /// [`LlmLinguaConfig`](super::types::LlmLinguaConfig). This performs no
    /// validation — callers should validate the owning config first.
    #[must_use]
    pub fn fit(tokens: &[String], order: usize, add_k: f64, lambda: f64) -> Self {
        Self::fit_with_reference(tokens, &[], order, add_k, lambda)
    }

    /// Fits the model on `tokens` after seeding its counts with an optional
    /// larger `reference` corpus (also normalised tokens).
    ///
    /// Both streams contribute counts identically; the reference lets callers
    /// widen the model's statistical support beyond the immediate context.
    /// `order` is clamped to at least `1`.
    #[must_use]
    pub fn fit_with_reference(
        tokens: &[String],
        reference: &[String],
        order: usize,
        add_k: f64,
        lambda: f64,
    ) -> Self {
        let order = order.max(1);
        let mut unigram_counts: HashMap<String, u64> = HashMap::new();
        let mut higher: Vec<OrderTable> = (0..=order).map(|_| OrderTable::default()).collect();

        for stream in [reference, tokens] {
            for token in stream {
                *unigram_counts.entry(token.clone()).or_insert(0) += 1;
            }
            for o in 2..=order {
                if stream.len() < o {
                    continue;
                }
                for window in stream.windows(o) {
                    let ctx_key = window[..o - 1].join(&KEY_SEP.to_string());
                    let gram_key = format!("{ctx_key}{KEY_SEP}{}", window[o - 1]);
                    let table = &mut higher[o];
                    *table.grams.entry(gram_key).or_insert(0) += 1;
                    *table.context_totals.entry(ctx_key).or_insert(0) += 1;
                }
            }
        }

        let total_tokens = tokens.len() + reference.len();
        Self {
            order,
            add_k,
            lambda,
            unigram_counts,
            total_tokens,
            higher,
        }
    }

    /// The model order.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// The number of distinct normalised tokens (the vocabulary size `V`).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.unigram_counts.len()
    }

    /// The total number of tokens the model was fitted on (`N`).
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// The distinct normalised tokens in the vocabulary (unordered).
    #[must_use]
    pub fn vocabulary(&self) -> Vec<&str> {
        self.unigram_counts.keys().map(String::as_str).collect()
    }

    /// Add-k smoothed unigram probability, non-zero for every token including
    /// out-of-vocabulary ones (which draw the reserved `<unk>` mass).
    #[allow(clippy::cast_precision_loss)]
    fn unigram_probability(&self, token: &str) -> f64 {
        let vocab = self.unigram_counts.len() as f64;
        let n = self.total_tokens as f64;
        let count = self.unigram_counts.get(token).copied().unwrap_or(0) as f64;
        (count + self.add_k) / (n + self.add_k * (vocab + 1.0))
    }

    /// Recursive interpolated probability for a context of length `ctx.len()`
    /// (which must be `<= order - 1`); `ctx` is ordered oldest-to-newest.
    #[allow(clippy::cast_precision_loss)]
    fn probability_at(&self, ctx: &[&str], token: &str) -> f64 {
        if ctx.is_empty() {
            return self.unigram_probability(token);
        }
        // Backoff estimate drops the *oldest* context token.
        let lower = self.probability_at(&ctx[1..], token);
        let o = ctx.len() + 1;
        let table = &self.higher[o];
        let ctx_key = ctx.join(&KEY_SEP.to_string());
        let ctx_total = table.context_totals.get(&ctx_key).copied().unwrap_or(0);
        if ctx_total == 0 {
            // Unseen context: fully delegate to the lower order (keeps the
            // distribution normalised).
            return lower;
        }
        let gram_key = format!("{ctx_key}{KEY_SEP}{token}");
        let gram_count = table.grams.get(&gram_key).copied().unwrap_or(0);
        let max_likelihood = gram_count as f64 / ctx_total as f64;
        self.lambda * max_likelihood + (1.0 - self.lambda) * lower
    }

    /// Conditional probability `P(token | context)` under the smoothed model.
    ///
    /// `context` is the preceding tokens (oldest first); only its final
    /// `order - 1` entries are used. The result is always in `(0.0, 1.0]`.
    #[must_use]
    pub fn probability(&self, context: &[&str], token: &str) -> f64 {
        let max_ctx = self.order - 1;
        let start = context.len().saturating_sub(max_ctx);
        self.probability_at(&context[start..], token)
    }

    /// Token surprisal in **bits**: `-log2 P(token | context)`.
    ///
    /// Higher means less predictable, i.e. more information-dense — the tokens
    /// compression should preferentially keep. Always finite and `>= 0`
    /// because the smoothed probability is strictly positive.
    #[must_use]
    pub fn surprisal(&self, context: &[&str], token: &str) -> f64 {
        -self.probability(context, token).log2()
    }
}
