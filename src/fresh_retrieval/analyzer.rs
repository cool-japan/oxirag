//! Freshness analyzer: query time-sensitivity, freshness scoring, staleness flagging, re-ranking.

use crate::types::SearchResult;

use super::types::{FreshConfig, FreshError, FreshnessAssessment, TimeSensitivity};

// ── lexicons ──────────────────────────────────────────────────────────────────

/// Strong temporal markers implying *fast-changing* information when present.
///
/// "now", "today", "currently", "right now" pin the answer to the present
/// instant, which only matters when the underlying fact moves quickly.
const FAST_MARKERS: &[&str] = &["now", "today", "currently", "tonight"];

/// Temporal markers implying *at least slow-changing* recency-sensitivity.
///
/// "latest", "current", "recent", "newest", "this year" all ask for an
/// up-to-date answer without necessarily implying minute-by-minute volatility.
const RECENCY_MARKERS: &[&str] = &[
    "latest", "current", "recent", "recently", "newest", "modern", "upcoming", "ongoing",
    "nowadays", "today's",
];

/// Comparative superlatives that imply the user wants the most up-to-date item.
const COMPARATIVES: &[&str] = &["newest", "latest", "freshest"];

/// Fast-changing domain cues — facts in these domains move within days or less.
const FAST_DOMAINS: &[&str] = &[
    "price",
    "prices",
    "stock",
    "stocks",
    "weather",
    "score",
    "scores",
    "news",
    "rate",
    "rates",
    "forecast",
    "exchange",
    "traffic",
    "trending",
    "headline",
    "headlines",
    "standings",
];

// ── tokenizer ─────────────────────────────────────────────────────────────────

/// Split text into lowercase alphanumeric tokens (non-alphanumeric is a separator).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `token` looks like a plausible explicit calendar year (1900–2099).
fn is_year_token(token: &str) -> bool {
    if token.len() != 4 || !token.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    token
        .parse::<u32>()
        .is_ok_and(|y| (1900..=2099).contains(&y))
}

// ── FreshnessAnalyzer ─────────────────────────────────────────────────────────

/// Detects query time-sensitivity, scores document freshness, and flags staleness.
///
/// The analyzer is purely lexical and deterministic. [`Self::classify`] scans the
/// query for temporal markers, comparatives, fast-changing-domain cues, and
/// explicit calendar years, deriving a [`TimeSensitivity`] and a
/// `freshness_demand` in `[0,1]`. That demand then modulates how aggressively
/// [`Self::rerank`] lets document freshness override raw relevance.
///
/// Freshness itself decays exponentially with age:
/// ```text
/// freshness(age_days) = 0.5 ^ (age_days / half_life_days)
/// ```
#[derive(Debug, Clone, Default)]
pub struct FreshnessAnalyzer {
    /// Configuration controlling decay, staleness, and re-ranking strength.
    config: FreshConfig,
}

impl FreshnessAnalyzer {
    /// Create a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: FreshConfig) -> Self {
        Self { config }
    }

    /// Borrow the analyzer's configuration.
    #[must_use]
    pub fn config(&self) -> &FreshConfig {
        &self.config
    }

    /// Classify how time-sensitive `query` is.
    ///
    /// Detects temporal markers (`"current"`, `"latest"`, `"now"`, `"today"`,
    /// `"recent"`, `"this year"`), explicit calendar years, comparatives
    /// (`"newest"`), and fast-changing-domain cues (`"price"`, `"stock"`,
    /// `"weather"`, `"score"`, `"news"`). The strongest cue wins:
    ///
    /// - any fast marker or fast-changing-domain cue ⇒ [`TimeSensitivity::FastChanging`]
    /// - otherwise any recency marker, comparative, or explicit year ⇒
    ///   [`TimeSensitivity::SlowChanging`]
    /// - otherwise [`TimeSensitivity::Static`]
    ///
    /// `freshness_demand` starts from [`TimeSensitivity::base_demand`] and is
    /// nudged upward by each additional fired signal, saturating at `1.0`.
    ///
    /// # Errors
    ///
    /// Returns [`FreshError::EmptyQuery`] when `query` is empty or whitespace only.
    pub fn classify(&self, query: &str) -> Result<FreshnessAssessment, FreshError> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Err(FreshError::EmptyQuery);
        }

        let mut signals: Vec<String> = Vec::new();
        let mut fast = false;
        let mut recency = false;

        for token in &tokens {
            if FAST_MARKERS.contains(&token.as_str()) {
                fast = true;
                signals.push(format!("marker:{token}"));
            } else if RECENCY_MARKERS.contains(&token.as_str()) {
                recency = true;
                signals.push(format!("marker:{token}"));
            }

            if COMPARATIVES.contains(&token.as_str()) {
                recency = true;
                signals.push(format!("comparative:{token}"));
            }

            if FAST_DOMAINS.contains(&token.as_str()) {
                fast = true;
                signals.push(format!("domain:{token}"));
            }

            if is_year_token(token) {
                recency = true;
                signals.push(format!("year:{token}"));
            }
        }

        // "this year" / "this month" / "this week" two-word cue.
        for window in tokens.windows(2) {
            if window[0] == "this"
                && matches!(window[1].as_str(), "year" | "month" | "week" | "quarter")
            {
                recency = true;
                signals.push(format!("phrase:this_{}", window[1]));
            }
        }

        let sensitivity = if fast {
            TimeSensitivity::FastChanging
        } else if recency {
            TimeSensitivity::SlowChanging
        } else {
            TimeSensitivity::Static
        };

        let freshness_demand = Self::compute_demand(sensitivity, signals.len());

        Ok(FreshnessAssessment {
            sensitivity,
            freshness_demand,
            signals,
        })
    }

    /// Derive a `[0,1]` freshness demand from the category and signal count.
    ///
    /// Each fired signal adds a small reinforcement on top of the category's
    /// [`TimeSensitivity::base_demand`], saturating at `1.0`.
    fn compute_demand(sensitivity: TimeSensitivity, signal_count: usize) -> f32 {
        let base = sensitivity.base_demand();
        if sensitivity == TimeSensitivity::Static {
            return 0.0;
        }
        // Bounded to at most 4 → exactly representable in f32.
        #[allow(clippy::cast_precision_loss)]
        let reinforcement = 0.05 * signal_count.min(4) as f32;
        (base + reinforcement).clamp(0.0, 1.0)
    }

    /// Freshness score in `(0,1]` for a document of age `age_days`.
    ///
    /// Computed as `0.5^(age_days / half_life_days)`: age `0` scores `1.0`, age
    /// equal to the configured half-life scores `0.5`. Negative ages (documents
    /// dated in the future) are clamped to `0`, yielding `1.0`. A non-positive
    /// half-life disables decay and always returns `1.0`.
    #[must_use]
    pub fn freshness_score(&self, age_days: f64) -> f32 {
        let half_life = self.config.half_life_days;
        if half_life <= 0.0 {
            return 1.0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let age = age_days.max(0.0) as f32;
        0.5_f32.powf(age / half_life)
    }

    /// Whether an answer of age `answer_age_days` to `query` is likely stale.
    ///
    /// Returns `true` only when the query is [`TimeSensitivity::FastChanging`]
    /// **and** `answer_age_days` strictly exceeds
    /// [`FreshConfig::staleness_threshold_days`]. Static and slow-changing
    /// queries are never flagged stale, regardless of age.
    ///
    /// # Errors
    ///
    /// Returns [`FreshError::EmptyQuery`] when `query` is empty or whitespace only.
    pub fn is_stale(&self, query: &str, answer_age_days: f64) -> Result<bool, FreshError> {
        let assessment = self.classify(query)?;
        Ok(assessment.sensitivity == TimeSensitivity::FastChanging
            && answer_age_days > self.config.staleness_threshold_days)
    }

    /// Re-rank `results` by blending relevance with document freshness.
    ///
    /// The blend weight is `freshness_demand * freshness_weight`, so a static
    /// query (demand `0`) is an exact identity transform that preserves the
    /// input order, while a fast-changing query promotes fresher documents:
    ///
    /// ```text
    /// w           = demand * freshness_weight
    /// fresh_i     = freshness_score(ages_days[i])
    /// final_i     = (1 - w) * score_i + w * fresh_i
    /// ```
    ///
    /// `ages_days[i]` is the age of `results[i].document`. Results are returned
    /// sorted by `final` score descending with `rank` re-numbered from `0`.
    ///
    /// # Errors
    ///
    /// - [`FreshError::EmptyQuery`] when `query` is empty or whitespace only.
    /// - [`FreshError::LengthMismatch`] when `ages_days.len() != results.len()`.
    pub fn rerank(
        &self,
        query: &str,
        results: &[SearchResult],
        ages_days: &[f64],
    ) -> Result<Vec<SearchResult>, FreshError> {
        let assessment = self.classify(query)?;

        if ages_days.len() != results.len() {
            return Err(FreshError::LengthMismatch {
                ages: ages_days.len(),
                results: results.len(),
            });
        }

        let weight = assessment.freshness_demand * self.config.freshness_weight;

        // Identity fast path for fully static queries (no freshness influence).
        if weight == 0.0 {
            return Ok(results.to_vec());
        }

        let mut scored: Vec<SearchResult> = results
            .iter()
            .zip(ages_days.iter())
            .map(|(result, &age)| {
                let fresh = self.freshness_score(age);
                let final_score = (1.0 - weight).mul_add(result.score, weight * fresh);
                let mut updated = result.clone();
                updated.score = final_score;
                updated
            })
            .collect();

        // Stable descending sort by blended score; ties keep original order.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (rank, result) in scored.iter_mut().enumerate() {
            result.rank = rank;
        }

        Ok(scored)
    }
}
