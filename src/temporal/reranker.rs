//! Temporal re-ranker: applies recency decay to retrieval scores.

use crate::types::SearchResult;

use super::types::{TemporalConfig, TemporalError, TemporalScore};

// ── timestamp parsing ─────────────────────────────────────────────────────────

/// Parse an RFC 3339 / ISO 8601 timestamp from a string.
///
/// Supports the formats:
/// - `YYYY-MM-DDTHH:MM:SSZ`
/// - `YYYY-MM-DDTHH:MM:SS+00:00`
/// - `YYYY-MM-DD` (date only, treated as midnight UTC)
///
/// Returns seconds since Unix epoch on success.
fn parse_timestamp_secs(s: &str) -> Option<i64> {
    let s = s.trim();
    // Try date-only first
    if s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-') {
        let year: i64 = s[0..4].parse().ok()?;
        let month: i64 = s[5..7].parse().ok()?;
        let day: i64 = s[8..10].parse().ok()?;
        return Some(approx_unix_secs(year, month, day, 0, 0, 0));
    }
    // Try full datetime: at least YYYY-MM-DDTHH:MM:SS
    if s.len() >= 19 && s.chars().nth(4) == Some('-') && s.chars().nth(10) == Some('T') {
        let year: i64 = s[0..4].parse().ok()?;
        let month: i64 = s[5..7].parse().ok()?;
        let day: i64 = s[8..10].parse().ok()?;
        let hour: i64 = s[11..13].parse().ok()?;
        let min: i64 = s[14..16].parse().ok()?;
        let sec: i64 = s[17..19].parse().ok()?;
        return Some(approx_unix_secs(year, month, day, hour, min, sec));
    }
    // Try Unix epoch integer
    s.parse::<i64>().ok()
}

/// Approximate days per month (ignoring leap years for simplicity).
const DAYS_IN_MONTH: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Rough Unix seconds — sufficient for age-decay purposes (± a few days per century).
fn approx_unix_secs(year: i64, month: i64, day: i64, hour: i64, min: i64, sec: i64) -> i64 {
    let years_since_1970 = year - 1970;
    let leap_years = years_since_1970 / 4; // rough
    let mut days = years_since_1970 * 365 + leap_years;
    for m in 1..month {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let idx = (m as usize) - 1;
        days += DAYS_IN_MONTH[idx];
    }
    days += day - 1;
    days * 86_400 + hour * 3600 + min * 60 + sec
}

/// Extract the relevant timestamp (seconds since epoch) from a document's metadata.
fn document_timestamp(doc: &crate::types::Document, use_updated: bool) -> Option<i64> {
    let keys: &[&str] = if use_updated {
        &["updated_at", "created_at"]
    } else {
        &["created_at", "updated_at"]
    };
    for key in keys {
        if let Some(ts) = doc.metadata.get(*key)
            && let Some(secs) = parse_timestamp_secs(ts)
        {
            return Some(secs);
        }
    }
    None
}

/// Age in fractional days between `doc_secs` and `now_secs`.
#[allow(clippy::cast_precision_loss)]
fn age_days(doc_secs: i64, now_secs: i64) -> f64 {
    let delta = (now_secs - doc_secs).max(0);
    delta as f64 / 86_400.0
}

// ── TemporalReranker ──────────────────────────────────────────────────────────

/// Re-ranks search results by blending retrieval scores with recency decay.
///
/// Score formula:
/// ```text
/// decayed = original * decay(age_days)
/// final   = (1 - weight) * original + weight * decayed
/// ```
#[derive(Debug, Clone, Default)]
pub struct TemporalReranker;

impl TemporalReranker {
    /// Create a new [`TemporalReranker`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Re-rank `results` given the current time as Unix seconds.
    ///
    /// # Errors
    ///
    /// Returns [`TemporalError::EmptyResults`] when `results` is empty.
    pub fn rerank(
        &self,
        results: &[SearchResult],
        now_unix_secs: i64,
        config: &TemporalConfig,
    ) -> Result<(Vec<SearchResult>, Vec<TemporalScore>), TemporalError> {
        if results.is_empty() {
            return Err(TemporalError::EmptyResults);
        }

        let mut reranked: Vec<(SearchResult, TemporalScore)> = results
            .iter()
            .map(|r| {
                let age = document_timestamp(&r.document, config.use_updated)
                    .map_or(0.0, |ts| age_days(ts, now_unix_secs)); // unknown timestamp → treat as brand new

                #[allow(clippy::cast_possible_truncation)]
                let decay = config.decay.apply(age) as f32;
                let w = config.weight;
                let decayed = r.score * decay;
                let final_score = (1.0 - w) * r.score + w * decayed;

                let ts = TemporalScore {
                    original: r.score,
                    decayed: final_score,
                    age_days: age,
                };

                let mut updated = r.clone();
                updated.score = final_score;

                (updated, ts)
            })
            .collect();

        // Sort by final score descending
        reranked.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (docs, scores): (Vec<SearchResult>, Vec<TemporalScore>) = reranked.into_iter().unzip();

        Ok((docs, scores))
    }
}
