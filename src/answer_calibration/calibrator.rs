//! The [`AnswerCalibrator`]: answer-level confidence estimation and scaling.

use crate::answer_calibration::types::{
    AnswerCalibrationError, AnswerCalibratorConfig, ConfidenceSignals,
};

/// Numerically stable logistic sigmoid `1 / (1 + e^-x)`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Hedging words that signal *low* confidence; each maps the verbalized signal
/// toward `~0.3`.
const HEDGE_WORDS: &[&str] = &[
    "maybe",
    "possibly",
    "perhaps",
    "might",
    "guess",
    "unsure",
    "uncertain",
    "probably",
    "likely",
    "presumably",
    "seems",
    "apparently",
];

/// Confident words that signal *high* confidence; each maps the verbalized
/// signal toward `~0.9`.
const CONFIDENT_WORDS: &[&str] = &[
    "definitely",
    "certainly",
    "surely",
    "undoubtedly",
    "absolutely",
    "clearly",
    "obviously",
    "guaranteed",
    "without doubt",
];

/// Multi-word hedging phrase ("I think" and friends), checked against the lower-
/// cased text directly.
const HEDGE_PHRASES: &[&str] = &["i think", "i believe", "i suppose", "not sure", "could be"];

/// The verbalized-confidence value assigned to a hedge.
const HEDGE_CONFIDENCE: f32 = 0.3;
/// The verbalized-confidence value assigned to a confident cue.
const CONFIDENT_CONFIDENCE: f32 = 0.9;

/// Answer-level confidence calibrator.
///
/// Blends the [`ConfidenceSignals`] of a generated answer into a single
/// confidence, can extract a verbalized confidence from free text, and applies
/// an optional Platt-style sigmoid scaling (`sigmoid(a·raw + b)`) whose
/// parameters are fitted from labelled data.
///
/// Until [`fit_platt`](Self::fit_platt) is called the scaling is the identity-ish
/// default `a = 1`, `b = 0`, i.e. `calibrate(raw) = sigmoid(raw)`.
#[derive(Debug, Clone)]
pub struct AnswerCalibrator {
    /// Blending weights and binning configuration.
    pub config: AnswerCalibratorConfig,
    /// Platt slope parameter `a`. Defaults to `1.0`.
    pub platt_a: f32,
    /// Platt intercept parameter `b`. Defaults to `0.0`.
    pub platt_b: f32,
}

impl AnswerCalibrator {
    /// Create a calibrator with the given configuration and the identity-ish
    /// Platt parameters (`a = 1`, `b = 0`).
    #[must_use]
    pub fn new(config: AnswerCalibratorConfig) -> Self {
        Self {
            config,
            platt_a: 1.0,
            platt_b: 0.0,
        }
    }

    /// Extract a verbalized confidence from the answer text, in `[0.0, 1.0]`.
    ///
    /// Resolution order:
    /// 1. An explicit percentage such as "`90%`" → `0.90` (clamped to `[0,1]`).
    /// 2. A confident cue (e.g. "definitely", "certainly") → `~0.9`.
    /// 3. A hedge cue (e.g. "maybe", "possibly", "I think") → `~0.3`.
    ///
    /// Returns `None` when the text carries no recognisable signal. When both a
    /// confident and a hedge cue appear (but no percentage), the two are averaged
    /// so contradictory phrasing lands in the middle.
    #[must_use]
    pub fn extract_verbalized(&self, text: &str) -> Option<f32> {
        if let Some(pct) = parse_percentage(text) {
            return Some(pct.clamp(0.0, 1.0));
        }

        let lower = text.to_lowercase();
        let has_confident = CONFIDENT_WORDS.iter().any(|w| contains_word(&lower, w))
            || CONFIDENT_WORDS
                .iter()
                .filter(|w| w.contains(' '))
                .any(|w| lower.contains(*w));
        let has_hedge = HEDGE_WORDS.iter().any(|w| contains_word(&lower, w))
            || HEDGE_PHRASES.iter().any(|p| lower.contains(*p));

        match (has_confident, has_hedge) {
            (true, true) => Some(f32::midpoint(CONFIDENT_CONFIDENCE, HEDGE_CONFIDENCE)),
            (true, false) => Some(CONFIDENT_CONFIDENCE),
            (false, true) => Some(HEDGE_CONFIDENCE),
            (false, false) => None,
        }
    }

    /// Blend the supplied [`ConfidenceSignals`] into a single confidence in
    /// `[0.0, 1.0]`.
    ///
    /// Weights from the [`config`](Self::config) are normalised to sum to one. If
    /// [`verbalized`](ConfidenceSignals::verbalized) is `None` its weight is
    /// dropped and the agreement/support weights are renormalised over the
    /// remaining mass, so a missing verbalized signal neither inflates nor
    /// deflates the result. If every active weight is zero the blend falls back
    /// to the unweighted mean of the available signals.
    #[must_use]
    pub fn estimate_confidence(&self, signals: &ConfidenceSignals) -> f32 {
        let agreement = signals.agreement.clamp(0.0, 1.0);
        let support = signals.support.clamp(0.0, 1.0);

        let w_agree = self.config.agreement_weight.max(0.0);
        let w_support = self.config.support_weight.max(0.0);

        if let Some(verbalized) = signals.verbalized {
            let verbalized = verbalized.clamp(0.0, 1.0);
            let w_verb = self.config.verbalized_weight.max(0.0);
            let total = w_verb + w_agree + w_support;
            if total <= 0.0 {
                return ((verbalized + agreement + support) / 3.0).clamp(0.0, 1.0);
            }
            ((w_verb * verbalized + w_agree * agreement + w_support * support) / total)
                .clamp(0.0, 1.0)
        } else {
            let total = w_agree + w_support;
            if total <= 0.0 {
                return f32::midpoint(agreement, support).clamp(0.0, 1.0);
            }
            ((w_agree * agreement + w_support * support) / total).clamp(0.0, 1.0)
        }
    }

    /// Map a raw confidence to a calibrated one via the fitted Platt sigmoid
    /// `sigmoid(a·raw + b)`. The input is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn calibrate(&self, raw: f32) -> f32 {
        let r = raw.clamp(0.0, 1.0);
        sigmoid(self.platt_a * r + self.platt_b)
    }

    /// Fit the Platt scaling parameters `(a, b)` to labelled data by minimising
    /// the logistic loss with batch gradient descent.
    ///
    /// The fit targets `sigmoid(a·raw + b) ≈ correct`. Parameters start from the
    /// current `(a, b)` and are updated for a fixed number of deterministic
    /// gradient steps (learning rate `0.5`, `300` iterations), so the result is
    /// reproducible and depends only on the input. The fit is monotonic in the
    /// usual case where higher raw confidence co-occurs with higher accuracy:
    /// `a` converges to a positive slope, so a larger `raw` yields a larger
    /// calibrated value.
    ///
    /// # Errors
    ///
    /// Returns [`AnswerCalibrationError::EmptyData`] if `conf_correct` is empty.
    #[allow(clippy::cast_precision_loss)]
    pub fn fit_platt(
        &mut self,
        conf_correct: &[(f32, bool)],
    ) -> Result<(), AnswerCalibrationError> {
        if conf_correct.is_empty() {
            return Err(AnswerCalibrationError::EmptyData);
        }

        let n = conf_correct.len() as f32;
        let learning_rate = 0.5_f32;
        let iterations = 300;

        let mut a = self.platt_a;
        let mut b = self.platt_b;

        for _ in 0..iterations {
            let mut grad_a = 0.0_f32;
            let mut grad_b = 0.0_f32;
            for &(conf, correct) in conf_correct {
                let raw = conf.clamp(0.0, 1.0);
                let pred = sigmoid(a * raw + b);
                let target = if correct { 1.0 } else { 0.0 };
                let error = pred - target;
                grad_a += error * raw;
                grad_b += error;
            }
            a -= learning_rate * grad_a / n;
            b -= learning_rate * grad_b / n;
        }

        self.platt_a = a;
        self.platt_b = b;
        Ok(())
    }
}

impl Default for AnswerCalibrator {
    fn default() -> Self {
        Self::new(AnswerCalibratorConfig::default())
    }
}

/// Whether `needle` appears in `haystack` as a whole word (bounded by non-
/// alphanumeric characters), so "likely" does not match inside "unlikely".
fn contains_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let nlen = needle.len();
    if nlen == 0 {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let abs = start + pos;
        let before_ok = abs == 0 || !is_word_byte(bytes[abs - 1]);
        let after = abs + nlen;
        let after_ok = after >= bytes.len() || !is_word_byte(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

/// Whether `b` is an ASCII alphanumeric byte (a word-continuation character).
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

/// Parse the first percentage token (e.g. "`90%`", "`87.5 %`") in `text` and
/// return it as a fraction in `[0.0, 1.0]`-ish (caller clamps). Returns `None`
/// when no `<number>%` pattern is present.
fn parse_percentage(text: &str) -> Option<f32> {
    let bytes = text.as_bytes();
    let percent = text.find('%')?;

    // Walk left from the '%' over optional whitespace then the numeric token.
    let mut end = percent;
    while end > 0 && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t') {
        end -= 1;
    }
    let num_end = end;
    let mut begin = num_end;
    while begin > 0 {
        let c = bytes[begin - 1];
        if c.is_ascii_digit() || c == b'.' {
            begin -= 1;
        } else {
            break;
        }
    }
    if begin == num_end {
        return None;
    }
    let token = &text[begin..num_end];
    let value: f32 = token.parse().ok()?;
    Some(value / 100.0)
}
