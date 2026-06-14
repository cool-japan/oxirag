//! The RGB benchmark evaluator.
use crate::rgb_eval::types::{RgbAbility, RgbConfig, RgbError, RgbScores, RgbTestCase};
use std::collections::HashSet;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric chars, keep tokens of length ≥ 2,
/// lowercased.
fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Fraction of `gold` tokens that also appear in `answer` (`0.0` if gold empty).
fn gold_overlap(answer: &str, gold: &str) -> f32 {
    let gold_tokens = tokenize(gold);
    if gold_tokens.is_empty() {
        return 0.0;
    }
    let answer_tokens = tokenize(answer);
    let covered = gold_tokens
        .iter()
        .filter(|t| answer_tokens.contains(*t))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = covered as f32 / gold_tokens.len() as f32;
    ratio
}

// ── RgbEvaluator ──────────────────────────────────────────────────────────────

/// Evaluator that scores a RAG system's answers against labeled RGB test cases.
///
/// See the [module docs](crate::rgb_eval) for the four abilities and the scoring
/// rules. The evaluator is stateless beyond its [`RgbConfig`] and fully
/// deterministic.
#[derive(Debug, Clone, Default)]
pub struct RgbEvaluator {
    /// Configuration controlling thresholds and rejection phrases.
    pub config: RgbConfig,
}

impl RgbEvaluator {
    /// Create a new evaluator with the given config.
    #[must_use]
    pub fn new(config: RgbConfig) -> Self {
        Self { config }
    }

    /// Return `true` when `answer` lexically matches `gold`.
    ///
    /// Correctness is the fraction of `gold` tokens present in `answer`, compared
    /// against [`RgbConfig::match_threshold`].
    #[must_use]
    pub fn is_correct_answer(&self, answer: &str, gold: &str) -> bool {
        gold_overlap(answer, gold) >= self.config.match_threshold
    }

    /// Return `true` when `answer` contains any configured rejection phrase.
    ///
    /// Matching is case-insensitive substring containment over
    /// [`RgbConfig::rejection_phrases`].
    #[must_use]
    pub fn is_rejection(&self, answer: &str) -> bool {
        let lower = answer.to_lowercase();
        self.config
            .rejection_phrases
            .iter()
            .any(|p| !p.is_empty() && lower.contains(&p.to_lowercase()))
    }

    /// Return `true` when `system_answer` is correct for `case`'s ability.
    ///
    /// The rule depends on [`RgbTestCase::ability`]:
    /// - [`RgbAbility::NegativeRejection`] — correct iff the answer is a rejection.
    /// - [`RgbAbility::InformationIntegration`] — correct iff every
    ///   [`RgbTestCase::sub_answers`] entry is covered (and the answer is not a
    ///   rejection); falls back to the gold answer when no sub-answers are given.
    /// - otherwise — correct iff the answer matches the gold answer and is not a
    ///   rejection.
    #[must_use]
    pub fn evaluate_case(&self, case: &RgbTestCase, system_answer: &str) -> bool {
        match case.ability {
            RgbAbility::NegativeRejection => self.is_rejection(system_answer),
            RgbAbility::InformationIntegration => {
                if self.is_rejection(system_answer) {
                    return false;
                }
                if case.sub_answers.is_empty() {
                    return self.is_correct_answer(system_answer, &case.gold_answer);
                }
                case.sub_answers
                    .iter()
                    .all(|sub| self.is_correct_answer(system_answer, sub))
            }
            RgbAbility::NoiseRobustness | RgbAbility::CounterfactualRobustness => {
                !self.is_rejection(system_answer)
                    && self.is_correct_answer(system_answer, &case.gold_answer)
            }
        }
    }

    /// Evaluate all `cases` against their `answers`, returning per-ability
    /// accuracies and an overall mean.
    ///
    /// The overall score is the mean over only the abilities that appear in
    /// `cases`; abilities with no cases contribute `0.0` to their own field and
    /// are excluded from the overall mean.
    ///
    /// # Errors
    ///
    /// Returns [`RgbError::EmptyCases`] when `cases` is empty, or
    /// [`RgbError::LengthMismatch`] when `answers.len() != cases.len()`.
    pub fn evaluate(
        &self,
        cases: &[RgbTestCase],
        answers: &[String],
    ) -> Result<RgbScores, RgbError> {
        if cases.is_empty() {
            return Err(RgbError::EmptyCases);
        }
        if answers.len() != cases.len() {
            return Err(RgbError::LengthMismatch {
                answers: answers.len(),
                cases: cases.len(),
            });
        }

        // (correct, total) per ability.
        let mut noise = (0_usize, 0_usize);
        let mut negative = (0_usize, 0_usize);
        let mut integration = (0_usize, 0_usize);
        let mut counter = (0_usize, 0_usize);

        for (case, answer) in cases.iter().zip(answers.iter()) {
            let correct = self.evaluate_case(case, answer);
            let bucket = match case.ability {
                RgbAbility::NoiseRobustness => &mut noise,
                RgbAbility::NegativeRejection => &mut negative,
                RgbAbility::InformationIntegration => &mut integration,
                RgbAbility::CounterfactualRobustness => &mut counter,
            };
            bucket.1 += 1;
            if correct {
                bucket.0 += 1;
            }
        }

        let noise_robustness = accuracy(noise);
        let negative_rejection = accuracy(negative);
        let information_integration = accuracy(integration);
        let counterfactual_robustness = accuracy(counter);

        let present: Vec<f32> = [noise, negative, integration, counter]
            .into_iter()
            .zip([
                noise_robustness,
                negative_rejection,
                information_integration,
                counterfactual_robustness,
            ])
            .filter_map(|((_, total), acc)| (total > 0).then_some(acc))
            .collect();
        let overall = if present.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let mean = present.iter().sum::<f32>() / present.len() as f32;
            mean
        };

        Ok(RgbScores {
            noise_robustness,
            negative_rejection,
            information_integration,
            counterfactual_robustness,
            overall,
        })
    }
}

/// Accuracy `correct / total` for a `(correct, total)` pair (`0.0` if empty).
fn accuracy((correct, total): (usize, usize)) -> f32 {
    if total == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let acc = correct as f32 / total as f32;
        acc
    }
}
