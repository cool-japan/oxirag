//! [`DebateEngine`] — adversarial multi-agent debate round orchestration —
//! together with the deterministic [`MockDebatePersona`] and
//! [`MockDebateJudge`] implementations of the pluggable
//! [`super::types::DebatePersona`] / [`super::types::DebateJudge`] traits.
//!
//! ## Lexical helpers
//!
//! Deliberately self-contained (not imported from `searchain` or
//! `self_consistency`): each module in this crate owns its own small
//! lexical toolkit rather than sharing private helpers across module
//! boundaries. Content tokens are lowercase alphanumeric runs of at least
//! three characters, excluding a small stopword list. Hashing uses a
//! deterministic 64-bit FNV-1a seeded only from caller-visible text
//! (question, position, round number), so every value [`MockDebatePersona`]
//! and [`MockDebateJudge`] produce is exactly reproducible across runs —
//! No randomness (`rand`/`rand_distr`) and no array/tensor machinery
//! (`ndarray`/`SciRS2-Core`) is needed anywhere in this module — every
//! input and output is plain text, and every derived value traces back to
//! deterministic hashing of that text.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::types::{
    DebateArgument, DebateConfig, DebateError, DebateJudge, DebateJudgeWeights, DebateParticipant,
    DebatePersona, DebatePositionScore, DebateResult, DebateRound, DebateVerdict,
};

// ── lexical helpers ──────────────────────────────────────────────────────────

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Compute the deterministic `FNV-1a` 64-bit hash of `bytes`.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Stopwords excluded from the content-term vocabulary.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "did", "its", "let", "put", "say", "she", "too", "use", "that",
    "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then", "than",
    "into", "some", "such", "only", "also", "been", "more", "very", "will", "would", "there",
    "their", "which", "about", "could", "these", "those", "does", "still", "even", "point",
    "prior",
];

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep
/// non-empty fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Return `true` when `token` is a content token: at least three characters
/// and not a stopword.
fn is_content_token(token: &str) -> bool {
    token.chars().count() >= 3 && !STOPWORDS.contains(&token)
}

/// The distinct, sorted content-token vocabulary of `text`. A [`BTreeSet`]
/// (rather than a hash-based set) so that iterating the result is itself
/// deterministic across process runs, not merely within a single run.
pub(crate) fn content_terms(text: &str) -> BTreeSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| is_content_token(t))
        .collect()
}

/// Jaccard similarity (`|intersection| / |union|`) between two content-term
/// sets. `0.0` when both sets are empty.
pub(crate) fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    #[allow(clippy::cast_precision_loss)]
    {
        intersection as f32 / union as f32
    }
}

/// Deterministic connective phrases [`MockDebatePersona`] rotates through,
/// selected via [`fnv1a`].
const CONNECTIVES: &[&str] = &[
    "However,",
    "Moreover,",
    "Crucially,",
    "In contrast,",
    "Notably,",
    "Still,",
    "Even so,",
    "By the same token,",
];

/// Deterministically select a connective phrase for `(question, position,
/// round)`.
fn pick_connective(question: &str, position: &str, round: usize) -> &'static str {
    let seed = format!("{question}\u{1}{position}\u{1}{round}");
    let hash = fnv1a(seed.as_bytes());
    #[allow(clippy::cast_possible_truncation)]
    let index = (hash % CONNECTIVES.len() as u64) as usize;
    CONNECTIVES[index]
}

/// Deterministically pick the most "salient" content term of `text`: the
/// term whose [`fnv1a`] hash is largest, tie-broken by lexicographic order
/// (via the already-sorted [`BTreeSet`] iteration order that
/// [`content_terms`] returns).
pub(crate) fn salient_term(text: &str) -> Option<String> {
    content_terms(text)
        .into_iter()
        .max_by_key(|term| fnv1a(term.as_bytes()))
}

// ── MockDebatePersona ────────────────────────────────────────────────────────

/// Deterministic [`DebatePersona`] for tests and examples.
///
/// No live model is involved: [`MockDebatePersona::argue`] builds argument
/// text from the question, the assigned position, and — from round `1`
/// onward — a deterministically selected "salient" content term
/// ([`fnv1a`]-based, via [`salient_term`]) taken from the immediately
/// preceding round's opposing argument, so the produced text is a genuine
/// (if templated) reaction to what the opponent just said, not a canned
/// string repeated every round. Self-reported confidence is the Jaccard
/// overlap between the produced text's content terms and the assigned
/// position's content terms.
///
/// `label` is a short human-readable tag folded into the generated text
/// (e.g. a persona's name or role); it has no effect on control flow.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MockDebatePersona {
    /// A short human-readable tag folded into generated argument text.
    pub label: String,
}

impl MockDebatePersona {
    /// Create a new mock persona with the given human-readable `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }

    /// The most recent argument in `prior_arguments` whose position differs
    /// from `position` — i.e. the opponent's last point, from this
    /// persona's point of view.
    fn last_opposing<'a>(
        position: &str,
        prior_arguments: &'a [DebateArgument],
    ) -> Option<&'a DebateArgument> {
        prior_arguments
            .iter()
            .rev()
            .find(|a| a.position != position)
    }
}

impl DebatePersona for MockDebatePersona {
    fn argue(
        &self,
        question: &str,
        position: &str,
        prior_arguments: &[DebateArgument],
    ) -> Result<DebateArgument, DebateError> {
        let round = prior_arguments
            .iter()
            .filter(|a| a.position == position)
            .count();
        let connective = pick_connective(question, position, round);
        let label = self.label.trim();

        let text = match Self::last_opposing(position, prior_arguments) {
            None => {
                if label.is_empty() {
                    format!(
                        "Opening position on \"{position}\": {connective} the case for \"{position}\" directly answers: {question}"
                    )
                } else {
                    format!(
                        "{label} opens on \"{position}\": {connective} the case for \"{position}\" directly answers: {question}"
                    )
                }
            }
            Some(opponent) => {
                let salient =
                    salient_term(&opponent.text).unwrap_or_else(|| "that prior claim".to_string());
                let speaker = if label.is_empty() {
                    "this side".to_string()
                } else {
                    label.to_string()
                };
                format!(
                    "{speaker} rebuts persona {}'s point on \"{salient}\": {connective} \"{position}\" still stands against it, on: {question}",
                    opponent.persona_index
                )
            }
        };

        let confidence = jaccard(&content_terms(&text), &content_terms(position));

        // `round` and `position` are both genuinely derivable from this
        // call's own inputs (unlike `persona_index`, which is not — see the
        // trait-level documentation), so report them accurately rather than
        // leaving [`DebateArgument::new`]'s placeholders. `DebateEngine::run`
        // overwrites both unconditionally anyway, but a direct caller of
        // `argue` (bypassing the engine) sees correct values either way.
        let mut argument = DebateArgument::new(text, confidence);
        argument.position = position.to_string();
        argument.round = round;
        Ok(argument)
    }
}

// ── MockDebateJudge ──────────────────────────────────────────────────────────

/// Deterministic [`DebateJudge`] for tests and examples.
///
/// Scores every position on three dimensions — see [`DebatePositionScore`]
/// — combined via `self.weights`:
///
/// 1. **Argument count**, normalized against the transcript's highest
///    count.
/// 2. **Average self-reported confidence**, from each argument's
///    [`DebateArgument::confidence`].
/// 3. **Rebuttal engagement**: for every argument made after round `0`, the
///    best (maximum) Jaccard overlap between that argument's text and any
///    opposing argument from the immediately preceding round (the closest
///    reading of "did this side directly rebut the opponent's last
///    point"), averaged over the arguments that had an opposing point to
///    engage with at all.
///
/// The position with the highest weighted total wins; ties are broken by
/// ascending persona index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MockDebateJudge {
    /// The scoring weights this judge combines the three dimensions with.
    pub weights: DebateJudgeWeights,
}

impl MockDebateJudge {
    /// Create a new mock judge using the given scoring weights.
    #[must_use]
    pub fn new(weights: DebateJudgeWeights) -> Self {
        Self { weights }
    }

    /// Compute the [`DebatePositionScore`] for one side (`persona_index` /
    /// `position`), given the fully flattened transcript `flat` and the
    /// transcript-wide highest argument count `max_count`.
    #[allow(clippy::cast_precision_loss)]
    fn score_side(
        &self,
        flat: &[&DebateArgument],
        persona_index: usize,
        position: &str,
        max_count: usize,
    ) -> DebatePositionScore {
        let own: Vec<&DebateArgument> = flat
            .iter()
            .copied()
            .filter(|a| a.persona_index == persona_index)
            .collect();
        let argument_count = own.len();
        let normalized_argument_count = if max_count == 0 {
            0.0
        } else {
            argument_count as f32 / max_count as f32
        };

        let average_confidence = if argument_count == 0 {
            0.0
        } else {
            own.iter().map(|a| a.confidence).sum::<f32>() / argument_count as f32
        };

        let mut engagement_sum = 0.0_f32;
        let mut engagement_count = 0_usize;
        for &arg in &own {
            if arg.round == 0 {
                continue;
            }
            let preceding_round = arg.round - 1;
            let opposing: Vec<&DebateArgument> = flat
                .iter()
                .copied()
                .filter(|opp| opp.round == preceding_round && opp.persona_index != persona_index)
                .collect();
            if opposing.is_empty() {
                continue;
            }
            let best = opposing
                .iter()
                .map(|opp| jaccard(&content_terms(&arg.text), &content_terms(&opp.text)))
                .fold(0.0_f32, f32::max);
            engagement_sum += best;
            engagement_count += 1;
        }
        let rebuttal_engagement = if engagement_count == 0 {
            0.0
        } else {
            engagement_sum / engagement_count as f32
        };

        let total_score = self.weights.argument_count_weight * normalized_argument_count
            + self.weights.confidence_weight * average_confidence
            + self.weights.rebuttal_weight * rebuttal_engagement;

        DebatePositionScore {
            persona_index,
            position: position.to_string(),
            argument_count,
            normalized_argument_count,
            average_confidence,
            rebuttal_engagement,
            total_score,
        }
    }
}

/// Build a human-readable, transcript-referencing rationale for the winning
/// (first) entry of `scores`, naming the runner-up (if any) for contrast.
fn build_rationale(question: &str, round_count: usize, scores: &[DebatePositionScore]) -> String {
    let Some(winner) = scores.first() else {
        return format!("no positions were argued for \"{question}\"");
    };
    let mut rationale = format!(
        "After {round_count} round(s) on \"{question}\", position \"{}\" (persona {}) wins with total score {:.3} (argument_count={}, normalized_argument_count={:.3}, average_confidence={:.3}, rebuttal_engagement={:.3})",
        winner.position,
        winner.persona_index,
        winner.total_score,
        winner.argument_count,
        winner.normalized_argument_count,
        winner.average_confidence,
        winner.rebuttal_engagement,
    );
    if let Some(runner_up) = scores.get(1) {
        let _ = write!(
            rationale,
            "; runner-up position \"{}\" (persona {}) scored {:.3}",
            runner_up.position, runner_up.persona_index, runner_up.total_score
        );
    }
    rationale
}

impl DebateJudge for MockDebateJudge {
    fn judge(
        &self,
        question: &str,
        transcript: &[DebateRound],
    ) -> Result<DebateVerdict, DebateError> {
        if question.trim().is_empty() {
            return Err(DebateError::EmptyQuestion);
        }
        if transcript.is_empty() {
            return Err(DebateError::EmptyTranscript);
        }

        let flat: Vec<&DebateArgument> = transcript
            .iter()
            .flat_map(|round| round.arguments.iter())
            .collect();
        if flat.is_empty() {
            return Err(DebateError::EmptyTranscript);
        }

        // Distinct (persona_index, position) sides, in order of first
        // appearance, then re-sorted by ascending persona index.
        let mut sides: Vec<(usize, String)> = Vec::new();
        for arg in &flat {
            if !sides.iter().any(|(idx, _)| *idx == arg.persona_index) {
                sides.push((arg.persona_index, arg.position.clone()));
            }
        }
        sides.sort_by_key(|(idx, _)| *idx);

        let max_count = sides
            .iter()
            .map(|(idx, _)| flat.iter().filter(|a| a.persona_index == *idx).count())
            .max()
            .unwrap_or(0);

        let mut scores: Vec<DebatePositionScore> = sides
            .into_iter()
            .map(|(persona_index, position)| {
                self.score_side(&flat, persona_index, &position, max_count)
            })
            .collect();

        scores.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.persona_index.cmp(&b.persona_index))
        });

        let rationale = build_rationale(question, transcript.len(), &scores);
        let winning_position = scores
            .first()
            .map(|s| s.position.clone())
            .ok_or(DebateError::EmptyTranscript)?;
        let winning_persona_index = scores.first().map_or(0, |s| s.persona_index);

        Ok(DebateVerdict {
            winning_position,
            winning_persona_index,
            rationale,
            scores,
        })
    }
}

// ── DebateEngine ─────────────────────────────────────────────────────────────

/// Orchestrates an adversarial multi-agent debate: `N >= 2` participants
/// (each a [`DebatePersona`] paired with a distinct position) argue for
/// [`DebateConfig::max_rounds`] rounds, then a caller-supplied
/// [`DebateJudge`] reviews the complete transcript and decides a winner.
///
/// Every round, every participant is called exactly once, and every
/// participant receives the **same** snapshot of prior arguments: every
/// argument from every strictly earlier round, across every participant —
/// never that round's own not-yet-complete arguments from other
/// participants. Rounds are therefore round-synchronized: no participant is
/// advantaged merely by where it sits in the participant list, and every
/// [`DebateRound`] pushed onto the transcript is always complete (one
/// argument per participant) before the next round's calls begin.
///
/// See the [module-level documentation](crate::multi_agent_debate) for a
/// complete runnable example.
#[derive(Debug, Clone, Default)]
pub struct DebateEngine {
    /// Configuration for this engine.
    pub config: DebateConfig,
}

impl DebateEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: DebateConfig) -> Self {
        Self { config }
    }

    /// Run a complete debate: `max_rounds` rounds of argument (or fewer, if
    /// early stopping fires), then judging.
    ///
    /// # Errors
    ///
    /// - [`DebateError::EmptyQuestion`] if `question` is blank.
    /// - [`DebateError::TooFewPersonas`] if fewer than 2 participants are
    ///   supplied.
    /// - [`DebateError::ZeroRounds`] if `self.config.max_rounds` is `0`.
    /// - [`DebateError::EmptyPosition`] if a participant's position is
    ///   blank.
    /// - [`DebateError::DuplicatePosition`] if two participants share the
    ///   same position.
    /// - Whatever a participant's [`DebatePersona::argue`], or the supplied
    ///   [`DebateJudge::judge`], returns — propagated unchanged.
    pub fn run<J>(
        &self,
        question: &str,
        participants: &[DebateParticipant<'_>],
        judge: &J,
    ) -> Result<DebateResult, DebateError>
    where
        J: DebateJudge + ?Sized,
    {
        validate_inputs(question, participants, &self.config)?;

        let mut transcript: Vec<DebateRound> = Vec::with_capacity(self.config.max_rounds);
        let mut flat_streak = vec![0_usize; participants.len()];
        let mut last_confidence: Vec<Option<f32>> = vec![None; participants.len()];
        let mut stopped_early = false;
        let mut early_stop_reason: Option<String> = None;

        for round_index in 0..self.config.max_rounds {
            // A round-synchronized snapshot: every argument from every
            // strictly earlier (already-completed) round. Every participant
            // in this round sees this exact same slice.
            let history: Vec<DebateArgument> = transcript
                .iter()
                .flat_map(|round| round.arguments.iter().cloned())
                .collect();

            let mut round = DebateRound::new(round_index);
            for (i, participant) in participants.iter().enumerate() {
                let produced =
                    participant
                        .persona
                        .argue(question, participant.position, &history)?;

                // The engine — not the persona — owns identity/bookkeeping.
                let argument = DebateArgument {
                    persona_index: i,
                    position: participant.position.to_string(),
                    round: round_index,
                    text: produced.text,
                    confidence: produced.confidence.clamp(0.0, 1.0),
                };

                if let Some(previous) = last_confidence[i] {
                    if argument.confidence <= previous + self.config.flat_confidence_epsilon {
                        flat_streak[i] += 1;
                    } else {
                        flat_streak[i] = 0;
                    }
                }
                last_confidence[i] = Some(argument.confidence);

                round.arguments.push(argument);
            }
            transcript.push(round);

            if self.config.early_stop
                && let Some(i) = flat_streak
                    .iter()
                    .position(|&streak| streak >= self.config.flat_confidence_window)
            {
                stopped_early = true;
                early_stop_reason = Some(format!(
                    "persona {i} (position \"{}\") confidence was flat or declining for {} consecutive round(s)",
                    participants[i].position, flat_streak[i]
                ));
                break;
            }
        }

        let verdict = judge.judge(question, &transcript)?;
        let rounds_run = transcript.len();

        Ok(DebateResult {
            question: question.to_string(),
            transcript,
            verdict,
            rounds_run,
            stopped_early,
            early_stop_reason,
        })
    }
}

/// Validate a debate's inputs before any rounds are run.
fn validate_inputs(
    question: &str,
    participants: &[DebateParticipant<'_>],
    config: &DebateConfig,
) -> Result<(), DebateError> {
    if question.trim().is_empty() {
        return Err(DebateError::EmptyQuestion);
    }
    if participants.len() < 2 {
        return Err(DebateError::TooFewPersonas {
            count: participants.len(),
        });
    }
    if config.max_rounds == 0 {
        return Err(DebateError::ZeroRounds);
    }
    for (index, participant) in participants.iter().enumerate() {
        if participant.position.trim().is_empty() {
            return Err(DebateError::EmptyPosition { index });
        }
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for participant in participants {
        if !seen.insert(participant.position) {
            return Err(DebateError::DuplicatePosition {
                position: participant.position.to_string(),
            });
        }
    }
    Ok(())
}
