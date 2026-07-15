//! The generator abstraction `STaR` bootstraps — and a deterministic,
//! table-driven implementation of it whose policy *genuinely improves* as the
//! accumulated rationale set grows.
//!
//! # Why this module defines its own model trait
//!
//! `STaR` is an operation on a *learner*: something that generates rationales,
//! that can be handed an answer and asked to rationalize toward it, and — the
//! part that makes it "self-taught" — that **updates its policy** from an
//! accumulated set of correct rationales. No text-in/text-out trait elsewhere in
//! this crate exposes that third capability, so this module states its own
//! minimal requirement, [`ReasoningModel`]: *generate, rationalize, and learn.*
//! Any real backend (a fine-tunable LM, an in-context few-shot prompt that grows
//! its exemplar pool) can implement it; this module never needs to know which.
//!
//! # The deterministic fixture, and how its "learning" is real
//!
//! [`StaticReasoningModel`] is a **test fixture, not a language model** — but
//! unlike a lookup table, it does not merely memorize. It is a *rule inducer*
//! over a family of modular-arithmetic tasks `f(x) = (x + offset(class)) mod m`,
//! and the offset for each class is **unknown to it** until it induces it from
//! solved examples. That is what lets every claim `STaR` makes about
//! bootstrapping be checked against hand-computed numbers, with no weights, no
//! randomness, and no network:
//!
//! - It starts able to solve only a handful of **seed** problems (memorized
//!   worked examples). From the accumulated rationales it *recovers* each class's
//!   offset — `offset = (answer − x) mod m`, required to be consistent across all
//!   examples of that class — and can then solve unseen inputs. This is genuine
//!   generalization: two solved instances of a class pin its rule.
//! - Its confident region grows outward one step per round: it applies a learned
//!   rule to an input `x` only when `x` lies within
//!   [`generalization_radius`](StaticReasoningModel::generalization_radius) of an
//!   input it has already solved. This models a fine-tuned learner whose
//!   reliable extrapolation expands with its training coverage, and it is what
//!   produces a multi-round, strictly-increasing accuracy curve rather than a
//!   one-round jump.
//! - A **class it never sees a correct example of stays unsolved forever** in the
//!   forward pass — the hook the rationalization ablation hangs on.

use std::collections::{BTreeSet, HashMap};

use super::types::{StarGeneration, StarProblem, StarRationale};

// ── The trait ──────────────────────────────────────────────────────────────────

/// A generator that can reason forward, rationalize backward from a hint, and
/// update its policy from accumulated correct rationales.
///
/// The three methods are exactly the three things the `STaR` loop asks of a
/// learner, and each clause is load-bearing:
///
/// - [`generate`](ReasoningModel::generate) is the forward pass: solve the
///   problem from scratch. Its answer is checked against the gold answer; only
///   correct ones are kept.
/// - [`rationalize`](ReasoningModel::rationalize) is the backward pass: given the
///   gold answer as a `hint`, produce a rationale that reaches it. This is what
///   rescues problems the forward pass cannot solve — but a rationalization that
///   merely restates the hint is a cheat, and the engine rejects it, so a
///   faithful implementation actually derives the answer.
/// - [`update_from`](ReasoningModel::update_from) is the fine-tuning step:
///   improve the policy from the **full** accumulated set (not just the latest
///   round). Passing the whole set each round keeps the update idempotent and the
///   run reproducible.
pub trait ReasoningModel {
    /// Forward generation: produce a rationale and answer for `problem`.
    fn generate(&self, problem: &StarProblem) -> StarGeneration;

    /// Backward rationalization: produce a rationale that reaches `hint` (the
    /// gold answer) for a problem the forward pass failed.
    fn rationalize(&self, problem: &StarProblem, hint: &str) -> StarGeneration;

    /// Fine-tune: update the policy from the full accumulated rationale set.
    fn update_from(&mut self, accumulated: &[StarRationale]);
}

// ── Rationalization style ───────────────────────────────────────────────────────

/// How [`StaticReasoningModel`] writes its backward rationalizations.
///
/// A fixture needs to be able to produce *both* honest derivations and the cheats
/// the engine must reject, so the rationalization ablation and the cheat-rejection
/// test can each drive the exact behaviour they check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RationalizationStyle {
    /// Recover the offset from the hint and show the arithmetic — a genuine,
    /// problem-grounded derivation that the cheat detector accepts.
    #[default]
    Genuine,
    /// Echo the hint back ("the answer is `X` because the answer is `X`") with no
    /// derivation — a cheat the engine must reject.
    Cheat,
}

// ── The fixture ─────────────────────────────────────────────────────────────────

/// A fully-deterministic, rule-inducing generator over modular-arithmetic tasks.
///
/// See the [module documentation](self) for how it turns accumulated rationales
/// into a genuinely improving policy. It is public because it is as useful for
/// validating a caller's own `STaR` wiring as it is to this crate's tests.
///
/// Problems are expected to have statements of the form `"[<class>] x=<n>"`, e.g.
/// `"[A] x=7"`. The class label is arbitrary text; the fixture learns an
/// independent offset per class.
#[derive(Debug, Clone)]
pub struct StaticReasoningModel {
    /// The modulus `m` of the task's output space. A learner must know the range
    /// of its answers; it does **not** know the offsets.
    modulus: u64,
    /// How far beyond an already-solved input the model will apply a learned
    /// rule, in input-space. `1` extends the solved frontier by one each round.
    generalization_radius: u64,
    /// Memorized worked examples: `(class, x) -> answer`. These are the seed the
    /// model can solve before it has induced any rule.
    seeds: HashMap<(String, u64), u64>,
    /// How backward rationalizations are written.
    rationalization_style: RationalizationStyle,
    /// Induced per-class offset, `None` until enough consistent examples are seen.
    /// Rebuilt from scratch by every [`update_from`](ReasoningModel::update_from).
    learned_offset: HashMap<String, u64>,
    /// Inputs the model has a correct accumulated rationale for, per class. The
    /// frontier the learned rule may extend from. Rebuilt every `update_from`.
    solved_inputs: HashMap<String, BTreeSet<u64>>,
}

impl StaticReasoningModel {
    /// Create a fixture over modulus `m` with the given generalization radius.
    ///
    /// It starts knowing nothing: no seeds, no learned offsets. Add seeds with
    /// [`with_seed`](Self::with_seed).
    #[must_use]
    pub fn new(modulus: u64, generalization_radius: u64) -> Self {
        Self {
            modulus: modulus.max(1),
            generalization_radius,
            seeds: HashMap::new(),
            rationalization_style: RationalizationStyle::Genuine,
            learned_offset: HashMap::new(),
            solved_inputs: HashMap::new(),
        }
    }

    /// Add a memorized worked example: the model can solve `(class, x)` directly,
    /// answering `answer`, before it has induced any rule.
    #[must_use]
    pub fn with_seed(
        mut self,
        class: impl Into<String>,
        x: u64,
        answer: impl Into<String>,
    ) -> Self {
        let answer_value = answer.into();
        if let Ok(value) = answer_value.parse::<u64>() {
            self.seeds.insert((class.into(), x), value);
        }
        self
    }

    /// Set how backward rationalizations are written.
    #[must_use]
    pub fn with_rationalization_style(mut self, style: RationalizationStyle) -> Self {
        self.rationalization_style = style;
        self
    }

    /// The generalization radius: how far a learned rule reaches beyond a solved
    /// input each round.
    #[must_use]
    pub fn generalization_radius(&self) -> u64 {
        self.generalization_radius
    }

    /// The induced offset for `class`, if the model has recovered it.
    #[must_use]
    pub fn learned_offset(&self, class: &str) -> Option<u64> {
        self.learned_offset.get(class).copied()
    }

    /// Whether the model would currently solve `(class, x)` in the forward pass:
    /// the class's rule is known and `x` is within the generalization radius of a
    /// solved input.
    fn can_generalize(&self, class: &str, x: u64) -> bool {
        if !self.learned_offset.contains_key(class) {
            return false;
        }
        let Some(solved) = self.solved_inputs.get(class) else {
            return false;
        };
        solved
            .iter()
            .any(|&s| x.abs_diff(s) <= self.generalization_radius)
    }

    /// The answer the learned rule assigns to `x` for `class`.
    fn apply_rule(&self, class: &str, x: u64) -> Option<u64> {
        let offset = self.learned_offset.get(class)?;
        Some((x % self.modulus + offset) % self.modulus)
    }
}

impl ReasoningModel for StaticReasoningModel {
    fn generate(&self, problem: &StarProblem) -> StarGeneration {
        let Some((class, x)) = parse_problem(&problem.statement) else {
            return StarGeneration::new("malformed problem: cannot parse", "?");
        };

        // 1. If the class's rule is known and x is within the solved frontier,
        //    apply it — genuine forward reasoning.
        if self.can_generalize(&class, x)
            && let Some(answer) = self.apply_rule(&class, x)
        {
            let offset = self.learned_offset.get(&class).copied().unwrap_or(0);
            let rationale = format!(
                "class {class}, input x={x}: applying the learned offset {offset}, \
                 {x} + {offset} = {} which is {answer} mod {}",
                x + offset,
                self.modulus
            );
            return StarGeneration::new(rationale, answer.to_string());
        }

        // 2. A memorized seed the model can reproduce verbatim.
        if let Some(&answer) = self.seeds.get(&(class.clone(), x)) {
            let rationale = format!(
                "class {class}, input x={x}: recalling the worked example, the answer is {answer}"
            );
            return StarGeneration::new(rationale, answer.to_string());
        }

        // 3. No rule and no memory: commit the un-generalized guess (offset 0),
        //    which is wrong whenever the true offset is non-zero. STaR keeps only
        //    correct answers, so this is exactly the material the correctness
        //    filter must discard.
        let guess = x % self.modulus;
        let rationale =
            format!("class {class}, input x={x}: no rule learned yet, guessing {guess}");
        StarGeneration::new(rationale, guess.to_string())
    }

    fn rationalize(&self, problem: &StarProblem, hint: &str) -> StarGeneration {
        match self.rationalization_style {
            RationalizationStyle::Cheat => {
                // Echo the hint with no derivation. The engine must reject this.
                let rationale = format!("the answer is {hint} because the answer is {hint}");
                StarGeneration::new(rationale, hint.to_string())
            }
            RationalizationStyle::Genuine => {
                let Some((class, x)) = parse_problem(&problem.statement) else {
                    // Can't ground a derivation in an unparseable problem; echo
                    // the hint (and be honestly flagged as a cheat downstream).
                    return StarGeneration::new(format!("the answer is {hint}"), hint.to_string());
                };
                let Ok(hint_value) = hint.trim().parse::<u64>() else {
                    return StarGeneration::new(format!("the answer is {hint}"), hint.to_string());
                };
                // Recover the offset the hint implies and show the working. This
                // is a real derivation: it names the intermediate offset and the
                // sum, not just the answer.
                let x_mod = x % self.modulus;
                let offset = (hint_value + self.modulus - x_mod % self.modulus) % self.modulus;
                let answer = (x_mod + offset) % self.modulus;
                let rationale = format!(
                    "class {class}, input x={x}: the offset consistent with the target is {offset}, \
                     so {x} + {offset} = {}, and {} mod {} = {answer}",
                    x + offset,
                    x + offset,
                    self.modulus
                );
                StarGeneration::new(rationale, answer.to_string())
            }
        }
    }

    fn update_from(&mut self, accumulated: &[StarRationale]) {
        // Rebuild the entire learned state from the accumulated set, so the
        // policy is a pure, order-independent function of what has been learned.
        let mut candidates: HashMap<String, Vec<(u64, u64)>> = HashMap::new();
        for rationale in accumulated {
            if let Some((class, x)) = parse_problem(&rationale.statement)
                && let Ok(answer) = rationale.answer.trim().parse::<u64>()
            {
                candidates.entry(class).or_default().push((x, answer));
            }
        }

        self.learned_offset.clear();
        self.solved_inputs.clear();
        for (class, examples) in candidates {
            let offsets: BTreeSet<u64> = examples
                .iter()
                .map(|&(x, answer)| {
                    let x_mod = x % self.modulus;
                    (answer % self.modulus + self.modulus - x_mod) % self.modulus
                })
                .collect();
            // A rule is induced only when every example of the class agrees on
            // the offset. Contradictory data induces nothing — no plausible-but-
            // wrong rule is invented.
            if offsets.len() == 1
                && let Some(&offset) = offsets.iter().next()
            {
                self.learned_offset.insert(class.clone(), offset);
            }
            self.solved_inputs
                .insert(class, examples.into_iter().map(|(x, _)| x).collect());
        }
    }
}

/// Parse a problem statement of the form `"[<class>] x=<n>"` into
/// `(class, x)`, or `None` when it does not match.
fn parse_problem(statement: &str) -> Option<(String, u64)> {
    let open = statement.find('[')? + 1;
    let close = statement[open..].find(']')? + open;
    let class = statement[open..close].trim().to_string();
    if class.is_empty() {
        return None;
    }
    let after = statement.find("x=")? + 2;
    let digits: String = statement[after..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        return None;
    }
    let x = digits.parse::<u64>().ok()?;
    Some((class, x))
}
