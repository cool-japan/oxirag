//! The Program-of-Thoughts engine: generate a program, parse it, execute it.
//!
//! [`ProgramOfThoughtEngine`] wires the three pieces of the pattern together.
//! Given a question and a caller-supplied [`ProgramGenerator`], it:
//!
//! 1. asks the generator for DSL source that computes the answer,
//! 2. parses that source with [`parse_program`],
//! 3. executes it with the [`Interpreter`],
//! 4. and formats a natural-language [`PotOutput::answer_text`].
//!
//! The generator is supplied *per call* (mirroring the caller-supplies-executor
//! pattern used across the crate), so the engine itself stays free of any
//! model or I/O. [`MockProgramGenerator`] provides a deterministic generator
//! for tests and examples.

use crate::program_of_thought::interpreter::Interpreter;
use crate::program_of_thought::parser::parse_program;
use crate::program_of_thought::types::{PotError, Program};

// ── ProgramGenerator ────────────────────────────────────────────────────────

/// Produces Program-of-Thoughts DSL source for a question.
///
/// This is the only seam where a language model would plug in: an
/// implementation maps a natural-language `question` to DSL source (a sequence
/// of assignments and an optional `return`). Implementations are **pure sync**
/// — no I/O, no async. [`MockProgramGenerator`] is provided for tests.
pub trait ProgramGenerator {
    /// Generate a program (DSL source) that computes the answer to `question`.
    fn generate(&self, question: &str) -> String;
}

// ── MockProgramGenerator ────────────────────────────────────────────────────

/// Deterministic [`ProgramGenerator`] for tests and examples.
///
/// Returns the same pre-configured DSL `src` for every question, ignoring the
/// question text entirely. This isolates the parse-and-execute pipeline from any
/// real model so behaviour is fully reproducible.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockProgramGenerator {
    /// The DSL source returned by every [`ProgramGenerator::generate`] call.
    pub src: String,
}

impl MockProgramGenerator {
    /// Create a mock generator that always emits `src`.
    #[must_use]
    pub fn new(src: impl Into<String>) -> Self {
        Self { src: src.into() }
    }
}

impl ProgramGenerator for MockProgramGenerator {
    fn generate(&self, _question: &str) -> String {
        self.src.clone()
    }
}

// ── PotOutput ───────────────────────────────────────────────────────────────

/// The full result of a Program-of-Thoughts run.
#[derive(Debug, Clone, PartialEq)]
pub struct PotOutput {
    /// The parsed program that was executed.
    pub program: Program,
    /// The numeric value the interpreter computed.
    pub value: f64,
    /// A natural-language rendering of the answer, e.g. `"The answer is 12"`.
    pub answer_text: String,
}

// ── ProgramOfThoughtEngine ──────────────────────────────────────────────────

/// Drives the Program-of-Thoughts pattern end to end.
///
/// The engine holds no state; the [`ProgramGenerator`] is supplied per call to
/// [`ProgramOfThoughtEngine::run`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramOfThoughtEngine;

impl ProgramOfThoughtEngine {
    /// Create a new engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Run the full pipeline for `question` using `generator`.
    ///
    /// Generates DSL source, parses it into a [`Program`], evaluates it to a
    /// numeric value, and formats [`PotOutput::answer_text`].
    ///
    /// # Errors
    ///
    /// * [`PotError::EmptyQuestion`] if `question` is empty after trimming.
    /// * [`PotError::EmptyProgram`] if the generated source has no statements.
    /// * [`PotError::ParseError`] if the generated source fails to parse.
    /// * [`PotError::UndefinedVariable`] / [`PotError::DivisionByZero`] if
    ///   evaluation fails.
    #[allow(clippy::unused_self)]
    pub fn run<G>(&self, question: &str, generator: &G) -> Result<PotOutput, PotError>
    where
        G: ProgramGenerator + ?Sized,
    {
        if question.trim().is_empty() {
            return Err(PotError::EmptyQuestion);
        }

        let src = generator.generate(question);
        let program = parse_program(&src)?;
        let value = Interpreter::new().eval(&program)?;
        let answer_text = format_answer(value);

        Ok(PotOutput {
            program,
            value,
            answer_text,
        })
    }
}

// ── Answer formatting ───────────────────────────────────────────────────────

/// Render a numeric result as `"The answer is {value}"`.
///
/// Integral values print without a trailing `.0` (e.g. `12`, not `12.0`) so the
/// text reads naturally; non-integral values use the default `f64` formatting.
fn format_answer(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("The answer is {value:.0}")
    } else {
        format!("The answer is {value}")
    }
}
