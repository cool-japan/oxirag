//! Program-of-Thoughts (Chen et al. 2022) — separate **reasoning** from
//! **computation**.
//!
//! Instead of performing arithmetic in natural language (where language models
//! are error-prone), the model emits a small **program**: a sequence of variable
//! assignments over a tiny arithmetic expression language, optionally ending in a
//! `return`. A deterministic **interpreter** then executes that program to obtain
//! the numeric answer. Decoupling the brittle arithmetic from the reasoning that
//! produced the program is the whole point of the technique.
//!
//! This module implements a real, self-contained DSL:
//!
//! * [`parser`] — a recursive-descent parser with standard operator precedence
//!   and left-associativity, parenthesised grouping, and unary minus.
//! * [`interpreter`] — a deterministic evaluator over a name → value environment,
//!   with `return` short-circuiting and division-by-zero / undefined-variable
//!   errors.
//! * [`engine`] — [`ProgramOfThoughtEngine`] ties generation, parsing and
//!   execution together; the [`ProgramGenerator`] (where a model would plug in)
//!   is supplied per call.
//! * [`types`] — the AST ([`Op`], [`Expr`], [`Statement`], [`Program`]) and the
//!   [`PotError`] error type.
//!
//! # Grammar
//!
//! ```text
//! program    := statement (NEWLINE statement)*
//! statement  := assignment | return
//! assignment := identifier '=' expr
//! return     := 'return' expr
//! expr       := term (('+' | '-') term)*
//! term       := factor (('*' | '/') factor)*
//! factor     := number | identifier | '(' expr ')' | '-' factor
//! ```
//!
//! Blank lines and surrounding whitespace are ignored; each non-blank line is one
//! statement. If a program has no `return`, its result is the value of the last
//! assignment.
//!
//! # Example
//!
//! ```
//! use oxirag::program_of_thought::{
//!     MockProgramGenerator, ProgramOfThoughtEngine,
//! };
//!
//! // The "model" emits a program that computes the answer step by step.
//! let generator = MockProgramGenerator::new(
//!     "apples = 3\nbaskets = 4\nreturn apples * baskets",
//! );
//!
//! let engine = ProgramOfThoughtEngine::new();
//! let output = engine
//!     .run("How many apples across 4 baskets of 3?", &generator)
//!     .unwrap();
//!
//! assert_eq!(output.value, 12.0);
//! assert_eq!(output.answer_text, "The answer is 12");
//! ```
//!
//! Parsing and evaluation can also be used directly, without the engine:
//!
//! ```
//! use oxirag::program_of_thought::{parse_program, Interpreter};
//!
//! let program = parse_program("2 + 3 * 4").unwrap(); // precedence respected
//! let value = Interpreter::new().eval(&program).unwrap();
//! assert_eq!(value, 14.0);
//! ```

pub mod engine;
pub mod interpreter;
pub mod parser;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::{MockProgramGenerator, PotOutput, ProgramGenerator, ProgramOfThoughtEngine};
pub use interpreter::Interpreter;
pub use parser::parse_program;
pub use types::{Expr, Op, PotError, Program, Statement};
