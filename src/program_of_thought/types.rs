//! AST types and errors for the `program_of_thought` module.
//!
//! Program-of-Thoughts (Chen et al. 2022) separates **reasoning** from
//! **computation**: rather than performing arithmetic in natural language, a
//! model emits a small *program* — a sequence of variable assignments over an
//! arithmetic expression language — and a deterministic interpreter executes it
//! to obtain the numeric answer. This decouples error-prone language-model
//! arithmetic from the reasoning that produced the program.
//!
//! This file defines the abstract syntax tree for the tiny arithmetic DSL and
//! the module's error type:
//!
//! * [`Op`] — the four binary arithmetic operators.
//! * [`Expr`] — an arithmetic expression (numbers, variables, unary negation,
//!   and binary operations).
//! * [`Statement`] — a single program statement (an assignment or a return).
//! * [`Program`] — an ordered list of [`Statement`]s.
//! * [`PotError`] — parse and evaluation errors.
//!
//! The grammar parsed into this AST (see [`parser`](crate::program_of_thought::parser)) is:
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

use thiserror::Error;

// ── Op ──────────────────────────────────────────────────────────────────────

/// A binary arithmetic operator.
///
/// `Op` is shared by both [`Expr::Binary`] and [`Expr::Unary`]. For unary
/// negation only [`Op::Sub`] is meaningful (it denotes `0 - operand`); the
/// parser never produces a unary node with any other operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
}

impl Op {
    /// Return the source character that denotes this operator.
    #[must_use]
    pub fn as_char(self) -> char {
        match self {
            Op::Add => '+',
            Op::Sub => '-',
            Op::Mul => '*',
            Op::Div => '/',
        }
    }
}

// ── Expr ────────────────────────────────────────────────────────────────────

/// An arithmetic expression node.
///
/// Unary negation (`-x`) is represented by [`Expr::Unary`] carrying [`Op::Sub`],
/// which the interpreter evaluates as `0 - x`. This keeps the operator set
/// minimal while giving negation an explicit, well-typed AST node (chosen over
/// desugaring to `Binary { Sub, Num(0.0), .. }` so the original source shape is
/// preserved for inspection).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A numeric literal, e.g. `3` or `2.5`.
    Num(f64),
    /// A variable reference, e.g. `a`. Resolved against the environment at
    /// evaluation time; an unbound name yields [`PotError::UndefinedVariable`].
    Var(String),
    /// A unary operation. Only [`Op::Sub`] is produced by the parser, denoting
    /// arithmetic negation of the inner expression.
    Unary {
        /// The unary operator (always [`Op::Sub`] for negation).
        op: Op,
        /// The operand expression.
        expr: Box<Expr>,
    },
    /// A binary operation combining two sub-expressions.
    Binary {
        /// The binary operator.
        op: Op,
        /// The left-hand operand.
        lhs: Box<Expr>,
        /// The right-hand operand.
        rhs: Box<Expr>,
    },
}

impl Expr {
    /// Build an [`Expr::Num`] from any value convertible into `f64`.
    #[must_use]
    pub fn num(value: impl Into<f64>) -> Self {
        Expr::Num(value.into())
    }

    /// Build an [`Expr::Var`] from anything convertible into a `String`.
    #[must_use]
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    /// Build an [`Expr::Unary`] negation node (`-inner`).
    #[must_use]
    pub fn negate(inner: Expr) -> Self {
        Expr::Unary {
            op: Op::Sub,
            expr: Box::new(inner),
        }
    }

    /// Build an [`Expr::Binary`] node combining `lhs` and `rhs` with `op`.
    #[must_use]
    pub fn binary(op: Op, lhs: Expr, rhs: Expr) -> Self {
        Expr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// ── Statement ───────────────────────────────────────────────────────────────

/// A single statement in a [`Program`].
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Bind the value of `expr` to the variable `name` in the environment.
    Assign {
        /// The variable being assigned.
        name: String,
        /// The expression whose value is bound to `name`.
        expr: Expr,
    },
    /// Return the value of the wrapped expression, short-circuiting execution.
    Return(Expr),
}

impl Statement {
    /// Build a [`Statement::Assign`] binding `name` to `expr`.
    #[must_use]
    pub fn assign(name: impl Into<String>, expr: Expr) -> Self {
        Statement::Assign {
            name: name.into(),
            expr,
        }
    }

    /// Build a [`Statement::Return`] of `expr`.
    #[must_use]
    pub fn ret(expr: Expr) -> Self {
        Statement::Return(expr)
    }
}

// ── Program ─────────────────────────────────────────────────────────────────

/// A parsed Program-of-Thoughts program: an ordered list of [`Statement`]s.
///
/// Execution semantics (see
/// [`Interpreter`](crate::program_of_thought::interpreter::Interpreter)):
/// statements run top to bottom over a name → value environment. A
/// [`Statement::Return`] short-circuits and yields its value. If no `return` is
/// present, the program's result is the value of the **last** assignment.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    /// The program's statements, in source order.
    pub statements: Vec<Statement>,
}

impl Program {
    /// Create a program from a vector of statements.
    #[must_use]
    pub fn new(statements: Vec<Statement>) -> Self {
        Self { statements }
    }

    /// Return `true` when the program has no statements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }

    /// Return the number of statements in the program.
    #[must_use]
    pub fn len(&self) -> usize {
        self.statements.len()
    }
}

// ── PotError ────────────────────────────────────────────────────────────────

/// Errors produced while parsing or evaluating a Program-of-Thoughts program.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PotError {
    /// The question handed to the engine was empty after trimming.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// The program contained no statements.
    #[error("empty program")]
    EmptyProgram,
    /// The source failed to parse; the payload describes the failure.
    #[error("parse error: {0}")]
    ParseError(String),
    /// An expression referenced a variable that was never assigned.
    #[error("undefined variable: {0}")]
    UndefinedVariable(String),
    /// A division expression had a zero divisor.
    #[error("division by zero")]
    DivisionByZero,
}
