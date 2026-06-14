//! Deterministic interpreter for the Program-of-Thoughts arithmetic DSL.
//!
//! The [`Interpreter`] executes a parsed [`Program`] over a name → value
//! environment ([`std::collections::HashMap<String, f64>`]). Statements run in
//! source order:
//!
//! * An [`Statement::Assign`] evaluates its expression and binds the result to
//!   the named variable, overwriting any previous binding.
//! * A [`Statement::Return`] evaluates its expression and **short-circuits**:
//!   the program's result is that value and no later statements run.
//!
//! If the program contains no `return`, the result is the value of the **last**
//! assignment. Referencing an unbound variable yields
//! [`PotError::UndefinedVariable`]; dividing by zero yields
//! [`PotError::DivisionByZero`].

use std::collections::HashMap;

use crate::program_of_thought::types::{Expr, Op, PotError, Program, Statement};

// ── Interpreter ─────────────────────────────────────────────────────────────

/// A stateless evaluator for Program-of-Thoughts programs.
///
/// Each call to [`Interpreter::eval`] builds a fresh environment, so an
/// `Interpreter` value carries no run-to-run state and may be reused freely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Interpreter;

impl Interpreter {
    /// Create a new interpreter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Evaluate `program` and return its numeric result.
    ///
    /// Statements execute in order over a fresh environment. A
    /// [`Statement::Return`] short-circuits with its value; otherwise the value
    /// of the final [`Statement::Assign`] is returned.
    ///
    /// # Errors
    ///
    /// * [`PotError::EmptyProgram`] if `program` has no statements.
    /// * [`PotError::UndefinedVariable`] if an expression references an unbound
    ///   variable.
    /// * [`PotError::DivisionByZero`] if a division has a zero divisor.
    #[allow(clippy::unused_self)]
    pub fn eval(&self, program: &Program) -> Result<f64, PotError> {
        if program.is_empty() {
            return Err(PotError::EmptyProgram);
        }

        let mut env: HashMap<String, f64> = HashMap::new();
        let mut last_assigned: Option<f64> = None;

        for statement in &program.statements {
            match statement {
                Statement::Assign { name, expr } => {
                    let value = eval_expr(expr, &env)?;
                    env.insert(name.clone(), value);
                    last_assigned = Some(value);
                }
                Statement::Return(expr) => {
                    return eval_expr(expr, &env);
                }
            }
        }

        // No `return` encountered: the result is the last assignment's value.
        // `last_assigned` is always `Some` here because a non-empty program with
        // no `Return` must contain at least one `Assign`.
        last_assigned.ok_or(PotError::EmptyProgram)
    }
}

// ── Expression evaluation ───────────────────────────────────────────────────

/// Recursively evaluate `expr` against `env`.
fn eval_expr(expr: &Expr, env: &HashMap<String, f64>) -> Result<f64, PotError> {
    match expr {
        Expr::Num(value) => Ok(*value),
        Expr::Var(name) => env
            .get(name)
            .copied()
            .ok_or_else(|| PotError::UndefinedVariable(name.clone())),
        Expr::Unary { op, expr } => {
            let value = eval_expr(expr, env)?;
            // The parser only ever emits `Op::Sub` for a unary node (negation).
            // Any other operator is treated as the identity for total safety.
            match op {
                Op::Sub => Ok(-value),
                _ => Ok(value),
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let left = eval_expr(lhs, env)?;
            let right = eval_expr(rhs, env)?;
            match op {
                Op::Add => Ok(left + right),
                Op::Sub => Ok(left - right),
                Op::Mul => Ok(left * right),
                Op::Div => {
                    if right == 0.0 {
                        Err(PotError::DivisionByZero)
                    } else {
                        Ok(left / right)
                    }
                }
            }
        }
    }
}
