#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::default_constructed_unit_structs
)]
//! Unit tests for the `program_of_thought` module.

use crate::program_of_thought::engine::{
    MockProgramGenerator, PotOutput, ProgramGenerator, ProgramOfThoughtEngine,
};
use crate::program_of_thought::interpreter::Interpreter;
use crate::program_of_thought::parser::parse_program;
use crate::program_of_thought::types::{Expr, Op, PotError, Program, Statement};

// ── helpers ─────────────────────────────────────────────────────────────────

/// Parse `src` and evaluate it, returning the numeric result.
fn eval(src: &str) -> Result<f64, PotError> {
    let program = parse_program(src)?;
    Interpreter::new().eval(&program)
}

// ── Op ──────────────────────────────────────────────────────────────────────

#[test]
fn op_as_char_round_trips() {
    assert_eq!(Op::Add.as_char(), '+');
    assert_eq!(Op::Sub.as_char(), '-');
    assert_eq!(Op::Mul.as_char(), '*');
    assert_eq!(Op::Div.as_char(), '/');
}

#[test]
fn op_is_copy_and_eq() {
    let op = Op::Mul;
    let copy = op;
    assert_eq!(op, copy);
}

// ── Expr constructors ───────────────────────────────────────────────────────

#[test]
fn expr_num_constructor() {
    assert_eq!(Expr::num(3.0), Expr::Num(3.0));
    assert_eq!(Expr::num(7_i16), Expr::Num(7.0));
}

#[test]
fn expr_var_constructor() {
    assert_eq!(Expr::var("a"), Expr::Var("a".to_string()));
}

#[test]
fn expr_neg_uses_sub() {
    match Expr::negate(Expr::Num(5.0)) {
        Expr::Unary { op, expr } => {
            assert_eq!(op, Op::Sub);
            assert_eq!(*expr, Expr::Num(5.0));
        }
        other => panic!("expected Unary, got {other:?}"),
    }
}

#[test]
fn expr_binary_constructor() {
    let e = Expr::binary(Op::Add, Expr::Num(1.0), Expr::Num(2.0));
    assert_eq!(
        e,
        Expr::Binary {
            op: Op::Add,
            lhs: Box::new(Expr::Num(1.0)),
            rhs: Box::new(Expr::Num(2.0)),
        }
    );
}

// ── Statement constructors ──────────────────────────────────────────────────

#[test]
fn statement_assign_constructor() {
    assert_eq!(
        Statement::assign("x", Expr::Num(1.0)),
        Statement::Assign {
            name: "x".to_string(),
            expr: Expr::Num(1.0),
        }
    );
}

#[test]
fn statement_return_constructor() {
    assert_eq!(
        Statement::ret(Expr::Num(2.0)),
        Statement::Return(Expr::Num(2.0))
    );
}

// ── Program helpers ─────────────────────────────────────────────────────────

#[test]
fn program_new_and_len() {
    let p = Program::new(vec![Statement::assign("a", Expr::Num(1.0))]);
    assert_eq!(p.len(), 1);
    assert!(!p.is_empty());
}

#[test]
fn program_default_is_empty() {
    let p = Program::default();
    assert!(p.is_empty());
    assert_eq!(p.len(), 0);
}

// ── parse: simple assignment ────────────────────────────────────────────────

#[test]
fn parse_simple_assignment() {
    let program = parse_program("a = 3").unwrap();
    assert_eq!(
        program.statements,
        vec![Statement::Assign {
            name: "a".to_string(),
            expr: Expr::Num(3.0),
        }]
    );
}

#[test]
fn parse_assignment_with_decimal() {
    let program = parse_program("ratio = 2.5").unwrap();
    assert_eq!(
        program.statements,
        vec![Statement::assign("ratio", Expr::Num(2.5))]
    );
}

#[test]
fn parse_assignment_to_variable() {
    let program = parse_program("b = a").unwrap();
    assert_eq!(
        program.statements,
        vec![Statement::assign("b", Expr::Var("a".to_string()))]
    );
}

#[test]
fn parse_underscore_identifier() {
    let program = parse_program("total_count = 5").unwrap();
    assert_eq!(
        program.statements,
        vec![Statement::assign("total_count", Expr::Num(5.0))]
    );
}

#[test]
fn parse_return_statement() {
    let program = parse_program("return 42").unwrap();
    assert_eq!(program.statements, vec![Statement::Return(Expr::Num(42.0))]);
}

#[test]
fn parse_return_of_expression() {
    let program = parse_program("return a + 1").unwrap();
    assert_eq!(
        program.statements,
        vec![Statement::Return(Expr::binary(
            Op::Add,
            Expr::Var("a".to_string()),
            Expr::Num(1.0),
        ))]
    );
}

// ── parse + eval: the headline cases ────────────────────────────────────────

#[test]
fn eval_multiply_two_variables_is_twelve() {
    assert_eq!(eval("a = 3\nb = 4\nreturn a * b").unwrap(), 12.0);
}

#[test]
fn eval_precedence_add_then_mul() {
    assert_eq!(eval("2 + 3 * 4").unwrap(), 14.0);
}

#[test]
fn eval_precedence_mul_then_add() {
    assert_eq!(eval("3 * 4 + 2").unwrap(), 14.0);
}

#[test]
fn eval_parentheses_override_precedence() {
    assert_eq!(eval("(2 + 3) * 4").unwrap(), 20.0);
}

#[test]
fn eval_nested_parentheses() {
    assert_eq!(eval("((1 + 2) * (3 + 4))").unwrap(), 21.0);
}

#[test]
fn eval_unary_minus_then_add() {
    assert_eq!(eval("-5 + 2").unwrap(), -3.0);
}

#[test]
fn eval_unary_minus_on_parenthesised() {
    assert_eq!(eval("-(2 + 3)").unwrap(), -5.0);
}

#[test]
fn eval_double_unary_minus() {
    assert_eq!(eval("--5").unwrap(), 5.0);
}

#[test]
fn eval_unary_minus_in_multiplication() {
    assert_eq!(eval("3 * -2").unwrap(), -6.0);
}

#[test]
fn eval_subtraction() {
    assert_eq!(eval("10 - 4").unwrap(), 6.0);
}

#[test]
fn eval_subtraction_is_left_associative() {
    // (10 - 4) - 3 == 3, not 10 - (4 - 3) == 9
    assert_eq!(eval("10 - 4 - 3").unwrap(), 3.0);
}

#[test]
fn eval_division() {
    assert_eq!(eval("12 / 4").unwrap(), 3.0);
}

#[test]
fn eval_division_is_left_associative() {
    // (16 / 4) / 2 == 2, not 16 / (4 / 2) == 8
    assert_eq!(eval("16 / 4 / 2").unwrap(), 2.0);
}

#[test]
fn eval_division_producing_fraction() {
    assert_eq!(eval("7 / 2").unwrap(), 3.5);
}

#[test]
fn eval_mixed_operators() {
    // 100 - 50 / 5 + 2 * 3 == 100 - 10 + 6 == 96
    assert_eq!(eval("100 - 50 / 5 + 2 * 3").unwrap(), 96.0);
}

#[test]
fn eval_decimal_arithmetic() {
    assert_eq!(eval("1.5 + 2.5").unwrap(), 4.0);
}

// ── eval: errors ────────────────────────────────────────────────────────────

#[test]
fn eval_division_by_zero_errors() {
    assert_eq!(eval("1 / 0"), Err(PotError::DivisionByZero));
}

#[test]
fn eval_division_by_zero_variable_errors() {
    assert_eq!(eval("z = 0\nreturn 5 / z"), Err(PotError::DivisionByZero));
}

#[test]
fn eval_undefined_variable_errors() {
    assert_eq!(
        eval("return missing"),
        Err(PotError::UndefinedVariable("missing".to_string()))
    );
}

#[test]
fn eval_undefined_variable_in_assignment_errors() {
    assert_eq!(
        eval("a = b + 1"),
        Err(PotError::UndefinedVariable("b".to_string()))
    );
}

#[test]
fn eval_empty_program_via_interpreter_errors() {
    let empty = Program::default();
    assert_eq!(Interpreter::new().eval(&empty), Err(PotError::EmptyProgram));
}

// ── eval: multi-statement environment & control flow ────────────────────────

#[test]
fn eval_multi_statement_environment() {
    // a = 2; b = a + 3 (= 5); c = a * b (= 10); return c
    assert_eq!(eval("a = 2\nb = a + 3\nc = a * b\nreturn c").unwrap(), 10.0);
}

#[test]
fn eval_reassignment_uses_latest_value() {
    assert_eq!(eval("x = 1\nx = x + 10\nreturn x").unwrap(), 11.0);
}

#[test]
fn eval_chained_dependencies() {
    assert_eq!(
        eval("a = 1\nb = a + 1\nc = b + 1\nd = c + 1\nreturn d").unwrap(),
        4.0
    );
}

#[test]
fn eval_return_short_circuits_ignoring_later_statements() {
    // The `return` fires before `b` is ever assigned, so referencing a
    // would-be-later variable must not happen — result is purely from `a`.
    assert_eq!(eval("a = 7\nreturn a\nb = undefined_thing").unwrap(), 7.0);
}

#[test]
fn eval_return_short_circuit_does_not_evaluate_bad_tail() {
    // If the tail were evaluated, division by zero would error. It must not.
    let result = eval("a = 5\nreturn a\nc = 1 / 0");
    assert_eq!(result.unwrap(), 5.0);
}

#[test]
fn eval_no_return_uses_last_assignment() {
    assert_eq!(eval("a = 3\nb = 4").unwrap(), 4.0);
}

#[test]
fn eval_no_return_single_assignment() {
    assert_eq!(eval("answer = 99").unwrap(), 99.0);
}

#[test]
fn eval_no_return_last_assignment_uses_env() {
    // Last assignment depends on earlier ones; its value is the result.
    assert_eq!(eval("a = 10\nb = a * 2").unwrap(), 20.0);
}

// ── parse / program errors ──────────────────────────────────────────────────

#[test]
fn parse_empty_source_errors() {
    assert_eq!(parse_program(""), Err(PotError::EmptyProgram));
}

#[test]
fn parse_whitespace_only_source_errors() {
    assert_eq!(parse_program("   \n  \n\t"), Err(PotError::EmptyProgram));
}

#[test]
fn parse_unbalanced_paren_errors() {
    assert!(matches!(
        parse_program("a = (1 + 2"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_trailing_tokens_errors() {
    assert!(matches!(
        parse_program("a = 1 2"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_unexpected_character_errors() {
    assert!(matches!(
        parse_program("a = 1 % 2"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_missing_rhs_errors() {
    assert!(matches!(
        parse_program("a = 1 +"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_malformed_number_errors() {
    assert!(matches!(
        parse_program("a = 1.2.3"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_non_identifier_assignment_target_errors() {
    assert!(matches!(
        parse_program("3 = 4"),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn parse_bare_expression_without_assignment_is_ok() {
    // A single bare expression line is treated as the program's result.
    let program = parse_program("2 + 2").unwrap();
    assert_eq!(program.len(), 1);
}

// ── whitespace / blank-line tolerance ───────────────────────────────────────

#[test]
fn parse_tolerates_blank_lines() {
    let src = "\na = 3\n\n\nb = 4\n\nreturn a + b\n";
    assert_eq!(eval(src).unwrap(), 7.0);
}

#[test]
fn parse_tolerates_surrounding_whitespace() {
    let src = "   a = 3   \n   return a * 2   ";
    assert_eq!(eval(src).unwrap(), 6.0);
}

#[test]
fn parse_tolerates_internal_whitespace() {
    assert_eq!(eval("a=3\nb  =  4\nreturn a*b").unwrap(), 12.0);
}

// ── PotError messages ───────────────────────────────────────────────────────

#[test]
fn error_messages_match_spec() {
    assert_eq!(
        PotError::EmptyQuestion.to_string(),
        "question must not be empty"
    );
    assert_eq!(PotError::EmptyProgram.to_string(), "empty program");
    assert_eq!(
        PotError::ParseError("boom".to_string()).to_string(),
        "parse error: boom"
    );
    assert_eq!(
        PotError::UndefinedVariable("x".to_string()).to_string(),
        "undefined variable: x"
    );
    assert_eq!(PotError::DivisionByZero.to_string(), "division by zero");
}

// ── MockProgramGenerator ────────────────────────────────────────────────────

#[test]
fn mock_generator_returns_fixed_source() {
    let generator = MockProgramGenerator::new("a = 1");
    assert_eq!(generator.generate("anything"), "a = 1");
    assert_eq!(generator.generate("something else"), "a = 1");
}

#[test]
fn mock_generator_ignores_question() {
    let generator = MockProgramGenerator::new("return 5");
    assert_eq!(generator.generate(""), generator.generate("q"));
}

// ── ProgramOfThoughtEngine: end to end ──────────────────────────────────────

#[test]
fn engine_run_end_to_end() {
    let generator = MockProgramGenerator::new("a = 3\nb = 4\nreturn a * b");
    let engine = ProgramOfThoughtEngine::new();
    let output = engine.run("compute 3 times 4", &generator).unwrap();

    assert_eq!(output.value, 12.0);
    assert_eq!(output.answer_text, "The answer is 12");
    assert_eq!(output.program.len(), 3);
}

#[test]
fn engine_run_answer_text_for_fraction() {
    let generator = MockProgramGenerator::new("return 7 / 2");
    let engine = ProgramOfThoughtEngine::new();
    let output = engine.run("half of seven", &generator).unwrap();

    assert_eq!(output.value, 3.5);
    assert_eq!(output.answer_text, "The answer is 3.5");
}

#[test]
fn engine_run_answer_text_for_negative() {
    let generator = MockProgramGenerator::new("return -5 + 2");
    let engine = ProgramOfThoughtEngine::new();
    let output = engine.run("negative", &generator).unwrap();

    assert_eq!(output.value, -3.0);
    assert_eq!(output.answer_text, "The answer is -3");
}

#[test]
fn engine_empty_question_errors() {
    let generator = MockProgramGenerator::new("return 1");
    let engine = ProgramOfThoughtEngine::new();
    assert_eq!(engine.run("   ", &generator), Err(PotError::EmptyQuestion));
}

#[test]
fn engine_empty_program_errors() {
    let generator = MockProgramGenerator::new("\n\n");
    let engine = ProgramOfThoughtEngine::new();
    assert_eq!(
        engine.run("question", &generator),
        Err(PotError::EmptyProgram)
    );
}

#[test]
fn engine_propagates_parse_error() {
    let generator = MockProgramGenerator::new("a = (1 +");
    let engine = ProgramOfThoughtEngine::new();
    assert!(matches!(
        engine.run("question", &generator),
        Err(PotError::ParseError(_))
    ));
}

#[test]
fn engine_propagates_division_by_zero() {
    let generator = MockProgramGenerator::new("return 1 / 0");
    let engine = ProgramOfThoughtEngine::new();
    assert_eq!(
        engine.run("question", &generator),
        Err(PotError::DivisionByZero)
    );
}

#[test]
fn engine_propagates_undefined_variable() {
    let generator = MockProgramGenerator::new("return ghost");
    let engine = ProgramOfThoughtEngine::new();
    assert_eq!(
        engine.run("question", &generator),
        Err(PotError::UndefinedVariable("ghost".to_string()))
    );
}

#[test]
fn engine_accepts_dyn_generator() {
    // Confirm the `?Sized` bound lets a trait object be used.
    let generator: Box<dyn ProgramGenerator> = Box::new(MockProgramGenerator::new("return 6 * 7"));
    let engine = ProgramOfThoughtEngine::new();
    let output = engine.run("life", generator.as_ref()).unwrap();
    assert_eq!(output.value, 42.0);
}

// ── determinism ─────────────────────────────────────────────────────────────

#[test]
fn engine_is_deterministic_across_runs() {
    let generator = MockProgramGenerator::new("a = 6\nb = 7\nreturn a * b");
    let engine = ProgramOfThoughtEngine::new();

    let first = engine.run("q", &generator).unwrap();
    let second = engine.run("q", &generator).unwrap();
    let third = engine.run("q", &generator).unwrap();

    assert_eq!(first, second);
    assert_eq!(second, third);
}

#[test]
fn eval_is_deterministic() {
    let src = "a = 2\nb = 3\nc = a * b + 1\nreturn c";
    let r1 = eval(src).unwrap();
    let r2 = eval(src).unwrap();
    assert_eq!(r1, r2);
    assert_eq!(r1, 7.0);
}

#[test]
fn pot_output_is_clonable_and_eq() {
    let output = PotOutput {
        program: Program::new(vec![Statement::Return(Expr::Num(1.0))]),
        value: 1.0,
        answer_text: "The answer is 1".to_string(),
    };
    assert_eq!(output.clone(), output);
}

// ── interpreter reuse ───────────────────────────────────────────────────────

#[test]
fn interpreter_reuse_has_no_residual_state() {
    let interp = Interpreter::new();

    let p1 = parse_program("a = 5\nreturn a").unwrap();
    assert_eq!(interp.eval(&p1).unwrap(), 5.0);

    // `a` from the first program must NOT leak into the second.
    let p2 = parse_program("return a").unwrap();
    assert_eq!(
        interp.eval(&p2),
        Err(PotError::UndefinedVariable("a".to_string()))
    );
}

#[test]
fn interpreter_default_equals_new() {
    assert_eq!(Interpreter::default(), Interpreter::new());
}
