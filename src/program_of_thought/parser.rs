//! Recursive-descent parser for the Program-of-Thoughts arithmetic DSL.
//!
//! The parser turns DSL source into a [`Program`] AST. It is split into a small
//! `Token` lexer and a recursive-descent `Parser` that implements the
//! standard expression grammar with correct operator precedence and
//! left-associativity:
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
//! Blank lines and surrounding whitespace are ignored. Each non-blank line is a
//! single statement. All failures surface as [`PotError::ParseError`] with a
//! human-readable message.

use crate::program_of_thought::types::{Expr, Op, PotError, Program, Statement};

// ── Tokens ──────────────────────────────────────────────────────────────────

/// A lexical token within a single statement line.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A numeric literal.
    Number(f64),
    /// An identifier (variable name or the `return` keyword).
    Ident(String),
    /// An arithmetic operator.
    Op(Op),
    /// The assignment `=` sign.
    Equals,
    /// A left parenthesis `(`.
    LParen,
    /// A right parenthesis `)`.
    RParen,
}

/// Tokenize a single line of DSL source.
///
/// # Errors
///
/// Returns [`PotError::ParseError`] on an unexpected character or a malformed
/// numeric literal.
fn tokenize(line: &str) -> Result<Vec<Token>, PotError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => {
                i += 1;
            }
            '+' => {
                tokens.push(Token::Op(Op::Add));
                i += 1;
            }
            '-' => {
                tokens.push(Token::Op(Op::Sub));
                i += 1;
            }
            '*' => {
                tokens.push(Token::Op(Op::Mul));
                i += 1;
            }
            '/' => {
                tokens.push(Token::Op(Op::Div));
                i += 1;
            }
            '=' => {
                tokens.push(Token::Equals);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut seen_dot = false;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    if chars[i] == '.' {
                        if seen_dot {
                            return Err(PotError::ParseError(format!(
                                "malformed number with multiple '.' in `{line}`"
                            )));
                        }
                        seen_dot = true;
                    }
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                let value: f64 = text.parse().map_err(|_| {
                    PotError::ParseError(format!("invalid number literal `{text}`"))
                })?;
                tokens.push(Token::Number(value));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(text));
            }
            other => {
                return Err(PotError::ParseError(format!(
                    "unexpected character `{other}` in `{line}`"
                )));
            }
        }
    }

    Ok(tokens)
}

// ── Parser ──────────────────────────────────────────────────────────────────

/// Recursive-descent parser over a token stream for one statement line.
struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    /// The original line, kept for richer error messages.
    line: &'a str,
}

impl<'a> Parser<'a> {
    /// Build a parser over `tokens` lexed from `line`.
    fn new(tokens: &'a [Token], line: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            line,
        }
    }

    /// Peek at the current token without consuming it.
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// Consume and return the current token, advancing the cursor.
    fn advance(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Parse `expr := term (('+' | '-') term)*` with left-associativity.
    fn parse_expr(&mut self) -> Result<Expr, PotError> {
        let mut lhs = self.parse_term()?;
        while let Some(&Token::Op(op)) = self.peek() {
            if op == Op::Add || op == Op::Sub {
                self.advance();
                let rhs = self.parse_term()?;
                lhs = Expr::binary(op, lhs, rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    /// Parse `term := factor (('*' | '/') factor)*` with left-associativity.
    fn parse_term(&mut self) -> Result<Expr, PotError> {
        let mut lhs = self.parse_factor()?;
        while let Some(&Token::Op(op)) = self.peek() {
            if op == Op::Mul || op == Op::Div {
                self.advance();
                let rhs = self.parse_factor()?;
                lhs = Expr::binary(op, lhs, rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    /// Parse `factor := number | identifier | '(' expr ')' | '-' factor`.
    fn parse_factor(&mut self) -> Result<Expr, PotError> {
        // Clone the current token out of the borrow so error arms below can also
        // read `self.line` without overlapping the `&Token` borrow.
        let token = self.advance().cloned();
        match token {
            Some(Token::Number(value)) => Ok(Expr::Num(value)),
            Some(Token::Ident(name)) => Ok(Expr::Var(name)),
            Some(Token::Op(Op::Sub)) => {
                // Unary negation binds tighter than the binary operators and
                // recurses into another factor, so `--5` and `-(2 + 3)` parse.
                let inner = self.parse_factor()?;
                Ok(Expr::negate(inner))
            }
            Some(Token::LParen) => {
                let inner = self.parse_expr()?;
                match self.advance() {
                    Some(Token::RParen) => Ok(inner),
                    _ => Err(PotError::ParseError(format!(
                        "expected `)` in `{}`",
                        self.line
                    ))),
                }
            }
            Some(other) => Err(PotError::ParseError(format!(
                "unexpected token {other:?} in `{}`",
                self.line
            ))),
            None => Err(PotError::ParseError(format!(
                "unexpected end of expression in `{}`",
                self.line
            ))),
        }
    }

    /// Ensure the whole token stream was consumed.
    fn expect_eof(&self) -> Result<(), PotError> {
        if self.pos == self.tokens.len() {
            Ok(())
        } else {
            Err(PotError::ParseError(format!(
                "trailing tokens after expression in `{}`",
                self.line
            )))
        }
    }
}

// ── Statement / Program parsing ─────────────────────────────────────────────

/// Parse a single non-blank line into a [`Statement`].
fn parse_statement(line: &str) -> Result<Statement, PotError> {
    let tokens = tokenize(line)?;
    if tokens.is_empty() {
        return Err(PotError::ParseError(format!("empty statement in `{line}`")));
    }

    // `return expr`
    if let Token::Ident(kw) = &tokens[0]
        && kw == "return"
    {
        let mut parser = Parser::new(&tokens[1..], line);
        let expr = parser.parse_expr()?;
        parser.expect_eof()?;
        return Ok(Statement::Return(expr));
    }

    // `name = expr`
    if tokens.len() >= 2 && tokens[1] == Token::Equals {
        let name = match &tokens[0] {
            Token::Ident(name) => name.clone(),
            other => {
                return Err(PotError::ParseError(format!(
                    "assignment target must be an identifier, found {other:?} in `{line}`"
                )));
            }
        };
        let mut parser = Parser::new(&tokens[2..], line);
        let expr = parser.parse_expr()?;
        parser.expect_eof()?;
        return Ok(Statement::Assign { name, expr });
    }

    // Bare expression: the line is neither an assignment nor a `return`, so treat
    // it as the program's result (equivalent to `return <expr>`). This lets a
    // single-line program such as `2 + 3 * 4` be a valid program.
    let mut parser = Parser::new(&tokens, line);
    let expr = parser.parse_expr()?;
    parser.expect_eof()?;
    Ok(Statement::Return(expr))
}

/// Parse newline-separated DSL `src` into a [`Program`].
///
/// Blank lines (and lines consisting only of whitespace) are skipped. Each
/// remaining line is parsed as one [`Statement`]: an assignment `name = expr`
/// or a `return expr`. Expressions follow standard arithmetic precedence with
/// left-associative `+ - * /`, parenthesised grouping, and unary minus.
///
/// # Errors
///
/// Returns [`PotError::EmptyProgram`] when `src` has no non-blank lines, or
/// [`PotError::ParseError`] describing the first malformed line.
pub fn parse_program(src: &str) -> Result<Program, PotError> {
    let mut statements = Vec::new();

    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        statements.push(parse_statement(trimmed)?);
    }

    if statements.is_empty() {
        return Err(PotError::EmptyProgram);
    }

    Ok(Program::new(statements))
}
