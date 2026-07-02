//! Pluggable traits ([`RewooPlanSource`], [`RewooRetriever`],
//! [`RewooGenerator`]) with deterministic `Mock*` implementations, the
//! placeholder-substitution and arithmetic-evaluation helpers, and the
//! [`RewooPlanner`] / [`RewooWorker`] / [`RewooSolver`] / [`RewooPipeline`]
//! orchestration logic.
//!
//! ## Placeholder substitution ([`substitute_placeholders`])
//!
//! Placeholder tokens have the exact form `#E<digits>` (e.g. `#E1`, `#E10`).
//! Substitution is done with a small hand-rolled scanner using **maximal
//! munch**: once a `#E` prefix is found, *every* following ASCII digit is
//! consumed before the token is considered complete. This means `#E10` is
//! always read as placeholder `10`, never as placeholder `1` followed by a
//! stray literal `"0"` — the exact collision a naive `str::replace("#E1",
//! ...)` substring search would fall into. No naive substring search or
//! regex is used anywhere in this module.
//!
//! ## Arithmetic evaluation ([`evaluate_expression`])
//!
//! [`crate::rewoo::types::RewooAction::Compute`] expressions are evaluated by
//! a small recursive-descent parser supporting `+ - * /`, parentheses, unary
//! sign, and decimal literals, with the usual operator precedence
//! (`*`/`/` bind tighter than `+`/`-`).

use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::types::{
    PlaceholderVar, RewooAction, RewooConfig, RewooError, RewooEvidence, RewooOutcome, RewooPlan,
};

// ── Placeholder scanning / substitution ───────────────────────────────────────

/// Scans `text` for `#E<digits>` placeholder tokens using maximal-munch digit
/// matching. Returns the ordered list of `(byte_start, byte_end, var)`
/// matches, where `text[byte_start..byte_end]` is exactly the matched token
/// (e.g. `"#E10"`).
///
/// `#` and `E` and ASCII digits are all single-byte, non-continuation UTF-8
/// bytes (every byte value below `0x80`), so every offset this function
/// records — and every offset [`substitute_placeholders`] slices `text`
/// at — falls on a valid `char` boundary even when `text` also contains
/// multi-byte characters elsewhere.
fn scan_placeholder_tokens(text: &str) -> Vec<(usize, usize, PlaceholderVar)> {
    let bytes = text.as_bytes();
    let mut matches = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'#' && i + 1 < bytes.len() && bytes[i + 1] == b'E' {
            let digits_start = i + 2;
            let mut j = digits_start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits_start
                && let Ok(id) = text[digits_start..j].parse::<usize>()
            {
                matches.push((i, j, PlaceholderVar::new(id)));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    matches
}

/// Returns every placeholder referenced by `text`, in the order they appear
/// (duplicates included).
fn referenced_placeholders(text: &str) -> Vec<PlaceholderVar> {
    scan_placeholder_tokens(text)
        .into_iter()
        .map(|(_, _, var)| var)
        .collect()
}

/// Replaces every `#E<n>` placeholder token in `text` with its resolved
/// evidence text from `evidence`.
///
/// # Errors
///
/// Returns [`RewooError::UnresolvedPlaceholder`] if `text` references a
/// placeholder that is not present in `evidence`.
pub fn substitute_placeholders(text: &str, evidence: &RewooEvidence) -> Result<String, RewooError> {
    let tokens = scan_placeholder_tokens(text);
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (start, end, var) in tokens {
        output.push_str(&text[cursor..start]);
        let value = evidence
            .get(var)
            .ok_or(RewooError::UnresolvedPlaceholder(var))?;
        output.push_str(value);
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    Ok(output)
}

// ── Arithmetic expression evaluation ──────────────────────────────────────────

/// A single lexical token of an arithmetic expression.
#[derive(Debug, Clone, PartialEq)]
enum ExprToken {
    /// A numeric literal.
    Num(f64),
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `(`
    LParen,
    /// `)`
    RParen,
}

/// Tokenizes an arithmetic expression, skipping ASCII whitespace.
fn tokenize_expr(expr: &str) -> Result<Vec<ExprToken>, String> {
    let chars: Vec<char> = expr.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            '+' => {
                tokens.push(ExprToken::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(ExprToken::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(ExprToken::Star);
                i += 1;
            }
            '/' => {
                tokens.push(ExprToken::Slash);
                i += 1;
            }
            '(' => {
                tokens.push(ExprToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(ExprToken::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                let mut seen_dot = c == '.';
                i += 1;
                while i < chars.len() {
                    let cc = chars[i];
                    if cc.is_ascii_digit() {
                        i += 1;
                    } else if cc == '.' && !seen_dot {
                        seen_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                let value: f64 = text
                    .parse()
                    .map_err(|_| format!("invalid number literal {text:?}"))?;
                tokens.push(ExprToken::Num(value));
            }
            other => return Err(format!("unexpected character {other:?}")),
        }
    }
    Ok(tokens)
}

/// Recursive-descent parser/evaluator over a fixed token slice.
///
/// Grammar (highest to lowest precedence):
/// `primary := NUM | '(' expr ')'`;
/// `unary := ('+' | '-')? primary`;
/// `term := unary (('*' | '/') unary)*`;
/// `expr := term (('+' | '-') term)*`.
struct ExprParser<'a> {
    tokens: &'a [ExprToken],
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn new(tokens: &'a [ExprToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&ExprToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&ExprToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn parse_expr(&mut self) -> Result<f64, String> {
        let mut value = self.parse_term()?;
        loop {
            match self.peek() {
                Some(ExprToken::Plus) => {
                    self.pos += 1;
                    value += self.parse_term()?;
                }
                Some(ExprToken::Minus) => {
                    self.pos += 1;
                    value -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut value = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(ExprToken::Star) => {
                    self.pos += 1;
                    value *= self.parse_unary()?;
                }
                Some(ExprToken::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    if rhs == 0.0 {
                        return Err("division by zero".to_string());
                    }
                    value /= rhs;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_unary(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some(ExprToken::Minus) => {
                self.pos += 1;
                Ok(-self.parse_unary()?)
            }
            Some(ExprToken::Plus) => {
                self.pos += 1;
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<f64, String> {
        match self.advance() {
            Some(ExprToken::Num(n)) => Ok(*n),
            Some(ExprToken::LParen) => {
                let value = self.parse_expr()?;
                match self.advance() {
                    Some(ExprToken::RParen) => Ok(value),
                    _ => Err("expected a closing ')'".to_string()),
                }
            }
            other => Err(format!("expected a number or '(', found {other:?}")),
        }
    }
}

/// Formats a finite `f64` compactly: as a plain integer when it is
/// (numerically) whole, otherwise as a fixed-precision decimal with trailing
/// zeros trimmed.
#[allow(clippy::cast_possible_truncation)] // guarded by the `< 1e15` + near-zero-fract check
fn format_number(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        let fixed = format!("{value:.6}");
        let trimmed = fixed.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    }
}

/// Evaluates an arithmetic `expression` (already placeholder-substituted) and
/// returns its result formatted as evidence text.
///
/// # Errors
///
/// Returns [`RewooError::ComputeFailed`] if `expression` is empty, contains
/// an unexpected character, is malformed (unbalanced parentheses, trailing
/// tokens, ...), divides by zero, or evaluates to a non-finite value.
pub fn evaluate_expression(expression: &str) -> Result<String, RewooError> {
    let compute_failed = |reason: String| RewooError::ComputeFailed {
        expression: expression.to_string(),
        reason,
    };

    let trimmed = expression.trim();
    if trimmed.is_empty() {
        return Err(compute_failed("empty expression".to_string()));
    }

    let tokens = tokenize_expr(trimmed).map_err(compute_failed)?;
    if tokens.is_empty() {
        return Err(compute_failed("empty expression".to_string()));
    }

    let mut parser = ExprParser::new(&tokens);
    let value = parser.parse_expr().map_err(compute_failed)?;
    if parser.pos != tokens.len() {
        return Err(compute_failed(
            "trailing tokens after a complete expression".to_string(),
        ));
    }
    if !value.is_finite() {
        return Err(compute_failed(
            "expression did not evaluate to a finite number".to_string(),
        ));
    }

    Ok(format_number(value))
}

// ── RewooPlanSource ────────────────────────────────────────────────────────────

/// Produces a complete [`RewooPlan`] for a query in a single call.
///
/// This is where `ReWOO`'s efficiency property lives: [`RewooPlanner::plan`]
/// invokes a `RewooPlanSource` **exactly once** per query, no matter how many
/// steps the resulting plan has — contrast with an interleaved ReAct-style
/// loop (`crate::agentic`), which needs one model call *per step* because
/// planning and acting alternate turn by turn. A `RewooPlanSource` never sees
/// any actual retrieval results: it only ever sees the query text.
pub trait RewooPlanSource {
    /// Produces the ordered plan steps for `query`.
    fn generate_plan(&self, query: &str) -> RewooPlan;
}

/// Deterministic [`RewooPlanSource`] for tests and examples.
///
/// Returns the plan of the first `(substring, plan)` mapping whose
/// `substring` is contained in the query; if none match, `default_plan` is
/// returned (an empty-step plan echoing the query if none was configured).
#[derive(Debug, Clone, Default)]
pub struct MockRewooPlanSource {
    /// Ordered `(query substring, plan)` mappings.
    pub mappings: Vec<(String, RewooPlan)>,
    /// Plan returned when no mapping matches.
    pub default_plan: Option<RewooPlan>,
}

impl MockRewooPlanSource {
    /// Creates a mock plan source from `(query substring, plan)` mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, RewooPlan)>) -> Self {
        Self {
            mappings,
            default_plan: None,
        }
    }

    /// Creates a mock plan source that always returns `plan`, regardless of
    /// the query.
    #[must_use]
    pub fn single(plan: RewooPlan) -> Self {
        Self {
            mappings: Vec::new(),
            default_plan: Some(plan),
        }
    }

    /// Sets the plan returned when no mapping matches.
    #[must_use]
    pub fn with_default(mut self, plan: RewooPlan) -> Self {
        self.default_plan = Some(plan);
        self
    }
}

impl RewooPlanSource for MockRewooPlanSource {
    fn generate_plan(&self, query: &str) -> RewooPlan {
        for (needle, plan) in &self.mappings {
            if query.contains(needle.as_str()) {
                return plan.clone();
            }
        }
        self.default_plan
            .clone()
            .unwrap_or_else(|| RewooPlan::new(query, Vec::new()))
    }
}

// ── RewooRetriever ─────────────────────────────────────────────────────────────

/// Executes a resolved [`RewooAction::Search`] query against a knowledge
/// source, returning the retrieved evidence text.
///
/// Implementations are **pure sync**, mirroring the caller-supplies-executor
/// convention used across this crate. [`RewooWorker::run`] calls `retrieve`
/// exactly once per [`RewooAction::Search`] step — a step's evidence is never
/// re-fetched and the plan is never re-consulted mid-execution.
pub trait RewooRetriever {
    /// Retrieves the evidence text for `query` (with all earlier placeholders
    /// already substituted in).
    fn retrieve(&self, query: &str) -> String;
}

/// Deterministic [`RewooRetriever`] for tests and examples.
///
/// Returns the evidence of the first `(substring, evidence)` mapping whose
/// `substring` is contained in the query; if none match, the query itself is
/// echoed back, so retrieval is always observable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MockRewooRetriever {
    /// Ordered `(query substring, evidence text)` mappings.
    pub mappings: Vec<(String, String)>,
}

impl MockRewooRetriever {
    /// Creates a mock retriever from `(query substring, evidence text)`
    /// mappings.
    #[must_use]
    pub fn new(mappings: Vec<(String, String)>) -> Self {
        Self { mappings }
    }

    /// Creates a mock retriever that echoes every query as its own evidence.
    #[must_use]
    pub fn echo() -> Self {
        Self {
            mappings: Vec::new(),
        }
    }
}

impl RewooRetriever for MockRewooRetriever {
    fn retrieve(&self, query: &str) -> String {
        for (needle, evidence) in &self.mappings {
            if query.contains(needle.as_str()) {
                return evidence.clone();
            }
        }
        query.to_string()
    }
}

// ── RewooGenerator ─────────────────────────────────────────────────────────────

/// Synthesizes the final answer from the original query and the plan's fully
/// placeholder-substituted reasoning text.
pub trait RewooGenerator {
    /// Produces the final answer for `query`, given `substituted_plan` — the
    /// plan's step reasoning text with every `#E<n>` placeholder replaced by
    /// its resolved evidence.
    fn generate(&self, query: &str, substituted_plan: &str) -> String;
}

/// Deterministic [`RewooGenerator`] for tests and examples.
///
/// Fills the `{query}` and `{plan}` placeholders of a fixed template string.
/// The default template is `"{plan}"`: the substituted plan text is returned
/// verbatim as the answer, which is sufficient for tests that only need to
/// assert on the resolved evidence without depending on a real LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockRewooGenerator {
    /// The template string, with `{query}` and `{plan}` placeholders.
    pub template: String,
}

impl MockRewooGenerator {
    /// Creates a mock generator driven by `template`.
    #[must_use]
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }
}

impl Default for MockRewooGenerator {
    fn default() -> Self {
        Self::new("{plan}")
    }
}

impl RewooGenerator for MockRewooGenerator {
    fn generate(&self, query: &str, substituted_plan: &str) -> String {
        self.template
            .replace("{query}", query)
            .replace("{plan}", substituted_plan)
    }
}

// ── RewooPlanner ──────────────────────────────────────────────────────────────

/// Produces a complete plan for a query in a single upfront pass.
///
/// `RewooPlanner` is deliberately thin: all of the actual plan-generation
/// intelligence lives behind the caller-supplied [`RewooPlanSource`]. What the
/// planner *owns* is the structural contract the rest of `ReWOO` depends on —
/// the plan source is consulted **exactly once** per [`RewooPlanner::plan`]
/// call, and the resulting plan is validated (size bound, no duplicate
/// placeholder definitions, and — when
/// [`RewooConfig::validate_plan_structure`] is set — no step referencing a
/// placeholder that isn't defined by a strictly earlier step) before it is
/// handed off to [`RewooWorker`].
#[derive(Debug, Clone, Default)]
pub struct RewooPlanner {
    /// Configuration governing plan-size and structural validation.
    pub config: RewooConfig,
}

impl RewooPlanner {
    /// Creates a new planner with the given configuration.
    #[must_use]
    pub fn new(config: RewooConfig) -> Self {
        Self { config }
    }

    /// Generates and validates a plan for `query` via `source`.
    ///
    /// `source` is invoked **exactly once**, regardless of how many steps the
    /// produced plan has — this is `ReWOO`'s core efficiency property versus an
    /// interleaved reasoning loop.
    ///
    /// # Errors
    ///
    /// - [`RewooError::EmptyQuery`] if `query` is empty after trimming.
    /// - [`RewooError::PlanTooLong`] if the produced plan exceeds
    ///   [`RewooConfig::max_steps`].
    /// - [`RewooError::DuplicatePlaceholder`] if two steps define the same
    ///   evidence variable.
    /// - [`RewooError::ForwardReference`] (only when
    ///   [`RewooConfig::validate_plan_structure`] is `true`) if a step's
    ///   action references a placeholder not defined by a strictly earlier
    ///   step.
    pub fn plan<S>(&self, query: &str, source: &S) -> Result<RewooPlan, RewooError>
    where
        S: RewooPlanSource + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(RewooError::EmptyQuery);
        }
        let plan = source.generate_plan(query);
        self.validate(&plan)?;
        Ok(plan)
    }

    /// Validates the structural shape of `plan` against `self.config`.
    fn validate(&self, plan: &RewooPlan) -> Result<(), RewooError> {
        validate_plan_shape(plan, &self.config)
    }
}

/// Shared structural validation used by both [`RewooPlanner::plan`] and
/// [`RewooWorker::run`]: enforces the step-count bound, rejects duplicate
/// placeholder definitions, and (when `config.validate_plan_structure` is
/// set) rejects any step whose action references a placeholder that is not
/// defined by a strictly earlier step.
fn validate_plan_shape(plan: &RewooPlan, config: &RewooConfig) -> Result<(), RewooError> {
    if plan.len() > config.max_steps {
        return Err(RewooError::PlanTooLong {
            len: plan.len(),
            max: config.max_steps,
        });
    }

    let mut defined: BTreeSet<PlaceholderVar> = BTreeSet::new();
    for (idx, step) in plan.steps.iter().enumerate() {
        if config.validate_plan_structure {
            for var in referenced_placeholders(step.action.raw_text()) {
                if !defined.contains(&var) {
                    return Err(RewooError::ForwardReference { step: idx, var });
                }
            }
        }
        if !defined.insert(step.evidence_var) {
            return Err(RewooError::DuplicatePlaceholder(step.evidence_var));
        }
    }
    Ok(())
}

// ── RewooWorker ───────────────────────────────────────────────────────────────

/// Walks a [`RewooPlan`] in order, resolving each step's placeholder by
/// executing its action.
///
/// This is where `ReWOO`'s decoupling is enforced at runtime: the worker never
/// consults a [`RewooPlanSource`] and never alters the plan — it purely
/// resolves the placeholders the planner already committed to.
#[derive(Debug, Clone, Default)]
pub struct RewooWorker {
    /// Configuration governing plan-size and structural validation.
    pub config: RewooConfig,
}

impl RewooWorker {
    /// Creates a new worker with the given configuration.
    #[must_use]
    pub fn new(config: RewooConfig) -> Self {
        Self { config }
    }

    /// Resolves every step of `plan`, in order, and returns the full resolved
    /// evidence map.
    ///
    /// For each step, `#E<n>` placeholders in the action's text are first
    /// substituted using only the evidence resolved by strictly earlier steps
    /// (evidence from the current or any later step does not exist yet); the
    /// resulting text is then executed: [`RewooAction::Search`] calls
    /// `retriever` exactly once, [`RewooAction::Compute`] is evaluated
    /// arithmetically via [`evaluate_expression`]. Exactly one call is made to
    /// `retriever` per [`RewooAction::Search`] step and the plan itself is
    /// never re-consulted or regenerated mid-run — `ReWOO`'s efficiency
    /// property versus an interleaved ReAct-style loop.
    ///
    /// # Errors
    ///
    /// - [`RewooError::PlanTooLong`] if `plan` exceeds
    ///   [`RewooConfig::max_steps`].
    /// - [`RewooError::DuplicatePlaceholder`] if two steps define the same
    ///   evidence variable.
    /// - [`RewooError::ForwardReference`] (only when
    ///   [`RewooConfig::validate_plan_structure`] is `true`) if a step's
    ///   action references a placeholder not defined by a strictly earlier
    ///   step.
    /// - [`RewooError::UnresolvedPlaceholder`] if a step's action references
    ///   a placeholder that has not been resolved yet — this is the runtime
    ///   backstop that fires even when eager structural validation is
    ///   disabled, or when a plan was hand-built (bypassing
    ///   [`RewooPlanner`]) with a step that references a placeholder no
    ///   earlier step defines.
    /// - [`RewooError::ComputeFailed`] if a [`RewooAction::Compute`]
    ///   expression is malformed.
    pub fn run<R>(&self, plan: &RewooPlan, retriever: &R) -> Result<RewooEvidence, RewooError>
    where
        R: RewooRetriever + ?Sized,
    {
        validate_plan_shape(plan, &self.config)?;

        let mut evidence = RewooEvidence::new();
        for step in &plan.steps {
            if evidence.get(step.evidence_var).is_some() {
                return Err(RewooError::DuplicatePlaceholder(step.evidence_var));
            }

            let resolved_text = substitute_placeholders(step.action.raw_text(), &evidence)?;
            let evidence_text = match &step.action {
                RewooAction::Search(_) => retriever.retrieve(&resolved_text),
                RewooAction::Compute(_) => evaluate_expression(&resolved_text)?,
            };

            evidence.insert(step.evidence_var, evidence_text);
        }

        Ok(evidence)
    }
}

// ── RewooSolver ───────────────────────────────────────────────────────────────

/// Substitutes resolved evidence into a plan's reasoning text and synthesizes
/// the final answer.
#[derive(Debug, Clone, Default)]
pub struct RewooSolver {
    /// Configuration (currently unused beyond consistency with the other
    /// stages; reserved for future solver-level bounds).
    pub config: RewooConfig,
}

impl RewooSolver {
    /// Creates a new solver with the given configuration.
    #[must_use]
    pub fn new(config: RewooConfig) -> Self {
        Self { config }
    }

    /// Substitutes every placeholder in `plan`'s step reasoning texts with its
    /// resolved evidence from `evidence`, then asks `generator` to synthesize
    /// the final answer to `query` from the fully substituted plan text.
    ///
    /// Unlike [`RewooWorker::run`], which only allows a step's *action* to see
    /// evidence resolved by strictly earlier steps (the plan hasn't finished
    /// executing yet at that point), the solver runs strictly after the
    /// worker has resolved the whole plan, so every placeholder defined
    /// anywhere in `plan` is available for the *reasoning* text regardless of
    /// step order.
    ///
    /// # Errors
    ///
    /// - [`RewooError::EmptyQuery`] if `query` is empty after trimming.
    /// - [`RewooError::UnresolvedPlaceholder`] if a step's reasoning text
    ///   references a placeholder that is not present in `evidence`, or if a
    ///   step's own `evidence_var` is not present in `evidence` (i.e.
    ///   `evidence` does not fully match `plan`).
    pub fn solve<G>(
        &self,
        query: &str,
        plan: &RewooPlan,
        evidence: &RewooEvidence,
        generator: &G,
    ) -> Result<String, RewooError>
    where
        G: RewooGenerator + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(RewooError::EmptyQuery);
        }

        let mut substituted_plan = String::new();
        for step in &plan.steps {
            let text = substitute_placeholders(&step.reasoning, evidence)?;
            // Surface this step's own resolved evidence explicitly, even
            // when the reasoning text never spells out its own placeholder
            // token verbatim — otherwise a generator reading only the
            // substituted reasoning could be left with no way to see what
            // e.g. `#E1` actually resolved to.
            let value = evidence
                .get(step.evidence_var)
                .ok_or(RewooError::UnresolvedPlaceholder(step.evidence_var))?;
            if !substituted_plan.is_empty() {
                substituted_plan.push('\n');
            }
            let _ = write!(
                substituted_plan,
                "{}: {text} (= {value})",
                step.evidence_var
            );
        }

        Ok(generator.generate(query, &substituted_plan))
    }
}

// ── RewooPipeline ─────────────────────────────────────────────────────────────

/// Runs the full plan → resolve → solve pipeline for a query in one call.
///
/// This exists purely for ergonomics: it does nothing [`RewooPlanner`],
/// [`RewooWorker`], and [`RewooSolver`] don't already do individually.
/// Callers that want to inspect (or even let a human review) the plan between
/// stages — itself a capability `ReWOO`'s decoupled, single-upfront-pass design
/// uniquely offers over an interleaved loop — should call the three stages
/// directly instead.
///
/// # Example
///
/// ```
/// use oxirag::rewoo::{
///     MockRewooGenerator, MockRewooPlanSource, MockRewooRetriever, PlaceholderVar, RewooAction,
///     RewooConfig, RewooPipeline, RewooPlan, RewooStep,
/// };
///
/// let plan = RewooPlan::new(
///     "capital of France?",
///     vec![RewooStep::new(
///         "Look up the capital of France.",
///         RewooAction::Search("capital of France".to_string()),
///         PlaceholderVar::new(1),
///     )],
/// );
/// let plan_source = MockRewooPlanSource::single(plan);
/// let retriever = MockRewooRetriever::new(vec![(
///     "capital of France".to_string(),
///     "Paris".to_string(),
/// )]);
/// let generator = MockRewooGenerator::default();
///
/// let pipeline = RewooPipeline::new(RewooConfig::default());
/// let outcome = pipeline
///     .run("capital of France?", &plan_source, &retriever, &generator)
///     .unwrap();
///
/// assert_eq!(outcome.evidence.get(PlaceholderVar::new(1)), Some("Paris"));
/// assert!(outcome.answer.contains("Paris"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct RewooPipeline {
    /// Configuration shared by all three stages.
    pub config: RewooConfig,
}

impl RewooPipeline {
    /// Creates a new pipeline with the given configuration.
    #[must_use]
    pub fn new(config: RewooConfig) -> Self {
        Self { config }
    }

    /// Runs planning, evidence resolution, and answer synthesis for `query`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`RewooPlanner::plan`], [`RewooWorker::run`],
    /// or [`RewooSolver::solve`].
    pub fn run<S, R, G>(
        &self,
        query: &str,
        plan_source: &S,
        retriever: &R,
        generator: &G,
    ) -> Result<RewooOutcome, RewooError>
    where
        S: RewooPlanSource + ?Sized,
        R: RewooRetriever + ?Sized,
        G: RewooGenerator + ?Sized,
    {
        let planner = RewooPlanner::new(self.config.clone());
        let plan = planner.plan(query, plan_source)?;

        let worker = RewooWorker::new(self.config.clone());
        let evidence = worker.run(&plan, retriever)?;

        let solver = RewooSolver::new(self.config.clone());
        let answer = solver.solve(query, &plan, &evidence, generator)?;

        Ok(RewooOutcome {
            query: query.trim().to_string(),
            plan,
            evidence,
            answer,
        })
    }
}
