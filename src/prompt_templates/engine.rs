//! Template engine (stub — agents fill in).
use super::types::{PromptTemplate, PromptTemplateError, RenderContext};

/// AST node for template bodies.
#[derive(Debug, Clone)]
pub(crate) enum TemplateNode {
    /// Literal text.
    Literal(String),
    /// Variable substitution.
    Var(String),
    /// Conditional block.
    If {
        flag: String,
        body: Vec<TemplateNode>,
        else_body: Vec<TemplateNode>,
    },
    /// Unless block.
    Unless {
        flag: String,
        body: Vec<TemplateNode>,
    },
}

/// Zero-sized, stateless template engine.
#[derive(Debug, Default)]
pub struct TemplateEngine;

impl TemplateEngine {
    /// Create a new engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Render a template with the given context.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError::MissingVariable`] if a required variable is absent,
    /// or propagates parse/render errors from the template body.
    pub fn render(
        tmpl: &PromptTemplate,
        ctx: &RenderContext,
    ) -> Result<String, PromptTemplateError> {
        for v in &tmpl.required_vars {
            if ctx.get_var(v).is_none() {
                return Err(PromptTemplateError::MissingVariable(v.clone()));
            }
        }
        Self::render_str(&tmpl.body, ctx)
    }

    /// Render a raw body string.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError`] on parse failure or missing variables.
    pub fn render_str(body: &str, ctx: &RenderContext) -> Result<String, PromptTemplateError> {
        let nodes = Self::parse(body)?;
        Self::render_nodes(&nodes, ctx)
    }

    /// Parse a body into AST nodes.
    pub(crate) fn parse(body: &str) -> Result<Vec<TemplateNode>, PromptTemplateError> {
        let tokens = tokenize(body)?;
        let mut pos = 0usize;
        parse_nodes(&tokens, &mut pos, None)
    }

    /// Return variable names referenced in the body.
    ///
    /// # Errors
    /// Returns [`PromptTemplateError`] if the body cannot be parsed.
    pub fn referenced_vars(body: &str) -> Result<Vec<String>, PromptTemplateError> {
        let nodes = Self::parse(body)?;
        let mut vars = Vec::new();
        collect_vars(&nodes, &mut vars);
        vars.dedup();
        Ok(vars)
    }

    fn render_nodes(
        nodes: &[TemplateNode],
        ctx: &RenderContext,
    ) -> Result<String, PromptTemplateError> {
        let mut out = String::new();
        for node in nodes {
            match node {
                TemplateNode::Literal(s) => out.push_str(s),
                TemplateNode::Var(name) => {
                    let val = ctx
                        .get_var(name)
                        .ok_or_else(|| PromptTemplateError::MissingVariable(name.clone()))?;
                    out.push_str(val);
                }
                TemplateNode::If {
                    flag,
                    body,
                    else_body,
                } => {
                    if ctx.get_flag(flag) {
                        out.push_str(&Self::render_nodes(body, ctx)?);
                    } else {
                        out.push_str(&Self::render_nodes(else_body, ctx)?);
                    }
                }
                TemplateNode::Unless { flag, body } => {
                    if !ctx.get_flag(flag) {
                        out.push_str(&Self::render_nodes(body, ctx)?);
                    }
                }
            }
        }
        Ok(out)
    }
}

fn collect_vars(nodes: &[TemplateNode], out: &mut Vec<String>) {
    for node in nodes {
        match node {
            TemplateNode::Var(n) => out.push(n.clone()),
            TemplateNode::If {
                body, else_body, ..
            } => {
                collect_vars(body, out);
                collect_vars(else_body, out);
            }
            TemplateNode::Unless { body, .. } => collect_vars(body, out),
            TemplateNode::Literal(_) => {}
        }
    }
}

// ─── Tokenizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Token {
    Text(String),
    OpenIf(String),
    OpenUnless(String),
    Else,
    CloseIf,
    CloseUnless,
    Var(String),
}

fn tokenize(body: &str) -> Result<Vec<Token>, PromptTemplateError> {
    const DEPTH_CAP: usize = 64;
    let mut tokens = Vec::new();
    let mut chars = body.chars().peekable();
    let mut lit = String::new();

    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'{') {
            chars.next(); // consume second {
            if !lit.is_empty() {
                tokens.push(Token::Text(std::mem::take(&mut lit)));
            }
            // collect inner until }}
            let mut inner = String::new();
            loop {
                match chars.next() {
                    None => {
                        return Err(PromptTemplateError::ParseError(
                            "unterminated tag {{".to_string(),
                        ));
                    }
                    Some('}') if chars.peek() == Some(&'}') => {
                        chars.next();
                        break;
                    }
                    Some(ch) => inner.push(ch),
                }
            }
            let inner = inner.trim().to_string();
            if inner.is_empty() {
                return Err(PromptTemplateError::ParseError(
                    "empty tag {{}}".to_string(),
                ));
            }
            let tok = if let Some(flag) = inner.strip_prefix("#if ") {
                Token::OpenIf(flag.trim().to_string())
            } else if let Some(flag) = inner.strip_prefix("#unless ") {
                Token::OpenUnless(flag.trim().to_string())
            } else if inner == "else" {
                Token::Else
            } else if inner == "/if" {
                Token::CloseIf
            } else if inner == "/unless" {
                Token::CloseUnless
            } else if inner
                .chars()
                .all(|x| x.is_alphanumeric() || x == '_' || x == '-')
            {
                Token::Var(inner)
            } else {
                return Err(PromptTemplateError::ParseError(format!(
                    "unrecognised tag: {{{{{inner}}}}}"
                )));
            };
            tokens.push(tok);
            // guard against pathological depth not checked here; recursion cap handles it
            let _ = DEPTH_CAP;
        } else {
            lit.push(c);
        }
    }
    if !lit.is_empty() {
        tokens.push(Token::Text(lit));
    }
    Ok(tokens)
}

// ─── Recursive descent builder ────────────────────────────────────────────────

const MAX_DEPTH: usize = 64;

fn parse_nodes(
    tokens: &[Token],
    pos: &mut usize,
    stop: Option<&str>,
) -> Result<Vec<TemplateNode>, PromptTemplateError> {
    parse_nodes_depth(tokens, pos, stop, 0)
}

fn parse_nodes_depth(
    tokens: &[Token],
    pos: &mut usize,
    stop: Option<&str>,
    depth: usize,
) -> Result<Vec<TemplateNode>, PromptTemplateError> {
    if depth > MAX_DEPTH {
        return Err(PromptTemplateError::ParseError(
            "template nesting depth exceeded 64".to_string(),
        ));
    }
    let mut nodes = Vec::new();
    loop {
        match tokens.get(*pos) {
            None => {
                if let Some(s) = stop {
                    return Err(PromptTemplateError::UnclosedBlock(s.to_string()));
                }
                return Ok(nodes);
            }
            Some(Token::Text(s)) => {
                nodes.push(TemplateNode::Literal(s.clone()));
                *pos += 1;
            }
            Some(Token::Var(n)) => {
                nodes.push(TemplateNode::Var(n.clone()));
                *pos += 1;
            }
            Some(Token::OpenIf(flag)) => {
                let flag = flag.clone();
                *pos += 1;
                let body = parse_nodes_depth(tokens, pos, Some("if"), depth + 1)?;
                // check for else
                let else_body = if matches!(tokens.get(*pos), Some(Token::Else)) {
                    *pos += 1;
                    let eb = parse_nodes_depth(tokens, pos, Some("if"), depth + 1)?;
                    // consume closing /if (already consumed by recursive call stopping on Else... no)
                    // actually parse_nodes_depth returns when it hits Else or CloseIf
                    eb
                } else {
                    Vec::new()
                };
                // consume CloseIf
                match tokens.get(*pos) {
                    Some(Token::CloseIf) => {
                        *pos += 1;
                    }
                    _ => return Err(PromptTemplateError::UnclosedBlock("if".to_string())),
                }
                nodes.push(TemplateNode::If {
                    flag,
                    body,
                    else_body,
                });
            }
            Some(Token::OpenUnless(flag)) => {
                let flag = flag.clone();
                *pos += 1;
                let body = parse_nodes_depth(tokens, pos, Some("unless"), depth + 1)?;
                match tokens.get(*pos) {
                    Some(Token::CloseUnless) => {
                        *pos += 1;
                    }
                    _ => return Err(PromptTemplateError::UnclosedBlock("unless".to_string())),
                }
                nodes.push(TemplateNode::Unless { flag, body });
            }
            Some(Token::CloseIf) => match stop {
                Some("if") => return Ok(nodes),
                _ => return Err(PromptTemplateError::UnexpectedClosingTag("/if".to_string())),
            },
            Some(Token::CloseUnless) => match stop {
                Some("unless") => return Ok(nodes),
                _ => {
                    return Err(PromptTemplateError::UnexpectedClosingTag(
                        "/unless".to_string(),
                    ));
                }
            },
            Some(Token::Else) => {
                match stop {
                    Some("if") => return Ok(nodes), // caller handles Else
                    _ => {
                        return Err(PromptTemplateError::UnexpectedClosingTag(
                            "else".to_string(),
                        ));
                    }
                }
            }
        }
    }
}
