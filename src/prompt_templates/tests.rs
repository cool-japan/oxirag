//! Tests for the `prompt_templates` module.
use super::engine::TemplateEngine;
use super::registry::{PromptRegistry, TEMPLATE_ANSWER_SYNTHESIS, builtin_templates};
use super::types::{PromptTemplate, PromptTemplateError, RenderContext, TemplateId};

fn ctx() -> RenderContext {
    RenderContext::new()
        .with_var("name", "World")
        .with_var("lang", "Rust")
        .with_flag("show", true)
        .with_flag("hide", false)
}

// ── TemplateId ──────────────────────────────────────────────────────────────

#[test]
fn test_template_id_display() {
    let id = TemplateId::new("my-template");
    assert_eq!(id.to_string(), "my-template");
    assert_eq!(id.as_str(), "my-template");
}

#[test]
fn test_template_id_from_str() {
    let id: TemplateId = "foo".into();
    assert_eq!(id.as_str(), "foo");
}

#[test]
fn test_template_id_from_string() {
    let id: TemplateId = String::from("bar").into();
    assert_eq!(id.as_str(), "bar");
}

// ── PromptTemplate builders ─────────────────────────────────────────────────

#[test]
fn test_prompt_template_new() {
    let t = PromptTemplate::new("t", 1, "hello {{name}}");
    assert_eq!(t.id.as_str(), "t");
    assert_eq!(t.version, 1);
    assert!(t.required_vars.is_empty());
}

#[test]
fn test_prompt_template_with_required_var() {
    let t = PromptTemplate::new("t", 1, "body")
        .with_required_var("x")
        .with_required_var("y");
    assert_eq!(t.required_vars, vec!["x", "y"]);
}

#[test]
fn test_prompt_template_with_required_vars() {
    let t = PromptTemplate::new("t", 1, "body").with_required_vars(vec!["a".into(), "b".into()]);
    assert_eq!(t.required_vars, vec!["a", "b"]);
}

#[test]
fn test_prompt_template_with_description() {
    let t = PromptTemplate::new("t", 1, "body").with_description("desc");
    assert_eq!(t.description, "desc");
}

// ── RenderContext ────────────────────────────────────────────────────────────

#[test]
fn test_render_context_var() {
    let c = RenderContext::new().with_var("key", "val");
    assert_eq!(c.get_var("key"), Some("val"));
    assert_eq!(c.get_var("missing"), None);
}

#[test]
fn test_render_context_flag_default_false() {
    let c = RenderContext::new().with_flag("on", true);
    assert!(c.get_flag("on"));
    assert!(!c.get_flag("absent"));
}

// ── Engine: variable substitution ───────────────────────────────────────────

#[test]
fn test_render_str_single_var() {
    let out = TemplateEngine::render_str("Hello {{name}}!", &ctx()).unwrap();
    assert_eq!(out, "Hello World!");
}

#[test]
fn test_render_str_multiple_vars() {
    let out = TemplateEngine::render_str("{{name}} loves {{lang}}", &ctx()).unwrap();
    assert_eq!(out, "World loves Rust");
}

#[test]
fn test_render_str_missing_var_error() {
    let err = TemplateEngine::render_str("{{missing}}", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::MissingVariable(v) if v == "missing"));
}

#[test]
fn test_render_str_repeated_var() {
    let out = TemplateEngine::render_str("{{name}} {{name}}", &ctx()).unwrap();
    assert_eq!(out, "World World");
}

#[test]
fn test_render_str_empty_body() {
    let out = TemplateEngine::render_str("", &ctx()).unwrap();
    assert_eq!(out, "");
}

#[test]
fn test_render_str_adjacent_vars() {
    let out = TemplateEngine::render_str("{{name}}{{lang}}", &ctx()).unwrap();
    assert_eq!(out, "WorldRust");
}

#[test]
fn test_render_str_no_vars() {
    let out = TemplateEngine::render_str("plain text", &ctx()).unwrap();
    assert_eq!(out, "plain text");
}

// ── Engine: conditionals ─────────────────────────────────────────────────────

#[test]
fn test_render_if_true() {
    let out = TemplateEngine::render_str("{{#if show}}YES{{/if}}", &ctx()).unwrap();
    assert_eq!(out, "YES");
}

#[test]
fn test_render_if_false() {
    let out = TemplateEngine::render_str("{{#if hide}}YES{{/if}}", &ctx()).unwrap();
    assert_eq!(out, "");
}

#[test]
fn test_render_unless_true() {
    let out = TemplateEngine::render_str("{{#unless show}}NOPE{{/unless}}", &ctx()).unwrap();
    assert_eq!(out, "");
}

#[test]
fn test_render_unless_false() {
    let out = TemplateEngine::render_str("{{#unless hide}}SHOWN{{/unless}}", &ctx()).unwrap();
    assert_eq!(out, "SHOWN");
}

#[test]
fn test_render_if_else_true() {
    let out = TemplateEngine::render_str("{{#if show}}A{{else}}B{{/if}}", &ctx()).unwrap();
    assert_eq!(out, "A");
}

#[test]
fn test_render_if_else_false() {
    let out = TemplateEngine::render_str("{{#if hide}}A{{else}}B{{/if}}", &ctx()).unwrap();
    assert_eq!(out, "B");
}

#[test]
fn test_render_nested_if() {
    let c = ctx();
    let out = TemplateEngine::render_str("{{#if show}}{{#if show}}DEEP{{/if}}{{/if}}", &c).unwrap();
    assert_eq!(out, "DEEP");
}

#[test]
fn test_render_if_absent_flag_is_false() {
    let c = RenderContext::new();
    let out = TemplateEngine::render_str("{{#if ghost}}YES{{else}}NO{{/if}}", &c).unwrap();
    assert_eq!(out, "NO");
}

#[test]
fn test_render_nested_if_in_unless() {
    let c = ctx();
    let out =
        TemplateEngine::render_str("{{#unless hide}}{{#if show}}OK{{/if}}{{/unless}}", &c).unwrap();
    assert_eq!(out, "OK");
}

#[test]
fn test_render_var_inside_if() {
    let c = ctx();
    let out = TemplateEngine::render_str("{{#if show}}Hi {{name}}{{/if}}", &c).unwrap();
    assert_eq!(out, "Hi World");
}

// ── Engine: parse errors ─────────────────────────────────────────────────────

#[test]
fn test_parse_error_unclosed_if() {
    let err = TemplateEngine::render_str("{{#if show}}no close", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::UnclosedBlock(_)));
}

#[test]
fn test_parse_error_unclosed_unless() {
    let err = TemplateEngine::render_str("{{#unless hide}}no close", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::UnclosedBlock(_)));
}

#[test]
fn test_parse_error_dangling_close_if() {
    let err = TemplateEngine::render_str("{{/if}}", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::UnexpectedClosingTag(_)));
}

#[test]
fn test_parse_error_dangling_close_unless() {
    let err = TemplateEngine::render_str("{{/unless}}", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::UnexpectedClosingTag(_)));
}

#[test]
fn test_parse_error_unterminated_tag() {
    let err = TemplateEngine::render_str("hello {{ world", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::ParseError(_)));
}

#[test]
fn test_parse_error_empty_tag() {
    let err = TemplateEngine::render_str("{{}}", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::ParseError(_)));
}

#[test]
fn test_parse_error_dangling_else() {
    let err = TemplateEngine::render_str("{{else}}", &ctx()).unwrap_err();
    assert!(matches!(err, PromptTemplateError::UnexpectedClosingTag(_)));
}

// ── Engine: referenced_vars ──────────────────────────────────────────────────

#[test]
fn test_referenced_vars_simple() {
    let vars = TemplateEngine::referenced_vars("Hello {{name}} from {{lang}}").unwrap();
    assert!(vars.contains(&"name".to_string()));
    assert!(vars.contains(&"lang".to_string()));
}

#[test]
fn test_referenced_vars_dedup() {
    let vars = TemplateEngine::referenced_vars("{{a}} {{a}}").unwrap();
    assert_eq!(vars.iter().filter(|v| *v == "a").count(), 1);
}

// ── Registry ─────────────────────────────────────────────────────────────────

#[test]
fn test_registry_new_is_empty() {
    let r = PromptRegistry::new();
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}

#[test]
fn test_registry_register_get_latest() {
    let mut r = PromptRegistry::new();
    let t = PromptTemplate::new("t", 1, "body");
    r.register(t).unwrap();
    assert_eq!(r.get_latest(&TemplateId::new("t")).unwrap().version, 1);
}

#[test]
fn test_registry_multiple_versions_sorted() {
    let mut r = PromptRegistry::new();
    r.register(PromptTemplate::new("t", 3, "v3")).unwrap();
    r.register(PromptTemplate::new("t", 1, "v1")).unwrap();
    r.register(PromptTemplate::new("t", 2, "v2")).unwrap();
    let versions = r.versions(&TemplateId::new("t")).unwrap();
    assert_eq!(versions, vec![1, 2, 3]);
    assert_eq!(r.get_latest(&TemplateId::new("t")).unwrap().body, "v3");
}

#[test]
fn test_registry_get_version_found() {
    let mut r = PromptRegistry::new();
    r.register(PromptTemplate::new("t", 2, "v2")).unwrap();
    r.register(PromptTemplate::new("t", 1, "v1")).unwrap();
    assert_eq!(r.get_version(&TemplateId::new("t"), 1).unwrap().body, "v1");
}

#[test]
fn test_registry_get_version_not_found() {
    let mut r = PromptRegistry::new();
    r.register(PromptTemplate::new("t", 1, "v1")).unwrap();
    let err = r.get_version(&TemplateId::new("t"), 99).unwrap_err();
    assert!(matches!(err, PromptTemplateError::VersionNotFound(99)));
}

#[test]
fn test_registry_template_not_found() {
    let r = PromptRegistry::new();
    let err = r.get_latest(&TemplateId::new("missing")).unwrap_err();
    assert!(matches!(err, PromptTemplateError::TemplateNotFound(_)));
}

#[test]
fn test_registry_empty_id_rejected() {
    let mut r = PromptRegistry::new();
    let err = r.register(PromptTemplate::new("", 1, "body")).unwrap_err();
    assert!(matches!(err, PromptTemplateError::EmptyTemplateId));
}

#[test]
fn test_registry_overwrite_same_version() {
    let mut r = PromptRegistry::new();
    r.register(PromptTemplate::new("t", 1, "old")).unwrap();
    r.register(PromptTemplate::new("t", 1, "new")).unwrap();
    assert_eq!(r.get_latest(&TemplateId::new("t")).unwrap().body, "new");
}

#[test]
fn test_registry_ids() {
    let mut r = PromptRegistry::new();
    r.register(PromptTemplate::new("a", 1, "a")).unwrap();
    r.register(PromptTemplate::new("b", 1, "b")).unwrap();
    assert_eq!(r.len(), 2);
    assert_eq!(r.ids().len(), 2);
}

// ── Built-ins ────────────────────────────────────────────────────────────────

#[test]
fn test_builtins_loads_four() {
    let r = PromptRegistry::with_builtins();
    assert_eq!(r.len(), 4);
}

#[test]
fn test_builtin_answer_synthesis_renders() {
    let r = PromptRegistry::with_builtins();
    let ctx = RenderContext::new()
        .with_var("context", "doc1")
        .with_var("question", "What is X?");
    let out = r
        .render_latest(&TemplateId::new(TEMPLATE_ANSWER_SYNTHESIS), &ctx)
        .unwrap();
    assert!(out.contains("doc1"));
    assert!(out.contains("What is X?"));
}

#[test]
fn test_builtin_required_vars_validated() {
    let r = PromptRegistry::with_builtins();
    let ctx = RenderContext::new(); // missing required vars
    let err = r
        .render_latest(&TemplateId::new(TEMPLATE_ANSWER_SYNTHESIS), &ctx)
        .unwrap_err();
    assert!(matches!(err, PromptTemplateError::MissingVariable(_)));
}

#[test]
fn test_builtin_ids_stable() {
    let builtins = builtin_templates();
    let ids: Vec<&str> = builtins.iter().map(|t| t.id.as_str()).collect();
    assert!(ids.contains(&"answer-synthesis"));
    assert!(ids.contains(&"query-rewrite"));
    assert!(ids.contains(&"doc-grading"));
    assert!(ids.contains(&"citation"));
}

// ── render() requires required_vars ─────────────────────────────────────────

#[test]
fn test_render_missing_required_var() {
    let t = PromptTemplate::new("t", 1, "{{x}}")
        .with_required_var("x")
        .with_required_var("y");
    let ctx = RenderContext::new().with_var("x", "val"); // y missing
    let err = TemplateEngine::render(&t, &ctx).unwrap_err();
    assert!(matches!(err, PromptTemplateError::MissingVariable(v) if v == "y"));
}
