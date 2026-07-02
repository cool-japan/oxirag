//! Comprehensive tests for the `code_retrieval` module (AST/structure-aware
//! code search over a code corpus).

#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::default_trait_access,
    clippy::uninlined_format_args
)]

use crate::code_retrieval::parser::parse_source;
use crate::code_retrieval::{
    CodeRetrievalConfig, CodeRetrievalEngine, CodeRetrievalError, CodeRetrievalHit,
    CodeRetrievalIndex, CodeRetrievalLanguageHint, CodeRetrievalResult, CodeRetrievalSymbol,
    CodeRetrievalUnit, CodeRetrievalUnitKind,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cfg() -> CodeRetrievalConfig {
    CodeRetrievalConfig::default()
}

fn parse(text: &str) -> Vec<CodeRetrievalUnit> {
    parse_source("item", text, &cfg())
}

fn parse_with(text: &str, config: &CodeRetrievalConfig) -> Vec<CodeRetrievalUnit> {
    parse_source("item", text, config)
}

fn names(units: &[CodeRetrievalUnit]) -> Vec<String> {
    units.iter().map(|u| u.name.clone()).collect()
}

fn hit_for<'a>(result: &'a CodeRetrievalResult, source_id: &str) -> &'a CodeRetrievalHit {
    result
        .hits
        .iter()
        .find(|h| h.unit.source_id == source_id)
        .expect("hit for source id must exist")
}

/// Build the adversarial corpus/index used by the structural-vs-text tests.
///
/// `unit_a` shares the query's *comment vocabulary* but none of its identifiers
/// or calls; `unit_b` shares the query's *exact identifiers and call pattern*
/// but almost none of its vocabulary. The query is a code snippet carrying both
/// the shared comment and the shared code.
fn adversarial() -> (String, CodeRetrievalIndex, CodeRetrievalConfig) {
    let comment = "// traverse entire document structure recursively collecting distinct \
                   heading paragraph sentence phrase keyword footnote citation reference \
                   blockquote appendix glossary chapter section margin caption table figure";
    let query = format!(
        "{comment}\nfn walk(root) {{ fetch_child(alpha); render_label(beta); \
         accumulate_metrics(gamma); }}"
    );
    // unit_a: same comment vocabulary, disjoint identifiers/calls.
    let unit_a = format!("{comment}\nfn zeta(theta) {{ mu(iota); nu(kappa); xi(lambda); }}");
    // unit_b: the query's exact identifiers/calls, no shared prose.
    let unit_b =
        "fn walk(root) { fetch_child(alpha); render_label(beta); accumulate_metrics(gamma); }"
            .to_string();

    let config = CodeRetrievalConfig::default().with_embedding_dim(4096);
    let engine = CodeRetrievalEngine::new(config.clone());
    let index = engine
        .index(vec![
            ("unit_a".to_string(), unit_a),
            ("unit_b".to_string(), unit_b),
        ])
        .expect("index builds");
    (query, index, config)
}

// ── CodeRetrievalLanguageHint ─────────────────────────────────────────────────

#[test]
fn language_hint_as_str_and_is_auto() {
    assert_eq!(CodeRetrievalLanguageHint::Auto.as_str(), "auto");
    assert_eq!(CodeRetrievalLanguageHint::BraceDelimited.as_str(), "brace");
    assert_eq!(
        CodeRetrievalLanguageHint::IndentDelimited.as_str(),
        "indent"
    );
    assert!(CodeRetrievalLanguageHint::Auto.is_auto());
    assert!(!CodeRetrievalLanguageHint::BraceDelimited.is_auto());
    assert_eq!(
        CodeRetrievalLanguageHint::default(),
        CodeRetrievalLanguageHint::Auto
    );
}

// ── CodeRetrievalUnitKind ─────────────────────────────────────────────────────

#[test]
fn unit_kind_as_str_and_default() {
    assert_eq!(CodeRetrievalUnitKind::Function.as_str(), "function");
    assert_eq!(CodeRetrievalUnitKind::WholeFile.as_str(), "whole_file");
    assert_eq!(
        CodeRetrievalUnitKind::default(),
        CodeRetrievalUnitKind::Function
    );
}

// ── CodeRetrievalSymbol ───────────────────────────────────────────────────────

#[test]
fn symbol_new_sets_fields() {
    let symbol = CodeRetrievalSymbol::new("parse_header", 3);
    assert_eq!(symbol.name, "parse_header");
    assert_eq!(symbol.occurrences, 3);
}

// ── CodeRetrievalConfig ───────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = CodeRetrievalConfig::default();
    assert_eq!(config.blend_weight, 0.5);
    assert_eq!(config.min_identifier_len, 2);
    assert_eq!(config.embedding_dim, 256);
    assert_eq!(config.top_k, 10);
    assert!(config.use_identifiers);
    assert!(config.use_imports);
    assert!(config.use_call_sites);
    assert!(config.language_hint.is_auto());
    assert!(!config.keywords.is_empty());
}

#[test]
fn config_builders_chain() {
    let config = CodeRetrievalConfig::new()
        .with_blend_weight(0.75)
        .with_min_identifier_len(4)
        .with_embedding_dim(512)
        .with_top_k(3)
        .with_language_hint(CodeRetrievalLanguageHint::IndentDelimited)
        .with_keywords(["alpha", "beta"]);
    assert_eq!(config.blend_weight, 0.75);
    assert_eq!(config.min_identifier_len, 4);
    assert_eq!(config.embedding_dim, 512);
    assert_eq!(config.top_k, 3);
    assert_eq!(
        config.language_hint,
        CodeRetrievalLanguageHint::IndentDelimited
    );
    assert_eq!(
        config.keywords,
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn config_signal_toggles() {
    let config = CodeRetrievalConfig::new().with_signals(false, false, true);
    assert!(!config.use_identifiers);
    assert!(!config.use_imports);
    assert!(config.use_call_sites);
    assert!(config.any_structural_signal());

    let none = CodeRetrievalConfig::new()
        .with_use_identifiers(false)
        .with_use_imports(false)
        .with_use_call_sites(false);
    assert!(!none.any_structural_signal());
}

#[test]
fn config_validate_accepts_defaults() {
    assert!(CodeRetrievalConfig::default().validate().is_ok());
}

#[test]
fn config_validate_rejects_zero_dim() {
    let config = CodeRetrievalConfig::new().with_embedding_dim(0);
    assert_eq!(config.validate(), Err(CodeRetrievalError::ZeroEmbeddingDim));
}

#[test]
fn config_validate_rejects_bad_blend() {
    for weight in [-0.1f32, 1.5, f32::NAN, f32::INFINITY] {
        let config = CodeRetrievalConfig::new().with_blend_weight(weight);
        assert!(matches!(
            config.validate(),
            Err(CodeRetrievalError::InvalidBlendWeight { .. })
        ));
    }
    for weight in [0.0f32, 0.5, 1.0] {
        assert!(
            CodeRetrievalConfig::new()
                .with_blend_weight(weight)
                .validate()
                .is_ok()
        );
    }
}

// ── CodeRetrievalError ────────────────────────────────────────────────────────

#[test]
fn error_messages_are_specific() {
    assert!(CodeRetrievalError::EmptyQuery.to_string().contains("empty"));
    assert!(
        CodeRetrievalError::EmptyCorpus
            .to_string()
            .contains("corpus")
    );
    assert!(
        CodeRetrievalError::ZeroEmbeddingDim
            .to_string()
            .contains("dimension")
    );
    let blend = CodeRetrievalError::InvalidBlendWeight { weight: 2.0 };
    assert!(blend.to_string().contains('2'));
    let mismatch = CodeRetrievalError::DimensionMismatch {
        index_dim: 256,
        config_dim: 128,
    };
    assert!(mismatch.to_string().contains("256") && mismatch.to_string().contains("128"));
}

// ── CodeRetrievalUnit helpers ─────────────────────────────────────────────────

#[test]
fn unit_helpers_report_kind_and_span() {
    let units = parse("fn demo(a, b) {\n    helper(a);\n}");
    assert_eq!(units.len(), 1);
    let unit = &units[0];
    assert!(unit.is_function());
    assert!(!unit.is_whole_file());
    assert_eq!(unit.name, "demo");
    assert_eq!(unit.body_start_line, 0);
    assert_eq!(unit.body_end_line, 2);
    assert_eq!(unit.line_span(), 3);
    assert!(unit.identifier_set().contains("demo"));
    assert_eq!(unit.occurrences_of("nonexistent"), 0);
}

// ── Brace-language function detection ─────────────────────────────────────────

#[test]
fn brace_single_function_name_and_params() {
    let units = parse("fn add(first, second) { return first + second; }");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "add");
    assert_eq!(units[0].param_count, 2);
    assert_eq!(units[0].kind, CodeRetrievalUnitKind::Function);
    assert_eq!(units[0].language, CodeRetrievalLanguageHint::BraceDelimited);
}

#[test]
fn brace_zero_params() {
    let units = parse("fn noop() { }");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "noop");
    assert_eq!(units[0].param_count, 0);
}

#[test]
fn brace_c_style_typed_function() {
    let units = parse("int add(int a, int b) {\n    return a + b;\n}");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "add");
    assert_eq!(units[0].param_count, 2);
}

#[test]
fn brace_method_with_self_and_type_params() {
    let units = parse("fn method(&self, scale: i32) { let x = scale; }");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "method");
    assert_eq!(units[0].param_count, 2);
}

#[test]
fn brace_generic_param_comma_not_double_counted() {
    let units = parse("fn build(map: HashMap<Key, Value>, flag: bool) { }");
    assert_eq!(units.len(), 1);
    // The comma inside `<Key, Value>` must not inflate the parameter count.
    assert_eq!(units[0].param_count, 2);
}

#[test]
fn brace_multiple_functions_produce_multiple_units() {
    let src = "fn first() { }\nfn second(x) { }\nfn third(x, y) { }";
    let units = parse(src);
    assert_eq!(units.len(), 3);
    assert_eq!(names(&units), vec!["first", "second", "third"]);
    assert_eq!(units[0].unit_index, 0);
    assert_eq!(units[1].unit_index, 1);
    assert_eq!(units[2].unit_index, 2);
    assert_eq!(units[2].param_count, 2);
}

#[test]
fn brace_body_span_tracks_matching_brace() {
    let src = "fn outer() {\n    if (ready()) {\n        step();\n    }\n}\n";
    let units = parse(src);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "outer");
    assert_eq!(units[0].body_start_line, 0);
    assert_eq!(units[0].body_end_line, 4);
}

#[test]
fn brace_ignores_control_flow_constructs() {
    let src = "fn process(items) {\n    if (ready(items)) {\n        for (item in items) {\n            handle(item);\n        }\n    }\n}";
    let units = parse(src);
    assert_eq!(units.len(), 1, "only `process` is a function");
    assert_eq!(units[0].name, "process");
    assert!(units[0].call_sites.contains(&"ready".to_string()));
    assert!(units[0].call_sites.contains(&"handle".to_string()));
    assert!(!units[0].call_sites.contains(&"if".to_string()));
    assert!(!units[0].call_sites.contains(&"for".to_string()));
}

#[test]
fn brace_method_in_class_detected_class_is_not() {
    let src = "class Widget {\n    void render() {\n        draw(self);\n    }\n    int measure(int scale) {\n        return scale;\n    }\n}";
    let units = parse(src);
    assert_eq!(units.len(), 2);
    assert_eq!(names(&units), vec!["render", "measure"]);
    assert_eq!(units[1].param_count, 1);
    assert!(!names(&units).contains(&"Widget".to_string()));
}

#[test]
fn brace_multiline_signature_and_trailing_comma() {
    let src = "fn compute(\n    first,\n    second,\n) -> i32 {\n    first\n}";
    let units = parse(src);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "compute");
    assert_eq!(units[0].param_count, 2);
    assert_eq!(units[0].body_end_line, 5);
}

#[test]
fn brace_forward_declaration_is_not_a_function() {
    // A declaration with no body must fall through to the whole-file fallback.
    let units = parse("int lonely(int a);");
    assert_eq!(units.len(), 1);
    assert!(units[0].is_whole_file());
}

// ── Indentation-language function detection ───────────────────────────────────

#[test]
fn indent_single_def_name_and_params() {
    let units = parse("def greet(name):\n    return name");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "greet");
    assert_eq!(units[0].param_count, 1);
    assert_eq!(
        units[0].language,
        CodeRetrievalLanguageHint::IndentDelimited
    );
}

#[test]
fn indent_zero_params() {
    let units = parse("def tick():\n    return 1");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "tick");
    assert_eq!(units[0].param_count, 0);
}

#[test]
fn indent_async_def() {
    let units = parse("async def fetch(url, retries):\n    return url");
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "fetch");
    assert_eq!(units[0].param_count, 2);
}

#[test]
fn indent_body_span_by_indentation() {
    let src = "def outer():\n    a = compute()\n    b = refine()\n    return a\nafter = 3";
    let units = parse(src);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "outer");
    assert_eq!(units[0].body_start_line, 0);
    // Body ends at `return a` (line 3); the dedented `after = 3` (line 4) is out.
    assert_eq!(units[0].body_end_line, 3);
}

#[test]
fn indent_multiple_defs_with_blank_line() {
    let src = "def first():\n    return 1\n\ndef second():\n    return 2";
    let units = parse(src);
    assert_eq!(units.len(), 2);
    assert_eq!(names(&units), vec!["first", "second"]);
}

#[test]
fn indent_nested_def_absorbed_into_enclosing_unit() {
    let src = "def outer(x):\n    def inner(y):\n        return y\n    return inner";
    let units = parse(src);
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "outer");
    // The inner def's identifiers are folded into the enclosing unit.
    assert!(units[0].identifier_set().contains("inner"));
}

#[test]
fn indent_method_inside_class() {
    let src = "class Service:\n    def start(self):\n        boot(self)\n    def stop(self):\n        halt(self)";
    let units = parse(src);
    assert_eq!(units.len(), 2);
    assert_eq!(names(&units), vec!["start", "stop"]);
}

// ── Import extraction ─────────────────────────────────────────────────────────

#[test]
fn imports_rust_use_paths() {
    let src = "use std::collections::HashMap;\nuse crate::foo::Bar;\nfn f() { let m = HashMap; }";
    let units = parse(src);
    assert!(
        units[0]
            .imports
            .contains(&"std::collections::HashMap".to_string())
    );
    assert!(units[0].imports.contains(&"crate::foo::Bar".to_string()));
}

#[test]
fn imports_rust_use_group_takes_base() {
    let src = "use std::io::{Read, Write};\nfn f() { }";
    let units = parse(src);
    assert!(units[0].imports.contains(&"std::io".to_string()));
}

#[test]
fn imports_python_import_and_from() {
    let src = "import os\nfrom collections import defaultdict\ndef f():\n    return os";
    let units = parse(src);
    assert!(units[0].imports.contains(&"os".to_string()));
    assert!(units[0].imports.contains(&"collections".to_string()));
}

#[test]
fn imports_c_include_angle_and_quote() {
    let src = "#include <stdio.h>\n#include \"myheader.h\"\nint main() { return 0; }";
    let units = parse(src);
    assert!(units[0].imports.contains(&"stdio.h".to_string()));
    assert!(units[0].imports.contains(&"myheader.h".to_string()));
}

#[test]
fn imports_js_from_and_require() {
    let src = "import { readFile } from 'fs';\nconst p = require('path');\nfunction run() { return readFile; }";
    let units = parse(src);
    assert!(units[0].imports.contains(&"fs".to_string()));
    assert!(units[0].imports.contains(&"path".to_string()));
}

#[test]
fn imports_shared_across_all_units_of_a_file() {
    let src = "use std::io;\nfn a() { }\nfn b() { }";
    let units = parse(src);
    assert_eq!(units.len(), 2);
    assert_eq!(units[0].imports, vec!["std::io".to_string()]);
    assert_eq!(units[1].imports, vec!["std::io".to_string()]);
}

#[test]
fn imports_are_sorted_and_deduplicated() {
    let src = "use zeta::z;\nuse alpha::a;\nuse zeta::z;\nfn f() { }";
    let units = parse(src);
    assert_eq!(
        units[0].imports,
        vec!["alpha::a".to_string(), "zeta::z".to_string()]
    );
}

// ── Identifier extraction ─────────────────────────────────────────────────────

#[test]
fn identifiers_exclude_configured_keyword_list() {
    // A custom keyword list replaces the default one entirely.
    let config = CodeRetrievalConfig::default()
        .with_min_identifier_len(1)
        .with_keywords(["apple", "banana"]);
    let units = parse_with("fn f() { apple = banana; cherry = plum; }", &config);
    let ids = units[0].identifier_set();
    assert!(!ids.contains("apple"));
    assert!(!ids.contains("banana"));
    assert!(ids.contains("cherry"));
    assert!(ids.contains("plum"));
    // `let`-style default keywords no longer apply under the custom list.
    assert!(ids.contains("f"));
}

#[test]
fn identifiers_exclude_short_tokens() {
    let config = CodeRetrievalConfig::default().with_min_identifier_len(4);
    let units = parse_with("fn compute() { ab = 1; abcd = value; }", &config);
    let ids = units[0].identifier_set();
    assert!(ids.contains("compute"));
    assert!(ids.contains("abcd"));
    assert!(ids.contains("value"));
    assert!(!ids.contains("ab"));
}

#[test]
fn identifiers_exclude_numeric_but_keep_alphanumeric() {
    let units = parse("fn f() { total = item99 + 12345; }");
    let ids = units[0].identifier_set();
    assert!(ids.contains("item99"));
    assert!(ids.contains("total"));
    assert!(!ids.contains("12345"));
}

#[test]
fn identifiers_lowercased_and_counted() {
    let units = parse("fn Foo() { Bar(); Bar(); baz(); }");
    // The unit name preserves source case, but identifier symbols are lowercased.
    assert_eq!(units[0].name, "Foo");
    let ids = units[0].identifier_set();
    assert!(ids.contains("foo"));
    assert!(ids.contains("bar"));
    assert_eq!(units[0].occurrences_of("bar"), 2);
    assert_eq!(units[0].occurrences_of("baz"), 1);
}

#[test]
fn symbols_are_sorted_by_name() {
    let units = parse("fn f() { zebra(); apple(); mango(); }");
    let sorted: Vec<String> = units[0].symbols.iter().map(|s| s.name.clone()).collect();
    let mut expected = sorted.clone();
    expected.sort();
    assert_eq!(sorted, expected);
}

#[test]
fn identifiers_from_comments_are_excluded() {
    let units = parse("// zebra elephant giraffe\nfn f() { let cat = 1; }");
    let ids = units[0].identifier_set();
    assert!(!ids.contains("zebra"));
    assert!(!ids.contains("elephant"));
    assert!(!ids.contains("giraffe"));
    assert!(ids.contains("cat"));
}

// ── Call-site detection ───────────────────────────────────────────────────────

#[test]
fn calls_nested_are_all_detected() {
    let units = parse("fn f() { alpha(beta(gamma(x))); }");
    let calls = &units[0].call_sites;
    assert!(calls.contains(&"alpha".to_string()));
    assert!(calls.contains(&"beta".to_string()));
    assert!(calls.contains(&"gamma".to_string()));
}

#[test]
fn calls_exclude_control_keywords() {
    let src = "fn f() { if (test(x)) { while (check(y)) { run(z); } } }";
    let units = parse(src);
    let calls = &units[0].call_sites;
    assert!(calls.contains(&"test".to_string()));
    assert!(calls.contains(&"check".to_string()));
    assert!(calls.contains(&"run".to_string()));
    assert!(!calls.contains(&"if".to_string()));
    assert!(!calls.contains(&"while".to_string()));
}

#[test]
fn calls_come_from_body_not_signature_name() {
    let units = parse("fn process(data) { transform(data); }");
    let calls = &units[0].call_sites;
    assert!(calls.contains(&"transform".to_string()));
    assert!(
        !calls.contains(&"process".to_string()),
        "the function's own name must not be counted as a call"
    );
}

#[test]
fn calls_are_sorted_and_deduplicated() {
    let units = parse("fn f() { run(); run(); build(); }");
    assert_eq!(
        units[0].call_sites,
        vec!["build".to_string(), "run".to_string()]
    );
}

// ── Language resolution and whole-file fallback ───────────────────────────────

#[test]
fn auto_detects_brace_language() {
    let units = parse("fn f() { g(); }");
    assert_eq!(units[0].language, CodeRetrievalLanguageHint::BraceDelimited);
}

#[test]
fn auto_detects_indent_language() {
    let units = parse("def f():\n    return g()");
    assert_eq!(
        units[0].language,
        CodeRetrievalLanguageHint::IndentDelimited
    );
}

#[test]
fn explicit_language_hint_overrides_detection() {
    // Force indentation scanning even though a brace is present.
    let config = CodeRetrievalConfig::default()
        .with_language_hint(CodeRetrievalLanguageHint::IndentDelimited);
    let units = parse_with("def f():\n    return { 1 }", &config);
    assert_eq!(
        units[0].language,
        CodeRetrievalLanguageHint::IndentDelimited
    );
    assert_eq!(units[0].name, "f");
}

#[test]
fn whole_file_fallback_when_no_function() {
    let units = parse("total = compute(inputs) + refine(inputs);");
    assert_eq!(units.len(), 1);
    assert!(units[0].is_whole_file());
    assert_eq!(units[0].kind, CodeRetrievalUnitKind::WholeFile);
    assert_eq!(units[0].name, "item");
    // Structural features are still extracted for a whole-file unit.
    assert!(units[0].identifier_set().contains("compute"));
    assert!(units[0].call_sites.contains(&"compute".to_string()));
    assert!(units[0].call_sites.contains(&"refine".to_string()));
}

#[test]
fn parse_source_reexport_is_usable() {
    let units = parse_source("x.rs", "fn q() { }", &CodeRetrievalConfig::default());
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].source_id, "x.rs");
}

// ── Engine: index construction and errors ─────────────────────────────────────

#[test]
fn index_empty_corpus_errors() {
    let engine = CodeRetrievalEngine::new(cfg());
    let items: Vec<(String, String)> = Vec::new();
    assert_eq!(engine.index(items), Err(CodeRetrievalError::EmptyCorpus));
}

#[test]
fn index_zero_embedding_dim_errors() {
    let engine = CodeRetrievalEngine::new(cfg().with_embedding_dim(0));
    assert_eq!(
        engine.index(vec![("a", "fn f() { }")]),
        Err(CodeRetrievalError::ZeroEmbeddingDim)
    );
}

#[test]
fn index_multi_function_file_produces_multiple_units() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![("multi.rs", "fn a() { }\nfn b() { }\nfn c() { }")])
        .expect("index builds");
    assert_eq!(index.len(), 3);
    assert!(!index.is_empty());
    assert_eq!(index.embedding_dim(), 256);
    assert_eq!(index.units().len(), 3);
}

#[test]
fn engine_parse_convenience_matches_free_function() {
    let engine = CodeRetrievalEngine::new(cfg());
    let a = engine.parse("z.rs", "fn f(x) { g(x); }");
    let b = parse_source("z.rs", "fn f(x) { g(x); }", &cfg());
    assert_eq!(a, b);
}

// ── Engine: search errors ─────────────────────────────────────────────────────

#[test]
fn search_empty_query_errors() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine.index(vec![("a", "fn f() { }")]).expect("index");
    assert_eq!(
        engine.search(&index, ""),
        Err(CodeRetrievalError::EmptyQuery)
    );
    assert_eq!(
        engine.search(&index, "   \n\t "),
        Err(CodeRetrievalError::EmptyQuery)
    );
}

#[test]
fn search_dimension_mismatch_errors() {
    let build = CodeRetrievalEngine::new(cfg().with_embedding_dim(256));
    let index = build.index(vec![("a", "fn f() { }")]).expect("index");
    let searcher = CodeRetrievalEngine::new(cfg().with_embedding_dim(128));
    assert_eq!(
        searcher.search(&index, "query"),
        Err(CodeRetrievalError::DimensionMismatch {
            index_dim: 256,
            config_dim: 128,
        })
    );
}

#[test]
fn search_invalid_blend_weight_errors() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine.index(vec![("a", "fn f() { }")]).expect("index");
    let bad = CodeRetrievalEngine::new(cfg().with_blend_weight(2.0));
    assert!(matches!(
        bad.search(&index, "query"),
        Err(CodeRetrievalError::InvalidBlendWeight { .. })
    ));
}

// ── Engine: ranking and blending ──────────────────────────────────────────────

#[test]
fn search_ranks_structural_match_first() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![
            (
                "tokenizer",
                "fn scan(buffer) { let t = read_token(buffer); advance_cursor(buffer); }",
            ),
            ("math", "fn add(a, b) { return a + b; }"),
        ])
        .expect("index");
    let result = engine
        .search(&index, "read_token(buffer) then advance_cursor(buffer)")
        .expect("search");
    assert_eq!(result.units_searched, 2);
    let top = result.top().expect("a hit");
    assert_eq!(top.unit.name, "scan");
    assert!(top.structural_score > 0.0);
}

#[test]
fn search_result_helpers() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine.index(vec![("a", "fn f() { g(); }")]).expect("index");
    let result = engine.search(&index, "g()").expect("search");
    assert!(!result.is_empty());
    assert_eq!(result.len(), 1);
    assert!(result.top().is_some());
    assert_eq!(result.best_score(), result.hits[0].score);
}

#[test]
fn top_k_limits_returned_hits() {
    let engine = CodeRetrievalEngine::new(cfg().with_top_k(1));
    let index = engine
        .index(vec![(
            "m",
            "fn a() { p(); }\nfn b() { q(); }\nfn c() { r(); }",
        )])
        .expect("index");
    let result = engine.search(&index, "p() q() r()").expect("search");
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.units_searched, 3);
}

#[test]
fn score_breakdown_fields_are_in_range() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![("a", "fn f(x) { transform(x); }")])
        .expect("index");
    let result = engine.search(&index, "transform(x)").expect("search");
    let hit = result.top().expect("a hit");
    for value in [
        hit.score,
        hit.structural_score,
        hit.text_score,
        hit.identifier_overlap,
        hit.import_overlap,
        hit.call_site_overlap,
    ] {
        assert!((0.0..=1.0).contains(&value), "value {value} out of range");
    }
    assert!(hit.identifier_overlap > 0.0);
    assert!(hit.call_site_overlap > 0.0);
}

#[test]
fn blend_weight_one_is_structural_only() {
    let (query, index, config) = adversarial();
    let engine = CodeRetrievalEngine::new(config.with_blend_weight(1.0));
    let result = engine.search(&index, &query).expect("search");
    let a = hit_for(&result, "unit_a");
    let b = hit_for(&result, "unit_b");
    // Structural-only: the score equals the structural component exactly.
    assert_eq!(a.score, a.structural_score);
    assert_eq!(b.score, b.structural_score);
    // The structural match (B) outranks the lexical match (A).
    assert!(b.score > a.score);
    assert_eq!(result.top().expect("hit").unit.source_id, "unit_b");
}

#[test]
fn blend_weight_zero_is_text_only() {
    let (query, index, config) = adversarial();
    let engine = CodeRetrievalEngine::new(config.with_blend_weight(0.0));
    let result = engine.search(&index, &query).expect("search");
    let a = hit_for(&result, "unit_a");
    let b = hit_for(&result, "unit_b");
    // Text-only: the score equals the text component exactly.
    assert_eq!(a.score, a.text_score);
    assert_eq!(b.score, b.text_score);
    // The lexical match (A) outranks the structural match (B).
    assert!(a.score > b.score);
    assert_eq!(result.top().expect("hit").unit.source_id, "unit_a");
}

/// THE key test: the structural signal genuinely changes ranking versus pure
/// text similarity. With a high structural weight the structurally-matching
/// unit B wins; with structural weight 0 the ordering flips to the
/// lexically-matching unit A.
#[test]
fn structural_weight_changes_ranking_adversarially() {
    let (query, index, config) = adversarial();

    // Sanity: A shares the query's vocabulary; B shares its identifiers/calls.
    let probe = CodeRetrievalEngine::new(config.clone().with_blend_weight(0.5));
    let probe_result = probe.search(&index, &query).expect("search");
    let a = hit_for(&probe_result, "unit_a");
    let b = hit_for(&probe_result, "unit_b");
    assert_eq!(a.identifier_overlap, 0.0, "A shares no identifiers");
    assert_eq!(a.call_site_overlap, 0.0, "A shares no calls");
    assert!(b.identifier_overlap > 0.0, "B shares identifiers");
    assert!(b.call_site_overlap > 0.0, "B shares calls");
    assert!(
        a.text_score > b.text_score,
        "A is the stronger lexical match"
    );
    assert!(
        b.structural_score > a.structural_score,
        "B is the stronger structural match"
    );

    // High structural weight → B ranks first.
    let structural = CodeRetrievalEngine::new(config.clone().with_blend_weight(0.9));
    let structural_top = structural
        .search(&index, &query)
        .expect("search")
        .top()
        .expect("hit")
        .unit
        .source_id
        .clone();
    assert_eq!(structural_top, "unit_b");

    // Zero structural weight → the ordering flips to A.
    let textual = CodeRetrievalEngine::new(config.with_blend_weight(0.0));
    let textual_top = textual
        .search(&index, &query)
        .expect("search")
        .top()
        .expect("hit")
        .unit
        .source_id
        .clone();
    assert_eq!(textual_top, "unit_a");

    assert_ne!(structural_top, textual_top, "the ranking genuinely flipped");
}

#[test]
fn disabling_structural_signals_zeroes_structural_score() {
    let (query, index, config) = adversarial();
    let engine = CodeRetrievalEngine::new(config.with_signals(false, false, false));
    let result = engine.search(&index, &query).expect("search");
    for hit in &result.hits {
        assert_eq!(hit.structural_score, 0.0);
    }
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn search_is_deterministic() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![
            ("a", "fn alpha(x) { read(x); }"),
            ("b", "fn beta(y) { write(y); }"),
            ("c", "fn gamma(z) { read(z); write(z); }"),
        ])
        .expect("index");
    let first = engine.search(&index, "read(v) write(v)").expect("search");
    let second = engine.search(&index, "read(v) write(v)").expect("search");
    assert_eq!(first, second);
}

#[test]
fn indexing_is_deterministic() {
    let engine = CodeRetrievalEngine::new(cfg());
    let corpus = vec![("a", "fn f(x) { g(x); h(x); }")];
    let index_one = engine.index(corpus.clone()).expect("index");
    let index_two = engine.index(corpus).expect("index");
    assert_eq!(index_one.units(), index_two.units());
}

#[test]
fn tie_break_is_stable_by_source_id() {
    // Two identical units under a query that matches neither: equal scores must
    // resolve deterministically by source id.
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![("zzz", "fn same() { }"), ("aaa", "fn same() { }")])
        .expect("index");
    let result = engine.search(&index, "unrelated_symbol").expect("search");
    assert_eq!(result.hits[0].unit.source_id, "aaa");
    assert_eq!(result.hits[1].unit.source_id, "zzz");
}

// ── Whole-file units participate in search ────────────────────────────────────

#[test]
fn whole_file_unit_is_searchable() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![("script", "total = accumulate(values);")])
        .expect("index");
    let result = engine.search(&index, "accumulate(values)").expect("search");
    let top = result.top().expect("a hit");
    assert!(top.unit.is_whole_file());
    assert!(top.identifier_overlap > 0.0);
}

#[test]
fn comment_vocabulary_drives_text_not_structural() {
    let engine = CodeRetrievalEngine::new(cfg());
    let index = engine
        .index(vec![(
            "f",
            "// zebra elephant giraffe\nfn f() { let cat = 1; }",
        )])
        .expect("index");
    // Query words match only the comment, not any identifier.
    let result = engine
        .search(&index, "zebra elephant giraffe")
        .expect("search");
    let hit = result.top().expect("a hit");
    assert_eq!(hit.identifier_overlap, 0.0, "no identifier overlap");
    assert!(hit.text_score > 0.0, "comment words feed the text signal");
}
