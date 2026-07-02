#![allow(clippy::float_cmp, clippy::similar_names, clippy::too_many_lines)]

use crate::tool_retrieval::engine::{ArgumentGrounder, ToolRetrievalEngine, ToolRetrievalIndex};
use crate::tool_retrieval::types::{
    ArgumentGroundingStatus, GroundedArgument, ToolMatch, ToolParameter, ToolParameterType,
    ToolRetrievalConfig, ToolRetrievalError, ToolSpecEntry,
};

// ── ToolParameterType ─────────────────────────────────────────────────────────

#[test]
fn tool_parameter_type_name() {
    assert_eq!(ToolParameterType::String.name(), "string");
    assert_eq!(ToolParameterType::Number.name(), "number");
    assert_eq!(ToolParameterType::Boolean.name(), "boolean");
    assert_eq!(
        ToolParameterType::Enum(vec!["a".to_string()]).name(),
        "enum"
    );
}

// ── ToolParameter ─────────────────────────────────────────────────────────────

#[test]
fn tool_parameter_new_defaults() {
    let param = ToolParameter::new("city", ToolParameterType::String);
    assert_eq!(param.name, "city");
    assert_eq!(param.param_type, ToolParameterType::String);
    assert!(!param.required);
    assert_eq!(param.description, "");
}

#[test]
fn tool_parameter_with_required() {
    let param = ToolParameter::new("city", ToolParameterType::String).with_required(true);
    assert!(param.required);
}

#[test]
fn tool_parameter_with_description() {
    let param = ToolParameter::new("city", ToolParameterType::String)
        .with_description("the destination city");
    assert_eq!(param.description, "the destination city");
}

// ── ToolSpecEntry ─────────────────────────────────────────────────────────────

#[test]
fn tool_spec_entry_new_defaults() {
    let spec = ToolSpecEntry::new("weather", "looks up the weather");
    assert_eq!(spec.name, "weather");
    assert_eq!(spec.description, "looks up the weather");
    assert!(spec.parameters.is_empty());
}

#[test]
fn tool_spec_entry_with_parameter() {
    let spec = ToolSpecEntry::new("weather", "looks up the weather")
        .with_parameter(ToolParameter::new("city", ToolParameterType::String));
    assert_eq!(spec.parameters.len(), 1);
    assert_eq!(spec.parameters[0].name, "city");
}

#[test]
fn tool_spec_entry_with_parameters() {
    let params = vec![
        ToolParameter::new("city", ToolParameterType::String),
        ToolParameter::new("units", ToolParameterType::String),
    ];
    let spec = ToolSpecEntry::new("weather", "looks up the weather").with_parameters(params);
    assert_eq!(spec.parameters.len(), 2);
}

#[test]
fn tool_spec_entry_required_parameters() {
    let spec = ToolSpecEntry::new("weather", "looks up the weather")
        .with_parameter(ToolParameter::new("city", ToolParameterType::String).with_required(true))
        .with_parameter(ToolParameter::new("units", ToolParameterType::String));
    let required = spec.required_parameters();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0].name, "city");
}

#[test]
fn tool_spec_entry_parameter_lookup() {
    let spec = ToolSpecEntry::new("weather", "looks up the weather")
        .with_parameter(ToolParameter::new("city", ToolParameterType::String));
    assert!(spec.parameter("city").is_some());
    assert!(spec.parameter("missing").is_none());
}

// ── ArgumentGroundingStatus / GroundedArgument ────────────────────────────────

#[test]
fn argument_grounding_status_is_grounded() {
    assert!(ArgumentGroundingStatus::Grounded.is_grounded());
    assert!(!ArgumentGroundingStatus::UngroundedRequired.is_grounded());
    assert!(!ArgumentGroundingStatus::UngroundedOptional.is_grounded());
}

#[test]
fn argument_grounding_status_is_missing_required() {
    assert!(ArgumentGroundingStatus::UngroundedRequired.is_missing_required());
    assert!(!ArgumentGroundingStatus::UngroundedOptional.is_missing_required());
    assert!(!ArgumentGroundingStatus::Grounded.is_missing_required());
}

#[test]
fn grounded_argument_is_grounded_delegates_to_status() {
    let grounded = GroundedArgument {
        parameter_name: "x".to_string(),
        value: Some("y".to_string()),
        confidence: 0.9,
        source_span: Some("y".to_string()),
        status: ArgumentGroundingStatus::Grounded,
    };
    assert!(grounded.is_grounded());
}

// ── ToolRetrievalConfig ────────────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = ToolRetrievalConfig::default();
    assert_eq!(config.embedding_dim, 128);
    assert_eq!(config.description_weight, 0.65);
    assert_eq!(config.default_top_k, 5);
    assert_eq!(config.min_match_score, 0.05);
}

#[test]
fn config_builder_methods() {
    let config = ToolRetrievalConfig::new()
        .with_embedding_dim(64)
        .with_description_weight(0.5)
        .with_default_top_k(3)
        .with_min_match_score(0.1);
    assert_eq!(config.embedding_dim, 64);
    assert_eq!(config.description_weight, 0.5);
    assert_eq!(config.default_top_k, 3);
    assert_eq!(config.min_match_score, 0.1);
}

#[test]
fn config_validate_ok() {
    assert!(ToolRetrievalConfig::default().validate().is_ok());
}

#[test]
fn config_validate_invalid_embedding_dim() {
    let config = ToolRetrievalConfig::default().with_embedding_dim(0);
    assert_eq!(
        config.validate(),
        Err(ToolRetrievalError::InvalidEmbeddingDim)
    );
}

#[test]
fn config_validate_invalid_description_weight_low() {
    let config = ToolRetrievalConfig::default().with_description_weight(-0.1);
    assert_eq!(
        config.validate(),
        Err(ToolRetrievalError::InvalidDescriptionWeight(-0.1))
    );
}

#[test]
fn config_validate_invalid_description_weight_high() {
    let config = ToolRetrievalConfig::default().with_description_weight(1.1);
    assert_eq!(
        config.validate(),
        Err(ToolRetrievalError::InvalidDescriptionWeight(1.1))
    );
}

#[test]
fn config_validate_invalid_top_k() {
    let config = ToolRetrievalConfig::default().with_default_top_k(0);
    assert_eq!(config.validate(), Err(ToolRetrievalError::InvalidTopK));
}

// ── ToolRetrievalIndex::build ──────────────────────────────────────────────────

#[test]
fn index_build_empty_registry_errors() {
    let result = ToolRetrievalIndex::build(vec![], ToolRetrievalConfig::default());
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyRegistry);
}

#[test]
fn index_build_empty_tool_name_errors() {
    let tools = vec![ToolSpecEntry::new("", "does something")];
    let result = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default());
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyToolName);
}

#[test]
fn index_build_whitespace_tool_name_errors() {
    let tools = vec![ToolSpecEntry::new("   ", "does something")];
    let result = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default());
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyToolName);
}

#[test]
fn index_build_duplicate_name_errors_case_insensitive() {
    let tools = vec![
        ToolSpecEntry::new("Search", "first tool"),
        ToolSpecEntry::new("search", "second tool"),
    ];
    let result = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default());
    assert_eq!(
        result.unwrap_err(),
        ToolRetrievalError::DuplicateToolName("search".to_string())
    );
}

#[test]
fn index_build_invalid_config_propagates() {
    let tools = vec![ToolSpecEntry::new("a", "does something")];
    let config = ToolRetrievalConfig::default().with_embedding_dim(0);
    let result = ToolRetrievalIndex::build(tools, config);
    assert_eq!(result.unwrap_err(), ToolRetrievalError::InvalidEmbeddingDim);
}

#[test]
fn index_build_success_len_and_names() {
    let tools = vec![
        ToolSpecEntry::new("alpha", "first tool"),
        ToolSpecEntry::new("beta", "second tool"),
    ];
    let index = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default())
        .expect("build should succeed");
    assert_eq!(index.len(), 2);
    assert!(!index.is_empty());
    assert_eq!(index.tool_names(), vec!["alpha", "beta"]);
}

#[test]
fn index_get_unknown_returns_none() {
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let index = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default())
        .expect("build should succeed");
    assert!(index.get("alpha").is_some());
    assert!(index.get("missing").is_none());
}

#[test]
fn index_config_accessor_roundtrips() {
    let config = ToolRetrievalConfig::default().with_embedding_dim(32);
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let index = ToolRetrievalIndex::build(tools, config.clone()).expect("build should succeed");
    assert_eq!(index.config(), &config);
}

// ── ToolRetrievalEngine construction ───────────────────────────────────────────

#[test]
fn engine_build_empty_registry_errors() {
    let result = ToolRetrievalEngine::build(vec![], ToolRetrievalConfig::default());
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyRegistry);
}

#[test]
fn engine_new_wraps_index() {
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let index = ToolRetrievalIndex::build(tools, ToolRetrievalConfig::default())
        .expect("build should succeed");
    let engine = ToolRetrievalEngine::new(index);
    assert_eq!(engine.index().len(), 1);
}

#[test]
fn engine_index_accessor_exposes_tool_names() {
    let tools = vec![
        ToolSpecEntry::new("alpha", "first tool"),
        ToolSpecEntry::new("beta", "second tool"),
    ];
    let engine = ToolRetrievalEngine::build(tools, ToolRetrievalConfig::default()).expect("builds");
    assert_eq!(engine.index().tool_names(), vec!["alpha", "beta"]);
}

// ── ToolRetrievalEngine::retrieve ───────────────────────────────────────────────

/// The key differentiator: a tool with a deliberately unhelpful, generic
/// name whose *description* matches the query, alongside decoy tools that
/// share no vocabulary with the query at all (in either their name or their
/// description). `agentic`'s name-substring approach would find *nothing*
/// here, since no tool's name is a substring of the query; `tool_retrieval`
/// finds the right tool via its description.
#[test]
fn retrieve_ranks_by_description_not_name() {
    let generic_but_relevant = ToolSpecEntry::new(
        "util_seven",
        "Convert a distance value between kilometers and miles for travel planning",
    );
    let decoy_a = ToolSpecEntry::new(
        "handler_alpha",
        "Formats a person's mailing address into a standard postal layout",
    );
    let decoy_b = ToolSpecEntry::new(
        "module_nine",
        "Schedules a recurring calendar reminder at a fixed time each day",
    );

    let engine = ToolRetrievalEngine::build(
        vec![generic_but_relevant, decoy_a, decoy_b],
        ToolRetrievalConfig::default(),
    )
    .expect("build should succeed");

    let matches = engine
        .retrieve("convert distance between kilometers and miles", 3)
        .expect("retrieval should succeed");

    assert!(!matches.is_empty());
    assert_eq!(matches[0].tool_name, "util_seven");
}

/// Same scenario as [`retrieve_ranks_by_description_not_name`], but with
/// `description_weight` pinned to `1.0` so the ranking is driven *purely* by
/// the FNV-1a embedding cosine similarity (the lexical/Jaccard term
/// contributes nothing to the score) — proving the embedding channel alone,
/// not just the lexical channel, favours the semantically matching tool.
#[test]
fn retrieve_pure_cosine_channel_still_favors_semantic_match() {
    let generic_but_relevant = ToolSpecEntry::new(
        "util_seven",
        "Convert a distance value between kilometers and miles for travel planning",
    );
    let decoy = ToolSpecEntry::new(
        "handler_alpha",
        "Formats a person's mailing address into a standard postal layout",
    );

    let config = ToolRetrievalConfig::default()
        .with_description_weight(1.0)
        .with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(vec![generic_but_relevant, decoy], config)
        .expect("build should succeed");

    let matches = engine
        .retrieve("convert distance between kilometers and miles", 2)
        .expect("retrieval should succeed");

    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].tool_name, "util_seven");
    // With `description_weight == 1.0` the lexical channel contributes
    // nothing to the blend: `score` collapses to exactly `description_score`,
    // even though `lexical_score` is still independently computed/reported.
    assert_eq!(matches[0].score, matches[0].description_score);
    assert!(matches[0].description_score > 0.0);
}

/// Full ordering across four tools with a strong, a partial, and two
/// zero-overlap matches, pinned to the pure-lexical (Jaccard) channel via
/// `description_weight = 0.0` so the expected order is exactly
/// hand-computable rational arithmetic, independent of any hash-bucket
/// behaviour.
#[test]
fn retrieve_ranking_correctness_multi_tool_registry() {
    let strong = ToolSpecEntry::new(
        "util_seven",
        "Convert a distance value between kilometers and miles for travel planning",
    );
    let partial = ToolSpecEntry::new(
        "module_three",
        "Reports the current distance traveled by a vehicle in kilometers for fuel logs",
    );
    let none_a = ToolSpecEntry::new(
        "handler_alpha",
        "Formats a person's mailing address into a standard postal layout",
    );
    let none_z = ToolSpecEntry::new(
        "zzz_scheduler",
        "Schedules a recurring calendar reminder at a fixed time each day",
    );

    let config = ToolRetrievalConfig::default()
        .with_description_weight(0.0)
        .with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(vec![strong, partial, none_a, none_z], config)
        .expect("build should succeed");

    let matches = engine
        .retrieve("convert distance between kilometers and miles", 4)
        .expect("retrieval should succeed");

    assert_eq!(matches.len(), 4);
    assert_eq!(matches[0].tool_name, "util_seven");
    assert!((matches[0].score - 0.5).abs() < 1e-6);
    assert_eq!(matches[1].tool_name, "module_three");
    assert!((matches[1].score - 2.0 / 13.0).abs() < 1e-5);
    // Both zero-overlap tools tie at score 0.0; ascending name breaks the tie.
    assert_eq!(matches[2].tool_name, "handler_alpha");
    assert_eq!(matches[2].score, 0.0);
    assert_eq!(matches[3].tool_name, "zzz_scheduler");
    assert_eq!(matches[3].score, 0.0);
}

#[test]
fn retrieve_tie_break_by_ascending_name() {
    // Identical descriptions (differing only in the tool's own name) blended
    // at `description_weight = 0.0` produce an exact score tie, isolating
    // the tie-break comparator.
    let zzz = ToolSpecEntry::new(
        "zzz_tool",
        "Performs a specialized diagnostic scan of the system",
    );
    let aaa = ToolSpecEntry::new(
        "aaa_tool",
        "Performs a specialized diagnostic scan of the system",
    );

    let config = ToolRetrievalConfig::default()
        .with_description_weight(0.0)
        .with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(vec![zzz, aaa], config).expect("build should succeed");

    let matches = engine
        .retrieve("run a diagnostic scan", 2)
        .expect("retrieval should succeed");

    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].score, matches[1].score);
    assert_eq!(matches[0].tool_name, "aaa_tool");
    assert_eq!(matches[1].tool_name, "zzz_tool");
}

#[test]
fn retrieve_empty_query_errors() {
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let engine = ToolRetrievalEngine::build(tools, ToolRetrievalConfig::default()).expect("builds");
    let result = engine.retrieve("   ", 5);
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyQuery);
}

#[test]
fn retrieve_no_match_above_threshold_returns_empty_not_error() {
    let tools = vec![ToolSpecEntry::new(
        "helper_one",
        "Looks up account balance information",
    )];
    let config = ToolRetrievalConfig::default().with_min_match_score(0.99);
    let engine = ToolRetrievalEngine::build(tools, config).expect("builds");

    let result = engine.retrieve("check my account balance", 5);
    assert!(result.is_ok());
    assert!(result.expect("ok").is_empty());
}

#[test]
fn retrieve_top_k_zero_returns_empty() {
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let engine = ToolRetrievalEngine::build(tools, ToolRetrievalConfig::default()).expect("builds");
    let matches = engine.retrieve("first tool please", 0).expect("ok");
    assert!(matches.is_empty());
}

#[test]
fn retrieve_top_k_limits_results() {
    let tools = vec![
        ToolSpecEntry::new("alpha", "first tool"),
        ToolSpecEntry::new("beta", "second tool"),
        ToolSpecEntry::new("gamma", "third tool"),
    ];
    let config = ToolRetrievalConfig::default().with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(tools, config).expect("builds");
    let matches = engine.retrieve("some unrelated query text", 2).expect("ok");
    assert_eq!(matches.len(), 2);
}

#[test]
fn retrieve_is_deterministic() {
    let tools = vec![
        ToolSpecEntry::new("alpha", "converts currency amounts between two units"),
        ToolSpecEntry::new("beta", "schedules a meeting on a calendar"),
    ];
    let engine = ToolRetrievalEngine::build(tools, ToolRetrievalConfig::default()).expect("builds");

    let first = engine.retrieve("convert currency amounts", 5).expect("ok");
    let second = engine.retrieve("convert currency amounts", 5).expect("ok");
    assert_eq!(first, second);
}

#[test]
fn retrieve_scores_within_bounds() {
    let tools = vec![
        ToolSpecEntry::new("alpha", "converts currency amounts between two units"),
        ToolSpecEntry::new("beta", "schedules a meeting on a calendar"),
    ];
    let config = ToolRetrievalConfig::default().with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(tools, config).expect("builds");
    let matches = engine.retrieve("convert currency amounts", 5).expect("ok");
    for m in &matches {
        assert!((0.0..=1.0).contains(&m.score));
        assert!((0.0..=1.0).contains(&m.description_score));
        assert!((0.0..=1.0).contains(&m.lexical_score));
    }
}

#[test]
fn retrieve_score_matches_blend_formula() {
    let tools = vec![ToolSpecEntry::new(
        "alpha",
        "converts currency amounts between two units",
    )];
    let config = ToolRetrievalConfig::default()
        .with_description_weight(0.3)
        .with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(tools, config.clone()).expect("builds");
    let matches = engine.retrieve("convert currency amounts", 5).expect("ok");
    let top = &matches[0];
    let expected = config.description_weight * top.description_score
        + (1.0 - config.description_weight) * top.lexical_score;
    assert!((top.score - expected).abs() < 1e-5);
}

#[test]
fn retrieve_returns_empty_registry_error_is_unreachable_via_build_invariant() {
    // A `ToolRetrievalIndex` can never be built empty (see
    // `index_build_empty_registry_errors`), so `retrieve`'s own
    // `EmptyRegistry` guard is exercised indirectly: this test simply
    // documents that a successfully-built engine always has a non-empty
    // index, which is what makes that guard defensive-only.
    let tools = vec![ToolSpecEntry::new("alpha", "first tool")];
    let engine = ToolRetrievalEngine::build(tools, ToolRetrievalConfig::default()).expect("builds");
    assert!(!engine.index().is_empty());
}

// ── ArgumentGrounder / ToolRetrievalEngine::ground — String ────────────────────

#[test]
fn ground_string_quoted_anchored() {
    let spec = ToolSpecEntry::new("book_travel", "books travel").with_parameter(
        ToolParameter::new("destination", ToolParameterType::String).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground(r#"set the destination to "Berlin" now"#, &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded.len(), 1);
    assert_eq!(grounded[0].parameter_name, "destination");
    assert_eq!(grounded[0].value.as_deref(), Some("Berlin"));
    assert_eq!(grounded[0].source_span.as_deref(), Some("Berlin"));
    assert_eq!(grounded[0].confidence, 0.97);
    assert!(grounded[0].is_grounded());
}

#[test]
fn ground_string_quoted_unanchored() {
    let spec = ToolSpecEntry::new("book_travel", "books travel")
        .with_parameter(ToolParameter::new("location", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground(r#"please look up "Tokyo" for me"#, &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("Tokyo"));
    assert_eq!(grounded[0].confidence, 0.85);
}

#[test]
fn ground_string_capitalized_anchored() {
    let spec = ToolSpecEntry::new("assign_task", "assigns a task")
        .with_parameter(ToolParameter::new("assignee", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set assignee Bob for this task", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("Bob"));
    assert_eq!(grounded[0].confidence, 0.75);
}

#[test]
fn ground_string_plain_token_anchored() {
    let spec = ToolSpecEntry::new("configure_run", "configures a run")
        .with_parameter(ToolParameter::new("mode", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set mode fast for this run", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("fast"));
    assert_eq!(grounded[0].confidence, 0.75);
}

#[test]
fn ground_string_capitalized_fallback_unanchored() {
    let spec = ToolSpecEntry::new("send_report", "sends a report")
        .with_parameter(ToolParameter::new("recipient", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("send the report to Madrid", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("Madrid"));
    assert_eq!(grounded[0].confidence, 0.45);
}

#[test]
fn ground_string_required_no_value_flagged_ungrounded() {
    let spec = ToolSpecEntry::new("process_file", "processes a file").with_parameter(
        ToolParameter::new("target_file", ToolParameterType::String).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("please process this now", &spec)
        .expect("grounding should succeed");

    // The parameter is not silently omitted: it is still present in the
    // returned vec, just flagged as ungrounded.
    assert_eq!(grounded.len(), 1);
    assert_eq!(grounded[0].parameter_name, "target_file");
    assert!(grounded[0].value.is_none());
    assert!(grounded[0].source_span.is_none());
    assert_eq!(
        grounded[0].status,
        ArgumentGroundingStatus::UngroundedRequired
    );
    assert!(!grounded[0].is_grounded());
}

#[test]
fn ground_string_optional_no_value_flagged_ungrounded_optional() {
    let spec = ToolSpecEntry::new("annotate", "annotates something")
        .with_parameter(ToolParameter::new("notes", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("describe the general process flow", &spec)
        .expect("grounding should succeed");

    assert!(grounded[0].value.is_none());
    assert_eq!(
        grounded[0].status,
        ArgumentGroundingStatus::UngroundedOptional
    );
}

// ── Grounding — Number ──────────────────────────────────────────────────────────

#[test]
fn ground_number_anchored() {
    let spec = ToolSpecEntry::new("configure_retry", "configures retry behavior").with_parameter(
        ToolParameter::new("max_retries", ToolParameterType::Number).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set max_retries to 7 for safety", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("7"));
    assert_eq!(grounded[0].confidence, 0.8);
}

#[test]
fn ground_number_unanchored() {
    let spec = ToolSpecEntry::new("count_items", "counts items")
        .with_parameter(ToolParameter::new("quantity", ToolParameterType::Number));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("there are 12 apples in the basket", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("12"));
    assert_eq!(grounded[0].confidence, 0.4);
}

#[test]
fn ground_number_absent_ungrounded() {
    let spec = ToolSpecEntry::new("adjust_amount", "adjusts an amount")
        .with_parameter(ToolParameter::new("amount", ToolParameterType::Number));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("please increase the count significantly", &spec)
        .expect("grounding should succeed");

    assert!(grounded[0].value.is_none());
    assert_eq!(
        grounded[0].status,
        ArgumentGroundingStatus::UngroundedOptional
    );
}

#[test]
fn ground_number_supports_decimals_and_negatives() {
    let spec = ToolSpecEntry::new("adjust_temp", "adjusts a temperature")
        .with_parameter(ToolParameter::new("delta", ToolParameterType::Number));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("apply a delta of -3.5 degrees", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("-3.5"));
}

// ── Grounding — Boolean ──────────────────────────────────────────────────────────

#[test]
fn ground_boolean_strong_anchored() {
    let spec = ToolSpecEntry::new("configure_logging", "configures logging")
        .with_parameter(ToolParameter::new("verbose", ToolParameterType::Boolean));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set verbose to true please", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("true"));
    assert_eq!(grounded[0].confidence, 0.9);
}

#[test]
fn ground_boolean_weak_anchored() {
    let spec =
        ToolSpecEntry::new("configure_notifications", "configures notifications").with_parameter(
            ToolParameter::new("notifications", ToolParameterType::Boolean),
        );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("please enable notifications now", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("true"));
    assert_eq!(grounded[0].confidence, 0.65);
}

#[test]
fn ground_boolean_negative_cue_anchored() {
    let spec = ToolSpecEntry::new("configure_logging", "configures logging")
        .with_parameter(ToolParameter::new("debug", ToolParameterType::Boolean));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set debug to false today", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("false"));
}

#[test]
fn ground_boolean_unanchored() {
    let spec = ToolSpecEntry::new("configure_power", "configures power")
        .with_parameter(ToolParameter::new("power", ToolParameterType::Boolean));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("the setting is currently off by default", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("false"));
    assert_eq!(grounded[0].confidence, 0.35);
}

#[test]
fn ground_boolean_absent_ungrounded_optional() {
    let spec = ToolSpecEntry::new("describe_process", "describes a process")
        .with_parameter(ToolParameter::new("flag", ToolParameterType::Boolean));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("describe the general process flow", &spec)
        .expect("grounding should succeed");

    assert!(grounded[0].value.is_none());
    assert_eq!(
        grounded[0].status,
        ArgumentGroundingStatus::UngroundedOptional
    );
}

// ── Grounding — Enum ──────────────────────────────────────────────────────────────

#[test]
fn ground_enum_case_insensitive_anchored() {
    let allowed = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
    let spec = ToolSpecEntry::new("create_ticket", "creates a ticket").with_parameter(
        ToolParameter::new("priority", ToolParameterType::Enum(allowed)).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set priority to HIGH right away", &spec)
        .expect("grounding should succeed");

    // Canonical (allowed-list) casing in `value`, original query casing
    // preserved in `source_span`.
    assert_eq!(grounded[0].value.as_deref(), Some("high"));
    assert_eq!(grounded[0].source_span.as_deref(), Some("HIGH"));
    assert_eq!(grounded[0].confidence, 0.88);
}

#[test]
fn ground_enum_rejects_value_not_in_allowed_set() {
    let allowed = vec!["red".to_string(), "green".to_string(), "blue".to_string()];
    let spec = ToolSpecEntry::new("paint", "paints something").with_parameter(
        ToolParameter::new("color", ToolParameterType::Enum(allowed)).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set color to purple please", &spec)
        .expect("grounding should succeed");

    // "purple" is not in the allowed set, so it must not be grounded as if
    // it were valid.
    assert!(grounded[0].value.is_none());
    assert_eq!(
        grounded[0].status,
        ArgumentGroundingStatus::UngroundedRequired
    );
}

#[test]
fn ground_enum_unanchored() {
    let allowed = vec!["red".to_string(), "green".to_string(), "blue".to_string()];
    let spec = ToolSpecEntry::new("paint", "paints something").with_parameter(ToolParameter::new(
        "color",
        ToolParameterType::Enum(allowed),
    ));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("I think blue looks nice today", &spec)
        .expect("grounding should succeed");

    assert_eq!(grounded[0].value.as_deref(), Some("blue"));
    assert_eq!(grounded[0].confidence, 0.6);
}

#[test]
fn ground_enum_empty_allowed_list_never_grounds() {
    let spec = ToolSpecEntry::new("noop", "does nothing")
        .with_parameter(ToolParameter::new("mode", ToolParameterType::Enum(vec![])));
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("set mode to anything at all", &spec)
        .expect("grounding should succeed");

    assert!(grounded[0].value.is_none());
}

// ── Grounding — multiple parameters together ────────────────────────────────────

#[test]
fn ground_multiple_string_parameters_independently_disambiguated() {
    let spec = ToolSpecEntry::new("plan_trip", "plans a trip")
        .with_parameter(ToolParameter::new("city", ToolParameterType::String).with_required(true))
        .with_parameter(
            ToolParameter::new("country", ToolParameterType::String).with_required(true),
        );
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground(
            r#"set city to "Paris" and country to "France" please"#,
            &spec,
        )
        .expect("grounding should succeed");

    assert_eq!(grounded.len(), 2);
    let city = grounded
        .iter()
        .find(|g| g.parameter_name == "city")
        .expect("city argument present");
    let country = grounded
        .iter()
        .find(|g| g.parameter_name == "country")
        .expect("country argument present");
    // Each parameter must pick the quote *nearest its own mention*, not the
    // other parameter's quote.
    assert_eq!(city.value.as_deref(), Some("Paris"));
    assert_eq!(country.value.as_deref(), Some("France"));
}

#[test]
fn ground_multiple_heterogeneous_parameters_independently() {
    let allowed = vec![
        "economy".to_string(),
        "business".to_string(),
        "first".to_string(),
    ];
    let spec = ToolSpecEntry::new("book_flight", "books a flight")
        .with_parameter(
            ToolParameter::new("destination", ToolParameterType::String).with_required(true),
        )
        .with_parameter(
            ToolParameter::new("passenger_count", ToolParameterType::Number).with_required(true),
        )
        .with_parameter(ToolParameter::new("insurance", ToolParameterType::Boolean))
        .with_parameter(ToolParameter::new(
            "cabin_class",
            ToolParameterType::Enum(allowed),
        ));

    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground(
            r#"book a flight to "Berlin" for 3 passengers with insurance enabled in economy class"#,
            &spec,
        )
        .expect("grounding should succeed");

    assert_eq!(grounded.len(), 4);
    let by_name = |name: &str| -> &GroundedArgument {
        grounded
            .iter()
            .find(|g| g.parameter_name == name)
            .unwrap_or_else(|| panic!("missing grounded argument for {name}"))
    };
    assert_eq!(by_name("destination").value.as_deref(), Some("Berlin"));
    assert_eq!(by_name("passenger_count").value.as_deref(), Some("3"));
    assert_eq!(by_name("insurance").value.as_deref(), Some("true"));
    assert_eq!(by_name("cabin_class").value.as_deref(), Some("economy"));
}

// ── Grounding — errors, edge cases, determinism ─────────────────────────────────

#[test]
fn ground_empty_query_errors() {
    let spec = ToolSpecEntry::new("noop", "does nothing")
        .with_parameter(ToolParameter::new("x", ToolParameterType::String));
    let grounder = ArgumentGrounder::new();
    let result = grounder.ground("   ", &spec);
    assert_eq!(result.unwrap_err(), ToolRetrievalError::EmptyQuery);
}

#[test]
fn ground_spec_with_no_parameters_returns_empty_vec() {
    let spec = ToolSpecEntry::new("noop", "does nothing");
    let grounder = ArgumentGrounder::new();
    let grounded = grounder
        .ground("anything at all", &spec)
        .expect("grounding should succeed");
    assert!(grounded.is_empty());
}

#[test]
fn ground_is_deterministic() {
    let spec = ToolSpecEntry::new("book_travel", "books travel").with_parameter(
        ToolParameter::new("destination", ToolParameterType::String).with_required(true),
    );
    let grounder = ArgumentGrounder::new();
    let first = grounder
        .ground(r#"set the destination to "Berlin" now"#, &spec)
        .expect("ok");
    let second = grounder
        .ground(r#"set the destination to "Berlin" now"#, &spec)
        .expect("ok");
    assert_eq!(first, second);
}

#[test]
fn ground_confidence_quoted_exceeds_fallback_inference() {
    let string_param = |name: &str| ToolParameter::new(name, ToolParameterType::String);

    let quoted_spec = ToolSpecEntry::new("book_travel", "books travel")
        .with_parameter(string_param("destination"));
    let fallback_spec = ToolSpecEntry::new("send_report", "sends a report")
        .with_parameter(string_param("recipient"));

    let grounder = ArgumentGrounder::new();
    let quoted = grounder
        .ground(r#"set the destination to "Berlin" now"#, &quoted_spec)
        .expect("ok");
    let fallback = grounder
        .ground("send the report to Madrid", &fallback_spec)
        .expect("ok");

    assert!(quoted[0].confidence > fallback[0].confidence);
}

#[test]
fn engine_ground_delegates_to_grounder() {
    let spec = ToolSpecEntry::new("book_travel", "books travel").with_parameter(
        ToolParameter::new("destination", ToolParameterType::String).with_required(true),
    );
    let engine = ToolRetrievalEngine::build(vec![spec.clone()], ToolRetrievalConfig::default())
        .expect("builds");
    let grounded = engine
        .ground(r#"set the destination to "Berlin" now"#, &spec)
        .expect("grounding should succeed");
    assert_eq!(grounded[0].value.as_deref(), Some("Berlin"));
}

// ── Error Display ────────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    assert_eq!(
        ToolRetrievalError::EmptyQuery.to_string(),
        "query must not be empty"
    );
    assert_eq!(
        ToolRetrievalError::EmptyRegistry.to_string(),
        "tool registry must not be empty"
    );
    assert_eq!(
        ToolRetrievalError::DuplicateToolName("x".to_string()).to_string(),
        "duplicate tool name in registry: x"
    );
}

// ── ToolMatch struct sanity ────────────────────────────────────────────────────

#[test]
fn tool_match_fields_are_populated() {
    let tools = vec![ToolSpecEntry::new("alpha", "converts currency")];
    let config = ToolRetrievalConfig::default().with_min_match_score(0.0);
    let engine = ToolRetrievalEngine::build(tools, config).expect("builds");
    let matches: Vec<ToolMatch> = engine.retrieve("convert currency", 5).expect("ok");
    assert_eq!(matches[0].tool_name, "alpha");
}
