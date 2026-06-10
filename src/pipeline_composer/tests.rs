//! Tests for the `pipeline_composer` module.

use super::composer::ComposedPipeline;
use super::stage::{
    FormatStage, KeywordFilterStage, PassThroughStage, SanitizerStage, TruncateStage,
};
use super::types::{ComposerError, PipelineStage, StageInput, StageOutput};

// ── StageInput ────────────────────────────────────────────────────────────────

#[test]
fn test_stage_input_new() {
    let input = StageInput::new("my query", "some context");
    assert_eq!(input.query, "my query");
    assert_eq!(input.context, "some context");
    assert!(input.metadata.is_empty());
}

#[test]
fn test_stage_input_with_metadata() {
    let input = StageInput::new("q", "ctx")
        .with_metadata("key1", "val1")
        .with_metadata("key2", "val2");
    assert_eq!(input.metadata.get("key1"), Some(&"val1".to_string()));
    assert_eq!(input.metadata.get("key2"), Some(&"val2".to_string()));
}

// ── StageOutput ───────────────────────────────────────────────────────────────

#[test]
fn test_stage_output_pass() {
    let out = StageOutput::pass("hello world");
    assert_eq!(out.content, "hello world");
    assert!(!out.blocked);
    assert!(!out.is_blocked());
}

#[test]
fn test_stage_output_block() {
    let out = StageOutput::block("not allowed");
    assert!(out.blocked);
    assert!(out.is_blocked());
    assert_eq!(
        out.metadata.get("block_reason"),
        Some(&"not allowed".to_string())
    );
}

#[test]
fn test_stage_output_is_blocked() {
    let pass = StageOutput::pass("ok");
    let block = StageOutput::block("no");
    assert!(!pass.is_blocked());
    assert!(block.is_blocked());
}

// ── PassThroughStage ──────────────────────────────────────────────────────────

#[test]
fn test_passthrough_stage() {
    let stage = PassThroughStage::new("my-pass");
    assert_eq!(stage.name(), "my-pass");
    let input = StageInput::new("q", "original context");
    let output = stage.process(input).expect("stage should succeed");
    assert_eq!(output.content, "original context");
    assert!(!output.is_blocked());
}

// ── FormatStage ───────────────────────────────────────────────────────────────

#[test]
fn test_format_stage() {
    let stage = FormatStage::new("[START] ", " [END]");
    let input = StageInput::new("q", "middle");
    let output = stage.process(input).expect("stage should succeed");
    assert_eq!(output.content, "[START] middle [END]");
    assert!(!output.is_blocked());
}

// ── TruncateStage ─────────────────────────────────────────────────────────────

#[test]
fn test_truncate_stage() {
    let stage = TruncateStage::new(5);
    let input = StageInput::new("q", "hello world");
    let output = stage.process(input).expect("stage should succeed");
    assert_eq!(output.content, "hello");
    assert!(!output.is_blocked());
}

// ── KeywordFilterStage ────────────────────────────────────────────────────────

#[test]
fn test_keyword_filter_stage_pass() {
    let stage = KeywordFilterStage::new(vec!["badword".to_string()]);
    let input = StageInput::new("q", "this is fine content");
    let output = stage.process(input).expect("stage should succeed");
    assert!(!output.is_blocked());
    assert_eq!(output.content, "this is fine content");
}

#[test]
fn test_keyword_filter_stage_block() {
    let stage = KeywordFilterStage::new(vec!["badword".to_string(), "evil".to_string()]);
    let input = StageInput::new("q", "this contains evil content");
    let output = stage.process(input).expect("stage should succeed");
    assert!(output.is_blocked());
    assert!(
        output
            .metadata
            .get("block_reason")
            .unwrap()
            .contains("evil")
    );
}

#[test]
fn test_keyword_filter_case_insensitive() {
    let stage = KeywordFilterStage::new(vec!["BadWord".to_string()]).with_case_insensitive(true);
    let input = StageInput::new("q", "this contains badword");
    let output = stage.process(input).expect("stage should succeed");
    assert!(output.is_blocked());
}

// ── SanitizerStage ────────────────────────────────────────────────────────────

#[test]
fn test_sanitizer_stage() {
    let stage = SanitizerStage::new();
    let input = StageInput::new("q", "  hello    world   ");
    let output = stage.process(input).expect("stage should succeed");
    assert_eq!(output.content, "hello world");
    assert!(!output.is_blocked());
}

// ── ComposedPipeline ──────────────────────────────────────────────────────────

#[test]
fn test_composer_empty_pipeline_error() {
    let pipeline = ComposedPipeline::new();
    let input = StageInput::new("q", "ctx");
    let err = pipeline.run(input);
    assert!(matches!(err, Err(ComposerError::EmptyPipeline)));
}

#[test]
fn test_composer_single_stage() {
    let pipeline = ComposedPipeline::new().add_stage(Box::new(PassThroughStage::new("p")));
    let input = StageInput::new("q", "hello");
    let output = pipeline.run(input).expect("pipeline should succeed");
    assert_eq!(output.content, "hello");
    assert!(!output.is_blocked());
}

#[test]
fn test_composer_chain_stages() {
    let pipeline = ComposedPipeline::new()
        .add_stage(Box::new(SanitizerStage::new()))
        .add_stage(Box::new(TruncateStage::new(10)))
        .add_stage(Box::new(FormatStage::new("[", "]")));

    let input = StageInput::new("q", "  hello    world   ");
    let output = pipeline.run(input).expect("pipeline should succeed");
    // After sanitise: "hello world"
    // After truncate(10): "hello worl"
    // After format: "[hello worl]"
    assert_eq!(output.content, "[hello worl]");
}

#[test]
fn test_composer_stop_on_block_true() {
    let pipeline = ComposedPipeline::new()
        .with_stop_on_block(true)
        .add_stage(Box::new(KeywordFilterStage::new(vec!["bad".to_string()])))
        .add_stage(Box::new(FormatStage::new("[", "]")));

    let input = StageInput::new("q", "contains bad content");
    let output = pipeline.run(input).expect("pipeline should succeed");
    assert!(output.is_blocked());
    // FormatStage should NOT have run — content is empty (blocked output)
    assert_eq!(output.content, "");
}

#[test]
fn test_composer_stop_on_block_false_continues() {
    // When stop_on_block=false, blocked outputs are treated as pass-throughs
    // that propagate the (empty) content forward.
    let pipeline = ComposedPipeline::new()
        .with_stop_on_block(false)
        .add_stage(Box::new(KeywordFilterStage::new(vec!["bad".to_string()])))
        .add_stage(Box::new(FormatStage::new("[", "]")));

    let input = StageInput::new("q", "contains bad content");
    let output = pipeline.run(input).expect("pipeline should succeed");
    // Pipeline continues: FormatStage wraps the empty blocked content
    assert_eq!(output.content, "[]");
    assert!(!output.is_blocked()); // final stage did not block
}

#[test]
fn test_composer_stage_count() {
    let pipeline = ComposedPipeline::new()
        .add_stage(Box::new(PassThroughStage::new("a")))
        .add_stage(Box::new(PassThroughStage::new("b")));
    assert_eq!(pipeline.stage_count(), 2);
}

#[test]
fn test_composer_stage_metadata_passes_through() {
    struct MetaStage;
    #[allow(clippy::unnecessary_literal_bound)]
    impl PipelineStage for MetaStage {
        fn name(&self) -> &str {
            "meta"
        }
        fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
            Ok(StageOutput::pass(input.context).with_metadata("added_by", "meta_stage"))
        }
    }

    struct CheckStage;
    #[allow(clippy::unnecessary_literal_bound)]
    impl PipelineStage for CheckStage {
        fn name(&self) -> &str {
            "check"
        }
        fn process(&self, input: StageInput) -> Result<StageOutput, ComposerError> {
            let found = input.metadata.contains_key("added_by");
            Ok(StageOutput::pass(format!("found_meta={found}")))
        }
    }

    let pipeline = ComposedPipeline::new()
        .add_stage(Box::new(MetaStage))
        .add_stage(Box::new(CheckStage));

    let input = StageInput::new("q", "data");
    let output = pipeline.run(input).expect("pipeline should succeed");
    assert_eq!(output.content, "found_meta=true");
}

#[test]
fn test_composer_add_stage_builder() {
    let pipeline = ComposedPipeline::new();
    assert_eq!(pipeline.stage_count(), 0);
    assert!(pipeline.is_empty());

    let pipeline = pipeline.add_stage(Box::new(PassThroughStage::new("first")));
    assert_eq!(pipeline.stage_count(), 1);
    assert!(!pipeline.is_empty());
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        ComposerError::StageBlocked("foo".to_string()).to_string(),
        "Stage 'foo' blocked the pipeline"
    );
    assert_eq!(
        ComposerError::StageFailed("bar".to_string(), "oops".to_string()).to_string(),
        "Stage 'bar' failed: oops"
    );
    assert_eq!(
        ComposerError::EmptyPipeline.to_string(),
        "Pipeline has no stages"
    );
}
