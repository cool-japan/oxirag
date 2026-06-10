//! Tests for the `agentic` module.

use std::collections::HashMap;

use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

use super::agent::ReActAgent;
use super::tool::{CalculatorTool, LookupTool, MockTool, Tool, ToolRegistry};
use super::types::{AgenticConfig, AgenticError};

use async_trait::async_trait;

// ── MockEcho ──────────────────────────────────────────────────────────────────

struct MockEcho {
    results: Vec<SearchResult>,
}

impl MockEcho {
    fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl Echo for MockEcho {
    async fn index(&mut self, document: Document) -> Result<DocumentId, EmbeddingError> {
        Ok(document.id.clone())
    }

    async fn index_batch(
        &mut self,
        documents: Vec<Document>,
    ) -> Result<Vec<DocumentId>, EmbeddingError> {
        Ok(documents.into_iter().map(|d| d.id).collect())
    }

    async fn search(
        &self,
        _query: &str,
        _top_k: usize,
        _min_score: Option<f32>,
    ) -> Result<Vec<SearchResult>, EmbeddingError> {
        Ok(self.results.clone())
    }

    async fn get(&self, _id: &DocumentId) -> Result<Option<Document>, EmbeddingError> {
        Ok(None)
    }

    async fn delete(&mut self, _id: &DocumentId) -> Result<bool, EmbeddingError> {
        Ok(false)
    }

    async fn count(&self) -> usize {
        self.results.len()
    }

    async fn clear(&mut self) -> Result<(), EmbeddingError> {
        Ok(())
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

fn empty_registry() -> ToolRegistry {
    ToolRegistry::new()
}

// ── AgentAction tests ─────────────────────────────────────────────────────────

#[test]
fn test_agent_action_variants() {
    let a1 = super::types::AgentAction::Search("rust".to_string());
    let a2 = super::types::AgentAction::UseTool {
        name: "calc".to_string(),
        input: "1 + 2".to_string(),
    };
    let a3 = super::types::AgentAction::Finish("done".to_string());
    matches!(a1, super::types::AgentAction::Search(_));
    matches!(a2, super::types::AgentAction::UseTool { .. });
    matches!(a3, super::types::AgentAction::Finish(_));
}

// ── AgentTrace tests ──────────────────────────────────────────────────────────

#[test]
fn test_agent_trace_new() {
    let trace = super::types::AgentTrace::new();
    assert!(trace.is_empty());
    assert_eq!(trace.len(), 0);
    assert!(!trace.truncated);
}

#[test]
fn test_agent_trace_default() {
    let trace = super::types::AgentTrace::default();
    assert!(trace.is_empty());
}

// ── AgenticConfig tests ───────────────────────────────────────────────────────

#[test]
fn test_agentic_config_defaults() {
    let cfg = AgenticConfig::default();
    assert_eq!(cfg.max_steps, 6);
    assert_eq!(cfg.top_k, 5);
    assert!(cfg.include_snippets);
}

#[test]
fn test_agentic_config_builders() {
    let cfg = AgenticConfig::default()
        .with_max_steps(3)
        .with_top_k(10)
        .with_include_snippets(false);
    assert_eq!(cfg.max_steps, 3);
    assert_eq!(cfg.top_k, 10);
    assert!(!cfg.include_snippets);
}

// ── ToolRegistry tests ────────────────────────────────────────────────────────

#[test]
fn test_tool_registry_empty() {
    let reg = ToolRegistry::new();
    assert!(reg.is_empty());
    assert_eq!(reg.len(), 0);
    assert!(reg.get("nonexistent").is_none());
}

#[test]
fn test_tool_registry_register_and_get() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(MockTool::new("alpha", "result")));
    assert_eq!(reg.len(), 1);
    assert!(reg.get("alpha").is_some());
    assert!(reg.get("beta").is_none());
}

#[test]
fn test_tool_registry_list_tools() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(MockTool::new("b_tool", "r")));
    reg.register(Box::new(MockTool::new("a_tool", "r")));
    let list = reg.list_tools();
    assert_eq!(list.len(), 2);
    // Sorted by name
    assert_eq!(list[0].0, "a_tool");
    assert_eq!(list[1].0, "b_tool");
}

#[test]
fn test_tool_registry_replace() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(MockTool::new("t", "first")));
    reg.register(Box::new(MockTool::new("t", "second")));
    assert_eq!(reg.len(), 1);
}

// ── CalculatorTool tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_calculator_addition() {
    let tool = CalculatorTool::new();
    let result = tool.invoke("3 + 4").await.expect("ok");
    assert_eq!(result.trim(), "7");
}

#[tokio::test]
async fn test_calculator_subtraction() {
    let tool = CalculatorTool::new();
    let result = tool.invoke("10 - 3").await.expect("ok");
    assert_eq!(result.trim(), "7");
}

#[tokio::test]
async fn test_calculator_multiplication() {
    let tool = CalculatorTool::new();
    let result = tool.invoke("6 * 7").await.expect("ok");
    assert_eq!(result.trim(), "42");
}

#[tokio::test]
async fn test_calculator_division() {
    let tool = CalculatorTool::new();
    let result = tool.invoke("15 / 3").await.expect("ok");
    assert_eq!(result.trim(), "5");
}

#[tokio::test]
async fn test_calculator_division_by_zero() {
    let tool = CalculatorTool::new();
    let err = tool.invoke("5 / 0").await.expect_err("should fail");
    assert!(matches!(err, AgenticError::ToolFailed(_)));
}

#[tokio::test]
async fn test_calculator_bad_format() {
    let tool = CalculatorTool::new();
    let err = tool.invoke("5").await.expect_err("should fail");
    assert!(matches!(err, AgenticError::ToolFailed(_)));
}

#[tokio::test]
async fn test_calculator_unknown_op() {
    let tool = CalculatorTool::new();
    let err = tool.invoke("5 ^ 2").await.expect_err("should fail");
    assert!(matches!(err, AgenticError::ToolFailed(_)));
}

// ── LookupTool tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lookup_tool_found() {
    let mut table = HashMap::new();
    table.insert(
        "rust".to_string(),
        "A systems programming language.".to_string(),
    );
    let tool = LookupTool::new(table);
    let result = tool.invoke("rust").await.expect("ok");
    assert!(result.contains("systems"));
}

#[tokio::test]
async fn test_lookup_tool_case_insensitive() {
    let mut table = HashMap::new();
    table.insert("rust".to_string(), "Rust language".to_string());
    let tool = LookupTool::new(table);
    let result = tool.invoke("Rust").await.expect("ok");
    assert!(result.contains("Rust"));
}

#[tokio::test]
async fn test_lookup_tool_not_found() {
    let tool = LookupTool::new(HashMap::new());
    let err = tool.invoke("unknown").await.expect_err("should fail");
    assert!(matches!(err, AgenticError::ToolFailed(_)));
}

// ── MockTool tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mock_tool_name_and_output() {
    let tool = MockTool::new("my_tool", "my_output");
    assert_eq!(tool.name(), "my_tool");
    let result = tool.invoke("any input").await.expect("ok");
    assert_eq!(result, "my_output");
}

// ── ReActAgent tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_empty_query() {
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = agent.run("", &echo).await.expect_err("should fail");
    assert!(matches!(err, AgenticError::EmptyQuery));
}

#[tokio::test]
async fn test_agent_whitespace_query() {
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default());
    let echo = MockEcho::new(vec![]);
    let err = agent.run("   ", &echo).await.expect_err("should fail");
    assert!(matches!(err, AgenticError::EmptyQuery));
}

#[tokio::test]
async fn test_agent_runs_to_completion() {
    let docs = vec![
        make_result("d1", "Rust is a systems language", 0.9),
        make_result("d2", "Rust focuses on safety", 0.8),
    ];
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default().with_max_steps(4));
    let echo = MockEcho::new(docs);
    let trace = agent
        .run("what is rust", &echo)
        .await
        .expect("should succeed");
    assert!(!trace.final_answer.is_empty(), "should have final answer");
}

#[tokio::test]
async fn test_agent_max_steps_produces_trace() {
    let docs = vec![make_result("d1", "some content", 0.9)];
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default().with_max_steps(2));
    let echo = MockEcho::new(docs);
    let trace = agent.run("query", &echo).await.expect("should succeed");
    assert!(!trace.final_answer.is_empty());
}

#[tokio::test]
async fn test_agent_with_calculator_tool() {
    let mut reg = ToolRegistry::new();
    reg.register(Box::new(CalculatorTool::new()));
    let agent = ReActAgent::new(reg, AgenticConfig::default().with_max_steps(4));
    let echo = MockEcho::new(vec![]);
    let trace = agent
        .run("calculate 3 + 4", &echo)
        .await
        .expect("should succeed");
    assert!(!trace.final_answer.is_empty());
}

#[tokio::test]
async fn test_agent_no_results_gives_answer() {
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default().with_max_steps(4));
    let echo = MockEcho::new(vec![]); // no docs
    let trace = agent
        .run("query about nothing", &echo)
        .await
        .expect("should succeed");
    assert!(
        !trace.final_answer.is_empty(),
        "should have a fallback answer"
    );
}

#[tokio::test]
async fn test_agent_trace_steps_recorded() {
    let docs = vec![make_result("d1", "content", 0.8)];
    let agent = ReActAgent::new(empty_registry(), AgenticConfig::default().with_max_steps(6));
    let echo = MockEcho::new(docs);
    let trace = agent.run("what is something", &echo).await.expect("ok");
    assert!(!trace.steps.is_empty(), "should record steps");
    for (i, step) in trace.steps.iter().enumerate() {
        assert_eq!(step.step_index, i, "step indices should be sequential");
    }
}
