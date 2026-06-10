//! `ReActAgent` — the `ReAct` Thought→Action→Observation agent loop.

use crate::types::SearchResult;

use super::tool::ToolRegistry;
use super::types::{AgentAction, AgentStep, AgentTrace, AgenticConfig, AgenticError};

// ── helpers ───────────────────────────────────────────────────────────────────

/// Strip a leading verb prefix from a query before passing it to a tool.
///
/// E.g. `"calculate 3 + 4"` → `"3 + 4"`, `"lookup capital"` → `"capital"`.
fn strip_trigger_prefix(query: &str, tool_name_lower: &str) -> String {
    let q = query.trim();
    let lower = q.to_lowercase();
    // Try exact tool name prefix ("calculator 3 + 4")
    for prefix in &[
        format!("{tool_name_lower} "),
        // common verb forms: "calculate", "compute", "evaluate", "lookup", "look up"
        "calculate ".to_string(),
        "compute ".to_string(),
        "evaluate ".to_string(),
        "look up ".to_string(),
        "lookup ".to_string(),
    ] {
        if let Some(rest_lower) = lower.strip_prefix(prefix.as_str()) {
            let stripped = q[q.len() - rest_lower.len()..].trim();
            if !stripped.is_empty() {
                return stripped.to_string();
            }
        }
    }
    q.to_string()
}

// ── heuristic action planner ──────────────────────────────────────────────────

/// Decide the next action based on the current state.
///
/// The heuristic works in priority order:
/// 1. If search results are available and a relevant tool is registered → `UseTool`.
/// 2. If query contains question words and no docs yet → `Search`.
/// 3. If enough information has been gathered (step >= half of max) → `Finish`.
/// 4. Default → `Search`.
fn plan_action(
    query: &str,
    step_index: usize,
    max_steps: usize,
    previous_observations: &[String],
    search_results: &[SearchResult],
    registry: &ToolRegistry,
) -> (String, AgentAction) {
    let lower = query.to_lowercase();

    // If there are tool hits matching query keywords, try a tool first
    let tool_list = registry.list_tools();
    for (tool_name, _desc) in &tool_list {
        let tool_lower = tool_name.to_lowercase();
        if lower.contains(&tool_lower)
            || tool_lower.contains("calculator") && lower.contains("calculate")
        {
            let thought =
                format!("The query mentions '{tool_name}'. I should use the {tool_name} tool.");
            // Strip leading verb triggers so "calculate 3 + 4" becomes "3 + 4"
            let tool_input = strip_trigger_prefix(query, &tool_lower);
            return (
                thought,
                AgentAction::UseTool {
                    name: tool_name.to_string(),
                    input: tool_input,
                },
            );
        }
    }

    // If we've gathered observations and have results, finish
    let has_info = !previous_observations.is_empty() || !search_results.is_empty();
    let halfway = step_index >= max_steps / 2;
    if has_info && halfway {
        let answer = if search_results.is_empty() {
            previous_observations
                .last()
                .cloned()
                .unwrap_or_else(|| "No information found.".to_string())
        } else {
            search_results
                .iter()
                .take(3)
                .map(|r| r.document.content.chars().take(120).collect::<String>())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let thought = "I have gathered enough information to answer.".to_string();
        return (thought, AgentAction::Finish(answer));
    }

    // Default: search
    let thought = format!("I need to search for information about: {query}");
    (thought, AgentAction::Search(query.to_string()))
}

// ── ReActAgent ────────────────────────────────────────────────────────────────

/// `ReAct` (Reasoning + Acting) agent with a configurable tool registry.
///
/// `E` is the [`Echo`] layer provided *per call* through `run<E>`.
///
/// [`Echo`]: crate::layer1_echo::traits::Echo
pub struct ReActAgent {
    /// Registry of tools available to the agent.
    pub registry: ToolRegistry,
    /// Agent configuration.
    pub config: AgenticConfig,
}

impl ReActAgent {
    /// Create a new [`ReActAgent`] with the given registry and config.
    #[must_use]
    pub fn new(registry: ToolRegistry, config: AgenticConfig) -> Self {
        Self { registry, config }
    }

    /// Run the `ReAct` loop for `query`, optionally using Echo for searches.
    ///
    /// # Errors
    ///
    /// Returns [`AgenticError::EmptyQuery`] when `query` is blank.
    /// Returns [`AgenticError::RetrievalFailed`] when Echo search fails.
    /// Returns [`AgenticError::ToolNotFound`] when an action requests an unknown tool.
    /// Returns [`AgenticError::ToolFailed`] when a tool invocation fails.
    pub async fn run<E>(&self, query: &str, echo: &E) -> Result<AgentTrace, AgenticError>
    where
        E: crate::layer1_echo::traits::Echo + ?Sized,
    {
        let query_trimmed = query.trim();
        if query_trimmed.is_empty() {
            return Err(AgenticError::EmptyQuery);
        }

        let mut trace = AgentTrace::new();
        let mut observations: Vec<String> = Vec::new();
        let mut search_results: Vec<SearchResult> = Vec::new();

        for step_index in 0..self.config.max_steps {
            let (thought, action) = plan_action(
                query_trimmed,
                step_index,
                self.config.max_steps,
                &observations,
                &search_results,
                &self.registry,
            );

            let observation = match &action {
                AgentAction::Search(q) => {
                    let results = echo
                        .search(q, self.config.top_k, None)
                        .await
                        .map_err(|e| AgenticError::RetrievalFailed(e.to_string()))?;

                    let obs = if results.is_empty() {
                        "No relevant documents found.".to_string()
                    } else if self.config.include_snippets {
                        results
                            .iter()
                            .take(3)
                            .map(|r| {
                                let snippet: String =
                                    r.document.content.chars().take(150).collect();
                                format!("[{:.2}] {snippet}", r.score)
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        format!("Found {} documents.", results.len())
                    };
                    search_results = results;
                    obs
                }
                AgentAction::UseTool { name, input } => {
                    let tool = self
                        .registry
                        .get(name)
                        .ok_or_else(|| AgenticError::ToolNotFound(name.clone()))?;
                    tool.invoke(input).await?
                }
                AgentAction::Finish(answer) => {
                    // Record the final step, then return
                    let step = AgentStep {
                        thought,
                        action: action.clone(),
                        observation: String::new(),
                        step_index,
                    };
                    trace.steps.push(step);
                    trace.final_answer = answer.clone();
                    return Ok(trace);
                }
            };

            observations.push(observation.clone());
            trace.steps.push(AgentStep {
                thought,
                action,
                observation,
                step_index,
            });
        }

        // Max steps reached — synthesize best answer from observations
        trace.truncated = true;
        trace.final_answer = if search_results.is_empty() {
            observations
                .last()
                .cloned()
                .unwrap_or_else(|| "No information found.".to_string())
        } else {
            search_results
                .iter()
                .take(2)
                .map(|r| r.document.content.chars().take(200).collect::<String>())
                .collect::<Vec<_>>()
                .join(" ")
        };

        Ok(trace)
    }
}
