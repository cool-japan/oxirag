//! Tests for the `query_planning` module.

use super::executor::{PlanExecutor, topo_sort};
use super::planner::QueryPlanner;
use super::types::{PlanStep, PlanStepKind, QueryPlan, QueryPlanningError, SynthesisStrategy};
use crate::error::EmbeddingError;
use crate::layer1_echo::traits::Echo;
use crate::types::{Document, DocumentId, SearchResult};

// ── MockEcho ──────────────────────────────────────────────────────────────────

struct MockEcho {
    results: Vec<SearchResult>,
}

impl MockEcho {
    fn new(results: Vec<SearchResult>) -> Self {
        Self { results }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
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

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
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

fn make_result(id: &str, content: &str, score: f32) -> SearchResult {
    SearchResult {
        document: Document::new(content).with_id(DocumentId::from_string(id)),
        score,
        rank: 0,
    }
}

// ── PlanStepKind ──────────────────────────────────────────────────────────────

#[test]
fn test_plan_step_kind_as_str() {
    assert_eq!(PlanStepKind::Retrieve.as_str(), "retrieve");
    assert_eq!(PlanStepKind::Filter.as_str(), "filter");
    assert_eq!(PlanStepKind::Aggregate.as_str(), "aggregate");
    assert_eq!(PlanStepKind::Compare.as_str(), "compare");
    assert_eq!(PlanStepKind::Verify.as_str(), "verify");
}

// ── PlanStep ──────────────────────────────────────────────────────────────────

#[test]
fn test_plan_step_is_root() {
    let root = PlanStep::new(0, PlanStepKind::Retrieve, "query");
    assert!(root.is_root());

    let dep = PlanStep::new(1, PlanStepKind::Aggregate, "query").with_depends_on(vec![0]);
    assert!(!dep.is_root());
}

// ── SynthesisStrategy ─────────────────────────────────────────────────────────

#[test]
fn test_synthesis_strategy_as_str() {
    assert_eq!(SynthesisStrategy::Simple.as_str(), "simple");
    assert_eq!(SynthesisStrategy::Comparative.as_str(), "comparative");
    assert_eq!(SynthesisStrategy::Aggregative.as_str(), "aggregative");
}

#[test]
fn test_synthesis_default() {
    let s = SynthesisStrategy::default();
    assert_eq!(s, SynthesisStrategy::Simple);
}

// ── QueryPlan ─────────────────────────────────────────────────────────────────

#[test]
fn test_query_plan_is_empty() {
    let empty = QueryPlan::new(vec![], SynthesisStrategy::Simple);
    assert!(empty.is_empty());

    let non_empty = QueryPlan::new(
        vec![PlanStep::new(0, PlanStepKind::Retrieve, "q")],
        SynthesisStrategy::Simple,
    );
    assert!(!non_empty.is_empty());
}

#[test]
fn test_query_plan_root_steps() {
    let plan = QueryPlan::new(
        vec![
            PlanStep::new(0, PlanStepKind::Retrieve, "a"),
            PlanStep::new(1, PlanStepKind::Retrieve, "b"),
            PlanStep::new(2, PlanStepKind::Aggregate, "c").with_depends_on(vec![0, 1]),
        ],
        SynthesisStrategy::Aggregative,
    );
    let roots = plan.root_steps();
    assert_eq!(roots.len(), 2);
    assert!(roots.iter().any(|s| s.id == 0));
    assert!(roots.iter().any(|s| s.id == 1));
}

#[test]
fn test_query_plan_max_parallelism() {
    let plan = QueryPlan::new(
        vec![
            PlanStep::new(0, PlanStepKind::Retrieve, "a"),
            PlanStep::new(1, PlanStepKind::Retrieve, "b"),
            PlanStep::new(2, PlanStepKind::Compare, "c").with_depends_on(vec![0, 1]),
        ],
        SynthesisStrategy::Comparative,
    );
    assert_eq!(plan.max_parallelism(), 2);
}

// ── QueryPlanner ──────────────────────────────────────────────────────────────

#[test]
fn test_planner_empty_query_error() {
    let planner = QueryPlanner::new();
    let err = planner.plan("   ");
    assert!(matches!(err, Err(QueryPlanningError::EmptyQuery)));
}

#[test]
fn test_planner_simple_query() {
    let planner = QueryPlanner::new();
    let plan = planner
        .plan("what is machine learning")
        .expect("plan should succeed");
    assert_eq!(plan.step_count(), 1);
    assert_eq!(plan.steps[0].kind, PlanStepKind::Retrieve);
    assert_eq!(plan.synthesis_strategy, SynthesisStrategy::Simple);
}

#[test]
fn test_planner_comparative_query() {
    let planner = QueryPlanner::new();
    let plan = planner
        .plan("compare Python vs Rust")
        .expect("plan should succeed");
    assert_eq!(plan.step_count(), 3);
    assert_eq!(plan.steps[0].kind, PlanStepKind::Retrieve);
    assert_eq!(plan.steps[1].kind, PlanStepKind::Retrieve);
    assert_eq!(plan.steps[2].kind, PlanStepKind::Compare);
    assert_eq!(plan.synthesis_strategy, SynthesisStrategy::Comparative);
}

#[test]
fn test_planner_aggregative_query() {
    let planner = QueryPlanner::new();
    let plan = planner
        .plan("how many documents are in the index")
        .expect("plan should succeed");
    assert_eq!(plan.step_count(), 2);
    assert_eq!(plan.steps[0].kind, PlanStepKind::Retrieve);
    assert_eq!(plan.steps[1].kind, PlanStepKind::Aggregate);
    assert_eq!(plan.synthesis_strategy, SynthesisStrategy::Aggregative);
}

#[test]
fn test_planner_multi_step_and_query() {
    let planner = QueryPlanner::new();
    let plan = planner
        .plan("Rust safety and performance")
        .expect("plan should succeed");
    // "and" triggers multi-step: 2 Retrieve + 1 Aggregate
    assert!(plan.step_count() >= 2);
    assert!(plan.steps.iter().any(|s| s.kind == PlanStepKind::Aggregate));
    assert_eq!(plan.synthesis_strategy, SynthesisStrategy::Aggregative);
}

#[test]
fn test_planner_verify_query() {
    let planner = QueryPlanner::new();
    let plan = planner
        .plan("does Rust prevent data races")
        .expect("plan should succeed");
    assert_eq!(plan.step_count(), 2);
    assert_eq!(plan.steps[0].kind, PlanStepKind::Retrieve);
    assert_eq!(plan.steps[1].kind, PlanStepKind::Verify);
}

// ── PlanExecutor ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_executor_empty_plan() {
    let executor = PlanExecutor::new();
    let plan = QueryPlan::new(vec![], SynthesisStrategy::Simple);
    let echo = MockEcho::new(vec![]);
    let results = executor
        .run(&plan, &echo)
        .await
        .expect("run should succeed");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_executor_single_retrieve() {
    let executor = PlanExecutor::new();
    let plan = QueryPlan::new(
        vec![PlanStep::new(0, PlanStepKind::Retrieve, "what is Rust")],
        SynthesisStrategy::Simple,
    );
    let echo = MockEcho::new(vec![
        make_result("d1", "Rust is a systems language", 0.9),
        make_result("d2", "Rust prevents data races", 0.8),
    ]);
    let results = executor
        .run(&plan, &echo)
        .await
        .expect("run should succeed");
    assert_eq!(results.len(), 1);
    assert!(results[0].answer.contains("Rust"));
}

#[tokio::test]
async fn test_executor_comparative_plan() {
    let executor = PlanExecutor::new();
    let plan = QueryPlan::new(
        vec![
            PlanStep::new(0, PlanStepKind::Retrieve, "Python features"),
            PlanStep::new(1, PlanStepKind::Retrieve, "Rust features"),
            PlanStep::new(2, PlanStepKind::Compare, "compare").with_depends_on(vec![0, 1]),
        ],
        SynthesisStrategy::Comparative,
    );
    let echo = MockEcho::new(vec![make_result("d1", "some content", 0.9)]);
    let results = executor
        .run(&plan, &echo)
        .await
        .expect("run should succeed");
    assert_eq!(results.len(), 3);
    let compare_result = results
        .iter()
        .find(|r| r.step_id == 2)
        .expect("compare result expected");
    assert!(compare_result.answer.contains("vs") || compare_result.answer.contains("A:"));
}

#[tokio::test]
async fn test_executor_aggregative_plan() {
    let executor = PlanExecutor::new();
    let plan = QueryPlan::new(
        vec![
            PlanStep::new(0, PlanStepKind::Retrieve, "count documents"),
            PlanStep::new(1, PlanStepKind::Aggregate, "total").with_depends_on(vec![0]),
        ],
        SynthesisStrategy::Aggregative,
    );
    let echo = MockEcho::new(vec![make_result("d1", "there are 5 documents", 0.9)]);
    let results = executor
        .run(&plan, &echo)
        .await
        .expect("run should succeed");
    assert_eq!(results.len(), 2);
    assert!(!results[1].answer.is_empty());
}

#[test]
fn test_executor_cyclic_plan_error() {
    let steps = vec![
        PlanStep::new(0, PlanStepKind::Retrieve, "a").with_depends_on(vec![1]),
        PlanStep::new(1, PlanStepKind::Aggregate, "b").with_depends_on(vec![0]),
    ];
    let err = topo_sort(&steps);
    assert!(matches!(err, Err(QueryPlanningError::CyclicDependency)));
}

// ── topo_sort ─────────────────────────────────────────────────────────────────

#[test]
fn test_topo_sort_linear() {
    let steps = vec![
        PlanStep::new(0, PlanStepKind::Retrieve, "q"),
        PlanStep::new(1, PlanStepKind::Aggregate, "q").with_depends_on(vec![0]),
    ];
    let order = topo_sort(&steps).expect("sort should succeed");
    assert_eq!(order.len(), 2);
    // 0 must come before 1
    let pos0 = order.iter().position(|&id| id == 0).unwrap();
    let pos1 = order.iter().position(|&id| id == 1).unwrap();
    assert!(pos0 < pos1);
}

#[test]
fn test_topo_sort_diamond() {
    // 0 → 1, 0 → 2, 1 → 3, 2 → 3
    let steps = vec![
        PlanStep::new(0, PlanStepKind::Retrieve, "a"),
        PlanStep::new(1, PlanStepKind::Retrieve, "b").with_depends_on(vec![0]),
        PlanStep::new(2, PlanStepKind::Retrieve, "c").with_depends_on(vec![0]),
        PlanStep::new(3, PlanStepKind::Aggregate, "d").with_depends_on(vec![1, 2]),
    ];
    let order = topo_sort(&steps).expect("sort should succeed");
    assert_eq!(order.len(), 4);
    let pos = |id: usize| order.iter().position(|&x| x == id).unwrap();
    assert!(pos(0) < pos(1));
    assert!(pos(0) < pos(2));
    assert!(pos(1) < pos(3));
    assert!(pos(2) < pos(3));
}

// ── Error display ─────────────────────────────────────────────────────────────

#[test]
fn test_error_display() {
    assert_eq!(
        QueryPlanningError::EmptyQuery.to_string(),
        "Query must not be empty"
    );
    assert_eq!(
        QueryPlanningError::CyclicDependency.to_string(),
        "Cyclic dependency detected in plan"
    );
    assert_eq!(
        QueryPlanningError::StepFailed("oops".to_string()).to_string(),
        "Step execution failed: oops"
    );
}
