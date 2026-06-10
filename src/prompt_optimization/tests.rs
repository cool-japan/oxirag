use crate::prompt_optimization::optimizer::PromptOptimizer;
use crate::prompt_optimization::types::{
    DemoPool, DemoSelectionStrategy, DemoSelector, Demonstration, DevExample, OutputScorer,
    PromptOptimizationConfig, PromptOptimizationError, PromptVariant, VariantScore,
};
use std::collections::HashSet;

// ── JaccardScorer ─────────────────────────────────────────────────────────

struct JaccardScorer;
impl OutputScorer for JaccardScorer {
    fn score(&self, produced: &str, expected: &str) -> f32 {
        let ta: HashSet<&str> = produced.split_whitespace().collect();
        let tb: HashSet<&str> = expected.split_whitespace().collect();
        if ta.is_empty() && tb.is_empty() {
            return 1.0;
        }
        let i = ta.intersection(&tb).count();
        let u = ta.union(&tb).count();
        #[allow(clippy::cast_precision_loss)]
        if u == 0 { 0.0 } else { i as f32 / u as f32 }
    }
}

// ── DemoSelectionStrategy ─────────────────────────────────────────────────

#[test]
fn demo_selection_strategy_default_is_knn() {
    assert_eq!(DemoSelectionStrategy::default(), DemoSelectionStrategy::Knn);
}

#[test]
fn demo_selection_strategy_variants_equality() {
    assert_eq!(DemoSelectionStrategy::Knn, DemoSelectionStrategy::Knn);
    assert_eq!(
        DemoSelectionStrategy::MmrDiverse,
        DemoSelectionStrategy::MmrDiverse
    );
    assert_eq!(
        DemoSelectionStrategy::Deterministic,
        DemoSelectionStrategy::Deterministic
    );
    assert_eq!(
        DemoSelectionStrategy::Hardest,
        DemoSelectionStrategy::Hardest
    );
    assert_ne!(
        DemoSelectionStrategy::Knn,
        DemoSelectionStrategy::MmrDiverse
    );
}

// ── Demonstration ─────────────────────────────────────────────────────────

#[test]
fn demonstration_new_fields() {
    let d = Demonstration::new("input text", "output text");
    assert_eq!(d.input, "input text");
    assert_eq!(d.output, "output text");
    assert!(d.embedding.is_empty());
}

#[test]
fn demonstration_new_default_quality_one() {
    let d = Demonstration::new("in", "out");
    assert!((d.quality - 1.0_f32).abs() < 1e-6);
}

#[test]
fn demonstration_with_quality_sets_value() {
    let d = Demonstration::new("in", "out").with_quality(0.3);
    assert!((d.quality - 0.3_f32).abs() < 1e-6);
}

// ── DemoPool ──────────────────────────────────────────────────────────────

#[test]
fn demo_pool_new_len() {
    let pool = DemoPool::new(vec![
        Demonstration::new("a", "b"),
        Demonstration::new("c", "d"),
    ]);
    assert_eq!(pool.len(), 2);
}

#[test]
fn demo_pool_add_increments_len() {
    let mut pool = DemoPool::new(vec![]);
    pool.add(Demonstration::new("x", "y"));
    assert_eq!(pool.len(), 1);
}

#[test]
fn demo_pool_is_empty_on_empty_pool() {
    let pool = DemoPool::new(vec![]);
    assert!(pool.is_empty());
}

#[test]
fn demo_pool_is_empty_false_when_nonempty() {
    let pool = DemoPool::new(vec![Demonstration::new("a", "b")]);
    assert!(!pool.is_empty());
}

#[test]
fn demo_pool_multiple_add() {
    let mut pool = DemoPool::default();
    for i in 0..5 {
        pool.add(Demonstration::new(format!("q{i}"), format!("a{i}")));
    }
    assert_eq!(pool.len(), 5);
}

// ── DemoSelector ─────────────────────────────────────────────────────────

#[test]
fn demo_selector_default_fields() {
    let sel = DemoSelector::default();
    assert_eq!(sel.strategy, DemoSelectionStrategy::Knn);
    assert!((sel.mmr_lambda - 0.5_f32).abs() < 1e-6);
    assert_eq!(sel.dim, 128);
}

#[test]
fn demo_selector_new_mmr_diverse() {
    let sel = DemoSelector::new(DemoSelectionStrategy::MmrDiverse, 64);
    assert_eq!(sel.strategy, DemoSelectionStrategy::MmrDiverse);
    assert_eq!(sel.dim, 64);
}

#[test]
fn demo_selector_with_mmr_lambda() {
    let sel = DemoSelector::default().with_mmr_lambda(0.8);
    assert!((sel.mmr_lambda - 0.8_f32).abs() < 1e-6);
}

#[test]
fn demo_selector_select_empty_pool_error() {
    let sel = DemoSelector::default();
    let pool = DemoPool::new(vec![]);
    let res = sel.select("query", &pool, 3);
    assert!(matches!(res, Err(PromptOptimizationError::EmptyPool)));
}

#[test]
fn demo_selector_select_empty_query_error() {
    let sel = DemoSelector::default();
    let pool = DemoPool::new(vec![Demonstration::new("a", "b")]);
    let res = sel.select("   ", &pool, 1);
    assert!(matches!(res, Err(PromptOptimizationError::EmptyQuery)));
}

#[test]
fn demo_selector_select_k_zero_returns_empty() {
    let sel = DemoSelector::default();
    let pool = DemoPool::new(vec![Demonstration::new("a", "b")]);
    let selected = sel.select("query", &pool, 0).unwrap();
    assert!(selected.is_empty());
}

#[test]
fn demo_selector_select_k_greater_than_pool_returns_all() {
    let sel = DemoSelector::default();
    let pool = DemoPool::new(vec![
        Demonstration::new("a", "b"),
        Demonstration::new("c", "d"),
    ]);
    let selected = sel.select("query", &pool, 100).unwrap();
    assert_eq!(selected.len(), 2);
}

#[test]
fn demo_selector_knn_most_similar_first() {
    // "rust programming" query — demo with "rust" tokens should rank higher
    let sel = DemoSelector::new(DemoSelectionStrategy::Knn, 128);
    let pool = DemoPool::new(vec![
        Demonstration::new("banana tropical fruit", "b"), // low similarity
        Demonstration::new("rust programming language", "r"), // high similarity
    ]);
    let selected = sel.select("rust programming", &pool, 1).unwrap();
    assert_eq!(selected[0].input, "rust programming language");
}

#[test]
fn demo_selector_knn_returns_requested_k() {
    let sel = DemoSelector::new(DemoSelectionStrategy::Knn, 64);
    let pool = DemoPool::new(vec![
        Demonstration::new("apple fruit", "a"),
        Demonstration::new("banana tropical", "b"),
        Demonstration::new("rust systems", "c"),
        Demonstration::new("python scripting", "d"),
    ]);
    let selected = sel.select("rust systems language", &pool, 2).unwrap();
    assert_eq!(selected.len(), 2);
}

#[test]
fn demo_selector_deterministic_returns_in_index_order() {
    let sel = DemoSelector::new(DemoSelectionStrategy::Deterministic, 64);
    let pool = DemoPool::new(vec![
        Demonstration::new("first", "a"),
        Demonstration::new("second", "b"),
        Demonstration::new("third", "c"),
    ]);
    let selected = sel.select("query", &pool, 3).unwrap();
    assert_eq!(selected[0].input, "first");
    assert_eq!(selected[1].input, "second");
    assert_eq!(selected[2].input, "third");
}

#[test]
fn demo_selector_deterministic_k_truncates() {
    let sel = DemoSelector::new(DemoSelectionStrategy::Deterministic, 64);
    let pool = DemoPool::new(vec![
        Demonstration::new("first", "a"),
        Demonstration::new("second", "b"),
        Demonstration::new("third", "c"),
    ]);
    let selected = sel.select("query", &pool, 2).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].input, "first");
}

#[test]
fn demo_selector_hardest_returns_lowest_quality_first() {
    let sel = DemoSelector::new(DemoSelectionStrategy::Hardest, 64);
    let pool = DemoPool::new(vec![
        Demonstration::new("easy", "a").with_quality(0.9),
        Demonstration::new("hard", "b").with_quality(0.1),
        Demonstration::new("medium", "c").with_quality(0.5),
    ]);
    let selected = sel.select("query", &pool, 1).unwrap();
    assert_eq!(selected[0].input, "hard");
}

#[test]
fn demo_selector_hardest_k2_lowest_two() {
    let sel = DemoSelector::new(DemoSelectionStrategy::Hardest, 64);
    let pool = DemoPool::new(vec![
        Demonstration::new("easy", "a").with_quality(1.0),
        Demonstration::new("hard", "b").with_quality(0.0),
        Demonstration::new("medium", "c").with_quality(0.5),
    ]);
    let selected = sel.select("query", &pool, 2).unwrap();
    // Should be hard (0.0) then medium (0.5)
    assert_eq!(selected[0].input, "hard");
    assert_eq!(selected[1].input, "medium");
}

#[test]
fn demo_selector_mmr_diverse_returns_k_results() {
    let sel = DemoSelector::new(DemoSelectionStrategy::MmrDiverse, 128).with_mmr_lambda(0.5);
    let pool = DemoPool::new(vec![
        Demonstration::new("rust systems programming", "a"),
        Demonstration::new("rust low level language", "b"),
        Demonstration::new("python machine learning", "c"),
    ]);
    let selected = sel.select("rust programming", &pool, 2).unwrap();
    assert_eq!(selected.len(), 2);
}

#[test]
fn demo_selector_mmr_diverse_prefers_diversity() {
    // Two near-identical demos vs one different one; MMR should select the different one
    let sel = DemoSelector::new(DemoSelectionStrategy::MmrDiverse, 128).with_mmr_lambda(0.5);
    // Demo 0 and 1 are nearly identical; demo 2 is very different.
    let pool = DemoPool::new(vec![
        Demonstration::new("rust systems language programming", "a"),
        Demonstration::new("rust systems language programming code", "b"),
        Demonstration::new("banana tropical fruit salad mango pineapple", "c"),
    ]);
    // Select 2: first picks best match for "rust", then MMR penalises demo1 (similar to demo0)
    let selected = sel.select("rust systems language", &pool, 2).unwrap();
    let names: Vec<&str> = selected.iter().map(|d| d.input.as_str()).collect();
    // One rust demo and the banana demo should be preferred over two rust demos
    assert!(names.contains(&"banana tropical fruit salad mango pineapple"));
}

// ── PromptVariant ─────────────────────────────────────────────────────────

#[test]
fn prompt_variant_new_fields() {
    let v = PromptVariant::new("v1", "Answer: {input}");
    assert_eq!(v.name, "v1");
    assert_eq!(v.template, "Answer: {input}");
    assert!(v.instruction.is_none());
}

#[test]
fn prompt_variant_render_substitutes_input() {
    let v = PromptVariant::new("v1", "Answer this: {input}");
    let rendered = v.render("What is Rust?");
    assert_eq!(rendered, "Answer this: What is Rust?");
}

#[test]
fn prompt_variant_render_no_placeholder_returns_template() {
    let v = PromptVariant::new("v1", "static template");
    let rendered = v.render("anything");
    assert_eq!(rendered, "static template");
}

#[test]
fn prompt_variant_with_instruction_sets_field() {
    let v = PromptVariant::new("v1", "t").with_instruction("Be concise.");
    assert_eq!(v.instruction.as_deref(), Some("Be concise."));
}

#[test]
fn prompt_variant_render_multiple_placeholders() {
    let v = PromptVariant::new("v1", "{input} then {input}");
    let rendered = v.render("hello");
    assert_eq!(rendered, "hello then hello");
}

// ── DevExample ────────────────────────────────────────────────────────────

#[test]
fn dev_example_new_fields() {
    let ex = DevExample::new("input here", "expected output");
    assert_eq!(ex.input, "input here");
    assert_eq!(ex.expected, "expected output");
}

// ── JaccardScorer ─────────────────────────────────────────────────────────

#[test]
fn jaccard_scorer_identical_strings_one() {
    let s = JaccardScorer;
    let score = s.score("hello world rust", "hello world rust");
    assert!((score - 1.0_f32).abs() < 1e-6);
}

#[test]
fn jaccard_scorer_disjoint_strings_zero() {
    let s = JaccardScorer;
    let score = s.score("apple banana", "rust python");
    assert!((score - 0.0_f32).abs() < 1e-6);
}

#[test]
fn jaccard_scorer_partial_overlap() {
    let s = JaccardScorer;
    // "rust python" ∩ "rust java" = {"rust"}, ∪ = {"rust","python","java"}
    let score = s.score("rust python", "rust java");
    assert!((score - (1.0_f32 / 3.0_f32)).abs() < 1e-6);
}

#[test]
fn jaccard_scorer_both_empty_one() {
    let s = JaccardScorer;
    let score = s.score("", "");
    assert!((score - 1.0_f32).abs() < 1e-6);
}

// ── PromptOptimizationConfig ──────────────────────────────────────────────

#[test]
fn prompt_optimization_config_default_values() {
    let cfg = PromptOptimizationConfig::default();
    assert_eq!(cfg.num_demos, 4);
    assert!((cfg.mmr_lambda - 0.5_f32).abs() < 1e-6);
    assert_eq!(cfg.dim, 128);
    assert_eq!(cfg.strategy, DemoSelectionStrategy::Knn);
}

#[test]
fn prompt_optimization_config_with_num_demos() {
    let cfg = PromptOptimizationConfig::default().with_num_demos(8);
    assert_eq!(cfg.num_demos, 8);
}

#[test]
fn prompt_optimization_config_with_mmr_lambda() {
    let cfg = PromptOptimizationConfig::default().with_mmr_lambda(0.7);
    assert!((cfg.mmr_lambda - 0.7_f32).abs() < 1e-6);
}

#[test]
fn prompt_optimization_config_with_dim() {
    let cfg = PromptOptimizationConfig::default().with_dim(256);
    assert_eq!(cfg.dim, 256);
}

#[test]
fn prompt_optimization_config_with_strategy() {
    let cfg = PromptOptimizationConfig::default().with_strategy(DemoSelectionStrategy::MmrDiverse);
    assert_eq!(cfg.strategy, DemoSelectionStrategy::MmrDiverse);
}

// ── PromptOptimizer ───────────────────────────────────────────────────────

#[test]
fn prompt_optimizer_new_stores_config() {
    let cfg = PromptOptimizationConfig::default().with_num_demos(2);
    let opt = PromptOptimizer::new(cfg);
    assert_eq!(opt.config.num_demos, 2);
}

#[test]
fn prompt_optimizer_default_created() {
    let opt = PromptOptimizer::default();
    assert_eq!(opt.config.num_demos, 4);
}

#[test]
fn evaluate_variants_empty_dev_set_error() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("v1", "{input}")];
    let res = opt.evaluate_variants(&variants, &[], &JaccardScorer);
    assert!(matches!(res, Err(PromptOptimizationError::EmptyDevSet)));
}

#[test]
fn evaluate_variants_returns_vec_of_same_len() {
    let opt = PromptOptimizer::default();
    let variants = vec![
        PromptVariant::new("v1", "{input}"),
        PromptVariant::new("v2", "Answer: {input}"),
    ];
    let dev = vec![DevExample::new("hello", "hello")];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    assert_eq!(scores.len(), 2);
}

#[test]
fn evaluate_variants_scores_in_unit_interval() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("v1", "{input}")];
    let dev = vec![
        DevExample::new("rust", "python"),
        DevExample::new("hello world", "hello world"),
    ];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    for vs in &scores {
        assert!((0.0_f32..=1.0_f32).contains(&vs.score));
    }
}

#[test]
fn evaluate_variants_identical_template_and_expected_high_score() {
    let opt = PromptOptimizer::default();
    // template renders input unchanged; dev expected == input → perfect match
    let variants = vec![PromptVariant::new("exact", "{input}")];
    let dev = vec![
        DevExample::new("hello world", "hello world"),
        DevExample::new("rust lang", "rust lang"),
    ];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    assert!((scores[0].score - 1.0_f32).abs() < 1e-6);
}

#[test]
fn evaluate_variants_no_overlap_near_zero_score() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("mismatch", "{input}")];
    let dev = vec![DevExample::new("apple banana mango", "rust python systems")];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    // Jaccard("apple banana mango", "rust python systems") = 0
    assert!((scores[0].score - 0.0_f32).abs() < 1e-6);
}

#[test]
fn evaluate_variants_per_example_len_equals_dev_set_len() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("v1", "{input}")];
    let dev = vec![
        DevExample::new("a", "a"),
        DevExample::new("b", "b"),
        DevExample::new("c", "c"),
    ];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    assert_eq!(scores[0].per_example.len(), 3);
}

#[test]
fn evaluate_variants_variant_name_matches_input() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("my-variant", "{input}")];
    let dev = vec![DevExample::new("q", "q")];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    assert_eq!(scores[0].variant_name, "my-variant");
}

// ── best ──────────────────────────────────────────────────────────────────

#[test]
fn best_empty_scores_returns_none() {
    let opt = PromptOptimizer::default();
    let variants: Vec<PromptVariant> = vec![];
    let scores: Vec<VariantScore> = vec![];
    assert!(opt.best(&variants, &scores).is_none());
}

#[test]
fn best_returns_highest_scoring_variant() {
    let opt = PromptOptimizer::default();
    let variants = vec![
        PromptVariant::new("low", "prefix {input}"),
        PromptVariant::new("high", "{input}"),
    ];
    let dev = vec![DevExample::new("rust", "rust")];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    let best = opt.best(&variants, &scores).unwrap();
    // "{input}" renders to "rust", which exactly matches "rust" (score 1.0)
    // "prefix {input}" renders to "prefix rust", Jaccard("prefix rust","rust") = 0.5
    assert_eq!(best.variant_name, "high");
}

#[test]
fn best_single_variant_returned() {
    let opt = PromptOptimizer::default();
    let variants = vec![PromptVariant::new("solo", "{input}")];
    let dev = vec![DevExample::new("test", "test")];
    let scores = opt
        .evaluate_variants(&variants, &dev, &JaccardScorer)
        .unwrap();
    let best = opt.best(&variants, &scores).unwrap();
    assert_eq!(best.variant_name, "solo");
}

// ── VariantScore fields ───────────────────────────────────────────────────

#[test]
fn variant_score_fields_accessible() {
    let vs = VariantScore {
        variant_name: "test".to_string(),
        score: 0.5,
        per_example: vec![0.5, 0.5],
    };
    assert_eq!(vs.variant_name, "test");
    assert!((vs.score - 0.5_f32).abs() < 1e-6);
    assert_eq!(vs.per_example.len(), 2);
}

// ── PromptOptimizationError display ───────────────────────────────────────

#[test]
fn prompt_optimization_error_empty_pool_display() {
    let e = PromptOptimizationError::EmptyPool;
    assert!(!e.to_string().is_empty());
}

#[test]
fn prompt_optimization_error_empty_dev_set_display() {
    let e = PromptOptimizationError::EmptyDevSet;
    assert!(!e.to_string().is_empty());
}

#[test]
fn prompt_optimization_error_empty_query_display() {
    let e = PromptOptimizationError::EmptyQuery;
    assert!(!e.to_string().is_empty());
}
