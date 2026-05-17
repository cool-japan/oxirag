//! Benchmark suite for OxiRAG.
//!
//! Run with: cargo bench --features "native,echo,judge,graphrag"

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use oxirag::layer1_echo::{
    CachedEmbeddingProvider, EmbeddingCacheConfig, EmbeddingProvider, InMemoryVectorStore,
    MockEmbeddingProvider,
};
use oxirag::layer1_echo::{Echo, EchoLayer};
use oxirag::layer1_echo::{cosine_similarity, dot_product, normalize};
use oxirag::layer2_speculator::{
    CalibrationMethod, ConfidenceCalibrator, FactualConsistencyStage, KeywordMatchStage,
    RuleBasedSpeculator, SemanticSimilarityStage, Speculator, SpeculatorConfig,
    VerificationPipeline,
};
use oxirag::layer3_judge::{AdvancedClaimExtractor, ClaimExtractor, ExplanationBuilder};
use oxirag::types::{
    ClaimStructure, Document, Draft, LogicalClaim, SearchResult, VerificationResult,
    VerificationStatus,
};

#[cfg(feature = "graphrag")]
use oxirag::layer4_graph::{
    EntityType, GraphEntity, GraphQuery, GraphRelationship, GraphStore, InMemoryGraphStore,
    RelationshipType, bfs_traverse,
};

// ============================================================================
// Layer 1: Echo Benchmarks
// ============================================================================

fn bench_similarity_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    // Test different vector dimensions
    for dim in [64, 128, 384, 768, 1024] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("cosine", dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| cosine_similarity(black_box(a), black_box(b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dot_product", dim),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| dot_product(black_box(a), black_box(b)));
            },
        );

        group.bench_with_input(BenchmarkId::new("normalize", dim), &a, |bench, a| {
            bench.iter(|| normalize(black_box(a)));
        });
    }

    group.finish();
}

fn bench_vector_store_insert(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("vector_store_insert");

    for doc_count in [10, 100, 1000] {
        group.throughput(Throughput::Elements(doc_count as u64));

        group.bench_with_input(
            BenchmarkId::new("in_memory", doc_count),
            &doc_count,
            |bench, &count| {
                bench.to_async(&rt).iter_batched(
                    || {
                        let provider = MockEmbeddingProvider::new(128);
                        let store = InMemoryVectorStore::new(128);
                        let echo = EchoLayer::new(provider, store);
                        let docs: Vec<Document> = (0..count)
                            .map(|i| {
                                Document::new(format!("Document number {i} with some content"))
                            })
                            .collect();
                        (echo, docs)
                    },
                    |(mut echo, docs)| async move {
                        for doc in docs {
                            let _ = echo.index(doc).await;
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_vector_store_search(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("vector_store_search");

    for doc_count in [100, 1000, 5000] {
        // Set up a pre-populated store
        let (echo, _) = rt.block_on(async {
            let provider = MockEmbeddingProvider::new(128);
            let store = InMemoryVectorStore::new(128);
            let mut echo = EchoLayer::new(provider, store);

            let docs: Vec<Document> = (0..doc_count)
                .map(|i| {
                    Document::new(format!(
                        "Document number {i} with various content about topic {}",
                        i % 10
                    ))
                })
                .collect();

            let ids = echo.index_batch(docs).await.unwrap();
            (echo, ids)
        });

        group.throughput(Throughput::Elements(doc_count as u64));

        group.bench_with_input(
            BenchmarkId::new("search_top_10", doc_count),
            &echo,
            |bench, echo| {
                bench.to_async(&rt).iter(|| async {
                    let _ = echo
                        .search(black_box("query about topic 5"), 10, None)
                        .await;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("search_top_100", doc_count),
            &echo,
            |bench, echo| {
                bench.to_async(&rt).iter(|| async {
                    let _ = echo
                        .search(black_box("query about various topics"), 100, None)
                        .await;
                });
            },
        );
    }

    group.finish();
}

fn bench_embedding_cache(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("embedding_cache");

    // Benchmark cache hits vs misses
    let provider = MockEmbeddingProvider::new(128);
    let config = EmbeddingCacheConfig::new(1000);
    let cached = std::sync::Arc::new(CachedEmbeddingProvider::new(provider, config));

    // Pre-warm the cache
    rt.block_on(async {
        for i in 0..100 {
            let _ = cached.embed(&format!("cached text {i}")).await;
        }
    });

    let cached_hit = cached.clone();
    group.bench_function("cache_hit", |bench| {
        let cached = cached_hit.clone();
        let mut i = 0;
        bench.to_async(&rt).iter(|| {
            let cached = cached.clone();
            i = (i + 1) % 100;
            let text = format!("cached text {i}");
            async move {
                let _ = cached.embed(black_box(&text)).await;
            }
        });
    });

    let cached_miss = cached.clone();
    group.bench_function("cache_miss", |bench| {
        let cached = cached_miss.clone();
        let mut i = 0;
        bench.to_async(&rt).iter(|| {
            let cached = cached.clone();
            i += 1;
            let text = format!("new uncached text {i}");
            async move {
                let _ = cached.embed(black_box(&text)).await;
            }
        });
    });

    group.finish();
}

// ============================================================================
// Layer 3: Judge Benchmarks
// ============================================================================

fn bench_claim_extraction(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("claim_extraction");

    let extractor = AdvancedClaimExtractor::new();

    // Different text lengths
    let short_text = "The sky is blue.";
    let medium_text = "The quick brown fox jumps over the lazy dog. The temperature is 25 degrees Celsius. This happens because of atmospheric scattering.";
    let long_text = "Climate change is a significant global challenge. The average temperature has risen by 1.5 degrees over the past century. This increase causes melting ice caps, rising sea levels, and extreme weather events. Scientists predict that temperatures will continue to rise unless we reduce carbon emissions. Many countries have committed to achieving carbon neutrality by 2050. Renewable energy sources like solar and wind power are becoming more cost-effective. Electric vehicles are gaining market share. Governments are implementing carbon taxes and cap-and-trade systems.";

    group.bench_with_input(
        BenchmarkId::new("extract", "short"),
        &short_text,
        |bench, text| {
            bench.to_async(&rt).iter(|| async {
                let _ = extractor.extract_claims(black_box(text), 10).await;
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("extract", "medium"),
        &medium_text,
        |bench, text| {
            bench.to_async(&rt).iter(|| async {
                let _ = extractor.extract_claims(black_box(text), 10).await;
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("extract", "long"),
        &long_text,
        |bench, text| {
            bench.to_async(&rt).iter(|| async {
                let _ = extractor.extract_claims(black_box(text), 20).await;
            });
        },
    );

    group.finish();
}

fn bench_explanation_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("explanation_generation");

    // Create a sample verification result
    let create_result = |claim_count: usize| {
        let mut result =
            VerificationResult::new(VerificationStatus::Verified).with_confidence(0.85);

        for i in 0..claim_count {
            let claim = oxirag::types::LogicalClaim::new(
                format!("Claim number {i} about some topic"),
                ClaimStructure::Predicate {
                    subject: "subject".to_string(),
                    predicate: "is".to_string(),
                    object: Some("object".to_string()),
                },
            )
            .with_confidence(0.9);

            let claim_result =
                oxirag::types::ClaimVerificationResult::new(claim, VerificationStatus::Verified)
                    .with_duration(10);

            result = result.with_claim_result(claim_result);
        }

        result
    };

    let builder = ExplanationBuilder::new();

    for claim_count in [1, 5, 10, 50] {
        let result = create_result(claim_count);

        group.bench_with_input(
            BenchmarkId::new("explain", claim_count),
            &result,
            |bench, result| {
                bench.iter(|| {
                    let _ = builder.explain(black_box(result));
                });
            },
        );
    }

    group.finish();
}

fn bench_smtlib_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("smtlib_encoding");

    let extractor = AdvancedClaimExtractor::new();

    // Different claim structures
    let predicate_claim = LogicalClaim::new(
        "temperature equals 25",
        ClaimStructure::Predicate {
            subject: "temperature".to_string(),
            predicate: "equals".to_string(),
            object: Some("25".to_string()),
        },
    );

    let comparison_claim = LogicalClaim::new(
        "temperature > 20",
        ClaimStructure::Comparison {
            left: "temperature".to_string(),
            operator: oxirag::types::ComparisonOp::GreaterThan,
            right: "20".to_string(),
        },
    );

    let and_claim = LogicalClaim::new(
        "claim1 and claim2 and claim3",
        ClaimStructure::And(vec![
            ClaimStructure::Raw("claim1".to_string()),
            ClaimStructure::Raw("claim2".to_string()),
            ClaimStructure::Raw("claim3".to_string()),
        ]),
    );

    let quantified_claim = LogicalClaim::new(
        "for all x: x > 0",
        ClaimStructure::Quantified {
            quantifier: oxirag::types::Quantifier::ForAll,
            variable: "x".to_string(),
            domain: "integers".to_string(),
            body: Box::new(ClaimStructure::Raw("x > 0".to_string())),
        },
    );

    group.bench_function("predicate", |bench| {
        bench.iter(|| {
            let _ = extractor.to_smtlib(black_box(&predicate_claim));
        });
    });

    group.bench_function("comparison", |bench| {
        bench.iter(|| {
            let _ = extractor.to_smtlib(black_box(&comparison_claim));
        });
    });

    group.bench_function("conjunction", |bench| {
        bench.iter(|| {
            let _ = extractor.to_smtlib(black_box(&and_claim));
        });
    });

    group.bench_function("quantified", |bench| {
        bench.iter(|| {
            let _ = extractor.to_smtlib(black_box(&quantified_claim));
        });
    });

    group.finish();
}

// ============================================================================
// OxiZ SMT Solver Benchmarks
// ============================================================================

#[cfg(feature = "judge")]
fn bench_oxiz_solver_verification(c: &mut Criterion) {
    use oxirag::layer3_judge::{JudgeConfig, OxizVerifier, SmtVerifier};
    use oxirag::types::{CausalStrength, Modality, TimeRelation};

    let rt = tokio::runtime::Runtime::new().expect("failed to create runtime");

    let mut group = c.benchmark_group("oxiz_solver");

    let verifier = OxizVerifier::new(JudgeConfig::default());

    // Benchmark predicate claims
    let predicate_claim = LogicalClaim::new(
        "temperature is high",
        ClaimStructure::Predicate {
            subject: "temperature".to_string(),
            predicate: "is_high".to_string(),
            object: Some("true".to_string()),
        },
    );

    group.bench_function("predicate_claim", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&predicate_claim)).await;
        });
    });

    // Benchmark numeric comparison
    let numeric_comparison = LogicalClaim::new(
        "10 > 5",
        ClaimStructure::Comparison {
            left: "10".to_string(),
            operator: oxirag::types::ComparisonOp::GreaterThan,
            right: "5".to_string(),
        },
    );

    group.bench_function("numeric_comparison", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&numeric_comparison)).await;
        });
    });

    // Benchmark symbolic comparison
    let symbolic_comparison = LogicalClaim::new(
        "x > y",
        ClaimStructure::Comparison {
            left: "x".to_string(),
            operator: oxirag::types::ComparisonOp::GreaterThan,
            right: "y".to_string(),
        },
    );

    group.bench_function("symbolic_comparison", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&symbolic_comparison)).await;
        });
    });

    // Benchmark temporal claims
    let temporal_claim = LogicalClaim::new(
        "event1 before event2",
        ClaimStructure::Temporal {
            event: "event1".to_string(),
            time_relation: TimeRelation::Before,
            reference: "event2".to_string(),
        },
    );

    group.bench_function("temporal_claim", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&temporal_claim)).await;
        });
    });

    // Benchmark causal claims
    let causal_claim = LogicalClaim::new(
        "rain causes wetness",
        ClaimStructure::Causal {
            cause: Box::new(ClaimStructure::Raw("rain".to_string())),
            effect: Box::new(ClaimStructure::Raw("wetness".to_string())),
            strength: CausalStrength::Direct,
        },
    );

    group.bench_function("causal_claim", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&causal_claim)).await;
        });
    });

    // Benchmark modal claims
    let modal_claim = LogicalClaim::new(
        "necessarily p",
        ClaimStructure::Modal {
            claim: Box::new(ClaimStructure::Raw("p".to_string())),
            modality: Modality::Necessary,
        },
    );

    group.bench_function("modal_claim", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&modal_claim)).await;
        });
    });

    // Benchmark conjunctions
    let conjunction_claim = LogicalClaim::new(
        "p and q and r",
        ClaimStructure::And(vec![
            ClaimStructure::Raw("p".to_string()),
            ClaimStructure::Raw("q".to_string()),
            ClaimStructure::Raw("r".to_string()),
        ]),
    );

    group.bench_function("conjunction", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claim(black_box(&conjunction_claim)).await;
        });
    });

    // Benchmark multiple claims verification
    let claims = vec![
        LogicalClaim::new(
            "10 > 5",
            ClaimStructure::Comparison {
                left: "10".to_string(),
                operator: oxirag::types::ComparisonOp::GreaterThan,
                right: "5".to_string(),
            },
        ),
        LogicalClaim::new(
            "20 = 20",
            ClaimStructure::Comparison {
                left: "20".to_string(),
                operator: oxirag::types::ComparisonOp::Equal,
                right: "20".to_string(),
            },
        ),
        LogicalClaim::new(
            "temp is high",
            ClaimStructure::Predicate {
                subject: "temp".to_string(),
                predicate: "is_high".to_string(),
                object: None,
            },
        ),
    ];

    group.bench_function("verify_multiple_claims", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier.verify_claims(black_box(&claims)).await;
        });
    });

    // Benchmark consistency checking
    let consistency_claims = vec![
        LogicalClaim::new(
            "x > 5",
            ClaimStructure::Comparison {
                left: "x".to_string(),
                operator: oxirag::types::ComparisonOp::GreaterThan,
                right: "5".to_string(),
            },
        ),
        LogicalClaim::new(
            "x < 10",
            ClaimStructure::Comparison {
                left: "x".to_string(),
                operator: oxirag::types::ComparisonOp::LessThan,
                right: "10".to_string(),
            },
        ),
    ];

    group.bench_function("consistency_check", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = verifier
                .check_consistency(black_box(&consistency_claims))
                .await;
        });
    });

    group.finish();
}

// ============================================================================
// Layer 2: Speculator Benchmarks
// ============================================================================

fn bench_calibration_platt_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_platt");

    // Pre-fit a Platt calibrator with training samples so calibrate() is non-trivial.
    let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::Platt);
    for i in 0..100_usize {
        let score = (i as f32) / 100.0;
        calibrator.add_sample(score, score > 0.5);
    }
    calibrator.fit().expect("bench setup: platt fit");

    group.bench_function("calibrate_1000_scores", |bench| {
        let mut idx: usize = 0;
        bench.iter(|| {
            // Cycle through 1000 distinct scores in [0, 1).
            let score = (idx % 1000) as f32 / 1000.0;
            idx = idx.wrapping_add(1);
            let _ = calibrator.calibrate(black_box(score));
        });
    });

    group.finish();
}

fn bench_calibration_temperature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_temperature");

    // Temperature scaling is fitted immediately (no gradient descent needed).
    let mut calibrator = ConfidenceCalibrator::new(CalibrationMethod::TemperatureScaling(1.5));
    // add_sample + fit to mark as fitted
    calibrator.add_sample(0.3, false);
    calibrator.add_sample(0.8, true);
    calibrator.fit().expect("bench setup: temperature fit");

    group.bench_function("calibrate_1000_scores", |bench| {
        let mut idx: usize = 0;
        bench.iter(|| {
            let score = (idx % 1000) as f32 / 1000.0;
            idx = idx.wrapping_add(1);
            let _ = calibrator.calibrate(black_box(score));
        });
    });

    group.finish();
}

fn bench_rule_based_speculator_verify(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let mut group = c.benchmark_group("rule_based_speculator");

    let speculator = RuleBasedSpeculator::new(SpeculatorConfig::default());

    // Build a realistic draft + context pair.
    let draft = Draft::new(
        "The capital of France is Paris. Paris is known for the Eiffel Tower and has a \
         population of about 2 million people in the city proper. The city is located along \
         the Seine river.",
        "What is the capital of France?",
    )
    .with_confidence(0.85);

    let context: Vec<SearchResult> = vec![
        SearchResult::new(
            Document::new("The capital of France is Paris. It is known for the Eiffel Tower."),
            0.92,
            0,
        ),
        SearchResult::new(
            Document::new(
                "Paris has a population of about 2 million in the city proper. \
                 The Seine river flows through the city.",
            ),
            0.87,
            1,
        ),
        SearchResult::new(
            Document::new("France is a country in Western Europe with Paris as its capital city."),
            0.80,
            2,
        ),
    ];

    group.bench_function("verify_draft_100x", |bench| {
        bench.to_async(&rt).iter(|| async {
            for _ in 0..100_usize {
                let _ = speculator
                    .verify_draft(black_box(&draft), black_box(&context))
                    .await;
            }
        });
    });

    group.finish();
}

fn bench_verification_pipeline_single_stage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let mut group = c.benchmark_group("verification_pipeline");

    // 500-word-equivalent document content.
    let long_content = "Climate change affects global temperatures significantly. \
        Scientists observe rising sea levels caused by melting ice caps. \
        Renewable energy sources including solar panels and wind turbines provide \
        sustainable power alternatives. Carbon emissions from fossil fuels drive \
        atmospheric warming. Governments worldwide implement carbon trading systems \
        and tax incentives to reduce pollution. Electric vehicles gain market share \
        as battery technology improves. Ocean acidification threatens marine ecosystems \
        and coral reef biodiversity. Extreme weather events increase in frequency and \
        intensity due to higher temperatures. International agreements like the Paris \
        Accord set emission reduction targets. Reforestation programs help absorb \
        atmospheric carbon dioxide. Arctic permafrost thawing releases stored methane. \
        Agriculture adapts to shifting rainfall patterns and drought conditions. \
        Urban heat islands intensify local temperature increases. Scientific consensus \
        supports anthropogenic climate change driven primarily by industrial activities.";

    let draft = Draft::new(
        "Climate change is driven by carbon emissions and causes rising temperatures, \
         melting ice caps, and extreme weather events. Renewable energy and electric \
         vehicles offer sustainable alternatives to fossil fuels.",
        "What causes climate change?",
    )
    .with_confidence(0.75);

    let context: Vec<SearchResult> = vec![SearchResult::new(Document::new(long_content), 0.88, 0)];

    let single_pipeline = VerificationPipeline::new().add_stage(Box::new(KeywordMatchStage::new()));

    group.bench_function("single_stage_500w", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = single_pipeline
                .verify(black_box(&draft), black_box(&context))
                .await;
        });
    });

    group.finish();
}

fn bench_verification_pipeline_multi_stage(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let mut group = c.benchmark_group("verification_pipeline");

    let long_content = "Climate change affects global temperatures significantly. \
        Scientists observe rising sea levels caused by melting ice caps. \
        Renewable energy sources including solar panels and wind turbines provide \
        sustainable power alternatives. Carbon emissions from fossil fuels drive \
        atmospheric warming. Governments worldwide implement carbon trading systems \
        and tax incentives to reduce pollution. Electric vehicles gain market share \
        as battery technology improves. Ocean acidification threatens marine ecosystems \
        and coral reef biodiversity. Extreme weather events increase in frequency and \
        intensity due to higher temperatures. International agreements like the Paris \
        Accord set emission reduction targets. Reforestation programs help absorb \
        atmospheric carbon dioxide. Arctic permafrost thawing releases stored methane. \
        Agriculture adapts to shifting rainfall patterns and drought conditions. \
        Urban heat islands intensify local temperature increases. Scientific consensus \
        supports anthropogenic climate change driven primarily by industrial activities.";

    let draft = Draft::new(
        "Climate change is driven by carbon emissions and causes rising temperatures, \
         melting ice caps, and extreme weather events. Renewable energy and electric \
         vehicles offer sustainable alternatives to fossil fuels.",
        "What causes climate change?",
    )
    .with_confidence(0.75);

    let context: Vec<SearchResult> = vec![
        SearchResult::new(Document::new(long_content), 0.88, 0),
        SearchResult::new(
            Document::new(
                "Fossil fuel combustion is the primary driver of greenhouse gas emissions.",
            ),
            0.82,
            1,
        ),
    ];

    let three_stage_pipeline = VerificationPipeline::new()
        .add_stage(Box::new(KeywordMatchStage::new()))
        .add_stage(Box::new(SemanticSimilarityStage::new()))
        .add_stage(Box::new(FactualConsistencyStage::new()));

    group.bench_function("three_stage_500w", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = three_stage_pipeline
                .verify(black_box(&draft), black_box(&context))
                .await;
        });
    });

    group.finish();
}

// ============================================================================
// Layer 4: Graph Benchmarks
// ============================================================================

#[cfg(feature = "graphrag")]
fn bench_graph_store_add_entity(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let mut group = c.benchmark_group("graph_store_add_entity");
    group.throughput(Throughput::Elements(100));

    group.bench_function("insert_100_entities", |bench| {
        bench.to_async(&rt).iter_batched(
            || {
                let store = InMemoryGraphStore::new();
                let entities: Vec<GraphEntity> = (0..100_usize)
                    .map(|i| {
                        GraphEntity::new(format!("Entity_{i}"), EntityType::Concept)
                            .with_confidence(0.9)
                    })
                    .collect();
                (store, entities)
            },
            |(mut store, entities)| async move {
                for entity in entities {
                    let _ = store.add_entity(black_box(entity)).await;
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

#[cfg(feature = "graphrag")]
fn bench_graph_store_find_by_name(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    // Pre-populate store with 200 entities.
    let store: InMemoryGraphStore = rt.block_on(async {
        let mut s = InMemoryGraphStore::new();
        for i in 0..200_usize {
            let entity = GraphEntity::new(
                format!("Entity_{i}"),
                if i % 2 == 0 {
                    EntityType::Person
                } else {
                    EntityType::Organization
                },
            );
            let _ = s.add_entity(entity).await;
        }
        s
    });

    let mut group = c.benchmark_group("graph_store_find_by_name");

    // Cycle through the 200 entity names by parameterising the bench.
    for idx in 0..200_usize {
        let name = format!("Entity_{idx}");
        group.bench_with_input(
            BenchmarkId::new("find_by_name", idx),
            &name,
            |bench, name| {
                bench.to_async(&rt).iter(|| async {
                    let _ = store.find_entities_by_name(black_box(name.as_str())).await;
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "graphrag")]
fn bench_graph_store_find_by_type(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let store: InMemoryGraphStore = rt.block_on(async {
        let mut s = InMemoryGraphStore::new();
        let entity_types = [
            EntityType::Person,
            EntityType::Organization,
            EntityType::Location,
            EntityType::Concept,
        ];
        for i in 0..200_usize {
            let entity = GraphEntity::new(
                format!("TypeEntity_{i}"),
                entity_types[i % entity_types.len()].clone(),
            );
            let _ = s.add_entity(entity).await;
        }
        s
    });

    let mut group = c.benchmark_group("graph_store_find_by_type");

    group.bench_function("person_type_50_entities", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = store
                .find_entities_by_type(black_box(&EntityType::Person))
                .await;
        });
    });

    group.bench_function("organization_type_50_entities", |bench| {
        bench.to_async(&rt).iter(|| async {
            let _ = store
                .find_entities_by_type(black_box(&EntityType::Organization))
                .await;
        });
    });

    group.finish();
}

#[cfg(feature = "graphrag")]
fn bench_graph_bfs_traversal(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    // Build a graph: root -> 10 nodes (hop 1) -> 10 nodes each (hop 2) -> 10 nodes each (hop 3).
    // 100 nodes total, 3-hop chains.
    let (store, root_id) = rt.block_on(async {
        let mut store = InMemoryGraphStore::new();

        // Root node
        let root = GraphEntity::new("Root", EntityType::Concept).with_confidence(1.0);
        let root_id = store.add_entity(root).await.expect("bench setup: add root");

        // Hop-1 children
        let mut hop1_ids: Vec<String> = Vec::new();
        for i in 0..10_usize {
            let child = GraphEntity::new(format!("Hop1_{i}"), EntityType::Person);
            let child_id = store
                .add_entity(child)
                .await
                .expect("bench setup: add hop1");
            let rel = GraphRelationship::new(
                root_id.clone(),
                child_id.clone(),
                RelationshipType::RelatedTo,
            );
            let _ = store.add_relationship(rel).await;
            hop1_ids.push(child_id);
        }

        // Hop-2 grandchildren
        let mut hop2_ids: Vec<String> = Vec::new();
        for (i, parent_id) in hop1_ids.iter().enumerate() {
            for j in 0..5_usize {
                let grandchild =
                    GraphEntity::new(format!("Hop2_{i}_{j}"), EntityType::Organization);
                let gc_id = store
                    .add_entity(grandchild)
                    .await
                    .expect("bench setup: add hop2");
                let rel = GraphRelationship::new(
                    parent_id.clone(),
                    gc_id.clone(),
                    RelationshipType::RelatedTo,
                );
                let _ = store.add_relationship(rel).await;
                hop2_ids.push(gc_id);
            }
        }

        // Hop-3 great-grandchildren (first 10 hop-2 nodes only, to stay at ~100 total)
        for (i, parent_id) in hop2_ids.iter().take(10).enumerate() {
            for j in 0..2_usize {
                let ggchild = GraphEntity::new(format!("Hop3_{i}_{j}"), EntityType::Technology);
                let gg_id = store
                    .add_entity(ggchild)
                    .await
                    .expect("bench setup: add hop3");
                let rel = GraphRelationship::new(
                    parent_id.clone(),
                    gg_id.clone(),
                    RelationshipType::RelatedTo,
                );
                let _ = store.add_relationship(rel).await;
            }
        }

        (store, root_id)
    });

    let mut group = c.benchmark_group("graph_bfs_traversal");

    group.bench_function("bfs_3hop_from_root", |bench| {
        let query = GraphQuery::new(vec![root_id.clone()]).with_max_hops(3);
        bench.to_async(&rt).iter(|| async {
            let _ = bfs_traverse(black_box(&store), black_box(&query)).await;
        });
    });

    group.bench_function("bfs_2hop_from_root", |bench| {
        let query = GraphQuery::new(vec![root_id.clone()]).with_max_hops(2);
        bench.to_async(&rt).iter(|| async {
            let _ = bfs_traverse(black_box(&store), black_box(&query)).await;
        });
    });

    group.finish();
}

#[cfg(feature = "graphrag")]
fn bench_graph_store_add_relationship(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("bench setup: runtime");

    let mut group = c.benchmark_group("graph_store_add_relationship");
    group.throughput(Throughput::Elements(100));

    group.bench_function("insert_100_relationships", |bench| {
        bench.to_async(&rt).iter_batched(
            || {
                // Pre-insert 200 entities, then build relationships between them.
                let (store, ids) = rt.block_on(async {
                    let mut store = InMemoryGraphStore::new();
                    let mut ids: Vec<String> = Vec::with_capacity(200);
                    for i in 0..200_usize {
                        let entity =
                            GraphEntity::new(format!("RelBenchEntity_{i}"), EntityType::Concept);
                        let id = store.add_entity(entity).await.expect("bench setup: entity");
                        ids.push(id);
                    }
                    (store, ids)
                });

                // Build 100 relationships (entity i -> entity i+100)
                let relationships: Vec<GraphRelationship> = (0..100_usize)
                    .map(|i| {
                        GraphRelationship::new(
                            ids[i].clone(),
                            ids[i + 100].clone(),
                            RelationshipType::RelatedTo,
                        )
                    })
                    .collect();

                (store, relationships)
            },
            |(mut store, relationships)| async move {
                for rel in relationships {
                    let _ = store.add_relationship(black_box(rel)).await;
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

// Layer 1 + Layer 2 + Layer 3 (judge feature) group
#[cfg(feature = "judge")]
criterion_group!(
    benches,
    bench_similarity_functions,
    bench_vector_store_insert,
    bench_vector_store_search,
    bench_embedding_cache,
    bench_claim_extraction,
    bench_explanation_generation,
    bench_smtlib_encoding,
    bench_oxiz_solver_verification,
    bench_calibration_platt_scaling,
    bench_calibration_temperature_scaling,
    bench_rule_based_speculator_verify,
    bench_verification_pipeline_single_stage,
    bench_verification_pipeline_multi_stage,
);

// Layer 1 + Layer 2 + Layer 3 (no judge) group
#[cfg(not(feature = "judge"))]
criterion_group!(
    benches,
    bench_similarity_functions,
    bench_vector_store_insert,
    bench_vector_store_search,
    bench_embedding_cache,
    bench_claim_extraction,
    bench_explanation_generation,
    bench_smtlib_encoding,
    bench_calibration_platt_scaling,
    bench_calibration_temperature_scaling,
    bench_rule_based_speculator_verify,
    bench_verification_pipeline_single_stage,
    bench_verification_pipeline_multi_stage,
);

// Layer 4: Graph benchmarks (requires graphrag feature)
#[cfg(feature = "graphrag")]
criterion_group!(
    layer4_graph_benches,
    bench_graph_store_add_entity,
    bench_graph_store_find_by_name,
    bench_graph_store_find_by_type,
    bench_graph_bfs_traversal,
    bench_graph_store_add_relationship,
);

#[cfg(feature = "graphrag")]
criterion_main!(benches, layer4_graph_benches);

#[cfg(not(feature = "graphrag"))]
criterion_main!(benches);
