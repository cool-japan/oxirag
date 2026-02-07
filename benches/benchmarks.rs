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
use oxirag::layer3_judge::{AdvancedClaimExtractor, ClaimExtractor, ExplanationBuilder};
use oxirag::types::{
    ClaimStructure, Document, LogicalClaim, VerificationResult, VerificationStatus,
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
// Criterion Configuration
// ============================================================================

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
);

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
);

criterion_main!(benches);
