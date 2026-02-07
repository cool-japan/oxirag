//! Similarity computation for vector operations.
//!
//! Provides efficient similarity metrics for comparing embedding vectors.

use crate::layer1_echo::traits::SimilarityMetric;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "sse2")
))]
use super::similarity_simd;

/// Compute similarity between two vectors using the specified metric.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
/// * `metric` - The similarity metric to use
///
/// # Returns
/// The similarity score. Higher values indicate more similar vectors.
///
/// # Panics
/// Panics if the vectors have different lengths.
#[must_use]
pub fn compute_similarity(a: &[f32], b: &[f32], metric: SimilarityMetric) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    match metric {
        SimilarityMetric::Cosine => cosine_similarity(a, b),
        SimilarityMetric::Euclidean => euclidean_to_similarity(a, b),
        SimilarityMetric::DotProduct => dot_product(a, b),
    }
}

/// Compute cosine similarity between two vectors.
///
/// Cosine similarity = (a · b) / (||a|| * ||b||)
/// Range: [-1, 1], where 1 means identical direction.
///
/// Uses SIMD acceleration when available for improved performance.
#[inline]
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    ))]
    {
        similarity_simd::cosine_similarity_simd(a, b)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

/// Compute dot product between two vectors.
///
/// Uses SIMD acceleration when available for improved performance.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    ))]
    {
        similarity_simd::dot_product_simd(a, b)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Compute Euclidean distance and convert to similarity.
///
/// similarity = 1 / (1 + distance)
/// Range: (0, 1], where 1 means identical vectors.
#[must_use]
pub fn euclidean_to_similarity(a: &[f32], b: &[f32]) -> f32 {
    let distance: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    1.0 / (1.0 + distance)
}

/// Normalize a vector to unit length.
///
/// Uses SIMD acceleration when available for improved performance.
#[inline]
#[must_use]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    ))]
    {
        similarity_simd::normalize_simd(v)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 {
            return v.to_vec();
        }

        v.iter().map(|x| x / norm).collect()
    }
}

/// Batch compute similarities between a query and multiple vectors.
#[inline]
#[must_use]
pub fn batch_similarities(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: SimilarityMetric,
) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| compute_similarity(query, v, metric))
        .collect()
}

/// Find top-k most similar vectors.
///
/// Returns indices and scores sorted by descending similarity.
/// Optimized to avoid sorting the entire result set.
#[must_use]
pub fn top_k_similar(
    query: &[f32],
    vectors: &[Vec<f32>],
    k: usize,
    metric: SimilarityMetric,
    min_score: Option<f32>,
) -> Vec<(usize, f32)> {
    if k == 0 || vectors.is_empty() {
        return Vec::new();
    }

    let effective_k = k.min(vectors.len());

    // Pre-allocate result vector
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(vectors.len());

    // Compute all similarities
    for (i, v) in vectors.iter().enumerate() {
        let score = compute_similarity(query, v, metric);
        if min_score.is_none_or(|min| score >= min) {
            scored.push((i, score));
        }
    }

    // Use partial_sort for better performance when k << n
    if scored.len() > effective_k {
        scored.select_nth_unstable_by(effective_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(effective_k);
    }

    // Sort only the top-k elements
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
}

#[cfg(disabled)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_to_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = euclidean_to_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_to_similarity_different() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0]; // distance = 5
        let sim = euclidean_to_similarity(&a, &b);
        assert!((sim - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = normalize(&v);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = normalize(&v);
        assert_eq!(normalized, v);
    }

    #[test]
    fn test_batch_similarities() {
        let query = vec![1.0, 0.0];
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.707, 0.707]];

        let sims = batch_similarities(&query, &vectors, SimilarityMetric::Cosine);

        assert!((sims[0] - 1.0).abs() < 1e-6); // identical
        assert!(sims[1].abs() < 1e-6); // orthogonal
        assert!((sims[2] - 0.707).abs() < 0.01); // 45 degrees
    }

    #[test]
    fn test_top_k_similar() {
        let query = vec![1.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0],  // score ~1.0
            vec![0.0, 1.0],  // score ~0.0
            vec![0.8, 0.6],  // score ~0.8
            vec![-1.0, 0.0], // score ~-1.0
        ];

        let top = top_k_similar(&query, &vectors, 2, SimilarityMetric::Cosine, None);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // index 0 has highest score
        assert_eq!(top[1].0, 2); // index 2 is second
    }

    #[test]
    fn test_top_k_similar_with_min_score() {
        let query = vec![1.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0], // score ~1.0
            vec![0.0, 1.0], // score ~0.0
            vec![0.8, 0.6], // score ~0.8
        ];

        let top = top_k_similar(&query, &vectors, 10, SimilarityMetric::Cosine, Some(0.5));

        assert_eq!(top.len(), 2); // only 2 vectors meet the threshold
    }

    #[test]
    fn test_compute_similarity_dispatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let cos = compute_similarity(&a, &b, SimilarityMetric::Cosine);
        let euc = compute_similarity(&a, &b, SimilarityMetric::Euclidean);
        let dot = compute_similarity(&a, &b, SimilarityMetric::DotProduct);

        assert!((cos - 1.0).abs() < 1e-6);
        assert!((euc - 1.0).abs() < 1e-6);
        assert!((dot - 14.0).abs() < 1e-6);
    }

    // Property-based tests
    #[cfg(disabled)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        // Strategy for generating non-empty vectors of f32
        fn vec_strategy() -> impl Strategy<Value = Vec<f32>> {
            prop::collection::vec(-10.0f32..10.0, 2..50)
        }

        proptest! {
            /// Cosine similarity is commutative: sim(a, b) == sim(b, a)

            fn cosine_similarity_is_commutative(
                vec1 in vec_strategy(),
                vec2 in vec_strategy()
            ) {
                // Make vectors same length
                let len = vec1.len().min(vec2.len());
                let v1 = &vec1[..len];
                let v2 = &vec2[..len];

                let sim1 = cosine_similarity(v1, v2);
                let sim2 = cosine_similarity(v2, v1);

                prop_assert!((sim1 - sim2).abs() < 1e-5,
                    "Cosine similarity not commutative: sim({:?}, {:?}) = {}, sim({:?}, {:?}) = {}",
                    v1, v2, sim1, v2, v1, sim2);
            }

            /// Cosine similarity is in range [-1, 1]
            #[test]
            fn cosine_similarity_in_range(
                vec1 in vec_strategy(),
                vec2 in vec_strategy()
            ) {
                let len = vec1.len().min(vec2.len());
                let v1 = &vec1[..len];
                let v2 = &vec2[..len];

                let sim = cosine_similarity(v1, v2);
                prop_assert!(sim >= -1.0 && sim <= 1.0,
                    "Cosine similarity out of range: {} for vectors {:?}, {:?}", sim, v1, v2);
            }

            /// Normalized vectors have unit length (or are zero vectors)
            #[test]
            fn normalized_vectors_have_unit_length(vec in vec_strategy()) {
                let normalized = normalize(&vec);
                let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();

                // Check if input was zero vector
                let input_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if input_norm < 1e-6 {
                    // Zero vector should remain zero
                    prop_assert!(norm < 1e-6, "Normalized zero vector should remain zero, got norm {}", norm);
                } else {
                    // Non-zero vector should have unit norm
                    prop_assert!((norm - 1.0).abs() < 1e-4,
                        "Normalized vector should have unit length, got {} for input {:?}", norm, vec);
                }
            }

            /// Dot product is commutative: dot(a, b) == dot(b, a)
            #[test]
            fn dot_product_is_commutative(
                vec1 in vec_strategy(),
                vec2 in vec_strategy()
            ) {
                let len = vec1.len().min(vec2.len());
                let v1 = &vec1[..len];
                let v2 = &vec2[..len];

                let dot1 = dot_product(v1, v2);
                let dot2 = dot_product(v2, v1);

                prop_assert!((dot1 - dot2).abs() < 1e-4,
                    "Dot product not commutative: {} vs {}", dot1, dot2);
            }

            /// Euclidean similarity is always positive
            #[test]
            fn euclidean_similarity_is_positive(
                vec1 in vec_strategy(),
                vec2 in vec_strategy()
            ) {
                let len = vec1.len().min(vec2.len());
                let v1 = &vec1[..len];
                let v2 = &vec2[..len];

                let sim = euclidean_to_similarity(v1, v2);
                prop_assert!(sim > 0.0 && sim <= 1.0,
                    "Euclidean similarity should be in (0, 1], got {}", sim);
            }

            /// Identical vectors have maximum cosine similarity
            #[test]
            fn identical_vectors_max_similarity(vec in vec_strategy()) {
                let sim = cosine_similarity(&vec, &vec);

                // Check if vector is zero
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-6 {
                    prop_assert!((sim - 0.0).abs() < 1e-5,
                        "Zero vector similarity should be 0, got {}", sim);
                } else {
                    prop_assert!((sim - 1.0).abs() < 1e-4,
                        "Identical non-zero vectors should have similarity ~1.0, got {}", sim);
                }
            }

            /// Batch similarities should match individual computations
            #[test]
            fn batch_similarities_match_individual(
                query in vec_strategy(),
                vectors in prop::collection::vec(vec_strategy(), 1..10)
            ) {
                // Make all vectors same length as query
                let len = query.len();
                let vectors_resized: Vec<Vec<f32>> = vectors.iter()
                    .map(|v| if v.len() >= len { v[..len].to_vec() } else {
                        let mut new_v = v.clone();
                        new_v.resize(len, 0.0);
                        new_v
                    })
                    .collect();

                let batch_sims = batch_similarities(&query, &vectors_resized, SimilarityMetric::Cosine);

                for (i, vec) in vectors_resized.iter().enumerate() {
                    let individual_sim = compute_similarity(&query, vec, SimilarityMetric::Cosine);
                    prop_assert!((batch_sims[i] - individual_sim).abs() < 1e-5,
                        "Batch similarity mismatch at index {}: {} vs {}", i, batch_sims[i], individual_sim);
                }
            }

            /// Top-k similar should return at most k results
            #[test]
            fn top_k_returns_at_most_k(
                query in vec_strategy(),
                vectors in prop::collection::vec(vec_strategy(), 1..20),
                k in 1usize..10
            ) {
                let len = query.len();
                let vectors_resized: Vec<Vec<f32>> = vectors.iter()
                    .map(|v| if v.len() >= len { v[..len].to_vec() } else {
                        let mut new_v = v.clone();
                        new_v.resize(len, 0.0);
                        new_v
                    })
                    .collect();

                let results = top_k_similar(&query, &vectors_resized, k, SimilarityMetric::Cosine, None);

                prop_assert!(results.len() <= k,
                    "top_k should return at most k results, got {} for k={}", results.len(), k);
                prop_assert!(results.len() <= vectors_resized.len(),
                    "top_k should return at most n results, got {} for n={}", results.len(), vectors_resized.len());
            }

            /// Top-k results should be sorted in descending order
            #[test]
            fn top_k_is_sorted_descending(
                query in vec_strategy(),
                vectors in prop::collection::vec(vec_strategy(), 2..15),
                k in 2usize..8
            ) {
                let len = query.len();
                let vectors_resized: Vec<Vec<f32>> = vectors.iter()
                    .map(|v| if v.len() >= len { v[..len].to_vec() } else {
                        let mut new_v = v.clone();
                        new_v.resize(len, 0.0);
                        new_v
                    })
                    .collect();

                let results = top_k_similar(&query, &vectors_resized, k, SimilarityMetric::Cosine, None);

                for i in 0..results.len().saturating_sub(1) {
                    prop_assert!(results[i].1 >= results[i + 1].1,
                        "top_k results not sorted: {} at {} > {} at {}",
                        results[i].1, i, results[i+1].1, i+1);
                }
            }

            /// Normalization is idempotent: normalize(normalize(v)) ≈ normalize(v)
            #[test]
            fn normalization_is_idempotent(vec in vec_strategy()) {
                let norm1 = normalize(&vec);
                let norm2 = normalize(&norm1);

                for i in 0..norm1.len() {
                    prop_assert!((norm1[i] - norm2[i]).abs() < 1e-4,
                        "Normalization not idempotent at index {}: {} vs {}", i, norm1[i], norm2[i]);
                }
            }
        }
    }
}
