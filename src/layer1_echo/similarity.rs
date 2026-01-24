//! Similarity computation for vector operations.
//!
//! Provides efficient similarity metrics for comparing embedding vectors.

use crate::layer1_echo::traits::SimilarityMetric;

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
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute dot product between two vectors.
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
#[must_use]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return v.to_vec();
    }

    v.iter().map(|x| x / norm).collect()
}

/// Batch compute similarities between a query and multiple vectors.
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
#[must_use]
pub fn top_k_similar(
    query: &[f32],
    vectors: &[Vec<f32>],
    k: usize,
    metric: SimilarityMetric,
    min_score: Option<f32>,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, compute_similarity(query, v, metric)))
        .filter(|(_, score)| min_score.is_none_or(|min| *score >= min))
        .collect();

    // Sort by descending score
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored.truncate(k);
    scored
}

#[cfg(test)]
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
}
