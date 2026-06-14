#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
//! Tests for the `product_quantization` module.
//!
//! Shared fixtures live here; the actual assertions are split by concern into
//! the `config`, `quantizer` and `index` submodules.

mod config;
mod index;
mod quantizer;

use super::types::PqConfig;

/// Squared L2 distance between two equal-length slices.
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Deterministic, well-separated training vectors over `dim` dimensions.
///
/// Produces `count` vectors drawn from one of four tight clusters. Every
/// dimension of a clustered vector shares the cluster's base value (plus a tiny
/// deterministic jitter), so a 4-centroid codebook reconstructs each subvector
/// with near-zero residual and k-means converges cleanly.
fn make_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    // Four clusters, far apart relative to the 0.001 jitter.
    let anchors = [0.0f32, 20.0, 40.0, 60.0];
    (0..count)
        .map(|i| {
            let base = anchors[i % anchors.len()];
            (0..dim)
                .map(|d| {
                    // Tiny deterministic per-dimension jitter (< 0.005).
                    let jitter = ((i * 7 + d * 3) % 5) as f32 * 0.001;
                    base + jitter
                })
                .collect()
        })
        .collect()
}

/// A compact config: 8-dim vectors, 2 subspaces of 4 dims, 4 centroids each.
fn small_config() -> PqConfig {
    PqConfig::new()
        .with_dim(8)
        .with_num_subspaces(2)
        .with_codebook_bits(2)
        .with_kmeans_iters(10)
}
