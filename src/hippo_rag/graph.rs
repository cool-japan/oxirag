//! Entity extraction and the Personalized `PageRank` (PPR) random walk.
//!
//! `HippoRAG` (Gutiérrez et al., 2024) models long-term memory as an *entity*
//! graph: every passage contributes the entities mentioned within it, and two
//! entities that co-occur in the same passage are linked, with the edge weight
//! growing with the number of shared passages. At query time, a Personalized
//! `PageRank` walk seeded from the query's entities spreads relevance across this
//! graph in a single step, so passages connected to the query only through an
//! intermediate entity (a genuine *multi-hop* path) are still surfaced.
//!
//! This module holds the two pure building blocks — entity extraction via
//! [`extract_entities`] and the restart-biased power iteration via
//! [`personalized_pagerank`] — that [`HippoRagIndex`] composes.
//!
//! [`HippoRagIndex`]: crate::hippo_rag::HippoRagIndex

// ── Entity extraction ─────────────────────────────────────────────────────────

/// Extract entity surface forms from `text`.
///
/// An entity is a capitalised, multi-character alphanumeric token: the token's
/// first character is uppercase and it has at least two characters. Tokens are
/// returned lowercased so that the same entity is recognised regardless of the
/// casing of its first letter across passages, and in first-seen order with
/// duplicates removed.
///
/// The tokeniser splits on every non-alphanumeric character, matching the
/// convention shared across `OxiRAG`'s lexical modules.
#[must_use]
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.chars().count() < 2 {
            continue;
        }
        let Some(first) = token.chars().next() else {
            continue;
        };
        if !first.is_uppercase() {
            continue;
        }
        let lowered = token.to_lowercase();
        if !seen.contains(&lowered) {
            seen.push(lowered);
        }
    }
    seen
}

// ── Personalized `PageRank` ─────────────────────────────────────────────────────

/// Run Personalized `PageRank` over a weighted, undirected entity graph.
///
/// The graph is supplied as a symmetric, row-wise weighted adjacency list:
/// `adjacency[i]` lists `(neighbour, weight)` pairs for entity `i`. The walk is
/// the standard restart-biased power iteration
///
/// `r ← (1 - d) · s + d · Wᵀ r`
///
/// where `s` is the personalization (seed) distribution, `d` is `damping`, and
/// `W` is the row-stochastic transition matrix obtained by normalising each
/// node's outgoing edge weights. Restart mass always returns to the seed
/// distribution `s` rather than the uniform distribution, which is what makes
/// the walk *personalized* toward the query entities. Dangling nodes (no
/// outgoing edges) and the no-seed degenerate case both redistribute their
/// continuation mass via `s`, so the returned vector always sums to `≈ 1.0`.
///
/// `seeds` lists the entity indices to teleport toward; seed mass is split
/// uniformly across them. With no seeds the result is the all-zero vector
/// (there is nothing to personalize toward). `damping` is clamped to
/// `[0.0, 1.0]`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn personalized_pagerank(
    adjacency: &[Vec<(usize, f32)>],
    seeds: &[usize],
    damping: f32,
    iterations: usize,
) -> Vec<f32> {
    let n = adjacency.len();
    if n == 0 {
        return Vec::new();
    }

    // Build the personalization (seed) distribution: uniform over valid seeds.
    let mut seed_dist = vec![0.0f32; n];
    let valid_seeds: Vec<usize> = seeds.iter().copied().filter(|&s| s < n).collect();
    if valid_seeds.is_empty() {
        // Nothing to personalize toward: every restart would have zero mass and
        // the walk can never inject probability, so the stationary result is the
        // zero vector. Returning it explicitly keeps the contract total.
        return vec![0.0f32; n];
    }
    let seed_mass = 1.0 / valid_seeds.len() as f32;
    for &s in &valid_seeds {
        seed_dist[s] += seed_mass;
    }

    let damping = damping.clamp(0.0, 1.0);

    // Per-node total outgoing weight, used to normalise edge contributions.
    let out_weight: Vec<f32> = adjacency
        .iter()
        .map(|edges| edges.iter().map(|&(_, w)| w).sum())
        .collect();

    // Initialise the rank vector at the seed distribution; this is both a valid
    // probability vector and a sensible starting point for the personalized walk.
    let mut rank = seed_dist.clone();

    for _ in 0..iterations {
        let mut next = vec![0.0f32; n];

        // Mass held by dangling nodes (no usable outgoing weight) is rerouted
        // through the seed distribution to preserve total probability.
        let mut dangling_mass = 0.0f32;
        for (i, &w) in out_weight.iter().enumerate() {
            if w <= 0.0 {
                dangling_mass += rank[i];
            }
        }

        // Push each node's rank to its neighbours proportionally to edge weight.
        for (i, edges) in adjacency.iter().enumerate() {
            let total = out_weight[i];
            if total <= 0.0 {
                continue;
            }
            let share = rank[i] / total;
            for &(j, w) in edges {
                next[j] += share * w;
            }
        }

        // Combine: teleport to seeds with prob (1 - d), follow edges with prob d,
        // and reinject dangling mass via the seed distribution.
        for i in 0..n {
            next[i] = (1.0 - damping) * seed_dist[i]
                + damping * next[i]
                + damping * dangling_mass * seed_dist[i];
        }

        rank = next;
    }

    rank
}
