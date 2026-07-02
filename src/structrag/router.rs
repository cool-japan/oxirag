//! [`StructRagRouter`] — deterministic heuristic inference of the optimal
//! [`StructRagStructureKind`] for a query.

use std::fmt::Write as _;

use super::types::{
    StructRagConfig, StructRagError, StructRagRoutingDecision, StructRagStructureKind,
};

// ── cue tables ────────────────────────────────────────────────────────────────

/// A weighted cue phrase associated with a [`StructRagStructureKind`].
/// Multi-word, more specific phrases carry a higher weight than generic
/// single words, so a query with one strong, specific cue can outweigh
/// several weak, generic ones.
struct Cue {
    /// The phrase to search for (already lowercase).
    phrase: &'static str,
    /// The phrase's contribution to its kind's raw score when matched.
    weight: f32,
}

const fn cue(phrase: &'static str, weight: f32) -> Cue {
    Cue { phrase, weight }
}

/// Cues for [`StructRagStructureKind::Table`]: comparison and statistics
/// across several items.
const TABLE_CUES: &[Cue] = &[
    cue("comparison", 1.5),
    cue("compare", 1.5),
    cue(" versus ", 1.5),
    cue(" vs ", 1.5),
    cue("vs.", 1.0),
    cue("statistics", 1.5),
    cue("statistic", 1.0),
    cue("which is higher", 2.0),
    cue("which is lower", 2.0),
    cue("which has more", 2.0),
    cue("highest", 1.0),
    cue("lowest", 1.0),
    cue("most expensive", 2.0),
    cue("cheapest", 1.5),
    cue("average", 1.0),
    cue("how many", 1.0),
    cue("price of", 1.0),
    cue("cost of", 1.0),
    cue("rank the", 1.5),
    cue("table of", 1.5),
    cue("numbers for", 1.0),
];

/// Cues for [`StructRagStructureKind::Graph`]: relationships and
/// connections between entities.
const GRAPH_CUES: &[Cue] = &[
    cue("relationship between", 2.0),
    cue("relationships", 1.0),
    cue("relationship", 1.0),
    cue("related to", 1.5),
    cue("connection between", 2.0),
    cue("connection", 1.0),
    cue("connected to", 1.5),
    cue("network of", 1.5),
    cue("linked to", 1.5),
    cue("influence", 1.0),
    cue("influences", 1.0),
    cue("interacts with", 1.5),
    cue("interaction between", 2.0),
    cue("depends on", 1.5),
    cue("affects", 1.0),
    cue("connects to", 1.5),
    cue("associated with", 1.5),
];

/// Cues for [`StructRagStructureKind::Tree`]: hierarchy, taxonomy, and
/// decomposition.
const TREE_CUES: &[Cue] = &[
    cue("hierarchy", 1.5),
    cue("hierarchical", 1.5),
    cue("taxonomy", 1.5),
    cue("classification of", 1.5),
    cue("categories of", 1.5),
    cue("category", 0.75),
    cue("subcategories", 1.5),
    cue("sub-categories", 1.5),
    cue("breakdown of", 1.5),
    cue("decompose", 1.5),
    cue("decomposition", 1.5),
    cue("structure of", 1.5),
    cue("parent and child", 1.5),
    cue("is a type of", 1.5),
    cue("subtype", 1.0),
    cue("subtypes", 1.0),
    cue("levels", 0.75),
    cue("tiers", 1.0),
];

/// Cues for [`StructRagStructureKind::Catalogue`]: enumerated lists and
/// inventories.
const CATALOGUE_CUES: &[Cue] = &[
    cue("list of", 1.5),
    cue("list all", 1.5),
    cue("enumerate", 1.5),
    cue("inventory of", 2.0),
    cue("inventory", 1.0),
    cue("catalogue of", 2.0),
    cue("catalog of", 2.0),
    cue("collection of", 1.5),
    cue("examples of", 1.5),
    cue("options for", 1.5),
    cue("varieties of", 1.5),
    cue("what are the", 1.0),
    cue("items in", 1.5),
    cue("types of", 1.0),
    cue("kinds of", 1.0),
];

/// Cues for [`StructRagStructureKind::Algorithm`]: step-by-step procedures.
const ALGORITHM_CUES: &[Cue] = &[
    cue("step by step", 2.0),
    cue("step-by-step", 2.0),
    cue("steps to", 1.5),
    cue("steps for", 1.5),
    cue("procedure for", 1.5),
    cue("procedure to", 1.5),
    cue("process of", 1.0),
    cue("how to", 1.5),
    cue("instructions for", 1.5),
    cue("instructions to", 1.5),
    cue("guide to", 1.0),
    cue("algorithm for", 1.5),
    cue("workflow for", 1.5),
    cue("sequence of steps", 2.0),
    cue("method for", 1.0),
    cue("how do i", 1.5),
    cue("how do you", 1.5),
];

fn cues_for(kind: StructRagStructureKind) -> &'static [Cue] {
    match kind {
        StructRagStructureKind::Table => TABLE_CUES,
        StructRagStructureKind::Graph => GRAPH_CUES,
        StructRagStructureKind::Tree => TREE_CUES,
        StructRagStructureKind::Catalogue => CATALOGUE_CUES,
        StructRagStructureKind::Algorithm => ALGORITHM_CUES,
    }
}

/// Sum the weights of every cue in `cues` that matches (as a substring)
/// somewhere in the already-lowercased `haystack`, alongside the list of
/// matched phrases (for the routing rationale).
fn cue_score(haystack: &str, cues: &[Cue]) -> (f32, Vec<&'static str>) {
    let mut score = 0.0;
    let mut matched = Vec::new();
    for c in cues {
        if haystack.contains(c.phrase) {
            score += c.weight;
            matched.push(c.phrase);
        }
    }
    (score, matched)
}

/// Rank of `kind` in `config.tie_break_order`; kinds absent from that list
/// rank after every listed kind, ordered by [`StructRagStructureKind::all`].
fn tie_break_rank(config: &StructRagConfig, kind: StructRagStructureKind) -> usize {
    config
        .tie_break_order
        .iter()
        .position(|k| *k == kind)
        .unwrap_or_else(|| {
            config.tie_break_order.len()
                + StructRagStructureKind::all()
                    .iter()
                    .position(|k| *k == kind)
                    .unwrap_or(StructRagStructureKind::all().len())
        })
}

// ── StructRagRouter ──────────────────────────────────────────────────────────

/// Classifies the optimal [`StructRagStructureKind`] for a query via
/// deterministic, weighted keyword-cue matching — no LLM call, no
/// randomness.
///
/// Every [`StructRagStructureKind`] variant is scored independently by
/// summing the weights of its matched cue phrases (see e.g. `TABLE_CUES`);
/// raw scores are normalized into a `[0.0, 1.0]` confidence by dividing by
/// the total evidence collected across all five kinds. The highest-scoring
/// *enabled* kind wins, subject to [`StructRagConfig::min_confidence`] (a
/// low-confidence winner is replaced by [`StructRagConfig::default_kind`],
/// when enabled) and [`StructRagConfig::tie_break_order`] (exact ties are
/// broken deterministically).
#[derive(Debug, Clone, Copy, Default)]
pub struct StructRagRouter;

impl StructRagRouter {
    /// Create a new router. Stateless: all behaviour is parameterized by the
    /// `config` passed to [`StructRagRouter::route`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Infer the optimal structure kind for `query`.
    ///
    /// # Errors
    ///
    /// - [`StructRagError::EmptyQuery`] if `query` is blank.
    /// - [`StructRagError::NoEnabledStructureKinds`] if
    ///   [`StructRagConfig::enabled_kinds`] is empty.
    pub fn route(
        &self,
        query: &str,
        config: &StructRagConfig,
    ) -> Result<StructRagRoutingDecision, StructRagError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(StructRagError::EmptyQuery);
        }
        if config.enabled_kinds.is_empty() {
            return Err(StructRagError::NoEnabledStructureKinds);
        }

        let lower = trimmed.to_lowercase();

        // Raw (score, matched-phrase) per kind, in canonical order.
        let raw: Vec<(StructRagStructureKind, f32, Vec<&'static str>)> =
            StructRagStructureKind::all()
                .into_iter()
                .map(|k| {
                    let (score, matched) = cue_score(&lower, cues_for(k));
                    (k, score, matched)
                })
                .collect();

        let total: f32 = raw.iter().map(|(_, score, _)| *score).sum();

        let mut scores: Vec<(StructRagStructureKind, f32)> = raw
            .iter()
            .map(|(k, score, _)| {
                let normalized = if total > 0.0 { *score / total } else { 0.0 };
                (*k, normalized)
            })
            .collect();

        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| tie_break_rank(config, a.0).cmp(&tie_break_rank(config, b.0)))
        });

        let enabled_sorted: Vec<(StructRagStructureKind, f32)> = scores
            .iter()
            .filter(|(k, _)| config.is_enabled(*k))
            .copied()
            .collect();

        let Some(&(top_kind, top_confidence)) = enabled_sorted.first() else {
            return Err(StructRagError::NoEnabledStructureKinds);
        };

        let mut kind = top_kind;
        let mut confidence = top_confidence;
        let mut used_default = false;

        if confidence < config.min_confidence && config.is_enabled(config.default_kind) {
            kind = config.default_kind;
            confidence = scores
                .iter()
                .find(|(k, _)| *k == kind)
                .map_or(0.0, |(_, s)| *s);
            used_default = true;
        }

        let matched_desc = raw
            .iter()
            .find(|(k, _, _)| *k == kind)
            .map(|(_, _, matched)| matched.clone())
            .unwrap_or_default();
        let matched_text = if matched_desc.is_empty() {
            "no cue phrases matched".to_string()
        } else {
            matched_desc.join(", ")
        };

        let mut rationale = format!(
            "query=\"{trimmed}\" matched cues for {kind}: [{matched_text}] -> confidence={confidence:.2}"
        );
        if used_default {
            let _ = write!(
                rationale,
                " (top enabled kind {top_kind} scored {top_confidence:.2}, below threshold {:.2}; falling back to default kind {kind})",
                config.min_confidence
            );
        }
        if enabled_sorted.len() < StructRagStructureKind::all().len() {
            let _ = write!(
                rationale,
                " ({} of {} kinds enabled)",
                enabled_sorted.len(),
                StructRagStructureKind::all().len()
            );
        }

        Ok(StructRagRoutingDecision {
            kind,
            confidence,
            rationale,
            scores,
        })
    }
}
