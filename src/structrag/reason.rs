//! [`StructRagReasoner`] — answers a query by reading over a restructured
//! [`StructRagKnowledgeStructure`]: scans a table's numeric columns,
//! traverses a graph, walks a tree's ancestor/descendant links, looks up a
//! catalogue entry, or walks an algorithm's ordered steps.

use std::collections::{HashSet, VecDeque};
use std::fmt::Write as _;

use super::lexical::tokenize;
use super::types::{
    StructRagAlgorithm, StructRagCatalogue, StructRagCatalogueItem, StructRagGraph,
    StructRagKnowledgeStructure, StructRagTable, StructRagTableRow, StructRagTree,
};

// ── StructRagReasoner ─────────────────────────────────────────────────────────

/// Produces a natural-language answer by reading over a restructured
/// [`StructRagKnowledgeStructure`], choosing the reading strategy from the
/// structure's own kind: aggregate/compare over a table, traverse a graph,
/// walk a tree, look an entry up in a catalogue, or walk an algorithm's
/// steps in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct StructRagReasoner;

impl StructRagReasoner {
    /// Create a new reasoner. Stateless: all behaviour is parameterized by
    /// the `query` and `structure` passed to [`StructRagReasoner::reason`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Produce an answer to `query` by reading `structure`.
    #[must_use]
    pub fn reason(&self, query: &str, structure: &StructRagKnowledgeStructure) -> String {
        match structure {
            StructRagKnowledgeStructure::Table(t) => reason_table(query, t),
            StructRagKnowledgeStructure::Graph(g) => reason_graph(query, g),
            StructRagKnowledgeStructure::Tree(t) => reason_tree(query, t),
            StructRagKnowledgeStructure::Catalogue(c) => reason_catalogue(query, c),
            StructRagKnowledgeStructure::Algorithm(a) => reason_algorithm(a),
        }
    }
}

// ── shared query-matching helpers ────────────────────────────────────────────

/// Distinct lowercase tokens shared between `query_terms` and `candidate`.
fn overlap_score(query_terms: &HashSet<String>, candidate: &str) -> usize {
    let candidate_terms: HashSet<String> = tokenize(candidate).into_iter().collect();
    query_terms.intersection(&candidate_terms).count()
}

// ── Table ─────────────────────────────────────────────────────────────────────

const MAX_WORDS: &[&str] = &[
    "highest", "most", "maximum", "max", "largest", "greatest", "top",
];
const MIN_WORDS: &[&str] = &["lowest", "least", "minimum", "min", "smallest", "cheapest"];

/// A row's display subject: the value of a `subject`/`key`/`name`/`item`
/// column when present and non-empty, else the row's first non-empty cell,
/// else its source passage id.
fn row_subject(table: &StructRagTable, row: &StructRagTableRow) -> String {
    for candidate in ["subject", "key", "name", "item"] {
        if let Some(idx) = table.column_index(candidate)
            && let Some(cell) = row.cells.get(idx)
            && !cell.is_empty()
        {
            return cell.clone();
        }
    }
    row.cells
        .iter()
        .find(|c| !c.is_empty())
        .cloned()
        .unwrap_or_else(|| row.source_passage_id.clone())
}

/// `"col1=v1, col2=v2, ..."` for a row's non-empty cells.
fn summarize_row(table: &StructRagTable, row: &StructRagTableRow) -> String {
    table
        .columns
        .iter()
        .zip(row.cells.iter())
        .filter(|(_, v)| !v.is_empty())
        .map(|(c, v)| format!("{c}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The best numeric column to read for an aggregate query: the first column
/// (with at least one numeric cell) whose name is echoed in the query, else
/// the first column with any numeric cell at all.
fn best_numeric_column(query: &str, table: &StructRagTable) -> Option<String> {
    let lower_query = query.to_lowercase();
    for col in &table.columns {
        if lower_query.contains(&col.to_lowercase())
            && table.numeric_column(col).iter().any(Option::is_some)
        {
            return Some(col.clone());
        }
    }
    table
        .columns
        .iter()
        .find(|col| table.numeric_column(col).iter().any(Option::is_some))
        .cloned()
}

/// Row indices whose cells share at least one query term.
fn matching_rows(query: &str, table: &StructRagTable) -> Vec<usize> {
    let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
    if query_terms.is_empty() {
        return Vec::new();
    }
    table
        .rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            row.cells
                .iter()
                .any(|cell| overlap_score(&query_terms, cell) > 0)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Resolve an aggregate (`highest`/`lowest`/...) query by scanning
/// [`best_numeric_column`] for the extreme value and naming its row's
/// subject. `None` when the query carries no aggregate cue, no numeric
/// column exists, or the column has no numeric cells at all.
fn try_aggregate_answer(
    query: &str,
    table: &StructRagTable,
    wants_max: bool,
    wants_min: bool,
) -> Option<String> {
    if !(wants_max || wants_min) {
        return None;
    }
    let col = best_numeric_column(query, table)?;
    let values = table.numeric_column(&col);

    let mut best_idx: Option<usize> = None;
    for (i, v) in values.iter().enumerate() {
        let Some(v) = v else { continue };
        let better = match best_idx {
            None => true,
            Some(b) => {
                let bv = values.get(b).copied().flatten().unwrap_or(0.0);
                if wants_max { *v > bv } else { *v < bv }
            }
        };
        if better {
            best_idx = Some(i);
        }
    }

    let i = best_idx?;
    let row = table.rows.get(i)?;
    let value = values.get(i).copied().flatten()?;
    let subject = row_subject(table, row);
    let comparator = if wants_max { "highest" } else { "lowest" };
    Some(format!(
        "Reading the restructured table ({} rows x {} columns): {subject} has the {comparator} {col} at {value}.",
        table.rows.len(),
        table.columns.len()
    ))
}

/// Read `table` to answer `query`: an aggregate (`highest`/`lowest`/...)
/// query is resolved by scanning [`best_numeric_column`] for the extreme
/// value and naming its row's subject; otherwise rows whose cells overlap
/// the query's terms are surfaced; failing that, the whole table is
/// summarized.
fn reason_table(query: &str, table: &StructRagTable) -> String {
    if table.rows.is_empty() {
        return "The restructured table has no rows to reason over.".to_string();
    }
    let lower_query = query.to_lowercase();
    let wants_max = MAX_WORDS.iter().any(|w| lower_query.contains(w));
    let wants_min = MIN_WORDS.iter().any(|w| lower_query.contains(w));

    if let Some(answer) = try_aggregate_answer(query, table, wants_max, wants_min) {
        return answer;
    }

    let matches = matching_rows(query, table);
    if !matches.is_empty() {
        let parts: Vec<String> = matches
            .iter()
            .filter_map(|&i| table.rows.get(i).map(|row| summarize_row(table, row)))
            .collect();
        return format!("Reading the restructured table: {}", parts.join("; "));
    }

    let all_rows: Vec<String> = table.rows.iter().map(|r| summarize_row(table, r)).collect();
    format!(
        "Restructured into a table with {} rows and columns [{}]; no row specifically matched the query, so here is the full table: {}",
        table.rows.len(),
        table.columns.join(", "),
        all_rows.join("; ")
    )
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// Breadth-first shortest path from `start` to `goal` (node ids), treating
/// edges as undirected for reachability. `Some([start])` when they are
/// equal; `None` when no path exists.
fn shortest_path(graph: &StructRagGraph, start: &str, goal: &str) -> Option<Vec<String>> {
    if start == goal {
        return Some(vec![start.to_string()]);
    }
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<Vec<String>> = VecDeque::new();
    visited.insert(start.to_string());
    queue.push_back(vec![start.to_string()]);

    while let Some(path) = queue.pop_front() {
        let Some(last) = path.last() else { continue };
        for neighbor in graph.neighbors(last) {
            if neighbor == goal {
                let mut full = path.clone();
                full.push(neighbor);
                return Some(full);
            }
            if visited.insert(neighbor.clone()) {
                let mut next = path.clone();
                next.push(neighbor);
                queue.push_back(next);
            }
        }
    }
    None
}

/// Describe a node-id `path` as `"relation1 -> relation2 -> ..."`, reading
/// each hop's edge relation (in either direction) from `graph`.
fn describe_path(graph: &StructRagGraph, path: &[String]) -> String {
    let mut parts = Vec::new();
    for window in path.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        let relation = graph
            .edges
            .iter()
            .find(|e| (e.source == *a && e.target == *b) || (e.source == *b && e.target == *a))
            .map_or_else(
                || "related_to".to_string(),
                |e| e.relation.replace('_', " "),
            );
        parts.push(relation);
    }
    parts.join(" -> ")
}

/// Read `graph` to answer `query`: when two or more query-matched entities
/// exist, a shortest path is traced between the first two; when exactly one
/// matches, its outgoing relations are listed; otherwise the most-mentioned
/// entities are surfaced as a summary.
fn reason_graph(query: &str, graph: &StructRagGraph) -> String {
    if graph.nodes.is_empty() {
        return "The restructured graph has no entities to reason over.".to_string();
    }
    let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
    let mentioned: Vec<&super::types::StructRagGraphNode> = graph
        .nodes
        .iter()
        .filter(|n| overlap_score(&query_terms, &n.label) > 0)
        .collect();

    if mentioned.len() >= 2 {
        if let Some(path) = shortest_path(graph, &mentioned[0].id, &mentioned[1].id) {
            let relation_desc = describe_path(graph, &path);
            return format!(
                "Traversing the restructured graph: {} is connected to {} via {relation_desc}.",
                mentioned[0].label, mentioned[1].label
            );
        }
        return format!(
            "The restructured graph contains {} and {} but no path connects them within the retrieved passages.",
            mentioned[0].label, mentioned[1].label
        );
    }

    if let Some(node) = mentioned.first() {
        let edges = graph.edges_from(&node.id);
        if edges.is_empty() {
            return format!(
                "{} appears in the restructured graph with no recorded relations.",
                node.label
            );
        }
        let desc: Vec<String> = edges
            .iter()
            .map(|e| {
                let target_label = graph
                    .nodes
                    .iter()
                    .find(|n| n.id == e.target)
                    .map_or(e.target.as_str(), |n| n.label.as_str());
                format!(
                    "{} {} {}",
                    node.label,
                    e.relation.replace('_', " "),
                    target_label
                )
            })
            .collect();
        return format!(
            "Traversing the restructured graph from {}: {}.",
            node.label,
            desc.join("; ")
        );
    }

    let mut by_mentions: Vec<&super::types::StructRagGraphNode> = graph.nodes.iter().collect();
    by_mentions.sort_by(|a, b| b.mentions.cmp(&a.mentions).then_with(|| a.id.cmp(&b.id)));
    let top: Vec<String> = by_mentions
        .iter()
        .take(3)
        .map(|n| format!("{} ({} mentions)", n.label, n.mentions))
        .collect();
    format!(
        "Restructured into a graph with {} entities and {} relations; most prominent: {}.",
        graph.nodes.len(),
        graph.edges.len(),
        top.join(", ")
    )
}

// ── Tree ──────────────────────────────────────────────────────────────────────

/// Read `tree` to answer `query`: a query-matched node reports its ancestor
/// chain and children; otherwise the tree's roots and depth are summarized.
fn reason_tree(query: &str, tree: &StructRagTree) -> String {
    if tree.nodes.is_empty() {
        return "The restructured tree has no nodes to reason over.".to_string();
    }
    let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
    let matched = tree
        .nodes
        .iter()
        .find(|n| overlap_score(&query_terms, &n.label) > 0);

    if let Some(node) = matched {
        let ancestors: Vec<String> = tree
            .ancestors(node.id)
            .iter()
            .filter_map(|&id| tree.nodes.iter().find(|n| n.id == id))
            .map(|n| n.label.clone())
            .collect();
        let children: Vec<String> = node
            .children
            .iter()
            .filter_map(|&id| tree.nodes.iter().find(|n| n.id == id))
            .map(|n| n.label.clone())
            .collect();

        let mut desc = format!(
            "In the restructured tree, {} is at depth {}",
            node.label, node.depth
        );
        if !ancestors.is_empty() {
            let _ = write!(desc, ", under {}", ancestors.join(" -> "));
        }
        if children.is_empty() {
            desc.push_str(", a leaf node");
        } else {
            let _ = write!(desc, ", with children [{}]", children.join(", "));
        }
        desc.push('.');
        return desc;
    }

    let roots: Vec<String> = tree
        .root_ids
        .iter()
        .filter_map(|&id| tree.nodes.iter().find(|n| n.id == id))
        .map(|n| n.label.clone())
        .collect();
    let max_depth = tree.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    format!(
        "Restructured into a tree of {} nodes ({} levels deep) rooted at [{}].",
        tree.nodes.len(),
        max_depth + 1,
        roots.join(", ")
    )
}

// ── Catalogue ─────────────────────────────────────────────────────────────────

/// Format a single item as `"key (attr1=v1, attr2=v2)"`.
fn describe_item(item: &StructRagCatalogueItem) -> String {
    let attrs = item
        .attributes
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} ({attrs})", item.key)
}

/// Read `catalogue` to answer `query`: items whose key overlaps the query's
/// terms are looked up and described; otherwise every item's key is listed.
fn reason_catalogue(query: &str, catalogue: &StructRagCatalogue) -> String {
    if catalogue.items.is_empty() {
        return "The restructured catalogue has no items to reason over.".to_string();
    }
    let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
    let matched: Vec<&StructRagCatalogueItem> = catalogue
        .items
        .iter()
        .filter(|item| overlap_score(&query_terms, &item.key) > 0)
        .collect();

    if !matched.is_empty() {
        let parts: Vec<String> = matched.iter().map(|item| describe_item(item)).collect();
        return format!(
            "Looked up in the restructured catalogue: {}.",
            parts.join("; ")
        );
    }

    let keys: Vec<&str> = catalogue.items.iter().map(|i| i.key.as_str()).collect();
    format!(
        "Restructured into a catalogue of {} items: {}.",
        catalogue.items.len(),
        keys.join(", ")
    )
}

// ── Algorithm ─────────────────────────────────────────────────────────────────

/// Walk `algorithm`'s steps in order, formatting each as `"Step N:
/// action"`.
fn reason_algorithm(algorithm: &StructRagAlgorithm) -> String {
    if algorithm.steps.is_empty() {
        return "The restructured procedure has no steps to reason over.".to_string();
    }
    let steps: Vec<String> = algorithm
        .steps
        .iter()
        .map(|s| format!("Step {}: {}", s.index, s.action))
        .collect();
    format!(
        "Walking the restructured procedure ({} steps): {}",
        algorithm.steps.len(),
        steps.join(" ")
    )
}
