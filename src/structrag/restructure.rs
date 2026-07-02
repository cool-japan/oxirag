//! [`StructRagRestructurer`] — transforms retrieved passages into a concrete
//! [`StructRagKnowledgeStructure`] of a chosen [`StructRagStructureKind`].

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};

use super::lexical::{
    extract_entity_spans, extract_key_value_pairs, find_ci, first_entity_or_token,
    first_number_token, is_stopword, leading_indent_depth, split_list, split_sentences,
    strip_leading_article, strip_list_marker, strip_step_marker, tokenize, truncate_for_display,
};
use super::types::{
    StructRagAlgorithm, StructRagCatalogue, StructRagCatalogueItem, StructRagConfig,
    StructRagGraph, StructRagGraphEdge, StructRagGraphNode, StructRagKnowledgeStructure,
    StructRagPassage, StructRagStep, StructRagStructureKind, StructRagTable, StructRagTableRow,
    StructRagTree, StructRagTreeNode,
};

// ── StructRagRestructurer ────────────────────────────────────────────────────

/// Builds a concrete [`StructRagKnowledgeStructure`] of a given
/// [`StructRagStructureKind`] from retrieved passages.
///
/// This is the "restructuring" half of `StructRAG`'s novelty: rather than
/// extracting records against a fixed schema (see
/// [`crate::structured_extraction`]), it re-shapes free-form passages into
/// whichever of five general knowledge-representation forms best suits the
/// query, as chosen by [`crate::structrag::StructRagRouter`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StructRagRestructurer;

impl StructRagRestructurer {
    /// Create a new restructurer. Stateless: all behaviour is parameterized
    /// by the `kind` and `config` passed to
    /// [`StructRagRestructurer::restructure`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Restructure `passages` into a [`StructRagKnowledgeStructure`] of
    /// `kind`, respecting `config.max_structure_size`.
    #[must_use]
    pub fn restructure(
        &self,
        kind: StructRagStructureKind,
        passages: &[StructRagPassage],
        config: &StructRagConfig,
    ) -> StructRagKnowledgeStructure {
        let structure = match kind {
            StructRagStructureKind::Table => {
                StructRagKnowledgeStructure::Table(build_table(passages))
            }
            StructRagStructureKind::Graph => {
                StructRagKnowledgeStructure::Graph(build_graph(passages))
            }
            StructRagStructureKind::Tree => StructRagKnowledgeStructure::Tree(build_tree(passages)),
            StructRagStructureKind::Catalogue => {
                StructRagKnowledgeStructure::Catalogue(build_catalogue(passages))
            }
            StructRagStructureKind::Algorithm => {
                StructRagKnowledgeStructure::Algorithm(build_algorithm(passages))
            }
        };
        truncate_to_max_size(structure, config.max_structure_size)
    }
}

// ── size cap ──────────────────────────────────────────────────────────────────

/// Deterministically truncate `structure` to at most `max_size` primary
/// elements, repairing any references (graph edges, tree parent/child links)
/// that would otherwise dangle.
fn truncate_to_max_size(
    structure: StructRagKnowledgeStructure,
    max_size: usize,
) -> StructRagKnowledgeStructure {
    match structure {
        StructRagKnowledgeStructure::Table(mut t) => {
            t.rows.truncate(max_size);
            StructRagKnowledgeStructure::Table(t)
        }
        StructRagKnowledgeStructure::Graph(mut g) => {
            g.nodes.truncate(max_size);
            let kept_ids: HashSet<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
            g.edges.retain(|e| {
                kept_ids.contains(e.source.as_str()) && kept_ids.contains(e.target.as_str())
            });
            g.edges.truncate(max_size);
            StructRagKnowledgeStructure::Graph(g)
        }
        StructRagKnowledgeStructure::Tree(mut tr) => {
            if tr.nodes.len() > max_size {
                tr.nodes.truncate(max_size);
                let kept: HashSet<usize> = tr.nodes.iter().map(|n| n.id).collect();
                for node in &mut tr.nodes {
                    if let Some(p) = node.parent
                        && !kept.contains(&p)
                    {
                        node.parent = None;
                    }
                    node.children.retain(|c| kept.contains(c));
                }
                let mut root_ids: HashSet<usize> = tr
                    .root_ids
                    .iter()
                    .copied()
                    .filter(|id| kept.contains(id))
                    .collect();
                for node in &tr.nodes {
                    if node.parent.is_none() {
                        root_ids.insert(node.id);
                    }
                }
                tr.root_ids = root_ids.into_iter().collect();
                tr.root_ids.sort_unstable();
                recompute_tree_depths(&mut tr);
            }
            StructRagKnowledgeStructure::Tree(tr)
        }
        StructRagKnowledgeStructure::Catalogue(mut c) => {
            c.items.truncate(max_size);
            StructRagKnowledgeStructure::Catalogue(c)
        }
        StructRagKnowledgeStructure::Algorithm(mut a) => {
            a.steps.truncate(max_size);
            for (i, step) in a.steps.iter_mut().enumerate() {
                step.index = i + 1;
            }
            StructRagKnowledgeStructure::Algorithm(a)
        }
    }
}

// ── Table ─────────────────────────────────────────────────────────────────────

/// Build a [`StructRagTable`] with one row per (non-blank) passage.
///
/// Each passage first tries explicit `key: value` extraction
/// ([`extract_key_value_pairs`]); when a passage yields none (ordinary
/// prose, no markup), it falls back to a `subject`/`value`/`detail` triad
/// so free-text numeric passages ("Product A costs $9.99...") still produce
/// a genuinely comparable row. Columns are the union of keys across all
/// passages, in first-appearance order; a row's cell is empty when its
/// passage did not contribute that column.
fn build_table(passages: &[StructRagPassage]) -> StructRagTable {
    let mut columns: Vec<String> = Vec::new();
    let mut per_passage: Vec<(Vec<(String, String)>, String)> = Vec::new();

    for passage in passages {
        let text = passage.text.trim();
        if text.is_empty() {
            continue;
        }
        let mut pairs = extract_key_value_pairs(text);
        if pairs.is_empty() {
            pairs = fallback_subject_value_detail(text);
        }
        for (key, _) in &pairs {
            if !columns.iter().any(|c: &String| c.eq_ignore_ascii_case(key)) {
                columns.push(key.clone());
            }
        }
        per_passage.push((pairs, passage.id.clone()));
    }

    let rows = per_passage
        .into_iter()
        .map(|(pairs, source_passage_id)| {
            let cells = columns
                .iter()
                .map(|col| {
                    pairs
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(col))
                        .map(|(_, v)| v.clone())
                        .unwrap_or_default()
                })
                .collect();
            StructRagTableRow {
                cells,
                source_passage_id,
            }
        })
        .collect();

    StructRagTable { columns, rows }
}

/// Fallback row shape for passages with no explicit `key: value` markup: a
/// `subject` (first entity mention or leading token), a `value` (first
/// number found, currency/percent preserved), and a `detail` (truncated raw
/// text) — enough structure for the reasoner to compare rows numerically.
fn fallback_subject_value_detail(text: &str) -> Vec<(String, String)> {
    vec![
        ("subject".to_string(), first_entity_or_token(text)),
        (
            "value".to_string(),
            first_number_token(text).unwrap_or_default(),
        ),
        ("detail".to_string(), truncate_for_display(text, 80)),
    ]
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// Build a [`StructRagGraph`] by scanning every sentence of every passage
/// for capitalized entity spans ([`extract_entity_spans`]). Every span found
/// registers (or increments the mention count of) a node; when a sentence
/// contains two or more spans, an edge is drawn between the first two, with
/// the intervening (stopword-filtered) text as the relation label —
/// `"related_to"` when that text is empty.
fn build_graph(passages: &[StructRagPassage]) -> StructRagGraph {
    let mut nodes: Vec<StructRagGraphNode> = Vec::new();
    let mut node_index: HashMap<String, usize> = HashMap::new();
    let mut edges: Vec<StructRagGraphEdge> = Vec::new();

    for passage in passages {
        let text = passage.text.trim();
        if text.is_empty() {
            continue;
        }
        for sentence in split_sentences(text) {
            let spans = extract_entity_spans(sentence);
            for label in &spans {
                register_graph_node(&mut nodes, &mut node_index, label);
            }
            if spans.len() >= 2 {
                let source_id = normalize_label(&spans[0]);
                let target_id = normalize_label(&spans[1]);
                if source_id != target_id {
                    let relation = extract_relation_between(sentence, &spans[0], &spans[1]);
                    edges.push(StructRagGraphEdge {
                        source: source_id,
                        target: target_id,
                        relation,
                        source_passage_id: passage.id.clone(),
                    });
                }
            }
        }
    }

    StructRagGraph { nodes, edges }
}

fn normalize_label(label: &str) -> String {
    label.trim().to_lowercase()
}

fn register_graph_node(
    nodes: &mut Vec<StructRagGraphNode>,
    index: &mut HashMap<String, usize>,
    label: &str,
) {
    let id = normalize_label(label);
    if let Some(&i) = index.get(&id) {
        if let Some(node) = nodes.get_mut(i) {
            node.mentions += 1;
        }
    } else {
        index.insert(id.clone(), nodes.len());
        nodes.push(StructRagGraphNode {
            id,
            label: label.trim().to_string(),
            mentions: 1,
        });
    }
}

/// The relation label between two entity mentions in `sentence`: the
/// stopword-filtered words strictly between the end of `first` and the
/// start of `second`, joined with `_`; `"related_to"` when that text is
/// empty or either span cannot be located.
fn extract_relation_between(sentence: &str, first: &str, second: &str) -> String {
    let fallback = "related_to".to_string();
    let Some(start1) = sentence.find(first) else {
        return fallback;
    };
    let after_first = start1 + first.len();
    let Some(tail) = sentence.get(after_first..) else {
        return fallback;
    };
    let Some(rel_len) = tail.find(second) else {
        return fallback;
    };
    let between = &tail[..rel_len];
    let words: Vec<String> = tokenize(between)
        .into_iter()
        .filter(|w| !is_stopword(w))
        .collect();
    if words.is_empty() {
        fallback
    } else {
        words.join("_")
    }
}

// ── Tree ──────────────────────────────────────────────────────────────────────

/// Phrase patterns denoting `"<child> is-a <parent>"`.
const HIER_IS_A: &[&str] = &[" is a type of ", " is a kind of ", " is a subclass of "];
/// Phrase patterns denoting `"<child> is-part-of <parent>"`.
const HIER_PART_OF: &[&str] = &[" is part of ", " belongs to "];
/// Phrase patterns denoting `"<parent> consists-of <children...>"`.
const HIER_CONSISTS: &[&str] = &[" consists of ", " comprises "];
/// Phrase patterns denoting `"<parent> includes <children...>"`.
const HIER_INCLUDES: &[&str] = &[" includes ", " contains "];

/// A single parent/children relation extracted from one sentence.
struct TreeRelation {
    parent: String,
    children: Vec<String>,
}

/// Extract at most one [`TreeRelation`] from `sentence` by testing each
/// hierarchy phrase pattern in turn; the first pattern that matches wins.
fn extract_tree_relations(sentence: &str) -> Vec<TreeRelation> {
    for pat in HIER_IS_A.iter().chain(HIER_PART_OF.iter()) {
        if let Some(pos) = find_ci(sentence, pat) {
            let child = strip_leading_article(sentence[..pos].trim());
            let parent = strip_leading_article(
                sentence[pos + pat.len()..]
                    .trim()
                    .trim_end_matches('.')
                    .trim(),
            );
            if !child.is_empty() && !parent.is_empty() {
                return vec![TreeRelation {
                    parent: parent.to_string(),
                    children: vec![child.to_string()],
                }];
            }
        }
    }
    for pat in HIER_CONSISTS.iter().chain(HIER_INCLUDES.iter()) {
        if let Some(pos) = find_ci(sentence, pat) {
            let parent = strip_leading_article(sentence[..pos].trim());
            let rest = sentence[pos + pat.len()..]
                .trim()
                .trim_end_matches('.')
                .trim();
            let children = split_list(rest);
            if !parent.is_empty() && !children.is_empty() {
                return vec![TreeRelation {
                    parent: parent.to_string(),
                    children,
                }];
            }
        }
    }
    Vec::new()
}

fn get_or_create_tree_node(
    nodes: &mut Vec<StructRagTreeNode>,
    index: &mut HashMap<String, usize>,
    label: &str,
    source_passage_id: &str,
) -> usize {
    let key = label.trim().to_lowercase();
    if let Some(&id) = index.get(&key) {
        return id;
    }
    let id = nodes.len();
    index.insert(key, id);
    nodes.push(StructRagTreeNode {
        id,
        label: label.trim().to_string(),
        parent: None,
        children: Vec::new(),
        depth: 0,
        source_passage_id: source_passage_id.to_string(),
    });
    id
}

/// `true` when `candidate` is `node_id` itself or lies within `node_id`'s
/// subtree — used to reject a parent assignment that would create a cycle.
fn is_descendant(nodes: &[StructRagTreeNode], candidate: usize, node_id: usize) -> bool {
    if candidate == node_id {
        return true;
    }
    nodes.get(node_id).is_some_and(|node| {
        node.children
            .iter()
            .any(|&c| is_descendant(nodes, candidate, c))
    })
}

/// Assign `parent_id` as the parent of `child_id`, unless: they are the same
/// node, `child_id` already has a parent (first relation wins — later,
/// possibly-contradictory relations are ignored, for determinism), or the
/// assignment would create a cycle.
fn assign_parent(nodes: &mut [StructRagTreeNode], child_id: usize, parent_id: usize) {
    if child_id == parent_id || nodes[child_id].parent.is_some() {
        return;
    }
    if is_descendant(nodes, parent_id, child_id) {
        return;
    }
    nodes[child_id].parent = Some(parent_id);
    if !nodes[parent_id].children.contains(&child_id) {
        nodes[parent_id].children.push(child_id);
    }
}

/// Recompute every node's `depth` via breadth-first traversal from
/// `tree.root_ids`. Nodes unreachable from any root (should not happen for a
/// tree built by [`build_tree`], but possible after aggressive truncation)
/// keep depth `0`.
fn recompute_tree_depths(tree: &mut StructRagTree) {
    let children_of: HashMap<usize, Vec<usize>> = tree
        .nodes
        .iter()
        .map(|n| (n.id, n.children.clone()))
        .collect();
    let mut depth_of: HashMap<usize, usize> = HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &root in &tree.root_ids {
        depth_of.insert(root, 0);
        queue.push_back(root);
    }
    while let Some(id) = queue.pop_front() {
        let d = depth_of.get(&id).copied().unwrap_or(0);
        if let Some(children) = children_of.get(&id) {
            for &child in children {
                if let Entry::Vacant(e) = depth_of.entry(child) {
                    e.insert(d + 1);
                    queue.push_back(child);
                }
            }
        }
    }
    for node in &mut tree.nodes {
        node.depth = depth_of.get(&node.id).copied().unwrap_or(0);
    }
}

/// Build a [`StructRagTree`] in two passes over the passages, in order:
///
/// 1. **Phrase relations** (primary): each sentence is tested against the
///    `is a type of` / `is part of` / `consists of` / `includes` family of
///    hierarchy phrases (see [`extract_tree_relations`]); a match creates or
///    reuses parent/child nodes and links them.
/// 2. **Indentation/bullet fallback**: a passage that yields no phrase
///    relation becomes a single node, nested under the most recently seen
///    node at one shallower [`leading_indent_depth`] — a standard
///    outline-parsing technique, letting pre-formatted bullet-list passages
///    contribute hierarchy even without explicit relational phrasing.
///
/// A later relation is never allowed to move a node that already has a
/// parent, nor to introduce a cycle (see [`assign_parent`]), which keeps the
/// result deterministic and well-formed regardless of passage order.
fn build_tree(passages: &[StructRagPassage]) -> StructRagTree {
    let mut nodes: Vec<StructRagTreeNode> = Vec::new();
    let mut label_index: HashMap<String, usize> = HashMap::new();
    let mut last_at_depth: Vec<Option<usize>> = Vec::new();

    for passage in passages {
        let text = passage.text.trim();
        if text.is_empty() {
            continue;
        }

        let mut passage_had_relation = false;
        for sentence in split_sentences(text) {
            for relation in extract_tree_relations(sentence) {
                passage_had_relation = true;
                let parent_id = get_or_create_tree_node(
                    &mut nodes,
                    &mut label_index,
                    &relation.parent,
                    &passage.id,
                );
                for child_label in &relation.children {
                    let child_id = get_or_create_tree_node(
                        &mut nodes,
                        &mut label_index,
                        child_label,
                        &passage.id,
                    );
                    assign_parent(&mut nodes, child_id, parent_id);
                }
            }
        }

        if !passage_had_relation {
            let depth = leading_indent_depth(text);
            let label = strip_list_marker(text).unwrap_or(text).trim();
            let label = if label.is_empty() { text } else { label };
            let node_id = get_or_create_tree_node(&mut nodes, &mut label_index, label, &passage.id);

            while last_at_depth.len() <= depth {
                last_at_depth.push(None);
            }
            if depth > 0
                && let Some(Some(parent_id)) = last_at_depth.get(depth - 1)
            {
                assign_parent(&mut nodes, node_id, *parent_id);
            }
            last_at_depth.truncate(depth + 1);
            last_at_depth[depth] = Some(node_id);
        }
    }

    let mut root_ids: Vec<usize> = nodes
        .iter()
        .filter(|n| n.parent.is_none())
        .map(|n| n.id)
        .collect();
    root_ids.sort_unstable();

    let mut tree = StructRagTree { nodes, root_ids };
    recompute_tree_depths(&mut tree);
    tree
}

// ── Catalogue ─────────────────────────────────────────────────────────────────

/// Build a [`StructRagCatalogue`] with one item per (non-blank) passage.
///
/// The item's `key` reuses [`first_entity_or_token`]. Attributes prefer
/// explicit `key: value` pairs ([`extract_key_value_pairs`]); when a passage
/// has none, the text after its first `:` becomes an `"items"` attribute
/// (e.g. `"Ingredients: flour, sugar, eggs"`), or — lacking even that — the
/// truncated passage text becomes a `"detail"` attribute. Unlike
/// [`StructRagTable`], items are not forced onto a shared column set, so
/// heterogeneous attribute sets across items are preserved as-is.
fn build_catalogue(passages: &[StructRagPassage]) -> StructRagCatalogue {
    let mut items = Vec::new();
    for passage in passages {
        let text = passage.text.trim();
        if text.is_empty() {
            continue;
        }

        let key = first_entity_or_token(text);
        let mut attributes = extract_key_value_pairs(text);
        if attributes.is_empty() {
            if let Some(colon_pos) = text.find(':') {
                let items_text = text[colon_pos + 1..].trim().trim_end_matches('.').trim();
                if !items_text.is_empty() {
                    attributes.push(("items".to_string(), items_text.to_string()));
                }
            }
            if attributes.is_empty() {
                attributes.push(("detail".to_string(), truncate_for_display(text, 96)));
            }
        }

        items.push(StructRagCatalogueItem {
            key,
            attributes,
            source_passage_id: passage.id.clone(),
        });
    }
    StructRagCatalogue { items }
}

// ── Algorithm ─────────────────────────────────────────────────────────────────

/// One candidate step gathered while scanning passages, before final
/// ordering.
struct StepCandidate {
    explicit_order: Option<usize>,
    action: String,
    source_passage_id: String,
    encounter: usize,
}

/// Build a [`StructRagAlgorithm`] by splitting every passage into sentences
/// and treating each non-blank sentence as a candidate step
/// ([`strip_step_marker`] both detects an explicit ordinal like `"Step 2:"`
/// or `"3."` and strips leading marker/connective text from the action).
///
/// When at least one candidate carries an explicit ordinal, the full step
/// list is stably re-sorted by that ordinal (ties, and steps with no
/// ordinal, keep their original encounter order); otherwise the natural
/// passage-then-sentence encounter order is already the procedural order and
/// is left untouched. Steps are re-numbered `1..=N` after ordering.
fn build_algorithm(passages: &[StructRagPassage]) -> StructRagAlgorithm {
    let mut candidates: Vec<StepCandidate> = Vec::new();
    let mut encounter = 0usize;

    for passage in passages {
        let text = passage.text.trim();
        if text.is_empty() {
            continue;
        }
        for sentence in split_sentences(text) {
            let (explicit_order, action) = strip_step_marker(sentence);
            if action.trim().is_empty() {
                continue;
            }
            candidates.push(StepCandidate {
                explicit_order,
                action,
                source_passage_id: passage.id.clone(),
                encounter,
            });
            encounter += 1;
        }
    }

    if candidates.iter().any(|c| c.explicit_order.is_some()) {
        candidates.sort_by_key(|c| (c.explicit_order.unwrap_or(usize::MAX), c.encounter));
    }

    let steps = candidates
        .into_iter()
        .enumerate()
        .map(|(i, c)| StructRagStep {
            index: i + 1,
            action: c.action,
            source_passage_id: c.source_passage_id,
        })
        .collect();

    StructRagAlgorithm { steps }
}
