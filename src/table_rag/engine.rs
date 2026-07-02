//! [`TableRagEngine`] — `TableRAG`'s two-stage retrieval: schema retrieval
//! (relevant columns), cell retrieval (relevant rows via distinct-value-
//! encoded probes), and sub-table assembly.

use std::collections::{HashMap, HashSet};

use super::types::{
    CellProbe, TableColumnSpec, TableRagConfig, TableRagError, TableRagIndex, TableRagResult,
    TableRagSubTable, TableRagTable,
};

// ── FNV-1a pseudo-embeddings (self-contained; see e.g. `semantic_router`,
// `eigenscore` for the same technique applied elsewhere in this crate) ──────

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Deterministic `FNV-1a` 64-bit hash of a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute a deterministic pseudo-embedding for `text`: whitespace-split
/// tokens are `FNV-1a`-hashed into one of `dim` buckets (incrementing a
/// per-bucket count), and the resulting histogram is L2-normalized.
///
/// Two texts that share vocabulary hash into similar histograms, giving a
/// cheap, dependency-free, fully deterministic stand-in for a real sentence
/// embedding.
#[allow(clippy::cast_possible_truncation)]
fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }

    let mut buckets = vec![0.0_f64; dim];
    for token in text.split_whitespace() {
        let idx = (fnv1a(token.as_bytes()) as usize) % dim;
        buckets[idx] += 1.0;
    }

    let norm: f64 = buckets.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return vec![0.0; dim];
    }
    buckets.iter().map(|x| (x / norm) as f32).collect()
}

/// Cosine similarity between two equal-length pseudo-embeddings, clamped to
/// `[-1.0, 1.0]` to absorb floating-point rounding from normalization.
///
/// Returns `0.0` for empty or mismatched-length inputs (never panics).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(-1.0, 1.0)
}

// ── lexical helpers ──────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep
/// non-empty fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The distinct token vocabulary of `text` (see [`tokenize`]).
fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

/// Stopwords excluded when expanding a query into cell-retrieval probe
/// candidates (kept small and specific to this module, per this crate's
/// convention of each module owning its own lexical toolkit).
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "with", "by", "from", "as", "it", "this", "that", "what",
    "which", "who", "whom", "does", "do", "did",
];

/// `(original index, score)` pairs ranked best-first: either an index into
/// `TableSchema::columns` (schema retrieval) or an index into
/// `TableRagTable::rows` (cell retrieval).
type ScoredIndices = Vec<(usize, f32)>;

// ── column scoring (schema retrieval) ────────────────────────────────────

/// Token-overlap ratio between `query_tokens` and `column_name`'s own
/// (underscore/hyphen-split) tokens, in `[0.0, 1.0]`.
#[allow(clippy::cast_precision_loss)]
fn column_lexical_overlap(query_tokens: &HashSet<String>, column_name: &str) -> f32 {
    let name_tokens = token_set(&column_name.replace(['_', '-'], " "));
    if name_tokens.is_empty() {
        return 0.0;
    }
    let hits = name_tokens.intersection(query_tokens).count();
    hits as f32 / name_tokens.len() as f32
}

/// Bonus applied when the column's full (humanized) name literally appears
/// as a substring of the (lowercased) query text.
fn column_phrase_bonus(query_lower: &str, column_name: &str) -> f32 {
    let phrase = column_name.replace(['_', '-'], " ").to_lowercase();
    if !phrase.is_empty() && query_lower.contains(&phrase) {
        0.3
    } else {
        0.0
    }
}

/// Bonus applied when the query contains one of the column type's
/// distinguishing keywords (`column.column_type.keywords()`).
fn type_match_bonus(query_tokens: &HashSet<String>, column: &TableColumnSpec) -> f32 {
    if column
        .column_type
        .keywords()
        .iter()
        .any(|kw| query_tokens.contains(*kw))
    {
        0.25
    } else {
        0.0
    }
}

/// Combined schema-retrieval score for a single column: a weighted mix of a
/// lexical component (token overlap + phrase containment + column-type
/// keyword bonus, capped at `1.0`) and a pseudo-embedding cosine-similarity
/// component (floored at `0.0`).
fn column_score(
    query_lower: &str,
    query_tokens: &HashSet<String>,
    query_embedding: &[f32],
    column: &TableColumnSpec,
    config: &TableRagConfig,
) -> f32 {
    let lexical = (column_lexical_overlap(query_tokens, &column.name)
        + column_phrase_bonus(query_lower, &column.name)
        + type_match_bonus(query_tokens, column))
    .min(1.0);

    let humanized_name = column.name.replace(['_', '-'], " ");
    let embed_similarity = cosine(
        query_embedding,
        &embed(&humanized_name, config.pseudo_embedding_dim),
    )
    .max(0.0);

    config.lexical_weight * lexical + config.embedding_weight * embed_similarity
}

/// Score every column of `table` against the query, preserving schema
/// order.
fn score_table_columns(
    table: &TableRagTable,
    query_lower: &str,
    query_tokens: &HashSet<String>,
    query_embedding: &[f32],
    config: &TableRagConfig,
) -> Vec<f32> {
    table
        .schema
        .columns
        .iter()
        .map(|column| column_score(query_lower, query_tokens, query_embedding, column, config))
        .collect()
}

/// Pick the index of the table containing the single highest-scoring column
/// across the whole index (ties keep the earliest table).
///
/// Selecting by each table's *best* column (rather than e.g. summing all of
/// a table's column scores) guarantees that whenever *any* column anywhere
/// in the index clears [`TableRagConfig::min_column_score`], the selected
/// table is one that contains such a column — a table with many mediocre
/// columns can never out-rank a table with one genuinely relevant column.
fn select_table_index(per_table_scores: &[Vec<f32>]) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f32::MIN;
    for (idx, scores) in per_table_scores.iter().enumerate() {
        let table_best = scores.iter().copied().fold(f32::MIN, f32::max);
        if table_best > best_score {
            best_score = table_best;
            best_idx = idx;
        }
    }
    best_idx
}

/// Filter `scores` (one entry per column, in schema order) down to those at
/// or above [`TableRagConfig::min_column_score`], rank them best-first
/// (ties broken by ascending original column index), and truncate to
/// [`TableRagConfig::top_k_columns`].
fn ranked_relevant_columns(scores: &[f32], config: &TableRagConfig) -> ScoredIndices {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, value)| value >= config.min_column_score)
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(config.top_k_columns);
    ranked
}

// ── cell scoring (cell retrieval) ────────────────────────────────────────

/// Expand `query` into a deduplicated, deterministically ordered list of
/// probe-value candidates: the full cleaned query (its tokens rejoined with
/// single spaces — punctuation-free, so its pseudo-embedding is computed
/// from the same token vocabulary as every unigram/bigram probe below, not
/// from raw whitespace-split words), then every non-stopword unigram, then
/// every bigram of consecutive tokens.
fn expand_query_terms(query: &str) -> Vec<String> {
    let tokens = tokenize(query);
    let mut seen: HashSet<String> = HashSet::new();
    let mut terms: Vec<String> = Vec::new();

    let full = tokens.join(" ");
    if !full.is_empty() && seen.insert(full.clone()) {
        terms.push(full);
    }

    for token in &tokens {
        if !STOPWORDS.contains(&token.as_str()) && seen.insert(token.clone()) {
            terms.push(token.clone());
        }
    }

    for pair in tokens.windows(2) {
        let bigram = format!("{} {}", pair[0], pair[1]);
        if seen.insert(bigram.clone()) {
            terms.push(bigram);
        }
    }

    terms
}

/// Score a candidate `probe` against an actual distinct cell `value`.
///
/// Three tiers, deliberately non-overlapping so the source of a match is
/// unambiguous from its score: an exact (case-insensitive) match scores
/// `1.0`; a containment match (either direction) scores in `[0.55, 0.90]`
/// depending on the length ratio; otherwise the pseudo-embedding cosine
/// similarity is used, damped to `[0.0, 0.5]` so a fuzzy match can never
/// outscore a genuine containment or exact match.
#[allow(clippy::cast_precision_loss)]
fn value_match_score(probe: &str, value: &str, dim: usize) -> f32 {
    let probe = probe.trim().to_lowercase();
    let value = value.trim().to_lowercase();
    if probe.is_empty() || value.is_empty() {
        return 0.0;
    }
    if probe == value {
        return 1.0;
    }
    if value.contains(probe.as_str()) || probe.contains(value.as_str()) {
        let shorter = probe.len().min(value.len()) as f32;
        let longer = probe.len().max(value.len()) as f32;
        return 0.55 + 0.35 * (shorter / longer);
    }
    let embed_similarity = cosine(&embed(&probe, dim), &embed(&value, dim)).max(0.0);
    embed_similarity * 0.5
}

/// A column's distinct-value dictionary, capped at
/// [`TableRagConfig::max_distinct_values_per_column`] — the mechanism that
/// bounds cell-retrieval matching cost on large columns.
///
/// Values are tracked in first-seen row order; once the cap is reached, new
/// (never-before-seen) values are no longer tracked (their occurrences
/// become unreachable through probe matching, which is the intended
/// cost/recall trade-off), while further occurrences of an *already
/// tracked* value keep accumulating row indices.
#[derive(Debug)]
struct DistinctValueIndex {
    /// Distinct values tracked, in first-seen order (length `<= cap`).
    values: Vec<String>,
    /// `rows_by_value[i]` lists every row index holding `values[i]`.
    rows_by_value: Vec<Vec<usize>>,
    /// `true` when the column holds more distinct values than the cap
    /// allowed tracking.
    capped: bool,
}

impl DistinctValueIndex {
    /// Build the distinct-value dictionary for `table`'s column
    /// `column_index`, tracking at most `cap` distinct values.
    fn build(table: &TableRagTable, column_index: usize, cap: usize) -> Self {
        let mut values: Vec<String> = Vec::new();
        let mut rows_by_value: Vec<Vec<usize>> = Vec::new();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        let mut capped = false;

        for (row_index, row) in table.rows.iter().enumerate() {
            let Some(cell) = row.get(column_index) else {
                continue;
            };
            let key = cell.to_lowercase();
            if let Some(&pos) = index_of.get(&key) {
                rows_by_value[pos].push(row_index);
            } else if values.len() < cap {
                index_of.insert(key, values.len());
                values.push(cell.clone());
                rows_by_value.push(vec![row_index]);
            } else {
                capped = true;
            }
        }

        Self {
            values,
            rows_by_value,
            capped,
        }
    }

    /// The tracked distinct value that best matches `probe`, together with
    /// the row indices it appears in. Ties keep the earliest-tracked value.
    /// Returns `None` when no values are tracked at all.
    fn best_match(&self, probe: &str, dim: usize) -> Option<(f32, &[usize])> {
        let mut best: Option<(f32, usize)> = None;
        for (i, value) in self.values.iter().enumerate() {
            let score = value_match_score(probe, value, dim);
            let replace = match best {
                None => true,
                Some((best_score, _)) => score > best_score,
            };
            if replace {
                best = Some((score, i));
            }
        }
        best.map(|(score, i)| (score, self.rows_by_value[i].as_slice()))
    }
}

/// Rank `(row index, score)` pairs best-first (ties broken by ascending row
/// index) and truncate to [`TableRagConfig::top_k_rows`].
fn ranked_relevant_rows(mut rows: ScoredIndices, config: &TableRagConfig) -> ScoredIndices {
    rows.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    rows.truncate(config.top_k_rows);
    rows
}

// ── sub-table assembly ───────────────────────────────────────────────────

/// Intersect `ranked_columns` × `ranked_rows` of `table` into a
/// [`TableRagSubTable`], preserving each ranking's order and recording
/// provenance.
fn assemble_sub_table(
    table: &TableRagTable,
    ranked_columns: &[(usize, f32)],
    ranked_rows: &[(usize, f32)],
) -> TableRagSubTable {
    let column_indices: Vec<usize> = ranked_columns.iter().map(|&(i, _)| i).collect();
    let columns: Vec<TableColumnSpec> = column_indices
        .iter()
        .map(|&i| table.schema.columns[i].clone())
        .collect();

    let row_indices: Vec<usize> = ranked_rows.iter().map(|&(r, _)| r).collect();
    let rows: Vec<Vec<String>> = row_indices
        .iter()
        .map(|&r| {
            column_indices
                .iter()
                .map(|&c| table.rows[r][c].clone())
                .collect()
        })
        .collect();

    TableRagSubTable {
        table_name: table.name.clone(),
        columns,
        column_indices,
        rows,
        row_indices,
    }
}

// ── TableRagEngine ────────────────────────────────────────────────────────

/// Drives `TableRAG`'s two-stage retrieval over a [`TableRagIndex`]: schema
/// retrieval (relevant columns), cell retrieval (relevant rows), and
/// sub-table assembly.
#[derive(Debug, Clone, Default)]
pub struct TableRagEngine {
    /// The indexed tables this engine retrieves over.
    pub index: TableRagIndex,
    /// Retrieval configuration.
    pub config: TableRagConfig,
}

impl TableRagEngine {
    /// Create a new engine over `index` with the given `config`.
    #[must_use]
    pub fn new(index: TableRagIndex, config: TableRagConfig) -> Self {
        Self { index, config }
    }

    /// Create a new engine over `index` with default configuration.
    #[must_use]
    pub fn with_default_config(index: TableRagIndex) -> Self {
        Self::new(index, TableRagConfig::default())
    }

    /// Run the full two-stage retrieval pipeline for `query_text`: schema
    /// retrieval selects the most relevant table and its most relevant
    /// columns; cell retrieval expands the query into [`CellProbe`]s and
    /// matches them against those columns' (distinct-value-encoded) data to
    /// select the most relevant rows; sub-table assembly intersects the two
    /// into a compact, provenance-carrying [`TableRagResult`].
    ///
    /// A query whose columns match but whose cell probes match no row still
    /// succeeds, returning a [`TableRagResult`] whose sub-table has columns
    /// but zero rows (see [`TableRagResult::is_empty`]) — cell retrieval
    /// finding nothing is a legitimate outcome, not a failure.
    ///
    /// # Errors
    ///
    /// - [`TableRagError::EmptyQuery`] when `query_text` is blank.
    /// - [`TableRagError::EmptyIndex`] when the engine's index has no
    ///   tables.
    /// - [`TableRagError::NoRelevantColumns`] when no column, in any table,
    ///   scores at or above [`TableRagConfig::min_column_score`] for this
    ///   query.
    pub fn query(&self, query_text: &str) -> Result<TableRagResult, TableRagError> {
        let trimmed = query_text.trim();
        if trimmed.is_empty() {
            return Err(TableRagError::EmptyQuery);
        }
        if self.index.is_empty() {
            return Err(TableRagError::EmptyIndex);
        }

        let (table, relevant_columns) = self.retrieve_schema(trimmed)?;
        let (ranked_rows, probes, capped_columns) =
            self.retrieve_cells(table, &relevant_columns, trimmed);

        let sub_table = assemble_sub_table(table, &relevant_columns, &ranked_rows);
        let encoded_text = sub_table.to_encoded_text();
        let column_scores = relevant_columns
            .iter()
            .map(|&(i, score)| (table.schema.columns[i].name.clone(), score))
            .collect();

        Ok(TableRagResult {
            query: trimmed.to_string(),
            sub_table,
            encoded_text,
            column_scores,
            row_scores: ranked_rows,
            probes,
            capped_columns,
        })
    }

    /// Stage 1 — schema retrieval: score every column of every table against
    /// `trimmed_query`, select the table containing the single
    /// highest-scoring column (see [`select_table_index`]), then rank and
    /// filter that table's own columns.
    ///
    /// # Errors
    ///
    /// [`TableRagError::NoRelevantColumns`] when the selected table's
    /// filtered column list is empty (equivalently: no column anywhere in
    /// the index cleared the threshold).
    fn retrieve_schema(
        &self,
        trimmed_query: &str,
    ) -> Result<(&TableRagTable, ScoredIndices), TableRagError> {
        let query_lower = trimmed_query.to_lowercase();
        let query_tokens = token_set(trimmed_query);
        let query_embedding = embed(trimmed_query, self.config.pseudo_embedding_dim);

        let per_table_scores: Vec<Vec<f32>> = self
            .index
            .tables
            .iter()
            .map(|table| {
                score_table_columns(
                    table,
                    &query_lower,
                    &query_tokens,
                    &query_embedding,
                    &self.config,
                )
            })
            .collect();

        let table_idx = select_table_index(&per_table_scores);
        let table = &self.index.tables[table_idx];
        let relevant_columns = ranked_relevant_columns(&per_table_scores[table_idx], &self.config);

        if relevant_columns.is_empty() {
            return Err(TableRagError::NoRelevantColumns);
        }

        Ok((table, relevant_columns))
    }

    /// Stage 2 — cell retrieval: expand `trimmed_query` into probe-value
    /// candidates, match each against every relevant column's
    /// distinct-value dictionary, and aggregate per-row scores (summed
    /// across distinct matching columns, taking each column's single best
    /// probe match so overlapping probes within one column cannot
    /// double-count).
    ///
    /// Returns the ranked `(row index, score)` pairs, every [`CellProbe`]
    /// that cleared [`TableRagConfig::min_cell_score`] (best-first), and the
    /// names of any columns whose distinct-value dictionary was capped.
    fn retrieve_cells(
        &self,
        table: &TableRagTable,
        relevant_columns: &[(usize, f32)],
        trimmed_query: &str,
    ) -> (ScoredIndices, Vec<CellProbe>, Vec<String>) {
        let probe_terms = expand_query_terms(trimmed_query);
        let row_count = table.row_count();
        let mut row_scores: Vec<f32> = vec![0.0; row_count];
        let mut touched: Vec<bool> = vec![false; row_count];
        let mut probes: Vec<CellProbe> = Vec::new();
        let mut capped_columns: Vec<String> = Vec::new();

        for &(col_idx, _) in relevant_columns {
            let column = &table.schema.columns[col_idx];
            let dict = DistinctValueIndex::build(
                table,
                col_idx,
                self.config.max_distinct_values_per_column,
            );
            if dict.capped {
                capped_columns.push(column.name.clone());
            }

            let mut best_in_column: Vec<Option<f32>> = vec![None; row_count];
            for term in &probe_terms {
                let Some((score, rows)) = dict.best_match(term, self.config.pseudo_embedding_dim)
                else {
                    continue;
                };
                if score < self.config.min_cell_score {
                    continue;
                }
                probes.push(CellProbe::new(column.name.clone(), term.clone(), score));
                for &row in rows {
                    let slot = &mut best_in_column[row];
                    let replace = match slot {
                        Some(current) => score > *current,
                        None => true,
                    };
                    if replace {
                        *slot = Some(score);
                    }
                }
            }

            for (row, slot) in best_in_column.into_iter().enumerate() {
                if let Some(score) = slot {
                    row_scores[row] += score;
                    touched[row] = true;
                }
            }
        }

        let candidate_rows: Vec<(usize, f32)> = (0..row_count)
            .filter(|&i| touched[i])
            .map(|i| (i, row_scores[i]))
            .collect();
        let ranked_rows = ranked_relevant_rows(candidate_rows, &self.config);

        probes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.column_name.cmp(&b.column_name))
                .then(a.probe_value.cmp(&b.probe_value))
        });

        (ranked_rows, probes, capped_columns)
    }
}
