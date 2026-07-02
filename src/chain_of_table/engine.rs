//! [`ChainOfTableEngine`] — the Chain-of-Table planner and drive loop.
//!
//! The engine is the reasoning controller. Given a natural-language question
//! and an initial [`CotTableState`], it repeatedly:
//!
//! 1. asks the **planner** ([`ChainOfTableEngine::plan_next`]) for the next
//!    table operation to apply, chosen by heuristic keyword analysis of the
//!    question and inspection of the current table, and
//! 2. **applies** that operation with [`super::ops::apply_with_config`],
//!    recording a [`CotOperationTrace`],
//!
//! until a stop condition is reached (a terminal aggregate scalar, no further
//! applicable operation, or the [`ChainOfTableConfig::max_steps`] budget), then
//! **extracts** a [`CotAnswer`] from the final table.
//!
//! Each operation kind is emitted at most once per run, so the chain is
//! bounded by the six-operation vocabulary and always terminates. The planner
//! is deliberately deterministic: identical `(question, table, config)` inputs
//! always yield an identical chain and answer.
//!
//! The heuristic routing, in priority order:
//!
//! | intent in the question | operation |
//! |---|---|
//! | "difference"/"ratio"/"product"/"times" over two numeric columns | `f_add_column` |
//! | a filter (`col is/=/>/< value`, "where …") | `f_select_row` |
//! | "per"/"each"/"group by" a column | `f_group_by` |
//! | an explicit sort verb, or an entity superlative ("which … most/least") | `f_sort_by` |
//! | "how many"/"total"/"average"/"max"/"min" | `f_aggregate` |
//! | an explicit projection ("columns", "only the …") | `f_select_column` |

use std::collections::HashSet;

use super::ops::apply_with_config;
use super::types::{
    ChainOfTableConfig, ChainOfTableError, ChainOfTableResult, CotAddRule, CotAggregate, CotAnswer,
    CotAnswerValue, CotCell, CotColumnType, CotComparator, CotOperationKind, CotOperationTrace,
    CotPredicate, CotTableOperation, CotTableState,
};

// ── keyword vocabularies ─────────────────────────────────────────────────────

/// Superlatives implying a descending order (largest first).
const SUPERLATIVE_DESC: &[&str] = &[
    "most", "highest", "largest", "top", "maximum", "greatest", "best", "biggest", "longest",
];

/// Superlatives implying an ascending order (smallest first).
const SUPERLATIVE_ASC: &[&str] = &[
    "least", "lowest", "smallest", "bottom", "minimum", "fewest", "worst", "shortest",
];

/// Interrogatives that ask for an *entity* (a row) rather than a value, which
/// routes a superlative to `f_sort_by` + take-top rather than an aggregate.
const ENTITY_INTERROGATIVE: &[&str] = &["which", "who", "whom"];

/// Explicit sort verbs.
const SORT_VERB: &[&str] = &["sort", "sorted", "rank", "ranked", "arrange", "arranged"];

/// Keywords for the count aggregation.
const COUNT_WORD: &[&str] = &["count", "number", "many"];

/// Keywords for the sum aggregation.
const SUM_WORD: &[&str] = &["total", "sum", "combined", "altogether"];

/// Keywords for the mean aggregation.
const MEAN_WORD: &[&str] = &["average", "mean", "avg"];

// ── ChainOfTableEngine ───────────────────────────────────────────────────────

/// Drives the Chain-of-Table pattern: plan an operation, apply it, repeat, then
/// extract the answer.
///
/// The engine holds only its [`ChainOfTableConfig`]; the table and question are
/// supplied per call to [`ChainOfTableEngine::run`].
#[derive(Debug, Clone, Default)]
pub struct ChainOfTableEngine {
    /// Configuration for this engine.
    pub config: ChainOfTableConfig,
}

impl ChainOfTableEngine {
    /// Create an engine with the given configuration.
    #[must_use]
    pub fn new(config: ChainOfTableConfig) -> Self {
        Self { config }
    }

    /// Apply a single operation to `state` using this engine's configuration.
    ///
    /// A thin wrapper over [`super::ops::apply_with_config`] that threads the
    /// engine's [`ChainOfTableConfig`].
    ///
    /// # Errors
    ///
    /// See [`super::ops::apply_with_config`].
    pub fn apply(
        &self,
        operation: &CotTableOperation,
        state: &CotTableState,
    ) -> ChainOfTableResult<CotTableState> {
        apply_with_config(operation, state, &self.config)
    }

    /// Run the full Chain-of-Table pipeline for `question` over `table`.
    ///
    /// Plans and applies operations until a terminal aggregate, an
    /// out-of-intents state, or the [`ChainOfTableConfig::max_steps`] budget,
    /// then extracts and returns the [`CotAnswer`].
    ///
    /// # Errors
    ///
    /// - [`ChainOfTableError::EmptyQuestion`] when `question` is blank.
    /// - [`ChainOfTableError::EmptyTable`] when `table` has no columns.
    /// - [`ChainOfTableError::NoOperationsEnabled`] when the configuration
    ///   enables no operations.
    /// - Any error propagated from applying an operation (see
    ///   [`super::ops::apply_with_config`]).
    pub fn run(&self, question: &str, table: &CotTableState) -> ChainOfTableResult<CotAnswer> {
        if question.trim().is_empty() {
            return Err(ChainOfTableError::EmptyQuestion);
        }
        if table.is_empty() {
            return Err(ChainOfTableError::EmptyTable);
        }
        if !CotOperationKind::all()
            .iter()
            .any(|&kind| self.config.enabled_operations.is_enabled(kind))
        {
            return Err(ChainOfTableError::NoOperationsEnabled);
        }

        let mut state = table.clone();
        let mut chain: Vec<CotOperationTrace> = Vec::new();
        let mut applied: HashSet<CotOperationKind> = HashSet::new();

        for step in 0..self.config.max_steps {
            let Some(operation) = self.plan_next(question, &state, &applied) else {
                break;
            };
            let kind = operation.kind();
            let rows_before = state.row_count();

            let next_state = self.apply(&operation, &state)?;
            let description = describe(&operation, rows_before, &next_state);
            chain.push(CotOperationTrace::new(
                step,
                operation,
                rows_before,
                next_state.row_count(),
                next_state.column_count(),
                description,
            ));
            applied.insert(kind);
            state = next_state;

            // An aggregate is a natural terminal: it reduces the table to a
            // scalar (flat) or one row per group (grouped), and nothing more
            // is planned on top of it.
            if kind == CotOperationKind::Aggregate {
                break;
            }
        }

        let value = extract_answer_value(question, &state, &chain);
        Ok(CotAnswer {
            question: question.to_string(),
            value,
            final_state: state,
            chain,
        })
    }

    /// Choose the next operation to apply, or `None` when no enabled,
    /// not-yet-applied operation matches the question.
    ///
    /// Operations are considered in [`CotOperationKind::all`] priority order;
    /// the first kind that is enabled, has not already been applied this run,
    /// and can be built from the question is returned.
    #[must_use]
    pub fn plan_next(
        &self,
        question: &str,
        state: &CotTableState,
        applied: &HashSet<CotOperationKind>,
    ) -> Option<CotTableOperation> {
        let analysis = QuestionAnalysis::new(question, &self.config);
        for kind in CotOperationKind::all() {
            if applied.contains(&kind) || !self.config.enabled_operations.is_enabled(kind) {
                continue;
            }
            let candidate = match kind {
                CotOperationKind::AddColumn => analysis.plan_add_column(state),
                CotOperationKind::SelectRow => analysis.plan_select_row(state),
                CotOperationKind::GroupBy => analysis.plan_group_by(state),
                CotOperationKind::SortBy => analysis.plan_sort_by(state),
                CotOperationKind::Aggregate => analysis.plan_aggregate(state, &self.config),
                CotOperationKind::SelectColumn => analysis.plan_select_column(state),
            };
            if candidate.is_some() {
                return candidate;
            }
        }
        None
    }
}

// ── QuestionAnalysis ─────────────────────────────────────────────────────────

/// A one-shot lexical analysis of a question, shared by every per-operation
/// planner. Holds the lowercased question, its cleaned/lowercased word tokens
/// (for word-membership tests), and its whitespace tokens (for value
/// extraction, preserving case and decimal points).
struct QuestionAnalysis {
    lower: String,
    words: Vec<String>,
    raw_tokens: Vec<String>,
    case_sensitive: bool,
}

impl QuestionAnalysis {
    /// Build the analysis for `question`.
    fn new(question: &str, config: &ChainOfTableConfig) -> Self {
        let raw_tokens: Vec<String> = question.split_whitespace().map(str::to_string).collect();
        let words: Vec<String> = raw_tokens.iter().map(|t| clean_word(t)).collect();
        Self {
            lower: question.to_lowercase(),
            words,
            raw_tokens,
            case_sensitive: config.case_sensitive,
        }
    }

    /// `true` when any cleaned word equals `needle`.
    fn has_word(&self, needle: &str) -> bool {
        self.words.iter().any(|w| w == needle)
    }

    /// `true` when any cleaned word is in `needles`.
    fn has_any_word(&self, needles: &[&str]) -> bool {
        self.words.iter().any(|w| needles.contains(&w.as_str()))
    }

    /// `true` when the lowercased question contains `phrase`.
    fn has_phrase(&self, phrase: &str) -> bool {
        self.lower.contains(phrase)
    }

    /// `true` when the question asks for an entity via a superlative
    /// ("which … most", "who … best"), which routes to `f_sort_by`.
    fn is_entity_superlative(&self) -> bool {
        self.has_any_word(ENTITY_INTERROGATIVE)
            && (self.has_any_word(SUPERLATIVE_DESC) || self.has_any_word(SUPERLATIVE_ASC))
    }

    /// Resolve a single cleaned word to a column's exact name, honouring case
    /// sensitivity. Used where a column is expected to be a lone token.
    fn column_for_word(&self, word: &str, state: &CotTableState) -> Option<String> {
        state
            .columns
            .iter()
            .find(|column| {
                if self.case_sensitive {
                    column.name == word
                } else {
                    column.name.eq_ignore_ascii_case(word)
                }
            })
            .map(|column| column.name.clone())
    }

    /// Find the column whose (lowercased) name is the longest one appearing as
    /// a substring of the question. Returns the column's exact name.
    fn find_column(&self, state: &CotTableState) -> Option<String> {
        self.find_column_where(state, |_| true)
    }

    /// Find the longest-named *numeric* column mentioned in the question.
    fn find_numeric_column(&self, state: &CotTableState) -> Option<String> {
        self.find_column_where(state, CotColumnType::is_numeric)
    }

    /// Shared column-search: longest matching name whose type passes `accept`.
    fn find_column_where(
        &self,
        state: &CotTableState,
        accept: impl Fn(CotColumnType) -> bool,
    ) -> Option<String> {
        let mut best: Option<(usize, usize, String)> = None; // (name len, position, name)
        for column in &state.columns {
            if !accept(column.column_type) {
                continue;
            }
            let name_lower = column.name.to_lowercase();
            if let Some(position) = self.lower.find(&name_lower) {
                let candidate = (name_lower.len(), position, column.name.clone());
                match &best {
                    Some((best_len, best_pos, _))
                        if *best_len > candidate.0
                            || (*best_len == candidate.0 && *best_pos <= candidate.1) => {}
                    _ => best = Some(candidate),
                }
            }
        }
        best.map(|(_, _, name)| name)
    }

    /// The names of the numeric columns mentioned in the question, ordered by
    /// first appearance.
    fn numeric_columns_in_order(&self, state: &CotTableState) -> Vec<String> {
        let mut hits: Vec<(usize, String)> = state
            .columns
            .iter()
            .filter(|column| column.column_type.is_numeric())
            .filter_map(|column| {
                self.lower
                    .find(&column.name.to_lowercase())
                    .map(|position| (position, column.name.clone()))
            })
            .collect();
        hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        hits.into_iter().map(|(_, name)| name).collect()
    }

    /// The sole numeric column, when the table has exactly one.
    #[allow(clippy::unused_self)]
    fn sole_numeric_column(&self, state: &CotTableState) -> Option<String> {
        let mut numeric = state
            .columns
            .iter()
            .filter(|column| column.column_type.is_numeric());
        let first = numeric.next()?;
        if numeric.next().is_none() {
            Some(first.name.clone())
        } else {
            None
        }
    }

    /// Resolve the numeric column a numeric aggregation should target: the one
    /// named in the question, else the sole numeric column.
    fn numeric_target(&self, state: &CotTableState) -> Option<String> {
        self.find_numeric_column(state)
            .or_else(|| self.sole_numeric_column(state))
    }

    // ── per-operation planners ───────────────────────────────────────────────

    /// Plan an `f_add_column` derivation from a "difference/ratio/product"
    /// phrase over two numeric columns.
    fn plan_add_column(&self, state: &CotTableState) -> Option<CotTableOperation> {
        if self.has_word("difference") || self.has_phrase("subtract") {
            let columns = self.numeric_columns_in_order(state);
            if columns.len() >= 2 {
                return Some(CotTableOperation::AddColumn {
                    name: "difference".to_string(),
                    rule: CotAddRule::Difference {
                        left: columns[0].clone(),
                        right: columns[1].clone(),
                    },
                });
            }
        }
        if self.has_word("ratio") || self.has_phrase("divided by") || self.has_phrase("per unit") {
            let columns = self.numeric_columns_in_order(state);
            if columns.len() >= 2 {
                return Some(CotTableOperation::AddColumn {
                    name: "ratio".to_string(),
                    rule: CotAddRule::Ratio {
                        numerator: columns[0].clone(),
                        denominator: columns[1].clone(),
                    },
                });
            }
        }
        if self.has_word("product")
            || self.has_word("times")
            || self.has_phrase("multiplied by")
            || self.has_phrase("multiply")
        {
            let columns = self.numeric_columns_in_order(state);
            if columns.len() >= 2 {
                return Some(CotTableOperation::AddColumn {
                    name: "product".to_string(),
                    rule: CotAddRule::Product {
                        columns: vec![columns[0].clone(), columns[1].clone()],
                    },
                });
            }
        }
        None
    }

    /// Plan an `f_select_row` from a `column <comparator> value` phrase, where
    /// the comparator immediately follows a column token.
    fn plan_select_row(&self, state: &CotTableState) -> Option<CotTableOperation> {
        for ci in 0..self.words.len() {
            let Some(column) = self.column_for_word(&self.words[ci], state) else {
                continue;
            };
            let Some((comparator, consumed)) = parse_comparator_at(&self.words, ci + 1) else {
                continue;
            };
            let value_index = ci + 1 + consumed;
            if value_index >= self.raw_tokens.len() {
                continue;
            }
            let value = CotCell::parse(&clean_value(&self.raw_tokens[value_index]));
            return Some(CotTableOperation::SelectRow {
                predicate: CotPredicate::new(column, comparator, value),
            });
        }
        None
    }

    /// Plan an `f_group_by` from a "per"/"each"/"group by" cue naming a column.
    fn plan_group_by(&self, state: &CotTableState) -> Option<CotTableOperation> {
        for (index, word) in self.words.iter().enumerate() {
            let is_cue = word == "per"
                || word == "each"
                || (word == "by"
                    && index > 0
                    && (self.words[index - 1] == "group" || self.words[index - 1] == "grouped"));
            if !is_cue {
                continue;
            }
            for follow in &self.words[index + 1..] {
                if let Some(column) = self.column_for_word(follow, state) {
                    return Some(CotTableOperation::GroupBy { column });
                }
            }
        }
        None
    }

    /// Plan an `f_sort_by` for an explicit sort verb or an entity superlative.
    fn plan_sort_by(&self, state: &CotTableState) -> Option<CotTableOperation> {
        let sort_verb = self.has_any_word(SORT_VERB) || self.has_phrase("order by");
        if !sort_verb && !self.is_entity_superlative() {
            return None;
        }
        let ascending = if self.has_any_word(SUPERLATIVE_ASC)
            || self.has_word("ascending")
            || self.has_word("asc")
        {
            true
        } else {
            !(self.has_any_word(SUPERLATIVE_DESC)
                || self.has_word("descending")
                || self.has_word("desc"))
        };
        let column = self
            .find_numeric_column(state)
            .or_else(|| self.find_column(state))
            .or_else(|| self.sole_numeric_column(state))?;
        Some(CotTableOperation::SortBy { column, ascending })
    }

    /// Plan an `f_aggregate` from a count/sum/mean/max/min cue, unless the
    /// question is an entity superlative (which routes to `f_sort_by`).
    fn plan_aggregate(
        &self,
        state: &CotTableState,
        config: &ChainOfTableConfig,
    ) -> Option<CotTableOperation> {
        if self.is_entity_superlative() {
            return None;
        }
        let (agg, column) = self.detect_aggregate(state, config)?;
        Some(CotTableOperation::Aggregate { column, agg })
    }

    /// Determine the aggregation function and its target column.
    fn detect_aggregate(
        &self,
        state: &CotTableState,
        config: &ChainOfTableConfig,
    ) -> Option<(CotAggregate, String)> {
        if self.has_phrase("how many")
            || self.has_phrase("number of")
            || self.has_any_word(COUNT_WORD)
        {
            let column = self
                .find_column(state)
                .or_else(|| state.columns.first().map(|c| c.name.clone()))?;
            return Some((CotAggregate::Count, column));
        }
        if self.has_any_word(SUM_WORD) {
            return Some((CotAggregate::Sum, self.numeric_target(state)?));
        }
        if self.has_any_word(MEAN_WORD) {
            return Some((CotAggregate::Mean, self.numeric_target(state)?));
        }
        if self.has_any_word(SUPERLATIVE_DESC) || self.has_word("max") || self.has_word("maximum") {
            return Some((CotAggregate::Max, self.numeric_target(state)?));
        }
        if self.has_any_word(SUPERLATIVE_ASC) || self.has_word("min") || self.has_word("minimum") {
            return Some((CotAggregate::Min, self.numeric_target(state)?));
        }
        if self.has_word("aggregate") || self.has_word("summarize") {
            let agg = config.default_aggregate;
            let column = if agg.requires_numeric() {
                self.numeric_target(state)?
            } else {
                self.find_column(state)
                    .or_else(|| state.columns.first().map(|c| c.name.clone()))?
            };
            return Some((agg, column));
        }
        None
    }

    /// Plan an `f_select_column` only for an explicit projection cue.
    fn plan_select_column(&self, state: &CotTableState) -> Option<CotTableOperation> {
        let cued = self.has_word("columns")
            || self.has_phrase("only the")
            || self.has_phrase("just the")
            || self.has_phrase("select column")
            || self.has_phrase("project");
        if !cued {
            return None;
        }
        let mut hits: Vec<(usize, String)> = state
            .columns
            .iter()
            .filter_map(|column| {
                self.lower
                    .find(&column.name.to_lowercase())
                    .map(|position| (position, column.name.clone()))
            })
            .collect();
        hits.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let columns: Vec<String> = hits.into_iter().map(|(_, name)| name).collect();
        if columns.is_empty() {
            None
        } else {
            Some(CotTableOperation::SelectColumn { columns })
        }
    }
}

// ── answer extraction ────────────────────────────────────────────────────────

/// Extract the answer value from the final table state.
///
/// A 1×1 table is a scalar. Otherwise, when the question is an entity
/// superlative and the last operation was a sort, the top row is taken (the
/// "which X has the most Y" case). Everything else returns the whole small
/// resulting table.
fn extract_answer_value(
    question: &str,
    state: &CotTableState,
    chain: &[CotOperationTrace],
) -> CotAnswerValue {
    if state.is_scalar() {
        let cell = state.cell(0, 0).cloned().unwrap_or(CotCell::Empty);
        return CotAnswerValue::Scalar(cell);
    }

    let analysis_words: Vec<String> = question.split_whitespace().map(clean_word).collect();
    let entity_superlative = analysis_words
        .iter()
        .any(|w| ENTITY_INTERROGATIVE.contains(&w.as_str()))
        && analysis_words.iter().any(|w| {
            SUPERLATIVE_DESC.contains(&w.as_str()) || SUPERLATIVE_ASC.contains(&w.as_str())
        });
    let last_was_sort = chain
        .last()
        .is_some_and(|trace| trace.operation.kind() == CotOperationKind::SortBy);

    if entity_superlative && last_was_sort && state.row_count() > 1 {
        let top = CotTableState::new(state.columns.clone(), vec![state.rows[0].clone()]);
        return CotAnswerValue::Table(top);
    }

    CotAnswerValue::Table(state.clone())
}

// ── trace descriptions ───────────────────────────────────────────────────────

/// Build a short human-readable description of an applied operation.
fn describe(operation: &CotTableOperation, rows_before: usize, after: &CotTableState) -> String {
    match operation {
        CotTableOperation::AddColumn { name, .. } => {
            format!("added column '{name}'")
        }
        CotTableOperation::SelectRow { predicate } => format!(
            "kept {} of {} rows where {} {} {}",
            after.row_count(),
            rows_before,
            predicate.column,
            predicate.comparator.as_str(),
            predicate.value.to_display_string(),
        ),
        CotTableOperation::SelectColumn { columns } => {
            format!("projected to columns [{}]", columns.join(", "))
        }
        CotTableOperation::GroupBy { column } => {
            format!("grouped rows by '{column}'")
        }
        CotTableOperation::SortBy { column, ascending } => {
            let direction = if *ascending {
                "ascending"
            } else {
                "descending"
            };
            format!("sorted by '{column}' {direction}")
        }
        CotTableOperation::Aggregate { column, agg } => {
            format!("aggregated {} of '{column}'", agg.as_str())
        }
    }
}

// ── lexical helpers ──────────────────────────────────────────────────────────

/// Trim leading/trailing non-alphanumeric characters from `token` and
/// lowercase the result — the canonical form used for word-membership and
/// column-name matching.
fn clean_word(token: &str) -> String {
    clean_value(token).to_lowercase()
}

/// Trim leading/trailing non-alphanumeric characters from `token`, preserving
/// interior characters (so decimal points inside numbers survive) and case.
fn clean_value(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// Match a comparator phrase starting at word index `start`, returning the
/// [`CotComparator`] and the number of words it consumed (1 or 2).
///
/// Two-word phrases ("greater than", "at least", …) are matched before
/// single-word ones ("is", "above", "=", …).
fn parse_comparator_at(words: &[String], start: usize) -> Option<(CotComparator, usize)> {
    if start >= words.len() {
        return None;
    }
    if start + 1 < words.len() {
        let two = format!("{} {}", words[start], words[start + 1]);
        let two_word = match two.as_str() {
            "greater than" | "more than" | "larger than" | "higher than" => Some(CotComparator::Gt),
            "less than" | "fewer than" | "lower than" | "smaller than" => Some(CotComparator::Lt),
            "at least" | "minimum of" => Some(CotComparator::Ge),
            "at most" | "maximum of" => Some(CotComparator::Le),
            "equal to" | "equals to" => Some(CotComparator::Eq),
            "not equal" | "is not" | "different from" => Some(CotComparator::Ne),
            _ => None,
        };
        if let Some(comparator) = two_word {
            return Some((comparator, 2));
        }
    }
    let one_word = match words[start].as_str() {
        "is" | "equals" | "equal" => Some(CotComparator::Eq),
        "above" | "over" | "exceeds" | "exceeding" => Some(CotComparator::Gt),
        "below" | "under" | "beneath" => Some(CotComparator::Lt),
        _ => None,
    };
    one_word.map(|comparator| (comparator, 1))
}
