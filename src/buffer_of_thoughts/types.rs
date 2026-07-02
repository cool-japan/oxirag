//! Types and traits for the `buffer_of_thoughts` module.
//!
//! Buffer of Thoughts (Yang et al. 2024, "Buffer of Thoughts: Thought-Augmented
//! Reasoning with Large Language Models") maintains a **persistent, growing**
//! meta-buffer of distilled `ThoughtTemplate`s. Unlike a fixed module bank, the
//! buffer accumulates value over time: every successfully solved problem is a
//! candidate for distillation into a new, reusable template (or reinforcement
//! of an existing one), so the buffer gets *better* the more problems it sees.
//!
//! # Problem signature scheme
//!
//! A [`ProblemSignature`] is a structural fingerprint of the *kind* of problem,
//! never the literal problem text. It has two components:
//!
//! - **`operation_tags`** — a set of coarse operation-type labels detected via
//!   keyword matching (e.g. `"arithmetic"`, `"comparison"`, `"causal"`,
//!   `"combinatorial"`, `"classification"`, `"search_lookup"`,
//!   `"optimization"`, `"sequence"`, `"logical_deduction"`, or `"generic"` when
//!   nothing else matches).
//! - **`structural_features`** — shape features of the problem statement (e.g.
//!   `"has_numbers"`, `"is_question"`, `"multi_clause"`, `"negation"`,
//!   `"list_like"`, and a token-count bucket: `"short_form"`, `"medium_form"`,
//!   or `"long_form"`).
//!
//! Both sets are computed by keyword/shape detectors ([`detect_operation_tags`]
//! and [`detect_structural_features`]) that operate on lower-cased, tokenized
//! text — never on raw problem strings — so two superficially different
//! problems of the *same kind* (e.g. "What is the sum of 12 and 7?" and "What
//! is the sum of 30 and 5?") produce identical or near-identical signatures.
//!
//! # Similarity
//!
//! [`ProblemSignature::similarity`] combines a weighted Jaccard overlap of both
//! tag sets: operation tags are weighted `0.7` (the operation kind is the
//! dominant signal for whether a reasoning strategy transfers) and structural
//! features `0.3`. This is the "structural-feature-overlap" option explicitly
//! permitted for this module (as an alternative to an FNV-1a pseudo-embedding
//! cosine similarity, which is used elsewhere in this crate, e.g.
//! `crate::semantic_router`). Structural overlap was chosen here because it is
//! fully interpretable and lets retrieval thresholds be reasoned about exactly
//! in terms of *which* tags matched, which matters for a buffer whose entire
//! value proposition is precise, explainable reuse.

use std::collections::{BTreeSet, HashSet};

use thiserror::Error;

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Tokenize `text` into lower-cased alphanumeric tokens of length ≥ 2.
///
/// The split boundary is any non-alphanumeric character. Tokens shorter than
/// two characters are discarded.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(str::to_lowercase)
}

/// Extract every maximal run of digits (optionally containing a decimal
/// point) from `text`, in order of appearance.
///
/// Used both to detect the `"has_numbers"` structural feature and to derive
/// the per-instance `{opN}` placeholders substituted during instantiation and
/// distillation.
pub(crate) fn extract_numbers(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_digit() && c != '.')
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .map(str::to_string)
        .collect()
}

/// Returns `true` if any keyword in `keywords` appears in `tokens`.
fn any_keyword(tokens: &HashSet<String>, keywords: &[&str]) -> bool {
    keywords.iter().any(|k| tokens.contains(*k))
}

// ── Operation-tag detection ───────────────────────────────────────────────────

const ARITHMETIC_KEYWORDS: &[&str] = &[
    "sum",
    "total",
    "add",
    "plus",
    "subtract",
    "minus",
    "difference",
    "multiply",
    "product",
    "times",
    "divide",
    "quotient",
    "percent",
    "percentage",
    "average",
    "mean",
    "calculate",
    "compute",
];
const COMPARISON_KEYWORDS: &[&str] = &[
    "more",
    "less",
    "greater",
    "smaller",
    "compare",
    "comparison",
    "versus",
    "maximum",
    "minimum",
    "largest",
    "smallest",
    "biggest",
    "highest",
    "lowest",
];
const SEQUENCE_KEYWORDS: &[&str] = &["next", "sequence", "pattern", "series", "continue"];
const LOGICAL_KEYWORDS: &[&str] = &[
    "therefore",
    "thus",
    "implies",
    "valid",
    "invalid",
    "syllogism",
    "premise",
    "conclude",
    "conclusion",
];
const CAUSAL_KEYWORDS: &[&str] = &[
    "why", "cause", "causes", "caused", "effect", "result", "results", "leads", "due", "reason",
];
const COMBINATORIAL_KEYWORDS: &[&str] = &[
    "ways",
    "combinations",
    "combination",
    "permutations",
    "permutation",
    "choose",
    "arrange",
    "arrangement",
];
const CLASSIFICATION_KEYWORDS: &[&str] = &[
    "classify",
    "category",
    "categorize",
    "type",
    "kind",
    "label",
    "group",
];
const SEARCH_KEYWORDS: &[&str] = &["who", "where", "when", "find", "locate", "which", "name"];
const OPTIMIZATION_KEYWORDS: &[&str] = &[
    "best",
    "optimal",
    "maximize",
    "maximise",
    "minimize",
    "minimise",
    "efficient",
    "optimum",
];

/// Detect coarse operation-type tags for `problem` via keyword matching.
///
/// Returns a sorted, de-duplicated set of tags such as `"arithmetic"` or
/// `"causal"`. When no keyword category matches, the singleton set
/// `{"generic"}` is returned so every problem has at least one operation tag.
pub(crate) fn detect_operation_tags(problem: &str) -> BTreeSet<String> {
    let tokens: HashSet<String> = tokenize(problem).collect();
    let mut tags = BTreeSet::new();

    if any_keyword(&tokens, ARITHMETIC_KEYWORDS) {
        tags.insert("arithmetic".to_string());
    }
    if any_keyword(&tokens, COMPARISON_KEYWORDS) {
        tags.insert("comparison".to_string());
    }
    if any_keyword(&tokens, SEQUENCE_KEYWORDS) {
        tags.insert("sequence".to_string());
    }
    if any_keyword(&tokens, LOGICAL_KEYWORDS) {
        tags.insert("logical_deduction".to_string());
    }
    if any_keyword(&tokens, CAUSAL_KEYWORDS) {
        tags.insert("causal".to_string());
    }
    if any_keyword(&tokens, COMBINATORIAL_KEYWORDS) {
        tags.insert("combinatorial".to_string());
    }
    if any_keyword(&tokens, CLASSIFICATION_KEYWORDS) {
        tags.insert("classification".to_string());
    }
    if any_keyword(&tokens, SEARCH_KEYWORDS) {
        tags.insert("search_lookup".to_string());
    }
    if any_keyword(&tokens, OPTIMIZATION_KEYWORDS) {
        tags.insert("optimization".to_string());
    }

    if tags.is_empty() {
        tags.insert("generic".to_string());
    }
    tags
}

// ── Structural-feature detection ──────────────────────────────────────────────

const QUESTION_WORDS: &[&str] = &[
    "what", "why", "how", "who", "where", "when", "which", "is", "are", "does", "do", "can",
    "could", "would", "will", "should",
];

fn is_question(problem: &str) -> bool {
    if problem.trim().ends_with('?') {
        return true;
    }
    tokenize(problem)
        .next()
        .is_some_and(|first| QUESTION_WORDS.contains(&first.as_str()))
}

fn is_multi_clause(problem: &str) -> bool {
    problem.contains(',')
        || problem.contains(';')
        || tokenize(problem).any(|t| t == "and" || t == "or")
}

fn has_negation(problem: &str) -> bool {
    let lower = problem.to_lowercase();
    let tokens: HashSet<String> = tokenize(problem).collect();
    tokens.contains("not") || tokens.contains("never") || lower.contains("n't")
}

fn is_list_like(problem: &str) -> bool {
    problem.matches(',').count() >= 2
}

fn length_bucket(problem: &str) -> &'static str {
    match tokenize(problem).count() {
        0..=6 => "short_form",
        7..=15 => "medium_form",
        _ => "long_form",
    }
}

/// Detect structural shape features for `problem`.
///
/// Returns a sorted, de-duplicated set of features such as `"has_numbers"`,
/// `"is_question"`, `"multi_clause"`, `"negation"`, `"list_like"`, and exactly
/// one length bucket (`"short_form"`, `"medium_form"`, or `"long_form"`).
pub(crate) fn detect_structural_features(problem: &str) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    if !extract_numbers(problem).is_empty() {
        features.insert("has_numbers".to_string());
    }
    if is_question(problem) {
        features.insert("is_question".to_string());
    }
    if is_multi_clause(problem) {
        features.insert("multi_clause".to_string());
    }
    if has_negation(problem) {
        features.insert("negation".to_string());
    }
    if is_list_like(problem) {
        features.insert("list_like".to_string());
    }
    features.insert(length_bucket(problem).to_string());
    features
}

/// Weighted Jaccard similarity between two sorted, de-duplicated tag lists.
///
/// Two empty lists are defined as perfectly similar (`1.0`); an empty list
/// compared against a non-empty one is defined as maximally dissimilar
/// (`0.0`), since one has no tags in common with the other by construction.
fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: HashSet<&str> = b.iter().map(String::as_str).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let ratio = intersection as f32 / union as f32;
        ratio
    }
}

// ── ProblemSignature ──────────────────────────────────────────────────────────

/// A structural fingerprint of the *kind* of problem a
/// [`ThoughtTemplate`](crate::buffer_of_thoughts::types::ThoughtTemplate)
/// applies to — never the literal problem text.
///
/// See the module documentation for the full signature scheme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProblemSignature {
    /// Sorted, de-duplicated operation-type tags (e.g. `"arithmetic"`).
    pub operation_tags: Vec<String>,
    /// Sorted, de-duplicated structural shape features (e.g.
    /// `"has_numbers"`).
    pub structural_features: Vec<String>,
}

impl ProblemSignature {
    /// Relative weight given to `operation_tags` overlap in
    /// [`ProblemSignature::similarity`].
    pub const OPERATION_WEIGHT: f32 = 0.7;
    /// Relative weight given to `structural_features` overlap in
    /// [`ProblemSignature::similarity`].
    pub const STRUCTURAL_WEIGHT: f32 = 0.3;

    /// Compute the structural signature of `problem`.
    ///
    /// This never stores or re-derives the literal problem text; only the
    /// detected operation tags and structural features are retained.
    #[must_use]
    pub fn compute(problem: &str) -> Self {
        Self {
            operation_tags: detect_operation_tags(problem).into_iter().collect(),
            structural_features: detect_structural_features(problem).into_iter().collect(),
        }
    }

    /// Weighted-Jaccard similarity to `other`, in `[0.0, 1.0]`.
    ///
    /// `operation_tags` overlap contributes [`ProblemSignature::OPERATION_WEIGHT`]
    /// and `structural_features` overlap contributes
    /// [`ProblemSignature::STRUCTURAL_WEIGHT`] of the final score.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        let op_sim = jaccard(&self.operation_tags, &other.operation_tags);
        let struct_sim = jaccard(&self.structural_features, &other.structural_features);
        Self::OPERATION_WEIGHT * op_sim + Self::STRUCTURAL_WEIGHT * struct_sim
    }

    /// Render a canonical, human-readable form of this signature, useful for
    /// debugging and logging.
    #[must_use]
    pub fn canonical(&self) -> String {
        format!(
            "op:[{}]|struct:[{}]",
            self.operation_tags.join(","),
            self.structural_features.join(",")
        )
    }
}

// ── EvictionPolicy ────────────────────────────────────────────────────────────

/// Eviction policy applied by [`ThoughtBuffer`](crate::buffer_of_thoughts::engine::ThoughtBuffer)
/// when [`BotConfig::max_buffer_size`] is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Evict the least-recently-used template (smallest `last_used_tick`).
    ///
    /// Ties are broken by evicting the lowest `id` (the older template).
    #[default]
    Lru,
    /// Evict the template with the lowest `success_rate`.
    ///
    /// Ties are broken by the fewest `usage_count`, then by the lowest `id`.
    LowestSuccessRate,
}

// ── BotConfig ──────────────────────────────────────────────────────────────────

/// Configuration for [`BotEngine`](crate::buffer_of_thoughts::engine::BotEngine).
#[derive(Debug, Clone, PartialEq)]
pub struct BotConfig {
    /// Minimum [`ProblemSignature::similarity`] required for a buffered
    /// template to be considered a match during retrieval.
    ///
    /// Defaults to `0.5`. Clamped to `[0.0, 1.0]` by
    /// [`BotConfig::with_similarity_threshold`].
    pub similarity_threshold: f32,

    /// Maximum number of templates the buffer retains before evicting.
    ///
    /// Defaults to `200`. Clamped to at least `1` by
    /// [`BotConfig::with_max_buffer_size`].
    pub max_buffer_size: usize,

    /// Eviction policy applied once `max_buffer_size` is reached.
    ///
    /// Defaults to [`EvictionPolicy::Lru`].
    pub eviction_policy: EvictionPolicy,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            max_buffer_size: 200,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

impl BotConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the similarity threshold, clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn with_similarity_threshold(mut self, similarity_threshold: f32) -> Self {
        self.similarity_threshold = similarity_threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum buffer size, clamped to at least `1`.
    #[must_use]
    pub fn with_max_buffer_size(mut self, max_buffer_size: usize) -> Self {
        self.max_buffer_size = max_buffer_size.max(1);
        self
    }

    /// Set the eviction policy.
    #[must_use]
    pub fn with_eviction_policy(mut self, eviction_policy: EvictionPolicy) -> Self {
        self.eviction_policy = eviction_policy;
        self
    }
}

// ── ThoughtTemplate ────────────────────────────────────────────────────────────

/// A distilled, reusable reasoning scaffold stored in the
/// [`ThoughtBuffer`](crate::buffer_of_thoughts::engine::ThoughtBuffer).
///
/// `template_text` is a generalized strategy with two kinds of placeholders:
/// a literal `{problem}` placeholder (substituted with the new problem's raw
/// text) and numbered `{op1}`, `{op2}`, … placeholders (substituted with the
/// numeric operands extracted from the new problem, in order). The internal
/// `instantiate_template` helper performs this substitution during
/// [`crate::buffer_of_thoughts::engine::BotEngine::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct ThoughtTemplate {
    /// Unique, monotonically-assigned identifier within a
    /// [`ThoughtBuffer`](crate::buffer_of_thoughts::engine::ThoughtBuffer).
    pub id: u64,
    /// The structural signature of the problem *kind* this template solves.
    pub signature: ProblemSignature,
    /// The generalized, placeholder-bearing reasoning scaffold.
    pub template_text: String,
    /// Number of times this template has been (re)used, including the
    /// original distillation.
    pub usage_count: usize,
    /// Number of those uses that were reported successful.
    pub success_count: usize,
    /// `success_count as f32 / usage_count as f32`, kept in sync internally
    /// whenever the template is reinforced via
    /// [`crate::buffer_of_thoughts::engine::BotEngine::record_outcome`].
    pub success_rate: f32,
    /// Logical clock value (not wall-clock time) of the last use, for LRU
    /// eviction.
    pub last_used_tick: u64,
}

impl ThoughtTemplate {
    /// Create a freshly-distilled template.
    ///
    /// A template is born from one successful solve, so it starts with
    /// `usage_count = 1`, `success_count = 1`, and `success_rate = 1.0`.
    pub(crate) fn new(
        id: u64,
        signature: ProblemSignature,
        template_text: String,
        tick: u64,
    ) -> Self {
        Self {
            id,
            signature,
            template_text,
            usage_count: 1,
            success_count: 1,
            success_rate: 1.0,
            last_used_tick: tick,
        }
    }

    /// Record one more (re)use of this template, updating `usage_count`,
    /// `success_count`, `success_rate`, and `last_used_tick`.
    pub(crate) fn reinforce(&mut self, success: bool, tick: u64) {
        self.usage_count = self.usage_count.saturating_add(1);
        if success {
            self.success_count = self.success_count.saturating_add(1);
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = self.success_count as f32 / self.usage_count as f32;
        self.success_rate = rate;
        self.last_used_tick = tick;
    }
}

/// Substitute the `{problem}` and `{op1}`, `{op2}`, … placeholders in
/// `template_text` with values derived from `problem`.
///
/// `{problem}` is replaced with the literal `problem` text. Each `{opN}`
/// placeholder present in `template_text` is replaced with the `N`-th number
/// extracted from `problem` (1-indexed); if `problem` has fewer numbers than
/// the template references, the remaining placeholders are replaced with
/// `"?"` rather than left dangling, so the instantiated scaffold is always
/// well-formed prose. Scanning stops after 64 consecutive placeholders (a
/// generous bound no legitimate template should approach) to guarantee
/// termination even for adversarial template text.
pub(crate) fn instantiate_template(template_text: &str, problem: &str) -> String {
    let mut result = template_text.replace("{problem}", problem);
    let numbers = extract_numbers(problem);

    for n in 1..=64usize {
        let placeholder = format!("{{op{n}}}");
        if !result.contains(&placeholder) {
            break;
        }
        let value = numbers.get(n - 1).map_or("?", String::as_str);
        result = result.replace(&placeholder, value);
    }

    result
}

/// The generic, stateless fallback scaffold used when no buffered template is
/// similar enough to the new problem (a "no match" outcome).
///
/// This constant is never stored in the buffer and never grows — it exists so
/// the engine can still produce a reasonable scaffold instead of forcing a
/// bad match.
pub const DEFAULT_TEMPLATE_TEXT: &str = "Step 1: Read the problem carefully and identify what is being asked.\nStep 2: Identify the relevant facts, values, and relationships.\nStep 3: Apply appropriate step-by-step reasoning to derive intermediate results.\nStep 4: State the final answer for: {problem}";

// ── SolveOutcome ───────────────────────────────────────────────────────────────

/// Caller-supplied outcome signal for a solved problem, used to decide
/// whether to distill a new template or reinforce a reused one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveOutcome {
    /// The produced answer was correct / accepted.
    Success,
    /// The produced answer was incorrect / rejected.
    Failure,
}

// ── DistillAction ──────────────────────────────────────────────────────────────

/// What [`BotEngine::record_outcome`](crate::buffer_of_thoughts::engine::BotEngine::record_outcome)
/// did to the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillAction {
    /// A brand-new template was distilled and inserted, with this `id`.
    Distilled(u64),
    /// An existing template (this `id`) was reinforced (its `usage_count` /
    /// `success_rate` updated).
    Reinforced(u64),
    /// No buffer write-back occurred (a fresh, unmatched problem that was not
    /// solved successfully — distilling from a failed solve would pollute
    /// the buffer with an ungeneralizable template, so it is skipped).
    Skipped,
}

// ── BotSolveResult ─────────────────────────────────────────────────────────────

/// The result of [`BotEngine::solve`](crate::buffer_of_thoughts::engine::BotEngine::solve),
/// carrying everything [`BotEngine::record_outcome`](crate::buffer_of_thoughts::engine::BotEngine::record_outcome)
/// needs to perform write-back.
#[derive(Debug, Clone, PartialEq)]
pub struct BotSolveResult {
    /// The original problem text.
    pub problem: String,
    /// The computed structural signature of `problem`.
    pub signature: ProblemSignature,
    /// `Some(id)` of the buffered template that was matched and used, or
    /// `None` if retrieval found no sufficiently similar template (the
    /// generic default scaffold was used instead).
    pub matched_template_id: Option<u64>,
    /// The similarity score of the match, when `matched_template_id` is
    /// `Some`.
    pub similarity: Option<f32>,
    /// The fully-instantiated reasoning scaffold that conditioned the
    /// generator.
    pub instantiated_scaffold: String,
    /// The generator's answer.
    pub answer: String,
}

// ── BotGenerator ───────────────────────────────────────────────────────────────

/// The generator interface driving `solve` and `distill`.
///
/// Implementations are **pure-sync** — no I/O, no async, consistent with
/// `self_discover::SelfDiscoverModel` and
/// `analogical_prompting::AnalogicalModel`. The caller supplies a concrete
/// generator (e.g. wrapping an LLM); [`MockBotGenerator`] is provided for
/// tests.
pub trait BotGenerator {
    /// Produce an answer for `problem`, conditioned on the fully-instantiated
    /// `instantiated_scaffold`.
    fn generate(&self, problem: &str, instantiated_scaffold: &str) -> String;

    /// Distill a new, generalized template from a problem that was just
    /// solved successfully with no buffered match.
    ///
    /// `signature` is the already-computed structural signature of
    /// `problem`; implementations should use it (rather than re-deriving one)
    /// to decide, e.g., which operation tag to bake into the generalized
    /// strategy description. The returned text should contain a literal
    /// `{problem}` placeholder and, where appropriate, `{op1}`, `{op2}`, …
    /// placeholders so [`crate::buffer_of_thoughts::types::instantiate_template`]
    /// can re-specialize it for future problems of the same kind.
    fn distill(&self, problem: &str, answer: &str, signature: &ProblemSignature) -> String;
}

// ── MockBotGenerator ───────────────────────────────────────────────────────────

/// Deterministic [`BotGenerator`] for tests and examples.
///
/// [`BotGenerator::generate`] deterministically combines the instantiated
/// scaffold and the problem text. [`BotGenerator::distill`] builds a
/// generalized scaffold whose operand placeholders (`{op1}`, `{op2}`, …)
/// mirror the count of numbers found in the originating problem, and whose
/// strategy description names the signature's primary (first, alphabetically)
/// operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MockBotGenerator;

impl BotGenerator for MockBotGenerator {
    fn generate(&self, problem: &str, instantiated_scaffold: &str) -> String {
        format!("{instantiated_scaffold}\n=> Answer for: {problem}")
    }

    fn distill(&self, problem: &str, _answer: &str, signature: &ProblemSignature) -> String {
        let primary_tag = signature
            .operation_tags
            .first()
            .cloned()
            .unwrap_or_else(|| "general".to_string());

        let operand_count = extract_numbers(problem).len();
        let operands_desc = if operand_count == 0 {
            "the relevant values in the problem".to_string()
        } else {
            (1..=operand_count)
                .map(|n| format!("{{op{n}}}"))
                .collect::<Vec<_>>()
                .join(" and ")
        };

        format!(
            "Step 1: Identify {operands_desc}, relevant to a {primary_tag} task.\nStep 2: Apply {primary_tag} reasoning to relate them.\nStep 3: Verify the intermediate result makes sense.\nStep 4: State the final answer for: {{problem}}"
        )
    }
}

// ── BotError ───────────────────────────────────────────────────────────────────

/// Errors from the `buffer_of_thoughts` module.
#[derive(Debug, Error)]
pub enum BotError {
    /// The problem string was empty after trimming.
    #[error("problem must not be empty")]
    EmptyProblem,
    /// A generator's distilled template text was empty after trimming.
    #[error("distilled template text must not be empty")]
    EmptyTemplateText,
    /// [`BotEngine::record_outcome`](crate::buffer_of_thoughts::engine::BotEngine::record_outcome)
    /// referenced a template id that is no longer in the buffer (e.g. it was
    /// evicted between `solve` and `record_outcome`).
    #[error("no thought template found in the buffer with id {0}")]
    TemplateNotFound(u64),
}
