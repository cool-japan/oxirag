//! Types for the `self_consistency` module.

use thiserror::Error;

// ── ReasoningPath ───────────────────────────────────────────────────────────────

/// A single sampled chain of thought together with its extracted final answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningPath {
    /// The verbal reasoning / chain of thought produced for the question.
    pub reasoning: String,
    /// The final answer extracted from the reasoning.
    pub answer: String,
}

impl ReasoningPath {
    /// Create a new [`ReasoningPath`] from a reasoning trace and a final answer.
    #[must_use]
    pub fn new(reasoning: impl Into<String>, answer: impl Into<String>) -> Self {
        Self {
            reasoning: reasoning.into(),
            answer: answer.into(),
        }
    }

    /// Number of alphanumeric tokens in the reasoning trace.
    #[must_use]
    pub fn reasoning_token_count(&self) -> usize {
        tokenize(&self.reasoning).count()
    }
}

// ── ReasoningSampler ────────────────────────────────────────────────────────────

/// A source of diverse reasoning paths for a question.
///
/// Implementors replace greedy decoding: each call should produce one
/// *independently sampled* reasoning path. The `index` argument enables
/// deterministic diversity across the `K` samples drawn for a single question.
pub trait ReasoningSampler {
    /// Produce one reasoning path for `question`.
    ///
    /// `index` enables deterministic diversity across samples (it is the
    /// zero-based ordinal of this sample within the batch).
    fn sample(&self, question: &str, index: usize) -> ReasoningPath;
}

/// Deterministic [`ReasoningSampler`] backed by a fixed list of paths.
///
/// Sampling returns `paths[index % paths.len()]`, cycling through the supplied
/// paths. Useful for testing and for replaying recorded chains of thought.
#[derive(Debug, Clone)]
pub struct MockReasoningSampler {
    /// The fixed reasoning paths cycled through by [`Self::sample`].
    pub paths: Vec<ReasoningPath>,
}

impl MockReasoningSampler {
    /// Create a new mock sampler from a list of reasoning paths.
    #[must_use]
    pub fn new(paths: Vec<ReasoningPath>) -> Self {
        Self { paths }
    }
}

impl ReasoningSampler for MockReasoningSampler {
    fn sample(&self, _question: &str, index: usize) -> ReasoningPath {
        if self.paths.is_empty() {
            return ReasoningPath::new(String::new(), String::new());
        }
        self.paths[index % self.paths.len()].clone()
    }
}

// ── VoteWeighting ───────────────────────────────────────────────────────────────

/// How much each reasoning path contributes to its answer cluster's vote total.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VoteWeighting {
    /// Every path contributes a weight of `1.0`.
    #[default]
    Uniform,
    /// Weight `1 + ln(1 + reasoning_token_count)`, rewarding more-developed
    /// chains of thought.
    ByReasoningLength,
}

impl VoteWeighting {
    /// Short human-readable label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Uniform => "uniform",
            Self::ByReasoningLength => "by_reasoning_length",
        }
    }

    /// Compute the vote weight contributed by `path` under this weighting.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn weight_of(&self, path: &ReasoningPath) -> f32 {
        match self {
            Self::Uniform => 1.0,
            Self::ByReasoningLength => {
                let count = path.reasoning_token_count() as f32;
                1.0 + (1.0 + count).ln()
            }
        }
    }
}

// ── AnswerCluster ───────────────────────────────────────────────────────────────

/// A group of reasoning paths whose final answers are semantically equivalent.
#[derive(Debug, Clone)]
pub struct AnswerCluster {
    /// Canonical (normalized) representative of the cluster's answer.
    pub canonical: String,
    /// Indices (into the path list) of the paths that voted for this cluster.
    pub members: Vec<usize>,
    /// Sum of the member paths' vote weights.
    pub votes: f32,
}

// ── SelfConsistencyConfig ───────────────────────────────────────────────────────

/// Configuration for [`SelfConsistencyEngine`](crate::self_consistency::SelfConsistencyEngine).
#[derive(Debug, Clone)]
pub struct SelfConsistencyConfig {
    /// Number of reasoning paths to sample. Defaults to `5`.
    pub num_paths: usize,
    /// Token-set Jaccard threshold above which two answers are equivalent.
    ///
    /// Defaults to `0.8`.
    pub equivalence_threshold: f32,
    /// How votes are weighted across paths. Defaults to [`VoteWeighting::Uniform`].
    pub weighting: VoteWeighting,
}

impl Default for SelfConsistencyConfig {
    fn default() -> Self {
        Self {
            num_paths: 5,
            equivalence_threshold: 0.8,
            weighting: VoteWeighting::Uniform,
        }
    }
}

impl SelfConsistencyConfig {
    /// Create a config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of reasoning paths to sample.
    #[must_use]
    pub fn with_num_paths(mut self, num_paths: usize) -> Self {
        self.num_paths = num_paths;
        self
    }

    /// Set the answer-equivalence Jaccard threshold.
    #[must_use]
    pub fn with_equivalence_threshold(mut self, equivalence_threshold: f32) -> Self {
        self.equivalence_threshold = equivalence_threshold;
        self
    }

    /// Set the vote weighting scheme.
    #[must_use]
    pub fn with_weighting(mut self, weighting: VoteWeighting) -> Self {
        self.weighting = weighting;
        self
    }
}

// ── SelfConsistencyOutput ───────────────────────────────────────────────────────

/// Result of marginalizing over sampled reasoning paths.
#[derive(Debug, Clone)]
pub struct SelfConsistencyOutput {
    /// The winning answer (canonical form of the highest-vote cluster).
    pub answer: String,
    /// Confidence — the winning cluster's share of total vote mass, in [0, 1].
    pub confidence: f32,
    /// All answer clusters, sorted by descending votes (then by tie-break order).
    pub clusters: Vec<AnswerCluster>,
    /// The reasoning paths that were marginalized over.
    pub paths: Vec<ReasoningPath>,
}

// ── SelfConsistencyError ────────────────────────────────────────────────────────

/// Errors from the `self_consistency` module.
#[derive(Debug, Error)]
pub enum SelfConsistencyError {
    /// The question was empty after trimming.
    #[error("question must not be empty")]
    EmptyQuestion,
    /// No reasoning paths were available to marginalize over.
    #[error("no reasoning paths")]
    NoPaths,
}

// ── Tokenization & normalization helpers ────────────────────────────────────────

/// Tokenize `text` into lowercase alphanumeric tokens.
pub(crate) fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
}

/// Normalize a raw answer string for equivalence comparison.
///
/// Lowercases, trims, strips surrounding (leading/trailing) punctuation,
/// collapses internal whitespace runs to a single space, and canonicalizes a
/// pure-integer answer to its bare digits (e.g. `"42."` ⇒ `"42"`,
/// `"+007"` ⇒ `"7"`).
#[must_use]
pub(crate) fn normalize_answer(answer: &str) -> String {
    let lowered = answer.to_lowercase();

    // Integer fast path (sign-aware): trim trailing non-alphanumerics and any
    // leading characters that are neither a sign nor alphanumeric, so a leading
    // `+`/`-` that belongs to the number survives (e.g. "-5" stays "-5"). This
    // must run before the general punctuation strip, which would drop the sign.
    {
        let head_trimmed =
            lowered.trim_start_matches(|c: char| !(c.is_alphanumeric() || c == '+' || c == '-'));
        let int_candidate = head_trimmed.trim_end_matches(|c: char| !c.is_alphanumeric());
        if let Some(canon) = canonicalize_integer(int_candidate) {
            return canon;
        }
    }

    // Strip surrounding punctuation/whitespace, keeping inner content intact.
    let trimmed = lowered.trim_matches(|c: char| !c.is_alphanumeric());

    // Collapse internal whitespace.
    let collapsed: String = {
        let mut out = String::with_capacity(trimmed.len());
        let mut prev_ws = false;
        for c in trimmed.chars() {
            if c.is_whitespace() {
                if !prev_ws {
                    out.push(' ');
                }
                prev_ws = true;
            } else {
                out.push(c);
                prev_ws = false;
            }
        }
        out.trim().to_string()
    };

    collapsed
}

/// If `s` denotes an integer (optional `+`/`-` then digits), return its
/// canonical decimal form; otherwise `None`.
fn canonicalize_integer(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (sign_neg, digits) = match bytes[0] {
        b'+' => (false, &s[1..]),
        b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: i128 = digits.parse().ok()?;
    let signed = if sign_neg { -value } else { value };
    Some(signed.to_string())
}

/// Token-set Jaccard similarity between two answer strings.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub(crate) fn answer_jaccard(a: &str, b: &str) -> f32 {
    let set_a: std::collections::HashSet<String> = tokenize(a).collect();
    let set_b: std::collections::HashSet<String> = tokenize(b).collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = set_a.intersection(&set_b).count();
    intersection as f32 / union as f32
}

/// Two answers are equivalent if their normalized forms are identical, or if
/// their token-set Jaccard similarity is at least `threshold`.
#[must_use]
pub(crate) fn answers_equivalent(a: &str, b: &str, threshold: f32) -> bool {
    if normalize_answer(a) == normalize_answer(b) {
        return true;
    }
    answer_jaccard(a, b) >= threshold
}
