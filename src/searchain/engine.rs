//! [`SearChainEngine`] — Search-in-the-Chain: chain-of-query construction
//! (Phase 1), Interactive Reasoning-Verification with dependency-aware
//! backtracking (Phase 2), and final-answer assembly (Phase 3).

use std::collections::{HashSet, VecDeque};

use super::types::{
    ChainGenerator, ChainNode, Generator, NodeVerdict, Retriever, SearChainConfig, SearChainError,
    SearChainResult, SearChainStatus,
};

// ── lexical helpers (answer-vs-evidence comparison) ─────────────────────────
//
// Deliberately self-contained (not imported from `knowledge_conflict` or
// `dragin`): each module in this crate owns its own small lexical toolkit
// rather than sharing private helpers across module boundaries.

/// Tokenize `text`: split on non-alphanumeric boundaries, lowercase, keep
/// non-empty fragments.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Stopwords excluded from the *content* vocabulary used for overlap and
/// subject-matching decisions.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old",
    "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "that", "this", "with", "from", "they", "have", "were", "what", "your", "when", "them", "then",
    "than", "into", "some", "such", "only", "also", "been", "more", "very", "will", "would",
    "there", "their", "which", "about", "could", "these", "those", "does",
];

/// Return `true` when `token` is a content token: at least three characters
/// and not a stopword.
fn is_content_token(token: &str) -> bool {
    token.chars().count() >= 3 && !STOPWORDS.contains(&token)
}

/// The distinct content-token vocabulary of `text`.
fn content_terms(text: &str) -> HashSet<String> {
    tokenize(text)
        .into_iter()
        .filter(|t| is_content_token(t) && !t.chars().all(|c| c.is_ascii_digit()))
        .collect()
}

/// Distinct content tokens shared between two texts.
fn shared_terms(a: &str, b: &str) -> HashSet<String> {
    let terms_a = content_terms(a);
    let terms_b = content_terms(b);
    terms_a.intersection(&terms_b).cloned().collect()
}

/// Negation markers: whole-token markers plus the contracted `n't` suffix.
const NEGATION_TOKENS: &[&str] = &[
    "not", "no", "never", "without", "fails", "fail", "cannot", "false", "neither", "nor",
];

/// Return `true` when `text` exhibits a negation signal.
fn has_negation(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains("n't") {
        return true;
    }
    tokenize(&lower)
        .iter()
        .any(|t| NEGATION_TOKENS.contains(&t.as_str()))
}

/// Extract numeric tokens from `text`, excluding 4-digit year-like values
/// (which are handled separately by [`extract_years`]).
fn extract_numbers(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
                return None;
            }
            let digit_count = cleaned.chars().filter(char::is_ascii_digit).count();
            if digit_count == 4
                && !cleaned.contains('.')
                && cleaned
                    .parse::<u32>()
                    .is_ok_and(|value| (1000..=2100).contains(&value))
            {
                return None;
            }
            cleaned.parse::<f64>().ok()
        })
        .collect()
}

/// Extract 4-digit year tokens in the range `1000..=2100`.
fn extract_years(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|token| {
            let cleaned: String = token.chars().filter(char::is_ascii_digit).collect();
            if cleaned.len() == 4 {
                cleaned
                    .parse::<u32>()
                    .ok()
                    .filter(|&y| (1000..=2100).contains(&y))
            } else {
                None
            }
        })
        .collect()
}

/// Compare `answer` against retrieved `evidence`, producing a [`NodeVerdict`].
///
/// - No evidence at all → [`NodeVerdict::Unverified`] (nothing to confirm or
///   refute).
/// - Evidence sharing at least `config.min_shared_terms` content tokens with
///   `answer` (i.e. plausibly about the same subject) that disagrees via
///   negation polarity, a mismatched year, or a mismatched number →
///   [`NodeVerdict::Conflicting`].
/// - Otherwise, when at least `config.min_support_overlap` of `answer`'s
///   content terms are echoed by the evidence → [`NodeVerdict::Verified`].
/// - Otherwise → [`NodeVerdict::Unverified`].
fn compare_answer_to_evidence(
    answer: &str,
    evidence: &[String],
    config: &SearChainConfig,
) -> NodeVerdict {
    if evidence.is_empty() {
        return NodeVerdict::Unverified;
    }
    let combined = evidence.join(" ");

    let shared = shared_terms(answer, &combined);
    if shared.len() >= config.min_shared_terms {
        if has_negation(answer) ^ has_negation(&combined) {
            return NodeVerdict::Conflicting;
        }

        let years_answer = extract_years(answer);
        let years_evidence = extract_years(&combined);
        if !years_answer.is_empty()
            && !years_evidence.is_empty()
            && !years_answer.iter().any(|y| years_evidence.contains(y))
        {
            return NodeVerdict::Conflicting;
        }

        let nums_answer = extract_numbers(answer);
        let nums_evidence = extract_numbers(&combined);
        if !nums_answer.is_empty()
            && !nums_evidence.is_empty()
            && !nums_answer
                .iter()
                .any(|x| nums_evidence.iter().any(|y| (x - y).abs() < f64::EPSILON))
        {
            return NodeVerdict::Conflicting;
        }
    }

    let answer_terms = content_terms(answer);
    if answer_terms.is_empty() {
        return NodeVerdict::Unverified;
    }
    let evidence_terms = content_terms(&combined);
    let hit = answer_terms.intersection(&evidence_terms).count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = hit as f32 / answer_terms.len() as f32;
    if ratio >= config.min_support_overlap {
        NodeVerdict::Verified
    } else {
        NodeVerdict::Unverified
    }
}

/// Pick the single best answer candidate from ranked `evidence` — the
/// first, best-ranked passage, trimmed. Callers should only invoke this with
/// non-empty evidence.
fn best_answer_from_evidence(evidence: &[String]) -> String {
    evidence
        .first()
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Apply a freshly computed `verdict`/`evidence` pair to `node`, updating
/// `final_answer` per the IRV revision rule and setting `revised` whenever
/// the resulting text differs from what `node.final_answer` held before this
/// call.
///
/// `candidate_answer` is the answer text that was actually compared against
/// `evidence` to produce `verdict` — the node's current answer on a
/// first-pass check, or a backtracking-patched candidate on a cascade
/// re-verification.
fn apply_verdict(
    node: &mut ChainNode,
    verdict: NodeVerdict,
    evidence: Vec<String>,
    candidate_answer: String,
) {
    let previous = node.final_answer.clone();
    let resolved = match verdict {
        NodeVerdict::Verified => candidate_answer,
        NodeVerdict::Unverified | NodeVerdict::Conflicting => {
            if evidence.is_empty() {
                candidate_answer
            } else {
                best_answer_from_evidence(&evidence)
            }
        }
    };

    node.evidence = evidence;
    node.verdict = Some(verdict);
    node.final_answer = resolved;
    if node.final_answer != previous {
        node.revised = true;
    }
}

// ── dependency graph helpers ─────────────────────────────────────────────────

/// Build a forward adjacency list: `dependents[k]` lists every node id that
/// declared `k` in its own `depends_on`, deduplicated and sorted ascending.
fn build_dependents(chain: &[ChainNode]) -> Vec<Vec<usize>> {
    let mut dependents: Vec<HashSet<usize>> = vec![HashSet::new(); chain.len()];
    for node in chain {
        for &dep in &node.depends_on {
            if let Some(set) = dependents.get_mut(dep) {
                set.insert(node.id);
            }
        }
    }
    dependents
        .into_iter()
        .map(|set| {
            let mut ids: Vec<usize> = set.into_iter().collect();
            ids.sort_unstable();
            ids
        })
        .collect()
}

/// Breadth-first transitive closure of `dependents`, starting from `seeds`
/// (each seed is included in the result). Returned in ascending order.
fn transitive_dependents(dependents: &[Vec<usize>], seeds: &[usize]) -> Vec<usize> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &seed in seeds {
        if seen.insert(seed) {
            queue.push_back(seed);
        }
    }
    while let Some(id) = queue.pop_front() {
        if let Some(children) = dependents.get(id) {
            for &child in children {
                if seen.insert(child) {
                    queue.push_back(child);
                }
            }
        }
    }
    let mut result: Vec<usize> = seen.into_iter().collect();
    result.sort_unstable();
    result
}

// ── SearChainEngine ──────────────────────────────────────────────────────────

/// Drives the three phases of Search-in-the-Chain (`SearChain`, Xu et al.
/// 2024): (1) upfront chain-of-query construction with no retrieval, (2)
/// Interactive Reasoning-Verification (IRV) with dependency-aware
/// backtracking, and (3) final-answer assembly.
///
/// The generator, retriever, and answer-assembler are supplied *per call*
/// through generic trait bounds, mirroring the caller-supplies-executor
/// pattern used across this crate (see e.g. `dragin::DraginEngine`).
#[derive(Debug, Clone, Default)]
pub struct SearChainEngine {
    /// Configuration for this engine.
    pub config: SearChainConfig,
}

impl SearChainEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SearChainConfig) -> Self {
        Self { config }
    }

    /// Phase 1: build and validate the complete chain-of-query for `query`.
    ///
    /// The `chain_generator` is consulted exactly once, upfront; no
    /// retrieval occurs during this phase. The returned chain is validated:
    /// node ids must be sequential starting at `0`, and every dependency
    /// must reference a strictly earlier node.
    ///
    /// # Errors
    ///
    /// - [`SearChainError::EmptyQuery`] if `query` is blank.
    /// - [`SearChainError::EmptyChain`] if the generator returns no nodes.
    /// - [`SearChainError::NonSequentialIds`] if node ids are not
    ///   `0, 1, 2, ...` in order.
    /// - [`SearChainError::InvalidDependency`] if a node depends on itself
    ///   or a later node.
    pub fn build_chain<C>(
        &self,
        query: &str,
        chain_generator: &C,
    ) -> Result<Vec<ChainNode>, SearChainError>
    where
        C: ChainGenerator + ?Sized,
    {
        if query.trim().is_empty() {
            return Err(SearChainError::EmptyQuery);
        }

        let chain = chain_generator.generate_chain(query);
        if chain.is_empty() {
            return Err(SearChainError::EmptyChain);
        }

        for (position, node) in chain.iter().enumerate() {
            if node.id != position {
                return Err(SearChainError::NonSequentialIds {
                    expected: position,
                    found: node.id,
                });
            }
            for &dep in &node.depends_on {
                if dep >= node.id {
                    return Err(SearChainError::InvalidDependency {
                        node_id: node.id,
                        dep_id: dep,
                    });
                }
            }
        }

        Ok(chain)
    }

    /// Phase 2: Interactive Reasoning-Verification (IRV) with
    /// dependency-aware backtracking.
    ///
    /// Runs in two sub-passes:
    ///
    /// 1. **First sweep.** Every node, in id order, has its *current* answer
    ///    (initially the Phase-1 tentative answer) checked against freshly
    ///    retrieved evidence, producing a verdict. `Unverified` and
    ///    `Conflicting` verdicts replace the answer with the evidence's best
    ///    candidate.
    /// 2. **Backtracking.** Every node that came out `Conflicting` is
    ///    propagated to its *direct dependents*: for each dependent, any
    ///    literal occurrence of the ancestor's original tentative answer
    ///    inside the dependent's current answer is patched to the ancestor's
    ///    corrected answer, and the dependent is genuinely **re-verified**
    ///    against fresh evidence — regardless of what verdict it received on
    ///    the first sweep, since that verdict may have rested on the
    ///    now-invalidated premise. If the re-verified dependent is itself
    ///    `Conflicting`, its own dependents are queued in turn, so a
    ///    correction cascades transitively down the chain.
    ///
    /// Each dependent re-verification counts against
    /// [`SearChainConfig::max_backtrack_iterations`]. If the budget is
    /// exhausted, no more re-verifications are performed; every node that
    /// was consequently never re-checked (plus its own transitive
    /// dependents) is reported, honestly, via
    /// [`SearChainStatus::PartialBacktrackExhausted`] rather than being
    /// silently left in a possibly-wrong state.
    ///
    /// Returns the termination status together with the total number of
    /// backtracking-triggered re-verifications performed.
    pub fn verify_chain<R>(
        &self,
        chain: &mut [ChainNode],
        retriever: &R,
    ) -> (SearChainStatus, usize)
    where
        R: Retriever + ?Sized,
    {
        let node_count = chain.len();
        if node_count == 0 {
            return (SearChainStatus::Complete, 0);
        }

        let dependents = build_dependents(chain);

        // ── First sweep ──────────────────────────────────────────────────
        let mut conflicting_first_pass: Vec<usize> = Vec::new();
        for (id, node) in chain.iter_mut().enumerate() {
            let candidate = node.final_answer.clone();
            let evidence = retriever.retrieve(&node.sub_query);
            let verdict = compare_answer_to_evidence(&candidate, &evidence, &self.config);
            if verdict == NodeVerdict::Conflicting {
                conflicting_first_pass.push(id);
            }
            apply_verdict(node, verdict, evidence, candidate);
        }

        // ── Backtracking: propagate invalidation to dependents ─────────────
        let mut propagate_from: VecDeque<usize> = conflicting_first_pass.into_iter().collect();
        let mut backtrack_events: usize = 0;
        let mut unresolved_seeds: Vec<usize> = Vec::new();

        while let Some(ancestor_id) = propagate_from.pop_front() {
            // The stable, never-mutated reference text that a dependent may
            // have textually embedded, and what it should now read instead.
            let old_answer = chain[ancestor_id].tentative_answer.clone();
            let new_answer = chain[ancestor_id].final_answer.clone();

            for &dep_id in &dependents[ancestor_id] {
                if backtrack_events >= self.config.max_backtrack_iterations {
                    unresolved_seeds.push(dep_id);
                    continue;
                }
                backtrack_events += 1;

                let before = chain[dep_id].final_answer.clone();
                let candidate = if !old_answer.is_empty() && before.contains(old_answer.as_str()) {
                    before.replace(old_answer.as_str(), new_answer.as_str())
                } else {
                    before
                };

                let evidence = retriever.retrieve(&chain[dep_id].sub_query);
                let verdict = compare_answer_to_evidence(&candidate, &evidence, &self.config);
                if verdict == NodeVerdict::Conflicting {
                    propagate_from.push_back(dep_id);
                }
                apply_verdict(&mut chain[dep_id], verdict, evidence, candidate);
            }
        }

        let status = if unresolved_seeds.is_empty() {
            SearChainStatus::Complete
        } else {
            SearChainStatus::PartialBacktrackExhausted {
                unresolved_node_ids: transitive_dependents(&dependents, &unresolved_seeds),
            }
        };

        (status, backtrack_events)
    }

    /// Phase 3: assemble the final answer from the fully processed `chain`.
    #[allow(clippy::unused_self)]
    pub fn assemble_answer<G>(
        &self,
        original_query: &str,
        chain: &[ChainNode],
        generator: &G,
    ) -> String
    where
        G: Generator + ?Sized,
    {
        generator.generate(original_query, chain)
    }

    /// Run the complete `SearChain` pipeline: Phase 1 chain construction,
    /// Phase 2 Interactive Reasoning-Verification with backtracking, and
    /// Phase 3 final-answer assembly.
    ///
    /// # Errors
    ///
    /// See [`SearChainEngine::build_chain`].
    pub fn run<C, R, G>(
        &self,
        query: &str,
        chain_generator: &C,
        retriever: &R,
        generator: &G,
    ) -> Result<SearChainResult, SearChainError>
    where
        C: ChainGenerator + ?Sized,
        R: Retriever + ?Sized,
        G: Generator + ?Sized,
    {
        let mut chain = self.build_chain(query, chain_generator)?;
        let (status, backtrack_events) = self.verify_chain(&mut chain, retriever);
        let final_answer = self.assemble_answer(query, &chain, generator);

        Ok(SearChainResult {
            original_query: query.to_string(),
            chain,
            final_answer,
            status,
            backtrack_events,
        })
    }
}
