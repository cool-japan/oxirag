//! [`ToCEngine`] — builds, prunes, and aggregates a [`ClarificationTree`].

use crate::tree_of_clarifications::tree::{ClarificationNode, ClarificationTree};
use crate::tree_of_clarifications::types::{
    ClarificationAnswerer, ClarificationRetriever, Disambiguator, ToCConfig, ToCError,
};

// ── ToCEngine ──────────────────────────────────────────────────────────────────

/// Tree-of-Clarifications engine: disambiguates an ambiguous question into a
/// tree of readings, retrieves and answers each, recursively prunes weakly
/// supported branches, and aggregates the survivors into one long-form answer.
///
/// The disambiguator, retriever, and answerer are fixed at construction time
/// (unlike the caller-supplies-executor-per-call pattern used elsewhere in this
/// crate), since [`ToCEngine::run`] must invoke all three repeatedly while
/// walking the tree it builds.
#[derive(Debug, Clone)]
pub struct ToCEngine<D, R, A>
where
    D: Disambiguator,
    R: ClarificationRetriever,
    A: ClarificationAnswerer,
{
    /// Configuration for this engine.
    pub config: ToCConfig,
    /// Proposes disambiguated readings of a question.
    pub disambiguator: D,
    /// Retrieves supporting passages for a single reading.
    pub retriever: R,
    /// Drafts an answer for a single reading from its retrieved passages.
    pub answerer: A,
}

impl<D, R, A> ToCEngine<D, R, A>
where
    D: Disambiguator,
    R: ClarificationRetriever,
    A: ClarificationAnswerer,
{
    /// Create a new engine from its configuration and three components.
    #[must_use]
    pub fn new(config: ToCConfig, disambiguator: D, retriever: R, answerer: A) -> Self {
        Self {
            config,
            disambiguator,
            retriever,
            answerer,
        }
    }

    /// Run the full Tree-of-Clarifications pipeline for `question`.
    ///
    /// 1. Disambiguate `question` into candidate readings (recursively, up to
    ///    `config.max_depth`), building one [`ClarificationNode`] per reading
    ///    and retrieving + answering each along the way.
    /// 2. Prune branches whose `relevance_score` falls below
    ///    `config.prune_threshold`.
    /// 3. Aggregate every surviving leaf's answer into one long-form answer.
    ///
    /// Returns the pruned tree together with the aggregated answer.
    ///
    /// # Errors
    ///
    /// Returns [`ToCError::EmptyQuestion`] when `question` is empty (or
    /// whitespace-only) after trimming. Returns [`ToCError::InvalidConfig`]
    /// when `self.config` fails [`ToCConfig::validate`].
    pub fn run(&self, question: &str) -> Result<(ClarificationTree, String), ToCError> {
        let trimmed = question.trim();
        if trimmed.is_empty() {
            return Err(ToCError::EmptyQuestion);
        }
        self.config.validate()?;

        let mut tree = self.build_tree(trimmed);
        tree.prune(self.config.prune_threshold);
        let answer = aggregate(&tree);

        Ok((tree, answer))
    }

    /// Build the (unpruned) [`ClarificationTree`] for `question`.
    fn build_tree(&self, question: &str) -> ClarificationTree {
        let top_candidates = self.disambiguator.disambiguate(question);

        let root_questions = if top_candidates.is_empty() {
            // Not ambiguous: a single root-equals-leaf node for the original
            // question. Building it at `max_depth` guarantees it is a genuine
            // leaf (no further disambiguation is attempted).
            vec![self.build_node(question.to_string(), self.config.max_depth)]
        } else {
            top_candidates
                .into_iter()
                .map(|candidate| self.build_node(candidate, 1))
                .collect()
        };

        ClarificationTree { root_questions }
    }

    /// Build a single node for `question` at tree `depth`, recursively
    /// disambiguating it further while `depth < config.max_depth`.
    fn build_node(&self, question: String, depth: usize) -> ClarificationNode {
        let passages = self.retriever.retrieve(&question, self.config.top_k);
        let relevance_score = lexical_relevance(&question, &passages);
        let answer = self.answerer.answer(&question, &passages);

        let children = if depth < self.config.max_depth {
            self.disambiguator
                .disambiguate(&question)
                .into_iter()
                .map(|child_question| self.build_node(child_question, depth + 1))
                .collect()
        } else {
            Vec::new()
        };

        ClarificationNode {
            question,
            relevance_score,
            answer: Some(answer),
            children,
        }
    }
}

// ── aggregate ──────────────────────────────────────────────────────────────────

/// Synthesize a long-form answer from every surviving leaf of `tree`.
///
/// Each leaf's answer is attributed to its disambiguated question, one per
/// line, e.g.:
///
/// ```text
/// Regarding What is the capital of France?: Paris.
/// Regarding What is the capital of a company?: Company capital ...
/// ```
///
/// Returns a fixed fallback message when `tree` has no leaves at all (every
/// branch was pruned away).
#[must_use]
pub fn aggregate(tree: &ClarificationTree) -> String {
    let leaves = tree.leaves();
    if leaves.is_empty() {
        return "No disambiguated reading survived pruning; unable to produce an answer."
            .to_string();
    }

    leaves
        .iter()
        .map(|leaf| {
            let answer = leaf
                .answer
                .as_deref()
                .unwrap_or("No answer was produced for this reading.");
            format!("Regarding {}: {answer}", leaf.question)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ── lexical_relevance ────────────────────────────────────────────────────────

/// Tokenise `text` into lower-cased, non-empty, alphanumeric-run tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Retrieval-support-based confidence: the fraction of `question`'s distinct
/// tokens that also appear among `passages`' tokens, in `[0.0, 1.0]`.
///
/// Returns `0.0` when there are no passages or the question has no tokens at
/// all (nothing to overlap with means no measurable support).
fn lexical_relevance(question: &str, passages: &[String]) -> f32 {
    if passages.is_empty() {
        return 0.0;
    }

    let question_tokens: std::collections::BTreeSet<String> =
        tokenize(question).into_iter().collect();
    if question_tokens.is_empty() {
        return 0.0;
    }

    let passage_text = passages.join(" ");
    let passage_tokens: std::collections::BTreeSet<String> =
        tokenize(&passage_text).into_iter().collect();

    let overlap = question_tokens
        .iter()
        .filter(|token| passage_tokens.contains(*token))
        .count();

    #[allow(clippy::cast_precision_loss)]
    let score = overlap as f32 / question_tokens.len() as f32;
    score.clamp(0.0, 1.0)
}
