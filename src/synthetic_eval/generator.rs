//! Synthetic evaluation-set generation over a document corpus.
//!
//! [`SyntheticEvalGenerator`] ties a [`QuestionTemplater`] together with
//! salience-driven sentence selection and hard-negative distractor mining to
//! produce [`SyntheticQa`] tuples for offline RAG evaluation.

use crate::synthetic_eval::templater::{
    HeuristicTemplater, QuestionTemplater, is_stopword, salience, sentences, tokenize,
};
use crate::synthetic_eval::types::{SyntheticEvalConfig, SyntheticEvalError, SyntheticQa};
use crate::types::{Document, DocumentId};
use std::collections::HashSet;

// ── SyntheticEvalGenerator ────────────────────────────────────────────────────

/// Generates synthetic evaluation tuples from a corpus using a [`QuestionTemplater`].
#[derive(Debug, Clone)]
pub struct SyntheticEvalGenerator<T: QuestionTemplater> {
    /// Generation configuration.
    pub config: SyntheticEvalConfig,
    /// The templater used to turn sentences into questions.
    pub templater: T,
}

impl SyntheticEvalGenerator<HeuristicTemplater> {
    /// Create a generator using the default [`HeuristicTemplater`].
    #[must_use]
    pub fn new(config: SyntheticEvalConfig) -> Self {
        Self {
            config,
            templater: HeuristicTemplater::new(),
        }
    }
}

impl<T: QuestionTemplater> SyntheticEvalGenerator<T> {
    /// Create a generator with a custom [`QuestionTemplater`].
    #[must_use]
    pub fn with_templater(config: SyntheticEvalConfig, templater: T) -> Self {
        Self { config, templater }
    }

    /// The set of salient (non-stopword) tokens present in `text`.
    fn salient_terms(text: &str) -> HashSet<String> {
        tokenize(text)
            .into_iter()
            .filter(|t| !is_stopword(t))
            .collect()
    }

    /// Rank the sentences of `doc` by descending total salience (deterministic).
    ///
    /// Sentences shorter than `min_sentence_tokens` are dropped. Ties are broken
    /// by original sentence order, which is stable because the sort is stable and
    /// the input is already in document order.
    fn ranked_sentences(&self, doc: &Document) -> Vec<String> {
        let mut scored: Vec<(f32, String)> = sentences(&doc.content)
            .into_iter()
            .filter_map(|s| {
                let tokens = tokenize(&s);
                if tokens.len() < self.config.min_sentence_tokens {
                    return None;
                }
                let sal = salience(&tokens);
                let total: f32 = sal.values().sum();
                Some((total, s))
            })
            .collect();
        // Stable sort by descending salience; equal scores keep document order.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, s)| s).collect()
    }

    /// Mine up to `num_distractors` hard negatives for one generated question.
    ///
    /// A distractor is a document that (1) is not the source, (2) shares at least
    /// one salient term with the question, and (3) does not contain the answer.
    /// Candidates are ranked by descending shared-term overlap; ties are broken by
    /// corpus order for determinism.
    fn mine_distractors(
        &self,
        question: &str,
        answer: &str,
        source_id: &DocumentId,
        corpus: &[Document],
    ) -> Vec<DocumentId> {
        if self.config.num_distractors == 0 {
            return Vec::new();
        }
        let question_terms = Self::salient_terms(question);
        if question_terms.is_empty() {
            return Vec::new();
        }
        let answer_terms = Self::salient_terms(answer);

        let mut scored: Vec<(usize, usize, DocumentId)> = Vec::new();
        for (corpus_index, candidate) in corpus.iter().enumerate() {
            if &candidate.id == source_id {
                continue;
            }
            let candidate_terms = Self::salient_terms(&candidate.content);
            let overlap = question_terms.intersection(&candidate_terms).count();
            if overlap == 0 {
                continue;
            }
            // Reject candidates that actually contain the answer (not hard negatives).
            if Self::contains_answer(&candidate_terms, &answer_terms) {
                continue;
            }
            scored.push((overlap, corpus_index, candidate.id.clone()));
        }
        // Highest overlap first; ties broken by earliest corpus index.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored
            .into_iter()
            .take(self.config.num_distractors)
            .map(|(_, _, id)| id)
            .collect()
    }

    /// Return `true` if `candidate_terms` contain every salient answer term.
    ///
    /// An empty answer-term set (e.g. the answer is all stopwords) is treated as
    /// "not contained" so such pairs can still mine distractors.
    fn contains_answer(candidate_terms: &HashSet<String>, answer_terms: &HashSet<String>) -> bool {
        if answer_terms.is_empty() {
            return false;
        }
        answer_terms.is_subset(candidate_terms)
    }

    /// Generate up to `questions_per_doc` tuples for a single document.
    ///
    /// Sentences are visited in descending salience order; within each sentence,
    /// questions are produced in canonical template order. Distractors are mined
    /// from `corpus`. The result is capped at `questions_per_doc`.
    #[must_use]
    pub fn generate_for(&self, doc: &Document, corpus: &[Document]) -> Vec<SyntheticQa> {
        let mut out: Vec<SyntheticQa> = Vec::new();
        for sentence in self.ranked_sentences(doc) {
            if out.len() >= self.config.questions_per_doc {
                break;
            }
            let tuples = self.templater.generate(&sentence, doc, &self.config.types);
            for (question, answer, question_type) in tuples {
                if out.len() >= self.config.questions_per_doc {
                    break;
                }
                if question.trim().is_empty() || answer.trim().is_empty() {
                    continue;
                }
                let distractor_ids = self.mine_distractors(&question, &answer, &doc.id, corpus);
                out.push(SyntheticQa {
                    question,
                    answer,
                    source_id: doc.id.clone(),
                    source_sentence: sentence.clone(),
                    distractor_ids,
                    question_type,
                });
            }
        }
        out
    }

    /// Generate synthetic evaluation tuples for the entire corpus.
    ///
    /// Documents are processed in corpus order; each document contributes at most
    /// `questions_per_doc` tuples.
    ///
    /// # Errors
    ///
    /// Returns [`SyntheticEvalError::EmptyCorpus`] when `corpus` is empty.
    pub fn generate(&self, corpus: &[Document]) -> Result<Vec<SyntheticQa>, SyntheticEvalError> {
        if corpus.is_empty() {
            return Err(SyntheticEvalError::EmptyCorpus);
        }
        let mut out: Vec<SyntheticQa> = Vec::new();
        for doc in corpus {
            out.extend(self.generate_for(doc, corpus));
        }
        Ok(out)
    }
}
