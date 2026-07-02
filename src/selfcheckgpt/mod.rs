//! `SelfCheckGPT`: zero-resource hallucination detection via sampling
//! consistency (Manakul et al., EMNLP 2023).
//!
//! The core idea is that a hallucinated statement is unlikely to be
//! reproduced consistently when a stochastic generator is re-sampled: if the
//! model *knows* a fact, repeatedly sampling the same prompt should keep
//! restating it (perhaps paraphrased); if the model is confabulating, each
//! sample tends to diverge. `SelfCheckGPT` exploits this **without querying any
//! external knowledge base**:
//!
//! 1. Sample the same prompt `K` times from the (stochastic) generator to
//!    obtain `K` additional responses.
//! 2. Split the *main* response — the one actually returned to the user —
//!    into sentences.
//! 3. For each main-response sentence, measure how well it is corroborated by
//!    the `K` samples using one of three [`SelfCheckVariant`]s.
//! 4. Sentences whose inconsistency score clears
//!    [`SelfCheckConfig::hallucination_threshold`] are flagged as likely
//!    hallucinations; the mean over all sentences is the document-level
//!    [`SelfCheckScore::overall_score`].
//!
//! ## Honest heuristic-proxy notice
//!
//! The original paper proposes several variants, two of which normally rely
//! on trained models we do not have access to in this pure-Rust, zero-dependency
//! crate. Rather than fabricate a fake model, each variant here is an honestly
//! documented, purely lexical/statistical stand-in:
//!
//! | Variant | Paper's method | This crate's proxy |
//! |---------|-----------------|---------------------|
//! | [`SelfCheckVariant::NGram`] | n-gram LM trained on the samples | Pooled n-gram frequency counting (no training, just counting) |
//! | [`SelfCheckVariant::NliLite`] | Trained NLI classifier scoring entailment/contradiction | Jaccard lexical overlap against each sample's best-matching sentence |
//! | [`SelfCheckVariant::QaLite`] | LLM-generated question + QA model answer-matching | Salient-keyword cloze masking + exact keyword presence check |
//!
//! None of these variants call an external LLM, NLI classifier, or QA model;
//! they operate purely on the text supplied by the caller.
//!
//! ## Distinct from other hallucination-adjacent modules
//!
//! | Module | Compares against | Produces |
//! |--------|-------------------|----------|
//! | `hallucination_detector` | Retrieved **source documents** | Per-claim support score vs. sources |
//! | `self_consistency` | Multiple **reasoning paths** for the same question | A single majority-vote final answer (marginalization) |
//! | `semantic_entropy` | Multiple **answer samples**, clustered by meaning | Document-level entropy over meaning clusters |
//! | **`selfcheckgpt`** | Multiple **raw response samples** to the same prompt | Per-*sentence* inconsistency score for a *fixed* main response |
//!
//! `selfcheckgpt` never touches retrieved context (unlike
//! `hallucination_detector`), never picks a winning answer (unlike
//! `self_consistency`), and never clusters answers into meanings (unlike
//! `semantic_entropy`) — it scores each sentence of one fixed response by how
//! often its content recurs across independently sampled responses.
//!
//! # Example
//!
//! ```
//! use oxirag::selfcheckgpt::{SelfCheckConfig, SelfCheckScorer, SelfCheckVariant};
//!
//! // Three stochastic re-samples of the same prompt: all three agree that
//! // Paris is the capital of France, but none of them ever mention "unicorns".
//! let samples = vec![
//!     "Paris is the capital of France and a major cultural center.".to_string(),
//!     "The capital of France is Paris, known for its cultural heritage.".to_string(),
//!     "Paris, the capital city of France, is famous worldwide.".to_string(),
//! ];
//!
//! // The main response repeats the well-corroborated fact, then adds a
//! // fabricated claim that appears in none of the samples.
//! let main_response =
//!     "Paris is the capital of France. Paris was founded by unicorns in 1200.";
//!
//! let scorer = SelfCheckScorer::new(SelfCheckConfig::new().with_variant(SelfCheckVariant::NGram));
//! let result = scorer.score(main_response, &samples).expect("enough samples");
//!
//! assert_eq!(result.sentence_checks.len(), 2);
//! // The corroborated sentence is judged less inconsistent than the
//! // fabricated one.
//! assert!(
//!     result.sentence_checks[0].inconsistency_score
//!         < result.sentence_checks[1].inconsistency_score
//! );
//! // The fabricated sentence clears the default hallucination threshold.
//! assert!(result.sentence_checks[1].is_hallucination);
//! ```

mod ngram;
mod nli_lite;
mod qa_lite;
pub mod scorer;
#[cfg(test)]
mod tests;
pub mod types;

pub use scorer::SelfCheckScorer;
pub use types::{SelfCheckConfig, SelfCheckError, SelfCheckScore, SelfCheckVariant, SentenceCheck};
