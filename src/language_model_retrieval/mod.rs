//! Probabilistic language-model retrieval: query likelihood with three
//! smoothing schemes, the Divergence-From-Randomness family (`PL2`, `DPH`),
//! and `RM3` relevance-model feedback.
//!
//! This is the crate's second scoring family, and it is a genuinely different
//! *kind* of model from the first. Everything else that ranks a document by
//! its words here — [`hybrid_search::bm25`](crate::hybrid_search::bm25), and
//! the feature-gated `bm25f_retrieval`, `sparse_retrieval` and
//! `coil_retrieval` — descends from BM25: a saturating
//! `tf` term, an `IDF` term, and a length-normalization term, multiplied
//! together and summed. Every one of them asks *"how strongly does this
//! document's term profile resemble the query's?"*.
//!
//! The models here ask two entirely different questions.
//!
//! **Query likelihood** treats the document as a *generative* probability
//! distribution over words and asks: *"if I had sampled words at random from
//! this document, how likely is it that I would have typed this query?"*
//!
//! ```text
//! score(D, Q) = Σ_{w ∈ Q} qtf(w) · log P(w | D)
//! ```
//!
//! There is no `IDF` anywhere in it. Rare terms are not up-weighted by fiat;
//! they matter because a document that contains a rare term has a *far* higher
//! probability of generating it than the collection at large does. The
//! discrimination emerges from the probability model rather than being bolted
//! on, and the entire art of the method lies in the one thing BM25 never has
//! to face: a maximum-likelihood language model assigns probability **zero**
//! to every word the document happens not to contain, so `log P(w | D) = -∞`
//! and a single missing query term annihilates the score. Fixing that is
//! *smoothing*, and it is not a numerical detail — the choice of smoothing
//! scheme *is* the retrieval model, and different schemes produce genuinely
//! different rankings. See [`smoothing`].
//!
//! **Divergence From Randomness** asks instead: *"how improbable is this
//! term's frequency in this document under a model in which words are
//! sprinkled at random — and how much of that improbability should I actually
//! believe?"* The score is the surprisal of the observed frequency under a
//! randomness model (Poisson, for `PL2`), discounted by an after-effect factor
//! that accounts for terms being bursty. See [`dfr`].
//!
//! **`RM3`** closes the loop. Query likelihood has no notion of relevance in it
//! at all; `RM1` (Lavrenko & Croft, 2001) estimates one — a distribution
//! `P(w | R)` over the words a relevant document would generate — as the
//! query-likelihood-weighted mixture of the *smoothed language models* of the
//! top-ranked documents, and `RM3` interpolates it back into the query. See
//! [`rm3`], which also contrasts it, in detail, against this crate's two
//! *non*-probabilistic feedback mechanisms (Rocchio centroid arithmetic in
//! [`relevance_feedback`](crate::relevance_feedback) and rank-heuristic term
//! counting in [`query_expansion::prf`](crate::query_expansion::prf)).
//!
//! # How this differs from BM25 (and from the crate's other sparse scorers)
//!
//! | | `hybrid_search::bm25`, `bm25f_retrieval` | `sparse_retrieval` (`SPLADE`), `coil_retrieval` | `language_model_retrieval` (this module) |
//! |---|---|---|---|
//! | What is estimated | a relevance *score* | learned term *weights* | a probability *distribution* |
//! | Term weighting | `IDF`, an axiomatic multiplier | learned from data by a neural encoder | none — discrimination emerges from `P(w \| D)` vs `P(w \| C)` |
//! | Length handling | `b`-interpolated `dl/avgdl` divisor | absorbed into the learned weights | falls out of the smoothing scheme: Dirichlet smooths *less* as `\|D\|` grows |
//! | Absent query term | contributes `0` | contributes `0` | contributes `log(K_D · P(w \| C))` — a *finite, negative*, length- and rarity-dependent penalty |
//! | Zero-frequency problem | does not arise | does not arise | **the central problem**; see [`smoothing`] |
//! | Feedback | none | none | `RM1` / `RM3` relevance models |
//!
//! The row that matters most is "absent query term". BM25 shrugs at a missing
//! term. A language model *cannot*: the term's probability under `D` is not
//! zero (that is what smoothing buys), so it contributes a real, negative
//! number whose magnitude depends on how common the term is in the collection
//! and on how much smoothing mass `D` received. Documents are penalized for
//! what they *lack*, in a quantity the model derives rather than one a
//! designer chose. That is the whole reason both scoring families are worth
//! having.
//!
//! # Numerical hazards, and what this module does about them
//!
//! Every formula here is a logarithm of a probability, and the interesting
//! part of the implementation is the set of places where the mathematics
//! diverges but the answer must not. Each is derived and resolved in the
//! relevant file; briefly:
//!
//! | hazard | where | resolution |
//! |---|---|---|
//! | `log P(w \| D) = -∞` for an unseen term | [`smoothing`] | smoothing itself: every scheme backs off to `P(w \| C) > 0` |
//! | `cf(w) = 0` (out of vocabulary) — even `P(w \| C)` is zero | [`smoothing`] | floor at `1 / (\|C\| + 1)`, strictly below the rarest possible *seen* term |
//! | `μ = 0` / `λ = 0` / `δ = 0` — smoothing switched off entirely | [`smoothing`] | clamp `P(w \| D)` at [`LM_MIN_PROBABILITY`], keeping `log` finite |
//! | `\|D\| = 0` — `tf / \|D\|` is `0 / 0` | [`smoothing`] | back off wholly to the collection model (`K_D = 1`) |
//! | `tfn → 0` in `PL2` — Stirling's `½·log₂(2π·tfn) → -∞` | [`dfr`] | absent term ⇒ contribute `0`; present-but-vanishing ⇒ floor `tfn` at [`PL2_MIN_TFN`] |
//! | `tf = dl` in `DPH` — `0 · (-∞)` | [`dfr`] | the limit is exactly `0` (`u²·log u → 0`), and that is what is returned |
//! | `exp(score_QL)` underflows to `0` for every feedback document | [`rm3`] | log-sum-exp: normalize `P(D \| Q)` by max-subtraction |
//!
//! The last one is the least obvious and the most destructive. A
//! query-likelihood score is a sum of log-probabilities, so `-800` is an
//! ordinary value for a long query — and `f64::exp(-800.0)` is exactly `0.0`.
//! Computing `P(D | Q) ∝ exp(score)` directly does not lose a little
//! precision: it produces a vector of zeros, and the normalization then
//! divides `0` by `0`. `RM1` would silently return `NaN` for every term. See
//! [`rm3`].
//!
//! # Determinism
//!
//! Nothing in this module is random. Ranking is by descending score with ties
//! broken by ascending `document_id`, term selection by descending weight with
//! ties broken by ascending term, and all float ordering goes through
//! [`f64::total_cmp`]. Identical inputs produce identical, stably-ordered
//! output.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "language-model-retrieval")]
//! # {
//! use oxirag::language_model_retrieval::{
//!     LmRetrievalConfig, LmRetrievalIndex, LmSmoothing, Rm3Config,
//! };
//!
//! // Dirichlet smoothing; mu is small because the corpus is tiny.
//! let config = LmRetrievalConfig::query_likelihood(LmSmoothing::Dirichlet { mu: 10.0 });
//! let mut index = LmRetrievalIndex::new(config).expect("valid configuration");
//!
//! index.add_document("d1", "renal renal kidney kidney kidney failure").expect("unique id");
//! index.add_document("d2", "renal kidney kidney kidney disease").expect("unique id");
//! index.add_document("d3", "kidney kidney kidney transplant surgery").expect("unique id");
//! index.add_document("d4", "automobile engine transmission gearbox").expect("unique id");
//! index.build().expect("valid configuration");
//!
//! // The query "renal" ranks d1 and d2 first -- and misses d3 entirely, which
//! // never uses the word.
//! let hits = index.search("renal", 3).expect("built index, non-empty query");
//! assert_eq!(hits[0].document_id, "d1");
//! assert!(!hits.iter().any(|hit| hit.document_id == "d3"));
//!
//! // RM3 reads "kidney" off the feedback documents and expands the query with it.
//! let rm3 = Rm3Config::new(2, 4, 0.5);
//! let expanded = index.expand_query_rm3("renal", &rm3).expect("built index");
//! assert!(expanded.contains("kidney"));
//! assert!(expanded.weight_of("kidney") > 0.1);
//!
//! // ...and the expanded query now finds d3.
//! let hits = index.search_rm3("renal", &rm3, 3).expect("built index");
//! assert!(hits.iter().any(|hit| hit.document_id == "d3"));
//! # }
//! ```

pub mod dfr;
pub mod index;
pub mod rm3;
pub mod smoothing;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use dfr::{dph_term_score, pl2_term_score};
pub use index::{LmDocumentStats, LmRetrievalIndex};
pub use rm3::{LmRelevanceModel, lm_log_sum_exp, lm_query_posteriors};
pub use smoothing::{LmCollectionModel, LmSmoothingComponents};
pub use types::{
    DfrModel, LM_MIN_PROBABILITY, LmHit, LmRetrievalConfig, LmRetrievalError, LmRetrievalResult,
    LmScoringModel, LmSmoothing, PL2_MIN_TFN, Rm3Config, Rm3ExpandedQuery, Rm3Term, lm_tokenize,
};
