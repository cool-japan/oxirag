//! [`LmRetrievalIndex`] — the in-memory index that owns the collection
//! statistics and evaluates every model in this module against them.
//!
//! The index is deliberately self-contained: it keeps its own term
//! frequencies, collection frequencies, posting lists and token totals rather
//! than borrowing another module's. Query likelihood, `PL2`, `DPH` and `RM3`
//! all need `cf(w)`, `|C|`, `|D|`, `|D|_unique`, `df(w)`, `N` and `avgdl`
//! under one roof, and no existing structure in this crate exposes that set.
//!
//! # Which documents get scored
//!
//! The two scoring families need different candidate sets, and the difference
//! is a matter of correctness, not of optimization:
//!
//! - **Query likelihood scores every document.** A smoothed language model
//!   assigns non-zero probability to every term, so `P(Q | D)` is defined for
//!   every document — including those containing none of the query's terms.
//!   Skipping them would not merely omit hopeless candidates: under Dirichlet
//!   smoothing a document containing a query term exactly once can score
//!   *below* one that does not contain it at all, if the former is far longer
//!   (`(1 + μp)/(|D| + μ)` falls below `μp/(|D'| + μ)` once `|D| ≫ |D'|`), so
//!   the posting-list union is not even an upper set of the ranking.
//! - **`DFR` scores only the posting-list union.** `DFR` is not generative; a
//!   term that does not occur contributes exactly `0` (see [`super::dfr`]), so
//!   every document outside the union would score exactly `0` and carry no
//!   evidence whatsoever. Returning them would be noise.
//!
//! # Determinism
//!
//! Ranking is by descending score, ties broken by ascending `document_id`.
//! Float comparison uses [`f64::total_cmp`] throughout: `partial_cmp` returns
//! `None` on `NaN`, and feeding `sort_by` a comparator that is not a total
//! order produces arbitrary output.

// Counting statistics (`tf`, `|D|`, `cf`, `|C|`, `N`) are stored as exact
// integers and converted to `f64` at the point of use, because every formula
// in this module is real-valued. The conversion is lossy only above 2^53
// tokens -- some nine petabytes of text -- and the alternative (storing counts
// as floats) would lose exactness where it actually matters, in the counting.
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet};

use super::rm3::{LmRelevanceModel, lm_query_posteriors, sort_terms_by_weight};
use super::smoothing::LmCollectionModel;
use super::types::{
    LmHit, LmRetrievalConfig, LmRetrievalError, LmRetrievalResult, LmScoringModel, Rm3Config,
    Rm3ExpandedQuery, lm_tokenize,
};

// ── LmDocumentStats ──────────────────────────────────────────────────────────

/// Everything the models need to know about a single indexed document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LmDocumentStats {
    /// The identifier the document was added under.
    pub document_id: String,
    /// `tf(w, D)` for every term the document contains.
    pub term_frequencies: HashMap<String, u64>,
    /// `|D|` — the document's length in tokens (the sum of its term
    /// frequencies).
    pub length: u64,
    /// `|D|_unique` — the number of *distinct* terms in the document. Absolute
    /// discounting needs this and nothing else does.
    pub unique_terms: u64,
}

impl LmDocumentStats {
    /// Build the statistics of a document from its token stream.
    #[must_use]
    pub fn from_terms(document_id: impl Into<String>, terms: &[String]) -> Self {
        let mut term_frequencies: HashMap<String, u64> = HashMap::new();
        for term in terms {
            *term_frequencies.entry(term.clone()).or_insert(0) += 1;
        }
        let length = terms.len() as u64;
        let unique_terms = term_frequencies.len() as u64;
        Self {
            document_id: document_id.into(),
            term_frequencies,
            length,
            unique_terms,
        }
    }

    /// `tf(term, D)`, or `0` if the document does not contain the term.
    #[must_use]
    pub fn term_frequency(&self, term: &str) -> u64 {
        self.term_frequencies.get(term).copied().unwrap_or(0)
    }
}

// ── LmRetrievalIndex ─────────────────────────────────────────────────────────

/// An in-memory language-model retrieval index.
///
/// Add documents with [`Self::add_document`], call [`Self::build`] to derive
/// the collection statistics, then [`Self::search`]. Adding a document after a
/// build invalidates the derived statistics and requires another
/// [`Self::build`] — searching in between returns [`LmRetrievalError::NotBuilt`]
/// rather than silently scoring against stale term statistics.
#[derive(Debug, Clone)]
pub struct LmRetrievalIndex {
    /// The scoring model, smoothing scheme and tokenizer settings.
    config: LmRetrievalConfig,
    /// Per-document statistics, in insertion order.
    documents: Vec<LmDocumentStats>,
    /// Maps a `document_id` to its position in the `documents` vector.
    document_index: HashMap<String, usize>,
    /// `cf(w)` — total occurrences of each term across the whole collection.
    collection_frequencies: HashMap<String, u64>,
    /// Posting lists: `term -> ascending document positions`.
    postings: HashMap<String, Vec<usize>>,
    /// `|C|` — the collection's total token count.
    total_tokens: u64,
    /// Whether the derived statistics above are current.
    built: bool,
}

impl LmRetrievalIndex {
    /// Create an empty index.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] if any of the
    /// configuration's hyper-parameters is outside the range in which its
    /// model is well-posed (see [`LmRetrievalConfig::validate`]).
    pub fn new(config: LmRetrievalConfig) -> LmRetrievalResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            documents: Vec::new(),
            document_index: HashMap::new(),
            collection_frequencies: HashMap::new(),
            postings: HashMap::new(),
            total_tokens: 0,
            built: false,
        })
    }

    /// The index's configuration.
    #[must_use]
    pub fn config(&self) -> &LmRetrievalConfig {
        &self.config
    }

    /// Add a document, tokenizing its text with [`lm_tokenize`].
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::DuplicateDocumentId`] if a document with
    /// the same identifier has already been added.
    pub fn add_document(
        &mut self,
        document_id: impl Into<String>,
        text: &str,
    ) -> LmRetrievalResult<()> {
        let terms = lm_tokenize(text, self.config.lowercase);
        self.add_document_terms(document_id, &terms)
    }

    /// Add a pre-tokenized document, bypassing the built-in tokenizer.
    ///
    /// Use this to plug in stemming, stop-word removal, or any other analyzer.
    /// An empty term list is accepted: it produces a document of length zero,
    /// whose language model backs off entirely to the collection model (see
    /// [`super::smoothing`]).
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::DuplicateDocumentId`] if a document with
    /// the same identifier has already been added.
    pub fn add_document_terms(
        &mut self,
        document_id: impl Into<String>,
        terms: &[String],
    ) -> LmRetrievalResult<()> {
        let document_id = document_id.into();
        if self.document_index.contains_key(&document_id) {
            return Err(LmRetrievalError::DuplicateDocumentId { document_id });
        }
        let position = self.documents.len();
        self.document_index.insert(document_id.clone(), position);
        self.documents
            .push(LmDocumentStats::from_terms(document_id, terms));
        // The derived statistics no longer describe the corpus.
        self.built = false;
        Ok(())
    }

    /// Derive the collection statistics — `cf(w)`, `|C|`, and the posting
    /// lists — from the documents added so far.
    ///
    /// Recomputes from scratch, so it is safe (if not cheap) to call after
    /// every batch of additions. An index with no documents builds
    /// successfully and answers every search with an empty result list.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::InvalidParameter`] if the configuration has
    /// been left in an invalid state (it is validated in [`Self::new`], so
    /// this cannot happen through the public API).
    pub fn build(&mut self) -> LmRetrievalResult<()> {
        self.config.validate()?;

        self.collection_frequencies.clear();
        self.postings.clear();
        self.total_tokens = 0;

        for (position, document) in self.documents.iter().enumerate() {
            self.total_tokens += document.length;
            for (term, frequency) in &document.term_frequencies {
                *self.collection_frequencies.entry(term.clone()).or_insert(0) += *frequency;
                self.postings
                    .entry(term.clone())
                    .or_default()
                    .push(position);
            }
        }
        // `documents` is walked in order, but a HashMap iteration is not, so
        // the postings of a given term arrive in document order while the
        // terms themselves do not. Sorting is therefore a no-op that documents
        // the invariant the candidate-set union relies on.
        for posting in self.postings.values_mut() {
            posting.sort_unstable();
        }

        self.built = true;
        Ok(())
    }

    /// Whether the derived statistics are current.
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// `N` — the number of indexed documents.
    #[must_use]
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Whether the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// `|C|` — the collection's total token count.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// `|V|` — the number of distinct terms in the collection.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.collection_frequencies.len()
    }

    /// `avgdl` — the mean document length, or `0.0` for an empty index.
    #[must_use]
    pub fn average_document_length(&self) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }
        self.total_tokens as f64 / self.documents.len() as f64
    }

    /// `cf(w)` — the term's total occurrences across the collection.
    #[must_use]
    pub fn collection_frequency(&self, term: &str) -> u64 {
        self.collection_frequencies.get(term).copied().unwrap_or(0)
    }

    /// `df(w)` — the number of documents containing the term at least once.
    #[must_use]
    pub fn document_frequency(&self, term: &str) -> u64 {
        self.postings.get(term).map_or(0, |p| p.len() as u64)
    }

    /// `P(w | C) = cf(w) / |C|`, floored for out-of-vocabulary terms. See
    /// [`LmCollectionModel::probability`].
    #[must_use]
    pub fn collection_probability(&self, term: &str) -> f64 {
        self.collection_model().probability(term)
    }

    /// The statistics of the document with the given identifier.
    #[must_use]
    pub fn document_stats(&self, document_id: &str) -> Option<&LmDocumentStats> {
        self.document_index
            .get(document_id)
            .and_then(|position| self.documents.get(*position))
    }

    /// Every indexed document's statistics, in insertion order.
    #[must_use]
    pub fn documents(&self) -> &[LmDocumentStats] {
        &self.documents
    }

    /// A borrowing view of the collection model.
    fn collection_model(&self) -> LmCollectionModel<'_> {
        LmCollectionModel::new(&self.collection_frequencies, self.total_tokens)
    }

    // ── Query preparation ────────────────────────────────────────────────────

    /// Tokenize a query and collapse it into `(term, qtf)` pairs.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::EmptyQuery`] if the tokenizer yields
    /// nothing.
    pub fn query_weights(&self, query: &str) -> LmRetrievalResult<Vec<(String, f64)>> {
        let tokens = lm_tokenize(query, self.config.lowercase);
        if tokens.is_empty() {
            return Err(LmRetrievalError::EmptyQuery);
        }
        let mut counts: HashMap<String, f64> = HashMap::new();
        for token in tokens {
            *counts.entry(token).or_insert(0.0) += 1.0;
        }
        let mut weights: Vec<(String, f64)> = counts.into_iter().collect();
        // Deterministic order; the score is a sum and so order-independent up
        // to floating-point associativity, but reproducibility is cheap.
        weights.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(weights)
    }

    // ── Scoring ──────────────────────────────────────────────────────────────

    /// `score_QL(D, Q) = Σ_w qtf(w) · log P(w | D)` for the document at
    /// `position`, using the configured smoothing scheme.
    ///
    /// Always query likelihood, whatever [`LmRetrievalConfig::model`] says:
    /// `RM3` is defined in terms of query likelihood and needs this even when
    /// a `DFR` model produced the first-pass ranking.
    ///
    /// Returns `0.0` (the log-probability of the empty query) for an
    /// out-of-range position.
    #[must_use]
    pub fn query_likelihood_score(&self, position: usize, query_weights: &[(String, f64)]) -> f64 {
        let Some(document) = self.documents.get(position) else {
            return 0.0;
        };
        let collection = self.collection_model();
        let smoothing = self.config.smoothing;

        query_weights
            .iter()
            .map(|(term, weight)| {
                let log_probability = smoothing.log_document_probability(
                    document.term_frequency(term),
                    document.length,
                    document.unique_terms,
                    collection.probability(term),
                );
                weight * log_probability
            })
            .sum()
    }

    /// The `DFR` score of the document at `position` under the configured
    /// `DFR` model.
    ///
    /// Returns `0.0` for an out-of-range position, and for a document
    /// containing none of the query's terms (every term contributes zero — see
    /// [`super::dfr`]).
    #[must_use]
    pub fn dfr_score(&self, position: usize, query_weights: &[(String, f64)]) -> f64 {
        let Some(document) = self.documents.get(position) else {
            return 0.0;
        };
        let LmScoringModel::Dfr(model) = self.config.model else {
            return 0.0;
        };
        let num_documents = self.documents.len() as u64;
        let avg_doc_length = self.average_document_length();

        query_weights
            .iter()
            .map(|(term, weight)| {
                model.term_score(
                    *weight,
                    document.term_frequency(term),
                    document.length,
                    avg_doc_length,
                    self.collection_frequency(term),
                    num_documents,
                )
            })
            .sum()
    }

    /// The document's score under the configured model.
    #[must_use]
    pub fn score_document(&self, position: usize, query_weights: &[(String, f64)]) -> f64 {
        match self.config.model {
            LmScoringModel::QueryLikelihood => self.query_likelihood_score(position, query_weights),
            LmScoringModel::Dfr(_) => self.dfr_score(position, query_weights),
        }
    }

    /// The set of documents the configured model is willing to score. See the
    /// module documentation for why the two families differ.
    fn candidate_positions(&self, query_weights: &[(String, f64)]) -> Vec<usize> {
        match self.config.model {
            LmScoringModel::QueryLikelihood => (0..self.documents.len()).collect(),
            LmScoringModel::Dfr(_) => {
                let mut union: HashSet<usize> = HashSet::new();
                for (term, _) in query_weights {
                    if let Some(posting) = self.postings.get(term) {
                        union.extend(posting.iter().copied());
                    }
                }
                let mut positions: Vec<usize> = union.into_iter().collect();
                positions.sort_unstable();
                positions
            }
        }
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Search with a free-text query.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::NotBuilt`] if [`Self::build`] has not been
    /// called since the last document was added, or
    /// [`LmRetrievalError::EmptyQuery`] if the query tokenizes to nothing.
    pub fn search(&self, query: &str, top_k: usize) -> LmRetrievalResult<Vec<LmHit>> {
        let weights = self.query_weights(query)?;
        self.search_weighted(&weights, top_k)
    }

    /// Search with an explicitly weighted query — the form `RM3` produces.
    ///
    /// Weights need not be integers, and need not sum to anything in
    /// particular; they multiply each term's contribution linearly.
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::NotBuilt`] if the index is not current, or
    /// [`LmRetrievalError::EmptyQuery`] if `query_weights` is empty.
    pub fn search_weighted(
        &self,
        query_weights: &[(String, f64)],
        top_k: usize,
    ) -> LmRetrievalResult<Vec<LmHit>> {
        if !self.built {
            return Err(LmRetrievalError::NotBuilt);
        }
        if query_weights.is_empty() {
            return Err(LmRetrievalError::EmptyQuery);
        }
        if top_k == 0 || self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(usize, f64)> = self
            .candidate_positions(query_weights)
            .into_iter()
            .map(|position| (position, self.score_document(position, query_weights)))
            .collect();

        scored.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                self.documents[left.0]
                    .document_id
                    .cmp(&self.documents[right.0].document_id)
            })
        });
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .enumerate()
            .map(|(rank, (position, score))| LmHit {
                document_id: self.documents[position].document_id.clone(),
                score,
                rank,
            })
            .collect())
    }

    /// Search with an [`Rm3ExpandedQuery`].
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::NotBuilt`] if the index is not current, or
    /// [`LmRetrievalError::EmptyQuery`] if the expanded query has no terms.
    pub fn search_expanded(
        &self,
        query: &Rm3ExpandedQuery,
        top_k: usize,
    ) -> LmRetrievalResult<Vec<LmHit>> {
        self.search_weighted(&query.as_weighted_terms(), top_k)
    }

    // ── RM3 ──────────────────────────────────────────────────────────────────

    /// Estimate the `RM1` relevance model `P(w | R)` from the top `fb_docs`
    /// documents of the first-pass ranking.
    ///
    /// The feedback *set* is chosen by the configured scoring model — if the
    /// index is configured for `DFR`, `DFR` selects the feedback documents —
    /// but the posteriors `P(D | Q)` that weight them are **always query
    /// likelihoods**, because that is what the relevance model is defined in
    /// terms of (`P(D | Q) ∝ exp(score_QL(D, Q))`; a `DFR` divergence weight is
    /// not a log-probability and exponentiating it would mean nothing).
    ///
    /// # Errors
    ///
    /// Returns [`LmRetrievalError::NotBuilt`] if the index is not current,
    /// [`LmRetrievalError::EmptyQuery`] if the query tokenizes to nothing, or
    /// [`LmRetrievalError::InvalidParameter`] if `config` is invalid.
    pub fn relevance_model(
        &self,
        query: &str,
        config: &Rm3Config,
    ) -> LmRetrievalResult<LmRelevanceModel> {
        let weights = self.query_weights(query)?;
        self.relevance_model_weighted(&weights, config)
    }

    /// [`Self::relevance_model`], for an already-weighted query.
    ///
    /// # Errors
    ///
    /// As [`Self::relevance_model`].
    pub fn relevance_model_weighted(
        &self,
        query_weights: &[(String, f64)],
        config: &Rm3Config,
    ) -> LmRetrievalResult<LmRelevanceModel> {
        if !self.built {
            return Err(LmRetrievalError::NotBuilt);
        }
        if query_weights.is_empty() {
            return Err(LmRetrievalError::EmptyQuery);
        }
        config.validate()?;

        let feedback = self.search_weighted(query_weights, config.fb_docs)?;

        // P(D | Q) ∝ exp(score_QL(D, Q)), normalized over the feedback set in
        // log space. A naive exp() of these scores underflows to zero -- see
        // `super::rm3`.
        let positions: Vec<usize> = feedback
            .iter()
            .filter_map(|hit| self.document_index.get(&hit.document_id).copied())
            .collect();
        let log_scores: Vec<f64> = positions
            .iter()
            .map(|position| self.query_likelihood_score(*position, query_weights))
            .collect();
        let posteriors = lm_query_posteriors(&log_scores);

        // P(w | R) = Σ_D π_D · (S_D(w) + K_D · P(w|C))
        //          = [Σ_D π_D · S_D(w)] + [Σ_D π_D · K_D] · P(w|C)
        let smoothing = self.config.smoothing;
        let mut sparse: HashMap<String, f64> = HashMap::new();
        let mut background_coefficient = 0.0_f64;
        let mut document_posteriors: Vec<(String, f64)> = Vec::with_capacity(positions.len());

        for (position, posterior) in positions.iter().zip(posteriors.iter()) {
            let document = &self.documents[*position];
            document_posteriors.push((document.document_id.clone(), *posterior));

            // K_D is w-free, so read it off any term (the components of a term
            // the document does not contain have foreground 0 and the right
            // K_D).
            let components = smoothing.components(0, document.length, document.unique_terms);
            background_coefficient += posterior * components.background_coefficient;

            for (term, frequency) in &document.term_frequencies {
                let foreground = smoothing
                    .components(*frequency, document.length, document.unique_terms)
                    .foreground;
                *sparse.entry(term.clone()).or_insert(0.0) += posterior * foreground;
            }
        }

        Ok(LmRelevanceModel::from_parts(
            sparse,
            background_coefficient,
            document_posteriors,
            |term| self.collection_probability(term),
        ))
    }

    /// Materialize a relevance model over the *entire* vocabulary:
    /// `P(w | R) = S(w) + K · P(w | C)` for every term in the collection,
    /// heaviest first.
    ///
    /// This is the full distribution, and it sums to `1` (up to floating-point
    /// error) — which is why it exists: [`LmRelevanceModel`] stores a sparse
    /// closed form for efficiency, and this is the brute-force expansion that
    /// the closed form is equivalent to.
    #[must_use]
    pub fn materialize_relevance_model(&self, model: &LmRelevanceModel) -> Vec<(String, f64)> {
        let collection = self.collection_model();
        let mut distribution: Vec<(String, f64)> = self
            .collection_frequencies
            .keys()
            .map(|term| {
                (
                    term.clone(),
                    model.probability(term, collection.probability(term)),
                )
            })
            .collect();
        sort_terms_by_weight(&mut distribution);
        distribution
    }

    /// Run `RM3`: estimate the relevance model, interpolate it with the
    /// original query, and return the weighted expanded query.
    ///
    /// # Errors
    ///
    /// As [`Self::relevance_model`].
    pub fn expand_query_rm3(
        &self,
        query: &str,
        config: &Rm3Config,
    ) -> LmRetrievalResult<Rm3ExpandedQuery> {
        let weights = self.query_weights(query)?;
        let model = self.relevance_model_weighted(&weights, config)?;
        Ok(
            model.expand_query(&weights, config.alpha, config.fb_terms, |term| {
                self.collection_probability(term)
            }),
        )
    }

    /// The full `RM3` retrieval pipeline: first-pass search, relevance model,
    /// query expansion, second-pass search.
    ///
    /// Falls back to the unexpanded query if expansion produces nothing (only
    /// possible with an empty index).
    ///
    /// # Errors
    ///
    /// As [`Self::relevance_model`].
    pub fn search_rm3(
        &self,
        query: &str,
        config: &Rm3Config,
        top_k: usize,
    ) -> LmRetrievalResult<Vec<LmHit>> {
        let expanded = self.expand_query_rm3(query, config)?;
        if expanded.is_empty() {
            return self.search(query, top_k);
        }
        self.search_expanded(&expanded, top_k)
    }
}
