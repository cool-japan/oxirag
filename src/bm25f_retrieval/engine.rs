//! [`Bm25fIndex`] — per-field term statistics and classical BM25F scoring.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::types::{Bm25fConfig, Bm25fDocument, Bm25fError, Bm25fHit, Bm25fResult};

// ── tokenizer ────────────────────────────────────────────────────────────────

/// Tokenize `text`: split on non-alphanumeric boundaries (whitespace and
/// punctuation alike), lowercase, and drop empty fragments.
///
/// Deliberately simple and self-contained, matching the tokenizer used by
/// every other lexical module in this crate (see e.g. `searchain::engine` or
/// `sparse_retrieval::encoder`): no stemming, no stopword removal, no
/// minimum-length filter. BM25F's field-weighting behaviour does not depend
/// on any of that, and keeping the tokenizer this simple makes hand-computed
/// test fixtures tractable.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Normalise an arbitrary caller-supplied `term` (as passed to the
/// introspection accessors) into the same canonical token form used
/// internally, by running it through [`tokenize`] and taking the first
/// token. Returns `None` when `term` contains no alphanumeric content at
/// all.
fn lookup_key(term: &str) -> Option<String> {
    tokenize(term).into_iter().next()
}

/// Per-field statistics accumulated for a single field occurrence on a
/// single document during indexing: its raw per-term frequencies and its
/// total token length.
struct FieldOccurrence {
    term_freq: HashMap<String, u32>,
    len: usize,
}

// ── Bm25fIndex ───────────────────────────────────────────────────────────────

/// An in-memory BM25F index: per-document, per-term combined cross-field
/// pseudo-frequencies, corpus-wide document frequencies, and per-field
/// average lengths, ready to be scored against queries.
///
/// See the [module documentation](crate::bm25f_retrieval) for the exact
/// pseudo-frequency and scoring formulas this implements.
#[derive(Debug, Clone)]
pub struct Bm25fIndex {
    /// Configuration this index was built with.
    config: Bm25fConfig,
    /// Whether [`build`](Self::build) has populated the statistics below.
    built: bool,
    /// Document ids, in corpus (insertion) order; `doc_ids[i]` corresponds
    /// to every other `Vec` field's index `i`.
    doc_ids: Vec<String>,
    /// Maps a document id back to its position in `doc_ids`, for the
    /// introspection accessors. When a corpus contains duplicate ids, the
    /// later document's position wins.
    doc_id_index: HashMap<String, usize>,
    /// `doc_term_pseudo_freq[i]` maps every term with non-zero combined
    /// pseudo-frequency in document `i` to that pseudo-frequency `tf~(t,
    /// d)`. Terms absent from the map have an implicit pseudo-frequency of
    /// `0.0`.
    doc_term_pseudo_freq: Vec<HashMap<String, f32>>,
    /// Number of documents (out of the whole corpus) containing each term
    /// in at least one field, regardless of that field's configured weight.
    doc_freq: HashMap<String, usize>,
    /// Precomputed BM25 inverse document frequency per term.
    idf: HashMap<String, f32>,
    /// Corpus-wide average token length of each *named* field (e.g. the
    /// average length of every document's `"title"` field, separately from
    /// the average length of every document's `"body"` field). A document
    /// missing a given field contributes a length of `0` to that field's
    /// average, matching the convention that missing fields behave exactly
    /// like empty ones.
    avg_field_len: BTreeMap<String, f32>,
}

impl Bm25fIndex {
    /// Create a new, empty index with the given configuration.
    #[must_use]
    pub fn new(config: Bm25fConfig) -> Self {
        Self {
            config,
            built: false,
            doc_ids: Vec::new(),
            doc_id_index: HashMap::new(),
            doc_term_pseudo_freq: Vec::new(),
            doc_freq: HashMap::new(),
            idf: HashMap::new(),
            avg_field_len: BTreeMap::new(),
        }
    }

    /// The configuration this index was constructed with.
    #[must_use]
    pub fn config(&self) -> &Bm25fConfig {
        &self.config
    }

    /// Number of documents currently held by the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Return `true` when the index holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Whether [`build`](Self::build) has been called successfully at least
    /// once.
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Fit the index on `documents`: compute per-field average lengths,
    /// per-document combined cross-field pseudo-frequencies, corpus-wide
    /// document frequencies, and per-term IDF.
    ///
    /// Calling `build` again (e.g. with an updated corpus) fully replaces
    /// the previous statistics.
    ///
    /// # Errors
    ///
    /// Returns [`Bm25fError::EmptyCorpus`] when `documents` is empty. A
    /// non-empty corpus whose documents have no fields (or only
    /// blank-text fields) is *not* an error: `build` succeeds and produces
    /// an index that legitimately matches nothing (see
    /// [`search`](Self::search)).
    pub fn build(&mut self, documents: &[Bm25fDocument]) -> Bm25fResult<()> {
        if documents.is_empty() {
            return Err(Bm25fError::EmptyCorpus);
        }

        let num_docs = documents.len();
        #[allow(clippy::cast_precision_loss)]
        let num_docs_f = num_docs as f32;

        let avg_field_len = Self::compute_average_field_lengths(documents, num_docs_f);

        let mut doc_ids: Vec<String> = Vec::with_capacity(num_docs);
        let mut doc_id_index: HashMap<String, usize> = HashMap::with_capacity(num_docs);
        let mut doc_term_pseudo_freq: Vec<HashMap<String, f32>> = Vec::with_capacity(num_docs);
        let mut doc_freq: HashMap<String, usize> = HashMap::new();

        for (position, doc) in documents.iter().enumerate() {
            doc_ids.push(doc.id.clone());
            doc_id_index.insert(doc.id.clone(), position);

            let (combined, terms_in_doc) = self.combine_document_fields(doc, &avg_field_len);

            for term in &terms_in_doc {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }

            doc_term_pseudo_freq.push(combined);
        }

        let idf = Self::compute_idf(&doc_freq, num_docs_f, self.config.idf_smoothing);

        self.doc_ids = doc_ids;
        self.doc_id_index = doc_id_index;
        self.doc_term_pseudo_freq = doc_term_pseudo_freq;
        self.doc_freq = doc_freq;
        self.idf = idf;
        self.avg_field_len = avg_field_len;
        self.built = true;
        Ok(())
    }

    /// Pass 1 of indexing: the corpus-wide average token length of every
    /// named field, keyed by field name. A field name's total is summed
    /// across every occurrence of that name in the corpus (documents
    /// lacking the field, or repeating it, both fall out naturally); the
    /// average always divides by the full document count `num_docs_f`, so a
    /// field that only a minority of documents carry is not artificially
    /// inflated.
    fn compute_average_field_lengths(
        documents: &[Bm25fDocument],
        num_docs_f: f32,
    ) -> BTreeMap<String, f32> {
        let mut total_field_len: HashMap<String, usize> = HashMap::new();
        for doc in documents {
            for field in &doc.fields {
                let len = tokenize(&field.text).len();
                *total_field_len.entry(field.name.clone()).or_insert(0) += len;
            }
        }

        let mut avg_field_len: BTreeMap<String, f32> = BTreeMap::new();
        for (name, total_len) in &total_field_len {
            #[allow(clippy::cast_precision_loss)]
            let avg = *total_len as f32 / num_docs_f;
            avg_field_len.insert(name.clone(), avg);
        }
        avg_field_len
    }

    /// Pass 2 of indexing, for one document: the combined cross-field
    /// pseudo-frequency `tf~(t, d)` of every term the document contains in
    /// at least one non-zero-weight field, plus the full set of terms
    /// present in *any* field (including zero-weight ones), used by the
    /// caller to update document frequency.
    ///
    /// Fields are folded in `doc.fields` order (not via an intermediate hash
    /// map), which is what keeps this routine — and therefore the whole
    /// index — deterministic: floating-point addition is not perfectly
    /// associative, so the *order* in which per-field contributions are
    /// summed into a term's combined value must not depend on hash-map
    /// iteration order.
    fn combine_document_fields(
        &self,
        doc: &Bm25fDocument,
        avg_field_len: &BTreeMap<String, f32>,
    ) -> (HashMap<String, f32>, HashSet<String>) {
        let mut combined: HashMap<String, f32> = HashMap::new();
        let mut terms_in_doc: HashSet<String> = HashSet::new();

        for field in &doc.fields {
            let tokens = tokenize(&field.text);
            if tokens.is_empty() {
                continue;
            }

            let occurrence = Self::count_field_terms(&tokens);
            let field_weight = self.config.field_weight(&field.name);
            let avg_len = avg_field_len.get(&field.name).copied().unwrap_or(0.0);

            #[allow(clippy::cast_precision_loss)]
            let field_len = occurrence.len as f32;
            let ratio = if avg_len > 0.0 {
                field_len / avg_len
            } else {
                0.0
            };
            // `max(f32::EPSILON)` is a defence-in-depth guard: with `b_f` in
            // its conventional `[0.0, 1.0]` range this denominator is always
            // strictly positive (raw_tf > 0 in this branch implies
            // field_len > 0, which in turn implies avg_len > 0 and ratio >
            // 0), so the guard is unreachable in practice, but it keeps a
            // pathological out-of-range `b_f` from ever producing a zero or
            // negative denominator.
            let norm = (1.0 - field_weight.b + field_weight.b * ratio).max(f32::EPSILON);

            for (term, &raw_tf) in &occurrence.term_freq {
                terms_in_doc.insert(term.clone());

                if field_weight.weight <= 0.0 {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let raw_tf_f = raw_tf as f32;
                let contribution = field_weight.weight * raw_tf_f / norm;
                *combined.entry(term.clone()).or_insert(0.0) += contribution;
            }
        }

        (combined, terms_in_doc)
    }

    /// Raw per-term frequency within one field's token stream, plus the
    /// field's total token count.
    fn count_field_terms(tokens: &[String]) -> FieldOccurrence {
        let mut term_freq: HashMap<String, u32> = HashMap::new();
        for tok in tokens {
            *term_freq.entry(tok.clone()).or_insert(0) += 1;
        }
        FieldOccurrence {
            term_freq,
            len: tokens.len(),
        }
    }

    /// Pass 3 of indexing: standard BM25 IDF per term, smoothed to stay
    /// finite and strictly positive even when a term appears in every
    /// document:
    ///
    /// `idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + smoothing))`
    fn compute_idf(
        doc_freq: &HashMap<String, usize>,
        num_docs_f: f32,
        idf_smoothing: f32,
    ) -> HashMap<String, f32> {
        let mut idf: HashMap<String, f32> = HashMap::with_capacity(doc_freq.len());
        for (term, &df) in doc_freq {
            #[allow(clippy::cast_precision_loss)]
            let df_f = df as f32;
            let value = (1.0 + (num_docs_f - df_f + 0.5) / (df_f + idf_smoothing)).ln();
            idf.insert(term.clone(), value);
        }
        idf
    }

    /// Search the index for the `k` documents most relevant to `query`,
    /// under the BM25F scoring formula described in the
    /// [module documentation](crate::bm25f_retrieval).
    ///
    /// The query is tokenized and deduplicated (repeated query words do not
    /// double-count); each unique query term contributes
    /// `IDF(t) * tf~(t, d) * (k1 + 1) / (k1 + tf~(t, d))` to a document's
    /// score, and a document's total score is the sum of its per-term
    /// contributions. Documents with a total score of `0.0` (no matching
    /// signal at all) are omitted rather than reported. Hits are sorted by
    /// descending score, ties broken by ascending document id for
    /// determinism, then truncated to `k`.
    ///
    /// # Errors
    ///
    /// Returns [`Bm25fError::NotBuilt`] if [`build`](Self::build) has not
    /// been called yet, and [`Bm25fError::EmptyQuery`] if `query` is blank.
    /// A non-blank query that tokenizes to nothing meaningful (e.g. pure
    /// punctuation), or that matches no document, is not an error: it
    /// yields an empty result vector.
    pub fn search(&self, query: &str, k: usize) -> Bm25fResult<Vec<Bm25fHit>> {
        if !self.built {
            return Err(Bm25fError::NotBuilt);
        }
        if query.trim().is_empty() {
            return Err(Bm25fError::EmptyQuery);
        }

        let unique_terms = Self::unique_query_terms(query);
        if unique_terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let mut hits: Vec<Bm25fHit> = Vec::new();
        for (doc_idx, doc_id) in self.doc_ids.iter().enumerate() {
            let score = self.score_document(doc_idx, &unique_terms);
            if score > 0.0 {
                hits.push(Bm25fHit {
                    doc_id: doc_id.clone(),
                    score,
                });
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Tokenize `query` and deduplicate, preserving first-occurrence order.
    ///
    /// Order preservation (rather than e.g. collecting into a `HashSet`) is
    /// what makes multi-term scoring deterministic: the per-term
    /// contributions are later summed in this exact order, and hash-set
    /// iteration order is not guaranteed stable across process runs.
    fn unique_query_terms(query: &str) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut unique_terms: Vec<String> = Vec::new();
        for tok in tokenize(query) {
            if seen.insert(tok.clone()) {
                unique_terms.push(tok);
            }
        }
        unique_terms
    }

    /// Sum the BM25 saturation contribution of every query term for one
    /// document, by its position `doc_idx` in `doc_ids`.
    fn score_document(&self, doc_idx: usize, unique_terms: &[String]) -> f32 {
        let pseudo = &self.doc_term_pseudo_freq[doc_idx];
        unique_terms
            .iter()
            .map(|term| {
                let tf_tilde = pseudo.get(term).copied().unwrap_or(0.0);
                if tf_tilde <= 0.0 {
                    return 0.0;
                }
                let idf = self.idf.get(term).copied().unwrap_or(0.0);
                idf * (tf_tilde * (self.config.k1 + 1.0)) / (self.config.k1 + tf_tilde)
            })
            .sum()
    }

    /// Number of documents containing `term` in at least one field,
    /// regardless of that field's configured weight. Returns `0` for a term
    /// never seen anywhere in the corpus, or before the index is built.
    #[must_use]
    pub fn document_frequency(&self, term: &str) -> usize {
        lookup_key(term)
            .and_then(|key| self.doc_freq.get(&key).copied())
            .unwrap_or(0)
    }

    /// The precomputed BM25 inverse document frequency of `term`. Returns
    /// `0.0` for a term never seen anywhere in the corpus, or before the
    /// index is built — a value the real formula can never itself produce,
    /// since it is strictly positive for every term that was actually
    /// observed.
    #[must_use]
    pub fn idf(&self, term: &str) -> f32 {
        lookup_key(term)
            .and_then(|key| self.idf.get(&key).copied())
            .unwrap_or(0.0)
    }

    /// The combined cross-field BM25F pseudo-frequency `tf~(term, doc)` for
    /// one document and term. Returns `0.0` when either the document id or
    /// the term is not found (including: the term appears only in a
    /// zero-weight field of that document).
    #[must_use]
    pub fn pseudo_frequency(&self, doc_id: &str, term: &str) -> f32 {
        let Some(&doc_idx) = self.doc_id_index.get(doc_id) else {
            return 0.0;
        };
        let Some(key) = lookup_key(term) else {
            return 0.0;
        };
        self.doc_term_pseudo_freq[doc_idx]
            .get(&key)
            .copied()
            .unwrap_or(0.0)
    }

    /// The corpus-wide average token length of the named field
    /// `field_name`. Returns `0.0` for a field name never seen in the
    /// corpus, or before the index is built.
    #[must_use]
    pub fn average_field_length(&self, field_name: &str) -> f32 {
        self.avg_field_len.get(field_name).copied().unwrap_or(0.0)
    }
}

impl Default for Bm25fIndex {
    /// An empty, unbuilt index using [`Bm25fConfig::default`].
    fn default() -> Self {
        Self::new(Bm25fConfig::default())
    }
}
