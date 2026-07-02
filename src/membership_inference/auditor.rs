//! Canary generation, shadow querying, AUC-style leakage scoring, and the
//! response-perturbation defense for the `membership_inference` module.

use std::cmp::Ordering;

use super::types::{
    Canary, CanaryKind, CanaryPair, CanaryPairSignal, LeakageReport, MembershipInferenceConfig,
    MembershipInferenceError, ProbeResult, RiskLevel,
};

// ── FNV-1a deterministic pseudo-randomness ────────────────────────────────────

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Tolerance below which two pooled leakage signals are treated as tied for
/// [`CanaryAuditor::compute_auc`] ranking purposes. Signals are simple
/// weighted sums of values in `[0, 1]`, so this comfortably separates
/// "genuinely equal" from "genuinely different" without being sensitive to
/// `f32` rounding.
const AUC_TIE_EPSILON: f64 = 1e-9;

/// Deterministic `FNV-1a` 64-bit hash of a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Deterministic `FNV-1a` hash of `(seed, canary_id, salt)`, used to derive
/// every pseudo-random choice the canary generator makes.
///
/// Identical inputs always produce an identical hash, which is what keeps
/// canary generation (and hence every test built on it) fully reproducible
/// without depending on `rand`/`rand_distr`.
fn seeded_hash(seed: u64, canary_id: u64, salt: u64) -> u64 {
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(&seed.to_le_bytes());
    bytes.extend_from_slice(&canary_id.to_le_bytes());
    bytes.extend_from_slice(&salt.to_le_bytes());
    fnv1a(&bytes)
}

/// Small deterministic word bank used to build innocuous filler sentences
/// around an embedded canary marker.
const FILLER_WORDS: &[&str] = &[
    "internal",
    "record",
    "summary",
    "reference",
    "quarterly",
    "account",
    "document",
    "archive",
    "review",
    "note",
    "case",
    "department",
    "regarding",
    "status",
    "update",
    "attachment",
    "meeting",
    "project",
    "schedule",
    "outline",
    "draft",
    "revision",
    "log",
    "entry",
];

/// Build one deterministic filler "sentence" from `word_count` words drawn
/// (with repetition) from [`FILLER_WORDS`], seeded by `(seed, canary_id,
/// sentence_index)`.
fn filler_sentence(seed: u64, canary_id: u64, sentence_index: u64, word_count: usize) -> String {
    let mut words = Vec::with_capacity(word_count.max(1));
    for word_index in 0..word_count.max(1) {
        #[allow(clippy::cast_possible_truncation)]
        let salt = sentence_index
            .wrapping_mul(1000)
            .wrapping_add(word_index as u64);
        let hash = seeded_hash(seed, canary_id, salt);
        #[allow(clippy::cast_possible_truncation)]
        let idx = (hash % FILLER_WORDS.len() as u64) as usize;
        words.push(FILLER_WORDS[idx]);
    }
    let mut sentence = words.join(" ");
    // Capitalize the first letter so the filler reads as a sentence.
    if let Some(first) = sentence.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    sentence.push('.');
    sentence
}

/// Build a single rare, synthetic marker token from `(seed, canary_id,
/// token_index)`: an 8-hex-digit hash body behind a `zc` prefix that is
/// vanishingly unlikely to occur in natural text.
fn marker_token(seed: u64, canary_id: u64, token_index: usize) -> String {
    #[allow(clippy::cast_possible_truncation)]
    let hash = seeded_hash(seed, canary_id, 500_000 + token_index as u64);
    format!("zc{:08x}", hash & 0xFFFF_FFFF)
}

/// Generate the marker string for a canary: `marker_token_count` space-joined
/// rare tokens.
fn generate_marker(seed: u64, canary_id: u64, marker_token_count: usize) -> String {
    (0..marker_token_count.max(1))
        .map(|i| marker_token(seed, canary_id, i))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Generate the full synthetic canary document: `filler_sentence_count`
/// filler sentences, the embedded marker, then `filler_sentence_count` more
/// filler sentences.
fn generate_text(seed: u64, canary_id: u64, marker: &str, filler_sentence_count: usize) -> String {
    let mut sentences: Vec<String> = (0..filler_sentence_count)
        .map(|i| filler_sentence(seed, canary_id, i as u64, 6))
        .collect();
    sentences.push(format!("Reference code: {marker}."));
    for i in 0..filler_sentence_count {
        sentences.push(filler_sentence(seed, canary_id, 1000 + i as u64, 6));
    }
    sentences.join(" ")
}

/// Build the shadow-query probe text: the leading `probe_token_count` tokens
/// of `marker`.
fn shadow_query_from_marker(marker: &str, probe_token_count: usize) -> String {
    marker
        .split_whitespace()
        .take(probe_token_count.max(1))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── word n-gram helpers ────────────────────────────────────────────────────────

/// Lowercased whitespace-separated words of `text`.
fn lower_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_lowercase).collect()
}

/// Fraction of `marker`'s word `n`-grams that appear verbatim (as a
/// case-insensitive substring) somewhere in `response_text`, in `[0.0, 1.0]`.
///
/// When `marker` has fewer than `n` words the whole marker is treated as a
/// single unit (present or absent). An empty marker yields `0.0`.
#[allow(clippy::cast_precision_loss)]
fn ngram_coverage(response_text: &str, marker: &str, n: usize) -> f32 {
    let marker_words = lower_words(marker);
    if marker_words.is_empty() {
        return 0.0;
    }
    let response_lower = response_text.to_lowercase();
    let n = n.max(1);

    if marker_words.len() <= n {
        let phrase = marker_words.join(" ");
        return if response_lower.contains(phrase.as_str()) {
            1.0
        } else {
            0.0
        };
    }

    let ngrams: Vec<String> = marker_words.windows(n).map(|w| w.join(" ")).collect();
    let hits = ngrams
        .iter()
        .filter(|gram| response_lower.contains(gram.as_str()))
        .count();
    hits as f32 / ngrams.len() as f32
}

// ── RagProbe ──────────────────────────────────────────────────────────────────

/// A pluggable probe over an audited RAG system's retrieval/generation
/// pipeline.
///
/// Implementations wrap any retriever/generator in this crate (or an
/// external black-box endpoint) and turn a shadow query into a
/// [`ProbeResult`]. They are **pure sync** — no async, no trait-object
/// lifetime gymnastics — mirroring the `Retriever`/`UncertaintyGenerator`
/// traits used by the `dragin` module. [`MockRagProbe`] is provided for
/// tests.
pub trait RagProbe {
    /// Run `query` against the audited system and report what came back.
    fn probe(&self, query: &str) -> ProbeResult;
}

// ── MockRagProbe ──────────────────────────────────────────────────────────────

/// Deterministic [`RagProbe`] for tests.
///
/// Holds a `corpus` of indexed document texts (the caller decides which
/// canaries' text, if any, to insert). When a query is a substring of an
/// indexed document, the mock reports a "hit"; behavior on a hit is governed
/// by `simulate_leak`:
///
/// * `simulate_leak = true` — echoes the matched document back verbatim as
///   both the sole retrieved passage and the generated response, and (when a
///   `baseline_confidence` is configured) adds `leak_confidence_boost` on
///   top of it — simulating a system that regurgitates indexed content and
///   exposes an elevated similarity/certainty score for it (a leaky system).
/// * `simulate_leak = false` — reports a generic, non-revealing "relevant
///   document found" response with only `baseline_confidence` — simulating a
///   system whose external behavior does not depend on *which* document
///   matched (a non-leaky system).
///
/// A query that matches no indexed document always gets a generic "nothing
/// found" response at `baseline_confidence`, regardless of `simulate_leak`.
/// When `baseline_confidence` is `None`, the mock never exposes a confidence
/// score at all, on a hit or a miss.
#[derive(Debug, Clone, Default)]
pub struct MockRagProbe {
    /// Document texts the mock treats as indexed in the audited corpus.
    pub corpus: Vec<String>,
    /// Whether a corpus hit is echoed back verbatim (leaky) or masked behind
    /// a generic response (non-leaky).
    pub simulate_leak: bool,
    /// Confidence reported when no indexed document matches the query (and
    /// the baseline confidence reported alongside a masked hit); `None`
    /// means this probe never exposes a confidence score.
    pub baseline_confidence: Option<f32>,
    /// Additional confidence added on top of `baseline_confidence` when
    /// `simulate_leak` is `true` and the query matches an indexed document.
    /// Has no effect when `baseline_confidence` is `None`.
    pub leak_confidence_boost: f32,
}

impl MockRagProbe {
    /// Create a mock probe over `corpus` that echoes matched documents back
    /// verbatim (a leaky system).
    #[must_use]
    pub fn leaky(
        corpus: Vec<String>,
        baseline_confidence: f32,
        leak_confidence_boost: f32,
    ) -> Self {
        Self {
            corpus,
            simulate_leak: true,
            baseline_confidence: Some(baseline_confidence),
            leak_confidence_boost,
        }
    }

    /// Create a mock probe over `corpus` that never reveals matched document
    /// content or an elevated confidence (a non-leaky system).
    #[must_use]
    pub fn non_leaky(corpus: Vec<String>, baseline_confidence: Option<f32>) -> Self {
        Self {
            corpus,
            simulate_leak: false,
            baseline_confidence,
            leak_confidence_boost: 0.0,
        }
    }
}

impl RagProbe for MockRagProbe {
    fn probe(&self, query: &str) -> ProbeResult {
        let hit = self.corpus.iter().find(|doc| doc.contains(query));
        match hit {
            Some(doc) if self.simulate_leak => ProbeResult {
                retrieved: vec![doc.clone()],
                generated: Some(doc.clone()),
                confidence: self
                    .baseline_confidence
                    .map(|baseline| baseline + self.leak_confidence_boost),
            },
            Some(_) => ProbeResult {
                retrieved: vec!["[a relevant document was found]".to_string()],
                generated: Some("I found a relevant document.".to_string()),
                confidence: self.baseline_confidence,
            },
            None => ProbeResult {
                retrieved: Vec::new(),
                generated: Some("No relevant information was found.".to_string()),
                confidence: self.baseline_confidence,
            },
        }
    }
}

// ── CanaryAuditor ─────────────────────────────────────────────────────────────

/// Canary-based membership-inference privacy auditor for RAG corpora.
///
/// A [`CanaryAuditor`] generates matched member/non-member canary pairs
/// ([`generate_pairs`](Self::generate_pairs)), runs each canary's shadow
/// query against a pluggable [`RagProbe`]
/// ([`audit_pairs`](Self::audit_pairs)), and reduces the resulting
/// member/non-member signal distributions to a single rank-based AUC-style
/// leakage score ([`compute_auc`](Self::compute_auc)). See the [module
/// docs](super) for the full mechanism.
#[derive(Debug, Clone, Default)]
pub struct CanaryAuditor {
    /// Generation, scoring, and risk-classification configuration.
    config: MembershipInferenceConfig,
}

impl CanaryAuditor {
    /// Create a new auditor with the given configuration.
    #[must_use]
    pub fn new(config: MembershipInferenceConfig) -> Self {
        Self { config }
    }

    /// Access the auditor's configuration.
    #[must_use]
    pub fn config(&self) -> &MembershipInferenceConfig {
        &self.config
    }

    /// Generate `n_pairs` matched member/non-member canary pairs.
    ///
    /// Generation is fully deterministic: calling this method twice on
    /// auditors built from equal configurations produces byte-identical
    /// results. Pair `i` uses canary ids `2*i` (member) and `2*i + 1`
    /// (non-member).
    ///
    /// # Errors
    ///
    /// Returns [`MembershipInferenceError::InvalidConfig`] when `n_pairs` is
    /// `0`, or when the auditor's configuration fails
    /// [`MembershipInferenceConfig::validate`].
    pub fn generate_pairs(
        &self,
        n_pairs: usize,
    ) -> Result<Vec<CanaryPair>, MembershipInferenceError> {
        if n_pairs == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "n_pairs must be at least 1".to_string(),
            ));
        }
        self.config.validate()?;

        let mut pairs = Vec::with_capacity(n_pairs);
        for i in 0..n_pairs {
            #[allow(clippy::cast_possible_truncation)]
            let base_id = (i as u64).wrapping_mul(2);
            let member = self.generate_canary(base_id, CanaryKind::Member);
            let non_member = self.generate_canary(base_id + 1, CanaryKind::NonMember);
            pairs.push(CanaryPair { member, non_member });
        }
        Ok(pairs)
    }

    /// Generate a single canary with the given `id` and `kind`.
    fn generate_canary(&self, id: u64, kind: CanaryKind) -> Canary {
        let marker = generate_marker(self.config.seed, id, self.config.marker_token_count);
        let shadow_query = shadow_query_from_marker(&marker, self.config.probe_token_count);
        let text = generate_text(
            self.config.seed,
            id,
            &marker,
            self.config.filler_sentence_count,
        );
        Canary {
            id,
            marker,
            shadow_query,
            text,
            kind,
        }
    }

    /// Compute the verbatim-overlap / exposed-confidence leakage signal for
    /// a single canary's probe response.
    ///
    /// `signal = overlap_weight * overlap + confidence_weight * confidence`
    /// when `result.confidence` is present; otherwise `signal = overlap`
    /// alone (there is nothing to blend). `overlap` is the fraction of the
    /// canary marker's word n-grams found verbatim in `result`'s combined
    /// text (see [`ngram_coverage`]).
    fn compute_signal(&self, canary: &Canary, result: &ProbeResult) -> f32 {
        let response_text = result.combined_text();
        let overlap = ngram_coverage(&response_text, &canary.marker, self.config.overlap_ngram);
        match result.confidence {
            Some(confidence) => {
                let confidence = confidence.clamp(0.0, 1.0);
                self.config.overlap_weight * overlap + self.config.confidence_weight * confidence
            }
            None => overlap,
        }
    }

    /// Run `probe`, optionally through `defense`, for a single `canary`, and
    /// return its leakage signal.
    fn probe_and_score(
        &self,
        probe: &dyn RagProbe,
        canary: &Canary,
        defense: Option<&MembershipDefense>,
        known_markers: &[&str],
    ) -> f32 {
        let raw = probe.probe(&canary.shadow_query);
        let result = match defense {
            Some(defense) => defense.mitigate(&raw, known_markers),
            None => raw,
        };
        self.compute_signal(canary, &result)
    }

    /// Audit `pairs` against `probe`: run every canary's shadow query,
    /// compute its leakage signal, and reduce the member/non-member signal
    /// distributions to a [`LeakageReport`].
    ///
    /// # Errors
    ///
    /// Returns [`MembershipInferenceError::EmptyCanarySet`] when `pairs` is
    /// empty, and propagates
    /// [`MembershipInferenceError::NonFiniteSignal`] from
    /// [`compute_auc`](Self::compute_auc) if any computed signal is
    /// non-finite (only possible when `probe` returns a non-finite
    /// confidence).
    pub fn audit_pairs(
        &self,
        probe: &dyn RagProbe,
        pairs: &[CanaryPair],
    ) -> Result<LeakageReport, MembershipInferenceError> {
        self.audit_pairs_inner(probe, pairs, None)
    }

    /// Audit `pairs` against `probe` with [`MembershipDefense::mitigate`]
    /// applied to every raw probe response before scoring.
    ///
    /// This is the same pipeline as [`audit_pairs`](Self::audit_pairs) except
    /// each canary's raw [`ProbeResult`] is passed through `defense` first,
    /// so the returned [`LeakageReport`] reflects the *mitigated* leakage
    /// signal — useful for directly comparing an undefended and a defended
    /// AUC on the same probe and canary set.
    ///
    /// # Errors
    ///
    /// Same conditions as [`audit_pairs`](Self::audit_pairs).
    pub fn audit_pairs_with_defense(
        &self,
        probe: &dyn RagProbe,
        pairs: &[CanaryPair],
        defense: &MembershipDefense,
    ) -> Result<LeakageReport, MembershipInferenceError> {
        self.audit_pairs_inner(probe, pairs, Some(defense))
    }

    /// Shared implementation of [`audit_pairs`](Self::audit_pairs) and
    /// [`audit_pairs_with_defense`](Self::audit_pairs_with_defense).
    fn audit_pairs_inner(
        &self,
        probe: &dyn RagProbe,
        pairs: &[CanaryPair],
        defense: Option<&MembershipDefense>,
    ) -> Result<LeakageReport, MembershipInferenceError> {
        if pairs.is_empty() {
            return Err(MembershipInferenceError::EmptyCanarySet);
        }

        let known_markers: Vec<&str> = pairs
            .iter()
            .flat_map(|pair| [pair.member.marker.as_str(), pair.non_member.marker.as_str()])
            .collect();

        let mut pair_signals = Vec::with_capacity(pairs.len());
        let mut member_signals = Vec::with_capacity(pairs.len());
        let mut non_member_signals = Vec::with_capacity(pairs.len());

        for (pair_index, pair) in pairs.iter().enumerate() {
            let member_signal = self.probe_and_score(probe, &pair.member, defense, &known_markers);
            let non_member_signal =
                self.probe_and_score(probe, &pair.non_member, defense, &known_markers);

            member_signals.push(member_signal);
            non_member_signals.push(non_member_signal);
            pair_signals.push(CanaryPairSignal {
                pair_index,
                member_id: pair.member.id,
                non_member_id: pair.non_member.id,
                member_signal,
                non_member_signal,
                member_signal_higher: member_signal > non_member_signal,
            });
        }

        let auc = Self::compute_auc(&member_signals, &non_member_signals)?;
        let risk = RiskLevel::classify(auc, &self.config);

        Ok(LeakageReport {
            pairs: pair_signals,
            auc,
            risk,
        })
    }

    /// Rank-based AUC (area-under-curve) separability statistic between a
    /// `member` leakage-signal group and a `non_member` control group.
    ///
    /// Implements the Mann-Whitney U / Wilcoxon rank-sum formulation:
    ///
    /// ```text
    /// AUC = (R_member - n_member * (n_member + 1) / 2) / (n_member * n_non_member)
    /// ```
    ///
    /// where `R_member` is the sum of the (tie-adjusted, average) ranks of
    /// the member signals within the pooled, ascending-sorted set of *all*
    /// signals. Tied values (within a configurable floating-point tolerance)
    /// share the average of the ranks they jointly occupy, the standard
    /// correction for the Mann-Whitney U statistic under ties.
    ///
    /// Equivalently, the result is the probability that a signal drawn
    /// uniformly at random from the member group exceeds one drawn from the
    /// non-member group, counting an exact tie as one-half a "win". `AUC =
    /// 0.5` means the two groups are statistically indistinguishable — no
    /// detectable leak; `AUC` near `1.0` (or, symmetrically, near `0.0`)
    /// means member canaries systematically produce a different response
    /// signal than non-member canaries — a leak. See [`RiskLevel::classify`]
    /// for turning this score into a risk band.
    ///
    /// # Errors
    ///
    /// Returns [`MembershipInferenceError::EmptyCanarySet`] when either
    /// group is empty, and [`MembershipInferenceError::NonFiniteSignal`] when
    /// any signal is `NaN` or infinite.
    pub fn compute_auc(
        member: &[f32],
        non_member: &[f32],
    ) -> Result<f32, MembershipInferenceError> {
        let n_member = member.len();
        let n_non_member = non_member.len();
        if n_member == 0 || n_non_member == 0 {
            return Err(MembershipInferenceError::EmptyCanarySet);
        }
        if member
            .iter()
            .chain(non_member.iter())
            .any(|value| !value.is_finite())
        {
            return Err(MembershipInferenceError::NonFiniteSignal);
        }

        let mut pooled: Vec<(f64, bool)> = member
            .iter()
            .map(|&value| (f64::from(value), true))
            .chain(non_member.iter().map(|&value| (f64::from(value), false)))
            .collect();
        pooled.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let n = pooled.len();
        let mut ranks = vec![0.0_f64; n];
        let mut i = 0usize;
        while i < n {
            let mut j = i;
            while j + 1 < n && (pooled[j + 1].0 - pooled[i].0).abs() < AUC_TIE_EPSILON {
                j += 1;
            }
            #[allow(clippy::cast_precision_loss)]
            let average_rank = ((i + 1 + j + 1) as f64) / 2.0;
            for rank in ranks.iter_mut().take(j + 1).skip(i) {
                *rank = average_rank;
            }
            i = j + 1;
        }

        let mut rank_sum_member = 0.0_f64;
        for (index, &(_, is_member)) in pooled.iter().enumerate() {
            if is_member {
                rank_sum_member += ranks[index];
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let n_member_f = n_member as f64;
        #[allow(clippy::cast_precision_loss)]
        let n_non_member_f = n_non_member as f64;
        let auc = (rank_sum_member - n_member_f * (n_member_f + 1.0) / 2.0)
            / (n_member_f * n_non_member_f);

        #[allow(clippy::cast_possible_truncation)]
        Ok(auc as f32)
    }
}

// ── MembershipDefense ─────────────────────────────────────────────────────────

/// Response-perturbation mitigation for a detected membership-inference leak
/// risk.
///
/// Applies two independent, composable mitigations to a raw [`ProbeResult`]:
///
/// 1. **Verbatim-span redaction.** Any run of `>= redact_min_ngram` marker
///    words found verbatim (case-insensitively) inside the retrieved or
///    generated text is replaced with `[REDACTED]`. This directly attacks the
///    overlap-based leakage channel: a system can no longer regurgitate
///    enough of a memorized canary to move the verbatim-overlap signal.
/// 2. **Confidence quantization.** Any exposed confidence score is rounded to
///    the nearest multiple of
///    [`MembershipInferenceConfig::confidence_bucket_width`]. Coarsening the
///    resolution of the exposed score collapses small, membership-correlated
///    confidence gaps into shared buckets (ties), which pulls the rank-based
///    AUC toward `0.5` — note this specifically requires *tie-inducing*
///    quantization rather than a linear rescaling, since the AUC statistic
///    is rank-based and therefore invariant to any strictly monotonic
///    transform of the underlying scores.
///
/// The caller is expected to apply [`mitigate`](Self::mitigate) to every raw
/// [`ProbeResult`] before it is scored (or shown to anyone) — see
/// [`CanaryAuditor::audit_pairs_with_defense`].
#[derive(Debug, Clone, Default)]
pub struct MembershipDefense {
    /// Configuration supplying `redact_min_ngram` and
    /// `confidence_bucket_width`.
    config: MembershipInferenceConfig,
}

impl MembershipDefense {
    /// Create a new defense with the given configuration.
    #[must_use]
    pub fn new(config: MembershipInferenceConfig) -> Self {
        Self { config }
    }

    /// Access the defense's configuration.
    #[must_use]
    pub fn config(&self) -> &MembershipInferenceConfig {
        &self.config
    }

    /// Apply verbatim-span redaction and confidence quantization to a raw
    /// probe result.
    ///
    /// `known_markers` should list every canary marker the auditor knows
    /// about (both member and non-member) — the audit tool generated every
    /// canary, so it always has this list available, even though it does not
    /// know in advance which markers will actually leak.
    #[must_use]
    pub fn mitigate(&self, result: &ProbeResult, known_markers: &[&str]) -> ProbeResult {
        let redact_ngram = self.config.redact_min_ngram.max(1);
        let phrases = redaction_phrases(known_markers, redact_ngram);

        let retrieved = result
            .retrieved
            .iter()
            .map(|text| redact_phrases(text, &phrases))
            .collect();
        let generated = result
            .generated
            .as_deref()
            .map(|text| redact_phrases(text, &phrases));
        let confidence = result
            .confidence
            .map(|value| self.quantize_confidence(value));

        ProbeResult {
            retrieved,
            generated,
            confidence,
        }
    }

    /// Round `confidence` to the nearest multiple of
    /// `confidence_bucket_width`.
    fn quantize_confidence(&self, confidence: f32) -> f32 {
        let width = self.config.confidence_bucket_width.max(1e-6);
        (confidence / width).round() * width
    }
}

/// Build the list of case-folded phrases the defense redacts: every
/// `n`-word window of every marker in `markers` (or the whole marker, when
/// it has fewer than `n` words), longest-first so a longer phrase is always
/// redacted before a shorter sub-phrase could partially match inside an
/// already-redacted span.
fn redaction_phrases(markers: &[&str], n: usize) -> Vec<String> {
    let mut phrases: Vec<String> = Vec::new();
    for marker in markers {
        let words = lower_words(marker);
        if words.is_empty() {
            continue;
        }
        if words.len() <= n {
            phrases.push(words.join(" "));
        } else {
            for window in words.windows(n) {
                phrases.push(window.join(" "));
            }
        }
    }
    phrases.sort();
    phrases.dedup();
    phrases.sort_by_key(|phrase| std::cmp::Reverse(phrase.len()));
    phrases
}

/// Redact every case-insensitive occurrence of every phrase in `phrases`
/// from `text`, replacing each match with `[REDACTED]`.
fn redact_phrases(text: &str, phrases: &[String]) -> String {
    let mut out = text.to_string();
    for phrase in phrases {
        out = redact_phrase_ci(&out, phrase);
    }
    out
}

/// Redact every case-insensitive occurrence of `phrase` in `text`.
///
/// Matching is byte-based on the lowercased copy of `text`; if lowercasing
/// changes the byte length (possible for some non-ASCII input) the text is
/// returned unchanged rather than risk slicing on a non-UTF-8 boundary. All
/// text this module itself generates is pure ASCII, so this only affects
/// (and only ever safely no-ops on) unusual externally-supplied probe
/// responses.
fn redact_phrase_ci(text: &str, phrase: &str) -> String {
    if phrase.is_empty() {
        return text.to_string();
    }
    let lower_text = text.to_lowercase();
    if lower_text.len() != text.len() {
        return text.to_string();
    }
    let lower_phrase = phrase.to_lowercase();

    let mut result = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(relative) = lower_text[cursor..].find(lower_phrase.as_str()) {
        let start = cursor + relative;
        let end = start + lower_phrase.len();
        result.push_str(&text[cursor..start]);
        result.push_str("[REDACTED]");
        cursor = end;
    }
    result.push_str(&text[cursor..]);
    result
}
