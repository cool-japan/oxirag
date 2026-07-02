//! Types for the `membership_inference` module.

use thiserror::Error;

// ── CanaryKind ────────────────────────────────────────────────────────────────

/// Whether a [`Canary`] was actually inserted into the corpus under audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanaryKind {
    /// The canary's full text was inserted into the audited corpus.
    ///
    /// A statistically significant response signal for this canary's shadow
    /// query is evidence that the RAG system's behavior reveals corpus
    /// membership.
    Member,
    /// The canary was generated with the same marker style and document
    /// structure as a member canary but was **never** inserted into the
    /// audited corpus.
    ///
    /// Non-member canaries form the control group: any response signal
    /// observed for one reflects background noise in the probe (e.g. the
    /// generator hallucinating plausible-looking content), not corpus
    /// membership.
    NonMember,
}

// ── Canary ────────────────────────────────────────────────────────────────────

/// A synthetic canary document used to audit a RAG deployment for membership
/// leakage.
///
/// Every canary carries a `marker` — a sequence of rare, synthetic tokens
/// vanishingly unlikely to occur in natural text — embedded inside a longer,
/// otherwise unremarkable synthetic document (`text`). The `shadow_query` is
/// a partial-knowledge probe built from a prefix of the marker: it models an
/// attacker who already knows a fragment of the target record and is
/// fishing to confirm the rest is present in the corpus.
///
/// Canaries are produced in matched member/non-member pairs by
/// [`CanaryAuditor::generate_pairs`](super::auditor::CanaryAuditor::generate_pairs);
/// see [`CanaryKind`] for what distinguishes the two groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Canary {
    /// Identifier unique within a single [`CanaryAuditor`](super::auditor::CanaryAuditor)
    /// generation run.
    pub id: u64,
    /// The rare, synthetic token sequence unique to this canary, e.g.
    /// `"zc1a2b3c zc9f8e7d zc00c0ffee"`.
    pub marker: String,
    /// The partial-marker probe query derived from a prefix of `marker`.
    ///
    /// Sent to the audited RAG system's [`RagProbe`](super::auditor::RagProbe)
    /// to test whether the system's response reveals knowledge of the full
    /// canary.
    pub shadow_query: String,
    /// The full synthetic canary document: filler sentences surrounding the
    /// embedded `marker`.
    ///
    /// For a [`CanaryKind::Member`] canary this text is meant to be inserted
    /// into the audited corpus; for [`CanaryKind::NonMember`] it must never
    /// be inserted anywhere the audited system can observe it.
    pub text: String,
    /// Whether this canary was inserted into the audited corpus.
    pub kind: CanaryKind,
}

impl Canary {
    /// `true` when this canary is a [`CanaryKind::Member`] (was inserted into
    /// the audited corpus).
    #[must_use]
    pub fn is_member(&self) -> bool {
        matches!(self.kind, CanaryKind::Member)
    }
}

// ── CanaryPair ────────────────────────────────────────────────────────────────

/// A matched member/non-member canary pair generated together, sharing the
/// same generation parameters but with independently derived markers.
///
/// Pairing member and non-member canaries this way keeps the two groups
/// structurally comparable — same synthetic style, same document length —
/// so any measured signal gap reflects corpus membership rather than a
/// confound in how the canaries were built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanaryPair {
    /// The canary intended to be inserted into the audited corpus.
    pub member: Canary,
    /// The control canary that must never be inserted anywhere.
    pub non_member: Canary,
}

// ── ProbeResult ───────────────────────────────────────────────────────────────

/// The audited RAG system's response to a single shadow query, as reported
/// by a [`RagProbe`](super::auditor::RagProbe) implementation.
///
/// Any subset of the fields may be populated depending on what the audited
/// system exposes: a retrieval-only system might only fill `retrieved`, a
/// black-box chat endpoint might only fill `generated`, and a system that
/// exposes similarity/certainty scores can additionally fill `confidence`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProbeResult {
    /// Retrieved passage texts, in retrieval-rank order.
    pub retrieved: Vec<String>,
    /// The generated (LLM) response text, if the probe exposes generation.
    pub generated: Option<String>,
    /// An optional confidence/certainty score exposed by the audited system
    /// (e.g. a top-passage similarity score or a generation log-probability
    /// proxy), nominally in `[0.0, 1.0]`.
    pub confidence: Option<f32>,
}

impl ProbeResult {
    /// Concatenate `retrieved` and `generated` into a single text blob for
    /// verbatim-overlap scoring.
    ///
    /// Retrieved passages are joined first (in order), followed by the
    /// generated text, each separated by a single space.
    #[must_use]
    pub fn combined_text(&self) -> String {
        let mut parts: Vec<&str> = self.retrieved.iter().map(String::as_str).collect();
        if let Some(generated) = self.generated.as_deref() {
            parts.push(generated);
        }
        parts.join(" ")
    }
}

// ── MembershipInferenceConfig ─────────────────────────────────────────────────

/// Configuration for [`CanaryAuditor`](super::auditor::CanaryAuditor) and
/// [`MembershipDefense`](super::auditor::MembershipDefense).
#[derive(Debug, Clone, PartialEq)]
pub struct MembershipInferenceConfig {
    /// Seed for the deterministic `FNV-1a`-based canary generator.
    ///
    /// Identical `seed` plus identical canary `id`s always reproduce
    /// byte-identical canaries, keeping audits (and their tests)
    /// reproducible. Default: `42`.
    pub seed: u64,
    /// Number of rare tokens making up a canary's `marker`.
    ///
    /// Default: `5`.
    pub marker_token_count: usize,
    /// Number of leading marker tokens exposed in the `shadow_query`.
    ///
    /// Must be `>= 1` and `<= marker_token_count`; models an attacker with
    /// partial knowledge of the target record. Default: `2`.
    pub probe_token_count: usize,
    /// Number of filler sentences generated on either side of the embedded
    /// marker in a canary's `text`.
    ///
    /// Default: `3`.
    pub filler_sentence_count: usize,
    /// Word n-gram window size used when scoring verbatim overlap between a
    /// probe response and a canary's marker.
    ///
    /// Default: `2`.
    pub overlap_ngram: usize,
    /// Weight applied to the verbatim-overlap component of the leakage
    /// signal. See
    /// [`CanaryAuditor::compute_auc`](super::auditor::CanaryAuditor::compute_auc)
    /// for how the signal is used. Default: `0.7`.
    pub overlap_weight: f32,
    /// Weight applied to the exposed-confidence component of the leakage
    /// signal, when a [`ProbeResult`] carries one. Default: `0.3`.
    pub confidence_weight: f32,
    /// Minimum length (in marker words) of a verbatim run that
    /// [`MembershipDefense`](super::auditor::MembershipDefense) treats as a
    /// leak-revealing span subject to redaction.
    ///
    /// Default: `2`.
    pub redact_min_ngram: usize,
    /// Width of the quantization bucket
    /// [`MembershipDefense`](super::auditor::MembershipDefense) rounds any
    /// exposed confidence score to.
    ///
    /// Default: `0.2`.
    pub confidence_bucket_width: f32,
    /// Separability threshold (see [`RiskLevel::classify`]) at or above which
    /// the audit is classified [`RiskLevel::Medium`].
    ///
    /// Default: `0.6`.
    pub low_risk_auc: f32,
    /// Separability threshold (see [`RiskLevel::classify`]) at or above which
    /// the audit is classified [`RiskLevel::High`].
    ///
    /// Default: `0.75`.
    pub high_risk_auc: f32,
}

impl Default for MembershipInferenceConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            marker_token_count: 5,
            probe_token_count: 2,
            filler_sentence_count: 3,
            overlap_ngram: 2,
            overlap_weight: 0.7,
            confidence_weight: 0.3,
            redact_min_ngram: 2,
            confidence_bucket_width: 0.2,
            low_risk_auc: 0.6,
            high_risk_auc: 0.75,
        }
    }
}

impl MembershipInferenceConfig {
    /// Create a new configuration with the default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the generator seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the number of rare tokens per marker.
    #[must_use]
    pub fn with_marker_token_count(mut self, marker_token_count: usize) -> Self {
        self.marker_token_count = marker_token_count;
        self
    }

    /// Set the number of leading marker tokens exposed in the shadow query.
    #[must_use]
    pub fn with_probe_token_count(mut self, probe_token_count: usize) -> Self {
        self.probe_token_count = probe_token_count;
        self
    }

    /// Set the number of filler sentences on either side of the marker.
    #[must_use]
    pub fn with_filler_sentence_count(mut self, filler_sentence_count: usize) -> Self {
        self.filler_sentence_count = filler_sentence_count;
        self
    }

    /// Set the word n-gram window used for the verbatim-overlap signal.
    #[must_use]
    pub fn with_overlap_ngram(mut self, overlap_ngram: usize) -> Self {
        self.overlap_ngram = overlap_ngram;
        self
    }

    /// Set the verbatim-overlap signal weight.
    #[must_use]
    pub fn with_overlap_weight(mut self, overlap_weight: f32) -> Self {
        self.overlap_weight = overlap_weight;
        self
    }

    /// Set the exposed-confidence signal weight.
    #[must_use]
    pub fn with_confidence_weight(mut self, confidence_weight: f32) -> Self {
        self.confidence_weight = confidence_weight;
        self
    }

    /// Set the minimum verbatim-run length (in marker words) the defense
    /// redacts.
    #[must_use]
    pub fn with_redact_min_ngram(mut self, redact_min_ngram: usize) -> Self {
        self.redact_min_ngram = redact_min_ngram;
        self
    }

    /// Set the confidence quantization bucket width.
    #[must_use]
    pub fn with_confidence_bucket_width(mut self, confidence_bucket_width: f32) -> Self {
        self.confidence_bucket_width = confidence_bucket_width;
        self
    }

    /// Set the low-risk separability threshold.
    #[must_use]
    pub fn with_low_risk_auc(mut self, low_risk_auc: f32) -> Self {
        self.low_risk_auc = low_risk_auc;
        self
    }

    /// Set the high-risk separability threshold.
    #[must_use]
    pub fn with_high_risk_auc(mut self, high_risk_auc: f32) -> Self {
        self.high_risk_auc = high_risk_auc;
        self
    }

    /// Validate this configuration, returning a descriptive
    /// [`MembershipInferenceError::InvalidConfig`] on the first violation
    /// found.
    ///
    /// # Errors
    ///
    /// Returns [`MembershipInferenceError::InvalidConfig`] when any count is
    /// zero (where a positive count is required), when `probe_token_count`
    /// exceeds `marker_token_count`, when either weight is negative or
    /// non-finite, when `confidence_bucket_width` is not a positive finite
    /// value, when either risk threshold falls outside `[0.0, 1.0]`, or when
    /// `low_risk_auc > high_risk_auc`.
    pub fn validate(&self) -> Result<(), MembershipInferenceError> {
        if self.marker_token_count == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "marker_token_count must be at least 1".to_string(),
            ));
        }
        if self.probe_token_count == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "probe_token_count must be at least 1".to_string(),
            ));
        }
        if self.probe_token_count > self.marker_token_count {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "probe_token_count ({}) must not exceed marker_token_count ({})",
                self.probe_token_count, self.marker_token_count
            )));
        }
        if self.filler_sentence_count == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "filler_sentence_count must be at least 1".to_string(),
            ));
        }
        if self.overlap_ngram == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "overlap_ngram must be at least 1".to_string(),
            ));
        }
        if self.redact_min_ngram == 0 {
            return Err(MembershipInferenceError::InvalidConfig(
                "redact_min_ngram must be at least 1".to_string(),
            ));
        }
        if !self.overlap_weight.is_finite() || self.overlap_weight < 0.0 {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "overlap_weight must be a non-negative finite value, got {}",
                self.overlap_weight
            )));
        }
        if !self.confidence_weight.is_finite() || self.confidence_weight < 0.0 {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "confidence_weight must be a non-negative finite value, got {}",
                self.confidence_weight
            )));
        }
        if !self.confidence_bucket_width.is_finite() || self.confidence_bucket_width <= 0.0 {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "confidence_bucket_width must be a positive finite value, got {}",
                self.confidence_bucket_width
            )));
        }
        if !(0.0..=1.0).contains(&self.low_risk_auc) {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "low_risk_auc must be in [0.0, 1.0], got {}",
                self.low_risk_auc
            )));
        }
        if !(0.0..=1.0).contains(&self.high_risk_auc) {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "high_risk_auc must be in [0.0, 1.0], got {}",
                self.high_risk_auc
            )));
        }
        if self.low_risk_auc > self.high_risk_auc {
            return Err(MembershipInferenceError::InvalidConfig(format!(
                "low_risk_auc ({}) must not exceed high_risk_auc ({})",
                self.low_risk_auc, self.high_risk_auc
            )));
        }
        Ok(())
    }
}

// ── RiskLevel ─────────────────────────────────────────────────────────────────

/// Risk classification band for a [`LeakageReport`], derived from the
/// audit's AUC-style separability score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Member and non-member canary signals are statistically indistinguishable
    /// (separability below [`MembershipInferenceConfig::low_risk_auc`]).
    Low,
    /// A moderate, likely-real separability was measured (between
    /// [`MembershipInferenceConfig::low_risk_auc`] and
    /// [`MembershipInferenceConfig::high_risk_auc`]).
    Medium,
    /// A strong separability was measured (at or above
    /// [`MembershipInferenceConfig::high_risk_auc`]) — the audited system's
    /// responses reliably reveal corpus membership.
    High,
}

impl RiskLevel {
    /// Classify an AUC-style leakage score into a [`RiskLevel`] band.
    ///
    /// `auc` is treated **symmetrically** around `0.5`: an AUC of `0.1` is
    /// just as informative to an attacker as an AUC of `0.9` (the attacker
    /// simply inverts their decision rule), so classification is driven by
    /// `separability = max(auc, 1 - auc)`, which always lies in `[0.5, 1.0]`.
    /// `separability` is compared against
    /// [`MembershipInferenceConfig::low_risk_auc`] and
    /// [`MembershipInferenceConfig::high_risk_auc`].
    #[must_use]
    pub fn classify(auc: f32, config: &MembershipInferenceConfig) -> Self {
        let separability = auc.max(1.0 - auc);
        if separability >= config.high_risk_auc {
            Self::High
        } else if separability >= config.low_risk_auc {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

// ── CanaryPairSignal ──────────────────────────────────────────────────────────

/// Per-pair leakage signal, one entry per audited [`CanaryPair`].
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryPairSignal {
    /// Index of the pair within the audited slice.
    pub pair_index: usize,
    /// [`Canary::id`] of the member canary in this pair.
    pub member_id: u64,
    /// [`Canary::id`] of the non-member canary in this pair.
    pub non_member_id: u64,
    /// Leakage signal computed from the member canary's shadow-query
    /// response.
    pub member_signal: f32,
    /// Leakage signal computed from the non-member canary's shadow-query
    /// response.
    pub non_member_signal: f32,
    /// `true` when `member_signal > non_member_signal` for this specific
    /// pair.
    ///
    /// This is an informal, single-pair indicator; the formal leakage
    /// statistic is [`LeakageReport::auc`], computed across *all* audited
    /// pairs.
    pub member_signal_higher: bool,
}

// ── LeakageReport ─────────────────────────────────────────────────────────────

/// Summary of a full canary-based membership-inference audit, produced by
/// [`CanaryAuditor::audit_pairs`](super::auditor::CanaryAuditor::audit_pairs).
#[derive(Debug, Clone, PartialEq)]
pub struct LeakageReport {
    /// Per-pair signal breakdown, in the same order as the audited slice.
    pub pairs: Vec<CanaryPairSignal>,
    /// The overall AUC-style leakage score across all pairs; see
    /// [`CanaryAuditor::compute_auc`](super::auditor::CanaryAuditor::compute_auc).
    pub auc: f32,
    /// Risk classification derived from `auc` via [`RiskLevel::classify`].
    pub risk: RiskLevel,
}

impl LeakageReport {
    /// Number of canary pairs this report was computed over.
    #[must_use]
    pub fn n_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// `true` when the risk classification is anything other than
    /// [`RiskLevel::Low`].
    #[must_use]
    pub fn is_leak_detected(&self) -> bool {
        !matches!(self.risk, RiskLevel::Low)
    }
}

// ── MembershipInferenceError ──────────────────────────────────────────────────

/// Errors produced by the `membership_inference` module.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MembershipInferenceError {
    /// No canary pairs (or an empty signal group) were supplied where at
    /// least one member and one non-member value are required.
    #[error("no canary pairs supplied; at least one member/non-member pair is required")]
    EmptyCanarySet,
    /// The configuration (or a directly-supplied argument standing in for
    /// one) is invalid, with a human-readable explanation.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A computed leakage signal was `NaN` or infinite, typically because a
    /// [`RagProbe`](super::auditor::RagProbe) implementation returned a
    /// non-finite confidence score.
    #[error("encountered a non-finite (NaN or infinite) leakage signal")]
    NonFiniteSignal,
}
