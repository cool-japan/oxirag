//! Canary-based membership-inference privacy audit for RAG corpora.
//!
//! This is authorized, defensive privacy red-teaming: it lets a RAG system
//! **owner** audit their **own** deployment for a specific privacy leak —
//! *does the system's observable behavior reveal whether a particular
//! document was present in its corpus?* — analogous in spirit to how
//! [`crate::poisoning_defense`] models corpus-integrity attacks and
//! [`crate::prompt_injection_defense`] models instruction-injection attacks,
//! but targeting a different threat entirely: **membership leakage**, not
//! answer steering or instruction hijacking. A leak here means, for example,
//! that querying around a specific person's private record reveals whether
//! that record is in a sensitive corpus, even without the system ever
//! printing the record itself.
//!
//! # Mechanism
//!
//! 1. **Canary injection.** A [`CanaryAuditor`] deterministically generates
//!    matched member/non-member [`Canary`] pairs
//!    ([`CanaryAuditor::generate_pairs`]). Each canary is a synthetic
//!    document carrying a `marker` — a sequence of rare, `FNV-1a`-hash-derived
//!    tokens (e.g. `zc1a2b3c zc9f8e7d`) vanishingly unlikely to occur in
//!    natural text — embedded in otherwise unremarkable filler sentences.
//!    The caller inserts every **member** canary's `text` into the corpus
//!    under audit and inserts **no non-member** canary anywhere — the
//!    non-member group is the control.
//! 2. **Shadow queries.** Each canary carries a `shadow_query`: a partial
//!    prefix of its own marker, modeling an attacker who already knows a
//!    fragment of a target record and is probing to confirm the rest is
//!    present. [`CanaryAuditor::audit_pairs`] sends every canary's shadow
//!    query through a pluggable [`RagProbe`] — implement this trait against
//!    any retriever/generator in this crate, or use [`MockRagProbe`] for
//!    tests — and collects the resulting [`ProbeResult`]s.
//! 3. **Leakage statistic.** Each [`ProbeResult`] is reduced to a scalar
//!    signal blending verbatim word-n-gram overlap between the response and
//!    the canary's marker with any exposed confidence score. The member and
//!    non-member signal distributions are then compared via a genuine
//!    rank-based statistic —
//!    [`CanaryAuditor::compute_auc`] — implementing the Mann-Whitney U /
//!    Wilcoxon rank-sum formulation:
//!
//!    ```text
//!    AUC = (R_member - n_member * (n_member + 1) / 2) / (n_member * n_non_member)
//!    ```
//!
//!    `AUC = 0.5` means member and non-member signals are statistically
//!    indistinguishable (no detectable leak); `AUC` near `1.0` (or,
//!    symmetrically, near `0.0`) means the audited system's behavior
//!    reliably reveals corpus membership.
//! 4. **Defense.** [`MembershipDefense`] mitigates a detected leak by
//!    redacting verbatim marker spans from generated/retrieved text and
//!    quantizing any exposed confidence score, measurably pulling a leaky
//!    probe's AUC back toward `0.5`.
//! 5. **Reporting.** [`CanaryAuditor::audit_pairs`] returns a
//!    [`LeakageReport`]: the per-pair signal breakdown, the overall AUC, and
//!    a [`RiskLevel`] classification.
//!
//! # Quick start
//!
//! ```rust
//! use oxirag::membership_inference::{
//!     CanaryAuditor, MembershipInferenceConfig, MockRagProbe, RiskLevel,
//! };
//!
//! let auditor = CanaryAuditor::new(MembershipInferenceConfig::new());
//! let pairs = auditor.generate_pairs(6).expect("valid config");
//!
//! // Insert only the member canaries' text into the audited corpus.
//! let corpus: Vec<String> = pairs.iter().map(|p| p.member.text.clone()).collect();
//! let probe = MockRagProbe::leaky(corpus, 0.1, 0.6);
//!
//! let report = auditor.audit_pairs(&probe, &pairs).expect("non-empty pairs");
//! assert_eq!(report.risk, RiskLevel::High);
//! assert!(report.auc > 0.9);
//! ```

pub mod auditor;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use auditor::{CanaryAuditor, MembershipDefense, MockRagProbe, RagProbe};
pub use types::{
    Canary, CanaryKind, CanaryPair, CanaryPairSignal, LeakageReport, MembershipInferenceConfig,
    MembershipInferenceError, ProbeResult, RiskLevel,
};
