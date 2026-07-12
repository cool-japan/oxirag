//! Tests for `knowledge_unlearning`.
//!
//! Fixture texts and thresholds below were calibrated empirically against
//! the actual hand-rolled `MinHash`/`Jaccard` implementation in
//! [`super::dedup`] (not hand-waved) — every asserted similarity/leakage
//! number reflects a real measurement, not an assumption.

use crate::types::Document;

use super::audit::UnlearningAuditor;
use super::dedup::UnlearningNearDuplicateDetector;
use super::engine::{UnlearnableStore, UnlearningEngine, UnlearningMemoryStore};
use super::scope::UnlearningArtifactRegistry;
use super::types::{
    UnlearningArtifactKind, UnlearningConfig, UnlearningError, UnlearningLeakageVerdict,
    UnlearningRequest, UnlearningScopeReason, UnlearningStatus, UnlearningTarget,
};

// ── shared fixtures ───────────────────────────────────────────────────────────

const FACT_A: &str = "The Meridian propulsion module uses a titanium alloy frame and draws \
                       4200 watts under peak load according to the internal engineering review.";
/// A paraphrase of [`FACT_A`] under a different document id, never named in
/// any request — the "hidden carrier" the headline test is about.
const PARAPHRASE_B: &str = "According to the internal engineering review, the Meridian \
                             propulsion module's frame is titanium alloy and it draws 4200 \
                             watts under peak load.";
/// Unrelated content that must never be swept up as a false positive.
const UNRELATED_C: &str = "The quarterly marketing budget increased by twelve percent \
                            following the launch of the new campaign in Southeast Asia.";

fn doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(id)
}

// ── headline test: delete-only is not enough ────────────────────────────────

#[test]
fn baseline_delete_only_leaves_the_fact_retrievable() {
    // A naive baseline: delete the named document and nothing else, then
    // audit what remains for traces of the deleted fact.
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("a", FACT_A));
    store.insert(doc("b", PARAPHRASE_B));
    store.insert(doc("c", UNRELATED_C));

    let a_id = crate::types::DocumentId::from_string("a");
    assert!(store.delete(&a_id), "doc a must have existed to delete");

    let auditor = UnlearningAuditor::new(UnlearningConfig::default());
    let verdict = auditor.audit(&store, FACT_A);

    match verdict {
        UnlearningLeakageVerdict::Clean => {
            panic!(
                "baseline delete-only must NOT be reported clean: doc b still paraphrases the deleted fact"
            )
        }
        UnlearningLeakageVerdict::ResidualLeakage {
            surviving_doc_ids, ..
        } => {
            assert_eq!(
                surviving_doc_ids,
                vec![crate::types::DocumentId::from_string("b")],
                "the audit must name doc b as the surviving carrier"
            );
        }
    }
}

#[test]
fn full_engine_catches_the_paraphrase_and_converges_clean() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("a", FACT_A));
    store.insert(doc("b", PARAPHRASE_B));
    store.insert(doc("c", UNRELATED_C));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("default config is valid");

    let a_id = crate::types::DocumentId::from_string("a");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            a_id.clone(),
        )))
        .expect("doc a exists");

    // Scope resolution must have found the paraphrase on its own, without it
    // ever being named in the request.
    let b_id = crate::types::DocumentId::from_string("b");
    assert!(
        certificate.scope.contains(&b_id),
        "scope resolution must catch the paraphrase as a near-duplicate carrier"
    );
    assert!(matches!(
        certificate
            .scope
            .items
            .iter()
            .find(|item| item.document_id == b_id)
            .map(|item| &item.reason),
        Some(UnlearningScopeReason::NearDuplicate { .. })
    ));

    // Both must actually be gone from the store.
    assert!(engine.store().get(&a_id).is_none());
    assert!(engine.store().get(&b_id).is_none());
    // The unrelated document must survive untouched.
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("c"))
            .is_some()
    );

    // And the re-audit must confirm the fact is genuinely unreachable now.
    assert!(
        certificate.is_clean(),
        "status was {:?}",
        certificate.status
    );
    assert_eq!(certificate.status, UnlearningStatus::Clean);
    assert!(
        certificate
            .audit_rounds
            .iter()
            .any(|round| round.verdict.is_clean())
    );
}

// ── derived artifacts cascade ────────────────────────────────────────────────

#[test]
fn baseline_delete_only_leaves_the_cached_summary_behind() {
    let source = doc("src", "Employee Jane Doe's home address is 42 Willow Lane.");
    let summary = doc(
        "summary",
        "Internal summary #445 generated for record retention purposes.",
    );
    let source_id = source.id.clone();
    let summary_id = summary.id.clone();

    let mut store = UnlearningMemoryStore::new();
    store.insert(source);
    store.insert(summary);
    store.delete(&source_id);

    assert!(
        store.get(&summary_id).is_some(),
        "a bare delete() of the source must leave its derived summary behind"
    );
}

#[test]
fn full_engine_cascades_deletion_to_derived_artifacts() {
    let source = doc(
        "src2",
        "Employee Jane Doe's home address is 42 Willow Lane.",
    );
    let summary = doc(
        "summary2",
        "Internal summary #445 generated for record retention purposes.",
    );
    let source_id = source.id.clone();
    let summary_id = summary.id.clone();

    let mut store = UnlearningMemoryStore::new();
    store.insert(source);
    store.insert(summary);

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("default config is valid");
    engine.registry_mut().register(
        source_id.clone(),
        summary_id.clone(),
        UnlearningArtifactKind::Summary,
    );

    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            source_id.clone(),
        )))
        .expect("source exists");

    assert!(certificate.is_clean());
    assert!(engine.store().get(&source_id).is_none());
    assert!(
        engine.store().get(&summary_id).is_none(),
        "the cached summary must be cascaded and deleted too"
    );
    assert!(matches!(
        certificate
            .scope
            .items
            .iter()
            .find(|item| item.document_id == summary_id)
            .map(|item| &item.reason),
        Some(UnlearningScopeReason::DerivedArtifact {
            kind: UnlearningArtifactKind::Summary,
            ..
        })
    ));
}

// ── the certificate cannot lie ───────────────────────────────────────────────

#[test]
fn certificate_honestly_reports_a_run_that_did_not_converge() {
    // `carrier` shares one distinctive verbatim phrase with `secret` (enough
    // to trip the leakage threshold) but is otherwise long and unrelated
    // (its overall shingle similarity stays below the auto-delete
    // similarity threshold) -- exactly the brief's "leakage below the
    // deletion threshold but above the leakage threshold" scenario. The
    // engine must refuse to auto-delete unrelated-looking content and must
    // not pretend the run converged.
    let secret = "Case reference ZX-88214 was resolved on July 3rd with a settlement of \
                  $12,450 paid to the claimant.";
    let carrier = "In an unrelated compliance memo, staff reviewed several older precedents \
                   spanning training schedules, facility budget allocations, and other \
                   operational notes with no connection to this matter, before noting that a \
                   prior dispute in a different department closed with a settlement of \
                   $12,450 paid to the claimant as documented in the appendix, followed by \
                   further commentary on scheduling and administrative logistics for next \
                   quarter and other routine business unrelated to the case.";

    let secret_doc = doc("secret", secret);
    let carrier_doc = doc("carrier", carrier);
    let secret_id = secret_doc.id.clone();
    let carrier_id = carrier_doc.id.clone();

    let mut store = UnlearningMemoryStore::new();
    store.insert(secret_doc);
    store.insert(carrier_doc);

    let config = UnlearningConfig::default();
    // Sanity check the calibration this test depends on before trusting the
    // engine's behavior on top of it.
    let detector = UnlearningNearDuplicateDetector::new(config.shingle_size, config.num_perm);
    let raw_similarity = detector.similarity(secret, carrier);
    assert!(
        raw_similarity < config.similarity_threshold,
        "fixture must stay below the auto-delete bar ({raw_similarity} vs {})",
        config.similarity_threshold
    );

    let mut engine = UnlearningEngine::new(config, store).expect("default config is valid");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            secret_id.clone(),
        )))
        .expect("secret exists");

    // The one non-negotiable assertion of this whole module: a run that
    // could not verify a clean result must never claim `Clean`.
    assert_eq!(certificate.status, UnlearningStatus::NotConverged);
    assert!(!certificate.is_clean());
    assert!(!certificate.status.is_clean());

    // The evidence must actually be present and honest, not swallowed.
    assert_eq!(certificate.audit_rounds.len(), 1);
    match &certificate.audit_rounds[0].verdict {
        UnlearningLeakageVerdict::ResidualLeakage {
            surviving_doc_ids, ..
        } => assert_eq!(surviving_doc_ids, &vec![carrier_id.clone()]),
        UnlearningLeakageVerdict::Clean => panic!("this run must not be Clean"),
    }
    assert!(
        certificate.audit_rounds[0]
            .escalated_document_ids
            .is_empty()
    );

    // The engine must not have destroyed data it wasn't confident about.
    assert!(engine.store().get(&carrier_id).is_some());
    assert_eq!(certificate.deleted_document_ids, vec![secret_id]);
}

#[test]
fn multi_round_escalation_converges_when_carriers_clear_the_bar() {
    // `first_carrier` is similar enough to `secret` to be caught directly by
    // scope resolution. `second_carrier` is only similar to `first_carrier`
    // (not to `secret` alone) and is discovered one round later, once
    // `first_carrier`'s own content is folded into what the audit probes
    // with -- proving the delete -> re-audit -> escalate loop does more than
    // a single static pass.
    let secret = "Zenith turbine specs remain confidential per engineering policy documents.";
    let first_carrier = "Zenith turbine specs remain confidential per engineering policy \
                          documents. Case file QR-77 is pending review next month.";
    let second_carrier = "Reports show the case file QR-77 is pending review next month \
                           within the department archive.";
    let unrelated = "Unrelated weekly cafeteria menu planning notes for the north campus \
                      building.";

    let secret_doc = doc("secret2", secret);
    let first_doc = doc("first", first_carrier);
    let second_doc = doc("second", second_carrier);
    let unrelated_doc = doc("unrelated", unrelated);
    let secret_id = secret_doc.id.clone();
    let first_id = first_doc.id.clone();
    let second_id = second_doc.id.clone();
    let unrelated_id = unrelated_doc.id.clone();

    let mut store = UnlearningMemoryStore::new();
    store.insert(secret_doc);
    store.insert(first_doc);
    store.insert(second_doc);
    store.insert(unrelated_doc);

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("default config is valid");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            secret_id.clone(),
        )))
        .expect("secret exists");

    assert_eq!(certificate.status, UnlearningStatus::Clean);
    assert!(
        certificate.audit_rounds.len() >= 2,
        "must take more than one round"
    );
    assert!(engine.store().get(&secret_id).is_none());
    assert!(engine.store().get(&first_id).is_none());
    assert!(
        engine.store().get(&second_id).is_none(),
        "the second-hop carrier must be found by escalation, not just the initial scan"
    );
    assert!(
        engine.store().get(&unrelated_id).is_some(),
        "unrelated content must never be swept up"
    );
    assert!(
        certificate
            .deleted_document_ids
            .iter()
            .any(|id| id == &second_id)
    );
}

// ── near-duplicate detection: exact / paraphrase / unrelated ────────────────

#[test]
fn near_duplicate_detector_catches_exact_copies() {
    let detector = UnlearningNearDuplicateDetector::default();
    assert!((detector.similarity(FACT_A, FACT_A) - 1.0).abs() < f64::EPSILON);
    assert!(detector.is_near_duplicate(FACT_A, FACT_A, 0.99));
}

#[test]
fn near_duplicate_detector_catches_paraphrases() {
    let config = UnlearningConfig::default();
    let detector = UnlearningNearDuplicateDetector::new(config.shingle_size, config.num_perm);
    let similarity = detector.similarity(FACT_A, PARAPHRASE_B);
    assert!(
        similarity >= config.similarity_threshold,
        "paraphrase similarity {similarity} must clear the configured threshold {}",
        config.similarity_threshold
    );
}

#[test]
fn near_duplicate_detector_does_not_false_positive_on_unrelated_text() {
    let config = UnlearningConfig::default();
    let detector = UnlearningNearDuplicateDetector::new(config.shingle_size, config.num_perm);
    let similarity = detector.similarity(FACT_A, UNRELATED_C);
    assert!(
        similarity < config.similarity_threshold,
        "unrelated documents must not be flagged as near-duplicates (got {similarity})"
    );
    assert!(!detector.is_near_duplicate(FACT_A, UNRELATED_C, config.similarity_threshold));
}

#[test]
fn scope_resolution_does_not_sweep_up_unrelated_documents() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("a", FACT_A));
    store.insert(doc("b", PARAPHRASE_B));
    store.insert(doc("c", UNRELATED_C));

    let engine = UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let scope = engine.resolve_scope(&UnlearningTarget::DocumentId(
        crate::types::DocumentId::from_string("a"),
    ));

    assert_eq!(
        scope.len(),
        2,
        "only a and its paraphrase b belong in scope"
    );
    assert!(!scope.contains(&crate::types::DocumentId::from_string("c")));
}

// ── idempotence ───────────────────────────────────────────────────────────────

#[test]
fn unlearning_an_already_unlearned_document_is_a_clean_no_op() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc(
        "solo",
        "A single, unremarkable fact with no duplicates anywhere.",
    ));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let target = UnlearningTarget::DocumentId(crate::types::DocumentId::from_string("solo"));

    let first = engine
        .unlearn(UnlearningRequest::new(target.clone()))
        .expect("document exists");
    assert_eq!(first.status, UnlearningStatus::Clean);

    let second = engine
        .unlearn(UnlearningRequest::new(target))
        .expect("idempotent re-request must not error");
    assert_eq!(second.status, UnlearningStatus::AlreadyUnlearned);
    assert!(second.scope.is_empty());
    assert!(second.deleted_document_ids.is_empty());
    assert!(second.audit_rounds.is_empty());
    assert!(second.is_clean());
}

// ── edge cases ────────────────────────────────────────────────────────────────

#[test]
fn target_not_found_is_an_honest_error() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("real", "Something that exists."));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let result = engine.unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
        crate::types::DocumentId::from_string("never-existed"),
    )));

    assert!(matches!(result, Err(UnlearningError::TargetNotFound(_))));
}

#[test]
fn empty_store_reports_target_not_found() {
    let store = UnlearningMemoryStore::new();
    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    assert!(engine.store().is_empty());

    let result = engine.unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
        crate::types::DocumentId::from_string("anything"),
    )));
    assert!(matches!(result, Err(UnlearningError::TargetNotFound(_))));
}

#[test]
fn deleting_the_only_document_converges_clean() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("only", "The last document standing in this store."));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            crate::types::DocumentId::from_string("only"),
        )))
        .expect("document exists");

    assert_eq!(certificate.status, UnlearningStatus::Clean);
    assert!(engine.store().is_empty());
}

#[test]
fn leakage_threshold_of_zero_tolerates_nothing() {
    // `e` and `f` share only the faintest trace of overlap: similarity is
    // positive but small enough to sit under the default leakage
    // threshold, so the default configuration reports Clean. A
    // zero-tolerance configuration must not.
    let e = "Quarterly infrastructure review notes covering server capacity, network \
             redundancy, and backup rotation schedules for the west data center.";
    let f = "This is a quarterly infrastructure summary focused only on marketing budget \
             allocation across regional teams for the spring campaign push.";

    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("f", f));

    let default_config = UnlearningConfig::default();
    let lenient = UnlearningAuditor::new(default_config.clone());
    assert!(
        lenient.audit(&store, e).is_clean(),
        "default threshold must tolerate this faint a signal"
    );

    let strict = UnlearningAuditor::new(default_config.with_leakage_threshold(0.0));
    let verdict = strict.audit(&store, e);
    assert!(
        !verdict.is_clean(),
        "a leakage_threshold of 0.0 must flag any detected signal, however small"
    );
}

// ── target variants: SourceId / ContentPattern ───────────────────────────────

#[test]
fn source_id_target_resolves_every_document_from_that_source() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(
        Document::new("Chunk one of the ingested PDF.")
            .with_id("chunk-1")
            .with_source("report.pdf"),
    );
    store.insert(
        Document::new("Chunk two of the ingested PDF.")
            .with_id("chunk-2")
            .with_source("report.pdf"),
    );
    store.insert(
        Document::new("An entirely different document.")
            .with_id("other")
            .with_source("other.pdf"),
    );

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::SourceId(
            "report.pdf".to_string(),
        )))
        .expect("source exists");

    assert!(certificate.is_clean());
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("chunk-1"))
            .is_none()
    );
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("chunk-2"))
            .is_none()
    );
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("other"))
            .is_some()
    );
}

#[test]
fn content_pattern_target_resolves_every_matching_document() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("p1", "The secret launch code is ALPHA-NINE-NINE."));
    store.insert(doc(
        "p2",
        "Reminder: the secret launch code is ALPHA-NINE-NINE, guard it.",
    ));
    store.insert(doc("p3", "Completely unrelated notes about lunch."));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let certificate = engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::ContentPattern(
            "alpha-nine-nine".to_string(),
        )))
        .expect("pattern matches");

    assert!(certificate.is_clean());
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("p1"))
            .is_none()
    );
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("p2"))
            .is_none()
    );
    assert!(
        engine
            .store()
            .get(&crate::types::DocumentId::from_string("p3"))
            .is_some()
    );
}

// ── config validation ────────────────────────────────────────────────────────

#[test]
fn config_rejects_zero_max_audit_rounds() {
    let config = UnlearningConfig::default().with_max_audit_rounds(0);
    assert!(matches!(
        config.validate(),
        Err(UnlearningError::InvalidConfig(_))
    ));
}

#[test]
fn config_rejects_out_of_range_thresholds() {
    assert!(
        UnlearningConfig::default()
            .with_similarity_threshold(1.5)
            .validate()
            .is_err()
    );
    assert!(
        UnlearningConfig::default()
            .with_leakage_threshold(-0.1)
            .validate()
            .is_err()
    );
}

#[test]
fn config_rejects_negative_weights() {
    assert!(
        UnlearningConfig::default()
            .with_verbatim_weight(-1.0)
            .validate()
            .is_err()
    );
    assert!(
        UnlearningConfig::default()
            .with_similarity_weight(f64::NAN)
            .validate()
            .is_err()
    );
}

#[test]
fn engine_construction_fails_fast_on_invalid_config() {
    let store = UnlearningMemoryStore::new();
    let result = UnlearningEngine::new(UnlearningConfig::default().with_max_audit_rounds(0), store);
    assert!(matches!(result, Err(UnlearningError::InvalidConfig(_))));
}

// ── UnlearningArtifactRegistry ───────────────────────────────────────────────

#[test]
fn artifact_registry_tracks_links() {
    let mut registry = UnlearningArtifactRegistry::new();
    assert!(registry.is_empty());

    let source = crate::types::DocumentId::from_string("src");
    let artifact_one = crate::types::DocumentId::from_string("art-1");
    let artifact_two = crate::types::DocumentId::from_string("art-2");
    registry.register(
        source.clone(),
        artifact_one.clone(),
        UnlearningArtifactKind::Summary,
    );
    registry.register(
        source.clone(),
        artifact_two.clone(),
        UnlearningArtifactKind::Other("triple".to_string()),
    );

    assert_eq!(registry.len(), 2);
    let derived = registry.derived_from(&source);
    assert_eq!(derived.len(), 2);
    assert!(derived.iter().any(|link| link.artifact_id == artifact_one));
    assert!(derived.iter().any(|link| link.artifact_id == artifact_two));
    assert!(
        registry
            .derived_from(&crate::types::DocumentId::from_string("nobody"))
            .is_empty()
    );
}

// ── UnlearningCertificate content hash ───────────────────────────────────────

#[test]
fn certificate_content_hash_is_deterministic_across_independent_runs() {
    let build_certificate = || {
        let mut store = UnlearningMemoryStore::new();
        store.insert(doc("a", FACT_A));
        store.insert(doc("b", PARAPHRASE_B));
        store.insert(doc("c", UNRELATED_C));
        let mut engine =
            UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
        engine
            .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
                crate::types::DocumentId::from_string("a"),
            )))
            .expect("doc a exists")
    };

    let first = build_certificate();
    let second = build_certificate();

    // Two independently-run certificates over identical inputs must hash
    // identically, since the hash deliberately excludes the wall-clock
    // `issued_at` field.
    assert_eq!(first.content_hash, second.content_hash);
}

#[test]
fn certificate_content_hash_changes_with_status() {
    let mut clean_store = UnlearningMemoryStore::new();
    clean_store.insert(doc("solo-x", "Nothing else references this."));
    let mut clean_engine =
        UnlearningEngine::new(UnlearningConfig::default(), clean_store).expect("valid config");
    let clean_certificate = clean_engine
        .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
            crate::types::DocumentId::from_string("solo-x"),
        )))
        .expect("doc exists");

    let mut leaky_store = UnlearningMemoryStore::new();
    leaky_store.insert(doc("solo-y", "Nothing else references this."));
    let mut leaky_engine =
        UnlearningEngine::new(UnlearningConfig::default(), leaky_store).expect("valid config");
    let idempotent_certificate = {
        leaky_engine
            .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
                crate::types::DocumentId::from_string("solo-y"),
            )))
            .expect("doc exists");
        leaky_engine
            .unlearn(UnlearningRequest::new(UnlearningTarget::DocumentId(
                crate::types::DocumentId::from_string("solo-y"),
            )))
            .expect("idempotent")
    };

    assert_ne!(
        clean_certificate.content_hash,
        idempotent_certificate.content_hash
    );
}

// ── UnlearningScope helpers ──────────────────────────────────────────────────

#[test]
fn scope_helpers_report_consistent_state() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("a", FACT_A));
    store.insert(doc("b", PARAPHRASE_B));

    let engine = UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let scope = engine.resolve_scope(&UnlearningTarget::DocumentId(
        crate::types::DocumentId::from_string("a"),
    ));

    assert!(!scope.is_empty());
    assert_eq!(scope.len(), scope.document_ids().len());
    for id in scope.document_ids() {
        assert!(scope.contains(&id));
    }
}

#[test]
fn history_status_is_recorded_after_a_run() {
    let mut store = UnlearningMemoryStore::new();
    store.insert(doc("track-me", "Track this document's history."));

    let mut engine =
        UnlearningEngine::new(UnlearningConfig::default(), store).expect("valid config");
    let target = UnlearningTarget::DocumentId(crate::types::DocumentId::from_string("track-me"));
    assert!(engine.history_status(&target).is_none());

    engine
        .unlearn(UnlearningRequest::new(target.clone()))
        .expect("doc exists");
    assert_eq!(
        engine.history_status(&target),
        Some(UnlearningStatus::Clean)
    );
}
