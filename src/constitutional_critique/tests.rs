//! Unit tests for the `constitutional_critique` module.

use crate::constitutional_critique::engine::ConstitutionalEngine;
use crate::constitutional_critique::types::{
    ConstitutionalConfig, ConstitutionalCritique, ConstitutionalError, ConstitutionalMatch,
    ConstitutionalPrinciple, ConstitutionalReviser, ConstitutionalTrigger,
    MockConstitutionalReviser,
};

// ── helpers ───────────────────────────────────────────────────────────────

fn engine_default() -> ConstitutionalEngine {
    ConstitutionalEngine::new(ConstitutionalConfig::default())
}

fn mock() -> MockConstitutionalReviser {
    MockConstitutionalReviser::new()
}

/// Assert that `principle` is violated by `violating_draft`, and that
/// revising it removes the violation (re-critiquing the revision against
/// the same principle no longer flags it).
fn assert_revision_clears_violation(principle: &ConstitutionalPrinciple, violating_draft: &str) {
    let reviser = mock();
    let critique = reviser.critique(violating_draft, principle).unwrap();
    assert!(
        critique.violated,
        "expected '{}' to violate principle '{}'",
        violating_draft, principle.id
    );

    let revision = reviser.revise(violating_draft, &critique).unwrap();
    let recheck = reviser.critique(&revision.after, principle).unwrap();
    assert!(
        !recheck.violated,
        "expected revision '{}' to clear the violation of principle '{}'",
        revision.after, principle.id
    );
}

// ── ConstitutionalTrigger ────────────────────────────────────────────────

#[test]
fn trigger_keyword_constructor_sets_word() {
    let trigger = ConstitutionalTrigger::keyword("kill");
    assert!(matches!(trigger, ConstitutionalTrigger::Keyword { word } if word == "kill"));
}

#[test]
fn trigger_absolute_constructor_sets_phrase_and_hedge() {
    let trigger = ConstitutionalTrigger::absolute("definitely", "likely");
    assert!(matches!(
        trigger,
        ConstitutionalTrigger::Absolute { phrase, hedge } if phrase == "definitely" && hedge == "likely"
    ));
}

#[test]
fn trigger_pii_shaped_constructor() {
    assert_eq!(
        ConstitutionalTrigger::pii_shaped(),
        ConstitutionalTrigger::PiiShaped
    );
}

#[test]
fn trigger_clone_equals_original() {
    let trigger = ConstitutionalTrigger::keyword("x");
    assert_eq!(trigger.clone(), trigger);
}

// ── ConstitutionalPrinciple ──────────────────────────────────────────────

#[test]
fn principle_new_sets_id_and_statement() {
    let principle = ConstitutionalPrinciple::new("my_id", "my statement");
    assert_eq!(principle.id, "my_id");
    assert_eq!(principle.statement, "my statement");
}

#[test]
fn principle_new_has_no_triggers() {
    assert!(
        ConstitutionalPrinciple::new("id", "stmt")
            .triggers
            .is_empty()
    );
}

#[test]
fn principle_with_trigger_appends_one() {
    let principle = ConstitutionalPrinciple::new("id", "stmt")
        .with_trigger(ConstitutionalTrigger::keyword("x"));
    assert_eq!(principle.triggers.len(), 1);
}

#[test]
fn principle_with_triggers_appends_many() {
    let principle = ConstitutionalPrinciple::new("id", "stmt").with_triggers(vec![
        ConstitutionalTrigger::keyword("x"),
        ConstitutionalTrigger::keyword("y"),
        ConstitutionalTrigger::pii_shaped(),
    ]);
    assert_eq!(principle.triggers.len(), 3);
}

#[test]
fn principle_clone_equals_original() {
    let principle = ConstitutionalPrinciple::avoid_harm();
    assert_eq!(principle.clone(), principle);
}

#[test]
fn default_constitution_has_five_principles() {
    assert_eq!(ConstitutionalPrinciple::default_constitution().len(), 5);
}

#[test]
fn default_constitution_ids_in_order() {
    let ids: Vec<String> = ConstitutionalPrinciple::default_constitution()
        .iter()
        .map(|p| p.id.clone())
        .collect();
    assert_eq!(
        ids,
        vec![
            "avoid_harm",
            "be_truthful",
            "respect_privacy",
            "be_respectful",
            "avoid_illegal_activity",
        ]
    );
}

#[test]
fn default_constitution_every_principle_has_triggers() {
    for principle in ConstitutionalPrinciple::default_constitution() {
        assert!(
            !principle.triggers.is_empty(),
            "principle '{}' has no triggers",
            principle.id
        );
    }
}

// ── default principles: fire / do-not-fire pairs ────────────────────────

#[test]
fn avoid_harm_fires_on_violating_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_harm();
    let critique = reviser
        .critique(
            "The villain wants to poison the town's water supply.",
            &principle,
        )
        .unwrap();
    assert!(critique.violated);
}

#[test]
fn avoid_harm_does_not_fire_on_clean_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_harm();
    let critique = reviser
        .critique(
            "The team wants to protect the town's water supply.",
            &principle,
        )
        .unwrap();
    assert!(!critique.violated);
}

#[test]
fn be_truthful_fires_on_violating_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::be_truthful();
    let critique = reviser
        .critique(
            "This method is guaranteed to solve every problem.",
            &principle,
        )
        .unwrap();
    assert!(critique.violated);
}

#[test]
fn be_truthful_does_not_fire_on_clean_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::be_truthful();
    let critique = reviser
        .critique("This method often solves many problems.", &principle)
        .unwrap();
    assert!(!critique.violated);
}

#[test]
fn respect_privacy_fires_on_violating_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique(
            "Reach me at jane.doe@example.com if you have questions.",
            &principle,
        )
        .unwrap();
    assert!(critique.violated);
}

#[test]
fn respect_privacy_does_not_fire_on_clean_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique(
            "Reach out to our support team if you have questions.",
            &principle,
        )
        .unwrap();
    assert!(!critique.violated);
}

#[test]
fn be_respectful_fires_on_violating_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::be_respectful();
    let critique = reviser
        .critique("You are being such an idiot right now.", &principle)
        .unwrap();
    assert!(critique.violated);
}

#[test]
fn be_respectful_does_not_fire_on_clean_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::be_respectful();
    let critique = reviser
        .critique("You are being very thoughtful right now.", &principle)
        .unwrap();
    assert!(!critique.violated);
}

#[test]
fn avoid_illegal_activity_fires_on_violating_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_illegal_activity();
    let critique = reviser
        .critique(
            "This tutorial explains how to hack into a router's admin panel.",
            &principle,
        )
        .unwrap();
    assert!(critique.violated);
}

#[test]
fn avoid_illegal_activity_does_not_fire_on_clean_example() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_illegal_activity();
    let critique = reviser
        .critique(
            "This tutorial explains how to configure a router's admin panel.",
            &principle,
        )
        .unwrap();
    assert!(!critique.violated);
}

// ── revision clears the violation ────────────────────────────────────────

#[test]
fn avoid_harm_revision_clears_violation() {
    assert_revision_clears_violation(
        &ConstitutionalPrinciple::avoid_harm(),
        "The villain wants to poison the town's water supply.",
    );
}

#[test]
fn be_truthful_revision_clears_violation() {
    assert_revision_clears_violation(
        &ConstitutionalPrinciple::be_truthful(),
        "This method is guaranteed to solve every problem.",
    );
}

#[test]
fn respect_privacy_revision_clears_violation() {
    assert_revision_clears_violation(
        &ConstitutionalPrinciple::respect_privacy(),
        "Reach me at jane.doe@example.com if you have questions.",
    );
}

#[test]
fn be_respectful_revision_clears_violation() {
    assert_revision_clears_violation(
        &ConstitutionalPrinciple::be_respectful(),
        "You are being such an idiot right now.",
    );
}

#[test]
fn avoid_illegal_activity_revision_clears_violation() {
    assert_revision_clears_violation(
        &ConstitutionalPrinciple::avoid_illegal_activity(),
        "This tutorial explains how to hack into a router's admin panel.",
    );
}

// ── PII sub-shape isolation ──────────────────────────────────────────────

#[test]
fn pii_shape_email_detected_alone() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique("Contact jane@example.com for details.", &principle)
        .unwrap();
    assert_eq!(critique.matches.len(), 1);
    assert!(critique.matches[0].trigger.contains("email"));
}

#[test]
fn pii_shape_phone_detected_alone() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique("Call 555-123-4567 for support.", &principle)
        .unwrap();
    assert_eq!(critique.matches.len(), 1);
    assert!(critique.matches[0].trigger.contains("phone"));
}

#[test]
fn pii_shape_ssn_detected_alone() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique("My ID is 123-45-6789 on file.", &principle)
        .unwrap();
    assert_eq!(critique.matches.len(), 1);
    assert!(critique.matches[0].trigger.contains("ssn"));
}

#[test]
fn pii_shape_redaction_uses_kind_specific_label() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::respect_privacy();
    let critique = reviser
        .critique("Contact jane@example.com for details.", &principle)
        .unwrap();
    let revision = reviser
        .revise("Contact jane@example.com for details.", &critique)
        .unwrap();
    assert!(revision.after.contains("[REDACTED-EMAIL]"));
    assert!(!revision.after.contains("jane@example.com"));
}

// ── multiple matches in one critique ─────────────────────────────────────

#[test]
fn multiple_keyword_matches_in_one_critique() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_harm();
    let critique = reviser
        .critique("The plan involves a bomb and a weapon cache.", &principle)
        .unwrap();
    assert_eq!(critique.matches.len(), 2);
}

#[test]
fn revise_applies_all_matches_in_one_pass() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::avoid_harm();
    let draft = "The plan involves a bomb and a weapon cache.";
    let critique = reviser.critique(draft, &principle).unwrap();
    let revision = reviser.revise(draft, &critique).unwrap();
    assert_eq!(revision.after.matches("[redacted]").count(), 2);
    assert!(!revision.after.contains("bomb"));
    assert!(!revision.after.contains("weapon"));
}

// ── MockConstitutionalReviser error paths ────────────────────────────────

#[test]
fn critique_empty_principle_id_errors() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::new("", "stmt");
    let err = reviser.critique("some draft", &principle).unwrap_err();
    assert!(matches!(err, ConstitutionalError::EmptyPrincipleId));
}

#[test]
fn critique_whitespace_principle_id_errors() {
    let reviser = mock();
    let principle = ConstitutionalPrinciple::new("   ", "stmt");
    let err = reviser.critique("some draft", &principle).unwrap_err();
    assert!(matches!(err, ConstitutionalError::EmptyPrincipleId));
}

#[test]
fn revise_non_violating_critique_errors() {
    let reviser = mock();
    let critique = ConstitutionalCritique {
        principle_id: "x".to_string(),
        violated: false,
        explanation: "ok".to_string(),
        matches: vec![],
    };
    let err = reviser.revise("draft", &critique).unwrap_err();
    assert!(
        matches!(err, ConstitutionalError::NoViolationToRevise { principle_id } if principle_id == "x")
    );
}

#[test]
fn revise_inconsistent_empty_matches_errors() {
    let reviser = mock();
    let critique = ConstitutionalCritique {
        principle_id: "y".to_string(),
        violated: true,
        explanation: "bad".to_string(),
        matches: vec![],
    };
    let err = reviser.revise("draft", &critique).unwrap_err();
    assert!(
        matches!(err, ConstitutionalError::InconsistentCritique { principle_id } if principle_id == "y")
    );
}

#[test]
fn revise_out_of_bounds_match_errors() {
    let reviser = mock();
    let critique = ConstitutionalCritique {
        principle_id: "z".to_string(),
        violated: true,
        explanation: "bad".to_string(),
        matches: vec![ConstitutionalMatch {
            trigger: "keyword: nope".to_string(),
            start: 100,
            end: 200,
            matched_text: "nope".to_string(),
            replacement: "[redacted]".to_string(),
        }],
    };
    let err = reviser.revise("short draft", &critique).unwrap_err();
    assert!(
        matches!(err, ConstitutionalError::InconsistentCritique { principle_id } if principle_id == "z")
    );
}

#[test]
fn mock_reviser_new_and_default_agree() {
    let principle = ConstitutionalPrinciple::avoid_harm();
    let via_new = MockConstitutionalReviser::new()
        .critique("clean text", &principle)
        .unwrap();
    let via_default = MockConstitutionalReviser
        .critique("clean text", &principle)
        .unwrap();
    assert_eq!(via_new.violated, via_default.violated);
}

// ── ConstitutionalEngine construction ────────────────────────────────────

#[test]
fn engine_default_uses_default_config() {
    assert_eq!(
        ConstitutionalEngine::default().config,
        ConstitutionalConfig::default()
    );
}

#[test]
fn engine_new_stores_config() {
    let cfg = ConstitutionalConfig::new().with_max_revisions(3);
    assert_eq!(ConstitutionalEngine::new(cfg.clone()).config, cfg);
}

#[test]
fn engine_clone_equals_original() {
    let engine = ConstitutionalEngine::new(ConstitutionalConfig::new().with_max_revisions(2));
    assert_eq!(engine.clone().config, engine.config);
}

#[test]
fn engine_accepts_trait_object() {
    let reviser: &dyn ConstitutionalReviser = &MockConstitutionalReviser::new();
    let principles = vec![ConstitutionalPrinciple::avoid_harm()];
    let result = engine_default()
        .run("I will poison the well.", &principles, reviser)
        .unwrap();
    assert_eq!(result.revisions_applied, 1);
}

// ── run: validation ───────────────────────────────────────────────────────

#[test]
fn run_empty_draft_errors() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let err = engine_default().run("", &principles, &mock()).unwrap_err();
    assert!(matches!(err, ConstitutionalError::EmptyDraft));
}

#[test]
fn run_whitespace_draft_errors() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let err = engine_default()
        .run("   \t\n", &principles, &mock())
        .unwrap_err();
    assert!(matches!(err, ConstitutionalError::EmptyDraft));
}

#[test]
fn run_empty_draft_error_display() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let err = engine_default().run("", &principles, &mock()).unwrap_err();
    assert_eq!(err.to_string(), "draft must not be empty");
}

// ── run: pass-through / clean ────────────────────────────────────────────

#[test]
fn run_empty_principle_list_passes_through() {
    let result = engine_default()
        .run("Hello world, this is fine.", &[], &mock())
        .unwrap();
    assert_eq!(result.final_draft, "Hello world, this is fine.");
    assert!(result.trace.is_empty());
    assert_eq!(result.revisions_applied, 0);
    assert!(!result.stopped_early);
}

#[test]
fn run_clean_draft_passes_through_unchanged() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This is a pleasant, calm, and thoughtful statement about the weather.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert!(result.is_unchanged());
    assert_eq!(result.final_draft, draft);
}

#[test]
fn run_clean_draft_has_full_trace_no_violations() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This is a pleasant, calm, and thoughtful statement about the weather.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert_eq!(result.trace.len(), 5);
    assert!(result.violated_principle_ids().is_empty());
    assert_eq!(result.revisions_applied, 0);
}

// ── run: single-principle violation ──────────────────────────────────────

#[test]
fn run_single_principle_violation_revises() {
    let principles = vec![ConstitutionalPrinciple::avoid_harm()];
    let result = engine_default()
        .run("I will poison the well.", &principles, &mock())
        .unwrap();
    assert_eq!(result.trace.len(), 1);
    assert!(result.trace[0].1.violated);
    assert!(result.trace[0].2.is_some());
    assert!(!result.final_draft.contains("poison"));
}

#[test]
fn run_trace_records_consistent_principle_id_across_triple() {
    let principles = vec![ConstitutionalPrinciple::avoid_harm()];
    let result = engine_default()
        .run("I will poison the well.", &principles, &mock())
        .unwrap();
    let (principle, critique, revision) = &result.trace[0];
    assert_eq!(principle.id, "avoid_harm");
    assert_eq!(critique.principle_id, "avoid_harm");
    assert_eq!(revision.as_ref().unwrap().principle_id, "avoid_harm");
}

// ── run: sequential principle application ────────────────────────────────

#[test]
fn sequential_application_second_principle_sees_revised_draft() {
    // Principle A hedges "foo" into "kill"; principle B then flags "kill".
    // The original draft never contains "kill" — so B can only fire if the
    // engine critiques the *current* (A-revised) draft, not the original.
    let draft = "please foo the bug";
    assert!(!draft.contains("kill"));

    let principle_a = ConstitutionalPrinciple::new("principle_a", "test A")
        .with_trigger(ConstitutionalTrigger::absolute("foo", "kill"));
    let principle_b = ConstitutionalPrinciple::new("principle_b", "test B")
        .with_trigger(ConstitutionalTrigger::keyword("kill"));
    let principles = vec![principle_a, principle_b];

    let result = engine_default().run(draft, &principles, &mock()).unwrap();

    assert_eq!(result.trace.len(), 2);
    assert!(result.trace[0].1.violated, "principle A should flag 'foo'");
    assert_eq!(
        result.trace[0].2.as_ref().unwrap().after,
        "please kill the bug"
    );
    assert!(
        result.trace[1].1.violated,
        "principle B should flag 'kill', which only exists after A's revision"
    );
}

#[test]
fn sequential_application_final_draft_and_counts() {
    let draft = "please foo the bug";
    let principle_a = ConstitutionalPrinciple::new("principle_a", "test A")
        .with_trigger(ConstitutionalTrigger::absolute("foo", "kill"));
    let principle_b = ConstitutionalPrinciple::new("principle_b", "test B")
        .with_trigger(ConstitutionalTrigger::keyword("kill"));
    let principles = vec![principle_a, principle_b];

    let result = engine_default().run(draft, &principles, &mock()).unwrap();

    assert_eq!(result.revisions_applied, 2);
    assert_eq!(result.final_draft, "please [redacted] the bug");
}

// ── run: custom principle lists ──────────────────────────────────────────

#[test]
fn custom_principle_list_not_default_works() {
    let principles = vec![
        ConstitutionalPrinciple::new("no_x", "no xylophones")
            .with_trigger(ConstitutionalTrigger::keyword("xylophone")),
        ConstitutionalPrinciple::new("no_y", "no yodeling")
            .with_trigger(ConstitutionalTrigger::keyword("yodeling")),
    ];
    let draft = "I love to play the xylophone and go yodeling in the mountains.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();

    let ids: Vec<String> = result.trace.iter().map(|(p, _, _)| p.id.clone()).collect();
    assert_eq!(ids, vec!["no_x", "no_y"]);
    assert_eq!(result.revisions_applied, 2);
}

#[test]
fn custom_single_principle_end_to_end() {
    let principles = vec![
        ConstitutionalPrinciple::new("no_urgent", "avoid false urgency")
            .with_trigger(ConstitutionalTrigger::keyword("URGENT")),
    ];
    let draft = "This is an URGENT request, please respond ASAP.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert!(!result.final_draft.to_lowercase().contains("urgent"));
}

// ── run: enabled_principle_ids filtering ─────────────────────────────────

#[test]
fn enabled_principle_ids_filters_others_out() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_enabled_principle_ids(["avoid_harm"]);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert_eq!(result.trace.len(), 1);
    assert_eq!(result.trace[0].0.id, "avoid_harm");
}

#[test]
fn enabled_principle_ids_final_draft_reflects_filter() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_enabled_principle_ids(["avoid_harm"]);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    // be_truthful was filtered out, so "definitely" survives; avoid_harm
    // was enabled, so "poison" was redacted.
    assert!(result.final_draft.contains("definitely"));
    assert!(!result.final_draft.contains("poison"));
}

#[test]
fn enabled_principle_ids_empty_list_disables_all() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_enabled_principle_ids(Vec::<String>::new());
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert!(result.trace.is_empty());
    assert!(result.is_unchanged());
}

// ── run: max_revisions cap ───────────────────────────────────────────────

#[test]
fn max_revisions_cap_stops_further_revision() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_max_revisions(1);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();
    assert_eq!(result.revisions_applied, 1);
}

#[test]
fn max_revisions_cap_still_records_critique_for_later_principle() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_max_revisions(1);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    // avoid_harm (index 0) gets revised; be_truthful (index 1) is still
    // critiqued (and still violated) but the cap prevents revision.
    assert_eq!(result.trace.len(), 5);
    assert!(result.trace[1].1.violated);
    assert!(result.trace[1].2.is_none());
}

#[test]
fn max_revisions_cap_final_draft_is_partial() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_max_revisions(1);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert!(!result.final_draft.contains("poison"));
    assert!(result.final_draft.contains("definitely"));
}

#[test]
fn max_revisions_zero_never_revises() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_max_revisions(0);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert_eq!(result.revisions_applied, 0);
    assert!(result.is_unchanged());
}

// ── run: stop_after_consecutive_clean ────────────────────────────────────

#[test]
fn stop_after_consecutive_clean_triggers() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_stop_after_consecutive_clean(2);
    let draft = "This will poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert!(result.stopped_early);
    // avoid_harm (violated), be_truthful (clean, 1), respect_privacy (clean, 2 -> stop).
    assert_eq!(result.trace.len(), 3);
}

#[test]
fn stop_after_consecutive_clean_records_correct_ids() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_stop_after_consecutive_clean(2);
    let draft = "This will poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    let ids: Vec<String> = result.trace.iter().map(|(p, _, _)| p.id.clone()).collect();
    assert_eq!(ids, vec!["avoid_harm", "be_truthful", "respect_privacy"]);
}

#[test]
fn stop_after_consecutive_clean_zero_disables_early_stop() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_stop_after_consecutive_clean(0);
    let draft = "This will poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert!(!result.stopped_early);
    assert_eq!(result.trace.len(), 5);
}

#[test]
fn stop_after_consecutive_clean_not_reached_processes_everything() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_stop_after_consecutive_clean(10);
    let draft = "This will poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert!(!result.stopped_early);
    assert_eq!(result.trace.len(), 5);
}

// ── run: determinism ──────────────────────────────────────────────────────

#[test]
fn run_is_deterministic_default_constitution() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This will definitely poison the well, and you are an idiot.";
    let a = engine_default().run(draft, &principles, &mock()).unwrap();
    let b = engine_default().run(draft, &principles, &mock()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn run_is_deterministic_with_cap_and_stop() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new()
        .with_max_revisions(1)
        .with_stop_after_consecutive_clean(2);
    let draft = "This will definitely poison the well.";
    let engine = ConstitutionalEngine::new(config);
    let a = engine.run(draft, &principles, &mock()).unwrap();
    let b = engine.run(draft, &principles, &mock()).unwrap();
    assert_eq!(a, b);
}

// ── run: trace order / completeness ──────────────────────────────────────

#[test]
fn trace_order_matches_principle_order() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This is a pleasant, calm, and thoughtful statement about the weather.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    let trace_ids: Vec<String> = result.trace.iter().map(|(p, _, _)| p.id.clone()).collect();
    let principle_ids: Vec<String> = principles.iter().map(|p| p.id.clone()).collect();
    assert_eq!(trace_ids, principle_ids);
}

#[test]
fn trace_length_equals_principles_when_no_early_stop() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This will definitely poison the well, and you are an idiot.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert_eq!(result.trace.len(), principles.len());
}

// ── ConstitutionalConfig ─────────────────────────────────────────────────

#[test]
fn config_default_values() {
    let config = ConstitutionalConfig::default();
    assert_eq!(config.enabled_principle_ids, None);
    assert_eq!(config.max_revisions, 10);
    assert_eq!(config.stop_after_consecutive_clean, None);
}

#[test]
fn config_new_equals_default() {
    assert_eq!(ConstitutionalConfig::new(), ConstitutionalConfig::default());
}

#[test]
fn config_with_max_revisions_sets_field() {
    assert_eq!(
        ConstitutionalConfig::new()
            .with_max_revisions(7)
            .max_revisions,
        7
    );
}

#[test]
fn config_with_enabled_principle_ids_sets_field() {
    let config = ConstitutionalConfig::new().with_enabled_principle_ids(["a", "b"]);
    assert_eq!(
        config.enabled_principle_ids,
        Some(vec!["a".to_string(), "b".to_string()])
    );
}

#[test]
fn config_with_stop_after_consecutive_clean_sets_field() {
    let config = ConstitutionalConfig::new().with_stop_after_consecutive_clean(4);
    assert_eq!(config.stop_after_consecutive_clean, Some(4));
}

#[test]
fn config_builders_chain_and_preserve_fields() {
    let config = ConstitutionalConfig::new()
        .with_max_revisions(2)
        .with_stop_after_consecutive_clean(3)
        .with_enabled_principle_ids(["only_this"]);
    assert_eq!(config.max_revisions, 2);
    assert_eq!(config.stop_after_consecutive_clean, Some(3));
    assert_eq!(
        config.enabled_principle_ids,
        Some(vec!["only_this".to_string()])
    );
}

#[test]
fn config_clone_equals_original() {
    let config = ConstitutionalConfig::new().with_max_revisions(9);
    assert_eq!(config.clone(), config);
}

// ── ConstitutionalResult convenience methods ─────────────────────────────

#[test]
fn result_is_unchanged_true_when_no_revisions() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This is a pleasant, calm, and thoughtful statement about the weather.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert!(result.is_unchanged());
}

#[test]
fn result_is_unchanged_false_when_revised() {
    let principles = vec![ConstitutionalPrinciple::avoid_harm()];
    let result = engine_default()
        .run("I will poison the well.", &principles, &mock())
        .unwrap();
    assert!(!result.is_unchanged());
}

#[test]
fn result_violated_principle_ids_content() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This will definitely poison the well.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert_eq!(
        result.violated_principle_ids(),
        vec!["avoid_harm".to_string(), "be_truthful".to_string()]
    );
}

#[test]
fn result_revised_principle_ids_differs_from_violated_under_cap() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let config = ConstitutionalConfig::new().with_max_revisions(1);
    let draft = "This will definitely poison the well.";
    let result = ConstitutionalEngine::new(config)
        .run(draft, &principles, &mock())
        .unwrap();

    assert_eq!(
        result.violated_principle_ids(),
        vec!["avoid_harm".to_string(), "be_truthful".to_string()]
    );
    assert_eq!(
        result.revised_principle_ids(),
        vec!["avoid_harm".to_string()]
    );
}

#[test]
fn result_clone_equals_original() {
    let principles = ConstitutionalPrinciple::default_constitution();
    let draft = "This will definitely poison the well.";
    let result = engine_default().run(draft, &principles, &mock()).unwrap();
    assert_eq!(result.clone(), result);
}

// ── ConstitutionalError ──────────────────────────────────────────────────

#[test]
fn error_empty_draft_display() {
    assert_eq!(
        ConstitutionalError::EmptyDraft.to_string(),
        "draft must not be empty"
    );
}

#[test]
fn error_empty_principle_id_display() {
    assert_eq!(
        ConstitutionalError::EmptyPrincipleId.to_string(),
        "principle id must not be empty"
    );
}

#[test]
fn error_no_violation_to_revise_display() {
    let err = ConstitutionalError::NoViolationToRevise {
        principle_id: "x".to_string(),
    };
    assert_eq!(err.to_string(), "no violation to revise for principle 'x'");
}

#[test]
fn error_inconsistent_critique_display() {
    let err = ConstitutionalError::InconsistentCritique {
        principle_id: "y".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "critique for principle 'y' is inconsistent: claims a violation but has no usable matches"
    );
}
