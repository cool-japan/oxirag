//! The four layers on Japanese input, and the draft assembly that feeds them.
//!
//! # What was measured before this file existed
//!
//! A seven-document Japanese corpus, indexed and queried through the whole pipeline:
//!
//! ```text
//! STATS  {"documents":7,"entities":0,"relationships":0}
//! claims: 0   status: Unknown   summary: "No verifiable claims found"
//! ```
//!
//! Layer 1 ranked correctly — it works on character bigrams and never needed words. Layers 3 and 4
//! returned nothing at all, because every heuristic in them assumed sentences end in `.` and words
//! are separated by spaces. Neither is true of Japanese, so the claim extractor saw one 40-character
//! "word" and bailed at its two-token minimum, and the entity extractor looked for capital letters
//! in a script that has no case.
//!
//! These tests are the floor under the fix. They assert the layers produce something, and that what
//! they produce is about the right things — not an exact count, which would break on any tuning.

#![cfg(not(target_arch = "wasm32"))]

use oxirag::layer2_speculator::{RuleBasedSpeculator, Speculator};
use oxirag::layer3_judge::AdvancedClaimExtractor;
use oxirag::layer3_judge::traits::ClaimExtractor;
#[cfg(feature = "graphrag")]
use oxirag::layer4_graph::extractor::{PatternEntityExtractor, PatternRelationshipExtractor};
#[cfg(feature = "graphrag")]
use oxirag::layer4_graph::traits::{EntityExtractor, RelationshipExtractor};
use oxirag::types::{Document, Draft, SearchResult};

/// Everyday Japanese, in the subject-topic-comment shape statements are normally written in.
const PENGUIN: &str = "ペンギンは鳥です。ペンギンは空を飛びません。ペンギンは海を泳ぎます。";
const FUJI: &str =
    "富士山は日本で一番高い山です。富士山の高さは3776メートルです。富士山は静岡県にあります。";
const BICYCLE: &str = "自転車はペダルをこいで進みます。自転車にはエンジンがありません。自転車は道路の左側を走ります。";

// ── Layer 3: claims ───────────────────────────────────────────────────────────

#[tokio::test]
async fn japanese_sentences_produce_claims() {
    let extractor = AdvancedClaimExtractor::new();
    let claims = extractor
        .extract_claims(PENGUIN, 10)
        .await
        .expect("extraction should succeed");

    assert_eq!(
        claims.len(),
        3,
        "three sentences, three claims — got {:?}",
        claims.iter().map(|c| &c.text).collect::<Vec<_>>()
    );
    assert!(claims.iter().all(|c| c.text.contains("ペンギン")));
}

#[tokio::test]
async fn japanese_negation_is_carried_into_the_structure() {
    let extractor = AdvancedClaimExtractor::new();
    let claims = extractor
        .extract_claims("ペンギンは空を飛びません。", 10)
        .await
        .expect("extraction should succeed");

    let claim = claims.first().expect("one sentence, one claim");
    let smtlib = extractor
        .to_smtlib(claim)
        .expect("a claim converts to SMT-LIB2");
    assert!(
        smtlib.contains("(not "),
        "`飛びません` is a negation and must survive as one: {smtlib}"
    );
}

#[tokio::test]
async fn a_claim_is_not_extracted_twice() {
    let extractor = AdvancedClaimExtractor::new();
    // The same sentence reaching the judge twice — the shape a draft plus its own context summary
    // used to have.
    let repeated = format!("{PENGUIN} {PENGUIN}");
    let claims = extractor
        .extract_claims(&repeated, 20)
        .await
        .expect("extraction should succeed");

    assert_eq!(
        claims.len(),
        3,
        "duplicates must collapse — got {:?}",
        claims.iter().map(|c| &c.text).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn english_extraction_still_works() {
    let extractor = AdvancedClaimExtractor::new();
    let claims = extractor
        .extract_claims(
            "OxiZ is a Pure Rust SMT solver. MeCrab is a morphological analyser.",
            10,
        )
        .await
        .expect("extraction should succeed");

    assert_eq!(claims.len(), 2);
    assert!(claims[0].text.contains("OxiZ"));
}

// ── Layer 4: entities and relationships ───────────────────────────────────────

#[cfg(feature = "graphrag")]
#[tokio::test]
async fn japanese_text_yields_entities() {
    let extractor = PatternEntityExtractor::new();
    let entities = extractor
        .extract_entities(FUJI)
        .await
        .expect("extraction should succeed");

    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"富士山"), "got {names:?}");
    assert!(names.contains(&"静岡県"), "got {names:?}");
    assert!(names.contains(&"日本"), "got {names:?}");
}

#[cfg(feature = "graphrag")]
#[tokio::test]
async fn a_japanese_place_suffix_classifies_the_entity() {
    let extractor = PatternEntityExtractor::new();
    let entities = extractor
        .extract_entities(FUJI)
        .await
        .expect("extraction should succeed");

    let shizuoka = entities
        .iter()
        .find(|e| e.name == "静岡県")
        .expect("静岡県 is extracted");
    assert_eq!(
        shizuoka.entity_type,
        oxirag::layer4_graph::types::EntityType::Location,
        "`県` names a prefecture the way `Prefecture` does in English"
    );
}

#[cfg(feature = "graphrag")]
#[tokio::test]
async fn a_relationship_is_not_read_across_an_intervening_entity() {
    let entity_extractor = PatternEntityExtractor::new();
    let relationship_extractor = PatternRelationshipExtractor::new();

    // Three entities in one sentence: カレー, インド, 料理. The span between カレー and 料理 holds
    // 「で生まれ」, so a pattern search over it used to report `カレー LOCATED_IN 料理`. The verb
    // belongs to インド, which sits between them.
    let text = "カレーはインドで生まれた料理です。";
    let entities = entity_extractor
        .extract_entities(text)
        .await
        .expect("extraction should succeed");
    let relationships = relationship_extractor
        .extract_relationships(text, &entities)
        .await
        .expect("extraction should succeed");

    let name_of = |id: &str| {
        entities
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.name.clone())
            .unwrap_or_default()
    };
    let pairs: Vec<(String, String, String)> = relationships
        .iter()
        .map(|r| {
            (
                name_of(&r.source_id),
                format!("{:?}", r.relationship_type),
                name_of(&r.target_id),
            )
        })
        .collect();

    assert!(
        !pairs.iter().any(|(source, kind, target)| source == "カレー"
            && target == "料理"
            && kind == "LocatedIn"),
        "read a relationship across an intervening entity: {pairs:?}"
    );
    assert!(
        pairs
            .iter()
            .any(|(source, _, target)| source == "カレー" && target == "インド"),
        "the relationship that IS stated was dropped: {pairs:?}"
    );
}

#[cfg(feature = "graphrag")]
#[tokio::test]
async fn a_stoplist_word_glued_to_a_stem_is_not_an_entity() {
    let extractor = PatternEntityExtractor::new();
    let entities = extractor
        .extract_entities("富士山は日本で一番高い山です。")
        .await
        .expect("extraction should succeed");

    let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    assert!(
        !names.contains(&"一番高"),
        "`一番` glued to the stem of `高い` is not a name: {names:?}"
    );
    assert!(names.contains(&"富士山"), "got {names:?}");
}

#[cfg(feature = "graphrag")]
#[tokio::test]
async fn japanese_entities_are_linked() {
    let entity_extractor = PatternEntityExtractor::new();
    let relationship_extractor = PatternRelationshipExtractor::new();

    let entities = entity_extractor
        .extract_entities(FUJI)
        .await
        .expect("extraction should succeed");
    let relationships = relationship_extractor
        .extract_relationships(FUJI, &entities)
        .await
        .expect("extraction should succeed");

    assert!(
        !relationships.is_empty(),
        "a Japanese corpus used to produce a graph with no edges at all"
    );
}

// ── Layer 2 / draft assembly ──────────────────────────────────────────────────

fn results() -> Vec<SearchResult> {
    vec![
        SearchResult::new(Document::new(PENGUIN), 0.9, 0),
        SearchResult::new(Document::new(BICYCLE), 0.3, 1),
        SearchResult::new(Document::new(FUJI), 0.2, 2),
    ]
}

#[tokio::test]
async fn a_revision_adds_evidence_rather_than_commentary() {
    let speculator = RuleBasedSpeculator::default();
    let context = results();
    let draft = Draft::new("ペンギンは鳥です。", "ペンギンは空を飛びますか");

    let speculation = speculator
        .verify_draft(&draft, &context)
        .await
        .expect("verification should succeed");
    let revised = speculator
        .revise_draft(&draft, &context, &speculation)
        .await
        .expect("revision should succeed");

    // The four things the old implementation put into the answer text, each of which then reached
    // the verdict table as a claim.
    assert!(
        !revised
            .content
            .contains("Based on the available information")
    );
    assert!(!revised.content.contains("Revision notes"));
    assert!(
        !revised.content.contains("エンジン"),
        "the bicycle document shares no query term and must not be pulled in: {}",
        revised.content
    );
    // No sentence appears twice.
    let sentences = oxirag::text::split_sentences(&revised.content);
    let mut normalized: Vec<String> = sentences
        .iter()
        .map(|s| oxirag::text::normalize_for_compare(s))
        .collect();
    let before = normalized.len();
    normalized.sort();
    normalized.dedup();
    assert_eq!(before, normalized.len(), "a sentence was added twice");
}

#[tokio::test]
async fn a_revision_never_truncates_mid_word() {
    let speculator = RuleBasedSpeculator::default();
    let context = vec![SearchResult::new(
        Document::new("MeCrab uses the IPADIC dictionary and segments text without spaces."),
        0.9,
        0,
    )];
    let draft = Draft::new("MeCrab", "which dictionary does MeCrab use");

    let speculation = speculator
        .verify_draft(&draft, &context)
        .await
        .expect("verification should succeed");
    let revised = speculator
        .revise_draft(&draft, &context, &speculation)
        .await
        .expect("revision should succeed");

    assert!(
        !revised.content.contains("dicti "),
        "the 100-character cut produced `IPADIC dicti`: {}",
        revised.content
    );
    assert!(revised.content.contains("dictionary"));
}
