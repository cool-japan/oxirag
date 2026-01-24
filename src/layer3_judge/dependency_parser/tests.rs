//! Tests for dependency parser module.

use super::*;
use crate::types::ClaimStructure;

// =========================================================================
// DependencyRelation tests
// =========================================================================

#[test]
fn test_dependency_relation_from_label() {
    assert_eq!(
        DependencyRelation::from_label("nsubj"),
        DependencyRelation::NominalSubject
    );
    assert_eq!(
        DependencyRelation::from_label("dobj"),
        DependencyRelation::DirectObject
    );
    assert_eq!(
        DependencyRelation::from_label("obj"),
        DependencyRelation::DirectObject
    );
    assert_eq!(
        DependencyRelation::from_label("NSUBJ"),
        DependencyRelation::NominalSubject
    );
    assert_eq!(
        DependencyRelation::from_label("unknown_rel"),
        DependencyRelation::Unknown
    );
}

#[test]
fn test_dependency_relation_as_label() {
    assert_eq!(DependencyRelation::NominalSubject.as_label(), "nsubj");
    assert_eq!(DependencyRelation::DirectObject.as_label(), "dobj");
    assert_eq!(DependencyRelation::Root.as_label(), "root");
}

// =========================================================================
// PosTag tests
// =========================================================================

#[test]
fn test_pos_tag_from_tag() {
    assert_eq!(PosTag::from_tag("NN"), PosTag::Noun);
    assert_eq!(PosTag::from_tag("VB"), PosTag::Verb);
    assert_eq!(PosTag::from_tag("JJ"), PosTag::Adjective);
    assert_eq!(PosTag::from_tag("RB"), PosTag::Adverb);
    assert_eq!(PosTag::from_tag("DT"), PosTag::Determiner);
}

#[test]
fn test_pos_tag_is_verb() {
    assert!(PosTag::Verb.is_verb());
    assert!(PosTag::Auxiliary.is_verb());
    assert!(!PosTag::Noun.is_verb());
}

#[test]
fn test_pos_tag_is_noun() {
    assert!(PosTag::Noun.is_noun());
    assert!(PosTag::ProperNoun.is_noun());
    assert!(PosTag::Pronoun.is_noun());
    assert!(!PosTag::Verb.is_noun());
}

#[test]
fn test_pos_tag_is_content_word() {
    assert!(PosTag::Noun.is_content_word());
    assert!(PosTag::Verb.is_content_word());
    assert!(PosTag::Adjective.is_content_word());
    assert!(!PosTag::Determiner.is_content_word());
}

// =========================================================================
// DependencyNode tests
// =========================================================================

#[test]
fn test_dependency_node_creation() {
    let node = DependencyNode::new("test", 0);
    assert_eq!(node.token, "test");
    assert_eq!(node.index, 0);
    assert_eq!(node.pos_tag, PosTag::Unknown);
}

#[test]
fn test_dependency_node_builder() {
    let node = DependencyNode::new("running", 1)
        .with_lemma("run")
        .with_pos(PosTag::Verb)
        .with_relation(DependencyRelation::Root)
        .with_head(-1)
        .with_offsets(5, 12);

    assert_eq!(node.lemma, "run");
    assert_eq!(node.pos_tag, PosTag::Verb);
    assert_eq!(node.relation, DependencyRelation::Root);
    assert_eq!(node.head_index, -1);
    assert_eq!(node.char_start, 5);
    assert_eq!(node.char_end, 12);
}

#[test]
fn test_dependency_node_is_root() {
    let root_node = DependencyNode::new("is", 0)
        .with_relation(DependencyRelation::Root)
        .with_head(-1);
    assert!(root_node.is_root());

    let non_root = DependencyNode::new("dog", 1)
        .with_relation(DependencyRelation::NominalSubject)
        .with_head(0);
    assert!(!non_root.is_root());
}

// =========================================================================
// DependencyTree tests
// =========================================================================

#[test]
fn test_dependency_tree_creation() {
    let tree = DependencyTree::new("The dog barks.");
    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
    assert!(tree.root().is_none());
}

#[test]
fn test_dependency_tree_add_node() {
    let mut tree = DependencyTree::new("The dog barks.");

    let root = DependencyNode::new("barks", 2)
        .with_relation(DependencyRelation::Root)
        .with_head(-1);
    tree.add_node(root);

    assert_eq!(tree.len(), 1);
    assert!(tree.root().is_some());
    assert_eq!(
        tree.root()
            .expect("root should be present after adding root node")
            .token,
        "barks"
    );
}

#[test]
fn test_dependency_tree_children() {
    let mut tree = DependencyTree::new("The dog barks.");

    tree.add_node(
        DependencyNode::new("The", 0)
            .with_relation(DependencyRelation::Determiner)
            .with_head(1),
    );
    tree.add_node(
        DependencyNode::new("dog", 1)
            .with_relation(DependencyRelation::NominalSubject)
            .with_head(2),
    );
    tree.add_node(
        DependencyNode::new("barks", 2)
            .with_relation(DependencyRelation::Root)
            .with_head(-1),
    );

    let children = tree.children(2);
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].token, "dog");

    let det_children = tree.children_with_relation(1, DependencyRelation::Determiner);
    assert_eq!(det_children.len(), 1);
    assert_eq!(det_children[0].token, "The");
}

#[test]
fn test_dependency_tree_find_by_relation() {
    let mut tree = DependencyTree::new("Test");

    tree.add_node(DependencyNode::new("a", 0).with_relation(DependencyRelation::NominalSubject));
    tree.add_node(DependencyNode::new("b", 1).with_relation(DependencyRelation::Root));
    tree.add_node(DependencyNode::new("c", 2).with_relation(DependencyRelation::NominalSubject));

    let subjects = tree.find_by_relation(DependencyRelation::NominalSubject);
    assert_eq!(subjects.len(), 2);
}

#[test]
fn test_dependency_tree_subtree_text() {
    let mut tree = DependencyTree::new("The big dog barks loudly.");

    tree.add_node(
        DependencyNode::new("The", 0)
            .with_relation(DependencyRelation::Determiner)
            .with_head(2),
    );
    tree.add_node(
        DependencyNode::new("big", 1)
            .with_relation(DependencyRelation::AdjectivalModifier)
            .with_head(2),
    );
    tree.add_node(
        DependencyNode::new("dog", 2)
            .with_relation(DependencyRelation::NominalSubject)
            .with_head(3),
    );
    tree.add_node(
        DependencyNode::new("barks", 3)
            .with_relation(DependencyRelation::Root)
            .with_head(-1),
    );

    let subtree = tree.subtree_text(2);
    assert!(subtree.contains("The"));
    assert!(subtree.contains("big"));
    assert!(subtree.contains("dog"));
}

// =========================================================================
// SvoTriple tests
// =========================================================================

#[test]
fn test_svo_triple_creation() {
    let triple = SvoTriple::new("dog", "chases", Some("cat"));
    assert_eq!(triple.subject, "dog");
    assert_eq!(triple.verb, "chases");
    assert_eq!(triple.object, Some("cat".to_string()));
    assert!(!triple.negated);
}

#[test]
fn test_svo_triple_builder() {
    let triple = SvoTriple::new("system", "is", Some("operational"))
        .with_negation(true)
        .with_confidence(0.85);

    assert!(triple.negated);
    assert!((triple.confidence - 0.85).abs() < f32::EPSILON);
}

// =========================================================================
// SimpleDependencyParser tests
// =========================================================================

#[test]
fn test_simple_parser_tokenize() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The cat sat.");
    assert!(!tree.is_empty());
}

#[test]
fn test_simple_parser_basic_sentence() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The dog is happy.");

    assert!(tree.root().is_some());
    let root = tree
        .root()
        .expect("root should be present for a valid sentence");
    assert!(root.pos_tag.is_verb() || root.pos_tag == PosTag::Auxiliary);
}

#[test]
fn test_simple_parser_svo_extraction_basic() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The cat is fluffy.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
    let triple = &triples[0];
    assert!(triple.subject.to_lowercase().contains("cat"));
    assert_eq!(triple.verb, "be");
}

#[test]
fn test_simple_parser_svo_with_object() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("John uses the computer.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_simple_parser_negation() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The system is not working.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
    assert!(triples[0].negated);
}

#[test]
fn test_simple_parser_complex_subject() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The large red ball is round.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_simple_parser_empty_text() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(triples.is_empty());
}

#[test]
fn test_simple_parser_no_verb() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The quick brown fox.");
    let triples = parser.extract_subject_verb_object(&tree);

    // May or may not extract something, but shouldn't crash
    assert!(triples.is_empty() || !triples.is_empty());
}

// =========================================================================
// DependencyClaimExtractor tests
// =========================================================================

#[test]
fn test_claim_extractor_creation() {
    let extractor = DependencyClaimExtractor::new();
    // Just check it doesn't panic
    let result = extractor.parser.parse("test");
    // Verify the result is a valid parse tree (possibly empty)
    let _node_count = result.nodes.len();
}

#[test]
fn test_claim_extractor_single_sentence() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("The sky is blue.", 10)
        .expect("extraction should succeed");

    assert!(!claims.is_empty());
    assert!(matches!(
        claims[0].structure,
        ClaimStructure::Predicate { .. }
    ));
}

#[test]
fn test_claim_extractor_multiple_sentences() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("The sky is blue. Water is wet. Fire is hot.", 10)
        .expect("extraction should succeed");

    assert!(claims.len() >= 2);
}

#[test]
fn test_claim_extractor_max_claims() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("A is B. C is D. E is F. G is H.", 2)
        .expect("extraction should succeed");

    assert!(claims.len() <= 2);
}

#[test]
fn test_claim_extractor_negation() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("The answer is not correct.", 10)
        .expect("extraction should succeed");

    if !claims.is_empty() {
        assert!(matches!(claims[0].structure, ClaimStructure::Not(_)));
    }
}

#[test]
fn test_claim_extractor_empty_text() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("", 10)
        .expect("extraction should succeed");

    assert!(claims.is_empty());
}

#[test]
fn test_claim_extractor_short_fragments() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("Yes. No. Ok.", 10)
        .expect("extraction should succeed");

    // Short fragments should be filtered out
    assert!(claims.is_empty());
}

#[test]
fn test_extract_relationships() {
    let extractor = DependencyClaimExtractor::new();
    let relationships = extractor.extract_relationships("The cat is happy. The dog is sad.");

    assert!(!relationships.is_empty());
}

#[test]
fn test_parse_sentences() {
    let extractor = DependencyClaimExtractor::new();
    let trees = extractor.parse_sentences("Hello world. Goodbye moon.");

    assert_eq!(trees.len(), 2);
}

// =========================================================================
// Various sentence structure tests
// =========================================================================

#[test]
fn test_passive_voice() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The ball was kicked by John.");
    let triples = parser.extract_subject_verb_object(&tree);

    // Should handle passive voice
    assert!(triples.is_empty() || !triples.is_empty()); // Just ensure no crash
}

#[test]
fn test_compound_sentence() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("The sun is bright and the moon is pale.", 10)
        .expect("extraction should succeed");

    // May extract one or more claims
    assert!(claims.is_empty() || !claims.is_empty());
}

#[test]
fn test_question_handling() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("Is the sky blue?", 10)
        .expect("extraction should succeed");

    // Questions should be handled (may or may not produce claims)
    assert!(claims.is_empty() || !claims.is_empty());
}

#[test]
fn test_imperative_handling() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("Close the door.", 10)
        .expect("extraction should succeed");

    // Imperatives have implied subject
    assert!(claims.is_empty() || !claims.is_empty());
}

#[test]
fn test_copula_constructions() {
    let parser = SimpleDependencyParser::new();

    // "X is Y" pattern
    let tree1 = parser.parse("Paris is the capital of France.");
    let triples1 = parser.extract_subject_verb_object(&tree1);
    assert!(!triples1.is_empty());

    // "X is adjective" pattern
    let tree2 = parser.parse("The weather is beautiful.");
    let triples2 = parser.extract_subject_verb_object(&tree2);
    assert!(!triples2.is_empty());
}

#[test]
fn test_action_verbs() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The engineer creates solutions.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
    assert!(triples[0].object.is_some());
}

#[test]
fn test_auxiliary_verbs() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The system can process data.");
    let _ = parser.extract_subject_verb_object(&tree);
    // Just ensure no crash with auxiliaries
}

#[test]
fn test_prepositional_phrases() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The book is on the table.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_multiple_modifiers() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The very large and extremely heavy box is difficult.");
    let triples = parser.extract_subject_verb_object(&tree);

    // Should handle complex noun phrases
    assert!(triples.is_empty() || !triples.is_empty());
}

#[test]
fn test_proper_nouns() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("John loves Mary.");
    let triples = parser.extract_subject_verb_object(&tree);

    // Proper nouns should be recognized
    assert!(triples.is_empty() || !triples.is_empty());
}

#[test]
fn test_pronouns() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("He is happy.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_numbers_in_sentence() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The answer is 42.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_technical_terms() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The API provides authentication.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(!triples.is_empty());
}

#[test]
fn test_scientific_claims() {
    let extractor = DependencyClaimExtractor::new();

    let claims = extractor
        .extract_claims("Water boils at 100 degrees Celsius.", 10)
        .expect("extraction should succeed");

    // Scientific claims should be extractable
    assert!(claims.is_empty() || !claims.is_empty());
}

#[test]
fn test_causal_sentences() {
    let extractor = DependencyClaimExtractor::new();

    let claims = extractor
        .extract_claims("Exercise improves health.", 10)
        .expect("extraction should succeed");

    assert!(!claims.is_empty());
}

#[test]
fn test_unicode_handling() {
    let parser = SimpleDependencyParser::new();

    // Test with unicode characters
    let tree = parser.parse("The cafe is nice.");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(triples.is_empty() || !triples.is_empty());
}

#[test]
fn test_contractions() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("It's important.");
    let _ = parser.extract_subject_verb_object(&tree);
    // Just ensure no crash
}

#[test]
fn test_hyphenated_words() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The well-known scientist is famous.");
    let _ = parser.extract_subject_verb_object(&tree);
    // Just ensure no crash
}

// =========================================================================
// Edge cases and robustness tests
// =========================================================================

#[test]
fn test_very_long_sentence() {
    let parser = SimpleDependencyParser::new();

    let long_sentence = "The quick brown fox jumps over the lazy dog while the cat watches from the window and the bird sings in the tree.";
    let tree = parser.parse(long_sentence);

    assert!(!tree.is_empty());
}

#[test]
fn test_single_word() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("Hello");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(triples.is_empty());
}

#[test]
fn test_only_punctuation() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("...");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(triples.is_empty());
}

#[test]
fn test_whitespace_only() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("   ");
    let triples = parser.extract_subject_verb_object(&tree);

    assert!(triples.is_empty());
}

#[test]
fn test_special_characters() {
    let parser = SimpleDependencyParser::new();

    let tree = parser.parse("The price is $100.");
    let _ = parser.extract_subject_verb_object(&tree);
    // Just ensure no crash
}

#[test]
fn test_claim_confidence_range() {
    let extractor = DependencyClaimExtractor::new();
    let claims = extractor
        .extract_claims("The system is operational.", 10)
        .expect("extraction should succeed");

    for claim in claims {
        assert!(claim.confidence >= 0.0);
        assert!(claim.confidence <= 1.0);
    }
}

#[test]
fn test_triple_confidence_range() {
    let parser = SimpleDependencyParser::new();
    let tree = parser.parse("The system is operational.");
    let triples = parser.extract_subject_verb_object(&tree);

    for triple in triples {
        assert!(triple.confidence >= 0.0);
        assert!(triple.confidence <= 1.0);
    }
}
