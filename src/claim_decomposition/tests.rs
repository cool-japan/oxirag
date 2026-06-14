#![allow(
    clippy::float_cmp,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::useless_vec,
    clippy::map_unwrap_or,
    clippy::manual_range_contains,
    clippy::many_single_char_names,
    clippy::match_wildcard_for_single_variants,
    clippy::default_constructed_unit_structs,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
//! Tests for the `claim_decomposition` module.

use crate::claim_decomposition::extractor::{split_clauses, split_sentences, tokenize};
use crate::claim_decomposition::{
    AtomicClaim, AtomicClaimExtractor, ClaimDecompConfig, ClaimDecompError, ClaimExtractor,
    HeuristicAtomicExtractor,
};

// ── ClaimDecompConfig defaults & builders ─────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = ClaimDecompConfig::default();
    assert_eq!(cfg.min_tokens, 3);
    assert!(cfg.decontextualize);
    assert_eq!(cfg.max_claims, 64);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(ClaimDecompConfig::new(), ClaimDecompConfig::default());
}

#[test]
fn test_config_with_min_tokens() {
    let cfg = ClaimDecompConfig::new().with_min_tokens(5);
    assert_eq!(cfg.min_tokens, 5);
}

#[test]
fn test_config_with_decontextualize() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    assert!(!cfg.decontextualize);
}

#[test]
fn test_config_with_max_claims() {
    let cfg = ClaimDecompConfig::new().with_max_claims(7);
    assert_eq!(cfg.max_claims, 7);
}

#[test]
fn test_config_builder_chain() {
    let cfg = ClaimDecompConfig::new()
        .with_min_tokens(2)
        .with_decontextualize(false)
        .with_max_claims(10);
    assert_eq!(cfg.min_tokens, 2);
    assert!(!cfg.decontextualize);
    assert_eq!(cfg.max_claims, 10);
}

#[test]
fn test_config_clone_eq() {
    let cfg = ClaimDecompConfig::new().with_min_tokens(4);
    assert_eq!(cfg.clone(), cfg);
}

// ── AtomicClaim construction ──────────────────────────────────────────────────

#[test]
fn test_atomic_claim_new() {
    let claim = AtomicClaim::new("Einstein was born in Germany", 0, false);
    assert_eq!(claim.text, "Einstein was born in Germany");
    assert_eq!(claim.source_sentence, 0);
    assert!(!claim.decontextualized);
}

#[test]
fn test_atomic_claim_eq() {
    let a = AtomicClaim::new("fact", 1, true);
    let b = AtomicClaim::new("fact", 1, true);
    assert_eq!(a, b);
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

#[test]
fn test_tokenize_lowercases() {
    assert_eq!(tokenize("Hello World"), vec!["hello", "world"]);
}

#[test]
fn test_tokenize_drops_single_char() {
    // "a" has length 1 and is dropped; "is" and "ok" survive.
    assert_eq!(tokenize("a is ok"), vec!["is", "ok"]);
}

#[test]
fn test_tokenize_splits_non_alphanumeric() {
    assert_eq!(tokenize("born-in,Germany"), vec!["born", "in", "germany"]);
}

#[test]
fn test_tokenize_empty() {
    assert!(tokenize("").is_empty());
}

// ── Sentence splitter ─────────────────────────────────────────────────────────

#[test]
fn test_split_sentences_basic() {
    let s = split_sentences("One thing. Two things! Three things?");
    assert_eq!(s.len(), 3);
    assert_eq!(s[0], "One thing");
    assert_eq!(s[1], "Two things");
    assert_eq!(s[2], "Three things");
}

#[test]
fn test_split_sentences_single() {
    let s = split_sentences("Just one sentence");
    assert_eq!(s.len(), 1);
}

#[test]
fn test_split_sentences_drops_empty() {
    let s = split_sentences("A.. B.");
    assert_eq!(s, vec!["A", "B"]);
}

// ── Clause splitter ───────────────────────────────────────────────────────────

#[test]
fn test_split_clauses_and() {
    let clauses = split_clauses("Einstein was born in Germany and developed relativity");
    assert_eq!(clauses.len(), 2);
    assert_eq!(clauses[0], "Einstein was born in Germany");
    assert_eq!(clauses[1], "developed relativity");
}

#[test]
fn test_split_clauses_but() {
    let clauses = split_clauses("It is small but it is fast");
    assert_eq!(clauses.len(), 2);
}

#[test]
fn test_split_clauses_semicolon() {
    let clauses = split_clauses("First part; second part");
    assert_eq!(clauses.len(), 2);
}

#[test]
fn test_split_clauses_comma() {
    let clauses = split_clauses("Paris, the capital of France");
    assert_eq!(clauses.len(), 2);
}

#[test]
fn test_split_clauses_which() {
    let clauses = split_clauses("Water boils which produces steam");
    assert_eq!(clauses.len(), 2);
}

#[test]
fn test_split_clauses_no_boundary() {
    let clauses = split_clauses("A single clause here");
    assert_eq!(clauses.len(), 1);
}

// ── Compound-sentence decomposition ───────────────────────────────────────────

#[test]
fn test_compound_sentence_two_claims() {
    let extractor = HeuristicAtomicExtractor::default();
    // Both clauses clear the default min_tokens of 3.
    let claims =
        extractor.extract("Einstein was born in Germany and developed the theory of relativity.");
    assert!(claims.len() >= 2);
    assert_eq!(claims[0].text, "Einstein was born in Germany");
    assert_eq!(claims[1].text, "developed the theory of relativity");
}

#[test]
fn test_compound_sentence_via_decompose() {
    let extractor = AtomicClaimExtractor::default();
    let claims = extractor
        .decompose("Einstein was born in Germany and developed the theory of relativity.")
        .expect("non-empty answer");
    assert!(claims.len() >= 2);
}

#[test]
fn test_compound_sentence_short_clause_dropped_by_default() {
    // The canonical FActScore example: "developed relativity" is only two tokens,
    // so under the default min_tokens of 3 it is dropped, leaving one claim.
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Einstein was born in Germany and developed relativity.");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].text, "Einstein was born in Germany");
}

#[test]
fn test_compound_sentence_short_clause_kept_with_min_tokens_two() {
    // Lowering min_tokens to 2 keeps the canonical two-token second clause.
    let cfg = ClaimDecompConfig::new().with_min_tokens(2);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Einstein was born in Germany and developed relativity.");
    assert_eq!(claims.len(), 2);
    assert_eq!(claims[1].text, "developed relativity");
}

#[test]
fn test_three_clauses() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims =
        extractor.extract("The dog ran fast and the cat slept quietly but the bird sang loudly.");
    assert_eq!(claims.len(), 3);
}

// ── min_tokens dropping ───────────────────────────────────────────────────────

#[test]
fn test_min_tokens_drops_tiny_clause() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    // "Yes" tokenises to 1 token (< 3) and is dropped; the long clause survives.
    let claims = extractor.extract("Yes and the experiment confirmed the hypothesis clearly.");
    assert_eq!(claims.len(), 1);
    assert_eq!(
        claims[0].text,
        "the experiment confirmed the hypothesis clearly"
    );
}

#[test]
fn test_min_tokens_high_drops_all() {
    let cfg = ClaimDecompConfig::new().with_min_tokens(100);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Einstein was born in Germany.");
    assert!(claims.is_empty());
}

#[test]
fn test_min_tokens_low_keeps_short() {
    let cfg = ClaimDecompConfig::new()
        .with_min_tokens(1)
        .with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Yes and no.");
    // Both "Yes" and "no" survive at min_tokens = 1.
    assert_eq!(claims.len(), 2);
}

// ── Decontextualization ───────────────────────────────────────────────────────

#[test]
fn test_decontextualize_replaces_leading_pronoun() {
    let extractor = HeuristicAtomicExtractor::default();
    // Subject = "Einstein" (first capitalised multi-char token).
    let claims = extractor.extract("Einstein was a physicist. He developed relativity.");
    // Second sentence's clause begins with "He" → rewritten to "Einstein".
    let second = claims
        .iter()
        .find(|c| c.source_sentence == 1)
        .expect("second sentence claim");
    assert_eq!(second.text, "Einstein developed relativity");
    assert!(second.decontextualized);
}

#[test]
fn test_decontextualize_sets_flag_true() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Mercury is a planet. It orbits the Sun quickly.");
    let orbiting = claims
        .iter()
        .find(|c| c.text.contains("orbits"))
        .expect("orbit claim");
    assert!(orbiting.decontextualized);
    assert!(orbiting.text.starts_with("Mercury"));
}

#[test]
fn test_decontextualize_its() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Saturn is large. Its rings are spectacular today.");
    let rings = claims
        .iter()
        .find(|c| c.text.contains("rings"))
        .expect("rings claim");
    assert!(rings.decontextualized);
    assert!(rings.text.starts_with("Saturn"));
}

#[test]
fn test_decontextualize_they() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Whales are mammals. They breathe air through lungs.");
    let breathe = claims
        .iter()
        .find(|c| c.text.contains("breathe"))
        .expect("breathe claim");
    assert!(breathe.decontextualized);
    assert!(breathe.text.starts_with("Whales"));
}

#[test]
fn test_decontextualize_no_pronoun_unchanged() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Einstein developed the theory of relativity.");
    assert_eq!(claims.len(), 1);
    assert!(!claims[0].decontextualized);
    assert_eq!(
        claims[0].text,
        "Einstein developed the theory of relativity"
    );
}

#[test]
fn test_decontextualize_no_subject_unchanged() {
    // No capitalised multi-char token → no subject → pronoun stays.
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("it orbits the sun quickly today");
    assert_eq!(claims.len(), 1);
    assert!(!claims[0].decontextualized);
    assert!(claims[0].text.starts_with("it"));
}

#[test]
fn test_decontextualize_false_leaves_pronoun() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Einstein was a physicist. He developed relativity.");
    let second = claims
        .iter()
        .find(|c| c.source_sentence == 1)
        .expect("second sentence claim");
    assert!(!second.decontextualized);
    assert!(second.text.starts_with("He"));
}

#[test]
fn test_decontextualize_false_flag_always_false() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Mercury is small. It orbits quickly around the star.");
    assert!(claims.iter().all(|c| !c.decontextualized));
}

// ── max_claims cap ────────────────────────────────────────────────────────────

#[test]
fn test_max_claims_cap() {
    let cfg = ClaimDecompConfig::new()
        .with_max_claims(2)
        .with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims =
        extractor.extract("The dog ran and the cat slept and the bird sang and the fish swam.");
    assert_eq!(claims.len(), 2);
}

#[test]
fn test_max_claims_one() {
    let cfg = ClaimDecompConfig::new().with_max_claims(1);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Einstein was born in Germany and developed relativity.");
    assert_eq!(claims.len(), 1);
}

#[test]
fn test_max_claims_zero() {
    let cfg = ClaimDecompConfig::new().with_max_claims(0);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor.extract("Einstein was born in Germany and developed relativity.");
    assert!(claims.is_empty());
}

// ── source_sentence indices ───────────────────────────────────────────────────

#[test]
fn test_source_sentence_indices() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor
        .extract("Alpha runs quickly today. Beta walks slowly and gamma jumps high every morning.");
    // Sentence 0 → one clause; sentence 1 → two clauses.
    assert_eq!(claims[0].source_sentence, 0);
    assert_eq!(claims[1].source_sentence, 1);
    assert_eq!(claims[2].source_sentence, 1);
}

#[test]
fn test_source_sentence_monotonic() {
    let cfg = ClaimDecompConfig::new().with_decontextualize(false);
    let extractor = HeuristicAtomicExtractor::new(cfg);
    let claims = extractor
        .extract("First sentence here today. Second sentence here now. Third sentence here soon.");
    let indices: Vec<usize> = claims.iter().map(|c| c.source_sentence).collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

// ── Single simple sentence ⇒ 1 claim ──────────────────────────────────────────

#[test]
fn test_single_simple_sentence_one_claim() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Einstein developed the theory of relativity.");
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0].source_sentence, 0);
}

#[test]
fn test_single_sentence_no_terminator() {
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("Einstein developed the theory of relativity");
    assert_eq!(claims.len(), 1);
}

// ── decompose / count / errors ────────────────────────────────────────────────

#[test]
fn test_decompose_empty_answer_error() {
    let extractor = AtomicClaimExtractor::default();
    let result = extractor.decompose("");
    assert!(matches!(result, Err(ClaimDecompError::EmptyAnswer)));
}

#[test]
fn test_decompose_whitespace_answer_error() {
    let extractor = AtomicClaimExtractor::default();
    let result = extractor.decompose("   \n\t  ");
    assert!(matches!(result, Err(ClaimDecompError::EmptyAnswer)));
}

#[test]
fn test_count_matches_decompose_len() {
    let extractor = AtomicClaimExtractor::default();
    let answer = "Einstein was born in Germany and developed relativity.";
    let count = extractor.count(answer).expect("non-empty");
    let claims = extractor.decompose(answer).expect("non-empty");
    assert_eq!(count, claims.len());
}

#[test]
fn test_count_empty_answer_error() {
    let extractor = AtomicClaimExtractor::default();
    assert!(matches!(
        extractor.count(""),
        Err(ClaimDecompError::EmptyAnswer)
    ));
}

#[test]
fn test_count_single_sentence() {
    let extractor = AtomicClaimExtractor::default();
    assert_eq!(
        extractor
            .count("Einstein developed the theory of relativity.")
            .expect("non-empty"),
        1
    );
}

#[test]
fn test_new_propagates_config() {
    let cfg = ClaimDecompConfig::new()
        .with_max_claims(3)
        .with_min_tokens(2);
    let extractor = AtomicClaimExtractor::new(cfg.clone());
    assert_eq!(extractor.config, cfg);
    assert_eq!(extractor.extractor.config, cfg);
}

#[test]
fn test_error_display() {
    let e = ClaimDecompError::EmptyAnswer;
    assert_eq!(e.to_string(), "answer must not be empty");
}

// ── Determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_determinism() {
    let extractor = AtomicClaimExtractor::default();
    let answer = "Einstein was born in Germany and developed relativity. He won the Nobel Prize.";
    let first = extractor.decompose(answer).expect("non-empty");
    let second = extractor.decompose(answer).expect("non-empty");
    assert_eq!(first, second);
}

#[test]
fn test_determinism_repeated_runs() {
    let extractor = HeuristicAtomicExtractor::default();
    let answer = "Mercury is small. It orbits the Sun quickly and it has no moons.";
    let baseline = extractor.extract(answer);
    for _ in 0..16 {
        assert_eq!(extractor.extract(answer), baseline);
    }
}

// ── Trait object usage ────────────────────────────────────────────────────────

#[test]
fn test_extract_via_trait_object() {
    let extractor = HeuristicAtomicExtractor::default();
    let dyn_extractor: &dyn ClaimExtractor = &extractor;
    let claims = dyn_extractor
        .extract("Einstein was born in Germany and developed the theory of relativity.");
    assert!(claims.len() >= 2);
}

#[test]
fn test_empty_after_terminators_only() {
    // No real sentences → no claims (but not an error path for the heuristic).
    let extractor = HeuristicAtomicExtractor::default();
    let claims = extractor.extract("...");
    assert!(claims.is_empty());
}
