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
//! Tests for the `quote_grounding` module.

use crate::quote_grounding::grounder::QuoteGrounder;
use crate::quote_grounding::types::{GroundedQuote, QuoteConfig, QuoteError};
use crate::types::{Document, DocumentId};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_doc(id: &str, content: &str) -> Document {
    Document::new(content).with_id(DocumentId::from_string(id))
}

/// Word-token count after whitespace splitting.
fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Normalise text to a whitespace-collapsed form with sentence-terminating
/// punctuation removed, so verbatim spans can be compared across the sentence
/// boundaries dropped by the splitter.
fn normalise(text: &str) -> String {
    text.split(['.', '!', '?'])
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

/// True when `needle` occurs as a contiguous whitespace-collapsed slice of some
/// document's content (the verbatim-span invariant). The quote is a contiguous
/// run of source words with the terminating `.`/`!`/`?` removed, so both sides
/// are normalised before the window comparison.
fn is_verbatim_span(needle: &str, docs: &[Document]) -> bool {
    let needle_norm = normalise(needle);
    let needle_words: Vec<&str> = needle_norm.split_whitespace().collect();
    if needle_words.is_empty() {
        return false;
    }
    docs.iter().any(|d| {
        let hay_norm = normalise(&d.content);
        let hay: Vec<&str> = hay_norm.split_whitespace().collect();
        hay.windows(needle_words.len())
            .any(|w| w == needle_words.as_slice())
    })
}

// ── QuoteConfig defaults ─────────────────────────────────────────────────────

#[test]
fn test_config_default_min_support() {
    assert_eq!(QuoteConfig::default().min_support, 0.3);
}

#[test]
fn test_config_default_max_quote_tokens() {
    assert_eq!(QuoteConfig::default().max_quote_tokens, 30);
}

#[test]
fn test_config_new_equals_default() {
    assert_eq!(QuoteConfig::new(), QuoteConfig::default());
}

// ── QuoteConfig builders ─────────────────────────────────────────────────────

#[test]
fn test_config_with_min_support() {
    let c = QuoteConfig::new().with_min_support(0.75);
    assert_eq!(c.min_support, 0.75);
    assert_eq!(c.max_quote_tokens, 30);
}

#[test]
fn test_config_with_max_quote_tokens() {
    let c = QuoteConfig::new().with_max_quote_tokens(5);
    assert_eq!(c.max_quote_tokens, 5);
    assert_eq!(c.min_support, 0.3);
}

#[test]
fn test_config_builders_chain() {
    let c = QuoteConfig::new()
        .with_min_support(0.5)
        .with_max_quote_tokens(12);
    assert_eq!(c.min_support, 0.5);
    assert_eq!(c.max_quote_tokens, 12);
}

#[test]
fn test_config_clone_eq() {
    let c = QuoteConfig::new().with_min_support(0.42);
    assert_eq!(c.clone(), c);
}

// ── GroundedQuote constructor ────────────────────────────────────────────────

#[test]
fn test_grounded_quote_new_fields() {
    let q = GroundedQuote::new("claim", "quote", "doc-1", 0.5);
    assert_eq!(q.claim, "claim");
    assert_eq!(q.quote, "quote");
    assert_eq!(q.source_id, DocumentId::from_string("doc-1"));
    assert_eq!(q.score, 0.5);
    assert_eq!(q.clone(), q);
}

// ── best_quote: finds supporting sentence ────────────────────────────────────

#[test]
fn test_best_quote_finds_supporting_sentence() {
    let docs = vec![make_doc(
        "d1",
        "Rust is a systems programming language. Python is dynamically typed.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Rust is a systems programming language", &docs)
        .expect("should ground");
    assert_eq!(q.quote, "Rust is a systems programming language");
}

#[test]
fn test_best_quote_picks_correct_source_id() {
    let docs = vec![
        make_doc("alpha", "The sky is blue today."),
        make_doc(
            "beta",
            "Rust guarantees memory safety without a garbage collector.",
        ),
    ];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Rust guarantees memory safety", &docs)
        .expect("should ground");
    assert_eq!(q.source_id, DocumentId::from_string("beta"));
}

#[test]
fn test_best_quote_selects_max_overlap_sentence() {
    let docs = vec![make_doc(
        "d1",
        "Cats are mammals. The Eiffel Tower is in Paris France.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("The Eiffel Tower is in Paris", &docs)
        .expect("should ground");
    assert_eq!(q.quote, "The Eiffel Tower is in Paris France");
}

#[test]
fn test_best_quote_is_substring_of_source() {
    let docs = vec![make_doc(
        "d1",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Photosynthesis converts sunlight into energy", &docs)
        .expect("should ground");
    assert!(is_verbatim_span(&q.quote, &docs));
}

#[test]
fn test_best_quote_across_multiple_docs() {
    let docs = vec![
        make_doc("d1", "Water boils at one hundred degrees Celsius."),
        make_doc("d2", "Mercury is the closest planet to the sun."),
        make_doc("d3", "Honey never spoils when stored correctly."),
    ];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Mercury is the closest planet", &docs)
        .expect("should ground");
    assert_eq!(q.source_id, DocumentId::from_string("d2"));
    assert!(is_verbatim_span(&q.quote, &docs));
}

// ── best_quote: min_support gating ───────────────────────────────────────────

#[test]
fn test_best_quote_none_when_no_overlap() {
    let docs = vec![make_doc("d1", "The ocean is vast and deep.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    assert!(
        g.best_quote("Quantum entanglement defies locality", &docs)
            .is_none()
    );
}

#[test]
fn test_best_quote_none_below_threshold() {
    // Only one of four claim tokens appears -> coverage 0.25 < 0.3.
    let docs = vec![make_doc("d1", "Bicycles have wheels.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    assert!(
        g.best_quote("Bicycles fly through outer space", &docs)
            .is_none()
    );
}

#[test]
fn test_best_quote_some_at_threshold_boundary() {
    // Two of four claim tokens present -> coverage 0.5 >= 0.5.
    let docs = vec![make_doc("d1", "Bicycles have two wheels.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.5));
    let q = g.best_quote("Bicycles need wheels often", &docs);
    assert!(q.is_some());
}

#[test]
fn test_best_quote_high_threshold_rejects_partial() {
    let docs = vec![make_doc("d1", "Rust is a systems programming language.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.99));
    assert!(
        g.best_quote("Rust is used for web frontends", &docs)
            .is_none()
    );
}

#[test]
fn test_best_quote_empty_docs_none() {
    let g = QuoteGrounder::new(QuoteConfig::default());
    assert!(g.best_quote("anything at all", &[]).is_none());
}

#[test]
fn test_best_quote_empty_claim_none() {
    let docs = vec![make_doc("d1", "Some meaningful content here.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    assert!(g.best_quote("", &docs).is_none());
}

#[test]
fn test_best_quote_punctuation_only_claim_none() {
    let docs = vec![make_doc("d1", "Some meaningful content here.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    assert!(g.best_quote("!!! ... ???", &docs).is_none());
}

// ── best_quote: max_quote_tokens trimming ────────────────────────────────────

#[test]
fn test_trim_long_sentence_to_max_tokens() {
    let long = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                quantum entanglement defies classical locality nu xi omicron pi rho sigma tau";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(6)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("quantum entanglement defies locality", &docs)
        .expect("should ground");
    assert!(word_count(&q.quote) <= 6, "quote was: {:?}", q.quote);
}

#[test]
fn test_trim_keeps_overlapping_span() {
    let long = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu \
                quantum entanglement defies classical locality nu xi omicron pi rho sigma tau";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(6)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("quantum entanglement defies locality", &docs)
        .expect("should ground");
    assert!(
        q.quote.contains("quantum") && q.quote.contains("entanglement"),
        "quote was: {:?}",
        q.quote
    );
}

#[test]
fn test_trim_window_is_verbatim_span() {
    let long = "one two three four five six seven eight nine ten eleven twelve \
                photosynthesis converts sunlight into chemical energy thirteen fourteen \
                fifteen sixteen seventeen eighteen nineteen twenty";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(7)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("photosynthesis converts sunlight energy", &docs)
        .expect("should ground");
    assert!(
        is_verbatim_span(&q.quote, &docs),
        "quote was: {:?}",
        q.quote
    );
    assert!(word_count(&q.quote) <= 7);
}

#[test]
fn test_no_trim_when_sentence_short() {
    let docs = vec![make_doc("d1", "Rust is fast and safe.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_max_quote_tokens(30));
    let q = g
        .best_quote("Rust is fast and safe", &docs)
        .expect("should ground");
    assert_eq!(q.quote, "Rust is fast and safe");
}

#[test]
fn test_trim_window_exact_max_tokens_unchanged() {
    // Sentence has exactly max_quote_tokens words -> returned unchanged.
    let docs = vec![make_doc("d1", "one two three four five.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_max_quote_tokens(5));
    let q = g
        .best_quote("one two three four five", &docs)
        .expect("should ground");
    assert_eq!(q.quote, "one two three four five");
    assert_eq!(word_count(&q.quote), 5);
}

#[test]
fn test_trim_window_centers_on_match_at_end() {
    // The matched tokens are clustered contiguously near the end; the centred
    // window must capture them and drop the leading filler.
    let long = "filler filler filler filler filler filler filler filler \
                supernova explosions enrich nebulae greatly";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(5)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("supernova explosions enrich nebulae", &docs)
        .expect("should ground");
    // The full matched span is captured by the centred window.
    assert!(q.quote.contains("supernova"), "quote was: {:?}", q.quote);
    assert!(q.quote.contains("explosions"), "quote was: {:?}", q.quote);
    assert!(q.quote.contains("nebulae"), "quote was: {:?}", q.quote);
    assert!(word_count(&q.quote) <= 5);
}

#[test]
fn test_trim_window_centers_on_wide_span_midpoint() {
    // When the matched span is wide, the window is centred on its midpoint and
    // need not contain the extreme matched words — but stays verbatim and short.
    let long = "filler filler filler filler filler filler filler filler \
                supernova explosions enrich the interstellar medium with heavy elements";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(5)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("supernova explosions enrich elements", &docs)
        .expect("should ground");
    assert!(word_count(&q.quote) <= 5, "quote was: {:?}", q.quote);
    assert!(
        is_verbatim_span(&q.quote, &docs),
        "quote was: {:?}",
        q.quote
    );
    assert!(!q.quote.contains("filler"), "quote was: {:?}", q.quote);
}

// ── score range ──────────────────────────────────────────────────────────────

#[test]
fn test_score_one_for_full_overlap() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Rust guarantees memory safety", &docs)
        .expect("should ground");
    assert_eq!(q.score, 1.0);
}

#[test]
fn test_score_half_for_half_overlap() {
    // Claim tokens: rust, fast, web, frontends (4). Source shares rust, fast.
    let docs = vec![make_doc("d1", "Rust is fast.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.0));
    let q = g
        .best_quote("Rust fast web frontends", &docs)
        .expect("should ground");
    assert_eq!(q.score, 0.5);
}

#[test]
fn test_score_monotonic_with_more_overlap() {
    let docs_low = vec![make_doc("d1", "Rust is here.")];
    let docs_high = vec![make_doc("d1", "Rust is fast and safe here.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.0));
    let low = g
        .best_quote("Rust is fast and safe", &docs_low)
        .unwrap()
        .score;
    let high = g
        .best_quote("Rust is fast and safe", &docs_high)
        .unwrap()
        .score;
    assert!(high > low);
}

// ── ground ───────────────────────────────────────────────────────────────────

#[test]
fn test_ground_one_quote_per_supported_claim() {
    let docs = vec![make_doc(
        "d1",
        "Rust guarantees memory safety. Python is dynamically typed.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground(
            "Rust guarantees memory safety. Python is dynamically typed.",
            &docs,
        )
        .expect("ok");
    assert_eq!(quotes.len(), 2);
}

#[test]
fn test_ground_skips_unsupported_claims() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground(
            "Rust guarantees memory safety. Dragons breathe purple fire constantly.",
            &docs,
        )
        .expect("ok");
    assert_eq!(quotes.len(), 1);
    assert_eq!(quotes[0].claim, "Rust guarantees memory safety");
}

#[test]
fn test_ground_preserves_claim_order() {
    let docs = vec![make_doc(
        "d1",
        "Apples are red. Bananas are yellow. Grapes are green.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground(
            "Apples are red. Bananas are yellow. Grapes are green.",
            &docs,
        )
        .expect("ok");
    let claims: Vec<&str> = quotes.iter().map(|q| q.claim.as_str()).collect();
    assert_eq!(
        claims,
        vec!["Apples are red", "Bananas are yellow", "Grapes are green"]
    );
}

#[test]
fn test_ground_quotes_are_verbatim_spans() {
    let docs = vec![
        make_doc("d1", "The mitochondria is the powerhouse of the cell."),
        make_doc("d2", "DNA carries the genetic instructions for life."),
    ];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground(
            "The mitochondria is the powerhouse. DNA carries genetic instructions.",
            &docs,
        )
        .expect("ok");
    assert!(!quotes.is_empty());
    for q in &quotes {
        assert!(
            is_verbatim_span(&q.quote, &docs),
            "quote was: {:?}",
            q.quote
        );
    }
}

#[test]
fn test_ground_empty_when_no_support() {
    let docs = vec![make_doc("d1", "The ocean is vast.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground("Neutron stars collapse under immense gravity.", &docs)
        .expect("ok");
    assert!(quotes.is_empty());
}

#[test]
fn test_ground_single_claim_no_terminator() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground("Rust guarantees memory safety", &docs)
        .expect("ok");
    assert_eq!(quotes.len(), 1);
}

#[test]
fn test_ground_empty_answer_errors() {
    let docs = vec![make_doc("d1", "content")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let err = g.ground("   ", &docs).unwrap_err();
    assert!(matches!(err, QuoteError::EmptyAnswer));
}

#[test]
fn test_ground_score_field_within_range() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety always.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let quotes = g
        .ground("Rust guarantees memory safety.", &docs)
        .expect("ok");
    for q in &quotes {
        assert!(q.score >= 0.0 && q.score <= 1.0);
        assert!(q.score >= g.config.min_support);
    }
}

// ── ungrounded ───────────────────────────────────────────────────────────────

#[test]
fn test_ungrounded_lists_unsupported_claims() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let un = g
        .ungrounded(
            "Rust guarantees memory safety. Dragons breathe purple fire constantly.",
            &docs,
        )
        .expect("ok");
    assert_eq!(un, vec!["Dragons breathe purple fire constantly"]);
}

#[test]
fn test_ungrounded_empty_when_all_supported() {
    let docs = vec![make_doc(
        "d1",
        "Rust guarantees memory safety. Python is dynamically typed.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let un = g
        .ungrounded(
            "Rust guarantees memory safety. Python is dynamically typed.",
            &docs,
        )
        .expect("ok");
    assert!(un.is_empty());
}

#[test]
fn test_ungrounded_all_when_no_docs() {
    let g = QuoteGrounder::new(QuoteConfig::default());
    let un = g
        .ungrounded("First claim here. Second claim there.", &[])
        .expect("ok");
    assert_eq!(un.len(), 2);
}

#[test]
fn test_ungrounded_empty_answer_errors() {
    let docs = vec![make_doc("d1", "content")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let err = g.ungrounded("", &docs).unwrap_err();
    assert!(matches!(err, QuoteError::EmptyAnswer));
}

#[test]
fn test_ground_and_ungrounded_partition_claims() {
    let docs = vec![make_doc("d1", "Rust guarantees memory safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let answer = "Rust guarantees memory safety. Unicorns roam the misty valleys.";
    let grounded = g.ground(answer, &docs).expect("ok").len();
    let ungrounded = g.ungrounded(answer, &docs).expect("ok").len();
    assert_eq!(grounded + ungrounded, 2);
}

// ── verbatim invariant (explicit) ────────────────────────────────────────────

#[test]
fn test_quote_is_verbatim_even_when_trimmed() {
    let long = "intro intro intro intro intro intro intro intro intro intro \
                tigers are the largest living cat species on earth \
                outro outro outro outro outro outro outro outro";
    let docs = vec![make_doc("d1", long)];
    let g = QuoteGrounder::new(
        QuoteConfig::new()
            .with_max_quote_tokens(8)
            .with_min_support(0.1),
    );
    let q = g
        .best_quote("tigers are the largest cat species", &docs)
        .expect("should ground");
    assert!(
        is_verbatim_span(&q.quote, &docs),
        "quote was: {:?}",
        q.quote
    );
}

#[test]
fn test_quote_never_invents_tokens() {
    let docs = vec![make_doc("d1", "Volcanoes erupt molten lava.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("Volcanoes erupt lava", &docs)
        .expect("should ground");
    // Every word of the quote must be drawn from the source content (compared
    // after stripping terminating punctuation dropped by the splitter).
    let src_norm = normalise(&docs[0].content);
    let src_words: Vec<&str> = src_norm.split_whitespace().collect();
    for w in q.quote.split_whitespace() {
        assert!(src_words.contains(&w), "invented word: {w}");
    }
}

// ── determinism ──────────────────────────────────────────────────────────────

#[test]
fn test_best_quote_deterministic() {
    let docs = vec![
        make_doc("d1", "Rust is a systems programming language."),
        make_doc("d2", "Rust prevents data races at compile time."),
    ];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let a = g.best_quote("Rust prevents data races", &docs);
    let b = g.best_quote("Rust prevents data races", &docs);
    assert_eq!(a, b);
}

#[test]
fn test_ground_deterministic() {
    let docs = vec![make_doc(
        "d1",
        "Saturn has prominent rings. Jupiter is the largest planet.",
    )];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let answer = "Saturn has prominent rings. Jupiter is the largest planet.";
    let a = g.ground(answer, &docs).expect("ok");
    let b = g.ground(answer, &docs).expect("ok");
    assert_eq!(a, b);
}

#[test]
fn test_ungrounded_deterministic() {
    let docs = vec![make_doc("d1", "Saturn has prominent rings.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let answer = "Saturn has rings. Pluto was reclassified as a dwarf.";
    let a = g.ungrounded(answer, &docs).expect("ok");
    let b = g.ungrounded(answer, &docs).expect("ok");
    assert_eq!(a, b);
}

// ── error display ────────────────────────────────────────────────────────────

#[test]
fn test_empty_answer_error_message() {
    assert_eq!(
        QuoteError::EmptyAnswer.to_string(),
        "answer must not be empty"
    );
}

// ── miscellaneous behaviour ──────────────────────────────────────────────────

#[test]
fn test_tokenizer_ignores_short_tokens() {
    // "is", "of", "a" are length < 3 but >= 2 retained; single-char dropped.
    // Claim has tokens a(1-dropped) systems language. Source must share to ground.
    let docs = vec![make_doc("d1", "Systems language design matters.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g.best_quote("a systems language", &docs).expect("ground");
    assert!(q.score > 0.0);
}

#[test]
fn test_case_insensitive_matching() {
    let docs = vec![make_doc("d1", "RUST guarantees MEMORY safety.")];
    let g = QuoteGrounder::new(QuoteConfig::default());
    let q = g
        .best_quote("rust guarantees memory safety", &docs)
        .expect("ground");
    assert_eq!(q.score, 1.0);
}

#[test]
fn test_grounder_clone() {
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.6));
    let g2 = g.clone();
    assert_eq!(g.config, g2.config);
}

#[test]
fn test_best_quote_prefers_strictly_higher_score() {
    // First doc has a weaker sentence; second has a stronger match. The stronger
    // one must win regardless of document order.
    let docs = vec![
        make_doc("weak", "Rust exists."),
        make_doc("strong", "Rust prevents data races at compile time."),
    ];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.0));
    let q = g.best_quote("Rust prevents data races", &docs).unwrap();
    assert_eq!(q.source_id, DocumentId::from_string("strong"));
}

#[test]
fn test_first_max_wins_on_tie() {
    // Two sentences with identical overlap; the first encountered must win
    // (strict `>` comparison keeps the earliest maximum).
    let docs = vec![make_doc("d1", "Rust is safe. Rust is safe again here.")];
    let g = QuoteGrounder::new(QuoteConfig::new().with_min_support(0.0));
    let q = g.best_quote("Rust is safe", &docs).unwrap();
    assert_eq!(q.quote, "Rust is safe");
}
