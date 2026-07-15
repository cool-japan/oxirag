//! Pillar 1 tests: heuristic rules **at the boundary**.
//!
//! For each rule, one document lands exactly *at* the threshold (must pass —
//! bounds are inclusive) and one lands one step past it (must fail). The
//! threshold value is written into each test, matching the default rule
//! configured by [`QualityRule::default_rule_set`].

use crate::corpus_curation::{QualityRule, evaluate_quality_rules};

fn only_passed(content: &str, rule: QualityRule) -> bool {
    evaluate_quality_rules(content, &[rule]).passed
}

#[test]
fn word_count_min_boundary_is_50() {
    let inside = (0..50)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let outside = (0..49)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(inside.split_whitespace().count(), 50);
    assert_eq!(outside.split_whitespace().count(), 49);

    let rule = QualityRule::WordCount {
        min: 50,
        max: 100_000,
    };
    assert!(
        only_passed(&inside, rule.clone()),
        "50 words must pass min=50"
    );
    assert!(!only_passed(&outside, rule), "49 words must fail min=50");
}

#[test]
fn word_count_max_boundary_is_10() {
    // A small max (rather than the production default of 100_000) keeps the
    // "just outside" fixture cheap; the boundary logic under test is
    // identical regardless of the threshold's magnitude.
    let inside = (0..10)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let outside = (0..11)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");

    let rule = QualityRule::WordCount { min: 1, max: 10 };
    assert!(
        only_passed(&inside, rule.clone()),
        "10 words must pass max=10"
    );
    assert!(!only_passed(&outside, rule), "11 words must fail max=10");
}

#[test]
fn mean_word_length_min_boundary_is_3() {
    let inside = ["cat"; 10].join(" "); // 3-char words -> mean 3.0
    let outside = ["at"; 10].join(" "); // 2-char words -> mean 2.0

    let rule = QualityRule::MeanWordLength {
        min: 3.0,
        max: 10.0,
    };
    assert!(
        only_passed(&inside, rule.clone()),
        "mean 3.0 must pass min=3.0"
    );
    assert!(!only_passed(&outside, rule), "mean 2.0 must fail min=3.0");
}

#[test]
fn mean_word_length_max_boundary_is_10() {
    let inside = ["abcdefghij"; 10].join(" "); // 10-char words -> mean 10.0
    let outside = ["abcdefghijk"; 10].join(" "); // 11-char words -> mean 11.0

    let rule = QualityRule::MeanWordLength {
        min: 3.0,
        max: 10.0,
    };
    assert!(
        only_passed(&inside, rule.clone()),
        "mean 10.0 must pass max=10.0"
    );
    assert!(!only_passed(&outside, rule), "mean 11.0 must fail max=10.0");
}

#[test]
fn symbol_to_word_ratio_boundary_is_0_1() {
    // 20 tokens total; 2 are `#` symbols -> ratio 0.10; 3 -> ratio 0.15.
    let inside = format!("{} # #", vec!["w"; 18].join(" "));
    let outside = format!("{} # # #", vec!["w"; 17].join(" "));
    assert_eq!(inside.split_whitespace().count(), 20);
    assert_eq!(outside.split_whitespace().count(), 20);

    let rule = QualityRule::SymbolToWordRatio { max: 0.1 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.10 must pass max=0.1"
    );
    assert!(!only_passed(&outside, rule), "ratio 0.15 must fail max=0.1");
}

#[test]
fn stopword_ratio_boundary_is_0_06() {
    let filler = |n: usize| {
        (0..n)
            .map(|i| format!("token{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    // 100 tokens total; 6 stop words -> ratio 0.06; 5 -> ratio 0.05.
    let inside = format!("{} {}", filler(94), ["the"; 6].join(" "));
    let outside = format!("{} {}", filler(95), ["the"; 5].join(" "));
    assert_eq!(inside.split_whitespace().count(), 100);
    assert_eq!(outside.split_whitespace().count(), 100);

    let rule = QualityRule::StopWordRatio { min: 0.06 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.06 must pass min=0.06"
    );
    assert!(
        !only_passed(&outside, rule),
        "ratio 0.05 must fail min=0.06"
    );
}

#[test]
fn duplicate_line_ratio_boundary_is_0_3() {
    // 100 lines total. Inside: 70 unique + 30 repeats -> ratio 0.30.
    let uniques: Vec<String> = (0..70).map(|i| format!("line{i}")).collect();
    let mut inside_lines = uniques.clone();
    inside_lines.extend(uniques[..30].iter().cloned());
    let inside = inside_lines.join("\n");

    // Outside: 69 unique + 31 repeats -> ratio 0.31.
    let uniques69: Vec<String> = (0..69).map(|i| format!("line{i}")).collect();
    let mut outside_lines = uniques69.clone();
    outside_lines.extend(uniques69[..31].iter().cloned());
    let outside = outside_lines.join("\n");

    assert_eq!(inside.lines().count(), 100);
    assert_eq!(outside.lines().count(), 100);

    let rule = QualityRule::DuplicateLineRatio { max: 0.3 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.30 must pass max=0.3"
    );
    assert!(!only_passed(&outside, rule), "ratio 0.31 must fail max=0.3");
}

#[test]
fn bullet_line_ratio_boundary_is_0_9() {
    // 100 lines total. Inside: 90 bulleted -> ratio 0.90.
    let mut inside_lines: Vec<String> = (0..90).map(|i| format!("- bullet {i}")).collect();
    inside_lines.extend((0..10).map(|i| format!("plain {i}")));
    let inside = inside_lines.join("\n");

    // Outside: 91 bulleted -> ratio 0.91.
    let mut outside_lines: Vec<String> = (0..91).map(|i| format!("- bullet {i}")).collect();
    outside_lines.extend((0..9).map(|i| format!("plain {i}")));
    let outside = outside_lines.join("\n");

    assert_eq!(inside.lines().count(), 100);
    assert_eq!(outside.lines().count(), 100);

    let rule = QualityRule::BulletLineRatio { max: 0.9 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.90 must pass max=0.9"
    );
    assert!(!only_passed(&outside, rule), "ratio 0.91 must fail max=0.9");
}

#[test]
fn ellipsis_line_ratio_boundary_is_0_3() {
    // 100 lines total. Inside: 30 ending in "..." -> ratio 0.30.
    let mut inside_lines: Vec<String> = (0..30).map(|i| format!("teaser {i}...")).collect();
    inside_lines.extend((0..70).map(|i| format!("plain {i}")));
    let inside = inside_lines.join("\n");

    // Outside: 31 ending in "..." -> ratio 0.31.
    let mut outside_lines: Vec<String> = (0..31).map(|i| format!("teaser {i}...")).collect();
    outside_lines.extend((0..69).map(|i| format!("plain {i}")));
    let outside = outside_lines.join("\n");

    assert_eq!(inside.lines().count(), 100);
    assert_eq!(outside.lines().count(), 100);

    let rule = QualityRule::EllipsisLineRatio { max: 0.3 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.30 must pass max=0.3"
    );
    assert!(!only_passed(&outside, rule), "ratio 0.31 must fail max=0.3");
}

#[test]
fn alphabetic_char_ratio_boundary_is_0_6() {
    // 100 non-whitespace chars total. Inside: 60 alphabetic -> ratio 0.60.
    let inside = format!("{}{}", "a".repeat(60), "1".repeat(40));
    // Outside: 59 alphabetic -> ratio 0.59.
    let outside = format!("{}{}", "a".repeat(59), "1".repeat(41));

    let rule = QualityRule::AlphabeticCharRatio { min: 0.6 };
    assert!(
        only_passed(&inside, rule.clone()),
        "ratio 0.60 must pass min=0.6"
    );
    assert!(!only_passed(&outside, rule), "ratio 0.59 must fail min=0.6");
}

#[test]
fn empty_rule_set_is_a_vacuous_pass() {
    let verdict = evaluate_quality_rules("anything at all, of any shape", &[]);
    assert!(verdict.passed);
    assert!(verdict.signals.is_empty());
    assert!((verdict.pass_rate() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn failed_signals_reports_exactly_the_rules_that_failed() {
    let content = "two words"; // fails WordCount(min 50), passes everything else
    let rules = vec![
        QualityRule::WordCount {
            min: 50,
            max: 100_000,
        },
        QualityRule::AlphabeticCharRatio { min: 0.0 },
    ];
    let verdict = evaluate_quality_rules(content, &rules);
    assert!(!verdict.passed);
    let failed: Vec<&str> = verdict.failed_signals().map(|s| s.name).collect();
    assert_eq!(failed, vec!["word_count"]);
    assert!((verdict.pass_rate() - 0.5).abs() < f64::EPSILON);
}
