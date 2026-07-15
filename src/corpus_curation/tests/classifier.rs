//! Pillar 2 tests: the classifier must **beat a baseline on a known
//! generating process**, measured on held-out data.

use crate::corpus_curation::{
    ClassifierConfig, ClassifierExample, CurationError, CurationRng, QualityClassifier,
    train_test_split,
};

use super::fixtures::{BAD_VOCAB, CONNECTIVES, GOOD_VOCAB};

/// Draw one 40-word example from `is_good`'s topic vocabulary, with a 25%
/// chance per content word of drawing from the *other* class's vocabulary
/// instead — simulated label noise / mixed content, leaving a real but
/// imperfect (75% vs 25%) majority signal so individual documents are
/// genuinely ambiguous and the task is not trivially, perfectly separable.
fn make_example(rng: &mut CurationRng, is_good: bool) -> ClassifierExample {
    let (primary, secondary) = if is_good {
        (GOOD_VOCAB, BAD_VOCAB)
    } else {
        (BAD_VOCAB, GOOD_VOCAB)
    };
    let mut words: Vec<&str> = Vec::with_capacity(40);
    for _ in 0..40 {
        if rng.next_bool(0.35)
            && let Some(w) = rng.choose(CONNECTIVES)
        {
            words.push(w);
            continue;
        }
        let vocab = if rng.next_bool(0.25) {
            secondary
        } else {
            primary
        };
        if let Some(w) = rng.choose(vocab) {
            words.push(w);
        }
    }
    let text = words.join(" ");
    if is_good {
        ClassifierExample::good(text)
    } else {
        ClassifierExample::bad(text)
    }
}

/// (b) Draw 300 "good" and 300 "bad" documents from two distinct word
/// distributions, train on a 70% split, and measure held-out accuracy
/// against a majority-class baseline computed from the *same* test set.
#[test]
fn classifier_beats_majority_baseline_on_held_out_split() {
    let mut generator = CurationRng::new(0xABCD_1234_5678);
    let mut examples = Vec::with_capacity(600);
    for _ in 0..300 {
        examples.push(make_example(&mut generator, true));
    }
    for _ in 0..300 {
        examples.push(make_example(&mut generator, false));
    }

    let mut split_rng = CurationRng::new(42);
    let (train, test) = train_test_split(&examples, 0.7, &mut split_rng);
    assert!(train.len() > 300, "train split should be the larger share");
    assert!(!test.is_empty());

    let config = ClassifierConfig::default();
    let model = QualityClassifier::train(&train, &config).expect("training succeeds");
    assert!(
        model.final_loss < model.loss_history[0],
        "loss should decrease during training: final {} vs initial {}",
        model.final_loss,
        model.loss_history[0]
    );

    let eval = model.evaluate(&test);

    let positives = test.iter().filter(|e| e.label).count();
    let negatives = test.len() - positives;
    #[allow(clippy::cast_precision_loss)]
    let majority_baseline = positives.max(negatives) as f64 / test.len() as f64;
    eprintln!(
        "[classifier measurement] n_test={} accuracy={:.4} precision={:.4} recall={:.4} f1={:.4} \
         majority_baseline={:.4} initial_loss={:.4} final_loss={:.4}",
        eval.n,
        eval.accuracy,
        eval.precision,
        eval.recall,
        eval.f1,
        majority_baseline,
        model.loss_history[0],
        model.final_loss
    );

    // Measured on this (fully deterministic, seeded) run: accuracy 0.9056,
    // majority baseline 0.5167 (the split is near-balanced but not exactly,
    // since the shuffle-split draws independently per class), a margin of
    // 0.3889. Thresholds below leave headroom below the observed values.
    assert!(
        eval.accuracy > 0.85,
        "held-out accuracy {} should exceed 0.85 (n={})",
        eval.accuracy,
        eval.n
    );
    assert!(
        eval.accuracy - majority_baseline > 0.3,
        "held-out accuracy {} should beat majority baseline {} by > 0.3",
        eval.accuracy,
        majority_baseline
    );
}

#[test]
fn train_rejects_empty_example_set() {
    let err = QualityClassifier::train(&[], &ClassifierConfig::default()).unwrap_err();
    assert_eq!(err, CurationError::EmptyTrainingSet);
}

#[test]
fn train_rejects_invalid_config() {
    let config = ClassifierConfig::default().with_epochs(0);
    let examples = vec![ClassifierExample::good("a"), ClassifierExample::bad("b")];
    let err = QualityClassifier::train(&examples, &config).unwrap_err();
    assert!(matches!(err, CurationError::InvalidConfig(_)));
}

#[test]
fn deterministic_training_is_reproducible() {
    let mut rng = CurationRng::new(7);
    let examples: Vec<ClassifierExample> = (0..40)
        .map(|i| make_example(&mut rng, i % 2 == 0))
        .collect();
    let config = ClassifierConfig::default();
    let model_a = QualityClassifier::train(&examples, &config).expect("training succeeds");
    let model_b = QualityClassifier::train(&examples, &config).expect("training succeeds");
    assert_eq!(model_a.weights, model_b.weights);
    assert!((model_a.bias - model_b.bias).abs() < f64::EPSILON);
}
