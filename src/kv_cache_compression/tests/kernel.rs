//! Tests for the attention kernel itself and for the accumulation of the
//! per-token attention statistics the eviction policies rank by.
//!
//! These come first, and are the strictest tests in the module: if the attention
//! is wrong, every eviction policy built on it is ranking noise.

use crate::kv_cache_compression::{
    KvAttentionStats, KvCacheTensor, KvCompressionError, KvScoreNormalization,
    scaled_dot_product_attention,
};

use super::SplitMix64;
// ═════════════════════════════════════════════════════════════════════════════
// 1. THE ATTENTION KERNEL
// ═════════════════════════════════════════════════════════════════════════════

/// A single hand-computed attention example, checked to the last decimal.
///
/// Geometry: 1 layer, 1 head, `d_head = 2`, 2 cached tokens.
///
/// ```text
/// k_0 = [1, 0]   v_0 = [1, 0]
/// k_1 = [0, 1]   v_1 = [0, 1]
/// q   = [sqrt(2)·ln 3, 0]           (the same query issued at positions 0 and 1)
/// ```
///
/// The scale is `1 / sqrt(d_head) = 1 / sqrt(2)`, so
///
/// ```text
/// logit_0 = (q · k_0) / sqrt(2) = (sqrt(2)·ln 3) / sqrt(2) = ln 3
/// logit_1 = (q · k_1) / sqrt(2) = 0 / sqrt(2)              = 0
/// ```
///
/// **Query at position 0.** The causal mask hides token 1 (its position, 1, is
/// greater than 0), so the softmax runs over `{token 0}` alone:
///
/// ```text
/// A[0] = [1, 0]                 (a softmax over one element is 1)
/// o[0] = 1·v_0 = [1, 0]
/// ```
///
/// **Query at position 1.** Both tokens are visible:
///
/// ```text
/// exp(ln 3) = 3,  exp(0) = 1,  denominator = 4
/// A[1] = [3/4, 1/4] = [0.75, 0.25]
/// o[1] = 0.75·[1, 0] + 0.25·[0, 1] = [0.75, 0.25]
/// ```
///
/// # On the tolerance
///
/// The *ideal* weights are the exact rationals `3/4` and `1/4`, but they cannot
/// be hit to `f64` precision through an `f32` cache, and no correct kernel could:
/// producing them requires the logit gap to be exactly `ln 3`, which requires the
/// query to be exactly `sqrt(2)·ln 3` — an irrational that `f32` can only hold to
/// a relative error of ~`2^-24`. Propagating that through the softmax
/// (`dw/dδ = w(1-w) = 0.1875`) bounds the achievable weight error at
/// `0.1875 · ln 3 · 2^-24 · (a few ulps) ≈ 4e-8`.
///
/// So this test asserts **both** halves, and neither is a weakened assertion:
///
/// - the kernel reproduces, to `1e-12`, an *independent* `f64` softmax computed
///   here from the same `f32`-rounded inputs — this pins the arithmetic exactly;
/// - and that result sits within `1e-7` of the ideal `3/4` — this pins the
///   arithmetic against the *hand-derived* answer, to the precision `f32` inputs
///   physically permit.
#[test]
fn attention_matches_a_hand_computed_example_exactly() {
    let mut cache = KvCacheTensor::new(1, 1, 2).expect("valid geometry");
    cache
        .append_token(&[1.0, 0.0], &[1.0, 0.0])
        .expect("well-formed token 0");
    cache
        .append_token(&[0.0, 1.0], &[0.0, 1.0])
        .expect("well-formed token 1");

    let gain = std::f32::consts::SQRT_2 * 3.0f32.ln();
    // [head][query][dim] with one head and two queries.
    let queries = vec![gain, 0.0, gain, 0.0];
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &[0, 1]).expect("well-formed call");

    let row_at_0 = attention.weight_row(0, 0).expect("row exists");
    assert!(
        (row_at_0[0] - 1.0).abs() < 1e-12,
        "a softmax over a single visible key must be exactly 1, got {}",
        row_at_0[0]
    );
    assert_eq!(
        row_at_0[1], 0.0,
        "the causal mask must give the future token exactly zero weight"
    );
    let output_at_0 = attention.output_row(0, 0).expect("row exists");
    assert!((output_at_0[0] - 1.0).abs() < 1e-6);
    assert!((output_at_0[1] - 0.0).abs() < 1e-6);

    let row_at_1 = attention.weight_row(0, 1).expect("row exists");

    // (a) Exact: an independent f64 softmax over the *same f32-rounded* inputs.
    //     logit_0 = q·k_0 / sqrt(2) = gain / sqrt(2);  logit_1 = 0.
    let logit_0 = f64::from(gain) / 2.0f64.sqrt();
    let exact_0 = logit_0.exp() / (logit_0.exp() + 1.0);
    let exact_1 = 1.0 / (logit_0.exp() + 1.0);
    assert!(
        (row_at_1[0] - exact_0).abs() < 1e-12,
        "the kernel must reproduce the softmax exactly: {} vs {exact_0}",
        row_at_1[0]
    );
    assert!((row_at_1[1] - exact_1).abs() < 1e-12);

    // (b) Hand-derived: exp(ln 3) / (exp(ln 3) + exp(0)) = 3/4, to the precision
    //     an f32-representable `sqrt(2)·ln 3` allows (see the doc comment).
    assert!(
        (row_at_1[0] - 0.75).abs() < 1e-7,
        "exp(ln 3) / (exp(ln 3) + exp(0)) = 3/4, got {}",
        row_at_1[0]
    );
    assert!(
        (row_at_1[1] - 0.25).abs() < 1e-7,
        "exp(0) / (exp(ln 3) + exp(0)) = 1/4, got {}",
        row_at_1[1]
    );
    let output_at_1 = attention.output_row(0, 1).expect("row exists");
    assert!(
        (output_at_1[0] - 0.75).abs() < 1e-6,
        "0.75·v_0[0] + 0.25·v_1[0] = 0.75, got {}",
        output_at_1[0]
    );
    assert!(
        (output_at_1[1] - 0.25).abs() < 1e-6,
        "0.75·v_0[1] + 0.25·v_1[1] = 0.25, got {}",
        output_at_1[1]
    );
}

/// The same exercise with **two heads**, which pins down the `[head][query][dim]`
/// indexing (a transposition bug here would silently mix heads and would not be
/// caught by the single-head example).
///
/// Geometry: 1 layer, 2 heads, `d_head = 2`, 2 tokens. Per token the key/value
/// buffers are laid out `[head 0 dims][head 1 dims]`.
///
/// ```text
/// token 0:  k = [ 1, 0 | 0, 2 ]   v = [ 1, 1 | 2, 0 ]
/// token 1:  k = [ 0, 1 | 1, 0 ]   v = [ 0, 2 | 0, 3 ]
///
/// head 0 query: [sqrt(2)·ln 3, 0]
///   logit_0 = (1·sqrt(2)·ln3 + 0·0) / sqrt(2) = ln 3   -> exp = 3
///   logit_1 = (0·sqrt(2)·ln3 + 1·0) / sqrt(2) = 0      -> exp = 1
///   A = [3/4, 1/4];  o = 0.75·[1,1] + 0.25·[0,2] = [0.75, 1.25]
///
/// head 1 query: [0, ln 4 / sqrt(2)]
///   logit_0 = (0·0 + 2·ln4/sqrt(2)) / sqrt(2) = ln 4   -> exp = 4
///   logit_1 = (1·0 + 0·ln4/sqrt(2)) / sqrt(2) = 0      -> exp = 1
///   A = [4/5, 1/5];  o = 0.8·[2,0] + 0.2·[0,3] = [1.6, 0.6]
/// ```
///
/// Tolerances are as in the single-head example: the weights are checked exactly
/// (`1e-12`) against an independent `f64` softmax of the same `f32`-rounded
/// inputs, and against the hand-derived rationals to `1e-7` — the precision an
/// `f32`-representable irrational query permits.
#[test]
fn attention_matches_a_hand_computed_two_head_example_exactly() {
    let mut cache = KvCacheTensor::new(1, 2, 2).expect("valid geometry");
    cache
        .append_token(&[1.0, 0.0, 0.0, 2.0], &[1.0, 1.0, 2.0, 0.0])
        .expect("well-formed token 0");
    cache
        .append_token(&[0.0, 1.0, 1.0, 0.0], &[0.0, 2.0, 0.0, 3.0])
        .expect("well-formed token 1");

    let head_0_gain = std::f32::consts::SQRT_2 * 3.0f32.ln();
    let head_1_gain = 4.0f32.ln() / std::f32::consts::SQRT_2;
    // [head][query][dim], one query at position 1.
    let queries = vec![head_0_gain, 0.0, 0.0, head_1_gain];
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &[1]).expect("well-formed call");

    // Head 0: exact against an independent f64 softmax, then against 3/4 : 1/4.
    let head_0 = attention.weight_row(0, 0).expect("row exists");
    let head_0_logit = f64::from(head_0_gain) / 2.0f64.sqrt();
    let head_0_exact = head_0_logit.exp() / (head_0_logit.exp() + 1.0);
    assert!(
        (head_0[0] - head_0_exact).abs() < 1e-12,
        "head 0 weights: {head_0:?} vs exact {head_0_exact}"
    );
    assert!(
        (head_0[0] - 0.75).abs() < 1e-7,
        "head 0 weights: {head_0:?}"
    );
    assert!(
        (head_0[1] - 0.25).abs() < 1e-7,
        "head 0 weights: {head_0:?}"
    );
    let head_0_output = attention.output_row(0, 0).expect("row exists");
    assert!((head_0_output[0] - 0.75).abs() < 1e-6);
    assert!((head_0_output[1] - 1.25).abs() < 1e-6);

    // Head 1: logit_0 = 2·gain / sqrt(2) = ln 4, so the weights are 4/5 : 1/5.
    let head_1 = attention.weight_row(1, 0).expect("row exists");
    let head_1_logit = 2.0 * f64::from(head_1_gain) / 2.0f64.sqrt();
    let head_1_exact = head_1_logit.exp() / (head_1_logit.exp() + 1.0);
    assert!(
        (head_1[0] - head_1_exact).abs() < 1e-12,
        "head 1 weights: {head_1:?} vs exact {head_1_exact}"
    );
    assert!((head_1[0] - 0.8).abs() < 1e-7, "head 1 weights: {head_1:?}");
    assert!((head_1[1] - 0.2).abs() < 1e-7, "head 1 weights: {head_1:?}");
    let head_1_output = attention.output_row(1, 0).expect("row exists");
    assert!((head_1_output[0] - 1.6).abs() < 1e-6);
    assert!((head_1_output[1] - 0.6).abs() < 1e-6);
}

#[test]
fn every_attention_row_sums_to_one() {
    let mut rng = SplitMix64::new(0x5EED_0001);
    let mut cache = KvCacheTensor::new(2, 3, 6).expect("valid geometry");
    for _ in 0..40 {
        let keys: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 2.0)
            .collect();
        let values: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 2.0)
            .collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }

    let positions: Vec<usize> = (0..40).collect();
    let queries: Vec<f32> = (0..3 * 40 * 6)
        .map(|_| rng.next_symmetric_unit() * 2.0)
        .collect();

    for layer in 0..2 {
        let attention = scaled_dot_product_attention(&cache, layer, &queries, &positions)
            .expect("well-formed call");
        for head in 0..3 {
            for query in 0..40 {
                let row = attention.weight_row(head, query).expect("row exists");
                let sum: f64 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "layer {layer} head {head} query {query}: softmax row sums to {sum}, not 1"
                );
                assert!(
                    row.iter().all(|&weight| (0.0..=1.0).contains(&weight)),
                    "softmax weights must be a probability distribution"
                );
            }
        }
    }
}

#[test]
fn the_causal_mask_gives_future_tokens_exactly_zero_weight() {
    let mut rng = SplitMix64::new(0x5EED_0002);
    let mut cache = KvCacheTensor::new(1, 2, 4).expect("valid geometry");
    for _ in 0..24 {
        let keys: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 3.0)
            .collect();
        let values: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 3.0)
            .collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }

    let positions: Vec<usize> = (0..24).collect();
    let queries: Vec<f32> = (0..2 * 24 * 4)
        .map(|_| rng.next_symmetric_unit() * 3.0)
        .collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &positions).expect("well-formed call");

    for query in 0..24 {
        for key in 0..24 {
            let visible = attention.is_visible(query, key);
            assert_eq!(
                visible,
                key <= query,
                "position {query} must see exactly the keys at positions <= {query}"
            );
            for head in 0..2 {
                let weight = attention.weight_row(head, query).expect("row exists")[key];
                if key > query {
                    assert_eq!(
                        weight, 0.0,
                        "query {query} must place *exactly* zero weight on future token {key}, got {weight}"
                    );
                } else {
                    assert!(
                        weight > 0.0,
                        "a visible key must get strictly positive weight (softmax has full support)"
                    );
                }
            }
        }
    }
}

/// The causal mask is driven by **absolute stream positions**, not slot indices,
/// so it stays correct after eviction has punched holes in the cache. If the mask
/// were (incorrectly) re-derived from slot indices, a compressed cache would let
/// a query attend to tokens from its own future.
#[test]
fn the_causal_mask_survives_eviction() {
    let mut rng = SplitMix64::new(0x5EED_0003);
    let mut cache = KvCacheTensor::new(1, 1, 4).expect("valid geometry");
    for _ in 0..20 {
        let keys: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
        let values: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }
    // Keep a gappy set: two "sinks", a heavy hitter, and a recent window.
    cache
        .retain_tokens(&[0, 1, 9, 16, 17, 18, 19])
        .expect("valid selection");
    assert_eq!(cache.positions(), &[0, 1, 9, 16, 17, 18, 19]);

    // A query at absolute position 16 may see positions 0, 1, 9, 16 -- and must
    // NOT see 17, 18, 19, which now sit at slots 4, 5, 6.
    let queries: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &[16]).expect("well-formed call");
    let row = attention.weight_row(0, 0).expect("row exists");

    for (slot, &position) in cache.positions().iter().enumerate() {
        if position <= 16 {
            assert!(
                row[slot] > 0.0,
                "slot {slot} (position {position}) is in the past and must be visible"
            );
        } else {
            assert_eq!(
                row[slot], 0.0,
                "slot {slot} (position {position}) is in the *future* of a query at position 16 \
                 and must get exactly zero weight -- the mask must follow positions, not slots"
            );
        }
    }
    let sum: f64 = row.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

/// A query that can see *nothing* (every key is in its future) has no softmax:
/// the distribution over the empty set does not exist. The kernel returns an
/// all-zero weight row and an all-zero output rather than inventing one.
#[test]
fn a_query_with_no_visible_key_yields_a_zero_row_and_zero_output() {
    let mut rng = SplitMix64::new(0x5EED_0004);
    let mut cache = KvCacheTensor::new(1, 1, 4).expect("valid geometry");
    for _ in 0..8 {
        let keys: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
        let values: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit() + 2.0).collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }
    // Retain only the tail: the surviving slots carry positions 4..=7.
    cache.retain_tokens(&[4, 5, 6, 7]).expect("valid selection");

    // Query at position 3 -- strictly before every surviving key.
    let queries: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &[3]).expect("well-formed call");

    let row = attention.weight_row(0, 0).expect("row exists");
    assert!(
        row.iter().all(|&weight| weight == 0.0),
        "an all-masked row must be all-zero, got {row:?}"
    );
    let output = attention.output_row(0, 0).expect("row exists");
    assert!(
        output.iter().all(|&value| value == 0.0),
        "an all-masked row must produce a zero output, got {output:?}"
    );
    assert!(output.iter().all(|value| value.is_finite()));
}

/// Extreme (but finite) magnitudes must not produce a `NaN` or an infinity
/// anywhere. The naive softmax overflows for logits above ~709; the
/// max-subtracted one cannot, and `f64` logit accumulation means even
/// `f32`-extremal keys cannot overflow the dot product.
#[test]
fn extreme_logits_stay_finite() {
    for magnitude in [1e-30f32, 1e-6, 1.0, 1e18, 1e30] {
        let mut rng = SplitMix64::new(0x5EED_0005);
        let mut cache = KvCacheTensor::new(1, 2, 8).expect("valid geometry");
        for _ in 0..16 {
            let keys: Vec<f32> = (0..cache.token_buffer_len())
                .map(|_| rng.next_symmetric_unit() * magnitude)
                .collect();
            let values: Vec<f32> = (0..cache.token_buffer_len())
                .map(|_| rng.next_symmetric_unit() * magnitude)
                .collect();
            cache.append_token(&keys, &values).expect("well-formed");
        }
        let positions: Vec<usize> = (0..16).collect();
        let queries: Vec<f32> = (0..2 * 16 * 8)
            .map(|_| rng.next_symmetric_unit() * magnitude)
            .collect();
        let attention = scaled_dot_product_attention(&cache, 0, &queries, &positions)
            .expect("well-formed call");

        assert!(
            attention.output().iter().all(|value| value.is_finite()),
            "magnitude {magnitude:e}: attention output contains a non-finite value"
        );
        assert!(
            attention.weights().iter().all(|weight| weight.is_finite()),
            "magnitude {magnitude:e}: attention weights contain a non-finite value"
        );
        for head in 0..2 {
            for query in 0..16 {
                let sum: f64 = attention
                    .weight_row(head, query)
                    .expect("row exists")
                    .iter()
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "magnitude {magnitude:e}: row ({head}, {query}) sums to {sum}"
                );
            }
        }

        // The output is a convex combination of the value vectors, so it can
        // never exceed the largest value element in magnitude.
        let mut largest_value = 0.0f32;
        for token in 0..16 {
            for head in 0..2 {
                let slice = cache.value(0, token, head).expect("in range");
                for element in slice {
                    largest_value = largest_value.max(element.abs());
                }
            }
        }
        for &value in attention.output() {
            assert!(
                value.abs() <= largest_value * 1.000_01 + 1e-30,
                "magnitude {magnitude:e}: attention manufactured magnitude ({value:e} > {largest_value:e})"
            );
        }
    }
}

#[test]
fn non_finite_inputs_are_rejected_at_the_boundary() {
    let mut cache = KvCacheTensor::new(1, 1, 2).expect("valid geometry");
    assert!(matches!(
        cache.append_token(&[1.0, f32::NAN], &[1.0, 1.0]),
        Err(KvCompressionError::NonFiniteInput {
            what: "key buffer",
            index: 1
        })
    ));
    assert!(matches!(
        cache.append_token(&[1.0, 1.0], &[f32::INFINITY, 1.0]),
        Err(KvCompressionError::NonFiniteInput {
            what: "value buffer",
            index: 0
        })
    ));
    cache
        .append_token(&[1.0, 0.0], &[1.0, 0.0])
        .expect("finite token");
    assert!(matches!(
        scaled_dot_product_attention(&cache, 0, &[f32::NAN, 0.0], &[0]),
        Err(KvCompressionError::NonFiniteInput {
            what: "query buffer",
            ..
        })
    ));
}

#[test]
fn malformed_attention_calls_are_rejected() {
    let empty = KvCacheTensor::new(1, 1, 2).expect("valid geometry");
    assert!(matches!(
        scaled_dot_product_attention(&empty, 0, &[1.0, 0.0], &[0]),
        Err(KvCompressionError::EmptyCache { .. })
    ));

    let mut cache = KvCacheTensor::new(2, 1, 2).expect("valid geometry");
    cache
        .append_token(&[1.0, 0.0, 1.0, 0.0], &[1.0, 0.0, 1.0, 0.0])
        .expect("well-formed");
    cache
        .append_token(&[0.0, 1.0, 0.0, 1.0], &[0.0, 1.0, 0.0, 1.0])
        .expect("well-formed");

    assert!(matches!(
        scaled_dot_product_attention(&cache, 2, &[1.0, 0.0], &[0]),
        Err(KvCompressionError::LayerOutOfRange {
            layer: 2,
            num_layers: 2
        })
    ));
    assert!(matches!(
        scaled_dot_product_attention(&cache, 0, &[1.0], &[0]),
        Err(KvCompressionError::ShapeMismatch {
            what: "query buffer",
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        scaled_dot_product_attention(&cache, 0, &[], &[]),
        Err(KvCompressionError::InvalidTokenSelection { .. })
    ));
    // Query positions must be chronological: score accumulation folds them in
    // order, and a decayed accumulation out of order is meaningless.
    assert!(matches!(
        scaled_dot_product_attention(&cache, 0, &[1.0, 0.0, 0.0, 1.0], &[1, 0]),
        Err(KvCompressionError::InvalidTokenSelection { .. })
    ));

    // Layers are genuinely independent storage: the same token has different
    // key/value blocks per layer, and layer 1's attention differs from layer 0's.
    let mut layered = KvCacheTensor::new(2, 1, 2).expect("valid geometry");
    layered
        .append_token(&[1.0, 0.0, 0.0, 1.0], &[5.0, 0.0, 0.0, 7.0])
        .expect("well-formed");
    let layer_0 =
        scaled_dot_product_attention(&layered, 0, &[1.0, 0.0], &[0]).expect("well-formed call");
    let layer_1 =
        scaled_dot_product_attention(&layered, 1, &[1.0, 0.0], &[0]).expect("well-formed call");
    assert_eq!(layer_0.output_row(0, 0).expect("row"), &[5.0, 0.0]);
    assert_eq!(layer_1.output_row(0, 0).expect("row"), &[0.0, 7.0]);
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. SCORE ACCUMULATION
// ═════════════════════════════════════════════════════════════════════════════

/// The accumulated score really is the attention matrix's column sum (head-mean),
/// and the visible-step count really is the number of softmax rows each token
/// competed in. Both are recomputed here straight from the weight matrix.
#[test]
fn accumulated_scores_are_the_attention_column_sums() {
    let mut rng = SplitMix64::new(0x5EED_0010);
    let mut cache = KvCacheTensor::new(1, 3, 4).expect("valid geometry");
    for _ in 0..12 {
        let keys: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 2.0)
            .collect();
        let values: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit() * 2.0)
            .collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }
    let positions: Vec<usize> = (0..12).collect();
    let queries: Vec<f32> = (0..3 * 12 * 4)
        .map(|_| rng.next_symmetric_unit() * 2.0)
        .collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &positions).expect("well-formed call");

    let mut stats = KvAttentionStats::new(12, 4);
    stats.accumulate(&attention, None).expect("stats match");

    for token in 0..12 {
        // Recompute the column sum from the raw weight matrix.
        let mut expected_mass = 0.0f64;
        let mut expected_visible = 0.0f64;
        for query in 0..12 {
            if !attention.is_visible(query, token) {
                continue;
            }
            expected_visible += 1.0;
            let head_mean: f64 = (0..3)
                .map(|head| attention.weight_row(head, query).expect("row exists")[token])
                .sum::<f64>()
                / 3.0;
            expected_mass += head_mean;
        }
        assert!(
            (stats.accumulated()[token] - expected_mass).abs() < 1e-12,
            "token {token}: accumulated {} != column sum {expected_mass}",
            stats.accumulated()[token]
        );
        assert!(
            (stats.visible_steps()[token] - expected_visible).abs() < 1e-12,
            "token {token}: visible {} != {expected_visible}",
            stats.visible_steps()[token]
        );
        // Under a causal mask, token t is visible to queries t..n-1.
        assert_eq!(expected_visible as usize, 12 - token);
    }
    assert_eq!(stats.steps_recorded(), 12);
    assert!(stats.has_history());

    // The column sums partition the total attention mass: every row sums to 1,
    // and there are 12 rows, so the column sums total exactly 12.
    let total: f64 = stats.accumulated().iter().sum();
    assert!(
        (total - 12.0).abs() < 1e-9,
        "column sums must total the number of query rows, got {total}"
    );
}

/// **The early-token bias, quantified exactly.**
///
/// Build a cache in which every key is identical, so every softmax row is exactly
/// uniform: `A[q, t] = 1 / (q + 1)` for `t <= q`. Then no token is more salient
/// than any other — yet the raw cumulative score still ranks them, purely because
/// of the causal mask's shape:
///
/// ```text
/// cumulative[t] = Σ_{q >= t} 1/(q+1) = H_n - H_t     (strictly decreasing in t)
/// visible[t]    = n - t
/// mean[t]       = (H_n - H_t) / (n - t)
/// ```
///
/// The correction removes *exactly* the visible-count factor, so the spread of
/// the cumulative score exceeds the spread of the mean by exactly `n`:
///
/// ```text
/// cumulative[0] / cumulative[n-1]  =  n · (mean[0] / mean[n-1])
/// ```
///
/// This is asserted to `1e-9`. (The mean does not make every token *equal* here,
/// and this test does not pretend it does: a token at position 0 genuinely does
/// receive more mass in the early, low-competition rows. What the mean removes is
/// the *summation-length* component of the bias — the dominant one, worth a factor
/// of `n`.)
#[test]
fn the_mean_normalization_removes_exactly_the_causal_mask_bias() {
    const N: usize = 64;
    let mut cache = KvCacheTensor::new(1, 1, 4).expect("valid geometry");
    for _ in 0..N {
        // Identical keys => identical logits => a perfectly uniform softmax row.
        cache
            .append_token(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0])
            .expect("well-formed");
    }
    let positions: Vec<usize> = (0..N).collect();
    let queries: Vec<f32> = (0..N).flat_map(|_| [1.0f32, 0.0, 0.0, 0.0]).collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &positions).expect("well-formed call");

    // Sanity: the rows really are uniform.
    for query in 0..N {
        let row = attention.weight_row(0, query).expect("row exists");
        let expected = 1.0 / (query + 1) as f64;
        for (token, &weight) in row.iter().enumerate().take(query + 1) {
            assert!(
                (weight - expected).abs() < 1e-12,
                "row {query} token {token}: {weight} != {expected}"
            );
        }
    }

    let mut stats = KvAttentionStats::new(N, 8);
    stats.accumulate(&attention, None).expect("stats match");

    let cumulative = stats.saliency(KvScoreNormalization::Cumulative);
    let mean = stats.saliency(KvScoreNormalization::MeanPerVisibleStep);

    // The raw column sum is strictly decreasing in position: the bias is real.
    for token in 1..N {
        assert!(
            cumulative[token] < cumulative[token - 1],
            "the raw column sum must decrease with position (the early-token bias): \
             cumulative[{token}] = {} >= cumulative[{}] = {}",
            cumulative[token],
            token - 1,
            cumulative[token - 1]
        );
    }

    // The harmonic identity: cumulative[t] = H_N - H_t.
    let harmonic = |k: usize| -> f64 { (1..=k).map(|i| 1.0 / i as f64).sum() };
    for token in 0..N {
        let expected = harmonic(N) - harmonic(token);
        assert!(
            (cumulative[token] - expected).abs() < 1e-9,
            "cumulative[{token}] = {} != H_{N} - H_{token} = {expected}",
            cumulative[token]
        );
    }

    let cumulative_spread = cumulative[0] / cumulative[N - 1];
    let mean_spread = mean[0] / mean[N - 1];
    assert!(
        (cumulative_spread - N as f64 * mean_spread).abs() < 1e-9,
        "the mean normalization must remove exactly the visible-count factor: \
         cumulative spread {cumulative_spread} != {N} · mean spread {mean_spread}"
    );
    assert!(
        cumulative_spread > 300.0 && mean_spread < 5.0,
        "for N = {N} the raw spread is ~304x and the mean's is ~4.7x, got {cumulative_spread} and {mean_spread}"
    );
}

/// The **forgetting factor**: a token that was a heavy hitter long ago but has
/// been ignored ever since must decay out of the heavy-hitter set instead of
/// squatting in it forever.
///
/// The workload switches what it cares about halfway through: queries `0..=29`
/// key on token 5, queries `30..` key on token 30. Without decay both tokens end
/// up with comparable accumulated mass. With a decay of `0.9` per step, token 5's
/// contribution has been multiplied by `0.9^34 ≈ 0.03` by the end, and its score
/// collapses relative to token 30's.
#[test]
fn the_forgetting_factor_decays_a_stale_heavy_hitter() {
    const N: usize = 64;
    const EARLY: usize = 5;
    const LATE: usize = 30;
    let mut rng = SplitMix64::new(0x5EED_0011);
    let mut cache = KvCacheTensor::new(1, 1, 8).expect("valid geometry");
    for position in 0..N {
        let mut keys = vec![0.0f32; 8];
        let mut values = vec![0.0f32; 8];
        if position == EARLY {
            keys[0] = 6.0; // keys on the "phase 1" query direction (e0)
        } else if position == LATE {
            keys[1] = 6.0; // keys on the "phase 2" query direction (e1)
        } else {
            for d in 2..8 {
                keys[d] = rng.next_symmetric_unit() * 0.5;
            }
        }
        for d in 0..8 {
            values[d] = rng.next_symmetric_unit();
        }
        cache.append_token(&keys, &values).expect("well-formed");
    }

    let positions: Vec<usize> = (0..N).collect();
    let mut queries = vec![0.0f32; N * 8];
    for query in 0..N {
        // Phase 1 keys on e0 (finds token 5); phase 2 keys on e1 (finds token 30).
        let axis = usize::from(query >= LATE);
        queries[query * 8 + axis] = 4.0;
    }
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &positions).expect("well-formed call");

    let mut undecayed = KvAttentionStats::new(N, 8);
    undecayed.accumulate(&attention, None).expect("stats match");
    let mut decayed = KvAttentionStats::new(N, 8);
    decayed
        .accumulate(&attention, Some(0.9))
        .expect("stats match");

    let flat = undecayed.saliency(KvScoreNormalization::Cumulative);
    let faded = decayed.saliency(KvScoreNormalization::Cumulative);

    let flat_ratio = flat[EARLY] / flat[LATE];
    let faded_ratio = faded[EARLY] / faded[LATE];

    assert!(
        flat_ratio > 0.4,
        "without forgetting, the stale heavy hitter keeps a comparable score \
         (got ratio {flat_ratio})"
    );
    assert!(
        faded_ratio < flat_ratio / 5.0,
        "with a 0.9 forgetting factor the stale heavy hitter must collapse relative to the \
         fresh one: faded ratio {faded_ratio} is not < {flat_ratio} / 5"
    );
    // Both tokens really were heavy hitters at some point -- this is not a
    // vacuous comparison of two near-zero scores.
    assert!(flat[EARLY] > 10.0 && flat[LATE] > 10.0);
}

#[test]
fn statistics_are_reindexed_onto_the_survivors() {
    let mut rng = SplitMix64::new(0x5EED_0012);
    let mut cache = KvCacheTensor::new(1, 1, 4).expect("valid geometry");
    for _ in 0..10 {
        let keys: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit() * 2.0).collect();
        let values: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit() * 2.0).collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }
    let positions: Vec<usize> = (0..10).collect();
    let queries: Vec<f32> = (0..10 * 4).map(|_| rng.next_symmetric_unit()).collect();
    let attention =
        scaled_dot_product_attention(&cache, 0, &queries, &positions).expect("well-formed call");
    let mut stats = KvAttentionStats::new(10, 4);
    stats.accumulate(&attention, None).expect("stats match");

    let before = stats.accumulated().to_vec();
    let keep = [1usize, 4, 7, 8, 9];
    stats.retain_tokens(&keep).expect("valid selection");
    cache.retain_tokens(&keep).expect("valid selection");

    assert_eq!(stats.num_tokens(), cache.seq_len());
    for (slot, &old_index) in keep.iter().enumerate() {
        assert_eq!(
            stats.accumulated()[slot],
            before[old_index],
            "slot {slot} of the compressed stats must carry the history of old slot {old_index}"
        );
    }
    assert_eq!(cache.positions(), &keep);

    // Freshly appended tokens start with a clean history.
    let keys: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
    let values: Vec<f32> = (0..4).map(|_| rng.next_symmetric_unit()).collect();
    cache.append_token(&keys, &values).expect("well-formed");
    stats.on_append(1);
    assert_eq!(stats.num_tokens(), cache.seq_len());
    assert_eq!(stats.accumulated()[5], 0.0);
    assert_eq!(stats.visible_steps()[5], 0.0);
    // The position counter is never rewound by eviction.
    assert_eq!(cache.positions().last(), Some(&10));
}

#[test]
fn stats_reject_a_mismatched_attention_matrix() {
    let mut cache = KvCacheTensor::new(1, 1, 2).expect("valid geometry");
    cache
        .append_token(&[1.0, 0.0], &[1.0, 0.0])
        .expect("well-formed");
    cache
        .append_token(&[0.0, 1.0], &[0.0, 1.0])
        .expect("well-formed");
    let attention =
        scaled_dot_product_attention(&cache, 0, &[1.0, 0.0], &[1]).expect("well-formed call");

    let mut stats = KvAttentionStats::new(5, 4);
    assert!(matches!(
        stats.accumulate(&attention, None),
        Err(KvCompressionError::StatsLengthMismatch {
            stats_tokens: 5,
            cache_tokens: 2
        })
    ));
    let mut good = KvAttentionStats::new(2, 4);
    assert!(matches!(
        good.accumulate(&attention, Some(1.5)),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
}
