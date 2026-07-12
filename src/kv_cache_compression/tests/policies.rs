//! Tests for the four eviction policies: the headline fidelity-under-a-budget
//! experiment, the `StreamingLLM` attention-sink demonstration, the budget-adherence
//! invariant, and the edge cases.

use std::collections::BTreeSet;

use crate::kv_cache_compression::{
    KvAttentionStats, KvCacheCompressor, KvCacheTensor, KvCompressionConfig, KvCompressionError,
    KvEvictionPolicy, KvScoreNormalization, KvSnapPooling, plan_recency_lru, plan_streaming_llm,
    pool_scores, scaled_dot_product_attention,
};

use super::{SplitMix64, bit_pattern, relative_deviation};
// ═════════════════════════════════════════════════════════════════════════════
// 3. THE HEADLINE WORKLOAD
// ═════════════════════════════════════════════════════════════════════════════

const HEADLINE_SEQ_LEN: usize = 128;
const HEADLINE_HEADS: usize = 2;
const HEADLINE_HEAD_DIM: usize = 8;
const HEADLINE_BUDGET: usize = 32;
/// Three genuinely high-salience tokens, all of them **old** -- deliberately far
/// from any recent window, so that a recency policy cannot stumble into them.
const HEADLINE_SALIENT: [usize; 3] = [7, 23, 51];
/// Distinct key gains, so the three salient tokens have strictly ordered scores
/// (equal scores would make `SnapKV`'s max-pool plateaus tie, and a tie-break is
/// not a demonstration of anything).
const HEADLINE_SALIENT_GAIN: [f32; 3] = [6.0, 6.2, 6.4];
const HEADLINE_QUERY_GAIN: f32 = 4.0;

/// A cache whose attention mass provably sits on three old tokens.
///
/// Every query in this workload points along `e0`. The salient tokens are the
/// only ones with a non-zero `e0` component in their key, so their logits are
/// large (`gain · 4 / sqrt(8) ≈ 8.5`) while **every filler token's logit is
/// exactly `0`** -- the filler keys have a zeroed `e0` component, so their dot
/// product with the query is exactly zero, by construction rather than by luck.
/// The salient tokens' values point along `e1` with a magnitude of 3, so an
/// attention output that has lost them is immediately visible.
fn build_headline_cache() -> KvCacheTensor {
    let mut rng = SplitMix64::new(0xC0FF_EE00_1234_5678);
    let mut cache =
        KvCacheTensor::new(1, HEADLINE_HEADS, HEADLINE_HEAD_DIM).expect("valid geometry");
    for position in 0..HEADLINE_SEQ_LEN {
        let salient = HEADLINE_SALIENT.iter().position(|&p| p == position);
        let mut keys = vec![0.0f32; cache.token_buffer_len()];
        let mut values = vec![0.0f32; cache.token_buffer_len()];
        for head in 0..HEADLINE_HEADS {
            let base = head * HEADLINE_HEAD_DIM;
            if let Some(index) = salient {
                keys[base] = HEADLINE_SALIENT_GAIN[index];
                values[base + 1] = 3.0;
            } else {
                // e0 stays exactly zero: filler cannot accidentally attract the
                // query. The rest of the key, and the whole value, is noise.
                for d in 1..HEADLINE_HEAD_DIM {
                    keys[base + d] = rng.next_symmetric_unit() * 0.5;
                }
                for d in 0..HEADLINE_HEAD_DIM {
                    values[base + d] = rng.next_symmetric_unit();
                }
            }
        }
        cache.append_token(&keys, &values).expect("well-formed");
    }
    cache
}

/// `[head][query][dim]` queries, all pointing along `e0`.
fn headline_queries(num_queries: usize) -> Vec<f32> {
    let mut queries = vec![0.0f32; HEADLINE_HEADS * num_queries * HEADLINE_HEAD_DIM];
    for head in 0..HEADLINE_HEADS {
        for query in 0..num_queries {
            queries[(head * num_queries + query) * HEADLINE_HEAD_DIM] = HEADLINE_QUERY_GAIN;
        }
    }
    queries
}

/// Prefill: run real causal attention over the whole sequence and fold the real
/// weight matrix into the statistics. This is the *history* the score-driven
/// policies get to see.
fn headline_prefill_stats(cache: &KvCacheTensor, config: &KvCompressionConfig) -> KvAttentionStats {
    let positions: Vec<usize> = (0..cache.seq_len()).collect();
    let queries = headline_queries(positions.len());
    let attention =
        scaled_dot_product_attention(cache, 0, &queries, &positions).expect("well-formed call");
    let mut stats = KvAttentionStats::for_cache(cache, config);
    stats
        .accumulate(&attention, config.score_decay)
        .expect("stats match the cache");
    stats
}

/// The **held-out** evaluation: one fresh query at position `seq_len`, issued
/// *after* the history the policies were scored on. This is what makes the
/// experiment non-tautological -- the policies never saw this query.
fn headline_eval_output(cache: &KvCacheTensor) -> Vec<f32> {
    let queries = headline_queries(1);
    let attention = scaled_dot_product_attention(cache, 0, &queries, &[HEADLINE_SEQ_LEN])
        .expect("well-formed call");
    attention.output().to_vec()
}

fn headline_config(policy: KvEvictionPolicy) -> KvCompressionConfig {
    let base = KvCompressionConfig::new(policy, HEADLINE_BUDGET);
    match policy {
        KvEvictionPolicy::H2O => base.with_recent_window(16),
        KvEvictionPolicy::StreamingLlm => base.with_sink_tokens(4).with_recent_window(28),
        // A width-7 max-pool around three peaks needs 21 prefix slots; the prefix
        // budget here is 32 - 8 = 24, so the plateaus fit. (Scaling the kernel to
        // the budget is a real requirement of the method, not a fudge: with a
        // prefix budget below the plateau size, the plateaus eat the budget and
        // the lowest-indexed peaks are lost to the tie-break.)
        KvEvictionPolicy::SnapKv => base
            .with_observation_window(8)
            .with_pooling(KvSnapPooling::Max, 7),
        KvEvictionPolicy::RecencyLru => base,
    }
}

/// Compress a fresh copy of the headline cache with `policy` and measure the
/// deviation of its held-out attention output from the uncompressed one.
fn headline_run(policy: KvEvictionPolicy, reference: &[f32]) -> (Vec<usize>, f64) {
    let mut cache = build_headline_cache();
    let compressor = KvCacheCompressor::new(headline_config(policy)).expect("valid config");
    let mut stats = headline_prefill_stats(&cache, compressor.config());
    let report = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");

    assert!(report.within_budget, "{policy:?} broke its budget");
    assert_eq!(report.tokens_after, HEADLINE_BUDGET);
    assert_eq!(report.tokens_evicted, HEADLINE_SEQ_LEN - HEADLINE_BUDGET);
    assert!((report.memory_saved_ratio() - 0.75).abs() < 1e-12);
    assert_eq!(stats.num_tokens(), cache.seq_len());

    let output = headline_eval_output(&cache);
    assert!(output.iter().all(|value| value.is_finite()));
    (
        report.retained_positions,
        relative_deviation(reference, &output),
    )
}

/// **THE HEADLINE EXPERIMENT: fidelity under a budget.**
///
/// A 128-token cache in which three *old* tokens (positions 7, 23, 51) genuinely
/// carry ~99.4% of every query's attention mass -- measured, not assumed -- buried
/// among 125 filler tokens whose logits against the query are *exactly* zero.
///
/// Each policy compresses to 32 tokens (a **4x** memory saving, 25% of the
/// sequence). Then a **held-out** query at position 128 -- one no policy ever saw
/// -- is run against each compressed cache, and its output is compared to the
/// output of the full, uncompressed cache.
///
/// The result is the module's entire justification:
///
/// - H2O and `SnapKV` *find* the salient tokens by their accumulated attention
///   mass and keep them, and their outputs are within a fraction of a percent of
///   full attention.
/// - `RecencyLru` evicts them, because they are old and it knows nothing but age.
///   Its output deviates by roughly its own magnitude -- it is destroyed.
/// - `StreamingLLM` sits with the baseline **on this workload by construction**:
///   the salient tokens are not in its sinks and not in its window. This is
///   recorded rather than hidden; `StreamingLLM`'s own claim is a different one,
///   and is demonstrated in the sink test below.
#[test]
fn headline_h2o_and_snapkv_preserve_attention_output_where_recency_destroys_it() {
    let full_cache = build_headline_cache();
    let reference = headline_eval_output(&full_cache);

    // First: verify the premise. The salient tokens really do carry the mass.
    let attention =
        scaled_dot_product_attention(&full_cache, 0, &headline_queries(1), &[HEADLINE_SEQ_LEN])
            .expect("well-formed call");
    let row = attention.weight_row(0, 0).expect("row exists");
    let salient_mass: f64 = HEADLINE_SALIENT.iter().map(|&token| row[token]).sum();
    assert!(
        salient_mass > 0.95,
        "the premise of this experiment is that 3 of 128 tokens carry the attention mass; \
         measured mass = {salient_mass}"
    );
    // ... and every filler logit is exactly zero, so filler weights are uniform.
    let filler_weight = row[100];
    assert!(
        filler_weight < 1e-4,
        "filler tokens must be attentionally irrelevant, got {filler_weight}"
    );

    let (h2o_kept, h2o_deviation) = headline_run(KvEvictionPolicy::H2O, &reference);
    let (snap_kept, snap_deviation) = headline_run(KvEvictionPolicy::SnapKv, &reference);
    let (stream_kept, stream_deviation) = headline_run(KvEvictionPolicy::StreamingLlm, &reference);
    let (recency_kept, recency_deviation) = headline_run(KvEvictionPolicy::RecencyLru, &reference);

    println!("── fidelity under a 32/128 (25%) budget ─────────────────────────");
    println!("  salient attention mass (measured) : {salient_mass:.6}");
    println!("  H2O           deviation           : {h2o_deviation:.6}");
    println!("  SnapKV        deviation           : {snap_deviation:.6}");
    println!("  StreamingLLM  deviation           : {stream_deviation:.6}");
    println!("  RecencyLru    deviation           : {recency_deviation:.6}");

    // ── The score-driven policies find the heavy hitters ─────────────────────
    for &salient in &HEADLINE_SALIENT {
        assert!(
            h2o_kept.contains(&salient),
            "H2O must retain heavy hitter {salient}; kept {h2o_kept:?}"
        );
        assert!(
            snap_kept.contains(&salient),
            "SnapKV must retain heavy hitter {salient}; kept {snap_kept:?}"
        );
        assert!(
            !recency_kept.contains(&salient),
            "the recency baseline is supposed to LOSE the old heavy hitter {salient} -- \
             if it kept it, this experiment is not testing what it claims to"
        );
        assert!(
            !stream_kept.contains(&salient),
            "on this workload StreamingLLM's sinks and window do not cover {salient}"
        );
    }

    // ── ... and that shows up in the attention output ────────────────────────
    assert!(
        h2o_deviation < 0.02,
        "H2O must stay close to full attention, got {h2o_deviation}"
    );
    assert!(
        snap_deviation < 0.02,
        "SnapKV must stay close to full attention, got {snap_deviation}"
    );
    assert!(
        recency_deviation > 0.5,
        "the recency baseline must be badly damaged (it threw the mass away), got {recency_deviation}"
    );
    assert!(
        recency_deviation > 20.0 * h2o_deviation,
        "the whole point: recency ({recency_deviation}) must deviate by more than 20x H2O \
         ({h2o_deviation})"
    );
    assert!(
        recency_deviation > 20.0 * snap_deviation,
        "the whole point: recency ({recency_deviation}) must deviate by more than 20x SnapKV \
         ({snap_deviation})"
    );
    assert!(
        stream_deviation > 0.5,
        "StreamingLLM has no sink structure to exploit on this workload, got {stream_deviation}"
    );
}

/// `SnapKV`'s pooling is supposed to make the retained prefix cluster into
/// **contiguous spans** around the informative regions, not a scatter of isolated
/// tokens. With a width-7 max-pool, each of the three peaks should drag its whole
/// `±3` neighbourhood into the cache.
#[test]
fn snapkv_pooling_retains_contiguous_spans_around_the_peaks() {
    let mut cache = build_headline_cache();
    let compressor =
        KvCacheCompressor::new(headline_config(KvEvictionPolicy::SnapKv)).expect("valid config");
    let mut stats = headline_prefill_stats(&cache, compressor.config());
    let report = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");
    let kept: BTreeSet<usize> = report.retained_positions.iter().copied().collect();

    for &peak in &HEADLINE_SALIENT {
        for offset in 0..=6usize {
            let neighbour = peak + offset - 3;
            assert!(
                kept.contains(&neighbour),
                "a width-7 max-pool must pull the whole ±3 span around peak {peak} into the \
                 cache; {neighbour} is missing. kept = {kept:?}"
            );
        }
    }

    // Without pooling, the same budget keeps the peaks but *not* their spans --
    // which is exactly the isolated-token failure the pooling exists to prevent.
    let unpooled_config =
        headline_config(KvEvictionPolicy::SnapKv).with_pooling(KvSnapPooling::None, 7);
    let mut unpooled_cache = build_headline_cache();
    let unpooled = KvCacheCompressor::new(unpooled_config).expect("valid config");
    let mut unpooled_stats = headline_prefill_stats(&unpooled_cache, unpooled.config());
    let unpooled_report = unpooled
        .compress(&mut unpooled_cache, &mut unpooled_stats)
        .expect("compressible");
    let unpooled_kept: BTreeSet<usize> =
        unpooled_report.retained_positions.iter().copied().collect();
    for &peak in &HEADLINE_SALIENT {
        assert!(
            unpooled_kept.contains(&peak),
            "the peaks themselves survive"
        );
    }
    let pooled_span_coverage = HEADLINE_SALIENT
        .iter()
        .flat_map(|&peak| peak - 3..=peak + 3)
        .filter(|slot| kept.contains(slot))
        .count();
    let unpooled_span_coverage = HEADLINE_SALIENT
        .iter()
        .flat_map(|&peak| peak - 3..=peak + 3)
        .filter(|slot| unpooled_kept.contains(slot))
        .count();
    assert!(
        pooled_span_coverage > unpooled_span_coverage,
        "max-pooling must retain strictly more of the peaks' neighbourhoods than no pooling \
         ({pooled_span_coverage} vs {unpooled_span_coverage})"
    );
}

/// H2O finds the heavy hitters under *either* normalization. The correction for
/// the early-token bias is not a crutch that the result depends on.
#[test]
fn h2o_finds_the_heavy_hitters_under_both_normalizations() {
    for normalization in [
        KvScoreNormalization::Cumulative,
        KvScoreNormalization::MeanPerVisibleStep,
    ] {
        let mut cache = build_headline_cache();
        let config = headline_config(KvEvictionPolicy::H2O).with_normalization(normalization);
        let compressor = KvCacheCompressor::new(config).expect("valid config");
        let mut stats = headline_prefill_stats(&cache, compressor.config());
        let report = compressor
            .compress(&mut cache, &mut stats)
            .expect("compressible");
        for &salient in &HEADLINE_SALIENT {
            assert!(
                report.retained_positions.contains(&salient),
                "{normalization:?}: H2O must retain heavy hitter {salient}"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. THE STREAMING-LLM SINK DEMONSTRATION
// ═════════════════════════════════════════════════════════════════════════════

const SINK_SEQ_LEN: usize = 64;
const SINK_SLOTS: [usize; 3] = [0, 1, 2];
const SINK_MIDDLE_SLOTS: [usize; 3] = [30, 31, 32];
/// The content the query actually needs sits in the *recent* window, which is the
/// situation `StreamingLLM` is designed for: local content, plus the sinks.
const SINK_CONTENT_SLOTS: [usize; 2] = [60, 61];
const SINK_HEAD_DIM: usize = 8;

/// A cache with a **real attention-sink structure**, of the kind trained models
/// are observed to develop.
///
/// - Tokens 0..2 ("sinks") have a large key component along the query's `e0`
///   direction, so every query's logit against them is high -- but their *values*
///   are near-zero noise. They are a place to dump probability mass, not a place
///   to read information from.
/// - Tokens 60, 61 ("content") key on `e1` (also present in the query, but with a
///   smaller key gain) and carry a strong, distinctive value along `e3`.
/// - Everything else is filler whose key has *exactly zero* overlap with the
///   query, so its logit is exactly `0`.
///
/// The sink structure is *given* here, as it is empirically given in a trained
/// model. What the test measures is the **consequence**: what happens to the
/// softmax when the mass-absorbing atoms are deleted.
fn build_sink_cache() -> KvCacheTensor {
    let mut rng = SplitMix64::new(0x5117_C0DE_0000_0001);
    let mut cache = KvCacheTensor::new(1, 1, SINK_HEAD_DIM).expect("valid geometry");
    for position in 0..SINK_SEQ_LEN {
        let mut keys = vec![0.0f32; SINK_HEAD_DIM];
        let mut values = vec![0.0f32; SINK_HEAD_DIM];
        if SINK_SLOTS.contains(&position) {
            keys[0] = 6.0;
            for d in 0..SINK_HEAD_DIM {
                values[d] = rng.next_symmetric_unit() * 0.2;
            }
        } else if SINK_CONTENT_SLOTS.contains(&position) {
            keys[1] = 5.0;
            values[3] = 4.0;
        } else {
            for d in 2..SINK_HEAD_DIM {
                keys[d] = rng.next_symmetric_unit() * 0.5;
            }
            for d in 0..SINK_HEAD_DIM {
                values[d] = rng.next_symmetric_unit() * 0.5;
            }
        }
        cache.append_token(&keys, &values).expect("well-formed");
    }
    cache
}

/// A query that overlaps both the sink direction (`e0`) and the content direction
/// (`e1`), and is orthogonal to every filler key.
fn sink_query() -> Vec<f32> {
    let mut query = vec![0.0f32; SINK_HEAD_DIM];
    query[0] = 4.0;
    query[1] = 4.0;
    query
}

fn sink_eval_output(cache: &KvCacheTensor) -> Vec<f32> {
    let attention = scaled_dot_product_attention(cache, 0, &sink_query(), &[SINK_SEQ_LEN])
        .expect("well-formed call");
    attention.output().to_vec()
}

/// **THE STREAMING-LLM CLAIM, DEMONSTRATED.**
///
/// Dropping the first few tokens is *catastrophic*; dropping the same number of
/// middle tokens is *free*. This is the counterintuitive, load-bearing result of
/// the paper, and it falls straight out of real attention arithmetic:
///
/// The sinks absorb ~86% of the row's mass (measured below). Deleting them does
/// not delete that mass -- softmax renormalizes, multiplying every survivor's
/// weight by `1 / (1 - 0.86) ≈ 7`. The output, previously a convex combination
/// dominated by a semantically empty sink value, becomes a convex combination
/// dominated by the content value, and it moves by many times its own magnitude.
///
/// Deleting three *middle* filler tokens removes a mass of ~0.02%, the
/// renormalizing factor is ~1.0002, and the output does not move.
#[test]
fn streaming_llm_sink_removal_is_catastrophic_middle_removal_is_not() {
    let full_cache = build_sink_cache();
    let reference = sink_eval_output(&full_cache);

    // ── First, MEASURE the sink mass. Do not assume it. ─────────────────────
    let attention = scaled_dot_product_attention(&full_cache, 0, &sink_query(), &[SINK_SEQ_LEN])
        .expect("well-formed call");
    let row = attention.weight_row(0, 0).expect("row exists");
    let sink_mass: f64 = SINK_SLOTS.iter().map(|&slot| row[slot]).sum();
    let middle_mass: f64 = SINK_MIDDLE_SLOTS.iter().map(|&slot| row[slot]).sum();
    let content_mass: f64 = SINK_CONTENT_SLOTS.iter().map(|&slot| row[slot]).sum();
    assert!(
        sink_mass > 0.5,
        "the premise: 3 of 64 tokens absorb the majority of the attention mass; measured {sink_mass}"
    );
    assert!(
        middle_mass < 0.001,
        "the 3 middle tokens must be attentionally negligible; measured {middle_mass}"
    );

    // ── A: drop the 3 sinks. B: drop 3 middle tokens. Same token count. ─────
    let mut without_sinks = build_sink_cache();
    let keep_after_sinks: Vec<usize> = (3..SINK_SEQ_LEN).collect();
    without_sinks
        .retain_tokens(&keep_after_sinks)
        .expect("valid selection");
    let sink_removed_output = sink_eval_output(&without_sinks);
    let sink_removal_deviation = relative_deviation(&reference, &sink_removed_output);

    let mut without_middle = build_sink_cache();
    let keep_after_middle: Vec<usize> = (0..SINK_SEQ_LEN)
        .filter(|slot| !SINK_MIDDLE_SLOTS.contains(slot))
        .collect();
    without_middle
        .retain_tokens(&keep_after_middle)
        .expect("valid selection");
    let middle_removed_output = sink_eval_output(&without_middle);
    let middle_removal_deviation = relative_deviation(&reference, &middle_removed_output);

    assert_eq!(without_sinks.seq_len(), without_middle.seq_len());

    println!("── the attention-sink asymmetry (3 tokens removed either way) ───");
    println!("  measured sink mass (tokens 0-2)   : {sink_mass:.6}");
    println!("  measured content mass (60, 61)    : {content_mass:.6}");
    println!("  measured middle mass (30-32)      : {middle_mass:.8}");
    println!("  deviation after dropping SINKS    : {sink_removal_deviation:.6}");
    println!("  deviation after dropping MIDDLE   : {middle_removal_deviation:.8}");

    assert!(
        sink_removal_deviation > 1.0,
        "dropping the attention sinks must move the output by MORE than its own magnitude \
         (catastrophic), got {sink_removal_deviation}"
    );
    assert!(
        middle_removal_deviation < 0.01,
        "dropping an equal number of middle tokens must be harmless, got {middle_removal_deviation}"
    );
    assert!(
        sink_removal_deviation > 100.0 * middle_removal_deviation,
        "the asymmetry is the whole point: sinks ({sink_removal_deviation}) must be >100x worse \
         than middle ({middle_removal_deviation})"
    );
}

/// The same asymmetry, now as a **policy** comparison at an identical budget.
///
/// `RecencyLru` *is* `StreamingLLM` with `sink_tokens = 0` -- it is the exact
/// ablation. Both keep the content tokens (they are recent). They differ in one
/// variable only: whether the 3 sinks survive. That one variable is the
/// difference between an output that is faithful and an output that is destroyed.
#[test]
fn streaming_llm_beats_its_own_sinkless_ablation_at_the_same_budget() {
    const BUDGET: usize = 16;
    let full_cache = build_sink_cache();
    let reference = sink_eval_output(&full_cache);

    let streaming_config = KvCompressionConfig::new(KvEvictionPolicy::StreamingLlm, BUDGET)
        .with_sink_tokens(3)
        .with_recent_window(13);
    let recency_config = KvCompressionConfig::new(KvEvictionPolicy::RecencyLru, BUDGET);

    // The ablation identity, asserted rather than asserted-about: StreamingLLM
    // with zero sinks selects exactly what the recency baseline selects.
    let sinkless = KvCompressionConfig::new(KvEvictionPolicy::StreamingLlm, BUDGET)
        .with_sink_tokens(0)
        .with_recent_window(BUDGET);
    assert_eq!(
        plan_streaming_llm(SINK_SEQ_LEN, &sinkless).expect("valid plan"),
        plan_recency_lru(SINK_SEQ_LEN, &recency_config).expect("valid plan"),
        "RecencyLru is exactly StreamingLLM with no sinks"
    );

    let mut streaming_cache = build_sink_cache();
    let streaming = KvCacheCompressor::new(streaming_config).expect("valid config");
    // Neither policy consults attention history, which is why an empty history is
    // enough for both -- a fact worth exercising.
    let mut streaming_stats = KvAttentionStats::new(SINK_SEQ_LEN, 8);
    assert!(!streaming_stats.has_history());
    let streaming_report = streaming
        .compress(&mut streaming_cache, &mut streaming_stats)
        .expect("compressible");

    let mut recency_cache = build_sink_cache();
    let recency = KvCacheCompressor::new(recency_config).expect("valid config");
    let mut recency_stats = KvAttentionStats::new(SINK_SEQ_LEN, 8);
    let recency_report = recency
        .compress(&mut recency_cache, &mut recency_stats)
        .expect("compressible");

    for &sink in &SINK_SLOTS {
        assert!(streaming_report.retained_positions.contains(&sink));
        assert!(!recency_report.retained_positions.contains(&sink));
    }
    for &content in &SINK_CONTENT_SLOTS {
        assert!(
            streaming_report.retained_positions.contains(&content),
            "both policies must keep the recent content, or this is not a controlled comparison"
        );
        assert!(recency_report.retained_positions.contains(&content));
    }

    let streaming_deviation = relative_deviation(&reference, &sink_eval_output(&streaming_cache));
    let recency_deviation = relative_deviation(&reference, &sink_eval_output(&recency_cache));

    println!("── sinks as the single controlled variable (budget 16/64) ───────");
    println!("  StreamingLLM (sinks kept)  deviation : {streaming_deviation:.6}");
    println!("  RecencyLru   (sinks lost)  deviation : {recency_deviation:.6}");

    assert!(
        streaming_deviation < 0.05,
        "keeping the sinks keeps the output faithful, got {streaming_deviation}"
    );
    assert!(
        recency_deviation > 1.0,
        "losing only the sinks destroys the output, got {recency_deviation}"
    );
    assert!(
        recency_deviation > 50.0 * streaming_deviation,
        "the sinks are worth >50x in output fidelity: {recency_deviation} vs {streaming_deviation}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. INVARIANTS
// ═════════════════════════════════════════════════════════════════════════════

/// **Budget adherence, for every policy and every budget.** The retained-token
/// count is `<= budget`. Always. No exceptions, no "approximately".
#[test]
fn every_policy_respects_every_budget() {
    let base = build_headline_cache();
    let seq_len = base.seq_len();

    for budget in 1..=seq_len + 8 {
        for policy in [
            KvEvictionPolicy::H2O,
            KvEvictionPolicy::StreamingLlm,
            KvEvictionPolicy::SnapKv,
            KvEvictionPolicy::RecencyLru,
        ] {
            // Scale the mandatory regions to the budget so that the floor is
            // satisfiable; a budget below the floor is a separate (error) test.
            let recent = (budget / 2).max(1).min(budget);
            let sinks = (budget / 4).min(budget.saturating_sub(recent));
            let observation = (budget / 2).max(1).min(budget);
            let config = KvCompressionConfig::new(policy, budget)
                .with_recent_window(if policy == KvEvictionPolicy::StreamingLlm {
                    budget - sinks
                } else {
                    recent
                })
                .with_sink_tokens(sinks)
                .with_observation_window(observation);

            let compressor = KvCacheCompressor::new(config)
                .unwrap_or_else(|error| panic!("{policy:?} budget {budget}: {error}"));
            let mut cache = base.clone();
            let mut stats = headline_prefill_stats(&cache, compressor.config());
            let report = compressor
                .compress(&mut cache, &mut stats)
                .unwrap_or_else(|error| panic!("{policy:?} budget {budget}: {error}"));

            assert!(
                report.tokens_after <= budget,
                "{policy:?} kept {} tokens against a budget of {budget}",
                report.tokens_after
            );
            assert!(report.within_budget);
            assert_eq!(cache.seq_len(), report.tokens_after);
            assert_eq!(stats.num_tokens(), cache.seq_len());
            assert_eq!(
                report.retained_positions.len() + report.evicted_positions.len(),
                seq_len
            );
            // When the cache is over budget, the budget is *spent*, not merely
            // respected: an under-full cache would be leaving accuracy on the table.
            if budget < seq_len {
                assert_eq!(
                    report.tokens_after, budget,
                    "{policy:?} must use its whole budget when the cache is over it"
                );
            }
            // Retained positions are ascending and are a subset of the original.
            assert!(
                report
                    .retained_positions
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            assert!(report.retained_positions.iter().all(|&p| p < seq_len));
        }
    }
}

/// A budget at or above the sequence length must be a **strict no-op**: not one
/// byte of the cache changes, and the attention output is *bitwise* identical to
/// the uncompressed one.
#[test]
fn a_budget_at_or_above_the_sequence_length_is_a_bitwise_no_op() {
    let base = build_headline_cache();
    let reference_output = headline_eval_output(&base);
    let reference_bits = bit_pattern(&reference_output);

    for budget in [HEADLINE_SEQ_LEN, HEADLINE_SEQ_LEN + 1, HEADLINE_SEQ_LEN * 2] {
        for policy in [
            KvEvictionPolicy::H2O,
            KvEvictionPolicy::StreamingLlm,
            KvEvictionPolicy::SnapKv,
            KvEvictionPolicy::RecencyLru,
        ] {
            let config = KvCompressionConfig::new(policy, budget)
                .with_recent_window(8)
                .with_sink_tokens(4)
                .with_observation_window(8);
            let compressor = KvCacheCompressor::new(config).expect("valid config");
            let mut cache = base.clone();
            let mut stats = headline_prefill_stats(&cache, compressor.config());
            let report = compressor
                .compress(&mut cache, &mut stats)
                .expect("compressible");

            assert!(
                report.is_noop(),
                "{policy:?} budget {budget} evicted tokens"
            );
            assert_eq!(report.tokens_evicted, 0);
            assert_eq!(report.tokens_after, HEADLINE_SEQ_LEN);
            assert_eq!(report.memory_saved_ratio(), 0.0);
            assert_eq!(report.eviction_ratio(), 0.0);
            assert!(report.evicted_positions.is_empty());
            assert_eq!(
                cache, base,
                "{policy:?}: the cache tensor must be untouched"
            );

            let output = headline_eval_output(&cache);
            assert_eq!(
                bit_pattern(&output),
                reference_bits,
                "{policy:?} budget {budget}: a no-op compression must leave the attention output \
                 BITWISE identical"
            );
        }
    }
}

/// A budget below a policy's mandatory floor is a configuration error, not a
/// silent truncation of the mandatory region.
#[test]
fn a_budget_below_the_policy_floor_is_an_honest_error() {
    // H2O: floor = recent_window.
    let h2o = KvCompressionConfig::new(KvEvictionPolicy::H2O, 16).with_recent_window(32);
    assert_eq!(h2o.policy_floor(), 32);
    assert!(matches!(
        KvCacheCompressor::new(h2o),
        Err(KvCompressionError::BudgetBelowFloor {
            policy: KvEvictionPolicy::H2O,
            budget: 16,
            floor: 32,
            ..
        })
    ));

    // StreamingLLM: floor = sink_tokens + recent_window. Silently truncating this
    // would turn StreamingLLM into a plain sliding window -- the exact thing the
    // policy exists to not be.
    let streaming = KvCompressionConfig::new(KvEvictionPolicy::StreamingLlm, 8)
        .with_sink_tokens(4)
        .with_recent_window(8);
    assert_eq!(streaming.policy_floor(), 12);
    assert!(matches!(
        KvCacheCompressor::new(streaming),
        Err(KvCompressionError::BudgetBelowFloor {
            policy: KvEvictionPolicy::StreamingLlm,
            budget: 8,
            floor: 12,
            ..
        })
    ));

    // SnapKV: floor = observation_window.
    let snap = KvCompressionConfig::new(KvEvictionPolicy::SnapKv, 8).with_observation_window(16);
    assert_eq!(snap.policy_floor(), 16);
    assert!(matches!(
        KvCacheCompressor::new(snap),
        Err(KvCompressionError::BudgetBelowFloor {
            policy: KvEvictionPolicy::SnapKv,
            budget: 8,
            floor: 16,
            ..
        })
    ));

    // A budget exactly *at* the floor is fine.
    let exact = KvCompressionConfig::new(KvEvictionPolicy::StreamingLlm, 12)
        .with_sink_tokens(4)
        .with_recent_window(8);
    assert!(KvCacheCompressor::new(exact).is_ok());
}

#[test]
fn structurally_invalid_configs_are_rejected() {
    assert!(matches!(
        KvCompressionConfig::new(KvEvictionPolicy::RecencyLru, 0).validate(),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
    // An even pooling kernel cannot be centred on a token.
    assert!(matches!(
        KvCompressionConfig::new(KvEvictionPolicy::SnapKv, 32)
            .with_pooling(KvSnapPooling::Max, 8)
            .validate(),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
    assert!(matches!(
        KvCompressionConfig::new(KvEvictionPolicy::SnapKv, 32)
            .with_pooling(KvSnapPooling::Max, 0)
            .validate(),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
    for decay in [0.0f64, -0.5, 1.5, f64::NAN] {
        assert!(
            matches!(
                KvCompressionConfig::new(KvEvictionPolicy::H2O, 32)
                    .with_score_decay(Some(decay))
                    .validate(),
                Err(KvCompressionError::InvalidConfig { .. })
            ),
            "score_decay = {decay} must be rejected"
        );
    }
    assert!(
        KvCompressionConfig::new(KvEvictionPolicy::H2O, 32)
            .with_score_decay(Some(1.0))
            .validate()
            .is_ok()
    );
    // SnapKV with no electorate cannot vote.
    assert!(matches!(
        KvCompressionConfig::new(KvEvictionPolicy::SnapKv, 32)
            .with_observation_window(0)
            .validate(),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
}

/// A score-driven policy with no attention history is an error, not a silent
/// fallback to "keep the earliest tokens" -- which is what a top-k over an
/// all-zero score vector would quietly do.
#[test]
fn score_driven_policies_refuse_to_guess_without_history() {
    let cache = build_headline_cache();
    for policy in [KvEvictionPolicy::H2O, KvEvictionPolicy::SnapKv] {
        assert!(policy.requires_attention_history());
        let compressor = KvCacheCompressor::new(headline_config(policy)).expect("valid config");
        let mut empty_cache = cache.clone();
        let mut stats = KvAttentionStats::new(HEADLINE_SEQ_LEN, 8);
        assert!(matches!(
            compressor.compress(&mut empty_cache, &mut stats),
            Err(KvCompressionError::NoAttentionHistory { .. })
        ));
    }
    for policy in [KvEvictionPolicy::StreamingLlm, KvEvictionPolicy::RecencyLru] {
        assert!(!policy.requires_attention_history());
        let compressor = KvCacheCompressor::new(headline_config(policy)).expect("valid config");
        let mut positional_cache = cache.clone();
        let mut stats = KvAttentionStats::new(HEADLINE_SEQ_LEN, 8);
        assert!(
            compressor
                .compress(&mut positional_cache, &mut stats)
                .is_ok(),
            "{policy:?} is purely positional and must not need history"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. EDGE CASES AND UNIT-LEVEL PIECES
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn an_empty_cache_is_an_honest_error_everywhere() {
    let mut cache = KvCacheTensor::new(1, 1, 4).expect("valid geometry");
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.memory_bytes(), 0);

    let compressor = KvCacheCompressor::new(
        KvCompressionConfig::new(KvEvictionPolicy::H2O, 8).with_recent_window(4),
    )
    .expect("valid config");
    let mut stats = KvAttentionStats::new(0, 8);
    assert!(matches!(
        compressor.compress(&mut cache, &mut stats),
        Err(KvCompressionError::EmptyCache {
            operation: "compress"
        })
    ));
    assert!(matches!(
        scaled_dot_product_attention(&cache, 0, &[1.0, 0.0, 0.0, 0.0], &[0]),
        Err(KvCompressionError::EmptyCache { .. })
    ));
}

#[test]
fn a_single_token_single_head_cache_behaves() {
    let mut cache = KvCacheTensor::new(1, 1, 1).expect("valid geometry");
    cache.append_token(&[2.0], &[7.0]).expect("well-formed");
    assert_eq!(cache.seq_len(), 1);
    assert_eq!(cache.memory_bytes(), 2 * 4);
    assert_eq!(cache.bytes_per_token(), 8);

    let attention =
        scaled_dot_product_attention(&cache, 0, &[3.0], &[0]).expect("well-formed call");
    // A softmax over a single visible key is exactly 1, whatever the logit.
    assert_eq!(attention.weight_row(0, 0).expect("row"), &[1.0]);
    assert_eq!(attention.output_row(0, 0).expect("row"), &[7.0]);

    let config = KvCompressionConfig::new(KvEvictionPolicy::H2O, 1).with_recent_window(1);
    let compressor = KvCacheCompressor::new(config).expect("valid config");
    let mut stats = KvAttentionStats::for_cache(&cache, compressor.config());
    stats.accumulate(&attention, None).expect("stats match");
    let report = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");
    assert!(report.is_noop());
    assert_eq!(report.tokens_after, 1);
    assert_eq!(report.retained_positions, vec![0]);
}

#[test]
fn cache_geometry_and_shape_errors() {
    assert!(matches!(
        KvCacheTensor::new(0, 1, 1),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
    assert!(matches!(
        KvCacheTensor::new(1, 0, 1),
        Err(KvCompressionError::InvalidConfig { .. })
    ));
    assert!(matches!(
        KvCacheTensor::new(1, 1, 0),
        Err(KvCompressionError::InvalidConfig { .. })
    ));

    let mut cache = KvCacheTensor::new(3, 2, 4).expect("valid geometry");
    assert_eq!(cache.token_buffer_len(), 3 * 2 * 4);
    assert_eq!(cache.per_layer_token_stride(), 8);
    assert!(matches!(
        cache.append_token(&[0.0; 5], &[0.0; 24]),
        Err(KvCompressionError::ShapeMismatch {
            what: "key buffer",
            expected: 24,
            actual: 5
        })
    ));
    assert!(matches!(
        cache.append_token(&[0.0; 24], &[0.0; 5]),
        Err(KvCompressionError::ShapeMismatch {
            what: "value buffer",
            expected: 24,
            actual: 5
        })
    ));

    // Per-layer, per-head slicing must be exact: write a distinct marker into
    // every (layer, head) slot and read each one back.
    let mut keys = vec![0.0f32; 24];
    let mut values = vec![0.0f32; 24];
    for layer in 0..3 {
        for head in 0..2 {
            let base = (layer * 2 + head) * 4;
            keys[base] = (layer * 10 + head) as f32;
            values[base] = (100 + layer * 10 + head) as f32;
        }
    }
    cache.append_token(&keys, &values).expect("well-formed");
    for layer in 0..3 {
        for head in 0..2 {
            assert_eq!(
                cache.key(layer, 0, head).expect("in range")[0],
                (layer * 10 + head) as f32
            );
            assert_eq!(
                cache.value(layer, 0, head).expect("in range")[0],
                (100 + layer * 10 + head) as f32
            );
        }
    }
    assert!(cache.key(3, 0, 0).is_none());
    assert!(cache.key(0, 1, 0).is_none());
    assert!(cache.key(0, 0, 2).is_none());

    assert!(matches!(
        cache.retain_tokens(&[5]),
        Err(KvCompressionError::InvalidTokenSelection { .. })
    ));
    let mut two = KvCacheTensor::new(1, 1, 1).expect("valid geometry");
    two.append_token(&[1.0], &[1.0]).expect("well-formed");
    two.append_token(&[2.0], &[2.0]).expect("well-formed");
    assert!(matches!(
        two.retain_tokens(&[1, 0]),
        Err(KvCompressionError::InvalidTokenSelection { .. })
    ));
    assert!(matches!(
        two.retain_tokens(&[0, 0]),
        Err(KvCompressionError::InvalidTokenSelection { .. })
    ));
}

/// Hand-computed pooling, including the edge-clamping rule.
///
/// ```text
/// scores = [0, 0, 5, 0, 0, 1, 0]      kernel = 3 (radius 1)
///
/// max-pool[i] = max over [i-1, i+1] clamped to the array:
///   i=0: max(0, 0)       = 0
///   i=1: max(0, 0, 5)    = 5
///   i=2: max(0, 5, 0)    = 5
///   i=3: max(5, 0, 0)    = 5
///   i=4: max(0, 0, 1)    = 1
///   i=5: max(0, 1, 0)    = 1
///   i=6: max(1, 0)       = 1
///
/// mean-pool divides by the CLAMPED width, not by the kernel:
///   i=0: (0 + 0) / 2     = 0
///   i=2: (0 + 5 + 0) / 3 = 5/3
///   i=6: (1 + 0) / 2     = 0.5
/// ```
#[test]
fn pooling_is_exact_and_clamps_at_the_edges() {
    let scores = [0.0f64, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0];

    let maxed = pool_scores(&scores, 3, KvSnapPooling::Max);
    assert_eq!(maxed, vec![0.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0]);

    let meaned = pool_scores(&scores, 3, KvSnapPooling::Mean);
    assert!((meaned[0] - 0.0).abs() < 1e-12);
    assert!((meaned[2] - 5.0 / 3.0).abs() < 1e-12);
    assert!((meaned[6] - 0.5).abs() < 1e-12);

    // No pooling is the identity; a kernel of 1 is too.
    assert_eq!(
        pool_scores(&scores, 7, KvSnapPooling::None),
        scores.to_vec()
    );
    assert_eq!(pool_scores(&scores, 1, KvSnapPooling::Max), scores.to_vec());
    assert!(pool_scores(&[], 3, KvSnapPooling::Max).is_empty());

    // A width-5 max-pool smears the peak across ±2.
    let wide = pool_scores(&scores, 5, KvSnapPooling::Max);
    assert_eq!(wide, vec![5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]);
}

#[test]
fn the_report_describes_what_actually_happened() {
    let mut cache = build_headline_cache();
    let bytes_per_token = cache.bytes_per_token();
    // keys and values (2) x layers x heads x head_dim x sizeof(f32).
    const LAYERS: usize = 1;
    let expected_bytes_per_token = 2 * LAYERS * HEADLINE_HEADS * HEADLINE_HEAD_DIM * 4;
    assert_eq!(bytes_per_token, expected_bytes_per_token);
    assert_eq!(cache.memory_bytes(), bytes_per_token * HEADLINE_SEQ_LEN);

    let compressor =
        KvCacheCompressor::new(headline_config(KvEvictionPolicy::H2O)).expect("valid config");
    let mut stats = headline_prefill_stats(&cache, compressor.config());
    let report = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");

    assert_eq!(report.policy, KvEvictionPolicy::H2O);
    assert_eq!(report.tokens_before, HEADLINE_SEQ_LEN);
    assert_eq!(report.tokens_after, HEADLINE_BUDGET);
    assert_eq!(report.tokens_evicted, HEADLINE_SEQ_LEN - HEADLINE_BUDGET);
    assert_eq!(report.budget, HEADLINE_BUDGET);
    assert!(report.within_budget);
    assert!(!report.is_noop());
    assert_eq!(report.bytes_before, bytes_per_token * HEADLINE_SEQ_LEN);
    assert_eq!(report.bytes_after, bytes_per_token * HEADLINE_BUDGET);
    assert!((report.memory_saved_ratio() - 0.75).abs() < 1e-12);
    assert!((report.eviction_ratio() - 0.75).abs() < 1e-12);
    assert_eq!(report.retained_positions.len(), HEADLINE_BUDGET);
    assert_eq!(
        report.evicted_positions.len(),
        HEADLINE_SEQ_LEN - HEADLINE_BUDGET
    );
    // Retained and evicted partition the original positions exactly.
    let mut all: Vec<usize> = report
        .retained_positions
        .iter()
        .chain(report.evicted_positions.iter())
        .copied()
        .collect();
    all.sort_unstable();
    assert_eq!(all, (0..HEADLINE_SEQ_LEN).collect::<Vec<_>>());
    // And the cache's surviving positions are exactly what the report claims.
    assert_eq!(cache.positions(), report.retained_positions.as_slice());

    // A report is serializable (it is a serving-system observability record).
    let json = serde_json::to_string(&report).expect("serializable");
    let restored: crate::kv_cache_compression::KvCompressionReport =
        serde_json::from_str(&json).expect("deserializable");
    assert_eq!(restored, report);

    let config_json = serde_json::to_string(compressor.config()).expect("config is serializable");
    let restored_config: KvCompressionConfig =
        serde_json::from_str(&config_json).expect("deserializable");
    assert_eq!(&restored_config, compressor.config());
}

/// Compression is idempotent: recompressing an already-compressed cache under the
/// same budget changes nothing, and repeated compression converges rather than
/// eroding the cache to nothing.
#[test]
fn repeated_compression_converges() {
    let mut cache = build_headline_cache();
    let compressor =
        KvCacheCompressor::new(headline_config(KvEvictionPolicy::H2O)).expect("valid config");
    let mut stats = headline_prefill_stats(&cache, compressor.config());

    let first = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");
    assert_eq!(first.tokens_after, HEADLINE_BUDGET);

    let snapshot = cache.clone();
    let second = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");
    assert!(
        second.is_noop(),
        "a within-budget cache must not be evicted again"
    );
    assert_eq!(cache, snapshot);
    assert_eq!(second.retained_positions, first.retained_positions);

    // The heavy hitters survive an arbitrary number of passes.
    for _ in 0..5 {
        compressor
            .compress(&mut cache, &mut stats)
            .expect("compressible");
    }
    for &salient in &HEADLINE_SALIENT {
        assert!(cache.positions().contains(&salient));
    }
    assert_eq!(cache.seq_len(), HEADLINE_BUDGET);
}

/// Eviction really does free memory in *every* layer and *every* head -- not just
/// in the layer whose attention was scored.
#[test]
fn eviction_shrinks_every_layer_and_every_head() {
    let mut rng = SplitMix64::new(0x5EED_0020);
    let mut cache = KvCacheTensor::new(4, 3, 6).expect("valid geometry");
    for _ in 0..40 {
        let keys: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit())
            .collect();
        let values: Vec<f32> = (0..cache.token_buffer_len())
            .map(|_| rng.next_symmetric_unit())
            .collect();
        cache.append_token(&keys, &values).expect("well-formed");
    }
    let before = cache.memory_bytes();

    let config = KvCompressionConfig::new(KvEvictionPolicy::RecencyLru, 10);
    let compressor = KvCacheCompressor::new(config).expect("valid config");
    let mut stats = KvAttentionStats::new(40, 8);
    let report = compressor
        .compress(&mut cache, &mut stats)
        .expect("compressible");

    assert_eq!(cache.seq_len(), 10);
    assert_eq!(cache.memory_bytes(), before / 4);
    assert!((report.memory_saved_ratio() - 0.75).abs() < 1e-12);
    // Every layer and every head must still be addressable at the new length --
    // and must NOT be addressable beyond it.
    for layer in 0..4 {
        for head in 0..3 {
            assert!(cache.key(layer, 9, head).is_some());
            assert!(cache.value(layer, 9, head).is_some());
            assert!(cache.key(layer, 10, head).is_none());
        }
    }
    assert_eq!(cache.positions(), (30..40).collect::<Vec<_>>().as_slice());
}
