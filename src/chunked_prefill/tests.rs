#![allow(
    // The exactness claim is *bit-for-bit*, so f32/f64 outputs are compared with
    // `assert_eq!`, not a tolerance. Every such comparison in this file is
    // exact-by-construction, and that is the whole point — see the module docs.
    clippy::float_cmp,
    // `cache`/`caches`, `keys`/`key`, `chunk`/`chunks`, `pos`/`positions` are the
    // vocabulary of this domain; renaming them to satisfy the lint would obscure
    // the maths.
    clippy::similar_names,
    // Token/slot/tick counts are `usize` and are widened to `u64`/`f64` for
    // arithmetic. Every count here is far below 2^53.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    // Fixture builders index parallel `[head][dim]` buffers by hand; a range loop
    // is the clearest way to say "for each head".
    clippy::needless_range_loop,
    // The exhaustive-composition and stall-free enumerations are long, linear
    // narratives; splitting them would make the measurement harder to follow.
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::unreadable_literal
)]
//! Tests for `Sarathi-Serve` chunked prefill.
//!
//! The suite is ordered by what it is allowed to assume:
//!
//! 1. **The piecewise mask** ([`mask`](super::mask)) in isolation — its shape, its
//!    two segments, its closed-form cell count. Everything downstream trusts it.
//! 2. **The exactness invariant** — the headline. Over **every** composition of a
//!    10-token prompt into chunks (all `2^9 = 512`), chunked prefill is asserted
//!    **bit-for-bit** equal to a one-shot prefill computed with
//!    [`scaled_dot_product_attention`] — an oracle this module did not write. Then
//!    the same 512 compositions again against a cache that already holds a shared
//!    prefix, and again against a *compressed* (gappy-position) cache.
//! 3. **Position bookkeeping** — absolute stream positions across chunk boundaries,
//!    element-wise against the one-shot positions.
//! 4. **The mutation check** — three deliberate boundary bugs, each shown to make
//!    the exactness test (and the mask audit) *fail*. A test that cannot fail is
//!    not a test.
//! 5. **The cost identities** — arithmetic is invariant across compositions; `KV`
//!    traffic is not, and grows by exactly `(m+1)/2`.
//! 6. **The stall-free property** — a pure integer invariant, enumerated
//!    exhaustively over small workloads, with the exact worst case reported, and
//!    shown to be *violated* by the classical baseline.
//! 7. **The scheduler↔executor seam** — a scheduled trace's chunk sizes, executed,
//!    still reproduce the one-shot prefill.
//! 8. **Errors and edge cases.**

use std::collections::BTreeSet;

use crate::kv_cache_compression::{KvAttentionOutput, KvCacheTensor, scaled_dot_product_attention};

use super::{
    ChunkedPrefill, ChunkedPrefillConfig, ChunkedPrefillError, PrefillGeometry,
    PrefillMaskGeometry, PrefillModel, PrefillPacking, PrefillPlan, PrefillScheduler,
    PrefillSequenceSpec, PrefillTensorModel, TokenBudget,
};

// ── Deterministic tensors ────────────────────────────────────────────────────

/// `SplitMix64` — a tiny, high-quality, fully deterministic generator, written
/// out here because the crate forbids a `rand` dependency and because a
/// reproducible seed is exactly what these fixtures want: any failure replays
/// from its seed alone.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// A finite `f32` in roughly `[-1.5, 1.5)`. Genuinely varied magnitudes, so
    /// the softmax is a real distribution and not a near-uniform one — which is
    /// what makes "the extra masked columns contribute exactly zero" a claim with
    /// something to prove.
    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (unit - 0.5) * 3.0
    }
}

/// A `[token][layer][head][dim]` buffer of `prompt_tokens` tokens, filled from
/// `seed`.
fn tensor(geometry: PrefillGeometry, prompt_tokens: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..prompt_tokens * geometry.token_stride())
        .map(|_| rng.next_f32())
        .collect()
}

/// A model whose `K`, `V` and `Q` are three independent deterministic streams.
fn build_model(geometry: PrefillGeometry, prompt_tokens: usize, seed: u64) -> PrefillTensorModel {
    PrefillTensorModel::new(
        geometry,
        prompt_tokens,
        tensor(geometry, prompt_tokens, seed ^ 0xA1),
        tensor(geometry, prompt_tokens, seed ^ 0xB2),
        tensor(geometry, prompt_tokens, seed ^ 0xC3),
    )
    .expect("fixture model")
}

/// A cache holding `prefix_tokens` committed tokens from a distinct stream, ready
/// to stand in for a shared prefix or an earlier conversational turn.
fn prefix_cache(geometry: PrefillGeometry, prefix_tokens: usize, seed: u64) -> KvCacheTensor {
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .expect("cache geometry");
    let mut rng_k = SplitMix64::new(seed ^ 0xD4);
    let mut rng_v = SplitMix64::new(seed ^ 0xE5);
    let stride = geometry.token_stride();
    for _ in 0..prefix_tokens {
        let keys: Vec<f32> = (0..stride).map(|_| rng_k.next_f32()).collect();
        let values: Vec<f32> = (0..stride).map(|_| rng_v.next_f32()).collect();
        cache.append_token(&keys, &values).expect("append prefix");
    }
    cache
}

/// One layer's `[head][query][dim]` query buffer covering **all** prompt tokens —
/// the one-shot oracle's queries.
fn full_query_buffer(
    model: &PrefillTensorModel,
    geometry: PrefillGeometry,
    layer: usize,
) -> Vec<f32> {
    let num_heads = geometry.num_heads();
    let head_dim = geometry.head_dim();
    let prompt = model.prompt_tokens();
    let mut buffer = vec![0.0f32; num_heads * prompt * head_dim];
    for token in 0..prompt {
        let query = model.token_query(layer, token).expect("token query");
        for head in 0..num_heads {
            let source = &query[head * head_dim..(head + 1) * head_dim];
            let start = (head * prompt + token) * head_dim;
            buffer[start..start + head_dim].copy_from_slice(source);
        }
    }
    buffer
}

/// The ground truth: append the whole prompt to `base` in one shot, then run one
/// attention call per layer over every prompt query. Returns the finished cache
/// and the per-layer outputs.
///
/// This is built entirely from [`KvCacheTensor::append_token`] and
/// [`scaled_dot_product_attention`] — code this module did not write.
fn one_shot(
    model: &PrefillTensorModel,
    geometry: PrefillGeometry,
    base: &KvCacheTensor,
) -> (KvCacheTensor, Vec<KvAttentionOutput>) {
    let mut cache = base.clone();
    let prefix = cache.seq_len();
    for token in 0..model.prompt_tokens() {
        cache
            .append_token(
                model.token_keys(token).expect("keys"),
                model.token_values(token).expect("values"),
            )
            .expect("append prompt token");
    }
    // The prompt's true absolute positions, read from the cache — which is the
    // whole point of positions surviving eviction. After a compression pass the
    // stream counter is *ahead* of `seq_len`, so these are gappy-aware and are
    // NOT simply `prefix..prefix + P`.
    let positions: Vec<usize> = cache.positions()[prefix..].to_vec();
    let outputs = (0..geometry.num_layers())
        .map(|layer| {
            let queries = full_query_buffer(model, geometry, layer);
            scaled_dot_product_attention(&cache, layer, &queries, &positions).expect("oracle attn")
        })
        .collect();
    (cache, outputs)
}

/// The `512` compositions of `n` into positive parts, as chunk-length vectors.
fn compositions(n: usize) -> Vec<Vec<usize>> {
    assert!(n >= 1);
    let cuts = n - 1;
    (0..1u32 << cuts)
        .map(|mask| {
            let mut lengths = Vec::new();
            let mut run = 1usize;
            for bit in 0..cuts {
                if mask >> bit & 1 == 1 {
                    lengths.push(run);
                    run = 1;
                } else {
                    run += 1;
                }
            }
            lengths.push(run);
            lengths
        })
        .collect()
}

/// Assert two attention outputs agree **bit for bit** on every output element.
fn assert_output_bit_identical(
    run: &super::PrefillRun,
    oracle: &[KvAttentionOutput],
    geometry: PrefillGeometry,
    context: &str,
) {
    for layer in 0..geometry.num_layers() {
        for head in 0..geometry.num_heads() {
            for token in 0..run.output().prompt_tokens() {
                let got = run.output().row(layer, head, token).expect("chunked row");
                let want = oracle[layer].output_row(head, token).expect("oracle row");
                assert_eq!(
                    got, want,
                    "{context}: output row (layer {layer}, head {head}, token {token}) diverged"
                );
            }
        }
    }
}

fn small_geometry() -> PrefillGeometry {
    PrefillGeometry::new(3, 2, 4).expect("geometry")
}

// ── 1. The piecewise mask in isolation ───────────────────────────────────────

#[test]
fn mask_glues_an_unmasked_prefix_to_an_inclusive_triangle() {
    // committed = 3, chunk = 4  ⇒  7 keys, 4 queries.
    let mask = PrefillMaskGeometry::new(3, 4);
    assert_eq!(mask.num_queries(), 4);
    assert_eq!(mask.num_keys(), 7);

    // The prefix segment (keys 0..3) is visible to every query.
    for query in 0..4 {
        for key in 0..3 {
            assert!(
                mask.is_visible(query, key),
                "prefix key {key} hidden from q{query}"
            );
        }
    }
    // The diagonal segment (keys 3..7) is an *inclusive* lower triangle: query j
    // sees chunk key i iff i <= j.
    let expected = [
        [true, false, false, false],
        [true, true, false, false],
        [true, true, true, false],
        [true, true, true, true],
    ];
    for query in 0..4 {
        for offset in 0..4 {
            assert_eq!(
                mask.is_visible(query, 3 + offset),
                expected[query][offset],
                "diagonal cell (q{query}, offset {offset})"
            );
        }
    }
}

#[test]
fn mask_cell_count_matches_the_closed_form_and_the_bitmap() {
    for committed in 0..6 {
        for chunk in 1..7 {
            let mask = PrefillMaskGeometry::new(committed, chunk);
            let counted = mask.to_bitmap().iter().filter(|&&v| v).count() as u64;
            let closed = (chunk * committed + chunk * (chunk + 1) / 2) as u64;
            assert_eq!(
                mask.cells(),
                closed,
                "closed form (c={committed}, n={chunk})"
            );
            assert_eq!(counted, closed, "bitmap count (c={committed}, n={chunk})");
            assert_eq!(mask.kv_slot_visits(), (committed + chunk) as u64);
        }
    }
}

// ── 2. THE exactness invariant: chunked == one-shot, bit for bit ──────────────

/// The headline. Over **all 512** compositions of a 10-token prompt, chunked
/// prefill produces a cache and an attention output that are *identical* — not
/// close, identical — to the one-shot oracle's. The tolerance is `0.0`.
#[test]
fn every_composition_of_a_10_token_prompt_matches_one_shot_exactly() {
    let geometry = small_geometry();
    let prompt_tokens = 10;
    let model = build_model(geometry, prompt_tokens, 0x5EED);
    let empty = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .expect("cache");
    let (oracle_cache, oracle_outputs) = one_shot(&model, geometry, &empty);

    let comps = compositions(prompt_tokens);
    assert_eq!(comps.len(), 512, "there are exactly 2^9 compositions of 10");

    for lengths in &comps {
        let plan = PrefillPlan::explicit(lengths).expect("plan");
        // A generous budget so the plan is legal; exactness does not depend on it.
        let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(
            TokenBudget::new(prompt_tokens, prompt_tokens - 1).expect("budget"),
        ));
        let mut cache = empty.clone();
        let run = engine.run(&model, &plan, &mut cache).expect("run");

        // (i) The KV tensor is bitwise identical (KvCacheTensor: PartialEq).
        assert_eq!(
            cache, oracle_cache,
            "KV tensor diverged for split {lengths:?}"
        );
        // (ii) Every attention output element is bitwise identical.
        assert_output_bit_identical(
            &run,
            &oracle_outputs,
            geometry,
            &format!("split {lengths:?}"),
        );
        // (iii) The mask audit ran on every chunk and passed (it is on by default).
        assert_eq!(run.chunks().len(), lengths.len());
    }
}

/// The same 512 compositions, but the cache already holds a **shared prefix** of
/// 5 committed tokens. This is the case that a mask built from slot *indices*
/// (rather than absolute positions) silently gets wrong.
#[test]
fn every_composition_matches_one_shot_over_a_shared_prefix() {
    let geometry = small_geometry();
    let prompt_tokens = 10;
    let prefix_tokens = 5;
    let model = build_model(geometry, prompt_tokens, 0xC0FFEE);
    let base = prefix_cache(geometry, prefix_tokens, 0x1234);
    let (oracle_cache, oracle_outputs) = one_shot(&model, geometry, &base);

    for lengths in &compositions(prompt_tokens) {
        let plan = PrefillPlan::explicit(lengths).expect("plan");
        let engine = ChunkedPrefill::new(ChunkedPrefillConfig::default());
        let mut cache = base.clone();
        let run = engine.run(&model, &plan, &mut cache).expect("run");

        assert_eq!(cache, oracle_cache, "KV diverged (prefix) for {lengths:?}");
        assert_output_bit_identical(
            &run,
            &oracle_outputs,
            geometry,
            &format!("prefix split {lengths:?}"),
        );
        // The prefix is never rewritten, and the prompt's absolute positions start
        // where the prefix left off.
        assert_eq!(run.stats().prefix_tokens(), prefix_tokens);
        assert_eq!(
            run.stream_positions(),
            (prefix_tokens..prefix_tokens + prompt_tokens).collect::<Vec<_>>()
        );
    }
}

/// The shared prefix, but **compressed** first, so its surviving positions are
/// gappy — `[0, 1, 57, 91, ...]`. The chunk's positions still start above the
/// largest surviving one, so the prefix stays fully visible; a slot-index mask
/// would misfire on exactly this.
#[test]
fn every_composition_matches_one_shot_over_a_gappy_compressed_prefix() {
    let geometry = small_geometry();
    let prompt_tokens = 8;

    // Build a long prefix, then evict the middle, leaving gappy positions.
    let mut base = prefix_cache(geometry, 12, 0x9999);
    base.retain_tokens(&[0, 1, 2, 9, 10, 11])
        .expect("evict middle");
    let survivors = base.positions().to_vec();
    assert_eq!(
        survivors,
        vec![0, 1, 2, 9, 10, 11],
        "prefix positions are gappy"
    );

    let model = build_model(geometry, prompt_tokens, 0x4242);
    let (oracle_cache, oracle_outputs) = one_shot(&model, geometry, &base);

    for lengths in &compositions(prompt_tokens) {
        let plan = PrefillPlan::explicit(lengths).expect("plan");
        let engine = ChunkedPrefill::new(ChunkedPrefillConfig::default());
        let mut cache = base.clone();
        let run = engine.run(&model, &plan, &mut cache).expect("run");
        assert_eq!(cache, oracle_cache, "KV diverged (gappy) for {lengths:?}");
        assert_output_bit_identical(
            &run,
            &oracle_outputs,
            geometry,
            &format!("gappy split {lengths:?}"),
        );
    }

    // The prompt's positions continue the stream past the *counter*, not past the
    // largest surviving slot index — 12, 13, ... — which is exactly why the gap
    // did not confuse the mask.
    let engine = ChunkedPrefill::new(ChunkedPrefillConfig::default());
    let mut cache = base.clone();
    let run = engine
        .run(
            &model,
            &PrefillPlan::uniform(prompt_tokens, 3).unwrap(),
            &mut cache,
        )
        .expect("run");
    assert_eq!(
        run.stream_positions(),
        (12..12 + prompt_tokens).collect::<Vec<_>>()
    );
}

// ── 3. Position bookkeeping ───────────────────────────────────────────────────

#[test]
fn absolute_positions_match_one_shot_element_wise_across_boundaries() {
    let geometry = small_geometry();
    let prompt_tokens = 10;
    let model = build_model(geometry, prompt_tokens, 0x7);

    for prefix_tokens in [0usize, 1, 4, 13] {
        let base = prefix_cache(geometry, prefix_tokens, 0x55);
        let one_shot_positions: Vec<usize> =
            (prefix_tokens..prefix_tokens + prompt_tokens).collect();

        for lengths in [vec![10], vec![3, 3, 4], vec![1; 10], vec![7, 1, 2]] {
            let plan = PrefillPlan::explicit(&lengths).expect("plan");
            let engine = ChunkedPrefill::new(ChunkedPrefillConfig::default());
            let mut cache = base.clone();
            let run = engine.run(&model, &plan, &mut cache).expect("run");

            // The whole prompt's positions, element for element.
            assert_eq!(run.stream_positions(), one_shot_positions);

            // And per chunk: each chunk's positions are the right absolute slice,
            // and its committed-key count is the prefix plus everything before it.
            let mut cursor = prefix_tokens;
            for report in run.chunks() {
                let n = report.chunk().token_count();
                assert_eq!(report.committed_keys(), cursor);
                assert_eq!(
                    report.stream_positions(),
                    (cursor..cursor + n).collect::<Vec<_>>()
                );
                cursor += n;
            }
        }
    }
}

// ── 4. The mutation check: the exactness test has teeth ───────────────────────

/// Re-run a prompt's chunks with a **deliberately broken** boundary and return
/// its per-layer output. When `chunk_local_positions` is set, each chunk hands
/// the kernel the positions `0..n` instead of the absolute
/// `committed..committed+n`.
///
/// This mirrors [`ChunkedPrefill::run`]'s inner loop closely enough to be a
/// faithful "what if the boundary were off by one", and it is confined to the
/// test module — the real executor is never mutated.
fn run_with_boundary_bug(
    model: &PrefillTensorModel,
    geometry: PrefillGeometry,
    base: &KvCacheTensor,
    lengths: &[usize],
    chunk_local_positions: bool,
) -> Vec<Vec<f32>> {
    let mut cache = base.clone();
    let head_dim = geometry.head_dim();
    let num_heads = geometry.num_heads();
    let prompt = model.prompt_tokens();
    let mut outputs = vec![vec![0.0f32; num_heads * prompt * head_dim]; geometry.num_layers()];
    let mut start = 0usize;
    for &n in lengths {
        let committed = cache.seq_len();
        for token in start..start + n {
            cache
                .append_token(
                    model.token_keys(token).unwrap(),
                    model.token_values(token).unwrap(),
                )
                .unwrap();
        }
        // THE BUG: chunk-local positions (0..n) instead of the absolute
        // committed..committed+n. For the first chunk over an empty base these
        // coincide; for any later chunk they are wrong by the committed width — a
        // query that should see the whole prefix now sees only the tokens whose
        // absolute position happens to fall below its chunk-local index.
        let positions: Vec<usize> = if chunk_local_positions {
            (0..n).collect()
        } else {
            cache.positions()[committed..].to_vec()
        };
        for layer in 0..geometry.num_layers() {
            let mut queries = vec![0.0f32; num_heads * n * head_dim];
            for offset in 0..n {
                let q = model.token_query(layer, start + offset).unwrap();
                for head in 0..num_heads {
                    let src = &q[head * head_dim..(head + 1) * head_dim];
                    let dst = (head * n + offset) * head_dim;
                    queries[dst..dst + head_dim].copy_from_slice(src);
                }
            }
            let attn = scaled_dot_product_attention(&cache, layer, &queries, &positions).unwrap();
            for head in 0..num_heads {
                for offset in 0..n {
                    let row = attn.output_row(head, offset).unwrap();
                    let dst = (head * prompt + start + offset) * head_dim;
                    outputs[layer][dst..dst + head_dim].copy_from_slice(row);
                }
            }
        }
        start += n;
    }
    outputs
}

#[test]
fn chunk_local_position_bug_makes_the_exactness_test_fail() {
    let geometry = small_geometry();
    let prompt_tokens = 6;
    let model = build_model(geometry, prompt_tokens, 0xBAD);
    let empty = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    let (_oracle_cache, oracle_outputs) = one_shot(&model, geometry, &empty);

    // A single chunk (the whole prompt): committed is 0, so 0..n *is* absolute —
    // the bug is inert, and the buggy path still matches the oracle. This proves
    // the harness is not rigged to always disagree.
    let single = run_with_boundary_bug(&model, geometry, &empty, &[prompt_tokens], true);
    for layer in 0..geometry.num_layers() {
        assert_eq!(
            single[layer].as_slice(),
            oracle_outputs[layer].output(),
            "one-chunk buggy run must still match: the bug is inert here"
        );
    }

    // Two chunks: now chunk 1's committed width is 3, so 0..3 is wrong, and the
    // output MUST diverge from the oracle. If it did not, the exactness test could
    // never catch this bug.
    let split = run_with_boundary_bug(&model, geometry, &empty, &[3, 3], true);
    let mut diverged = false;
    for layer in 0..geometry.num_layers() {
        if split[layer].as_slice() != oracle_outputs[layer].output() {
            diverged = true;
        }
    }
    assert!(
        diverged,
        "a chunk-local-position boundary bug did NOT change the output — the exactness \
         test would be toothless"
    );
}

#[test]
fn the_mask_audit_catches_the_chunk_local_position_bug() {
    // The audit is the tripwire that fires *before* the tensor is corrupted.
    // Reproduce chunk 1 of a [3, 3] split with the buggy positions and confirm the
    // geometry audit rejects it.
    let geometry = small_geometry();
    let model = build_model(geometry, 6, 0xBAD2);
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    for token in 0..6 {
        cache
            .append_token(
                model.token_keys(token).unwrap(),
                model.token_values(token).unwrap(),
            )
            .unwrap();
    }
    // Chunk 1: committed = 3, tokens = 3. Correct positions are 3, 4, 5.
    let mask = PrefillMaskGeometry::new(3, 3);
    let head_dim = geometry.head_dim();
    let num_heads = geometry.num_heads();
    let mut queries = vec![0.0f32; num_heads * 3 * head_dim];
    for offset in 0..3 {
        let q = model.token_query(0, 3 + offset).unwrap();
        for head in 0..num_heads {
            let src = &q[head * head_dim..(head + 1) * head_dim];
            let dst = (head * 3 + offset) * head_dim;
            queries[dst..dst + head_dim].copy_from_slice(src);
        }
    }
    let bug = scaled_dot_product_attention(&cache, 0, &queries, &[0, 1, 2]).unwrap();
    assert!(
        matches!(
            mask.audit(&bug),
            Err(ChunkedPrefillError::MaskDivergence { .. })
        ),
        "the audit must reject chunk-local positions"
    );

    // The correct positions pass.
    let ok = scaled_dot_product_attention(&cache, 0, &queries, &[3, 4, 5]).unwrap();
    assert!(
        mask.audit(&ok).is_ok(),
        "correct positions must pass the audit"
    );
}

#[test]
fn a_strict_triangle_mask_disagrees_with_the_kernel() {
    // A *geometry* off-by-one: the diagonal built with `<` (exclusive) instead of
    // `<=` (inclusive) loses self-attention. Checked against the real kernel, so
    // the disagreement is caught. Implemented as a local re-derivation, since the
    // real geometry cannot be mutated.
    let geometry = small_geometry();
    let model = build_model(geometry, 4, 0x33);
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    for token in 0..4 {
        cache
            .append_token(
                model.token_keys(token).unwrap(),
                model.token_values(token).unwrap(),
            )
            .unwrap();
    }
    let head_dim = geometry.head_dim();
    let num_heads = geometry.num_heads();
    let mut queries = vec![0.0f32; num_heads * 4 * head_dim];
    for offset in 0..4 {
        let q = model.token_query(0, offset).unwrap();
        for head in 0..num_heads {
            let src = &q[head * head_dim..(head + 1) * head_dim];
            let dst = (head * 4 + offset) * head_dim;
            queries[dst..dst + head_dim].copy_from_slice(src);
        }
    }
    let attention = scaled_dot_product_attention(&cache, 0, &queries, &[0, 1, 2, 3]).unwrap();

    // Correct geometry passes.
    let correct = PrefillMaskGeometry::new(0, 4);
    assert!(correct.audit(&attention).is_ok());

    // The strict variant, checked by hand: it hides the self-cell (i == j), which
    // the kernel shows as visible.
    let strict_visible = |query: usize, key: usize| -> bool { key < query };
    let mut caught = false;
    for query in 0..attention.num_queries() {
        for key in 0..attention.num_keys() {
            if strict_visible(query, key) != attention.is_visible(query, key) {
                caught = true;
            }
        }
    }
    assert!(
        caught,
        "a strict (exclusive) triangle must disagree with the kernel"
    );
}

// ── 5. Cost identities ────────────────────────────────────────────────────────

#[test]
fn attention_arithmetic_is_invariant_across_every_composition() {
    // Identity 1: the Q·K dot-product count is the same for every split of the
    // prompt — and equals the one-shot figure. Checked over all 512 compositions.
    let prompt_tokens = 10;
    for prefix in [0usize, 5, 40] {
        let one_shot_cells = {
            let p = prompt_tokens as u64;
            p * prefix as u64 + p * (p + 1) / 2
        };
        for lengths in &compositions(prompt_tokens) {
            let plan = PrefillPlan::explicit(lengths).unwrap();
            assert_eq!(
                plan.attention_cells(prefix),
                one_shot_cells,
                "split {lengths:?} at prefix {prefix} changed the arithmetic"
            );
        }
    }
}

#[test]
fn kv_traffic_amplifies_by_exactly_m_plus_one_over_two() {
    // Identity 2: uniform chunking of a 512-token prompt over an empty cache costs
    // (m+1)/2 the one-shot KV traffic — the exact table in the module docs.
    let prompt_tokens = 512;
    let expected: [(usize, u64, f64); 6] = [
        (512, 512, 1.0),
        (256, 768, 1.5),
        (128, 1280, 2.5),
        (64, 2304, 4.5),
        (32, 4352, 8.5),
        (16, 8448, 16.5),
    ];
    for (budget, visits, amp) in expected {
        let plan = PrefillPlan::uniform(prompt_tokens, budget).unwrap();
        assert_eq!(
            plan.attention_cells(0),
            131_328,
            "cells invariant at B={budget}"
        );
        assert_eq!(plan.kv_slot_visits(0), visits, "kv visits at B={budget}");
        assert_eq!(
            plan.kv_read_amplification(0),
            amp,
            "amplification at B={budget}"
        );
    }
}

#[test]
fn executed_run_reports_the_same_costs_the_plan_predicts() {
    // The measured run and the a-priori plan agree, and the run's arithmetic
    // equals the one-shot figure it carries alongside.
    let geometry = small_geometry();
    let prompt_tokens = 12;
    let model = build_model(geometry, prompt_tokens, 0x88);
    for budget in [2usize, 3, 5, 12] {
        let plan = PrefillPlan::uniform(prompt_tokens, budget).unwrap();
        let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(
            TokenBudget::new(budget, budget - 1).unwrap(),
        ));
        let mut cache = KvCacheTensor::new(
            geometry.num_layers(),
            geometry.num_heads(),
            geometry.head_dim(),
        )
        .unwrap();
        let run = engine.run(&model, &plan, &mut cache).unwrap();
        let stats = run.stats();
        assert_eq!(stats.attention_cells(), plan.attention_cells(0));
        assert_eq!(stats.attention_cells(), stats.one_shot_attention_cells());
        assert_eq!(stats.kv_slot_visits(), plan.kv_slot_visits(0));
        assert!(stats.max_chunk_tokens() <= budget, "budget adherence");
    }
}

// ── 6. The stall-free property: a pure integer invariant ──────────────────────

/// Enumerate every workload of up to three sequences (prompt 1..=4, output 0..=3)
/// under three budgets, and assert the stall-free bound **exhaustively**:
///
/// * no iteration exceeds `B` tokens (budget adherence),
/// * no decoding sequence is ever skipped (`max_token_gap_ticks <= 1`),
/// * no time-between-tokens exceeds `B`,
/// * the decode batch never exceeds the cap.
///
/// A workload that would overrun the decode cap is expected to *error*, never to
/// silently misbehave. The exact worst-case time-between-tokens is reported.
#[test]
fn stall_free_bound_holds_over_every_small_workload() {
    let budgets = [
        TokenBudget::new(2, 1).unwrap(),
        TokenBudget::new(3, 1).unwrap(),
        TokenBudget::new(4, 2).unwrap(),
    ];

    let specs: Vec<PrefillSequenceSpec> = (1..=4)
        .flat_map(|prompt| {
            (0..=3).map(move |output| PrefillSequenceSpec::new(prompt, output).unwrap())
        })
        .collect();

    let mut worst_time_between = 0u64;
    let mut worst_witness = String::new();
    let mut checked = 0u64;
    let mut cap_errors = 0u64;
    let mut tight_bound_seen: BTreeSet<usize> = BTreeSet::new();

    // All ordered workloads of length 1, 2 and 3 over the spec alphabet.
    let mut workloads: Vec<Vec<PrefillSequenceSpec>> = Vec::new();
    for a in &specs {
        workloads.push(vec![*a]);
        for b in &specs {
            workloads.push(vec![*a, *b]);
            for c in &specs {
                workloads.push(vec![*a, *b, *c]);
            }
        }
    }

    for budget in budgets {
        for workload in &workloads {
            let scheduler = PrefillScheduler::new(budget, PrefillPacking::StallFree);
            match scheduler.schedule(workload) {
                Ok(schedule) => {
                    checked += 1;
                    let stats = schedule.stats();

                    // (e) budget adherence, iteration by iteration.
                    for iteration in schedule.iterations() {
                        assert!(
                            iteration.total_tokens() <= budget.total(),
                            "iteration {} spent {} > B={}",
                            iteration.tick(),
                            iteration.total_tokens(),
                            budget.total()
                        );
                    }
                    // (b) nobody skipped, and the time-between-tokens bound.
                    assert!(stats.max_token_gap_ticks() <= 1, "a decode was skipped");
                    assert!(
                        stats.max_time_between_tokens() <= budget.total() as u64,
                        "time-between-tokens {} exceeded B={}",
                        stats.max_time_between_tokens(),
                        budget.total()
                    );
                    assert!(stats.peak_decode_sequences() <= budget.max_decode_slots());
                    assert!(stats.is_stall_free(budget));

                    if stats.max_time_between_tokens() > worst_time_between {
                        worst_time_between = stats.max_time_between_tokens();
                        worst_witness = format!("{workload:?} @ B={}", budget.total());
                    }
                    if stats.max_time_between_tokens() == budget.total() as u64 {
                        tight_bound_seen.insert(budget.total());
                    }
                }
                Err(ChunkedPrefillError::DecodeCapExceeded { running, cap }) => {
                    // The only acceptable failure: admission control (not this
                    // module's job) let too many decoders run at once.
                    cap_errors += 1;
                    assert!(running > cap);
                }
                Err(other) => panic!("unexpected scheduler error: {other}"),
            }
        }
    }

    assert!(
        checked > 5_000,
        "the enumeration should be large, got {checked}"
    );
    assert!(
        cap_errors > 0,
        "some workloads must exercise the decode-cap error path"
    );
    // The bound is not merely respected — it is *tight*: for every budget some
    // workload waits exactly B tokens between two tokens. A looser claim would be
    // suspicious.
    assert_eq!(
        tight_bound_seen.into_iter().collect::<Vec<_>>(),
        vec![2, 3, 4],
        "the B-token bound is achieved (is tight) for every budget"
    );
    // The measured worst case, reported for the record.
    assert_eq!(
        worst_time_between, 4,
        "worst-case time-between-tokens over the whole sweep, witnessed by {worst_witness}"
    );
}

/// The classical baseline *violates* the bound the chunked packing keeps — which
/// is the entire reason the module exists. Same work, catastrophically worse tail.
#[test]
fn exclusive_packing_stalls_where_stall_free_does_not() {
    let budget = TokenBudget::new(8, 4).unwrap();
    let work = [
        PrefillSequenceSpec::new(1, 6).unwrap(), // a decoder, mid-flight
        PrefillSequenceSpec::new(20, 1).unwrap(), // the 20-token prompt that lands on it
    ];

    let stall_free = PrefillScheduler::new(budget, PrefillPacking::StallFree)
        .schedule(&work)
        .unwrap();
    let exclusive = PrefillScheduler::new(budget, PrefillPacking::Exclusive)
        .schedule(&work)
        .unwrap();

    // Same total work, either way.
    assert_eq!(
        stall_free.stats().total_tokens(),
        exclusive.stats().total_tokens()
    );
    assert_eq!(stall_free.stats().total_tokens(), 26);

    // Chunked keeps the promise.
    assert!(stall_free.stats().is_stall_free(budget));
    assert_eq!(stall_free.stats().max_iteration_tokens(), 8);
    assert_eq!(stall_free.stats().max_token_gap_ticks(), 1);
    assert_eq!(stall_free.stats().max_time_between_tokens(), 8);

    // Classical breaks it: a whole 20-token prefill in one iteration, and the
    // decoder waits 21 tokens for its next token.
    assert!(!exclusive.stats().is_stall_free(budget));
    assert_eq!(exclusive.stats().max_iteration_tokens(), 20);
    assert_eq!(exclusive.stats().max_token_gap_ticks(), 2);
    assert_eq!(exclusive.stats().max_time_between_tokens(), 21);
}

#[test]
fn decode_cap_overflow_is_an_error_not_a_dropped_token() {
    // Five prompts that all finish prefilling and want to keep decoding, but the
    // budget reserves room for only two decoders. Because each prompt adds a
    // decoder every iteration and each decoder lingers for four (output_tokens
    // == 4 ⇒ three decode ticks), the decode set outgrows the cap. Dropping one
    // would be a stall; the module refuses instead.
    let budget = TokenBudget::new(4, 2).unwrap();
    let work: Vec<PrefillSequenceSpec> = (0..5)
        .map(|_| PrefillSequenceSpec::new(1, 4).unwrap())
        .collect();
    let result = PrefillScheduler::new(budget, PrefillPacking::StallFree).schedule(&work);
    assert!(matches!(
        result,
        Err(ChunkedPrefillError::DecodeCapExceeded { running: 3, cap: 2 })
    ));
}

// ── 7. The scheduler↔executor seam ────────────────────────────────────────────

#[test]
fn a_scheduled_trace_executes_to_the_one_shot_prefill() {
    // The chunk sizes the scheduler produced (the residuals the decodes left) are
    // handed straight to the executor, and still reproduce the one-shot prefill
    // bit for bit. This is the join between the two halves of the module.
    let geometry = small_geometry();
    let budget = TokenBudget::new(5, 2).unwrap();

    // A long prompt sharing iterations with two shorter, already-decoding ones.
    let work = [
        PrefillSequenceSpec::new(2, 4).unwrap(),
        PrefillSequenceSpec::new(2, 4).unwrap(),
        PrefillSequenceSpec::new(17, 1).unwrap(),
    ];
    let schedule = PrefillScheduler::new(budget, PrefillPacking::StallFree)
        .schedule(&work)
        .unwrap();

    // The long prompt was genuinely chunked (its residuals vary as decoders come
    // and go), and no chunk broke the budget.
    let trace = &schedule.sequences()[2];
    let plan = trace.plan().unwrap();
    assert!(
        plan.num_chunks() > 1,
        "the long prompt must have been split"
    );
    assert!(plan.max_chunk_tokens() <= budget.total());
    assert_eq!(plan.chunk_lengths().iter().sum::<usize>(), 17);

    // Execute exactly that plan and compare to a one-shot prefill of the same 17
    // tokens.
    let model = build_model(geometry, 17, 0xACE);
    let empty = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    let (oracle_cache, oracle_outputs) = one_shot(&model, geometry, &empty);
    let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(budget));
    let mut cache = empty.clone();
    let run = engine.run(&model, &plan, &mut cache).unwrap();

    assert_eq!(cache, oracle_cache);
    assert_output_bit_identical(&run, &oracle_outputs, geometry, "scheduled trace");
}

#[test]
fn a_prefill_only_sequence_never_enters_the_decode_set() {
    // output_tokens == 0: a scoring / embedding pass. It is prefilled and then
    // done — it never decodes, so it never competes for a decode slot.
    let budget = TokenBudget::new(4, 1).unwrap();
    let work = [PrefillSequenceSpec::new(9, 0).unwrap()];
    let schedule = PrefillScheduler::new(budget, PrefillPacking::StallFree)
        .schedule(&work)
        .unwrap();
    assert_eq!(schedule.stats().peak_decode_sequences(), 0);
    assert!(schedule.sequences()[0].token_ticks().is_empty());
    assert_eq!(
        schedule.sequences()[0]
            .chunk_lengths()
            .iter()
            .sum::<usize>(),
        9
    );
    // 9 tokens in chunks of 4 ⇒ 3 iterations, all within budget.
    assert!(schedule.stats().max_iteration_tokens() <= 4);
}

// ── 8. Errors and edge cases ──────────────────────────────────────────────────

#[test]
fn budgets_without_a_prefill_reserve_are_rejected() {
    assert!(matches!(
        TokenBudget::new(0, 0),
        Err(ChunkedPrefillError::InvalidConfig { .. })
    ));
    // decode cap == total leaves nothing for the chunk.
    assert!(matches!(
        TokenBudget::new(4, 4),
        Err(ChunkedPrefillError::InvalidConfig { .. })
    ));
    let budget = TokenBudget::new(4, 3).unwrap();
    assert_eq!(budget.prefill_reserve(), 1);
    assert_eq!(budget.max_chunk_tokens(), 4);
}

#[test]
fn a_chunk_over_budget_is_refused() {
    let geometry = small_geometry();
    let model = build_model(geometry, 6, 0x1);
    // Plan has a 6-token chunk; the budget only allows 4.
    let plan = PrefillPlan::one_shot(6).unwrap();
    let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(TokenBudget::new(4, 1).unwrap()));
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    assert!(matches!(
        engine.run(&model, &plan, &mut cache),
        Err(ChunkedPrefillError::ChunkOverBudget {
            ordinal: 0,
            tokens: 6,
            budget: 4
        })
    ));
}

#[test]
fn geometry_and_length_mismatches_are_refused() {
    let geometry = small_geometry();
    let model = build_model(geometry, 6, 0x2);
    let engine = ChunkedPrefill::new(ChunkedPrefillConfig::default());

    // Wrong cache geometry.
    let mut wrong = KvCacheTensor::new(3, 2, 8).unwrap();
    assert!(matches!(
        engine.run(&model, &PrefillPlan::one_shot(6).unwrap(), &mut wrong),
        Err(ChunkedPrefillError::GeometryMismatch { .. })
    ));

    // Plan and model disagree on the prompt length.
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    assert!(matches!(
        engine.run(&model, &PrefillPlan::one_shot(5).unwrap(), &mut cache),
        Err(ChunkedPrefillError::PromptLengthMismatch {
            plan_tokens: 5,
            model_tokens: 6
        })
    ));
}

#[test]
fn non_finite_and_empty_inputs_are_refused_at_the_boundary() {
    let geometry = small_geometry();
    let stride = geometry.token_stride();
    let mut keys = vec![0.5f32; stride];
    keys[2] = f32::NAN;
    assert!(matches!(
        PrefillTensorModel::new(geometry, 1, keys, vec![0.5; stride], vec![0.5; stride]),
        Err(ChunkedPrefillError::NonFiniteInput { .. })
    ));
    assert!(matches!(
        PrefillTensorModel::new(geometry, 0, vec![], vec![], vec![]),
        Err(ChunkedPrefillError::EmptyPrompt)
    ));
    assert!(matches!(
        PrefillPlan::uniform(0, 4),
        Err(ChunkedPrefillError::EmptyPrompt)
    ));
    assert!(matches!(
        PrefillPlan::explicit(&[]),
        Err(ChunkedPrefillError::EmptyPrompt)
    ));
    assert!(matches!(
        PrefillPlan::explicit(&[3, 0, 2]),
        Err(ChunkedPrefillError::EmptyChunk { ordinal: 1 })
    ));
}

#[test]
fn retained_attention_is_available_only_when_asked_for() {
    let geometry = small_geometry();
    let model = build_model(geometry, 5, 0x3);
    let mut cache = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    let engine = ChunkedPrefill::new(
        ChunkedPrefillConfig::new(TokenBudget::new(3, 1).unwrap()).with_retained_attention(true),
    );
    let run = engine
        .run(&model, &PrefillPlan::uniform(5, 3).unwrap(), &mut cache)
        .unwrap();
    // Each chunk kept one KvAttentionOutput per layer, and its query positions are
    // the chunk's absolute positions.
    let first = &run.chunks()[0];
    let attention = first.attention(0).expect("retained attention");
    assert_eq!(attention.query_positions(), first.stream_positions());

    // Off by default: no retention, no cost.
    let mut cache2 = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    let plain = ChunkedPrefill::new(ChunkedPrefillConfig::new(TokenBudget::new(3, 1).unwrap()));
    let run2 = plain
        .run(&model, &PrefillPlan::uniform(5, 3).unwrap(), &mut cache2)
        .unwrap();
    assert!(run2.chunks()[0].attention(0).is_none());
}

#[test]
fn disabling_the_mask_audit_still_produces_the_exact_answer() {
    // The audit is a check, not a load-bearing computation: turning it off changes
    // nothing about the output, only the safety net.
    let geometry = small_geometry();
    let model = build_model(geometry, 9, 0x4);
    let empty = KvCacheTensor::new(
        geometry.num_layers(),
        geometry.num_heads(),
        geometry.head_dim(),
    )
    .unwrap();
    let (oracle_cache, oracle_outputs) = one_shot(&model, geometry, &empty);

    let engine = ChunkedPrefill::new(
        ChunkedPrefillConfig::new(TokenBudget::new(4, 1).unwrap()).with_mask_audit(false),
    );
    let mut cache = empty.clone();
    let run = engine
        .run(&model, &PrefillPlan::uniform(9, 4).unwrap(), &mut cache)
        .unwrap();
    assert_eq!(cache, oracle_cache);
    assert_output_bit_identical(&run, &oracle_outputs, geometry, "audit disabled");
    assert!(!engine.config().audit_mask());
}
