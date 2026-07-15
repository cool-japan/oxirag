#![allow(
    // Attention outputs are compared bit-for-bit against an independent oracle;
    // the comparisons are exact by construction (identical IEEE-754 operations in
    // identical order), so `to_bits` equality is deliberate, not fragile.
    clippy::float_cmp,
    // `keys`/`key`, `table`/`tables`, `block`/`blocks` pairs are unavoidable in
    // this domain and renaming them would obscure the structure.
    clippy::similar_names,
    // Token/block/step counts are `usize`/`u64` and are freely cast to `f64` for
    // ratios; the values are tiny.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    // Some measurement narratives are one long linear story.
    clippy::too_many_lines,
    clippy::unreadable_literal,
    // Fixture builders index parallel buffers by hand.
    clippy::needless_range_loop,
    clippy::items_after_statements
)]
//! Tests for `PagedAttention`-style KV block allocation and iteration-level
//! batching.
//!
//! The suite is ordered by dependency, because the later measurements are only
//! meaningful if the earlier ones hold:
//!
//! 1. **The paged attention kernel** against an oracle this module did not write
//!    ([`crate::kv_cache_compression::scaled_dot_product_attention`]), over a
//!    deliberately fragmented pool — dense *and* gappy. If the paged gather is
//!    wrong, everything built on it measures noise, so it is tested first, and
//!    tested to the bit.
//! 2. **Copy-on-write** — the parent survives a child's write, bit for bit, and
//!    the refcount is the number of referring sequences.
//! 3. **Preemption** — recompute and swap both reproduce the pre-eviction KV bit
//!    for bit.
//! 4. **Allocator conservation** — under seeded random churn, checked after every
//!    single operation.
//! 5. **Prefix sharing** — the exact block count, measured, not asserted as a
//!    trend.
//! 6. **Iteration-level batching** — a finished sequence's slots are reclaimed
//!    immediately.

use crate::kv_cache_compression::{KvCacheTensor, scaled_dot_product_attention};

use super::allocator::{KvBlockAllocator, fold_prefix_hash};
use super::attention::paged_attention;
use super::engine::{BatchKvProducer, BatchSequence, ContinuousBatchEngine};
use super::table::KvBlockTable;
use super::types::{
    BatchPreemptionMode, ContinuousBatchConfig, ContinuousBatchError, ContinuousBatchResult,
    KvBlockId,
};

// ── Deterministic KV / query fixtures ────────────────────────────────────────

/// A pseudo-random `f32` in `[-1, 1)`, a pure function of its coordinates.
///
/// This stands in for a model's forward pass, and it satisfies the only property
/// [`BatchKvProducer`] depends on: it is deterministic in `(context, position)`.
/// That is what makes prefix sharing and recompute-preemption *checkable* — two
/// requests with the same prefix get the same bytes, so a shared block is a
/// correct block, and the tests assert exactly that.
fn kv_element(context: &[u32], position: usize, is_value: bool, index: usize) -> f32 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash ^= (position as u64).wrapping_mul(0x0000_0100_0000_01b3);
    for &token in context {
        hash ^= u64::from(token);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= u64::from(is_value) << 40;
    hash = hash.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 31;
    let bits = (hash >> 40) as u32; // 24 bits
    (bits as f32 / 8_388_608.0) - 1.0
}

/// Fill one token's `[layer][head][dim]` key and value buffers.
fn fill_token(
    config: &ContinuousBatchConfig,
    context: &[u32],
    position: usize,
    keys: &mut [f32],
    values: &mut [f32],
) {
    for index in 0..config.token_buffer_len() {
        keys[index] = kv_element(context, position, false, index);
        values[index] = kv_element(context, position, true, index);
    }
}

/// The [`BatchKvProducer`] the engine tests drive, calling exactly the same
/// [`kv_element`] the standalone builders use — so an engine-built sequence and a
/// standalone-built one are byte-identical.
struct HashProducer;

impl BatchKvProducer for HashProducer {
    fn produce_kv(
        &self,
        context: &[u32],
        position: usize,
        keys: &mut [f32],
        values: &mut [f32],
    ) -> ContinuousBatchResult<()> {
        for index in 0..keys.len() {
            keys[index] = kv_element(context, position, false, index);
            values[index] = kv_element(context, position, true, index);
        }
        Ok(())
    }
}

/// A pseudo-random query element in `[-1, 1)`, `[head][query][dim]`.
fn query_element(query_position: usize, head: usize, dim: usize) -> f32 {
    let mut hash = 0x243f_6a88_85a3_08d3_u64;
    hash ^= (query_position as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash = hash.wrapping_add((head as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    hash = hash.wrapping_add((dim as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 29;
    let bits = (hash >> 40) as u32;
    (bits as f32 / 8_388_608.0) - 1.0
}

/// Build a `[head][query][dim]` query buffer for the given query positions.
fn build_queries(config: &ContinuousBatchConfig, query_positions: &[usize]) -> Vec<f32> {
    let mut queries = vec![0.0f32; config.num_heads * query_positions.len() * config.head_dim];
    for head in 0..config.num_heads {
        for (query_index, &position) in query_positions.iter().enumerate() {
            for dim in 0..config.head_dim {
                let offset = (head * query_positions.len() + query_index) * config.head_dim + dim;
                queries[offset] = query_element(position, head, dim);
            }
        }
    }
    queries
}

/// Build an allocator whose free list is **exactly** `free_order`, in that order,
/// with every other block allocated (and inert).
///
/// This is how the attention tests fragment the pool: a sequence built on top of
/// this allocator grabs `free_order` in order, so its block table is scattered
/// and non-monotone — which is the whole point, because an identity block table
/// tests nothing.
fn fragmented_allocator(config: ContinuousBatchConfig, free_order: &[usize]) -> KvBlockAllocator {
    let total = config.total_blocks;
    let mut allocator = KvBlockAllocator::new(config).expect("valid config");
    let _all = allocator
        .allocate_blocks(total)
        .expect("pool holds total blocks");
    for &id in free_order {
        allocator
            .release_block(KvBlockId(id))
            .expect("id was allocated");
    }
    allocator
}

/// Append `token_ids` to a fresh table on `allocator`, computing each token's KV
/// with [`fill_token`] over the running prefix.
fn build_sequence(
    config: &ContinuousBatchConfig,
    allocator: &mut KvBlockAllocator,
    token_ids: &[u32],
) -> KvBlockTable {
    let mut table = KvBlockTable::new(config);
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    for (position, &token) in token_ids.iter().enumerate() {
        fill_token(
            config,
            &token_ids[..=position],
            position,
            &mut keys,
            &mut values,
        );
        table
            .append_token(allocator, token, &keys, &values)
            .expect("append");
    }
    table
}

/// Bit-for-bit `f32` slice equality (compares `to_bits`, so `+0.0` and `-0.0`
/// are distinguished — the strongest possible claim).
fn assert_bits_f32(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: f32 differ at {index}: {a} vs {b}"
        );
    }
}

/// Bit-for-bit `f64` slice equality.
fn assert_bits_f64(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: f64 differ at {index}: {a} vs {b}"
        );
    }
}

fn attention_config() -> ContinuousBatchConfig {
    ContinuousBatchConfig::new(3, 3, 8, 4, 40)
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE HEADLINE ORACLE — paged attention == contiguous attention, bit for bit
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn paged_attention_matches_contiguous_oracle_dense() {
    let config = attention_config();
    // A scattered, non-monotone free order: the sequence's six blocks will be
    // 13, 2, 29, 5, 20, 8 — not adjacent, not ascending.
    let free_order = [13usize, 2, 29, 5, 20, 8, 31, 1];
    let mut allocator = fragmented_allocator(config.clone(), &free_order);

    let token_ids: Vec<u32> = (0..23).map(|t| (t as u32) * 7 + 3).collect();
    let table = build_sequence(&config, &mut allocator, &token_ids);

    // The block table really is scattered — otherwise this test would be
    // checking a contiguous gather in disguise.
    assert_eq!(table.len(), 23);
    assert_eq!(table.num_blocks(), 6);
    assert_eq!(
        table.block_ids(),
        &[
            KvBlockId(13),
            KvBlockId(2),
            KvBlockId(29),
            KvBlockId(5),
            KvBlockId(20),
            KvBlockId(8)
        ]
    );

    // The oracle: the same tokens in a contiguous KvCacheTensor.
    let mut oracle =
        KvCacheTensor::new(config.num_layers, config.num_heads, config.head_dim).expect("geometry");
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    for (position, &token) in token_ids.iter().enumerate() {
        let _ = token;
        fill_token(
            &config,
            &token_ids[..=position],
            position,
            &mut keys,
            &mut values,
        );
        oracle.append_token(&keys, &values).expect("append");
    }

    let query_positions = [0usize, 4, 11, 22, 40];
    let queries = build_queries(&config, &query_positions);

    for layer in 0..config.num_layers {
        let paged = paged_attention(&table, &allocator, layer, &queries, &query_positions)
            .expect("paged attention");
        let reference = scaled_dot_product_attention(&oracle, layer, &queries, &query_positions)
            .expect("oracle attention");

        assert_eq!(paged.num_keys(), reference.num_keys());
        assert_eq!(paged.num_queries(), reference.num_queries());
        assert_bits_f32(
            paged.output(),
            reference.output(),
            &format!("dense output, layer {layer}"),
        );
        assert_bits_f64(
            paged.weights(),
            reference.weights(),
            &format!("dense weights, layer {layer}"),
        );
    }
}

#[test]
fn paged_attention_matches_contiguous_oracle_gappy() {
    let config = attention_config();
    let free_order: Vec<usize> = (0..40).rev().collect(); // 39, 38, 37, ...
    let mut allocator = fragmented_allocator(config.clone(), &free_order);

    let token_ids: Vec<u32> = (0..23).map(|t| (t as u32) * 5 + 1).collect();
    let mut table = build_sequence(&config, &mut allocator, &token_ids);

    // Keep a gappy subset of positions — as an eviction policy would leave behind.
    let keep = [1usize, 2, 3, 8, 9, 12, 18, 19, 20];
    table
        .compact_to_positions(&mut allocator, &keep)
        .expect("compact");
    assert_eq!(table.len(), keep.len());

    // The oracle builds the same gappy sequence independently: lay all 23 tokens
    // down densely, then retain the same slots (slot == position for a dense
    // build), which preserves their absolute positions exactly as compaction did.
    let mut oracle =
        KvCacheTensor::new(config.num_layers, config.num_heads, config.head_dim).expect("geometry");
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    for position in 0..23 {
        fill_token(
            &config,
            &token_ids[..=position],
            position,
            &mut keys,
            &mut values,
        );
        oracle.append_token(&keys, &values).expect("append");
    }
    oracle.retain_tokens(&keep).expect("retain");
    assert_eq!(oracle.positions(), &keep);
    assert_eq!(table.positions(&allocator).expect("positions"), keep);

    // Query 0 sits *below* the smallest surviving position (1): it sees nothing,
    // a fully-masked degenerate row that both kernels must define to zero.
    let query_positions = [0usize, 2, 9, 20, 40];
    let queries = build_queries(&config, &query_positions);

    for layer in 0..config.num_layers {
        let paged =
            paged_attention(&table, &allocator, layer, &queries, &query_positions).expect("paged");
        let reference = scaled_dot_product_attention(&oracle, layer, &queries, &query_positions)
            .expect("oracle");
        assert_bits_f32(
            paged.output(),
            reference.output(),
            &format!("gappy output, layer {layer}"),
        );
        assert_bits_f64(
            paged.weights(),
            reference.weights(),
            &format!("gappy weights, layer {layer}"),
        );

        // The degenerate row really is all-zero, not merely close.
        for head in 0..config.num_heads {
            let weight_row = paged.weight_row(head, 0).expect("row");
            assert!(
                weight_row.iter().all(|&w| w == 0.0),
                "query below all keys must be all-masked"
            );
            let output_row = paged.output_row(head, 0).expect("row");
            assert!(
                output_row.iter().all(|&o| o == 0.0),
                "all-masked query output must be zero"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. COPY-ON-WRITE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fork_shares_every_block_and_refcount_equals_referrers() {
    let config = ContinuousBatchConfig::new(2, 2, 4, 4, 40);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");

    let token_ids: Vec<u32> = (0..10).map(|t| t as u32 + 100).collect();
    let parent = build_sequence(&config, &mut allocator, &token_ids);
    assert_eq!(parent.num_blocks(), 3); // two full + one partial (2 slots)
    for &block in parent.block_ids() {
        assert_eq!(allocator.ref_count(block), 1);
    }

    // Fork three children; every block is now referenced by parent + 3 children.
    let child_a = parent.fork(&mut allocator).expect("fork");
    let child_b = parent.fork(&mut allocator).expect("fork");
    let child_c = parent.fork(&mut allocator).expect("fork");
    for &block in parent.block_ids() {
        assert_eq!(
            allocator.ref_count(block),
            4,
            "block shared by parent + 3 children"
        );
    }
    // Every fork points at the *same* physical blocks — not one byte was copied.
    assert_eq!(child_a.block_ids(), parent.block_ids());
    assert_eq!(child_b.block_ids(), parent.block_ids());
    assert_eq!(child_c.block_ids(), parent.block_ids());
    assert_eq!(allocator.cow_copies(), 0);

    drop(child_a);
    drop(child_b);
    drop(child_c);
}

#[test]
fn child_write_leaves_parent_bit_identical() {
    let config = ContinuousBatchConfig::new(2, 3, 4, 4, 40);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");

    let token_ids: Vec<u32> = (0..10).map(|t| t as u32 * 3 + 2).collect();
    let parent = build_sequence(&config, &mut allocator, &token_ids);

    // Snapshot the parent's KV before the fork.
    let before = parent.materialize(&allocator).expect("materialize");

    let mut child = parent.fork(&mut allocator).expect("fork");
    let shared_tail = *parent.block_ids().last().expect("tail");
    assert_eq!(
        allocator.ref_count(shared_tail),
        2,
        "tail shared before the write"
    );

    // The child appends token 10. Its tail block is the shared, partially filled
    // one, so this write must copy-on-write.
    let new_token = 999u32;
    let mut child_context = token_ids.clone();
    child_context.push(new_token);
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    fill_token(&config, &child_context, 10, &mut keys, &mut values);
    child
        .append_token(&mut allocator, new_token, &keys, &values)
        .expect("child append");

    // Exactly one copy-on-write happened.
    assert_eq!(allocator.cow_copies(), 1);
    // The parent kept the original tail, now private again; the child has a new one.
    assert_eq!(
        allocator.ref_count(shared_tail),
        1,
        "parent's tail is private after the child copied"
    );
    let child_tail = *child.block_ids().last().expect("tail");
    assert_ne!(child_tail, shared_tail, "child wrote into a fresh block");
    // The two full prefix blocks are still shared.
    for index in 0..2 {
        assert_eq!(
            allocator.ref_count(parent.block_ids()[index]),
            2,
            "full prefix blocks stay shared"
        );
    }

    // THE assertion: the parent's KV is unchanged, bit for bit.
    let after = parent.materialize(&allocator).expect("materialize");
    assert_eq!(
        before, after,
        "parent KV must be bit-identical after the child's write"
    );

    // And the child is correct too: its KV equals an independent 11-token build.
    let mut reference_allocator = KvBlockAllocator::new(config.clone()).expect("config");
    let reference_child = build_sequence(&config, &mut reference_allocator, &child_context);
    let child_kv = child.materialize(&allocator).expect("materialize");
    let reference_kv = reference_child
        .materialize(&reference_allocator)
        .expect("materialize");
    assert_eq!(
        child_kv, reference_kv,
        "child KV must equal an independent build of the same 11 tokens"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. PREEMPTION IS LOSSLESS, BIT FOR BIT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn recompute_reproduces_dense_kv_bit_for_bit() {
    let config = ContinuousBatchConfig::new(2, 2, 6, 4, 40);
    let mut allocator = fragmented_allocator(config.clone(), &(0..40).rev().collect::<Vec<_>>());

    let token_ids: Vec<u32> = (0..14).map(|t| t as u32 * 11 + 5).collect();
    let mut table = build_sequence(&config, &mut allocator, &token_ids);
    let snapshot = table.materialize(&allocator).expect("materialize");
    let positions = table.positions(&allocator).expect("positions");
    let next_position = table.next_position();

    // Preempt-by-recompute: drop every block.
    let freed = table.release(&mut allocator).expect("release");
    assert_eq!(freed, 4);
    assert!(table.is_empty());

    // Resume: re-drive the producer over the token stream, at the original
    // positions, into a fresh scattered set of blocks.
    let mut rebuilt = KvBlockTable::for_restore(&config, next_position);
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    for &position in &positions {
        fill_token(
            &config,
            &token_ids[..=position],
            position,
            &mut keys,
            &mut values,
        );
        rebuilt
            .restore_token(
                &mut allocator,
                token_ids[position],
                position,
                &keys,
                &values,
            )
            .expect("restore");
    }

    let restored = rebuilt.materialize(&allocator).expect("materialize");
    assert_eq!(
        restored, snapshot,
        "recompute must reproduce the pre-preemption KV bit for bit"
    );
    assert_eq!(
        rebuilt.next_position(),
        next_position,
        "the position counter must survive the round trip"
    );
}

#[test]
fn recompute_reproduces_gappy_kv_bit_for_bit() {
    let config = ContinuousBatchConfig::new(2, 2, 6, 4, 40);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");

    let token_ids: Vec<u32> = (0..14).map(|t| t as u32 * 13 + 1).collect();
    let mut table = build_sequence(&config, &mut allocator, &token_ids);

    // Compress it to a gappy set, exactly as an eviction policy on the resident
    // sequence would (this is the engine's non-dense recompute path).
    let keep = [1usize, 2, 3, 8, 9, 13];
    table
        .compact_to_positions(&mut allocator, &keep)
        .expect("compact");
    let snapshot = table.materialize(&allocator).expect("materialize");
    let next_position = table.next_position();
    assert_eq!(next_position, 14, "compaction never rewinds the counter");

    table.release(&mut allocator).expect("release");

    // Resume at exactly the surviving (gappy) positions.
    let mut rebuilt = KvBlockTable::for_restore(&config, next_position);
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];
    for &position in &keep {
        fill_token(
            &config,
            &token_ids[..=position],
            position,
            &mut keys,
            &mut values,
        );
        rebuilt
            .restore_token(
                &mut allocator,
                token_ids[position],
                position,
                &keys,
                &values,
            )
            .expect("restore");
    }

    let restored = rebuilt.materialize(&allocator).expect("materialize");
    assert_eq!(
        restored, snapshot,
        "gappy recompute must reproduce the KV bit for bit"
    );
    assert_eq!(rebuilt.positions(&allocator).expect("positions"), keep);
}

#[test]
fn swap_reproduces_kv_bit_for_bit_in_different_blocks() {
    let config = ContinuousBatchConfig::new(2, 2, 5, 4, 12);
    // Fragment so the round trip is forced into different physical blocks.
    let mut allocator = fragmented_allocator(config.clone(), &[7usize, 3, 11, 1, 9, 5, 0, 8]);

    let token_ids: Vec<u32> = (0..14).map(|t| t as u32 * 3 + 9).collect();
    let mut table = build_sequence(&config, &mut allocator, &token_ids);
    let original_blocks = table.block_ids().to_vec();
    let snapshot = table.materialize(&allocator).expect("materialize");

    let slab = table.swap_out(&mut allocator).expect("swap out");
    assert_eq!(slab.num_tokens(), 14);
    assert!(table.is_empty());

    let restored_table = KvBlockTable::swap_in(&mut allocator, &slab).expect("swap in");
    let restored = restored_table.materialize(&allocator).expect("materialize");
    assert_eq!(restored, snapshot, "swap must reproduce the KV bit for bit");
    assert_ne!(
        restored_table.block_ids(),
        original_blocks.as_slice(),
        "swap-in landed in different physical blocks"
    );
    for &block in restored_table.block_ids() {
        assert_eq!(allocator.ref_count(block), 1, "restored blocks are private");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. ALLOCATOR CONSERVATION UNDER SEEDED CHURN
// ═══════════════════════════════════════════════════════════════════════════

/// `splitmix64`: a tiny, dependency-free, fully deterministic PRNG for the churn.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[test]
fn allocator_conserves_blocks_under_seeded_churn() {
    let config = ContinuousBatchConfig::new(1, 2, 4, 4, 48);
    let total = config.total_blocks;
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");

    const SLOTS: usize = 10;
    let mut tables: Vec<Option<KvBlockTable>> = (0..SLOTS).map(|_| None).collect();
    let mut next_token: u32 = 0;
    let mut keys = vec![0.0f32; config.token_buffer_len()];
    let mut values = vec![0.0f32; config.token_buffer_len()];

    let mut rng = 0x0ddc0ffee0ddf00d_u64;
    let mut appends = 0u64;
    let mut forks = 0u64;
    let mut releases = 0u64;
    let mut compactions = 0u64;
    let mut swaps = 0u64;

    // The conservation law, checked after *every* operation. This closure is the
    // point of the whole test.
    let assert_conserved = |allocator: &KvBlockAllocator| {
        allocator.verify_invariants().expect("invariants hold");
        assert_eq!(
            allocator.free_blocks() + allocator.allocated_blocks(),
            total,
            "free + allocated == total"
        );
        let stats = allocator.stats();
        assert_eq!(stats.free_blocks + stats.allocated_blocks, total);
        assert_eq!(stats.allocated_blocks, total - stats.free_blocks);
    };

    for _ in 0..6000 {
        let op = splitmix64(&mut rng) % 10;
        let slot = (splitmix64(&mut rng) as usize) % SLOTS;

        match op {
            // Append a token (allocate / grow / copy-on-write).
            0..=3 => {
                if tables[slot].is_none() {
                    tables[slot] = Some(KvBlockTable::new(&config));
                }
                let token = next_token;
                next_token = next_token.wrapping_add(1);
                let position = tables[slot].as_ref().map_or(0, KvBlockTable::next_position);
                fill_token(&config, &[token], position, &mut keys, &mut values);
                let table = tables[slot].as_mut().expect("present");
                match table.append_token(&mut allocator, token, &keys, &values) {
                    Ok(_) => appends += 1,
                    Err(ContinuousBatchError::OutOfBlocks { .. }) => {}
                    Err(other) => panic!("unexpected append error: {other}"),
                }
            }
            // Fork into a free slot.
            4 | 5 => {
                let source = tables[slot].as_ref().filter(|table| !table.is_empty());
                if let Some(source) = source
                    && let Some(free_slot) = tables.iter().position(Option::is_none)
                {
                    match source.fork(&mut allocator) {
                        Ok(child) => {
                            tables[free_slot] = Some(child);
                            forks += 1;
                        }
                        Err(ContinuousBatchError::OutOfBlocks { .. }) => {}
                        Err(other) => panic!("unexpected fork error: {other}"),
                    }
                }
            }
            // Release.
            6 | 7 => {
                if let Some(mut table) = tables[slot].take() {
                    table.release(&mut allocator).expect("release");
                    releases += 1;
                }
            }
            // Compact to a random-ish subset of positions.
            8 => {
                if let Some(table) = tables[slot].as_mut() {
                    let positions = table.positions(&allocator).expect("positions");
                    if positions.len() >= 2 {
                        let keep: Vec<usize> = positions
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| index % 2 == 0)
                            .map(|(_, &position)| position)
                            .collect();
                        table
                            .compact_to_positions(&mut allocator, &keep)
                            .expect("compact shrinks, cannot OOM");
                        compactions += 1;
                    }
                }
            }
            // Swap out and immediately back in.
            _ => {
                if let Some(table) = tables[slot].as_mut()
                    && !table.is_empty()
                {
                    let slab = table.swap_out(&mut allocator).expect("swap out");
                    match KvBlockTable::swap_in(&mut allocator, &slab) {
                        Ok(restored) => {
                            *table = restored;
                            swaps += 1;
                        }
                        // A sharer's swap-out may free fewer blocks than the slab
                        // needs back; if so the sequence is dropped and its slab
                        // discarded. Conservation still holds.
                        Err(ContinuousBatchError::OutOfBlocks { .. }) => {
                            tables[slot] = None;
                        }
                        Err(other) => panic!("unexpected swap-in error: {other}"),
                    }
                }
            }
        }

        assert_conserved(&allocator);
    }

    // The churn actually exercised every operation — otherwise "conserved" would
    // be a vacuous claim.
    assert!(appends > 100, "expected many appends, got {appends}");
    assert!(forks > 10, "expected forks, got {forks}");
    assert!(releases > 10, "expected releases, got {releases}");
    assert!(compactions > 5, "expected compactions, got {compactions}");
    assert!(swaps > 5, "expected swaps, got {swaps}");

    // Drain everything and land back at an empty pool.
    for table in &mut tables {
        if let Some(mut table) = table.take() {
            table.release(&mut allocator).expect("release");
        }
    }
    assert_conserved(&allocator);
    assert_eq!(
        allocator.allocated_blocks(),
        0,
        "every block returned to the free list"
    );
    assert_eq!(allocator.free_blocks(), total);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. PREFIX SHARING SAVES BLOCKS — EXACT COUNT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn prefix_sharing_saves_exactly_the_predicted_blocks() {
    // block_size B = 4; prefix P = 16 tokens (4 full blocks); suffix S = 8 tokens
    // (2 blocks); N = 8 sequences.
    const B: usize = 4;
    const P: usize = 16;
    const S: usize = 8;
    const N: usize = 8;
    let config = ContinuousBatchConfig::new(2, 2, 4, B, 24)
        .with_max_running_sequences(N)
        .with_prefix_sharing(true);
    let mut engine = ContinuousBatchEngine::new(config).expect("config");

    let prefix: Vec<u32> = (0..P as u32).collect();
    for i in 0..N {
        let mut tokens = prefix.clone();
        // A distinct suffix per sequence, so only the *prefix* is shared.
        for j in 0..S {
            tokens.push(10_000 + (i * 100 + j) as u32);
        }
        tokens.push(20_000 + i as u32); // one decode token so it stays running
        engine.submit(tokens, P + S).expect("submit");
    }

    // One step admits all eight (the prefix of the seven followers is adopted).
    let record = engine.step(&HashProducer).expect("step");
    assert_eq!(record.admitted.len(), N);
    assert_eq!(record.running, N);

    // The exact predicted occupancy: blocks(P) + N * blocks(S).
    let predicted = P / B + N * (S / B);
    assert_eq!(predicted, 20);
    let allocated = engine.allocator().allocated_blocks();
    assert_eq!(
        allocated, predicted,
        "shared occupancy must be blocks(P) + N*blocks(suffix)"
    );

    // A non-sharing allocator would have needed N * blocks(P + S).
    let naive = N * ((P + S) / B);
    assert_eq!(naive, 48);
    assert_eq!(
        naive - allocated,
        (N - 1) * (P / B),
        "saving is exactly (N-1)*blocks(P)"
    );

    // The sharing factor is the exact ratio, measured.
    let stats = engine.allocator().stats();
    assert_eq!(stats.total_references, naive, "logical block references");
    assert_eq!(stats.blocks_saved_by_sharing(), naive - allocated);
    assert!(
        (stats.sharing_factor() - 2.4).abs() < 1e-12,
        "sharing factor {} != 2.4",
        stats.sharing_factor()
    );

    // Every one of the four prefix blocks is referenced by all N sequences.
    let first = engine.sequences()[0].table().block_ids().to_vec();
    for &block in first.iter().take(P / B) {
        assert_eq!(
            engine.allocator().ref_count(block),
            N,
            "prefix block shared by all N sequences"
        );
    }
    // The suffix blocks are private.
    for &block in first.iter().skip(P / B) {
        assert_eq!(
            engine.allocator().ref_count(block),
            1,
            "suffix block is private"
        );
    }

    // The token accounting: the prefix was computed once and adopted N-1 times.
    let engine_stats = engine.stats();
    assert_eq!(
        engine_stats.prefix_cached_tokens,
        ((N - 1) * P) as u64,
        "prefix computed once, adopted N-1 times"
    );
    assert_eq!(engine_stats.prefill_tokens, ((P + S) + (N - 1) * S) as u64);
    assert!((engine_stats.prefix_hit_rate() - 112.0 / 192.0).abs() < 1e-12);

    // And the shared prefix is *correct*: sequence 7's first P tokens attend
    // identically to a standalone build of the same prompt. (Sharing that
    // returned the wrong bytes would be caught here, not merely "saved blocks".)
    let shared_seq = &engine.sequences()[N - 1];
    // Its table holds the P+S resident (prompt) tokens; the final decode token has
    // not been produced yet, so the oracle is built over exactly those tokens.
    let resident = shared_seq.table().len();
    assert_eq!(resident, P + S);
    let query_positions = [0usize, 5, 15];
    let cfg = engine.config();
    let queries = build_queries(cfg, &query_positions);
    let paged = paged_attention(
        shared_seq.table(),
        engine.allocator(),
        0,
        &queries,
        &query_positions,
    )
    .expect("paged");

    let mut standalone_allocator = KvBlockAllocator::new(cfg.clone()).expect("config");
    let standalone = build_sequence(
        cfg,
        &mut standalone_allocator,
        &shared_seq.token_ids()[..resident],
    );
    let oracle = standalone
        .materialize(&standalone_allocator)
        .expect("materialize");
    let reference =
        scaled_dot_product_attention(&oracle, 0, &queries, &query_positions).expect("oracle");
    assert_bits_f32(
        paged.output(),
        reference.output(),
        "shared-prefix sequence attention",
    );
    assert_bits_f64(
        paged.weights(),
        reference.weights(),
        "shared-prefix sequence weights",
    );
}

#[test]
fn prefix_sharing_off_shares_nothing() {
    const B: usize = 4;
    const P: usize = 16;
    const S: usize = 8;
    const N: usize = 4;
    let config = ContinuousBatchConfig::new(2, 2, 4, B, 48)
        .with_max_running_sequences(N)
        .with_prefix_sharing(false);
    let mut engine = ContinuousBatchEngine::new(config).expect("config");

    let prefix: Vec<u32> = (0..P as u32).collect();
    for i in 0..N {
        let mut tokens = prefix.clone();
        for j in 0..S {
            tokens.push(10_000 + (i * 100 + j) as u32);
        }
        tokens.push(20_000 + i as u32);
        engine.submit(tokens, P + S).expect("submit");
    }
    engine.step(&HashProducer).expect("step");

    // With sharing off, each sequence pays full freight: N * blocks(P + S).
    assert_eq!(engine.allocator().allocated_blocks(), N * ((P + S) / B));
    assert_eq!(engine.stats().prefix_cached_tokens, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. ITERATION-LEVEL BATCHING RETIRES FINISHED SEQUENCES IMMEDIATELY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn continuous_batching_wastes_no_slots_on_finished_sequences() {
    // Eight requests, deliberately skewed so each cohort of four holds three short
    // sequences behind one long one. The pool is sized so that at most four
    // sequences ever run at once (`max_running`) and every one of them fits with
    // room to spare — so **no preemption happens**, and the only thing being
    // measured is *time*: how long a slot stays occupied.
    let prompt_len = 4;
    let lengths = [5usize, 5, 5, 84, 6, 6, 6, 82];
    let config = ContinuousBatchConfig::new(1, 2, 4, 4, 96)
        .with_max_running_sequences(4)
        .with_prefix_sharing(false);
    let mut engine = ContinuousBatchEngine::new(config).expect("config");

    for (i, &length) in lengths.iter().enumerate() {
        let tokens: Vec<u32> = (0..length as u32).map(|t| (i as u32) * 1000 + t).collect();
        engine.submit(tokens, prompt_len).expect("submit");
    }

    // Residency, measured: after each step, every still-running sequence occupied
    // a slot this step. Summing gives the total slot-steps the continuous loop
    // actually spent, and the per-sequence tally gives each sequence's residency.
    let mut residency = vec![0usize; lengths.len()];
    let mut continuous_slot_steps = 0usize;
    for _ in 0..5000 {
        if engine.is_idle() {
            break;
        }
        engine.step(&HashProducer).expect("step");
        for &id in engine.running() {
            residency[id.index()] += 1;
            continuous_slot_steps += 1;
        }
    }
    assert!(engine.is_idle());
    assert!(
        engine.sequences().iter().all(BatchSequence::is_complete),
        "every sequence finished"
    );
    // The clean regime: not a single preemption inflated the residency.
    assert_eq!(
        engine.stats().preemptions_recompute + engine.stats().preemptions_swap,
        0,
        "no preemption in a roomy pool"
    );

    // With prefill collapsed into one step and no preemption, a sequence is
    // resident for exactly `length - prompt_len` steps. That the *measured*
    // residency equals this closed form is itself a check on the loop.
    for (i, &length) in lengths.iter().enumerate() {
        assert_eq!(residency[i], length - prompt_len, "sequence {i} residency");
    }

    // In continuous batching every resident slot holds a genuinely decoding
    // sequence — there is no other kind of running slot — so all of it is useful.
    let useful_slot_steps: usize = residency.iter().sum();
    assert_eq!(continuous_slot_steps, useful_slot_steps);

    // A *static* batcher forms cohorts of `max_running` in submission order and
    // holds every member of a cohort until its **longest** member finishes: the
    // three short sequences keep their slots long after they are done. Charge each
    // cohort `cohort_size * max_residency`, using the very residencies just
    // measured.
    let mut static_slot_steps = 0usize;
    for cohort in residency.chunks(4) {
        let longest = cohort.iter().copied().max().unwrap_or(0);
        static_slot_steps += cohort.len() * longest;
    }

    let wasted = static_slot_steps - useful_slot_steps;
    let wasted_fraction = wasted as f64 / static_slot_steps as f64;
    // Three of every four slots in a cohort are held by a finished sequence for
    // most of the cohort's life: the static waste approaches (C-1)/C = 75%.
    assert!(
        wasted_fraction > 0.6,
        "static batching wastes >60% of slot-steps, measured {wasted_fraction:.3}"
    );
    assert!(
        continuous_slot_steps * 2 < static_slot_steps,
        "continuous uses less than half the slot-steps of static"
    );
}

#[test]
fn engine_decode_path_matches_oracle() {
    // Build a sequence through the *engine* (prefill + iteration-level decode) and
    // confirm its block table attends identically to a contiguous oracle — the
    // headline equivalence, but exercised through the real loop.
    let config = ContinuousBatchConfig::new(2, 3, 4, 4, 40).with_prefix_sharing(false);
    let mut engine = ContinuousBatchEngine::new(config.clone()).expect("config");

    let token_ids: Vec<u32> = (0..20).map(|t| t as u32 * 7 + 1).collect();
    let id = engine.submit(token_ids.clone(), 5).expect("submit"); // prompt 5, decode 15

    // Step until it has 19 tokens and is still running (target is 20).
    for _ in 0..15 {
        engine.step(&HashProducer).expect("step");
    }
    let sequence = engine.sequence(id).expect("sequence");
    assert_eq!(sequence.progress(), 19);
    assert!(!sequence.is_complete());

    let mut oracle_allocator = KvBlockAllocator::new(config.clone()).expect("config");
    let standalone = build_sequence(&config, &mut oracle_allocator, &token_ids[..19]);
    let oracle = standalone
        .materialize(&oracle_allocator)
        .expect("materialize");

    let query_positions = [0usize, 9, 18, 30];
    let queries = build_queries(&config, &query_positions);
    for layer in 0..config.num_layers {
        let paged = paged_attention(
            sequence.table(),
            engine.allocator(),
            layer,
            &queries,
            &query_positions,
        )
        .expect("paged");
        let reference = scaled_dot_product_attention(&oracle, layer, &queries, &query_positions)
            .expect("oracle");
        assert_bits_f32(
            paged.output(),
            reference.output(),
            &format!("engine decode output, layer {layer}"),
        );
        assert_bits_f64(
            paged.weights(),
            reference.weights(),
            &format!("engine decode weights, layer {layer}"),
        );
    }
}

#[test]
fn engine_completes_under_preemption_recompute() {
    engine_completes_under_preemption(BatchPreemptionMode::Recompute);
}

#[test]
fn engine_completes_under_preemption_swap() {
    engine_completes_under_preemption(BatchPreemptionMode::Swap);
}

/// A tiny pool forces preemption; the loop must still finish every sequence, and
/// the final KV of each must match a no-preemption reference run — proving the
/// eviction round trip (either mode) is lossless end to end.
fn engine_completes_under_preemption(mode: BatchPreemptionMode) {
    let lengths = [12usize, 9, 14, 7];

    // Reference: a pool big enough that nothing is ever preempted. Capture each
    // sequence's final KV in the step it completes, just before it is retired.
    let (reference_kv, reference_preemptions) =
        run_and_capture_final_kv_counting(&lengths, 200, BatchPreemptionMode::Recompute);
    assert_eq!(
        reference_preemptions, 0,
        "the roomy reference pool must never preempt"
    );

    // A pool barely large enough for the longest single sequence: preemption is
    // unavoidable, and both modes must reproduce the same final KV.
    let (preempted_kv, preemptions) = run_and_capture_final_kv_counting(&lengths, 6, mode);

    assert!(
        preemptions > 0,
        "the tiny pool must have forced at least one preemption"
    );
    assert_eq!(reference_kv.len(), preempted_kv.len());
    for (index, (reference, preempted)) in reference_kv.iter().zip(&preempted_kv).enumerate() {
        assert_eq!(
            reference, preempted,
            "sequence {index} final KV must survive {mode} preemption bit for bit"
        );
    }
}

/// Run the workload and, at the step each sequence finishes, materialise its KV
/// before the engine reclaims its blocks. Returns the per-sequence final KV and
/// the total number of preemptions.
fn run_and_capture_final_kv_counting(
    lengths: &[usize],
    total_blocks: usize,
    mode: BatchPreemptionMode,
) -> (Vec<KvCacheTensor>, u64) {
    let config = ContinuousBatchConfig::new(2, 2, 4, 4, total_blocks)
        .with_max_running_sequences(4)
        .with_prefix_sharing(false)
        .with_preemption_mode(mode);
    let mut engine = ContinuousBatchEngine::new(config).expect("config");

    let mut ids = Vec::new();
    for (i, &length) in lengths.iter().enumerate() {
        let tokens: Vec<u32> = (0..length as u32).map(|t| (i as u32) * 1000 + t).collect();
        ids.push(engine.submit(tokens, 3).expect("submit"));
    }

    let mut captured: Vec<Option<KvCacheTensor>> = lengths.iter().map(|_| None).collect();
    for _ in 0..4000 {
        if engine.is_idle() {
            break;
        }
        // Capture any running sequence that will complete this step, before the
        // step retires it. A sequence with progress == target - 1 that decodes
        // this step reaches target and is retired within `step`.
        let about_to_finish: Vec<usize> = engine
            .running()
            .iter()
            .filter_map(|&id| {
                let sequence = engine.sequence(id).ok()?;
                (sequence.progress() + 1 == sequence.target_len()).then_some(id.index())
            })
            .collect();
        for index in about_to_finish {
            if captured[index].is_none() {
                let sequence = &engine.sequences()[index];
                // Its KV *after* this step's decode is a standalone build of its
                // whole token stream; capture that reference now.
                let mut allocator = KvBlockAllocator::new(engine.config().clone()).expect("config");
                let table = build_sequence(engine.config(), &mut allocator, sequence.token_ids());
                captured[index] = Some(table.materialize(&allocator).expect("materialize"));
            }
        }
        engine.step(&HashProducer).expect("step");
    }

    assert!(engine.is_idle(), "workload finished");
    for &id in &ids {
        assert!(engine.sequence(id).expect("seq").is_complete());
    }
    let kv = captured
        .into_iter()
        .map(|entry| entry.expect("every sequence captured"))
        .collect();
    let preemptions = engine.stats().preemptions_recompute + engine.stats().preemptions_swap;
    (kv, preemptions)
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. FORK IN THE ENGINE (parallel sampling)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn engine_fork_shares_prompt_and_diverges_correctly() {
    let config = ContinuousBatchConfig::new(2, 2, 4, 4, 40).with_prefix_sharing(false);
    let mut engine = ContinuousBatchEngine::new(config.clone()).expect("config");

    let prompt: Vec<u32> = (0..12).collect();
    let mut parent_tokens = prompt.clone();
    // The parent has its own future beyond the fork point, so it is still running
    // (not finished) when the children branch off it.
    parent_tokens.extend([500, 501, 502, 503, 504, 505, 506, 507]);
    let parent = engine.submit(parent_tokens, 12).expect("submit");
    // Admit (prefill 12) and decode one token so the parent is running with 13.
    engine.step(&HashProducer).expect("prefill step");
    engine.step(&HashProducer).expect("decode step");
    assert_eq!(engine.sequence(parent).expect("seq").progress(), 13);

    // Fork two divergent continuations sharing the parent's first 13 tokens' KV.
    let mut child_a_tokens = engine.sequence(parent).expect("seq").token_ids()[..13].to_vec();
    child_a_tokens.extend([700, 701, 702]);
    let mut child_b_tokens = child_a_tokens[..13].to_vec();
    child_b_tokens.extend([800, 801]);
    let child_a = engine
        .fork_sequence(parent, child_a_tokens.clone())
        .expect("fork");
    let child_b = engine
        .fork_sequence(parent, child_b_tokens.clone())
        .expect("fork");

    // The fork copied no bytes: the parent's full blocks are referenced thrice.
    let shared = engine.sequences()[parent.index()].table().block_ids()[0];
    assert_eq!(
        engine.allocator().ref_count(shared),
        3,
        "first block shared by parent + 2 children"
    );

    engine.run_until_idle(&HashProducer, 200).expect("progress");
    assert!(engine.sequence(child_a).expect("seq").is_complete());
    assert!(engine.sequence(child_b).expect("seq").is_complete());
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CONFIG, ERRORS, AND SMALL INVARIANTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn config_validation_rejects_degenerate_geometry() {
    assert!(
        ContinuousBatchConfig::new(0, 2, 4, 4, 10)
            .validate()
            .is_err()
    );
    assert!(
        ContinuousBatchConfig::new(2, 0, 4, 4, 10)
            .validate()
            .is_err()
    );
    assert!(
        ContinuousBatchConfig::new(2, 2, 0, 4, 10)
            .validate()
            .is_err()
    );
    assert!(
        ContinuousBatchConfig::new(2, 2, 4, 0, 10)
            .validate()
            .is_err()
    );
    assert!(
        ContinuousBatchConfig::new(2, 2, 4, 4, 0)
            .validate()
            .is_err()
    );
    // Watermark must leave at least one admissible block.
    assert!(
        ContinuousBatchConfig::new(2, 2, 4, 4, 8)
            .with_watermark_blocks(8)
            .validate()
            .is_err()
    );
    assert!(
        ContinuousBatchConfig::new(2, 2, 4, 4, 8)
            .with_watermark_blocks(7)
            .validate()
            .is_ok()
    );
    assert!(KvBlockAllocator::new(ContinuousBatchConfig::new(0, 2, 4, 4, 10)).is_err());
}

#[test]
fn config_block_arithmetic() {
    let config = ContinuousBatchConfig::new(3, 4, 8, 16, 100);
    assert_eq!(config.token_buffer_len(), 3 * 4 * 8);
    assert_eq!(config.per_layer_token_stride(), 4 * 8);
    assert_eq!(config.blocks_for_tokens(0), 0);
    assert_eq!(config.blocks_for_tokens(1), 1);
    assert_eq!(config.blocks_for_tokens(16), 1);
    assert_eq!(config.blocks_for_tokens(17), 2);
    assert_eq!(
        config.block_bytes(),
        2 * 16 * (3 * 4 * 8) * std::mem::size_of::<f32>()
    );
}

#[test]
fn allocator_rejects_double_free_and_unknown_block() {
    let config = ContinuousBatchConfig::new(1, 1, 2, 4, 4);
    let mut allocator = KvBlockAllocator::new(config).expect("config");
    let id = allocator.allocate_block().expect("allocate");
    allocator.release_block(id).expect("release");
    assert!(
        matches!(
            allocator.release_block(id),
            Err(ContinuousBatchError::BlockNotAllocated { .. })
        ),
        "double free is refused"
    );
    assert!(matches!(
        allocator.release_block(KvBlockId(99)),
        Err(ContinuousBatchError::UnknownBlock { .. })
    ));
    assert!(matches!(
        allocator.copy_on_write(id),
        Err(ContinuousBatchError::BlockNotAllocated { .. })
    ));
}

#[test]
fn allocator_out_of_blocks_is_all_or_nothing() {
    let config = ContinuousBatchConfig::new(1, 1, 2, 4, 3);
    let mut allocator = KvBlockAllocator::new(config).expect("config");
    // Ask for more than the pool holds: nothing is taken.
    assert!(matches!(
        allocator.allocate_blocks(4),
        Err(ContinuousBatchError::OutOfBlocks { .. })
    ));
    assert_eq!(
        allocator.allocated_blocks(),
        0,
        "a failed multi-allocation rolls back"
    );
    assert_eq!(allocator.free_blocks(), 3);
    allocator.verify_invariants().expect("invariants");
}

#[test]
fn table_rejects_bad_buffers() {
    let config = ContinuousBatchConfig::new(2, 2, 4, 4, 8);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");
    let mut table = KvBlockTable::new(&config);
    let good = vec![0.0f32; config.token_buffer_len()];

    // Wrong length.
    assert!(matches!(
        table.append_token(&mut allocator, 0, &good[..3], &good),
        Err(ContinuousBatchError::ShapeMismatch { .. })
    ));
    // Non-finite.
    let mut bad = good.clone();
    bad[2] = f32::NAN;
    assert!(matches!(
        table.append_token(&mut allocator, 0, &bad, &good),
        Err(ContinuousBatchError::NonFiniteInput { .. })
    ));
    let mut inf = good.clone();
    inf[0] = f32::INFINITY;
    assert!(matches!(
        table.append_token(&mut allocator, 0, &good, &inf),
        Err(ContinuousBatchError::NonFiniteInput { .. })
    ));
    // The failed appends left the table empty.
    assert!(table.is_empty());
    assert_eq!(allocator.allocated_blocks(), 0);
}

#[test]
fn paged_attention_rejects_bad_calls() {
    let config = attention_config();
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");
    let empty = KvBlockTable::new(&config);
    let queries = build_queries(&config, &[0]);
    // Empty table.
    assert!(matches!(
        paged_attention(&empty, &allocator, 0, &queries, &[0]),
        Err(ContinuousBatchError::EmptyTable { .. })
    ));

    let table = build_sequence(&config, &mut allocator, &[1, 2, 3, 4, 5]);
    // Bad layer.
    assert!(matches!(
        paged_attention(&table, &allocator, 99, &queries, &[0]),
        Err(ContinuousBatchError::LayerOutOfRange { .. })
    ));
    // Non-increasing query positions.
    let two = build_queries(&config, &[3, 3]);
    assert!(matches!(
        paged_attention(&table, &allocator, 0, &two, &[3, 3]),
        Err(ContinuousBatchError::InvalidSelection { .. })
    ));
    // Wrong query buffer length.
    assert!(matches!(
        paged_attention(&table, &allocator, 0, &queries[..2], &[0]),
        Err(ContinuousBatchError::ShapeMismatch { .. })
    ));
}

#[test]
fn adopt_rejects_mismatched_prefix_block() {
    // A hash *hit* with a content *miss* must be rejected, not silently used.
    let config = ContinuousBatchConfig::new(1, 1, 2, 4, 12);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");
    let donor = build_sequence(&config, &mut allocator, &[1, 2, 3, 4]);
    let donor_block = donor.block_ids()[0];

    let mut adopter = KvBlockTable::new(&config);
    // Correct tokens: accepted.
    assert!(
        adopter
            .adopt_prefix_block(&mut allocator, donor_block, &[1, 2, 3, 4])
            .is_ok()
    );
    assert_eq!(allocator.ref_count(donor_block), 2);

    // Wrong tokens for the same block: refused.
    let mut other = KvBlockTable::new(&config);
    assert!(matches!(
        other.adopt_prefix_block(&mut allocator, donor_block, &[1, 2, 3, 9]),
        Err(ContinuousBatchError::PrefixMismatch { .. })
    ));
    assert_eq!(
        allocator.ref_count(donor_block),
        2,
        "a rejected adoption takes no reference"
    );
}

#[test]
fn fold_prefix_hash_is_deterministic_and_prefix_sensitive() {
    let seed = super::allocator::PREFIX_HASH_SEED;
    assert_eq!(
        fold_prefix_hash(seed, &[1, 2, 3]),
        fold_prefix_hash(seed, &[1, 2, 3])
    );
    // Rolling: folding [1,2] then [3] equals folding [1,2,3] in one go.
    assert_eq!(
        fold_prefix_hash(fold_prefix_hash(seed, &[1, 2]), &[3]),
        fold_prefix_hash(seed, &[1, 2, 3])
    );
    // Order matters.
    assert_ne!(
        fold_prefix_hash(seed, &[1, 2, 3]),
        fold_prefix_hash(seed, &[3, 2, 1])
    );
    // A differing earlier block changes the whole prefix key.
    assert_ne!(
        fold_prefix_hash(fold_prefix_hash(seed, &[9, 9]), &[3, 4]),
        fold_prefix_hash(fold_prefix_hash(seed, &[1, 1]), &[3, 4])
    );
}

#[test]
fn wasted_slots_is_bounded_by_block_size() {
    let config = ContinuousBatchConfig::new(1, 1, 2, 8, 20);
    let mut allocator = KvBlockAllocator::new(config.clone()).expect("config");
    for tokens in 1..=17usize {
        let ids: Vec<u32> = (0..tokens as u32).collect();
        let table = build_sequence(&config, &mut allocator, &ids);
        assert!(
            table.wasted_slots() < config.block_size,
            "internal fragmentation < block_size"
        );
        assert_eq!(
            table.num_blocks() * config.block_size - table.len(),
            table.wasted_slots()
        );
        let mut drop_it = table;
        drop_it.release(&mut allocator).expect("release");
    }
}

#[test]
fn stats_producer_calls_and_peak_are_consistent() {
    let config = ContinuousBatchConfig::new(1, 2, 4, 4, 30).with_prefix_sharing(false);
    let mut engine = ContinuousBatchEngine::new(config).expect("config");
    for i in 0..3u32 {
        let tokens: Vec<u32> = (0..10).map(|t| i * 100 + t).collect();
        engine.submit(tokens, 4).expect("submit");
    }
    engine.run_until_idle(&HashProducer, 500).expect("progress");
    let stats = engine.stats();
    // 3 sequences x 10 tokens, all computed exactly once (no sharing, no preempt).
    assert_eq!(stats.prefill_tokens + stats.tokens_decoded, 30);
    assert_eq!(stats.prefix_cached_tokens, 0);
    assert_eq!(stats.recomputed_tokens, 0);
    assert_eq!(stats.producer_calls(), 30);
    assert!(stats.peak_blocks_in_use <= engine.config().total_blocks);
}
