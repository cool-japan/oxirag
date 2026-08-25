//! **Chunked prefill** — `Sarathi-Serve` stall-free batching: split a prompt's
//! prefill into token-budgeted chunks that reproduce a one-shot prefill *exactly*,
//! and pack each one into a scheduling iteration alongside every running decode.
//!
//! # The problem
//!
//! An `LLM` serving iteration does one of two utterly different things.
//!
//! A **prefill** consumes a whole prompt at once — thousands of tokens, one huge
//! parallel forward pass. A **decode** consumes one token. Batch them the
//! classical way and an iteration is *either* a prefill *or* a batch of decodes,
//! with prefill first so that new requests get admitted. So when a 4096-token
//! prompt arrives, every sequence already generating **stops**, for as long as
//! that prefill takes. Their users watch the text freeze. That is a **generation
//! stall**, and its length is the *new* request's prompt length — a quantity the
//! stalled users have no relationship with and no defence against.
//!
//! Two knobs, and neither works. Refuse the prompt and you have traded a stall
//! for a queue. Cap the prompt length and you have a different product.
//!
//! # The idea
//!
//! Chop the prefill up. Give each scheduling iteration a **token budget `B`**,
//! fill it with **every runnable decode plus exactly one prefill chunk sized to
//! whatever the decodes left behind**, and carry the `KV` forward from chunk to
//! chunk:
//!
//! ```text
//! decode_tokens = (sequences currently decoding)        ≤ D_max
//! chunk_tokens  = min(prompt left, B − decode_tokens)   ≥ B − D_max ≥ 1
//! ```
//!
//! No iteration now exceeds `B` tokens — *the prompt length has dropped out of the
//! expression entirely* — and no decode is ever skipped, because decodes are taken
//! first and in full. A sequence emits a token in **every** iteration, and each
//! iteration costs at most `B`. So
//!
//! > **max time-between-tokens ≤ cost(`B` tokens), for any prompt length.**
//!
//! The stall is not shortened. It is *bounded*, and bounded by a number the
//! operator picks. See [`PrefillScheduler`] and
//! [`PrefillStallStats::is_stall_free`].
//!
//! # The catch, and why it is the whole design
//!
//! Chunk `i` must attend to **everything committed before it** *and* to **its own
//! causally-masked prefix**. So the mask a chunk runs under is neither a triangle
//! nor a rectangle — it is two segments glued at a boundary:
//!
//! ```text
//!            committed keys (c)          this chunk's keys (n)
//!         ┌─────────────────────────┬───────────────────────────┐
//!  q = 0  │ 1  1  1  1  1  1  1  1  │ 1  0  0  0  0             │
//!  q = 1  │ 1  1  1  1  1  1  1  1  │ 1  1  0  0  0             │
//!  q = 2  │ 1  1  1  1  1  1  1  1  │ 1  1  1  0  0             │
//!  q = 3  │ 1  1  1  1  1  1  1  1  │ 1  1  1  1  0             │
//!  q = 4  │ 1  1  1  1  1  1  1  1  │ 1  1  1  1  1             │
//!         └─────────────────────────┴───────────────────────────┘
//!            fully visible: no mask     inclusive lower triangle
//! ```
//!
//! and the *positions* its queries carry are **absolute stream positions**, not
//! offsets into the chunk. Get that boundary off by one and you produce a
//! perfectly finite, entirely plausible, completely wrong tensor: a query that
//! sees one token of its own future, or that cannot see itself. This is where real
//! implementations ship bugs, and [`PrefillMaskGeometry`] exists to make it
//! impossible to do so quietly — see [`mask`].
//!
//! # The invariant, and it is exact
//!
//! > **For *any* split of the prompt into chunks, chunked prefill produces the
//! > same `KV` tensor and the same attention outputs as a one-shot prefill.**
//!
//! Not "within a tolerance" — **bit for bit**. The tolerance is `0.0`, and the
//! tests assert equality, not closeness. The argument, for a query at absolute
//! position `p` (spelled out in full in [`executor`]):
//!
//! 1. The **visible key set is identical** — the causal mask admits the slots with
//!    position `<= p`, chunked or not, and they sit at the same slot indices,
//!    because the chunked cache is a *prefix* of the one-shot cache.
//! 2. The **logits are the same `f64`s** — the same `f32` vectors dotted in the
//!    same order.
//! 3. The **extra columns contribute exactly nothing** — a one-shot row is wider,
//!    but its extra entries are masked, so their weight is `exp(−inf − max) = 0.0`
//!    *exactly*; adding `0.0` to an `f64` accumulator is the identity, and the
//!    value sum skips zero weights outright.
//! 4. The **reductions run in the same order**.
//!
//! This is checked against an oracle built from code this module did not write —
//! [`KvCacheTensor::append_token`](crate::kv_cache_compression::KvCacheTensor::append_token)
//! and
//! [`scaled_dot_product_attention`](crate::kv_cache_compression::scaled_dot_product_attention)
//! — over **every composition** of a 10-token prompt into chunks: all `2^9 = 512`
//! of them, each asserted bit-identical, and again over every composition against
//! a cache that already holds a shared prefix.
//!
//! # What chunking costs, exactly
//!
//! Both figures below are *derived*, not benchmarked, and they point in opposite
//! directions.
//!
//! **Arithmetic: free.** A query at position `p` sees the keys at positions `<= p`.
//! That is a fact about the causal mask, and no chunk boundary can change it. So
//! the `Q·K` dot-product count is `P(P+1)/2` for **every** composition of a
//! `P`-token prompt. Chunking moves work between iterations; it creates none.
//!
//! **Memory traffic: not free.** Every chunk invocation needs the *whole* committed
//! cache resident, so it visits `c + start + n` `KV` slots. Over `m` uniform chunks
//! that is `P(m+1)/2` against `P` for one shot — the price of the bound. For a
//! 512-token prompt:
//!
//! | `B` | chunks | max iteration tokens | `Q·K` cells | `KV` slot visits | `KV` amplification |
//! |---:|---:|---:|---:|---:|---:|
//! | 512 (one-shot) | 1 | **512** | 131 328 | 512 | **1.0x** |
//! | 256 | 2 | 256 | 131 328 | 768 | 1.5x |
//! | 128 | 4 | 128 | 131 328 | 1 280 | 2.5x |
//! | 64 | 8 | **64** | 131 328 | 2 304 | **4.5x** |
//! | 32 | 16 | 32 | 131 328 | 4 352 | 8.5x |
//! | 16 | 32 | **16** | 131 328 | 8 448 | **16.5x** |
//!
//! An **8x** better worst-case time-between-tokens costs **4.5x** the `KV` traffic
//! and *exactly zero* arithmetic. A `FLOP` count alone would call chunking free,
//! and it is not — so both columns are reported. That trade is the entire content
//! of the token-budget knob.
//!
//! # The stall, measured
//!
//! One sequence generating 6 tokens; a 20-token prompt lands on top of it. `B = 8`,
//! decode cap 4. Both packings do **exactly the same 26 tokens of work** — only its
//! arrangement differs:
//!
//! | | max iteration | max gap | **max time-between-tokens** |
//! |---|---:|---:|---:|
//! | [`PrefillPacking::Exclusive`] (classical) | 20 tokens | 2 iterations | **21 tokens** |
//! | [`PrefillPacking::StallFree`] (`Sarathi-Serve`) | 8 tokens | 1 iteration | **8 tokens** |
//!
//! The decoder waited 21 tokens for its next token because a stranger's prompt was
//! 20 tokens long. Under the budget it waits 8 — and it would still wait 8 if the
//! prompt were 20 000.
//!
//! # Scope
//!
//! This module owns ***within*-request work splitting**, and nothing else.
//!
//! * **Where the bytes live** — the `KV` block allocator, copy-on-write, preemption
//!   — is `continuous_batching`'s.
//! * ***Across*-request ordering** — priority, deadlines, fairness, admission — is
//!   `request_scheduling`'s. Prompts here are prefilled in the order given, and if
//!   more sequences want to decode than the budget reserved room for, this module
//!   *errors* rather than dropping one, because dropping one is the stall.
//!
//! Everything is **synchronous**, and counted in **logical ticks and tokens**. No
//! clock is read, and none is simulated.
//!
//! # Example
//!
//! ```
//! use oxirag::chunked_prefill::{
//!     ChunkedPrefill, ChunkedPrefillConfig, PrefillGeometry, PrefillTensorModel, TokenBudget,
//! };
//! use oxirag::kv_cache_compression::KvCacheTensor;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let geometry = PrefillGeometry::new(2, 2, 4)?;
//! let prompt_tokens = 9;
//! let n = prompt_tokens * geometry.token_stride();
//! let ramp = |scale: f32| (0..n).map(|i| scale * ((i % 5) as f32 - 2.0)).collect::<Vec<f32>>();
//! let model = PrefillTensorModel::new(geometry, prompt_tokens, ramp(0.1), ramp(0.2), ramp(0.3))?;
//!
//! // A 4-token iteration budget: the prompt becomes chunks of 4, 4, 1.
//! let engine = ChunkedPrefill::new(ChunkedPrefillConfig::new(TokenBudget::new(4, 1)?));
//! let mut cache = KvCacheTensor::new(2, 2, 4)?;
//! let run = engine.prefill(&model, &mut cache)?;
//!
//! assert_eq!(run.chunks().len(), 3);
//! assert_eq!(run.stats().max_chunk_tokens(), 4);                   // never over budget
//! assert_eq!(run.stream_positions(), (0..9).collect::<Vec<_>>());  // positions survive
//! assert_eq!(cache.seq_len(), 9);                                  // the KV carried forward
//!
//! // Chunking cost no arithmetic...
//! let stats = run.stats();
//! assert_eq!(stats.attention_cells(), stats.one_shot_attention_cells());
//! // ...and cost exactly this much extra KV traffic.
//! assert_eq!(stats.kv_slot_visits(), 4 + 8 + 9);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! * Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov & Ramjee (2024),
//!   *Taming Throughput-Latency Tradeoff in `LLM` Inference with `Sarathi-Serve`*
//!   (`OSDI`) — the token budget, stall-free batching, and the observation that
//!   chunking is arithmetic-neutral but not memory-neutral.
//! * Agrawal, Panwar, Mohan, Kwatra, Gulavani & Ramjee (2023), *`SARATHI`:
//!   Efficient `LLM` Inference by Piggybacking Decodes with Chunked Prefills*.
//! * Holmes, Tanaka, Wyatt, Awan, Rasley, Rajbhandari, Aminabadi, Qin, Bakhtiari,
//!   Kurilenko & He (2024), *`DeepSpeed-FastGen`* — dynamic `SplitFuse`, the same
//!   packing arrived at independently.

pub mod executor;
pub mod mask;
pub mod model;
pub mod scheduler;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use executor::{
    ChunkedPrefill, PrefillChunkReport, PrefillOutput, PrefillRun, PrefillRunStats,
};
pub use mask::PrefillMaskGeometry;
pub use model::{PrefillModel, PrefillTensorModel, check_cache_geometry};
pub use scheduler::{
    PrefillIteration, PrefillIterationChunk, PrefillPacking, PrefillSchedule, PrefillScheduler,
    PrefillSequenceSpec, PrefillSequenceTrace, PrefillStallStats,
};
pub use types::{
    ChunkedPrefillConfig, ChunkedPrefillError, ChunkedPrefillResult, PrefillChunk, PrefillGeometry,
    PrefillPlan, TokenBudget,
};
