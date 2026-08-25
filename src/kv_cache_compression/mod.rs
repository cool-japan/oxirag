//! Attention-score-driven eviction from the transformer KV cache: **H2O**,
//! **`StreamingLLM`**, and **`SnapKV`**, over a real KV tensor with a real
//! scaled-dot-product attention kernel.
//!
//! During autoregressive decoding a transformer caches the key and value
//! vectors of every token it has already seen, so that each new token attends
//! over the whole history without recomputing it. That cache is linear in the
//! context length and, at long contexts, it — not the weights — is what fills
//! the accelerator's memory. Its size is
//!
//! ```text
//! bytes = 2 · num_layers · num_heads · head_dim · seq_len · sizeof(dtype)
//! ```
//!
//! and the only term a serving system can actually shrink at runtime is
//! `seq_len`. **Which tokens do you throw away?**
//!
//! The answer this module implements is: the ones the model *is not looking at*.
//! Attention says so directly — the column sums of the attention matrix are the
//! probability mass the model has spent on each cached token — so the eviction
//! decision can be driven by a measured quantity rather than a heuristic. That
//! is the idea behind H2O, `StreamingLLM` and `SnapKV`, and it is what this
//! module implements: a real KV tensor, a real `softmax(QKᵀ/√d)·V`, real
//! accumulated attention scores, and four eviction policies that rank tokens by
//! them.
//!
//! # How this differs from every other cache and every other pruner here
//!
//! This crate is full of things called caches and things that prune context.
//! **None of them evicts an individual cached token from an attention KV cache
//! by attention score.** The distinction is not pedantic — it is the difference
//! between shrinking the *prompt* and shrinking the *tensor the prompt has
//! already been turned into*:
//!
//! | Module | What it stores | What it evicts | By what signal |
//! |---|---|---|---|
//! | `prefix_cache` | a **whole prompt's** KV blob, keyed by a hash of the prompt string (its `kv_data` is a flat `Vec<f32>`, not a token-indexed tensor) | a whole cache **entry** | LRU / TTL |
//! | `hidden_states` | real per-layer, per-head `keys`/`values` tensors | a whole cache **entry** | recency (LRU); the KV tensor itself has no prune method at all |
//! | `memory_paging` | context **text**, as `String` pages | a page of **text**, swapped to archival storage | page heat / recency |
//! | `llmlingua`, `context_compression`, `context_pruning` | input **text** | **words and sentences of the prompt**, *before* the model ever sees them | surprisal, redundancy, heuristics |
//! | `kv_cache_compression` (this module) | a token-axis KV **tensor**, `[layer][token][head][dim]` | **individual token slots**, from every layer and head at once | **accumulated softmax attention mass**, actually computed |
//!
//! The three text-level pruners act *before* the forward pass and cannot know
//! what the model will attend to, because the model has not attended to anything
//! yet. This module acts *after*, on the cache, using the attention the model
//! actually produced. They are complementary, not alternatives: prune the prompt
//! with `llmlingua`, then evict from the cache with H2O.
//!
//! # The two things that are actually computed here
//!
//! **1. Attention.** [`scaled_dot_product_attention`] computes
//! `A = softmax(QKᵀ/√d_head + M)` and `O = A·V` with a numerically stable
//! (max-subtracted) softmax, `f64` logit accumulation, and a causal mask `M`
//! driven by each token's **absolute stream position** — which survives eviction,
//! so the mask stays correct after a compression pass has punched holes in the
//! middle of the cache. See the [`attention`] module docs for the proof that a
//! finite input cannot produce a non-finite output.
//!
//! **2. Saliency.** [`KvAttentionStats`] accumulates, per cached token, the
//! attention mass spent on it (`Σ_q A[q, t]`, the column sum) and the number of
//! softmax rows it competed in. The ratio of the two is the default saliency
//! score, and the reason for that ratio is the **early-token bias**: under a
//! causal mask a token at position `t` is visible to every query at position
//! `>= t`, so the raw column sum sums over more terms for older tokens and
//! prefers them for reasons that have nothing to do with saliency. See
//! [`KvScoreNormalization`].
//!
//! # The four policies
//!
//! | Policy | Keeps | Needs attention history? |
//! |---|---|---|
//! | [`KvEvictionPolicy::H2O`] | recent window ∪ top-`k` **heavy hitters** by accumulated attention | yes |
//! | [`KvEvictionPolicy::StreamingLlm`] | the first `n` **attention sinks** ∪ a sliding recent window; the middle is evicted | no (purely positional) |
//! | [`KvEvictionPolicy::SnapKv`] | the **observation window** ∪ the top-`k` prefix tokens *it* votes for (pooled, so selections form spans) | yes |
//! | [`KvEvictionPolicy::RecencyLru`] | the last `budget` tokens. Nothing else. The honest baseline. | no |
//!
//! All four obey one hard contract: after a successful
//! [`KvCacheCompressor::compress`], `cache.seq_len() <= budget`. Always.
//!
//! # The accuracy / memory tradeoff, measured
//!
//! Compression is not free, and this module does not pretend otherwise. Evicting
//! token `t` deletes its term from every future softmax and renormalizes the
//! rest, perturbing the attention output of a query `q` by
//!
//! ```text
//! ‖o_q - o_q^evicted‖ = (A[q, t] / (1 - A[q, t])) · ‖v_t - o_q^evicted‖
//! ```
//!
//! — small when `A[q, t]` is small, **unbounded** as `A[q, t] → 1`. Memory falls
//! linearly in the *number* of evicted tokens; accuracy falls with the *attention
//! mass* of the evicted tokens. Those are different quantities, and a good policy
//! is precisely one that decorrelates them: spend the eviction on tokens that
//! occupy many slots and carry little mass.
//!
//! The module's headline test measures exactly that. It builds a 128-token
//! workload in which three *old* tokens (positions 7, 23, 51) carry a **measured
//! 99.37%** of every query's attention mass, compresses to **32 tokens (25% of
//! the sequence, a 4x memory saving)** with each policy, and then runs a
//! **held-out** query — one no policy ever saw — against each compressed cache,
//! measuring the relative L2 deviation of its output from the uncompressed one:
//!
//! | Policy | Deviation from full attention at a 25% budget | Keeps the heavy hitters? |
//! |---|---|---|
//! | H2O | **0.0048** | yes |
//! | `SnapKV` | **0.0048** | yes |
//! | `StreamingLLM` | 1.0018 | no |
//! | `RecencyLru` | 1.0066 | no |
//!
//! A 4x memory saving for a 0.5% output perturbation — *if* the policy keeps the
//! right tokens. The recency baseline gets exactly the same 4x saving and
//! *destroys* the output (a deviation of ~1.0 means the output moved by as much
//! as its own magnitude), because the salient tokens are old and it knows nothing
//! but age. The 200x gap between those rows is the entire justification for this
//! module. (`StreamingLLM` sits with the baseline *on this particular workload* by
//! construction: this workload's semantic mass is not in its sinks. That is
//! recorded rather than hidden — `StreamingLLM`'s own load-bearing claim is a
//! different one, and is demonstrated separately below.)
//!
//! # The `StreamingLLM` sink result
//!
//! Separately, on a cache with a genuine attention-sink structure, the module
//! demonstrates the paper's counterintuitive claim by direct measurement. The
//! first three tokens absorb a **measured 85.75%** of the query's attention mass
//! while carrying no semantic value. Removing three tokens from that cache:
//!
//! | What is removed | Deviation of the attention output |
//! |---|---|
//! | the 3 **sink** tokens | **5.36** (catastrophic — the output moves by 5x its own magnitude) |
//! | 3 **middle** tokens (same count) | **0.00021** (free) |
//!
//! A **25,000x** asymmetry, from deleting the same number of tokens. The sinks are
//! not informative — they are *load-bearing*, because softmax renormalizes the
//! mass they were absorbing onto everything else. See [`streaming`] for the
//! derivation.
//!
//! # Eviction granularity
//!
//! Scores are pooled across heads, and eviction removes a token from every head
//! and every layer at once. The published policies are typically **per-head** — a
//! token may be a heavy hitter for head 3 and dead weight for head 5 — which needs
//! a *ragged* cache with a different surviving token set per head. That is a
//! strictly larger design and a strictly larger memory saving; pooling is the
//! conservative choice (a token that any head loves keeps a high pooled mean and
//! survives), and the ragged variant is named as follow-up work rather than
//! silently pretended.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "kv-cache-compression")]
//! # {
//! use oxirag::kv_cache_compression::{
//!     KvAttentionStats, KvCacheCompressor, KvCacheTensor, KvCompressionConfig,
//!     KvEvictionPolicy, scaled_dot_product_attention,
//! };
//!
//! // A tiny one-layer, one-head cache of 16 tokens.
//! let (layers, heads, dim) = (1, 1, 4);
//! let mut cache = KvCacheTensor::new(layers, heads, dim).expect("valid geometry");
//! for token in 0..16 {
//!     // Token 3 is the salient one: its key points hard along e0, so every
//!     // query that keys on e0 will spend most of its attention mass there.
//!     let magnitude = if token == 3 { 8.0 } else { 0.1 };
//!     let keys = vec![magnitude, 0.0, 0.0, 0.0];
//!     let values = vec![0.0, magnitude, 0.0, 0.0];
//!     cache.append_token(&keys, &values).expect("well-formed token");
//! }
//!
//! // Attend with queries that key on e0, and accumulate the real attention mass.
//! let config = KvCompressionConfig::new(KvEvictionPolicy::H2O, 8).with_recent_window(4);
//! let compressor = KvCacheCompressor::new(config).expect("valid config");
//! let mut stats = KvAttentionStats::for_cache(&cache, compressor.config());
//!
//! let positions: Vec<usize> = (0..16).collect();
//! let queries: Vec<f32> = positions.iter().flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();
//! let attention = scaled_dot_product_attention(&cache, 0, &queries, &positions)
//!     .expect("well-formed attention call");
//! compressor.observe(&mut stats, &attention).expect("stats match the cache");
//!
//! // Compress. The heavy hitter (token 3) is old -- a recency policy would have
//! // thrown it away -- but H2O keeps it, because the model was looking at it.
//! let report = compressor.compress(&mut cache, &mut stats).expect("compressible");
//! assert!(report.within_budget);
//! assert_eq!(report.tokens_after, 8);
//! assert!(report.retained_positions.contains(&3));
//! assert!(report.memory_saved_ratio() > 0.49);
//! # }
//! ```

pub mod attention;
pub mod compressor;
pub mod h2o;
pub mod recency;
pub mod snapkv;
pub mod streaming;
pub mod types;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub use attention::{KvAttentionOutput, KvAttentionStats, scaled_dot_product_attention};
pub use compressor::KvCacheCompressor;
pub use h2o::plan_h2o;
pub use recency::plan_recency_lru;
pub use snapkv::{plan_snap_kv, pool_scores};
pub use streaming::plan_streaming_llm;
pub use types::{
    KvCacheTensor, KvCompressionConfig, KvCompressionError, KvCompressionReport, KvEvictionPolicy,
    KvResult, KvScoreNormalization, KvSnapPooling,
};
