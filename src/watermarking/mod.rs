//! Statistical text watermarking for LLM-generated content: embedding and
//! detecting a provenance signal in generated text itself, entirely without
//! the model at detection time.
//!
//! This implements the green-list scheme of Kirchenbauer, Geiping, Wen,
//! Katz, Miers & Goldstein, 2023, "A Watermark for Large Language Models"
//! (<https://arxiv.org/abs/2301.10226>):
//!
//! * **Generation.** At each decoding step, hash the preceding
//!   [`WatermarkConfig::context_width`] tokens together with a secret key
//!   into a seed, and use that seed to deterministically partition the
//!   vocabulary into a *green list* (a fraction `gamma`) and a *red list*.
//!   Either add a bias `delta` to every green-list token's logit before
//!   deciding the next token ([`WatermarkMode::Soft`]), or restrict the
//!   decision to the green list outright ([`WatermarkMode::Hard`]).
//! * **Detection.** Given only the text and the key — never the model —
//!   recompute each token's green list from its own predecessor tokens,
//!   count how many of the text's tokens are green, and test that count
//!   against its null-hypothesis expectation (`gamma` per token, under the
//!   hypothesis that the text was produced *without* knowledge of the green
//!   lists) via a z-test.
//!
//! # Why this is fully deterministic
//!
//! The green list is derived by **hashing** the preceding tokens — it is a
//! pure function of `(secret_key, preceding tokens)`, recomputed identically
//! by the generator and the detector. Neither side stores or transmits any
//! RNG state; the "randomness" of which tokens are favored is entirely a
//! property of the hash, not of a stream either party has to keep in sync.
//! This mirrors this crate's broader convention of avoiding `rand` in favor
//! of hash-seeded determinism (see [`crate::ab_eval`]'s FNV-1a-driven
//! bootstrap); this module hand-rolls its own `splitmix64` avalanche step
//! and FNV-1a mixing rather than adding a dependency.
//!
//! # How this differs from every other citation/attribution module here
//!
//! Three modules in this crate sound adjacent but solve a different
//! problem entirely — they attribute **retrieved sources** to a generated
//! answer; none of them embeds or detects a statistical signal in the
//! generated text's *token choices themselves*:
//!
//! | Module | Operates on | Answers |
//! |---|---|---|
//! | [`attribution`](crate::attribution) | generated sentences + retrieved sources | "which retrieved source(s) does this sentence come from?" (inline `[1]`-style citations) |
//! | [`quote_grounding`](crate::quote_grounding) | a claim + retrieved documents | "what is the minimal verbatim source span that supports this claim?" |
//! | [`citation_verification`](crate::citation_verification) | a claim + its cited passage | "is this claim actually entailed by the passage it cites?" (lexical NLI-lite) |
//! | `watermarking` (this module) | the generated token sequence alone, plus a secret key | "was this text's *token sampling process* influenced by a key I know?" |
//!
//! All three of those modules need retrieved sources as input and reason
//! about *content* (which source, which span, is it entailed). Watermarking
//! needs no source documents at all — it needs a secret key and, at
//! generation time, the logits the decoder was about to sample from. It is
//! provenance of the *generation process*, not attribution of *content* to
//! *sources*, and it is undetectable without the key even when the sources
//! backing every sentence are fully known.
//!
//! # Components
//!
//! | Item | Responsibility |
//! |---|---|
//! | [`WatermarkConfig`] | `gamma`, `delta`, `context_width`, `secret_key`, `vocab_size`, `z_threshold` |
//! | [`WatermarkHasher`] | Deterministic green/red vocabulary partitioning from a predecessor context |
//! | [`WatermarkGenerator`] | Soft/hard logit biasing + greedy decoding |
//! | [`WatermarkDetector`] | Recompute green counts from text alone; z-score + p-value |
//! | [`WatermarkMode`] | [`Soft`](WatermarkMode::Soft) (bias) vs [`Hard`](WatermarkMode::Hard) (restrict) |
//! | [`WatermarkDetection`] | `green_count`, `total_scored`, `z_score`, `p_value`, `is_watermarked` |
//! | [`WatermarkError`] | Error variants for the whole module |
//!
//! # Limitations (documented, not hidden)
//!
//! * **Low-entropy text carries a weak watermark.** Soft mode's bias can
//!   only change a decoding step's outcome when the model's own logit
//!   margin was smaller than `delta`; a step the model was already certain
//!   about is unaffected, and un-affected steps contribute no signal to
//!   detection. Highly deterministic text (boilerplate, code, short
//!   factual answers) will therefore show a weaker z-score than the same
//!   *length* of open-ended prose, even though both were generated with an
//!   identical watermark configuration.
//! * **The first `context_width` tokens of any text are unscoreable** —
//!   they have no predecessor context to hash, so [`WatermarkDetector::detect`]
//!   excludes them from both the numerator and denominator of the count.
//! * **Robustness trades off against context width.** A wider
//!   `context_width` is harder for an adversary to reverse-engineer, but a
//!   single token substitution then corrupts more downstream positions'
//!   recomputed green lists (its own, plus up to `context_width` successors'
//!   contexts) — see the `robustness_*` tests in this module's test suite
//!   for a direct measurement.
//!
//! # Quick start
//!
//! ```rust
//! # #[cfg(feature = "watermarking")]
//! # {
//! use oxirag::watermarking::{WatermarkConfig, WatermarkDetector, WatermarkGenerator, WatermarkMode};
//!
//! let vocab_size = 32;
//! let config = WatermarkConfig::new(vocab_size, /* secret_key */ 0xC0FF_EE42)
//!     .with_gamma(0.5)
//!     .with_delta(4.0)
//!     .with_context_width(1);
//!
//! let generator = WatermarkGenerator::new(config.clone(), WatermarkMode::Soft)
//!     .expect("valid config");
//!
//! // A flat (maximum-entropy) logit distribution: every token is equally
//! // likely absent a watermark, so the green-list bias fully determines the
//! // decoding choice at every step past the first.
//! let flat_logits = vec![0.0_f64; vocab_size];
//! let steps: Vec<Vec<f64>> = (0..200).map(|_| flat_logits.clone()).collect();
//! let tokens = generator.generate_sequence(&steps).expect("valid steps");
//!
//! let detector = WatermarkDetector::new(config).expect("valid config");
//! let detection = detector.detect(&tokens).expect("enough tokens to score");
//!
//! // Every fully-contexted step had no genuine competition among logits, so
//! // the green-list bias won every one of them: the detector recovers a very
//! // strong signal.
//! assert!(detection.z_score > 10.0);
//! assert!(detection.is_watermarked);
//! # }
//! ```

pub mod detector;
pub mod generator;
pub mod hasher;
pub mod stats;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
pub mod types;

pub use detector::WatermarkDetector;
pub use generator::WatermarkGenerator;
pub use hasher::WatermarkHasher;
pub use types::{
    WatermarkConfig, WatermarkDetection, WatermarkError, WatermarkMode, WatermarkResult,
    WatermarkTokenId,
};
