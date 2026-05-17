# Layer 2: Speculator — Draft Verification

The Speculator is Layer 2 of OxiRAG's pipeline. After Layer 1 (Echo) retrieves
semantically similar documents, the Speculator takes a draft answer and the
retrieved context, then decides whether to Accept, Revise, or Reject the draft.

This implements OxiRAG's "Speculative RAG" vision: the vector cache is treated as
a source of draft candidates rather than final answers. The Speculator acts as a
lightweight verifier that prevents low-quality or inconsistent drafts from
propagating to users.

## What the Speculator Does

```
Draft answer + Retrieved context
          │
          ▼
┌───────────────────────┐
│      Speculator       │
│                       │
│  1. Analyze draft     │
│  2. Compare against   │
│     context           │
│  3. Score confidence  │
└───────────────────────┘
          │
          ▼
  SpeculationResult {
    decision: Accept | Revise | Reject,
    confidence: f32,
    explanation: String,
    issues: Vec<String>,
  }
```

If the decision is `Revise`, the pipeline calls `revise_draft` and re-verifies
up to `max_revisions` times before falling back to the original or rejecting.

Relevant source files:

- `src/layer2_speculator/mod.rs` — module re-exports
- `src/layer2_speculator/traits.rs` — `Speculator` trait and `RuleBasedSpeculator`
- `src/layer2_speculator/candle_slm.rs` — `CandleSlmSpeculator` and `CandleSLM`
- `src/layer2_speculator/calibration.rs` — `ConfidenceCalibrator`
- `src/layer2_speculator/verification.rs` — `VerificationPipeline`

## RuleBasedSpeculator (Default, Zero Dependencies)

`RuleBasedSpeculator` is the default verifier. It applies deterministic heuristics
without any model inference:

- Rejects empty drafts immediately (confidence 0.0).
- Checks for uncertainty markers (`"maybe"`, `"might"`, `"I think"`, etc.).
- Computes word-overlap ratio between draft and context.
- Adjusts confidence based on draft's own confidence score.

```rust
use oxirag::layer2_speculator::{RuleBasedSpeculator, SpeculatorConfig};
use oxirag::layer2_speculator::Speculator;
use oxirag::types::{Draft, SearchResult};

// Use defaults: accept_threshold=0.9, reject_threshold=0.3, max_revisions=2.
let speculator = RuleBasedSpeculator::default();

// Custom thresholds:
let config = SpeculatorConfig {
    accept_threshold: 0.85,
    reject_threshold: 0.2,
    max_revisions: 3,
    temperature: 0.3,
    ..SpeculatorConfig::default()
};
let speculator = RuleBasedSpeculator::new(config);
```

Example verification loop:

```rust
let draft = Draft::new(
    "Rust prevents data races at compile time through the ownership system.",
    "How does Rust handle concurrency safety?",
).with_confidence(0.88);

let context: Vec<SearchResult> = /* ... retrieved by EchoLayer ... */ vec![];

let result = speculator.verify_draft(&draft, &context).await?;
println!("Decision: {:?}", result.decision);
println!("Confidence: {:.2}", result.confidence);
println!("Explanation: {}", result.explanation);

if !result.issues.is_empty() {
    println!("Issues:");
    for issue in &result.issues {
        println!("  - {issue}");
    }
}
```

Performance: `RuleBasedSpeculator` runs in under 1 µs per verification call on
modern hardware. It is suitable for latency-sensitive paths and WASM deployments
where model inference is unavailable.

## CandleSlmSpeculator (SLM-backed, Requires `speculator` Feature)

`CandleSlmSpeculator` uses a small language model (Phi-2 or Phi-3) to perform
natural-language verification of drafts against context. This produces higher
quality verification decisions at the cost of model inference latency.

```toml
# Cargo.toml
oxirag = { version = "0.6", features = ["speculator"] }
```

```rust
use oxirag::layer2_speculator::{CandleSlmSpeculator, CandleSlmConfig, CandleSlmDevice};
use oxirag::layer2_speculator::Speculator;

// Load Phi-2 (~2.7B params, ~5.5 GB weights download on first use).
// The model is cached in HF_HOME after the first download.
let speculator = CandleSlmSpeculator::new_phi2()?;

// Or configure explicitly:
let config = CandleSlmConfig {
    model_id: "microsoft/phi-2".to_string(),
    revision: "main".to_string(),
    device: CandleSlmDevice::Cpu,
    max_tokens: 512,
    temperature: 0.2,
};
let speculator = CandleSlmSpeculator::new(config)?;
```

The SLM receives a structured prompt built from the verification template in
`src/layer2_speculator/traits.rs`:

```
Question: {query}

Context:
{retrieved_document_1}
{retrieved_document_2}

Draft Answer:
{draft}

Please verify this draft answer and provide your decision.
```

The model output is parsed for `ACCEPT`, `REVISE`, or `REJECT` keywords.

Performance (CPU inference, x86_64 Linux, no AVX2 BLAS):

| Model | Verification latency |
|---|---|
| Phi-2 (2.7B) | ~50 ms per verification |
| Phi-3-mini (3.8B) | ~80 ms per verification |

Enable CUDA for 10–20x inference speedup:

```rust
use oxirag::layer2_speculator::CandleSlmDevice;

let config = CandleSlmConfig {
    device: CandleSlmDevice::Cuda(0), // GPU 0
    ..CandleSlmConfig::default()
};
```

### MockSlmSpeculator (Testing)

For tests that need to exercise the SLM integration layer without downloading
models:

```rust
use oxirag::layer2_speculator::MockSlmSpeculator;

// Always accepts with confidence 0.9.
let speculator = MockSlmSpeculator::default();
```

## Confidence Calibration

Raw confidence scores from the Speculator may not be well-calibrated (a score of
0.9 should mean the draft is correct 90% of the time). `ConfidenceCalibrator`
applies statistical calibration to improve reliability.

```rust
use oxirag::layer2_speculator::{ConfidenceCalibrator, CalibrationMethod};

// Collect (raw_score, is_correct) pairs from labelled examples.
let calibration_data: Vec<(f32, bool)> = vec![
    (0.92, true), (0.85, true), (0.60, false), (0.75, true), (0.45, false),
    // ... hundreds of examples for production use
];

// Platt scaling: logistic regression over raw scores.
let calibrator = ConfidenceCalibrator::fit(
    &calibration_data,
    CalibrationMethod::PlattScaling,
)?;

let raw = 0.78;
let calibrated = calibrator.calibrate(raw);
println!("Calibrated confidence: {calibrated:.3}");

// Other methods:
// CalibrationMethod::TemperatureScaling  — single scalar temperature parameter
// CalibrationMethod::IsotonicRegression  — monotonic piecewise-linear mapping
// CalibrationMethod::HistogramBinning    — simple binned mapping
```

`CalibrationStats` reports the calibration quality:

```rust
let stats = calibrator.stats();
println!("Expected Calibration Error: {:.4}", stats.ece);
println!("Max Calibration Error: {:.4}", stats.mce);
```

## Multi-Stage VerificationPipeline

For fine-grained control, compose multiple verification stages into a
`VerificationPipeline`. Stages are applied in order; a stage can abort the
pipeline early if it reaches high enough confidence.

```rust
use oxirag::layer2_speculator::{
    VerificationPipeline, PipelineBuilder,
    KeywordMatchStage, SemanticSimilarityStage, FactualConsistencyStage,
    AggregationMethod,
};

let pipeline = PipelineBuilder::new()
    .add_stage(KeywordMatchStage::new(0.5))        // Stage 1: fast keyword overlap
    .add_stage(SemanticSimilarityStage::new(0.7))  // Stage 2: embedding similarity
    .add_stage(FactualConsistencyStage::new(0.85)) // Stage 3: fact consistency
    .with_aggregation(AggregationMethod::WeightedMean)
    .build();

let stage_result = pipeline.verify(draft, &context).await?;
println!("Aggregated confidence: {:.3}", stage_result.confidence);
for (stage, score) in stage_result.stage_scores.iter() {
    println!("  {stage}: {score:.3}");
}
```

Available built-in stages:

| Stage | What it checks | Typical threshold |
|---|---|---|
| `KeywordMatchStage` | Word overlap between draft and context | 0.3–0.5 |
| `SemanticSimilarityStage` | Cosine similarity of draft vs. context embeddings | 0.6–0.8 |
| `FactualConsistencyStage` | Cross-encoder style factual alignment | 0.8–0.9 |

Implement `VerificationStage` to add a custom stage:

```rust
use oxirag::layer2_speculator::{VerificationStage, StageResult};

struct DomainSpecificStage;

#[async_trait::async_trait]
impl VerificationStage for DomainSpecificStage {
    fn name(&self) -> &str { "domain-check" }

    async fn verify(
        &self,
        draft: &Draft,
        context: &[SearchResult],
    ) -> Result<StageResult, oxirag::error::SpeculatorError> {
        let score = if draft.content.contains("citing") { 0.9 } else { 0.6 };
        Ok(StageResult { confidence: score, passed: score > 0.7, details: None })
    }
}
```

## Hidden State Verification (Requires `hidden-states` Feature)

The `HiddenStateSpeculator` verifies drafts by comparing transformer hidden states
between the query and the candidate answer. Factual divergences manifest as
measurable shifts in intermediate layer activations.

```toml
oxirag = { version = "0.6", features = ["speculator", "hidden-states"] }
```

```rust
use oxirag::layer2_speculator::{HiddenStateSpeculator, HiddenStateSpeculatorConfig};
use oxirag::layer2_speculator::MockHiddenStateSpeculator;

// For testing (no model required):
let speculator = MockHiddenStateSpeculator::new(768);

// For production (uses CandleHiddenStateProvider):
// see src/hidden_states/candle_provider.rs for CandleHiddenStateConfig
let config = HiddenStateSpeculatorConfig {
    divergence_threshold: 0.15,
    num_verification_layers: 4,
    ..HiddenStateSpeculatorConfig::default()
};
let speculator = HiddenStateSpeculator::new(config, hidden_state_provider)?;
```

The `DivergencePoint` structure identifies which transformer layer showed the
highest divergence, enabling post-hoc analysis of which part of the reasoning
chain is suspect:

```rust
if let Some(divergence) = result.divergence_point {
    println!("Divergence detected at layer {}", divergence.layer_index);
    println!("Divergence magnitude: {:.4}", divergence.magnitude);
}
```

## Performance Reference

| Speculator | Latency | Accuracy | Model download |
|---|---|---|---|
| `RuleBasedSpeculator` | < 1 µs | Moderate (heuristic) | None |
| `MockSlmSpeculator` | < 1 µs | Fixed (test use only) | None |
| `CandleSlmSpeculator` (Phi-2, CPU) | ~50 ms | High | ~5.5 GB |
| `CandleSlmSpeculator` (Phi-2, CUDA) | ~3 ms | High | ~5.5 GB |
| `HiddenStateSpeculator` (BERT, CPU) | ~15 ms | High for factual | ~400 MB |

For most production deployments: start with `RuleBasedSpeculator` and upgrade to
`CandleSlmSpeculator` if you observe unacceptable false-positive rates in
acceptance decisions.
