# Layer 3: Judge — Logic Verification

The Judge is Layer 3 of OxiRAG's pipeline. After the Speculator (Layer 2)
decides whether a draft answer is generally consistent with the retrieved context,
the Judge extracts explicit logical claims from the draft and formally verifies
them using an SMT (Satisfiability Modulo Theories) solver. This catches structured
factual errors that semantic similarity checks cannot detect — for example, a
draft that says "Python 3 was released in 2008" when the context says 2008 is
correct but a numeric comparison would reveal a contradiction against another
claim in the same draft.

## What the Judge Does

```
Verified draft (from Speculator)
          │
          ▼
┌─────────────────────────┐
│    ClaimExtractor       │  Parses natural language into
│    (dependency parsing, │  LogicalClaim structures:
│     pattern matching)   │  - Predicate claims
└─────────────────────────┘  - Numeric comparisons
          │                  - Temporal claims
          ▼                  - Causal claims
  Vec<LogicalClaim>          - Modal claims
          │
          ▼
┌─────────────────────────┐
│     SmtVerifier         │  Encodes claims as SMT-LIB
│     (OxiZ solver or     │  assertions and calls the
│      mock rules)        │  solver to check satisfiability
└─────────────────────────┘
          │
          ▼
  VerificationResult {
    claims: Vec<ClaimVerificationResult>,
    overall_status: Verified | Falsified | Unknown | Timeout,
    explanation: String,
  }
```

Relevant source files:

- `src/layer3_judge/mod.rs` — module re-exports
- `src/layer3_judge/traits.rs` — `Judge`, `ClaimExtractor`, `SmtVerifier` traits
- `src/layer3_judge/claim_extractor.rs` — `AdvancedClaimExtractor`
- `src/layer3_judge/oxiz_verifier.rs` — `JudgeImpl`, `MockSmtVerifier`, `OxizVerifier`
- `src/layer3_judge/dependency_parser.rs` — SVO extraction for claim parsing
- `src/layer3_judge/incremental.rs` — `IncrementalConsistencyChecker`

## Using MockSmtVerifier (Default, Zero Dependencies)

`MockSmtVerifier` applies rule-based verification without invoking any external
solver. It is suitable for testing and for deployments where formal verification
overhead is unacceptable.

```rust
use oxirag::layer3_judge::{
    AdvancedClaimExtractor, JudgeImpl, MockSmtVerifier, JudgeConfig,
};
use oxirag::layer3_judge::Judge;
use oxirag::types::Draft;

let judge = JudgeImpl::new(
    AdvancedClaimExtractor::new(),
    MockSmtVerifier::default(),
    JudgeConfig::default(),
);

let draft = Draft::new(
    "Python 3.0 was released in December 2008. It introduced major breaking changes.",
    "When was Python 3 released?",
);

let result = judge.verify(&draft).await?;
println!("Status: {:?}", result.overall_status);
println!("{}", result.explanation);

for cv in &result.claims {
    println!("  Claim: {:?}", cv.claim);
    println!("  Status: {:?}", cv.status);
}
```

### JudgeConfig

```rust
use oxirag::layer3_judge::JudgeConfig;
use std::time::Duration;

let config = JudgeConfig {
    // Maximum number of claims to extract per draft.
    max_claims: 20,
    // SMT solver timeout per claim.
    solver_timeout: Duration::from_millis(100),
    // Require all claims to be verified for overall Verified status.
    require_all_verified: false,
    // Minimum claims that must pass for an overall Verified result.
    min_verified_claims: 1,
    ..JudgeConfig::default()
};

let judge = JudgeImpl::new(
    AdvancedClaimExtractor::new(),
    MockSmtVerifier::default(),
    config,
);
```

## OxiZ SMT Solver (Requires `judge` Feature)

For rigorous formal verification, use `OxizVerifier` which invokes the OxiZ
SMT solver. OxiZ is a pure-Rust SMT solver built on the theory of linear
arithmetic and propositional logic.

```toml
# Cargo.toml
oxirag = { version = "0.6", features = ["judge"] }
```

```rust
use oxirag::layer3_judge::{
    AdvancedClaimExtractor, JudgeImpl, OxizVerifier, JudgeConfig,
};
use std::time::Duration;

let config = JudgeConfig {
    solver_timeout: Duration::from_millis(500),
    ..JudgeConfig::default()
};

let judge = JudgeImpl::new(
    AdvancedClaimExtractor::new(),
    OxizVerifier::new(config.clone()),
    config,
);
```

`OxizVerifier::new` accepts the same `JudgeConfig` as `JudgeImpl`. The solver
timeout is enforced per-claim: if OxiZ does not produce a result within the
configured window, the claim is marked `Unknown` rather than blocking the pipeline.

## Claim Types

The `LogicalClaim` enum covers five categories of verifiable assertions:

### Predicate Claims

Simple subject-predicate assertions:

```
"Rust is a compiled language."
 │         │
 subject   predicate
```

```rust
use oxirag::types::{LogicalClaim, ClaimStructure};

// AdvancedClaimExtractor detects these automatically.
// Manual construction for testing:
let claim = LogicalClaim::Predicate {
    subject: "Rust".to_string(),
    predicate: "is a compiled language".to_string(),
};
```

### Numeric Claims

Comparison relationships between quantities:

```
"Python 3.0 was released in 2008."
          └── numeric: year(Python3.0) == 2008
```

```rust
use oxirag::types::{LogicalClaim, ClaimStructure, ComparisonOp};

let claim = LogicalClaim::Numeric {
    left: "year_python_3".to_string(),
    operator: ComparisonOp::Equal,
    right: "2008".to_string(),
};
```

`ComparisonOp` variants: `Equal`, `NotEqual`, `LessThan`, `LessOrEqual`,
`GreaterThan`, `GreaterOrEqual`.

### Temporal Claims

Ordering relationships between events in time:

```
"Python was created before Java."
 └── temporal: before(Python.created, Java.created)
```

### Causal Claims

Cause-and-effect relationships:

```
"The Rust borrow checker prevents use-after-free bugs."
 └── causal: causes(borrow_checker, prevents_uaf)
```

### Modal Claims

Necessity and possibility assertions:

```
"Rust programs must use the ownership system."
 └── modal: necessary(rust_programs, ownership_system)
```

## Claim Extraction

### AdvancedClaimExtractor

The default extractor combines pattern matching with linguistic analysis to
identify claim-like structures:

```rust
use oxirag::layer3_judge::AdvancedClaimExtractor;
use oxirag::layer3_judge::traits::ClaimExtractor;

let extractor = AdvancedClaimExtractor::new();
let text = "Python 3 was released in 2008. It requires more memory than Python 2.";
let claims = extractor.extract(text).await?;

for claim in &claims {
    println!("{claim:?}");
}
```

### PatternClaimExtractor

A lighter extractor based purely on regex patterns:

```rust
use oxirag::layer3_judge::traits::PatternClaimExtractor;

let extractor = PatternClaimExtractor::default();
```

### Dependency Parser

`DependencyClaimExtractor` leverages SVO (Subject-Verb-Object) triples extracted
by `SimpleDependencyParser` for more accurate claim boundaries in complex sentences:

```rust
use oxirag::layer3_judge::{
    DependencyClaimExtractor, SimpleDependencyParser,
};
use oxirag::layer3_judge::traits::ClaimExtractor;

let parser = SimpleDependencyParser::new();
let extractor = DependencyClaimExtractor::new(parser);

let text = "The borrow checker, which was introduced in Rust 1.0, prevents data races.";
let claims = extractor.extract(text).await?;
// The dependency parser correctly identifies "borrow checker" as the subject
// and "prevents data races" as the predicate, avoiding the relative clause trap.
```

`DependencyTree` and `SvoTriple` can be inspected directly:

```rust
use oxirag::layer3_judge::{SimpleDependencyParser, DependencyParser};

let tree = parser.parse("Rust prevents data races.")?;
for svo in tree.extract_svo_triples() {
    println!("Subject: {}", svo.subject);
    println!("Verb: {}", svo.verb);
    println!("Object: {}", svo.object.as_deref().unwrap_or("(none)"));
}
```

## Claim Normalization and Deduplication

Long documents often contain the same claim expressed in different ways.
`ClaimNormalizer` and `ClaimDeduplicator` canonicalize and deduplicate claims
before sending them to the SMT verifier, reducing solver overhead:

```rust
use oxirag::layer3_judge::{
    DefaultClaimNormalizer, ClaimDeduplicator, ClaimNormalizer,
};

let normalizer = DefaultClaimNormalizer::new();
let deduplicator = ClaimDeduplicator::new();

// Normalize: lowercase, strip punctuation, standardize tense.
let normalized = normalizer.normalize(raw_claim)?;

// Deduplicate a batch.
let unique_claims = deduplicator.deduplicate(claims);
println!("Reduced {} claims to {}", claims.len(), unique_claims.len());
```

## Incremental Consistency Checking

When indexing a stream of documents over time, `IncrementalConsistencyChecker`
maintains a running knowledge base and reports conflicts as new claims arrive:

```rust
use oxirag::layer3_judge::IncrementalConsistencyChecker;

let mut checker = IncrementalConsistencyChecker::new();

// Add claims from document 1.
let result1 = checker.add_claims(claims_from_doc1).await?;
assert_eq!(result1.conflicts.len(), 0);

// Add claims from document 2.
let result2 = checker.add_claims(claims_from_doc2).await?;
if !result2.conflicts.is_empty() {
    for conflict in &result2.conflicts {
        println!("Conflict detected:");
        println!("  New claim: {:?}", conflict.new_claim);
        println!("  Existing claim: {:?}", conflict.existing_claim);
        println!("  Type: {:?}", conflict.conflict_type);
    }
}
```

`ConflictType` variants: `Contradiction` (claims are logically incompatible),
`Inconsistency` (claims could coexist but are unlikely to both be true),
`Duplication` (same claim expressed differently).

## Explanation Generation

The explanation module generates human-readable summaries of verification results:

```rust
use oxirag::layer3_judge::explanation::{generate_summary, generate_counterexample};

let summary = generate_summary(&verification_result);
println!("{summary}");

// For Falsified claims, generate a counterexample if the solver produced one.
if let Some(counterexample) = generate_counterexample(&verification_result) {
    println!("Counterexample: {counterexample}");
}
```

`ExplanationBuilder` provides a fluent interface for constructing structured
explanations:

```rust
use oxirag::layer3_judge::ExplanationBuilder;

let explanation = ExplanationBuilder::new()
    .with_overall_status(verification_result.overall_status.clone())
    .with_verified_count(verified)
    .with_falsified_count(falsified)
    .with_unknown_count(unknown)
    .build();
```

## Performance Reference

| Component | Latency |
|---|---|
| `MockSmtVerifier` (per claim) | < 1 µs |
| `OxizVerifier` (predicate claim) | 0.1–2 ms |
| `OxizVerifier` (numeric comparison) | < 0.1 ms (direct arithmetic) |
| `OxizVerifier` (symbolic comparison) | 0.5–5 ms |
| `AdvancedClaimExtractor` (10 claims) | 0.5–2 ms |
| `DependencyClaimExtractor` (10 claims) | 2–5 ms |

The Judge is the most CPU-intensive layer when `OxizVerifier` is enabled. For
latency-sensitive paths, limit `max_claims` in `JudgeConfig` to 5–10 or use
`MockSmtVerifier` and reserve `OxizVerifier` for asynchronous post-processing.
