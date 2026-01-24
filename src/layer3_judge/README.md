# Layer 3: Judge

SMT-based logic verification layer.

## Overview

The Judge layer provides:
- Logical claim extraction from text
- SMT-LIB encoding of claims
- Formal verification using SMT solvers
- Consistency checking across claims

## Components

### Traits

- **`ClaimExtractor`**: Extract logical claims from text
  - `extract_claims()`: Parse text into structured claims
  - `to_smtlib()`: Convert claims to SMT-LIB format

- **`SmtVerifier`**: Verify claims using SMT solver
  - `verify_claim()`: Check single claim
  - `verify_claims()`: Batch verification
  - `check_consistency()`: Verify claims don't contradict

- **`Judge`**: Combined verification interface
  - `judge()`: Full verification with context
  - `quick_judge()`: Fast verification without SMT

### Claim Structures

```rust
pub enum ClaimStructure {
    Predicate { subject, predicate, object },
    Comparison { left, operator, right },
    And(Vec<ClaimStructure>),
    Or(Vec<ClaimStructure>),
    Not(Box<ClaimStructure>),
    Implies { premise, conclusion },
    Quantified { quantifier, variable, domain, body },
    Raw(String),
}
```

### Implementations

- **`PatternClaimExtractor`**: Keyword-based extraction
- **`AdvancedClaimExtractor`**: NLP heuristics for predicates, comparisons, conditionals
- **`MockSmtVerifier`**: Confidence-based mock verification
- **`OxizVerifier`**: Real SMT verification using OxiZ (requires `judge` feature)

## Usage

```rust
use oxirag::layer3_judge::*;

// Create judge
let judge = JudgeImpl::new(
    AdvancedClaimExtractor::new(),
    MockSmtVerifier::default(),
    JudgeConfig::default(),
);

// Verify a draft
let draft = Draft::new("The sky is blue.", "What color is the sky?");
let context = vec![/* search results */];

let result = judge.judge(&draft, &context).await?;

println!("Status: {:?}", result.status);
println!("Confidence: {:.2}", result.confidence);
println!("Claims verified: {}", result.claim_results.len());
```

## Configuration

```rust
let config = JudgeConfig {
    timeout_ms: 5000,           // SMT solver timeout
    max_claims: 10,             // Maximum claims to verify
    check_consistency: true,    // Check for contradictions
    min_claim_confidence: 0.5,  // Filter low-confidence claims
    generate_counterexamples: true,
};
```

## Verification Status

```rust
pub enum VerificationStatus {
    Verified,   // All claims proven true
    Falsified,  // At least one claim proven false
    Unknown,    // Could not determine
}
```

## SMT-LIB Output

Claims are converted to SMT-LIB format:

```lisp
; Predicate: "X is Y"
(assert (is |X| |Y|))

; Comparison: "A > B"
(assert (> |A| |B|))

; Implication: "if P then Q"
(assert (=> |P| |Q|))
```

## File Structure

```
layer3_judge/
├── mod.rs              # Module exports
├── traits.rs           # Core traits + PatternClaimExtractor
├── claim_extractor.rs  # AdvancedClaimExtractor
└── oxiz_verifier.rs    # JudgeImpl + SMT verifiers
```

## Advanced Usage

### Custom Claim Patterns

```rust
let extractor = PatternClaimExtractor::new()
    .with_pattern(ClaimPattern {
        name: "custom".to_string(),
        keywords: vec!["must".to_string(), "shall".to_string()],
        structure_type: PatternStructureType::Predicate,
    });
```

### Quick Verification

For fast verification without full SMT:

```rust
let result = judge.quick_judge(&draft).await?;
// Uses only claim extraction and confidence heuristics
```
