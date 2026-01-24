# Layer 2: Speculator

Draft verification layer using small language models.

## Overview

The Speculator layer:
- Verifies draft answers against retrieved context
- Identifies issues and inconsistencies
- Revises drafts when needed
- Makes accept/revise/reject decisions

## Components

### Traits

- **`Speculator`**: Core verification interface
  - `verify_draft()`: Analyze draft and return decision
  - `revise_draft()`: Improve draft based on issues found

### Implementations

- **`RuleBasedSpeculator`**: Heuristic verification using:
  - Content length checks
  - Context overlap analysis
  - Uncertainty marker detection
  - Confidence scoring

- **`MockSlmSpeculator`**: Simple heuristic verifier for testing

- **`CandleSlmSpeculator`**: Phi-2 based verification (requires `speculator` feature)

### Decisions

```rust
pub enum SpeculationDecision {
    Accept,  // Draft is good
    Revise,  // Draft needs improvement
    Reject,  // Draft is unusable
}
```

## Usage

```rust
use oxirag::layer2_speculator::*;

// Create speculator
let speculator = RuleBasedSpeculator::default();

// Verify a draft
let draft = Draft::new("Paris is the capital.", "What is the capital of France?");
let context = vec![/* search results */];

let result = speculator.verify_draft(&draft, &context).await?;

match result.decision {
    SpeculationDecision::Accept => println!("Draft accepted"),
    SpeculationDecision::Revise => {
        let revised = speculator.revise_draft(&draft, &context, &result).await?;
        println!("Revised: {}", revised.content);
    }
    SpeculationDecision::Reject => println!("Draft rejected"),
}
```

## Configuration

```rust
let config = SpeculatorConfig {
    temperature: 0.3,
    top_p: 0.9,
    max_tokens: 512,
    accept_threshold: 0.9,  // Auto-accept above this
    reject_threshold: 0.3,  // Auto-reject below this
    max_revisions: 2,
};

let speculator = RuleBasedSpeculator::new(config);
```

## Prompt Templates

The layer includes prompt templates for LLM-based verification:

- `VERIFICATION_SYSTEM`: System prompt for verification
- `VERIFICATION_TEMPLATE`: User prompt template
- `REVISION_TEMPLATE`: Prompt for draft revision

## File Structure

```
layer2_speculator/
├── mod.rs           # Module exports
├── traits.rs        # Speculator trait + RuleBasedSpeculator
└── candle_slm.rs    # Candle SLM implementation
```
