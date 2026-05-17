# ADR-0002: MultiModalEmbeddingProvider to EmbeddingProvider Blanket Impl

**Status:** Accepted
**Date:** 2026-05-17
**Deciders:** KitaSan

## Context

`EchoLayer<E, V>` is generic over `E: EmbeddingProvider`. The `EmbeddingProvider`
trait exposes `embed(&str)` and `embed_batch(&[&str])` — pure text operations.

When CLIP support was added (v0.4.0, feature `multimodal`), a new trait was
introduced:

```rust
// src/layer1_echo/traits.rs
pub trait MultiModalEmbeddingProvider {
    async fn embed_multi(&self, input: EmbeddingInput<'_>) -> Result<Vec<f32>, EmbeddingError>;
    async fn embed_multi_batch(&self, inputs: &[EmbeddingInput<'_>]) -> ...;
    fn dimension(&self) -> usize;
    fn model_id(&self) -> &str;
}
```

`EmbeddingInput<'a>` is an enum covering `Text(&'a str)`, `Image(&'a [u8])`, and
`TextAndImage { text, image }`. The concrete `CandleClipProvider` implements
`MultiModalEmbeddingProvider`.

The problem: `EchoLayer<CandleClipProvider, V>` would not compile without also
implementing `EmbeddingProvider` for `CandleClipProvider`, because `EchoLayer`
calls `self.embedding_provider.embed(query)` in its `search` path. Requiring
CLIP users to duplicate the `EmbeddingProvider` impl manually for every CLIP
provider would be error-prone and break the open/closed principle.

## Decision

Add a blanket implementation in `src/layer1_echo/traits.rs`:

```rust
// Native (Send + Sync required for async executor safety)
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl<T: MultiModalEmbeddingProvider + Send + Sync> EmbeddingProvider for T {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_multi(EmbeddingInput::Text(text)).await
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let inputs: Vec<EmbeddingInput<'_>> =
            texts.iter().map(|t| EmbeddingInput::Text(t)).collect();
        self.embed_multi_batch(&inputs).await
    }

    fn dimension(&self) -> usize {
        MultiModalEmbeddingProvider::dimension(self)
    }

    fn model_id(&self) -> &str {
        MultiModalEmbeddingProvider::model_id(self)
    }
}
```

A parallel `#[async_trait(?Send)]` impl covers WASM targets where `Send` is
not available.

With this blanket in place, `CandleClipProvider` automatically satisfies
`EmbeddingProvider`, and any CLIP provider can be used directly with `EchoLayer`:

```rust
use oxirag::layer1_echo::{CandleClipProvider, ClipPreset, EchoLayer, InMemoryVectorStore};

// Works because of the blanket impl — no code changes to EchoLayer required.
let provider = CandleClipProvider::from_preset(ClipPreset::VitBase32)?;
let echo = EchoLayer::new(provider, InMemoryVectorStore::new(512));
```

## Rationale

- Zero changes to existing pipeline code: callers that use `CandleEmbeddingProvider`
  or `MockEmbeddingProvider` are unaffected.
- Text-only queries through a CLIP model just work: `embed("query")` routes to
  `embed_multi(EmbeddingInput::Text("query"))`, which CLIP handles via its text
  encoder branch.
- The approach is consistent with standard Rust `From`/`Into` blanket patterns.
- Alternative: auto-deriving `EmbeddingProvider` via a proc-macro attribute on
  `MultiModalEmbeddingProvider` implementors. Rejected: proc-macros add build-time
  complexity for no additional expressiveness.

## Consequences

- A type cannot independently implement both `MultiModalEmbeddingProvider` and
  `EmbeddingProvider` due to the Rust orphan rule: the blanket impl is unconditional
  for all `T: MultiModalEmbeddingProvider + Send + Sync`. If a future provider
  needed different behavior for the text-only path, it would need to be wrapped
  in a newtype.
- The blanket impl uses `EmbeddingInput::Text` when called as `EmbeddingProvider`.
  This means image-query paths (e.g., searching with an image byte slice) require
  the caller to use `embed_multi` directly on the `MultiModalEmbeddingProvider`
  trait, not the `Echo` layer's `search` method.
- Users on WASM must use the `?Send` variant; the cfg guards handle this
  transparently and are invisible to library consumers.

## Alternatives Considered

### Newtype wrapper

`TextOnlyClipProvider(CandleClipProvider)` that manually implements
`EmbeddingProvider` by delegating to the text branch. Rejected: the ergonomic
overhead makes CLIP adoption unnecessarily friction-heavy. Every CLIP provider
variant would need its own newtype.
