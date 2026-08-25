# ADR-0007: `crate::text` — the script-aware substrate for Layers 2–4

**Status:** Accepted
**Date:** 2026-08-25
**Deciders:** KitaSan

## Context

OxiRAG's four layers were measured on a seven-document Japanese corpus — everyday
sentences about penguins, Mt Fuji, curry — indexed and queried through the whole
pipeline. The result:

```text
STATS  {"documents":7,"entities":0,"relationships":0,"vectorWidth":512}
Q: ペンギン 飛ぶ
  layers_used: ["Echo","Speculator","Judge"]
  hits:   ペンギン:0.403  自転車:0.091  コウモリ:0.052
  claims: 0   status: Unknown   summary: "No verifiable claims found"
GRAPH ペンギン  ents 0  rels 0  paths 0
```

Layer 1 was **correct**. It ranks on character bigrams and never needed word
boundaries, so the penguin question found the penguin document by a factor of
four. Every layer above it returned nothing, and reported doing so as a normal
outcome: `Unknown` is a legitimate verification status, and an empty graph is a
legitimate graph. Nothing failed. The engine simply had no opinion about any
Japanese text it was ever given.

The cause was two assumptions, written into every heuristic in Layers 2, 3 and 4:

1. **A sentence ends with `.`, `!` or `?`.** Japanese ends one with `。`, `！`,
   `？`. `split(['.', '!', '?'])` over a Japanese paragraph returns the paragraph.
2. **Words are separated by whitespace.** Japanese has none.
   `split_whitespace()` over a Japanese sentence returns one token, and
   `AdvancedClaimExtractor` bails at `tokens.len() < 2`.

`PatternEntityExtractor` had a third: it finds proper nouns by looking for
capitalised words, in a script with no case.

These are not obscure. They are the first two lines of every text heuristic in the
crate, and there are eighteen separate `split_sentences` functions carrying the
first one.

## Decision

Add `crate::text`: one module holding the script-aware primitives, and route the
four-layer path through it.

```rust
pub fn has_cjk(text: &str) -> bool;
pub fn split_sentences(text: &str) -> Vec<&str>;      // 。！？ .!? and \n
pub fn tokenize(sentence: &str) -> Vec<String>;       // whitespace OR particles
pub fn segment_japanese(sentence: &str) -> Vec<String>;
pub fn split_topic(sentence: &str) -> Option<(&str, &str)>;   // at は / が
pub fn strip_copula(phrase: &str) -> (&str, bool);            // です / である
pub fn is_negated(phrase: &str) -> bool;                      // ません / ない
pub fn japanese_noun_candidates(text: &str) -> Vec<String>;   // Han/Katakana runs
pub fn overlap_score(query_tokens: &[String], sentence: &str) -> f32;
pub fn query_tokens(query: &str) -> Vec<String>;
pub fn normalize_for_compare(sentence: &str) -> String;
```

Three properties are load-bearing:

**`tokenize` dispatches on script, and the English path is byte-identical to what
it replaced.** Text with no CJK in it takes the historical whitespace path. This
is not a compatibility gesture — it is what let the change ship without
re-validating every English extraction in the crate.

**Segmentation is three passes, not one.** Cutting at script transitions alone
splits `飛びません` into `飛` + `びません`, separating a verb from its own
negation — the half that changes the meaning. So: cut at transitions, glue a
hiragana run back onto the content word before it, then strip the grammatical
tail that gluing re-attached. `東京` + `は` → `東京`; `飛` + `びません` →
`飛びません`.

**Nothing here is morphological analysis.** There is no dictionary. `A は B です`
is recognised by its particles, not parsed. MeCrab does the real thing and is a
separate crate; pulling it in would make Japanese support cost an 11 MB dictionary
in a crate whose Layer 1 already handles Japanese for free.

### What changed on top of it

- `AdvancedClaimExtractor` gained a Japanese predicate path.
  `東京は日本の首都です` → `Predicate { 東京, です, 日本の首都 }`;
  `ペンギンは空を飛びません` → `Not(Predicate { ペンギン, 飛びません, 空 })`.
  It is kept as a separate function rather than folded into the existing strategy
  chain, because every strategy in that chain keys off an English keyword list
  (`if`, `because`, `might`) and running those over Japanese tokens produces
  confident nonsense rather than nothing.
- `PatternEntityExtractor` gained Han/Katakana noun candidates and suffix
  classification: `県`/`市`/`山`/`駅` → `Location`, `大学`/`会社` →
  `Organization`. A suffix is the closest available equivalent of a capital
  letter.
- `PatternRelationshipExtractor` gained Japanese patterns and, because Japanese is
  verb-final, now searches the span AFTER the second entity as well as between the
  two. `OxiRAG は OxiZ を使います` puts the verb that names the relationship where
  a between-the-entities search cannot see it.

## Consequences

**Positive.** The same corpus now produces claims, entities and relationships, and
`tests/japanese_pipeline.rs` holds the floor. The primitives are one module, so
the next extractor that needs them does not invent a nineteenth sentence splitter.

**Negative.** `crate::text` is a public module with a Japanese-shaped API in a
crate that is otherwise language-neutral, and it privileges Japanese over every
other non-space-separated language — Chinese, Thai, Khmer — which get the CJK
character-class path for `has_cjk` and nothing else. That is honest about what was
built and tested rather than a general solution; the module is where a general one
would go.

**Accepted limitation.** Recognition is by particle and suffix, so it fails on the
constructions those cues do not cover: relative clauses, `の`-chains longer than
one, embedded quotation. It will also mis-segment a compound whose parts are a
kanji run and a katakana run. The measured alternative was zero claims.

## Alternatives considered

**Depend on MeCrab.** Correct segmentation, and it is a sibling crate. Rejected
**for this crate**, and the reason is not size — it is that MeCrab needs a
dictionary *at run time*, which `cargo build` cannot satisfy and which every
consumer of a language-neutral retrieval library would have to arrange for
themselves. `crate::text` is the floor that works with nothing arranged.

It was NOT rejected as an approach. `EntityExtractor` and `ClaimExtractor` are
public traits precisely so that a consumer who can arrange a dictionary can
supply better implementations, and the COOLJAPAN Playground `/rag` demo does
exactly that: `crates/oxirag-wasm/src/morphology.rs` implements both against
MeCrab and falls back to the ones here when no dictionary is loaded. Measured
difference on that page — `一番高` stops being a graph node (it is `一番` glued
to the stem of `高い`, a seam no run-of-kanji heuristic can see), and `静岡県`
is typed `Location` because IPADIC says `名詞,固有名詞,地域` rather than because
of how its last character is spelled.

The honest summary of this ADR, then: what is here is what a library can do
without data. It is a floor, not a ceiling, and the ceiling is one trait impl
away.

**Unify all eighteen `split_sentences` implementations.** Tempting, and it is the
same bug eighteen times. Rejected as scope: the four-layer path is what was
measured and what is covered by tests. The other fifteen are unchanged and still
wrong on Japanese; `crate::text::split_sentences` is what they should call when
someone gets to them.

**Leave it and document that Japanese is unsupported.** Rejected on the evidence:
Layer 1 already worked, the crate ships to a Japanese-first demo site, and
"unsupported" would have meant documenting a silent `Unknown` as intended
behaviour.
