//! The entity linker: candidate generation, context disambiguation, and NIL.

use crate::entity_linking::catalog::EntityCatalog;
use crate::entity_linking::types::{
    EntityLinkConfig, EntityLinkError, EntityMention, LinkedEntity,
};

// ── Lexical pseudo-embedding ────────────────────────────────────────────────

/// Compute a deterministic lexical pseudo-embedding for `text`.
///
/// Algorithm: tokenise on non-alphanumeric boundaries (keeping tokens of length
/// at least two), hash each lower-cased token to a bucket with FNV-1a,
/// accumulate per-bucket counts, then L2-normalise to the requested `dim`.
#[must_use]
pub fn embed(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }
    let mut buckets = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 2 {
            continue;
        }
        // Deterministic hash: FNV-1a over the lower-cased token bytes.
        let mut h: u64 = 14_695_981_039_346_656_037;
        for b in token.to_lowercase().as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(1_099_511_628_211);
        }
        #[allow(clippy::cast_possible_truncation)]
        let idx = (h as usize) % dim;
        buckets[idx] += 1.0;
    }
    let norm: f32 = buckets.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut buckets {
            *x /= norm;
        }
    }
    buckets
}

/// Cosine similarity between two L2-normalised, equal-length vectors.
///
/// Returns `0.0` for mismatched lengths or empty inputs.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── EntityLinker ────────────────────────────────────────────────────────────

/// Links surface-form mentions to canonical entities in a catalog.
///
/// Candidate generation is an exact, case-insensitive surface-form lookup.
/// When a mention is unambiguous (one candidate) it is linked directly with a
/// confidence of `1.0`. When several candidates share the surface form, the
/// linker picks the candidate whose description is most similar to the
/// surrounding context (cosine over lexical pseudo-embeddings). If no candidate
/// exists, or the chosen candidate's confidence is below the configured floor,
/// the mention is left unlinked (NIL).
#[derive(Debug, Clone)]
pub struct EntityLinker {
    /// Linking configuration.
    config: EntityLinkConfig,
    /// The canonical entity catalog.
    catalog: EntityCatalog,
}

impl EntityLinker {
    /// Create a linker over `catalog` with the given `config`.
    #[must_use]
    pub fn new(config: EntityLinkConfig, catalog: EntityCatalog) -> Self {
        Self { config, catalog }
    }

    /// Borrow the linking configuration.
    #[must_use]
    pub fn config(&self) -> &EntityLinkConfig {
        &self.config
    }

    /// Borrow the underlying catalog.
    #[must_use]
    pub fn catalog(&self) -> &EntityCatalog {
        &self.catalog
    }

    /// Detect candidate mentions in `text`.
    ///
    /// A mention is a maximal run of contiguous capitalised words (each at least
    /// two characters, beginning with an uppercase character), joined by single
    /// spaces. Byte offsets refer to the start of the first word and the end of
    /// the last word of the run in the original `text`. Leading and trailing
    /// punctuation is stripped from the emitted text (and the offsets), so the
    /// surface form is directly usable for catalog lookup.
    #[must_use]
    pub fn detect_mentions(&self, text: &str) -> Vec<EntityMention> {
        let mut mentions = Vec::new();
        let mut run: Option<(usize, usize)> = None;
        for (offset, word) in word_spans(text) {
            if is_capitalised_word(word) {
                match run {
                    Some((start, _)) => run = Some((start, offset + word.len())),
                    None => run = Some((offset, offset + word.len())),
                }
            } else if let Some((start, end)) = run.take() {
                push_mention(&mut mentions, text, start, end);
            }
        }
        if let Some((start, end)) = run.take() {
            push_mention(&mut mentions, text, start, end);
        }
        mentions
    }

    /// Link a single `mention` to the catalog using `context` to disambiguate.
    ///
    /// With one candidate the link is accepted at confidence `1.0`; with several
    /// the highest context-versus-description cosine wins. The result is NIL
    /// (`entity_id` is `None`) when there is no candidate or when the best
    /// confidence is below [`EntityLinkConfig::min_confidence`].
    ///
    /// [`EntityLinkConfig::min_confidence`]: crate::entity_linking::EntityLinkConfig::min_confidence
    #[must_use]
    pub fn link(&self, mention: &str, context: &str) -> LinkedEntity {
        let candidates = self.catalog.candidates(mention);
        let Some((idx, confidence)) = self.best_candidate(&candidates, context) else {
            return LinkedEntity::nil(mention);
        };
        if confidence < self.config.min_confidence {
            return LinkedEntity::nil(mention);
        }
        match self.catalog.entity(idx) {
            Some(entity) => LinkedEntity::linked(mention, entity.id.clone(), confidence),
            None => LinkedEntity::nil(mention),
        }
    }

    /// Detect mentions in `text` and link each one using `text` as context.
    ///
    /// # Errors
    ///
    /// Returns [`EntityLinkError::EmptyCatalog`] when the catalog holds no
    /// entities.
    ///
    /// [`EntityLinkError::EmptyCatalog`]: crate::entity_linking::EntityLinkError::EmptyCatalog
    pub fn link_all(&self, text: &str) -> Result<Vec<LinkedEntity>, EntityLinkError> {
        if self.catalog.is_empty() {
            return Err(EntityLinkError::EmptyCatalog);
        }
        Ok(self
            .detect_mentions(text)
            .into_iter()
            .map(|m| self.link(&m.text, text))
            .collect())
    }

    /// Select the best candidate index and its confidence for `context`.
    ///
    /// Returns `None` for an empty candidate slice. A single candidate yields a
    /// confidence of `1.0`. Ties in cosine score are broken deterministically by
    /// the smaller catalog index.
    fn best_candidate(&self, candidates: &[usize], context: &str) -> Option<(usize, f32)> {
        match candidates {
            [] => None,
            [only] => Some((*only, 1.0)),
            many => {
                let ctx = embed(context, self.config.dim);
                let mut best: Option<(usize, f32)> = None;
                for &idx in many {
                    let score = match self.catalog.entity(idx) {
                        Some(entity) => cosine(&ctx, &embed(&entity.description, self.config.dim)),
                        None => 0.0,
                    };
                    let replace = match best {
                        None => true,
                        Some((_, best_score)) => score > best_score,
                    };
                    if replace {
                        best = Some((idx, score));
                    }
                }
                best
            }
        }
    }
}

// ── Mention-detection helpers ───────────────────────────────────────────────

/// Push the run `text[start..end]`, trimmed of leading/trailing punctuation,
/// onto `mentions` with offsets adjusted to the trimmed span.
///
/// Empty runs (all punctuation) are skipped.
fn push_mention(mentions: &mut Vec<EntityMention>, text: &str, start: usize, end: usize) {
    let raw = &text[start..end];
    let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return;
    }
    let lead = trimmed.as_ptr() as usize - raw.as_ptr() as usize;
    let new_start = start + lead;
    let new_end = new_start + trimmed.len();
    mentions.push(EntityMention::new(trimmed.to_string(), new_start, new_end));
}

/// Iterate over `(byte_offset, word)` pairs for whitespace-delimited words.
fn word_spans(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.split_whitespace().map(move |word| {
        // `split_whitespace` yields sub-slices of `text`; recover the offset via
        // pointer arithmetic against the base of the original string.
        let offset = word.as_ptr() as usize - text.as_ptr() as usize;
        (offset, word)
    })
}

/// Return `true` when `word`, stripped of trailing punctuation, is a
/// capitalised token of at least two characters.
fn is_capitalised_word(word: &str) -> bool {
    let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
    trimmed.chars().count() >= 2 && trimmed.chars().next().is_some_and(char::is_uppercase)
}
