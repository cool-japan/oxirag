//! [`PerplexityCompressor`]: the coarse-to-fine, perplexity-driven prompt
//! compression engine.
//!
//! The engine ties together the three pieces documented elsewhere in this
//! module:
//!
//! 1. **Surrogate LM** ([`PerplexityModel`])
//!    — a smoothed n-gram model fit on the input, giving every token a
//!    surprisal (bits of information) `-log2 P(token | history)`.
//! 2. **Stage A — coarse pruning** — split into segments, rank by information
//!    **density** (mean token surprisal), and drop the lowest-density segments
//!    first (never below [`LlmLinguaConfig::min_segments_retained`], and only
//!    while enough tokens remain to still meet the budget).
//! 3. **Stage B — fine pruning** — the
//!    [`BudgetController`] allocates the token
//!    budget across the survivors *proportionally to density*, then within each
//!    survivor the lowest-surprisal, unprotected tokens are dropped up to that
//!    segment's local allowance.
//!
//! Protected tokens (numbers, capitalized/proper-noun-looking words, and
//! sentence-boundary tokens, all configurable) are never dropped in Stage B, so
//! structurally important but locally predictable content survives.

use super::budget::BudgetController;
use super::ngram_model::{PerplexityModel, normalize_token, split_sentences, surface_tokens};
use super::types::{
    CompressionResult, CompressionTarget, LlmLinguaConfig, LlmLinguaError, SegmentStats,
};

/// Tokenised, scored representation of one input segment.
#[derive(Debug, Clone)]
struct Segment {
    /// Surface (original) token strings, in order.
    surfaces: Vec<String>,
    /// Whether each token is protected from fine pruning.
    protected: Vec<bool>,
    /// Per-token surprisal in bits, aligned with `surfaces`.
    surprisals: Vec<f32>,
}

impl Segment {
    /// Token count.
    fn len(&self) -> usize {
        self.surfaces.len()
    }

    /// Number of protected tokens.
    fn protected_count(&self) -> usize {
        self.protected.iter().filter(|&&p| p).count()
    }

    /// Mean token surprisal — the segment's information density (`0.0` when
    /// empty, which cannot occur for a real segment).
    #[allow(clippy::cast_precision_loss)]
    fn density(&self) -> f32 {
        if self.surprisals.is_empty() {
            return 0.0;
        }
        self.surprisals.iter().sum::<f32>() / self.surprisals.len() as f32
    }
}

/// Coarse-to-fine, perplexity-driven prompt compressor
/// (`LLMLingua`-style; Jiang et al., 2023).
///
/// Holds an [`LlmLinguaConfig`] and is otherwise stateless, so a single
/// instance is cheap to reuse and safe to share across threads. See the
/// [module documentation](self) for the algorithm.
#[derive(Debug, Clone)]
pub struct PerplexityCompressor {
    /// Configuration controlling model order, target, floors, and protection.
    pub config: LlmLinguaConfig,
}

impl PerplexityCompressor {
    /// Constructs a compressor with the supplied configuration.
    #[must_use]
    pub fn new(config: LlmLinguaConfig) -> Self {
        Self { config }
    }

    /// Compresses `text` according to [`self.config`](Self::config).
    ///
    /// # Errors
    ///
    /// - [`LlmLinguaError::InvalidConfig`] — the configuration fails
    ///   [`LlmLinguaConfig::validate`].
    /// - [`LlmLinguaError::EmptyInput`] — `text` yields no tokens.
    pub fn compress(&self, text: &str) -> Result<CompressionResult, LlmLinguaError> {
        self.config.validate()?;

        let mut segments = self.build_segments(text);
        if segments.is_empty() {
            return Err(LlmLinguaError::EmptyInput);
        }

        let total_tokens: usize = segments.iter().map(Segment::len).sum();
        if total_tokens == 0 {
            return Err(LlmLinguaError::EmptyInput);
        }

        self.score_surprisals(&mut segments);

        let densities: Vec<f32> = segments.iter().map(Segment::density).collect();
        let (target, requested_ratio) = self.resolve_target(total_tokens);

        let retained = stage_a_coarse(
            &segments,
            &densities,
            target,
            self.config.min_segments_retained,
        );

        let keep_counts = self.stage_b_allocate(&segments, &densities, &retained, target);

        Ok(assemble_result(
            &segments,
            &densities,
            &retained,
            &keep_counts,
            total_tokens,
            requested_ratio,
        ))
    }

    /// Splits `text` into scored-but-unscored [`Segment`]s (surprisals are
    /// filled later by [`Self::score_surprisals`]).
    fn build_segments(&self, text: &str) -> Vec<Segment> {
        split_sentences(text)
            .into_iter()
            .filter_map(|segment_text| {
                let surfaces = surface_tokens(&segment_text);
                if surfaces.is_empty() {
                    return None;
                }
                let protected = self.protection_flags(&surfaces);
                let surprisals = vec![0.0_f32; surfaces.len()];
                Some(Segment {
                    surfaces,
                    protected,
                    surprisals,
                })
            })
            .collect()
    }

    /// Computes the protection flag for each surface token of one segment.
    fn protection_flags(&self, surfaces: &[String]) -> Vec<bool> {
        let last = surfaces.len().saturating_sub(1);
        surfaces
            .iter()
            .enumerate()
            .map(|(i, surface)| {
                if self.config.protect_boundaries && (i == 0 || i == last) {
                    return true;
                }
                if self.config.protect_numbers && is_numeric(surface) {
                    return true;
                }
                if self.config.protect_capitalized && is_capitalized(surface) {
                    return true;
                }
                false
            })
            .collect()
    }

    /// Fits the surrogate LM on the whole normalised token stream and writes
    /// each token's surprisal back into its segment.
    fn score_surprisals(&self, segments: &mut [Segment]) {
        let flat_normals: Vec<String> = segments
            .iter()
            .flat_map(|segment| segment.surfaces.iter().map(|s| normalize_token(s)))
            .collect();

        let model = PerplexityModel::fit(
            &flat_normals,
            self.config.ngram_order,
            self.config.add_k,
            self.config.interpolation_lambda,
        );

        let refs: Vec<&str> = flat_normals.iter().map(String::as_str).collect();
        #[allow(clippy::cast_possible_truncation)]
        let surprisals: Vec<f32> = (0..refs.len())
            .map(|i| model.surprisal(&refs[..i], refs[i]) as f32)
            .collect();

        let mut offset = 0;
        for segment in segments.iter_mut() {
            let len = segment.len();
            segment
                .surprisals
                .clone_from_slice(&surprisals[offset..offset + len]);
            offset += len;
        }
    }

    /// Converts the configured [`CompressionTarget`] into an absolute token
    /// target and the requested retention ratio.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn resolve_target(&self, total_tokens: usize) -> (usize, f32) {
        match self.config.target {
            CompressionTarget::Ratio(ratio) => {
                let ratio = ratio.clamp(0.0, 1.0);
                let target = (f64::from(ratio) * total_tokens as f64).round() as usize;
                (target.min(total_tokens), ratio)
            }
            CompressionTarget::TokenBudget(budget) => {
                let target = budget.min(total_tokens);
                let requested = target as f32 / total_tokens as f32;
                (target, requested)
            }
        }
    }

    /// Runs the budget controller over the surviving segments, returning a
    /// keep-count per segment (`0` for segments dropped by Stage A).
    fn stage_b_allocate(
        &self,
        segments: &[Segment],
        densities: &[f32],
        retained: &[bool],
        target: usize,
    ) -> Vec<usize> {
        let controller = BudgetController::new(
            self.config.max_local_drop_ratio,
            self.config.density_emphasis,
        );

        let survivor_idx: Vec<usize> = (0..segments.len()).filter(|&i| retained[i]).collect();
        let lengths: Vec<usize> = survivor_idx.iter().map(|&i| segments[i].len()).collect();
        let survivor_densities: Vec<f32> = survivor_idx.iter().map(|&i| densities[i]).collect();
        let protected: Vec<usize> = survivor_idx
            .iter()
            .map(|&i| segments[i].protected_count())
            .collect();

        let allocated = controller.allocate(&lengths, &survivor_densities, &protected, target);

        let mut keep_counts = vec![0usize; segments.len()];
        for (slot, &seg_i) in survivor_idx.iter().enumerate() {
            keep_counts[seg_i] = allocated.get(slot).copied().unwrap_or(0);
        }
        keep_counts
    }
}

impl Default for PerplexityCompressor {
    fn default() -> Self {
        Self::new(LlmLinguaConfig::default())
    }
}

// ── free helpers ───────────────────────────────────────────────────────────────

/// Builds the final [`CompressionResult`] from the per-segment decisions.
#[allow(clippy::cast_precision_loss)]
fn assemble_result(
    segments: &[Segment],
    densities: &[f32],
    retained: &[bool],
    keep_counts: &[usize],
    total_tokens: usize,
    requested_ratio: f32,
) -> CompressionResult {
    let mut segment_stats = Vec::with_capacity(segments.len());
    let mut kept_texts: Vec<String> = Vec::new();
    let mut compressed_token_count = 0usize;

    for (index, segment) in segments.iter().enumerate() {
        if retained[index] {
            let kept = select_kept_tokens(segment, keep_counts[index]);
            let retained_tokens = kept.iter().filter(|&&k| k).count();
            let text = reconstruct(&segment.surfaces, &kept);
            compressed_token_count += retained_tokens;
            if !text.is_empty() {
                kept_texts.push(text.clone());
            }
            segment_stats.push(SegmentStats {
                index,
                original_tokens: segment.len(),
                retained_tokens,
                density: densities[index],
                retained: true,
                text,
            });
        } else {
            segment_stats.push(SegmentStats {
                index,
                original_tokens: segment.len(),
                retained_tokens: 0,
                density: densities[index],
                retained: false,
                text: String::new(),
            });
        }
    }

    let compressed_text = kept_texts.join(" ");
    let achieved_ratio = if total_tokens == 0 {
        0.0
    } else {
        compressed_token_count as f32 / total_tokens as f32
    };

    CompressionResult {
        compressed_text,
        original_token_count: total_tokens,
        compressed_token_count,
        requested_ratio,
        achieved_ratio,
        segment_stats,
    }
}

/// `true` when `surface` contains an ASCII digit.
fn is_numeric(surface: &str) -> bool {
    surface.chars().any(|c| c.is_ascii_digit())
}

/// `true` when `surface`'s first alphabetic character is uppercase
/// (proper-noun-looking).
fn is_capitalized(surface: &str) -> bool {
    surface
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(char::is_uppercase)
}

/// Coarse Stage A: returns a per-segment "retained" mask.
///
/// Segments are considered in ascending density order; the lowest-density ones
/// are dropped first, but only while (a) more than `min_segments_retained`
/// segments remain and (b) enough tokens would remain to still satisfy
/// `target`. Iteration stops at the first segment whose removal would undershoot
/// the budget, so the dropped set is the lowest-density prefix.
fn stage_a_coarse(
    segments: &[Segment],
    densities: &[f32],
    target: usize,
    min_segments_retained: usize,
) -> Vec<bool> {
    let n = segments.len();
    let mut retained = vec![true; n];
    if n <= min_segments_retained {
        return retained;
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        densities[a]
            .partial_cmp(&densities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut retained_tokens: usize = segments.iter().map(Segment::len).sum();
    let mut retained_count = n;

    for &idx in &order {
        if retained_count <= min_segments_retained {
            break;
        }
        let after = retained_tokens.saturating_sub(segments[idx].len());
        if after >= target {
            retained[idx] = false;
            retained_tokens = after;
            retained_count -= 1;
        } else {
            break;
        }
    }

    retained
}

/// Fine Stage B: selects which tokens of one segment to keep so that exactly
/// `keep` remain — all protected tokens plus the highest-surprisal unprotected
/// tokens. Ties break toward the earlier token for determinism.
fn select_kept_tokens(segment: &Segment, keep: usize) -> Vec<bool> {
    let m = segment.len();
    let mut kept = segment.protected.clone();
    let protected_count = segment.protected_count();
    let mut need = keep
        .saturating_sub(protected_count)
        .min(m - protected_count.min(m));

    let mut candidates: Vec<usize> = (0..m).filter(|&i| !segment.protected[i]).collect();
    candidates.sort_by(|&a, &b| {
        segment.surprisals[b]
            .partial_cmp(&segment.surprisals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    for &i in &candidates {
        if need == 0 {
            break;
        }
        kept[i] = true;
        need -= 1;
    }

    kept
}

/// Rejoins the kept surface tokens (in original order) into a single string.
fn reconstruct(surfaces: &[String], kept: &[bool]) -> String {
    surfaces
        .iter()
        .zip(kept.iter())
        .filter(|&(_, &keep)| keep)
        .map(|(surface, _)| surface.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}
