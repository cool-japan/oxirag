//! Core `EigenScore` detector logic: feature clipping, centering, the `K x K`
//! Gram matrix, regularization, and the mean-log-eigenvalue score.

use std::cmp::Ordering;

use super::jacobi::jacobi_eigenvalues;
use super::types::{EigenScoreConfig, EigenScoreError, EigenScoreResult};

// ── FNV-1a character-trigram pseudo-embeddings ────────────────────────────────

/// `FNV-1a` 64-bit offset basis.
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
/// `FNV-1a` 64-bit prime.
const FNV_PRIME: u64 = 1_099_511_628_211;

/// Length (in `char`s) of the n-grams hashed by [`embed_text`].
const NGRAM_N: usize = 3;

/// Deterministic `FNV-1a` 64-bit hash of a byte slice.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Derive a deterministic pseudo-embedding for `text`.
///
/// The lowercased, whitespace-normalized text is split into overlapping
/// character n-grams (`n = 3`; shorter texts fall back to a single gram over
/// the whole string). Each n-gram is `FNV-1a`-hashed into one of `dim`
/// buckets, incrementing a per-bucket count; the resulting histogram is then
/// L2-normalized. Two responses with similar character-level content hash
/// into similar histograms, giving a cheap, dependency-free stand-in for a
/// real sentence embedding, in the same spirit as the pseudo-embeddings used
/// by the `semantic_router` module.
///
/// Deterministic: identical input always produces an identical output vector.
fn embed_text(text: &str, dim: usize) -> Vec<f32> {
    if dim == 0 {
        return Vec::new();
    }

    let normalized: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let chars: Vec<char> = normalized.chars().collect();

    let mut buckets = vec![0.0_f64; dim];
    #[allow(clippy::cast_possible_truncation)]
    let bucket_of = |gram: &str| -> usize { (fnv1a(gram.as_bytes()) % dim as u64) as usize };

    if chars.len() < NGRAM_N {
        let gram: String = chars.iter().collect();
        buckets[bucket_of(&gram)] += 1.0;
    } else {
        for window in chars.windows(NGRAM_N) {
            let gram: String = window.iter().collect();
            buckets[bucket_of(&gram)] += 1.0;
        }
    }

    let norm: f64 = buckets.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        return vec![0.0; dim];
    }
    #[allow(clippy::cast_possible_truncation)]
    buckets.iter().map(|&x| (x / norm) as f32).collect()
}

// ── numerical helpers ──────────────────────────────────────────────────────────

/// Linear-interpolation empirical quantile of a pre-sorted slice (matches the
/// common "linear" convention, e.g. `numpy.quantile`'s default).
///
/// `sorted` must be non-empty and `q` is expected in `[0.0, 1.0]`; the
/// resulting fractional index is clamped into range so out-of-band `q` (or
/// floating-point rounding at the boundary) never produces an out-of-bounds
/// index.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let last = n - 1;
    let pos = (q * last as f64).clamp(0.0, last as f64);
    let lower = pos.floor() as usize;
    let upper = pos.ceil().min(last as f64) as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = pos - pos.floor();
        (1.0 - frac).mul_add(sorted[lower], frac * sorted[upper])
    }
}

/// Clip every coordinate of every embedding to the empirical `[p, 1-p]`
/// quantile band computed across the `K` samples for that coordinate.
///
/// This is INSIDE's test-time feature-clipping trick: it prevents a handful
/// of outlier feature dimensions (e.g. from an unlucky hash collision) from
/// dominating the Gram matrix and hence the eigenvalue spectrum.
fn clip_features(embeddings: &[Vec<f32>], p: f32, dim: usize) -> Vec<Vec<f64>> {
    let k = embeddings.len();
    let p64 = f64::from(p);

    let mut bounds: Vec<(f64, f64)> = Vec::with_capacity(dim);
    #[allow(clippy::needless_range_loop)]
    for d in 0..dim {
        let mut column: Vec<f64> = (0..k).map(|i| f64::from(embeddings[i][d])).collect();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let lower = quantile(&column, p64);
        let upper = quantile(&column, 1.0 - p64);
        bounds.push((lower.min(upper), lower.max(upper)));
    }

    embeddings
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(d, &v)| {
                    let (lower, upper) = bounds[d];
                    f64::from(v).clamp(lower, upper)
                })
                .collect()
        })
        .collect()
}

/// Center `data` (a `K x dim` matrix, one row per sample) across its `K` rows:
/// subtract the per-column mean from every entry.
fn center_samples(data: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let k = data.len();
    let mut means = vec![0.0_f64; dim];
    for row in data {
        for (m, &v) in means.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let k_f64 = k as f64;
    for m in &mut means {
        *m /= k_f64;
    }

    data.iter()
        .map(|row| row.iter().zip(means.iter()).map(|(&v, &m)| v - m).collect())
        .collect()
}

/// Compute the `K x K` Gram matrix `G_ij = <centered_i, centered_j>` of the
/// row-centered sample matrix.
fn gram_matrix(centered: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = centered.len();
    let mut gram = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in i..k {
            let dot: f64 = centered[i]
                .iter()
                .zip(centered[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            gram[i][j] = dot;
            gram[j][i] = dot;
        }
    }
    gram
}

// ── EigenScoreDetector ─────────────────────────────────────────────────────────

/// `INSIDE`/`EigenScore` hallucination detector (Chen et al., ICLR 2024).
///
/// Measures self-consistency across `K` sampled responses via the
/// differential entropy of their embedding covariance: low diversity
/// (consistent responses) yields a low score, high diversity (likely
/// hallucination) yields a high score. See the [module docs](super) for the
/// full derivation.
#[derive(Debug, Clone, Default)]
pub struct EigenScoreDetector {
    /// Configuration controlling regularization, feature clipping, the Jacobi
    /// solver's budget, the hallucination threshold, and the text-embedding
    /// dimension.
    pub config: EigenScoreConfig,
}

impl EigenScoreDetector {
    /// Construct a detector with the given configuration.
    #[must_use]
    pub fn new(config: EigenScoreConfig) -> Self {
        Self { config }
    }

    /// Score `K` pre-computed response embeddings.
    ///
    /// `embeddings` must contain at least two rows, all of the same
    /// (non-zero) length. The returned [`EigenScoreResult::eigenvalues`] has
    /// length `K` and is ascending.
    ///
    /// # Errors
    ///
    /// Returns [`EigenScoreError::InsufficientSamples`] when fewer than two
    /// embeddings are supplied; [`EigenScoreError::DimensionMismatch`] when
    /// embeddings have inconsistent lengths; [`EigenScoreError::NonFinite`]
    /// when any coordinate is `NaN`/infinite; and
    /// [`EigenScoreError::InvalidConfig`] when `regularization`,
    /// `clip_percentile`, `jacobi_max_sweeps`, or `jacobi_tol` is out of
    /// range, when the embedding dimension is zero, or when the regularized
    /// covariance nonetheless yields a non-positive eigenvalue (numerically
    /// degenerate; increasing `regularization` resolves this).
    pub fn score_embeddings(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Result<EigenScoreResult, EigenScoreError> {
        let k = embeddings.len();
        if k < 2 {
            return Err(EigenScoreError::InsufficientSamples { got: k, need: 2 });
        }

        let dim = embeddings[0].len();
        if dim == 0 {
            return Err(EigenScoreError::InvalidConfig(
                "embedding dimension must be greater than zero".to_string(),
            ));
        }
        for embedding in embeddings {
            if embedding.len() != dim {
                return Err(EigenScoreError::DimensionMismatch {
                    expected: dim,
                    got: embedding.len(),
                });
            }
            for &value in embedding {
                if !value.is_finite() {
                    return Err(EigenScoreError::NonFinite);
                }
            }
        }

        if !self.config.regularization.is_finite() || self.config.regularization < 0.0 {
            return Err(EigenScoreError::InvalidConfig(format!(
                "regularization must be a non-negative finite value, got {}",
                self.config.regularization
            )));
        }
        if let Some(p) = self.config.clip_percentile
            && !(0.0..0.5).contains(&p)
        {
            return Err(EigenScoreError::InvalidConfig(format!(
                "clip_percentile must be in [0.0, 0.5), got {p}"
            )));
        }
        if self.config.jacobi_max_sweeps == 0 {
            return Err(EigenScoreError::InvalidConfig(
                "jacobi_max_sweeps must be at least 1".to_string(),
            ));
        }
        if !self.config.jacobi_tol.is_finite() || self.config.jacobi_tol < 0.0 {
            return Err(EigenScoreError::InvalidConfig(format!(
                "jacobi_tol must be a non-negative finite value, got {}",
                self.config.jacobi_tol
            )));
        }

        // Step 0 (optional): clip outlier feature dimensions.
        let processed: Vec<Vec<f64>> = if let Some(p) = self.config.clip_percentile {
            clip_features(embeddings, p, dim)
        } else {
            embeddings
                .iter()
                .map(|e| e.iter().map(|&v| f64::from(v)).collect())
                .collect()
        };

        // Step 1: center across the K samples.
        let centered = center_samples(&processed, dim);

        // Step 2: K x K Gram matrix of the centered embeddings.
        let gram = gram_matrix(&centered);

        // Step 3: regularize, Sigma = (1/K) * G + alpha * I_K.
        let alpha = f64::from(self.config.regularization);
        #[allow(clippy::cast_precision_loss)]
        let k_f64 = k as f64;
        let mut sigma = vec![vec![0.0_f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                sigma[i][j] = gram[i][j] / k_f64;
            }
            sigma[i][i] += alpha;
        }

        // Step 4: eigen-decompose Sigma via cyclic Jacobi.
        let eigenvalues = jacobi_eigenvalues(
            &sigma,
            self.config.jacobi_max_sweeps,
            self.config.jacobi_tol,
        )?;

        // Step 5: EigenScore = (1/K) * sum_i ln(lambda_i).
        let mut log_sum = 0.0_f64;
        for &lambda in &eigenvalues {
            if lambda <= 0.0 {
                return Err(EigenScoreError::InvalidConfig(format!(
                    "eigen-decomposition produced a non-positive eigenvalue ({lambda}); \
                     increase regularization"
                )));
            }
            log_sum += lambda.ln();
        }
        let score64 = log_sum / k_f64;
        #[allow(clippy::cast_possible_truncation)]
        let score = score64 as f32;

        let is_hallucination = score >= self.config.hallucination_threshold;

        Ok(EigenScoreResult {
            score,
            #[allow(clippy::cast_possible_truncation)]
            eigenvalues: eigenvalues.iter().map(|&v| v as f32).collect(),
            is_hallucination,
        })
    }

    /// Score `K` sampled response strings by first deriving deterministic
    /// pseudo-embeddings (deterministic FNV-1a character-trigram hashing) at
    /// [`EigenScoreConfig::embed_dim`], then delegating to
    /// [`score_embeddings`](Self::score_embeddings).
    ///
    /// # Errors
    ///
    /// Returns [`EigenScoreError::InsufficientSamples`] when fewer than two
    /// responses are supplied, and otherwise propagates any error from
    /// [`score_embeddings`](Self::score_embeddings) (in particular,
    /// [`EigenScoreError::InvalidConfig`] when
    /// [`EigenScoreConfig::embed_dim`] is zero).
    pub fn score_responses(&self, responses: &[&str]) -> Result<EigenScoreResult, EigenScoreError> {
        let k = responses.len();
        if k < 2 {
            return Err(EigenScoreError::InsufficientSamples { got: k, need: 2 });
        }
        let dim = self.config.embed_dim;
        let embeddings: Vec<Vec<f32>> = responses.iter().map(|r| embed_text(r, dim)).collect();
        self.score_embeddings(&embeddings)
    }
}
