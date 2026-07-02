//! [`BudgetController`]: the proportional, density-aware token-budget allocator.
//!
//! Given a set of surviving segments and an overall keep-budget, the controller
//! decides **how many tokens each segment keeps**. The point of a controller
//! (rather than a flat per-segment percentage) is that denser — more
//! information-rich — segments should keep proportionally *more* of their own
//! tokens. This is a genuine constrained optimisation, not a single division:
//!
//! 1. Each segment `s` gets a base weight `w_s = len_s * density_s^gamma`, so
//!    its unconstrained keep share is `budget * w_s / sum(w)`. Because the
//!    weight scales with `len_s`, a segment's *retention fraction*
//!    `keep_s / len_s` scales with `density_s^gamma` — denser segments retain a
//!    higher fraction (`gamma` is [`density_emphasis`]).
//! 2. Each segment has a hard floor `L_s` (it may never drop below its
//!    protected tokens, may never delete more than the configured
//!    `max_local_drop_ratio`, and always keeps at least one token) and a
//!    ceiling `U_s = len_s`.
//! 3. A **water-filling** loop projects the proportional shares onto those
//!    `[L_s, U_s]` boxes: any share that violates a bound is fixed at that
//!    bound and its mass is removed from the pool, then the remaining budget is
//!    re-shared over the still-free segments — repeating until nothing new is
//!    clamped. This is the standard iterative solution to
//!    "maximise proportionality subject to per-item min/max caps".
//! 4. The real-valued solution is rounded to integers with the largest-
//!    remainder method, staying within every `[L_s, U_s]` box.
//!
//! When the floors collectively exceed the budget (heavy protection under a
//! very tight target) the achieved total simply lands at `sum(L_s)` — higher
//! than requested — which the compressor faithfully reports.
//!
//! [`density_emphasis`]: super::types::LlmLinguaConfig::density_emphasis

/// Smallest weight a segment can contribute, so a zero-density segment still
/// participates in allocation instead of vanishing.
const MIN_WEIGHT: f64 = 1e-9;

/// Proportional, density-weighted token-budget allocator with per-segment
/// min/max caps, solved by water-filling. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct BudgetController {
    /// Upper bound on the fraction of any one segment that may be deleted.
    max_local_drop_ratio: f32,
    /// Exponent `gamma` applied to density in the segment weight.
    density_emphasis: f32,
}

impl BudgetController {
    /// Creates a controller.
    ///
    /// `max_local_drop_ratio` (clamped to `[0, 1]`) caps how much of a single
    /// segment may be deleted; `density_emphasis` (clamped to `>= 0`) is the
    /// exponent applied to per-segment density when weighting the split.
    #[must_use]
    pub fn new(max_local_drop_ratio: f32, density_emphasis: f32) -> Self {
        Self {
            max_local_drop_ratio: max_local_drop_ratio.clamp(0.0, 1.0),
            density_emphasis: density_emphasis.max(0.0),
        }
    }

    /// Per-segment floor: keep at least the protected tokens, at least the
    /// `(1 - max_local_drop_ratio)` fraction, and always at least one token,
    /// never exceeding the segment length.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn lower_bound(&self, len: usize, protected: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let keep_fraction = 1.0_f32 - self.max_local_drop_ratio;
        let cap_floor = (keep_fraction * len as f32).ceil() as usize;
        protected.max(cap_floor).max(1).min(len)
    }

    /// Allocates a per-segment keep count summing to `budget` (as closely as
    /// the floors and ceilings permit).
    ///
    /// `lengths`, `densities`, and `protected_counts` are parallel arrays, one
    /// entry per surviving segment; the returned vector has the same length as
    /// `lengths`. `densities` / `protected_counts` entries missing relative to
    /// `lengths` default to `0`. The result respects, for every segment `s`,
    /// `lower_bound(len_s, protected_s) <= keep_s <= len_s`.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn allocate(
        &self,
        lengths: &[usize],
        densities: &[f32],
        protected_counts: &[usize],
        budget: usize,
    ) -> Vec<usize> {
        let n = lengths.len();
        if n == 0 {
            return Vec::new();
        }

        let lower: Vec<usize> = (0..n)
            .map(|i| self.lower_bound(lengths[i], protected_counts.get(i).copied().unwrap_or(0)))
            .collect();
        let upper: Vec<usize> = lengths.to_vec();
        let weight: Vec<f64> = (0..n)
            .map(|i| {
                let density = f64::from(densities.get(i).copied().unwrap_or(0.0)).max(0.0);
                let emphasised = density.powf(f64::from(self.density_emphasis));
                (lengths[i] as f64 * emphasised).max(MIN_WEIGHT)
            })
            .collect();

        let real_keep = Self::water_fill(&lower, &upper, &weight, budget);
        Self::round_allocation(&real_keep, &lower, &upper)
    }

    /// Projects the proportional split onto the `[lower, upper]` boxes via
    /// iterative water-filling, returning real-valued keep counts.
    #[allow(clippy::cast_precision_loss)]
    fn water_fill(lower: &[usize], upper: &[usize], weight: &[f64], budget: usize) -> Vec<f64> {
        let n = lower.len();
        let mut keep = vec![0.0_f64; n];
        let mut fixed = vec![false; n];
        let mut remaining = budget as f64;

        // At most `n` variables can become newly fixed, plus a final free pass.
        for _ in 0..=n {
            let active: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
            if active.is_empty() {
                break;
            }
            let total_weight: f64 = active.iter().map(|&i| weight[i]).sum();

            let mut newly_fixed = false;
            if total_weight <= 0.0 {
                // Degenerate weights: fall back to an even split of the pool.
                let share = remaining / active.len() as f64;
                for &i in &active {
                    let bounded = share.clamp(lower[i] as f64, upper[i] as f64);
                    keep[i] = bounded;
                    fixed[i] = true;
                }
                break;
            }

            for &i in &active {
                let share = remaining * weight[i] / total_weight;
                if share < lower[i] as f64 {
                    keep[i] = lower[i] as f64;
                    fixed[i] = true;
                    remaining -= keep[i];
                    newly_fixed = true;
                } else if share > upper[i] as f64 {
                    keep[i] = upper[i] as f64;
                    fixed[i] = true;
                    remaining -= keep[i];
                    newly_fixed = true;
                }
            }

            if !newly_fixed {
                // All remaining shares are within bounds: assign them directly.
                let total_weight: f64 = active.iter().map(|&i| weight[i]).sum();
                for &i in &active {
                    keep[i] = remaining * weight[i] / total_weight;
                }
                break;
            }
        }

        keep
    }

    /// Rounds real keep counts to integers via the largest-remainder method,
    /// preserving the (feasible) total and every `[lower, upper]` box.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn round_allocation(real_keep: &[f64], lower: &[usize], upper: &[usize]) -> Vec<usize> {
        let n = real_keep.len();
        let target_total: usize = real_keep.iter().sum::<f64>().round().max(0.0) as usize;

        let mut floors: Vec<usize> = (0..n)
            .map(|i| (real_keep[i].floor().max(0.0) as usize).clamp(lower[i], upper[i]))
            .collect();
        let assigned: usize = floors.iter().sum();

        if assigned < target_total {
            // Distribute the deficit to the largest fractional parts that still
            // have headroom below their ceiling.
            let mut order: Vec<usize> = (0..n).filter(|&i| floors[i] < upper[i]).collect();
            order.sort_by(|&a, &b| {
                let fa = real_keep[a] - real_keep[a].floor();
                let fb = real_keep[b] - real_keep[b].floor();
                fb.partial_cmp(&fa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut deficit = target_total - assigned;
            let mut cursor = 0;
            while deficit > 0 && !order.is_empty() {
                let i = order[cursor % order.len()];
                if floors[i] < upper[i] {
                    floors[i] += 1;
                    deficit -= 1;
                }
                cursor += 1;
                // Guard against a pathological loop when no headroom remains.
                if cursor > order.len() * (upper.iter().sum::<usize>() + 1) {
                    break;
                }
            }
        } else if assigned > target_total {
            // Trim the surplus from the smallest fractional parts that still
            // have room above their floor.
            let mut order: Vec<usize> = (0..n).filter(|&i| floors[i] > lower[i]).collect();
            order.sort_by(|&a, &b| {
                let fa = real_keep[a] - real_keep[a].floor();
                let fb = real_keep[b] - real_keep[b].floor();
                fa.partial_cmp(&fb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut surplus = assigned - target_total;
            let mut cursor = 0;
            while surplus > 0 && !order.is_empty() {
                let i = order[cursor % order.len()];
                if floors[i] > lower[i] {
                    floors[i] -= 1;
                    surplus -= 1;
                }
                cursor += 1;
                if cursor > order.len() * (upper.iter().sum::<usize>() + 1) {
                    break;
                }
            }
        }

        floors
    }
}
