//! Cardinality estimation for metadata predicates, and the cost-based rule
//! that turns an estimate into an execution plan.
//!
//! This is the piece that lets the index *decide*. Without it, a filtered
//! search has to guess which strategy to run, and every guess is wrong
//! somewhere: post-filtering collapses under a selective predicate,
//! pre-filtering degenerates into a full scan under a permissive one, and the
//! ACORN traversal pays a two-hop tax that neither of the other two owes. The
//! estimator's job is to tell the planner, *before any vector is touched*,
//! roughly how many records the predicate admits.
//!
//! # Statistics maintained
//!
//! Per attribute, [`FilteredAttributeStats`] keeps:
//!
//! - **Exact value counts** (`AttrValue -> count`) for the first
//!   [`max_tracked_values`](super::FilteredSearchConfig::max_tracked_values)
//!   distinct values seen. Counts for *tracked* values remain exact forever
//!   (they keep incrementing after the cap is hit); only *untracked* values
//!   need a heuristic, and the estimate reports itself as inexact when one is
//!   used.
//! - An **equi-width histogram** over the numeric (`Int`/`Float`) values, used
//!   for range constraints.
//!
//! # Compositional estimation
//!
//! Leaf selectivities come from the statistics above. Composite ones are
//! folded together with the **independence assumption**, the standard
//! (and standardly wrong) assumption of every textbook cardinality estimator:
//!
//! ```text
//! s(And(p1..pn)) = Π s(pi)
//! s(Or(p1..pn))  = 1 - Π (1 - s(pi))
//! s(Not(p))      = 1 - s(p)
//! ```
//!
//! For *correlated* attributes these are arbitrarily wrong — `And(country=JP,
//! language=ja)` will be badly underestimated, because the two attributes are
//! anything but independent. The estimate therefore reports
//! [`independence_assumed`](super::SelectivityEstimate::independence_assumed)
//! whenever it folded more than one sub-predicate, and callers who need a
//! hard guarantee should read [`FilterStrategy::PreFilter`],
//! which is exact regardless of what the estimator believed.
//!
//! Crucially, **a mis-estimate costs time, never correctness**: the worst a bad
//! estimate can do is route the query to a strategy that is slower than
//! necessary, or (in the `PostFilter` case) one that returns fewer than `k`
//! hits. Every hit any strategy returns has been verified against the real
//! predicate.

use std::collections::HashMap;

use super::predicate::FilterPredicate;
use super::types::{
    AttrValue, FilterBound, FilterStrategy, FilteredMetadata, FilteredSearchConfig,
    SelectivityEstimate, bump_count,
};

/// Relative tolerance used when deciding whether a histogram bucket is
/// *entirely* inside or outside a query range (in which case no
/// within-bucket-uniformity assumption is needed and the estimate stays exact).
const BUCKET_COVERAGE_EPSILON: f64 = 1e-12;

// ── FilteredNumericHistogram ─────────────────────────────────────────────────

/// An equi-width histogram over the numeric values observed for one attribute.
///
/// # Why the raw values are retained
///
/// Equi-width bucketing needs a domain, and the domain is not known until the
/// data has all arrived — but vectors arrive incrementally. This histogram
/// therefore retains the raw `f64` observations and **re-bins exactly**
/// whenever a new value falls outside the current `[min, max]` domain. For
/// i.i.d. data the domain expands `O(log n)` times, so `observe` is amortized
/// `O(1)`, and the memory cost is 8 bytes per observation per numeric
/// attribute — negligible beside the vectors themselves, which run to hundreds
/// of bytes each.
///
/// The alternatives were considered and rejected: a *fixed* domain with
/// overflow buckets cannot answer ranges in the overflow region at all, and
/// *dyadic domain doubling* (which merges adjacent bucket pairs — an exact
/// operation for equi-width bins, and therefore genuinely `O(buckets)` memory)
/// leaves the domain loosely bounding the data, wasting up to half the buckets
/// and coarsening every estimate. Since the surrounding index already retains
/// these values in its sorted numeric arrays for range resolution, the
/// tight-domain variant costs nothing extra in practice and estimates strictly
/// better.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredNumericHistogram {
    bucket_count: usize,
    minimum: f64,
    maximum: f64,
    buckets: Vec<u64>,
    values: Vec<f64>,
    /// Whether every observation so far has been an integer. See
    /// [`FilteredNumericHistogram::estimate_range`] for why this changes the
    /// interpolation rule — and why ignoring it makes an integer attribute's
    /// range estimates systematically wrong.
    all_integral: bool,
}

impl FilteredNumericHistogram {
    /// Create an empty histogram with `bucket_count` equi-width buckets
    /// (clamped to at least one).
    #[must_use]
    pub fn new(bucket_count: usize) -> Self {
        let bucket_count = bucket_count.max(1);
        Self {
            bucket_count,
            minimum: f64::INFINITY,
            maximum: f64::NEG_INFINITY,
            buckets: vec![0; bucket_count],
            values: Vec::new(),
            all_integral: true,
        }
    }

    /// Record one numeric observation.
    ///
    /// Non-finite values are ignored: a NaN satisfies no range constraint, and
    /// an infinity would collapse the domain onto a single bucket, destroying
    /// the histogram's resolution for every other value.
    pub fn observe(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if value.fract() != 0.0 {
            self.all_integral = false;
        }
        self.values.push(value);
        if value < self.minimum || value > self.maximum {
            self.minimum = self.minimum.min(value);
            self.maximum = self.maximum.max(value);
            self.rebin();
        } else {
            let bucket = self.bucket_of(value);
            self.buckets[bucket] += 1;
        }
    }

    /// Whether every observation has been an integer, in which case range
    /// estimates count *integer points* rather than continuous length.
    #[must_use]
    pub fn is_integral(&self) -> bool {
        self.all_integral && !self.values.is_empty()
    }

    /// Rebuild every bucket from the retained values against the current
    /// domain. Called only when the domain grows.
    fn rebin(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = 0;
        }
        // `self.values` is borrowed immutably while `self.buckets` is borrowed
        // mutably, so the bucket index is computed inline from the domain
        // rather than through `&self`.
        let width = self.bucket_width();
        let minimum = self.minimum;
        let last = self.bucket_count - 1;
        for value in &self.values {
            let bucket = if width > 0.0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let raw = ((value - minimum) / width) as usize;
                raw.min(last)
            } else {
                0
            };
            self.buckets[bucket] += 1;
        }
    }

    /// Width of one bucket, or `0.0` for a degenerate (single-value or empty)
    /// domain.
    fn bucket_width(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let width = (self.maximum - self.minimum) / self.bucket_count as f64;
        if width.is_finite() && width > 0.0 {
            width
        } else {
            0.0
        }
    }

    /// The bucket a value falls into, clamped into range.
    fn bucket_of(&self, value: f64) -> usize {
        let width = self.bucket_width();
        if width <= 0.0 {
            return 0;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let raw = ((value - self.minimum) / width) as usize;
        raw.min(self.bucket_count - 1)
    }

    /// Number of observations recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.values.len() as u64
    }

    /// The smallest value observed, or `None` when empty.
    #[must_use]
    pub fn minimum(&self) -> Option<f64> {
        (!self.values.is_empty()).then_some(self.minimum)
    }

    /// The largest value observed, or `None` when empty.
    #[must_use]
    pub fn maximum(&self) -> Option<f64> {
        (!self.values.is_empty()).then_some(self.maximum)
    }

    /// The bucket counts, low to high.
    #[must_use]
    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }

    /// Estimate how many observations fall inside `[lower, upper]`.
    ///
    /// Returns `(estimated_count, exact)`. The estimate is **exact** — not
    /// merely accurate — whenever no within-bucket-uniformity assumption was
    /// needed, which happens in more cases than one might expect:
    ///
    /// - the query range misses the domain entirely (the true count is
    ///   provably `0`);
    /// - the query range covers the domain entirely (the true count is
    ///   provably `total`);
    /// - the domain is degenerate (every observation has the same value), so
    ///   the bounds either accept all of them or none;
    /// - every bucket holding at least one observation is *fully* inside or
    ///   *fully* outside the query range.
    ///
    /// Only a bucket that is *partially* covered *and* non-empty forces the
    /// uniformity assumption, and only then is `exact` reported as `false`.
    ///
    /// # Discrete attributes get a discrete interpolation
    ///
    /// A bucket that is only partially covered has to be apportioned somehow,
    /// and the naive rule — apportion by *continuous length* — is badly wrong
    /// for integer attributes, which is most of them in practice (years,
    /// versions, counts, tenant ids, priority levels).
    ///
    /// Consider `bucket` uniform over `0..=99`, thirty-two equi-width buckets,
    /// and the predicate `bucket < 1`. The first bucket spans `[0, 3.09)` and
    /// therefore holds the integer values `{0, 1, 2, 3}` — four values, eighty
    /// records out of two thousand. Apportioning by length gives
    /// `80 * (1 / 3.09) = 25.9` records, a 30% over-estimate of the true 20 —
    /// which, at a `prefilter_threshold` of exactly 1%, is more than enough to
    /// send the query to the wrong execution plan. Apportioning by *integer
    /// points* gives `80 * (1 / 4) = 20`. Exactly right.
    ///
    /// So when every observation has been an integer, this counts the integer
    /// points a bucket contains and the integer points the query range covers,
    /// and apportions by their ratio. This is what a real query optimizer does
    /// for a discrete column, and it costs nothing: the discreteness is *real
    /// information* about the data, and throwing it away to reuse a continuous
    /// formula is a straightforwardly worse estimate.
    ///
    /// Bound strictness (inclusive versus exclusive) is honored exactly in the
    /// integer path, where it genuinely changes which values are admitted; in
    /// the continuous path it is ignored inside the interpolation, where it
    /// differs by a measure-zero set that no equi-width histogram can resolve.
    #[must_use]
    pub fn estimate_range(&self, lower: FilterBound, upper: FilterBound) -> (f64, bool) {
        let total = self.total();
        if total == 0 {
            return (0.0, true);
        }
        #[allow(clippy::cast_precision_loss)]
        let total_f64 = total as f64;

        // No observation can satisfy a lower bound that even the largest
        // observation fails, nor an upper bound that even the smallest fails.
        if !lower.accepts_as_lower(self.maximum) || !upper.accepts_as_upper(self.minimum) {
            return (0.0, true);
        }
        // Conversely, if the bounds accept both extremes they accept every
        // observation in between.
        if lower.accepts_as_lower(self.minimum) && upper.accepts_as_upper(self.maximum) {
            return (total_f64, true);
        }

        let width = self.bucket_width();
        if width <= 0.0 {
            // Degenerate domain: every observation equals `self.minimum`, and
            // the two checks above already decided the outcome unless the
            // bounds reject that single value — which, having reached here, they
            // do.
            return (0.0, true);
        }

        let integral = self.is_integral();
        let query_low = lower.as_lower_f64();
        let query_high = upper.as_upper_f64();
        // The integer points the query admits, as a closed interval on the
        // (extended) real line.
        let query_low_point = lowest_admissible_integer(lower);
        let query_high_point = highest_admissible_integer(upper);

        let mut estimate = 0.0_f64;
        let mut exact = true;
        let last = self.bucket_count - 1;

        for (index, &count) in self.buckets.iter().enumerate() {
            if count == 0 {
                // An empty bucket contributes nothing and forces no assumption,
                // however it overlaps the query range.
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let bucket_low = self.minimum + index as f64 * width;
            // Every bucket is half-open `[low, high)` except the last, which is
            // closed on `maximum` so that the largest observation has a home.
            let bucket_high = if index == last {
                self.maximum
            } else {
                bucket_low + width
            };

            let fraction = if integral {
                integer_coverage_fraction(
                    bucket_low,
                    bucket_high,
                    index == last,
                    query_low_point,
                    query_high_point,
                )
            } else {
                None
            }
            .unwrap_or_else(|| {
                // Continuous attribute (or a degenerate bucket holding no
                // integer point at all, which floating-point drift can just
                // barely produce): apportion by length.
                let overlap_low = bucket_low.max(query_low);
                let overlap_high = bucket_high.min(query_high);
                let overlap = (overlap_high - overlap_low).max(0.0);
                (overlap / width).clamp(0.0, 1.0)
            });

            if fraction > BUCKET_COVERAGE_EPSILON && fraction < 1.0 - BUCKET_COVERAGE_EPSILON {
                // Partially covered and non-empty: this is the one and only
                // place a uniformity assumption enters.
                exact = false;
            }
            #[allow(clippy::cast_precision_loss)]
            let bucket_total = count as f64;
            estimate += bucket_total * fraction;
        }

        (estimate.clamp(0.0, total_f64), exact)
    }
}

/// The smallest integer satisfying `bound` when it is used as a lower bound, or
/// `-inf` when unbounded.
fn lowest_admissible_integer(bound: FilterBound) -> f64 {
    match bound {
        FilterBound::Unbounded => f64::NEG_INFINITY,
        // Integers `n >= v`.
        FilterBound::Inclusive(value) => value.ceil(),
        // Integers `n > v`. `(v + 1).floor()` is correct for both integral and
        // fractional `v`: `2 -> 3`, `2.5 -> 3`, `-2.5 -> -2`.
        FilterBound::Exclusive(value) => (value + 1.0).floor(),
    }
}

/// The largest integer satisfying `bound` when it is used as an upper bound, or
/// `+inf` when unbounded.
fn highest_admissible_integer(bound: FilterBound) -> f64 {
    match bound {
        FilterBound::Unbounded => f64::INFINITY,
        // Integers `n <= v`.
        FilterBound::Inclusive(value) => value.floor(),
        // Integers `n < v`. `(v - 1).ceil()`: `3 -> 2`, `2.5 -> 2`, `-2.5 -> -3`.
        FilterBound::Exclusive(value) => (value - 1.0).ceil(),
    }
}

/// The fraction of a bucket's *integer points* that the query range covers, or
/// `None` when the bucket contains no integer point (which cannot happen for a
/// non-empty bucket of an all-integer attribute, but floating-point drift at a
/// bucket boundary is not worth betting correctness on).
fn integer_coverage_fraction(
    bucket_low: f64,
    bucket_high: f64,
    bucket_closed_above: bool,
    query_low_point: f64,
    query_high_point: f64,
) -> Option<f64> {
    let first = bucket_low.ceil();
    let last = if bucket_closed_above {
        bucket_high.floor()
    } else {
        // Half-open `[low, high)`: the largest integer strictly below `high`.
        (bucket_high - 1.0).ceil()
    };
    let points = last - first + 1.0;
    if !(points.is_finite() && points >= 1.0) {
        return None;
    }

    let covered_first = first.max(query_low_point);
    let covered_last = last.min(query_high_point);
    let covered = (covered_last - covered_first + 1.0).max(0.0);
    Some((covered / points).clamp(0.0, 1.0))
}

// ── FilteredAttributeStats ───────────────────────────────────────────────────

/// Per-attribute statistics: exact value counts up to a cardinality cap, plus
/// an equi-width histogram over the numeric values.
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredAttributeStats {
    present_count: u64,
    value_counts: HashMap<AttrValue, u64>,
    tracked_count: u64,
    untracked_count: u64,
    counts_saturated: bool,
    max_tracked_values: usize,
    histogram: FilteredNumericHistogram,
}

impl FilteredAttributeStats {
    /// Create empty statistics for one attribute.
    #[must_use]
    pub fn new(histogram_buckets: usize, max_tracked_values: usize) -> Self {
        Self {
            present_count: 0,
            value_counts: HashMap::new(),
            tracked_count: 0,
            untracked_count: 0,
            counts_saturated: false,
            max_tracked_values: max_tracked_values.max(1),
            histogram: FilteredNumericHistogram::new(histogram_buckets),
        }
    }

    /// Record one observation of this attribute.
    ///
    /// Distinct values are tracked exactly until `max_tracked_values` of them
    /// have been seen. After that, *already-tracked* values keep incrementing
    /// (so their counts stay exact — this is the classic "most common values"
    /// list), and observations of new values are only tallied into
    /// `untracked_count`, which the estimator uses to build a heuristic for
    /// them.
    pub fn observe(&mut self, value: &AttrValue) {
        self.present_count += 1;

        if self.value_counts.contains_key(value) {
            if let Some(count) = self.value_counts.get_mut(value) {
                *count += 1;
            }
            self.tracked_count += 1;
        } else if self.value_counts.len() < self.max_tracked_values {
            bump_count(&mut self.value_counts, value.clone());
            self.tracked_count += 1;
        } else {
            self.counts_saturated = true;
            self.untracked_count += 1;
        }

        if let Some(numeric) = value.as_f64() {
            self.histogram.observe(numeric);
        }
    }

    /// How many records carry this attribute at all.
    #[must_use]
    pub fn present_count(&self) -> u64 {
        self.present_count
    }

    /// Whether the distinct-value cap has been exceeded, meaning equality
    /// estimates for *untracked* values fall back to a heuristic.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        self.counts_saturated
    }

    /// The number of distinct values tracked exactly.
    #[must_use]
    pub fn tracked_distinct(&self) -> usize {
        self.value_counts.len()
    }

    /// The numeric histogram over this attribute's `Int`/`Float` values.
    #[must_use]
    pub fn histogram(&self) -> &FilteredNumericHistogram {
        &self.histogram
    }

    /// Estimate how many records satisfy `attr == value`.
    ///
    /// Returns `(estimated_count, exact)`. It is exact whenever the value is
    /// tracked (its count is a true count), and also whenever the counts are
    /// *unsaturated* and the value is absent (the true count is provably `0`).
    ///
    /// The one inexact case is an untracked value on a saturated attribute.
    /// There, the estimate assumes the untracked values share the average
    /// multiplicity of the tracked ones — `tracked_count / tracked_distinct` —
    /// capped by the total number of untracked observations. This is the
    /// standard `1 / NDV` heuristic with the number of distinct values
    /// necessarily underestimated, so it *over*-estimates; over-estimating
    /// selectivity is the safe direction, because it biases the planner away
    /// from `PostFilter` (whose failure mode is silent, missing results) and
    /// toward `InFilter`/`PreFilter` (whose failure mode is merely spending
    /// more time than strictly necessary).
    #[must_use]
    pub fn estimate_eq(&self, value: &AttrValue) -> (f64, bool) {
        if let Some(&count) = self.value_counts.get(value) {
            #[allow(clippy::cast_precision_loss)]
            return (count as f64, true);
        }
        if !self.counts_saturated {
            return (0.0, true);
        }
        if self.value_counts.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            return (self.untracked_count as f64, false);
        }
        #[allow(clippy::cast_precision_loss)]
        let average_multiplicity = self.tracked_count as f64 / self.value_counts.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let untracked = self.untracked_count as f64;
        (average_multiplicity.min(untracked), false)
    }

    /// Estimate how many records satisfy `attr != value` *and carry `attr`*.
    ///
    /// See the [predicate module](super::predicate) on why this is not the
    /// complement of `estimate_eq`.
    #[must_use]
    pub fn estimate_ne(&self, value: &AttrValue) -> (f64, bool) {
        let (equal, exact) = self.estimate_eq(value);
        #[allow(clippy::cast_precision_loss)]
        let present = self.present_count as f64;
        ((present - equal).max(0.0), exact)
    }

    /// Estimate how many records satisfy `attr in values`.
    ///
    /// The values are guaranteed distinct by
    /// [`FilterPredicate::validate`](super::FilterPredicate::validate), so
    /// their counts simply add. The sum is capped at the number of records
    /// carrying the attribute at all, which keeps the saturated-attribute
    /// heuristic from over-shooting into nonsense on a large `In` set.
    #[must_use]
    pub fn estimate_in(&self, values: &[AttrValue]) -> (f64, bool) {
        let mut total = 0.0_f64;
        let mut exact = true;
        for value in values {
            let (count, value_exact) = self.estimate_eq(value);
            total += count;
            exact &= value_exact;
        }
        #[allow(clippy::cast_precision_loss)]
        let present = self.present_count as f64;
        let capped = total.min(present);
        // Capping only loses exactness if it actually bit.
        let exact = exact && (capped - total).abs() <= f64::EPSILON;
        (capped, exact)
    }

    /// Estimate how many records satisfy a numeric range on this attribute.
    ///
    /// A zero-width inclusive range (`[x, x]`) is a *point* query, which an
    /// equi-width histogram fundamentally cannot answer — interpolating over a
    /// zero-width interval yields zero regardless of how many records actually
    /// hold `x`. Such queries are therefore routed to the exact value counts
    /// (trying both `Int(x)` and `Float(x)`, since the range does not say which
    /// variant the attribute holds), which is precisely the role a
    /// most-common-values list plays alongside a histogram in a real query
    /// optimizer.
    #[must_use]
    pub fn estimate_range(&self, lower: FilterBound, upper: FilterBound) -> (f64, bool) {
        if let (FilterBound::Inclusive(low), FilterBound::Inclusive(high)) = (lower, upper)
            && (low - high).abs() == 0.0
        {
            let mut total = 0.0_f64;
            let mut exact = true;
            #[allow(clippy::cast_possible_truncation)]
            let as_integer = low as i64;
            #[allow(clippy::cast_precision_loss)]
            let round_trips = (as_integer as f64 - low).abs() == 0.0;
            if round_trips {
                let (count, count_exact) = self.estimate_eq(&AttrValue::Int(as_integer));
                total += count;
                exact &= count_exact;
            }
            let (count, count_exact) = self.estimate_eq(&AttrValue::Float(low));
            total += count;
            exact &= count_exact;
            return (total, exact);
        }
        self.histogram.estimate_range(lower, upper)
    }
}

// ── SelectivityEstimator ─────────────────────────────────────────────────────

/// Maintains per-attribute statistics over a corpus and estimates predicate
/// selectivity compositionally over the AST.
///
/// The estimator sees every record the index sees, so an attribute it has
/// *never* observed is known to be absent from every record — an estimate of
/// zero for that attribute is therefore **exact**, not a guess.
#[derive(Debug, Clone)]
pub struct SelectivityEstimator {
    total: usize,
    attributes: HashMap<String, FilteredAttributeStats>,
    histogram_buckets: usize,
    max_tracked_values: usize,
}

impl SelectivityEstimator {
    /// Create an estimator over an empty corpus.
    #[must_use]
    pub fn new(histogram_buckets: usize, max_tracked_values: usize) -> Self {
        Self {
            total: 0,
            attributes: HashMap::new(),
            histogram_buckets: histogram_buckets.max(1),
            max_tracked_values: max_tracked_values.max(1),
        }
    }

    /// Create an estimator configured from a [`FilteredSearchConfig`].
    #[must_use]
    pub fn from_config(config: &FilteredSearchConfig) -> Self {
        Self::new(config.histogram_buckets, config.max_tracked_values)
    }

    /// Fold one record's metadata into the statistics.
    pub fn observe(&mut self, metadata: &FilteredMetadata) {
        self.total += 1;
        for (name, value) in metadata {
            self.attributes
                .entry(name.clone())
                .or_insert_with(|| {
                    FilteredAttributeStats::new(self.histogram_buckets, self.max_tracked_values)
                })
                .observe(value);
        }
    }

    /// The number of records observed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Statistics for one attribute, if it has ever been observed.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<&FilteredAttributeStats> {
        self.attributes.get(name)
    }

    /// The names of every attribute observed, sorted.
    #[must_use]
    pub fn attribute_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.attributes.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Estimate the fraction of the corpus that `predicate` admits.
    ///
    /// See the [module documentation](self) for the composition rules and the
    /// independence assumption they rest on.
    #[must_use]
    pub fn estimate(&self, predicate: &FilterPredicate) -> SelectivityEstimate {
        let (selectivity, exact_leaves) = self.estimate_fraction(predicate);
        SelectivityEstimate::new(
            selectivity,
            self.total,
            exact_leaves,
            predicate.requires_independence_assumption(),
        )
    }

    /// The recursive core: returns `(fraction_in_0_1, leaves_were_exact)`.
    fn estimate_fraction(&self, predicate: &FilterPredicate) -> (f64, bool) {
        match predicate {
            FilterPredicate::Eq { attr, value } => {
                self.leaf(attr, |stats| stats.estimate_eq(value))
            }
            FilterPredicate::Ne { attr, value } => {
                self.leaf(attr, |stats| stats.estimate_ne(value))
            }
            FilterPredicate::In { attr, values } => {
                self.leaf(attr, |stats| stats.estimate_in(values))
            }
            FilterPredicate::Range { attr, lower, upper } => {
                self.leaf(attr, |stats| stats.estimate_range(*lower, *upper))
            }
            FilterPredicate::Exists { attr } => self.leaf(attr, |stats| {
                #[allow(clippy::cast_precision_loss)]
                (stats.present_count() as f64, true)
            }),

            FilterPredicate::And(children) => {
                // Vacuously true over an empty conjunction.
                let mut fraction = 1.0_f64;
                let mut exact = true;
                for child in children {
                    let (child_fraction, child_exact) = self.estimate_fraction(child);
                    fraction *= child_fraction;
                    exact &= child_exact;
                }
                (fraction, exact)
            }

            FilterPredicate::Or(children) => {
                // Vacuously false over an empty disjunction.
                if children.is_empty() {
                    return (0.0, true);
                }

                // Merge equality disjuncts on the *same attribute* before doing
                // anything else. `lang = "ja" OR lang = "en"` are mutually
                // exclusive events — a record has one `lang`, not two — so their
                // selectivities simply **add**, exactly, and the independence
                // assumption is not merely unnecessary here but actively wrong:
                // it would compute `1 - (1-0.25)(1-0.25) = 0.4375` for two
                // quarter-of-the-corpus languages whose union is obviously
                // `0.5`. Real query optimizers special-case exactly this, and so
                // does the estimator.
                //
                // Groups are kept in first-appearance order, so the folding
                // arithmetic — and therefore the estimate — is deterministic.
                let mut groups: Vec<(&str, Vec<AttrValue>)> = Vec::new();
                let mut others: Vec<&FilterPredicate> = Vec::new();
                for child in children {
                    let (attr, values): (&str, &[AttrValue]) = match child {
                        FilterPredicate::Eq { attr, value } => {
                            (attr.as_str(), std::slice::from_ref(value))
                        }
                        FilterPredicate::In { attr, values } => (attr.as_str(), values.as_slice()),
                        other => {
                            others.push(other);
                            continue;
                        }
                    };
                    match groups.iter_mut().find(|(name, _)| *name == attr) {
                        Some((_, merged)) => merged.extend_from_slice(values),
                        None => groups.push((attr, values.to_vec())),
                    }
                }

                // Across distinct groups (and against the non-equality
                // disjuncts) independence is still assumed — two *different*
                // attributes really can co-occur.
                let mut complement = 1.0_f64;
                let mut exact = true;

                for (attr, values) in &groups {
                    // A value repeated across disjuncts must not be counted
                    // twice.
                    let mut deduplicated: Vec<AttrValue> = Vec::with_capacity(values.len());
                    for value in values {
                        if !deduplicated.contains(value) {
                            deduplicated.push(value.clone());
                        }
                    }
                    let (fraction, leaf_exact) =
                        self.leaf(attr, |stats| stats.estimate_in(&deduplicated));
                    complement *= 1.0 - fraction;
                    exact &= leaf_exact;
                }

                for other in others {
                    let (fraction, child_exact) = self.estimate_fraction(other);
                    complement *= 1.0 - fraction;
                    exact &= child_exact;
                }

                ((1.0 - complement).clamp(0.0, 1.0), exact)
            }

            FilterPredicate::Not(child) => {
                let (fraction, exact) = self.estimate_fraction(child);
                ((1.0 - fraction).clamp(0.0, 1.0), exact)
            }
        }
    }

    /// Evaluate a leaf against one attribute's statistics, converting the raw
    /// count the statistics return into a fraction of the corpus.
    ///
    /// An attribute the estimator has never seen is present in *no* record, so
    /// the leaf's true count is zero — exactly, not approximately.
    fn leaf<F>(&self, attr: &str, estimate: F) -> (f64, bool)
    where
        F: Fn(&FilteredAttributeStats) -> (f64, bool),
    {
        if self.total == 0 {
            return (0.0, true);
        }
        let Some(stats) = self.attributes.get(attr) else {
            return (0.0, true);
        };
        let (count, exact) = estimate(stats);
        #[allow(clippy::cast_precision_loss)]
        let fraction = count / self.total as f64;
        (fraction.clamp(0.0, 1.0), exact)
    }
}

// ── StrategySelector ─────────────────────────────────────────────────────────

/// Turns a [`SelectivityEstimate`] into a concrete [`FilterStrategy`].
///
/// # The cost model
///
/// Write `N` for the corpus size, `s` for the estimated selectivity, `k` for
/// the requested result count, `R` for the graph degree, `L` for the search
/// beam width, `gamma` for the ACORN neighbor-expansion factor and `m` for the
/// post-filter over-fetch multiplier. Roughly:
///
/// | Strategy | Distance computations | Recall |
/// |---|---|---|
/// | `PreFilter` | `~ s * N` | **1.0** — exact by construction |
/// | `PostFilter` | `~ L' * R` with `L' = max(L, k * m)` | `~ min(1, s * m)` — collapses for `s < 1/m` |
/// | `InFilter` | `~ L * R * gamma` | high, degrading gracefully as `s -> 0` |
///
/// The three regimes follow directly:
///
/// **Low `s` — pick `PreFilter`.** When `s * N` is small, an exhaustive scan of
/// the matching set costs less than a graph traversal *and* is exact. This is
/// also precisely the regime where `PostFilter` is broken and where `InFilter`
/// pays its highest tax (with almost no neighbor satisfying the predicate,
/// nearly every expansion triggers the two-hop step, and the traversal cannot
/// terminate early because its result heap never fills). Both of `InFilter`'s
/// rivals-in-cost arguments point the same way, so the exact strategy wins
/// outright.
///
/// The gate is on **both** the ratio and the absolute count: `s <= 0.01` means
/// something very different on a corpus of 1,000 vectors (10 matches — scan
/// them) than on one of a billion (10 million matches — do not). Hence
/// [`prefilter_max_matches`](super::FilteredSearchConfig::prefilter_max_matches).
///
/// **High `s` — pick `PostFilter`.** When almost everything matches, the
/// unconstrained top-`k * m` is almost entirely made of matching vectors, so
/// filtering it afterwards throws away almost nothing and the over-fetch is
/// nearly free. Its recall approaches the *unfiltered* recall of the graph,
/// which is the best any of the three can do.
///
/// The threshold must satisfy `postfilter_threshold >= 1 / m`, and
/// [`FilteredSearchConfig::validate`](super::FilteredSearchConfig::validate)
/// enforces it. That inequality is the whole safety argument for ever choosing
/// `PostFilter`: it is exactly the condition under which the over-fetch is
/// expected to contain at least `k` matching vectors.
///
/// **Everything in between — pick `InFilter`.** Too many matches to scan, too
/// few for a bounded over-fetch to reliably capture. This is the regime ACORN
/// was invented for, and the two-hop expansion is worth its cost precisely
/// here.
///
/// # A mis-estimate cannot corrupt results
///
/// If the estimator is wrong, the planner picks a slower strategy — or, in the
/// `PostFilter` case, one that returns fewer than `k` hits. It never returns a
/// hit that fails the predicate: every strategy verifies the real predicate
/// against the real metadata before admitting anything.
#[derive(Debug, Clone)]
pub struct StrategySelector {
    prefilter_threshold: f64,
    postfilter_threshold: f64,
    prefilter_max_matches: usize,
}

impl StrategySelector {
    /// Build a selector from an index configuration.
    #[must_use]
    pub fn from_config(config: &FilteredSearchConfig) -> Self {
        Self {
            prefilter_threshold: config.prefilter_threshold,
            postfilter_threshold: config.postfilter_threshold,
            prefilter_max_matches: config.prefilter_max_matches,
        }
    }

    /// Choose a concrete strategy for an estimate and a requested `top_k`.
    ///
    /// Never returns [`FilterStrategy::Auto`].
    #[must_use]
    pub fn select(&self, estimate: &SelectivityEstimate, top_k: usize) -> FilterStrategy {
        let matches = estimate.estimated_matches;
        #[allow(clippy::cast_precision_loss)]
        let max_scan = self.prefilter_max_matches as f64;
        #[allow(clippy::cast_precision_loss)]
        let requested = top_k as f64;

        // The matching set is small enough to scan exhaustively, and either the
        // predicate is selective enough to make that cheap, or the caller has
        // asked for at least as many results as there are matches (in which
        // case the entire matching set must be examined anyway, so an
        // approximate traversal of it can only lose recall for nothing).
        if matches <= max_scan
            && (estimate.selectivity <= self.prefilter_threshold || matches <= requested)
        {
            return FilterStrategy::PreFilter;
        }

        // Nearly everything matches: the over-fetch is nearly free and its
        // expected yield, `s * k * m`, comfortably exceeds `k`.
        if estimate.selectivity >= self.postfilter_threshold {
            return FilterStrategy::PostFilter;
        }

        FilterStrategy::InFilter
    }
}
