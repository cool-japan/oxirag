//! The inverted attribute index: turns a [`FilterPredicate`] into a small set
//! of candidate node ids without touching a single vector.
//!
//! This is what makes [`FilterStrategy::PreFilter`](super::FilterStrategy::PreFilter)
//! *cheap* as well as exact. Without it, "scan the matching set" would begin
//! with an `O(N)` predicate evaluation over the entire corpus just to discover
//! what the matching set *is* — and at that point the whole point of a
//! selective predicate has been thrown away.
//!
//! # Exactness bookkeeping
//!
//! Resolution is allowed to return a **superset** of the true matching set (the
//! caller re-verifies every candidate against the real predicate, so a superset
//! costs time, never correctness). But a superset is *poison* under negation:
//! complementing a superset yields a **subset** of the true complement, which
//! would silently drop matches. [`ResolvedCandidates`] therefore tracks whether
//! a set is exact or merely a superset, and `Not` refuses to complement
//! anything that is not exact — falling back to a full scan instead, which is
//! slower and unconditionally correct.
//!
//! The composition rules:
//!
//! | Node | Resolvable when | Result |
//! |---|---|---|
//! | `Eq` / `Ne` / `In` | the attribute has posting lists | exact |
//! | `Range` | the attribute is numeric | exact |
//! | `Exists` | always | exact |
//! | `And` | *at least one* child resolves | intersection of the resolved children — exact only if **all** children resolved exactly (an unresolved child simply drops a constraint, which widens the set) |
//! | `Or` | *every* child resolves | union — exact iff every child was exact (a union of supersets is a superset) |
//! | `Not` | the child resolves **exactly** | complement, exact |
//!
//! # Cardinality guards
//!
//! An attribute whose distinct-value count exceeds
//! [`max_postings_per_attr`](super::FilteredSearchConfig::max_postings_per_attr)
//! — a `f64` score with a distinct value per record, say — would otherwise
//! accumulate one singleton posting list per record, all overhead and no
//! benefit. Past the cap, its posting lists are dropped and equality lookups on
//! it degrade to a full scan. Ranges on it keep working, because the sorted
//! numeric array (one entry per record, not one list per distinct value) is
//! unaffected.

use std::collections::HashMap;

use super::predicate::FilterPredicate;
use super::types::{AttrValue, FilterBound, FilteredMetadata};

// ── Sorted-set helpers ───────────────────────────────────────────────────────

/// Intersection of two ascending, duplicate-free id lists.
fn intersect_sorted(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(left.len().min(right.len()));
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Union of two ascending, duplicate-free id lists.
fn union_sorted(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => {
                out.push(left[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(right[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(left[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

/// Set difference `left \ right` over ascending, duplicate-free id lists.
fn difference_sorted(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(left.len());
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < left.len() {
        while j < right.len() && right[j] < left[i] {
            j += 1;
        }
        if j < right.len() && right[j] == left[i] {
            i += 1;
            continue;
        }
        out.push(left[i]);
        i += 1;
    }
    out
}

/// Complement of `set` within `0..total`.
fn complement_sorted(set: &[u32], total: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(total.saturating_sub(set.len()));
    let mut cursor = 0_usize;
    for id in 0..total {
        #[allow(clippy::cast_possible_truncation)]
        let id = id as u32;
        while cursor < set.len() && set[cursor] < id {
            cursor += 1;
        }
        if cursor < set.len() && set[cursor] == id {
            continue;
        }
        out.push(id);
    }
    out
}

// ── ResolvedCandidates ───────────────────────────────────────────────────────

/// A candidate id set produced by the inverted index, tagged with whether it is
/// the true matching set or merely a superset of it.
///
/// See the [module documentation](self) for why the distinction is load-bearing
/// rather than cosmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedCandidates {
    /// Ascending, duplicate-free node ids.
    pub(crate) ids: Vec<u32>,
    /// `true` when `ids` is exactly the matching set; `false` when it is a
    /// superset. A superset is safe to *scan* and unsafe to *complement*.
    pub(crate) exact: bool,
}

// ── Per-attribute postings ───────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct AttributePostings {
    /// Every node carrying this attribute, ascending.
    present: Vec<u32>,
    /// Node ids per distinct value, each ascending. `None` once the distinct
    /// cardinality exceeds the configured cap.
    by_value: Option<HashMap<AttrValue, Vec<u32>>>,
    /// Numeric values, merged and sorted by value, for range resolution.
    numeric_sorted: Vec<(f64, u32)>,
    /// Numeric values inserted since the last [`FilteredAttributeIndex::finalize`],
    /// still unsorted. Scanned linearly during range resolution, which keeps
    /// ranges exact and available at all times without paying an `O(N)` memmove
    /// on every insert.
    numeric_pending: Vec<(f64, u32)>,
}

impl AttributePostings {
    fn observe(&mut self, id: u32, value: &AttrValue, max_postings: usize) {
        self.present.push(id);

        if let Some(by_value) = &mut self.by_value {
            if let Some(list) = by_value.get_mut(value) {
                list.push(id);
            } else if by_value.len() < max_postings {
                by_value.insert(value.clone(), vec![id]);
            } else {
                // Cardinality guard tripped: the posting lists are pure
                // overhead for this attribute. Drop them; equality lookups fall
                // back to a scan, ranges keep working off `numeric_sorted`.
                self.by_value = None;
            }
        }

        if let Some(numeric) = value.as_f64()
            && numeric.is_finite()
        {
            self.numeric_pending.push((numeric, id));
        }
    }

    fn finalize(&mut self) {
        if self.numeric_pending.is_empty() {
            return;
        }
        self.numeric_sorted.append(&mut self.numeric_pending);
        self.numeric_sorted
            .sort_by(|(left, left_id), (right, right_id)| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(left_id.cmp(right_id))
            });
    }

    /// Node ids whose numeric value satisfies both bounds, ascending.
    fn resolve_range(&self, lower: FilterBound, upper: FilterBound) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();

        // Binary-search the sorted prefix for the first value that could
        // satisfy the lower bound, then walk forward until the upper bound is
        // exceeded.
        let start = match lower {
            FilterBound::Unbounded => 0,
            FilterBound::Inclusive(bound) | FilterBound::Exclusive(bound) => self
                .numeric_sorted
                .partition_point(|(value, _)| *value < bound),
        };
        for &(value, id) in &self.numeric_sorted[start.min(self.numeric_sorted.len())..] {
            if !upper.accepts_as_upper(value) {
                // Sorted ascending: nothing beyond here can satisfy the upper
                // bound either.
                break;
            }
            if lower.accepts_as_lower(value) {
                ids.push(id);
            }
        }

        // The unsorted tail is scanned linearly. It is empty after `finalize`,
        // and bounded by the number of inserts since the last one otherwise.
        for &(value, id) in &self.numeric_pending {
            if lower.accepts_as_lower(value) && upper.accepts_as_upper(value) {
                ids.push(id);
            }
        }

        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

// ── FilteredAttributeIndex ───────────────────────────────────────────────────

/// Posting lists and sorted numeric arrays over every indexed attribute.
#[derive(Debug, Clone)]
pub(crate) struct FilteredAttributeIndex {
    total: usize,
    attributes: HashMap<String, AttributePostings>,
    max_postings_per_attr: usize,
}

impl FilteredAttributeIndex {
    /// Create an empty index.
    pub(crate) fn new(max_postings_per_attr: usize) -> Self {
        Self {
            total: 0,
            attributes: HashMap::new(),
            max_postings_per_attr: max_postings_per_attr.max(1),
        }
    }

    /// Record one node's metadata.
    ///
    /// Node ids must be handed in ascending order (which the index guarantees,
    /// since they are assigned sequentially on insertion) — that is what keeps
    /// every posting list sorted without ever sorting it.
    pub(crate) fn observe(&mut self, id: u32, metadata: &FilteredMetadata) {
        self.total += 1;
        let max_postings = self.max_postings_per_attr;
        for (name, value) in metadata {
            self.attributes
                .entry(name.clone())
                .or_default()
                .observe(id, value, max_postings);
        }
    }

    /// Merge each attribute's pending numeric values into its sorted array.
    pub(crate) fn finalize(&mut self) {
        for postings in self.attributes.values_mut() {
            postings.finalize();
        }
    }

    /// The number of nodes observed.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Resolve `predicate` to a candidate id set, or `None` when the inverted
    /// index cannot answer it and the caller must fall back to a full scan.
    ///
    /// The returned set may be a superset of the true matching set; the caller
    /// must re-verify each candidate with
    /// [`FilterPredicate::matches`](super::FilterPredicate::matches). See the
    /// [module documentation](self) for the composition rules.
    pub(crate) fn resolve(&self, predicate: &FilterPredicate) -> Option<ResolvedCandidates> {
        match predicate {
            FilterPredicate::Eq { .. }
            | FilterPredicate::Ne { .. }
            | FilterPredicate::In { .. }
            | FilterPredicate::Exists { .. }
            | FilterPredicate::Range { .. } => self.resolve_leaf(predicate),

            FilterPredicate::And(children) => self.resolve_conjunction(children),
            FilterPredicate::Or(children) => self.resolve_disjunction(children),

            FilterPredicate::Not(child) => {
                let resolved = self.resolve(child)?;
                // Complementing a *superset* yields a **subset** of the true
                // complement, which would silently drop real matches. Refuse,
                // and let the caller fall back to a scan.
                if !resolved.exact {
                    return None;
                }
                Some(ResolvedCandidates {
                    ids: complement_sorted(&resolved.ids, self.total),
                    exact: true,
                })
            }
        }
    }

    /// The empty, exactly-known candidate set — what an attribute nobody carries
    /// resolves to. It is a *conclusion*, not a failure: if the index has never
    /// observed the attribute, no record has it.
    fn empty_exact() -> ResolvedCandidates {
        ResolvedCandidates {
            ids: Vec::new(),
            exact: true,
        }
    }

    /// Resolve a leaf predicate from the posting lists and sorted numeric arrays.
    fn resolve_leaf(&self, predicate: &FilterPredicate) -> Option<ResolvedCandidates> {
        match predicate {
            FilterPredicate::Eq { attr, value } => {
                let Some(postings) = self.attributes.get(attr) else {
                    return Some(Self::empty_exact());
                };
                // `None` here means the cardinality guard dropped this
                // attribute's posting lists; the caller must scan.
                let by_value = postings.by_value.as_ref()?;
                Some(ResolvedCandidates {
                    ids: by_value.get(value).cloned().unwrap_or_default(),
                    exact: true,
                })
            }

            FilterPredicate::Ne { attr, value } => {
                let Some(postings) = self.attributes.get(attr) else {
                    return Some(Self::empty_exact());
                };
                let by_value = postings.by_value.as_ref()?;
                let equal = by_value.get(value).map_or(&[][..], Vec::as_slice);
                Some(ResolvedCandidates {
                    // `Ne` requires presence, so this is `present \ equal`, not
                    // the complement of `equal`.
                    ids: difference_sorted(&postings.present, equal),
                    exact: true,
                })
            }

            FilterPredicate::In { attr, values } => {
                let Some(postings) = self.attributes.get(attr) else {
                    return Some(Self::empty_exact());
                };
                let by_value = postings.by_value.as_ref()?;
                let mut ids: Vec<u32> = Vec::new();
                for value in values {
                    if let Some(list) = by_value.get(value) {
                        ids = union_sorted(&ids, list);
                    }
                }
                Some(ResolvedCandidates { ids, exact: true })
            }

            FilterPredicate::Exists { attr } => Some(ResolvedCandidates {
                ids: self
                    .attributes
                    .get(attr)
                    .map(|postings| postings.present.clone())
                    .unwrap_or_default(),
                exact: true,
            }),

            FilterPredicate::Range { attr, lower, upper } => Some(ResolvedCandidates {
                ids: self
                    .attributes
                    .get(attr)
                    .map(|postings| postings.resolve_range(*lower, *upper))
                    .unwrap_or_default(),
                exact: true,
            }),

            // The caller dispatches only leaves here.
            FilterPredicate::And(_) | FilterPredicate::Or(_) | FilterPredicate::Not(_) => None,
        }
    }

    /// Intersect whichever conjuncts the index can answer.
    ///
    /// An unresolved conjunct simply *drops a constraint*, which can only widen
    /// the candidate set — so the result stays a safe superset, and the caller's
    /// verification pass restores exactness. This is the classic
    /// "use the selective index, filter the rest" plan.
    fn resolve_conjunction(&self, children: &[FilterPredicate]) -> Option<ResolvedCandidates> {
        if children.is_empty() {
            // Vacuously true: every node is a candidate.
            #[allow(clippy::cast_possible_truncation)]
            let ids: Vec<u32> = (0..self.total as u32).collect();
            return Some(ResolvedCandidates { ids, exact: true });
        }

        let mut accumulated: Option<Vec<u32>> = None;
        // Exact only if *every* child resolved, and exactly: an unresolved child
        // drops a constraint, and a superset child widens one.
        let mut exact = true;
        for child in children {
            match self.resolve(child) {
                Some(resolved) => {
                    exact &= resolved.exact;
                    accumulated = Some(match accumulated {
                        Some(existing) => intersect_sorted(&existing, &resolved.ids),
                        None => resolved.ids,
                    });
                }
                None => exact = false,
            }
        }
        accumulated.map(|ids| ResolvedCandidates { ids, exact })
    }

    /// Union every disjunct — but only if *every* one of them resolves.
    ///
    /// Unlike a conjunction, a disjunction cannot afford to drop a term: an
    /// unresolved disjunct is one whose matches would simply be *missing* from
    /// the union, and no later verification pass can put back a candidate that
    /// was never proposed. One unanswerable disjunct therefore forces a scan.
    fn resolve_disjunction(&self, children: &[FilterPredicate]) -> Option<ResolvedCandidates> {
        if children.is_empty() {
            // Vacuously false: no node is a candidate.
            return Some(Self::empty_exact());
        }
        let mut ids: Vec<u32> = Vec::new();
        let mut exact = true;
        for child in children {
            let resolved = self.resolve(child)?;
            exact &= resolved.exact;
            ids = union_sorted(&ids, &resolved.ids);
        }
        Some(ResolvedCandidates { ids, exact })
    }
}
