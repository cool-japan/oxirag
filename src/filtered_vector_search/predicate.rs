//! The metadata predicate AST, its evaluator, and its validator.
//!
//! A [`FilterPredicate`] is a small boolean expression over the attributes of
//! a [`FilteredMetadata`] record. It is the *only* thing the three search
//! strategies disagree about how to apply — they all agree on what it *means*,
//! and that meaning is defined here, once, by [`FilterPredicate::matches`].
//!
//! # Semantics
//!
//! Every **leaf** predicate is `false` when the attribute it names is absent.
//! There is no implicit null propagation and no three-valued logic: a missing
//! attribute simply fails the leaf. This makes the negation rules explicit and
//! predictable:
//!
//! | Predicate | True when |
//! |---|---|
//! | `Eq { attr, value }` | `attr` is present **and** equals `value` (type-tagged equality; see [`AttrValue`]) |
//! | `Ne { attr, value }` | `attr` is present **and** differs from `value` |
//! | `Not(Eq { attr, value })` | `attr` is absent **or** differs from `value` |
//! | `Range { attr, .. }` | `attr` is present, is numeric, **and** lies within the bounds |
//! | `In { attr, values }` | `attr` is present **and** equals one of `values` |
//! | `Exists { attr }` | `attr` is present, whatever its value |
//!
//! Note carefully that `Ne` and `Not(Eq)` are **not** the same predicate: `Ne`
//! requires presence, `Not(Eq)` does not. Both are useful, and conflating them
//! is a classic source of silently wrong filters, so both are offered.
//!
//! # Empty connectives
//!
//! `And(vec![])` is vacuously **true** (a universally-quantified statement over
//! an empty set) and `Or(vec![])` is vacuously **false** (an
//! existentially-quantified one). [`FilterPredicate::all`] and
//! [`FilterPredicate::none`] are named constructors for exactly these, so that
//! callers never have to rely on remembering which way round it goes.

use serde::{Deserialize, Serialize};

use super::types::{AttrValue, FilterBound, FilteredMetadata, FilteredSearchError};

// ── FilterPredicate ──────────────────────────────────────────────────────────

/// A boolean expression over vector metadata.
///
/// See the [module documentation](self) for the full truth table, in
/// particular the treatment of absent attributes and the deliberate difference
/// between `Ne` and `Not(Eq)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterPredicate {
    /// `attr` is present and its value is exactly `value`.
    ///
    /// Equality is type-tagged: `Int(3)` does not equal `Float(3.0)`. See
    /// [`AttrValue`].
    Eq {
        /// The attribute name.
        attr: String,
        /// The value to compare against.
        value: AttrValue,
    },

    /// `attr` is present and its value differs from `value`.
    ///
    /// This is **not** `Not(Eq)`: a record missing `attr` entirely fails `Ne`
    /// but satisfies `Not(Eq)`.
    Ne {
        /// The attribute name.
        attr: String,
        /// The value to compare against.
        value: AttrValue,
    },

    /// `attr` is present, numeric, and lies inside the interval described by
    /// `lower` and `upper`.
    ///
    /// Both `Int` and `Float` attributes participate: bounds are `f64` and are
    /// compared against [`AttrValue::as_f64`]. A `Str` or `Bool` attribute
    /// never satisfies a range.
    Range {
        /// The attribute name.
        attr: String,
        /// The lower end of the interval.
        lower: FilterBound,
        /// The upper end of the interval.
        upper: FilterBound,
    },

    /// `attr` is present and its value equals one of `values`.
    ///
    /// An empty `values` set matches nothing (it is `Or` over an empty set).
    In {
        /// The attribute name.
        attr: String,
        /// The admissible values.
        values: Vec<AttrValue>,
    },

    /// `attr` is present, whatever its value.
    Exists {
        /// The attribute name.
        attr: String,
    },

    /// Conjunction. Empty is vacuously **true** — see [`FilterPredicate::all`].
    And(Vec<FilterPredicate>),

    /// Disjunction. Empty is vacuously **false** — see [`FilterPredicate::none`].
    Or(Vec<FilterPredicate>),

    /// Negation.
    Not(Box<FilterPredicate>),
}

impl FilterPredicate {
    // ── Constructors ─────────────────────────────────────────────────────

    /// `attr == value`.
    #[must_use]
    pub fn eq(attr: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        Self::Eq {
            attr: attr.into(),
            value: value.into(),
        }
    }

    /// `attr` is present and `attr != value`.
    #[must_use]
    pub fn ne(attr: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        Self::Ne {
            attr: attr.into(),
            value: value.into(),
        }
    }

    /// `attr` lies in the interval described by `lower` and `upper`.
    #[must_use]
    pub fn range(attr: impl Into<String>, lower: FilterBound, upper: FilterBound) -> Self {
        Self::Range {
            attr: attr.into(),
            lower,
            upper,
        }
    }

    /// `lower <= attr <= upper`.
    #[must_use]
    pub fn range_inclusive(attr: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self::range(
            attr,
            FilterBound::Inclusive(lower),
            FilterBound::Inclusive(upper),
        )
    }

    /// `attr >= lower`, with no upper bound.
    #[must_use]
    pub fn at_least(attr: impl Into<String>, lower: f64) -> Self {
        Self::range(attr, FilterBound::Inclusive(lower), FilterBound::Unbounded)
    }

    /// `attr <= upper`, with no lower bound.
    #[must_use]
    pub fn at_most(attr: impl Into<String>, upper: f64) -> Self {
        Self::range(attr, FilterBound::Unbounded, FilterBound::Inclusive(upper))
    }

    /// `attr` is one of `values`.
    #[must_use]
    pub fn in_set<V: Into<AttrValue>>(
        attr: impl Into<String>,
        values: impl IntoIterator<Item = V>,
    ) -> Self {
        Self::In {
            attr: attr.into(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// `attr` is present.
    #[must_use]
    pub fn exists(attr: impl Into<String>) -> Self {
        Self::Exists { attr: attr.into() }
    }

    /// Conjunction of `children`.
    #[must_use]
    pub fn and(children: impl IntoIterator<Item = Self>) -> Self {
        Self::And(children.into_iter().collect())
    }

    /// Disjunction of `children`.
    #[must_use]
    pub fn or(children: impl IntoIterator<Item = Self>) -> Self {
        Self::Or(children.into_iter().collect())
    }

    /// Negation of `child`.
    ///
    /// Named `not` to sit alongside [`and`](Self::and) and [`or`](Self::or) as a
    /// *constructor* of AST nodes, not as an implementation of
    /// [`std::ops::Not`] — which would take `self` by value and read as an
    /// operator rather than as the building of a syntax tree.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn not(child: Self) -> Self {
        Self::Not(Box::new(child))
    }

    /// The predicate satisfied by every record — the identity of `And`.
    ///
    /// A search under this predicate is an ordinary unconstrained ANN search.
    #[must_use]
    pub fn all() -> Self {
        Self::And(Vec::new())
    }

    /// The predicate satisfied by no record — the identity of `Or`.
    #[must_use]
    pub fn none() -> Self {
        Self::Or(Vec::new())
    }

    // ── Evaluation ───────────────────────────────────────────────────────

    /// Evaluate this predicate against a metadata record.
    ///
    /// This is the single source of truth for what the predicate *means*. All
    /// three search strategies, the inverted-index candidate resolver, and the
    /// test suite's brute-force ground truth funnel through it, which is what
    /// makes "`PreFilter` is exact" a claim about the strategies rather than a
    /// claim about two independent implementations of the same boolean logic.
    ///
    /// Cost is `O(|AST|)` with an early exit on `And`/`Or`.
    #[must_use]
    pub fn matches(&self, metadata: &FilteredMetadata) -> bool {
        match self {
            Self::Eq { attr, value } => metadata.get(attr) == Some(value),

            Self::Ne { attr, value } => match metadata.get(attr) {
                Some(actual) => actual != value,
                // Absent attribute fails every leaf, `Ne` included.
                None => false,
            },

            Self::Range { attr, lower, upper } => match metadata
                .get(attr)
                .and_then(AttrValue::as_f64)
            {
                // A NaN attribute value satisfies no bound: every comparison
                // against NaN is false, which is exactly what we want here.
                Some(numeric) => lower.accepts_as_lower(numeric) && upper.accepts_as_upper(numeric),
                None => false,
            },

            Self::In { attr, values } => match metadata.get(attr) {
                Some(actual) => values.iter().any(|candidate| candidate == actual),
                None => false,
            },

            Self::Exists { attr } => metadata.contains(attr),

            Self::And(children) => children.iter().all(|child| child.matches(metadata)),

            Self::Or(children) => children.iter().any(|child| child.matches(metadata)),

            Self::Not(child) => !child.matches(metadata),
        }
    }

    // ── Introspection ────────────────────────────────────────────────────

    /// Every attribute name referenced anywhere in this predicate, deduplicated
    /// and sorted for determinism.
    #[must_use]
    pub fn referenced_attributes(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.collect_attributes(&mut names);
        names.sort_unstable();
        names.dedup();
        names
    }

    fn collect_attributes(&self, out: &mut Vec<String>) {
        match self {
            Self::Eq { attr, .. }
            | Self::Ne { attr, .. }
            | Self::Range { attr, .. }
            | Self::In { attr, .. }
            | Self::Exists { attr } => out.push(attr.clone()),
            Self::And(children) | Self::Or(children) => {
                for child in children {
                    child.collect_attributes(out);
                }
            }
            Self::Not(child) => child.collect_attributes(out),
        }
    }

    /// The number of nodes in this AST, leaves included.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Eq { .. }
            | Self::Ne { .. }
            | Self::Range { .. }
            | Self::In { .. }
            | Self::Exists { .. } => 1,
            Self::And(children) | Self::Or(children) => {
                1 + children.iter().map(Self::node_count).sum::<usize>()
            }
            Self::Not(child) => 1 + child.node_count(),
        }
    }

    /// Whether this predicate is `all()` — i.e. an empty conjunction, which
    /// every record satisfies. Used to short-circuit to an unconstrained
    /// search.
    #[must_use]
    pub fn is_unconstrained(&self) -> bool {
        matches!(self, Self::And(children) if children.is_empty())
    }

    /// Whether estimating this predicate's selectivity requires the
    /// independence assumption.
    ///
    /// True whenever two or more sub-selectivities must be folded together
    /// without knowing how the underlying attributes co-vary — which is every
    /// multi-child `And`, and every multi-child `Or` **except** one case:
    ///
    /// A disjunction of equality tests on a *single* attribute
    /// (`lang = "ja" OR lang = "en"`, or several `In`s over the same attribute)
    /// describes **mutually exclusive** events, because a record has one value
    /// for an attribute, not several. Their selectivities therefore *add* —
    /// exactly, assuming nothing — and the estimator adds them. Independence
    /// there would not merely be unnecessary, it would be *wrong*: it computes
    /// `1 - (1 - 0.25)(1 - 0.25) = 0.4375` for two quarter-of-the-corpus
    /// languages whose union is plainly `0.5`.
    ///
    /// Surfaced through [`SelectivityEstimate::independence_assumed`](super::types::SelectivityEstimate::independence_assumed).
    #[must_use]
    pub fn requires_independence_assumption(&self) -> bool {
        match self {
            Self::Eq { .. }
            | Self::Ne { .. }
            | Self::Range { .. }
            | Self::In { .. }
            | Self::Exists { .. } => false,
            Self::And(children) => {
                children.len() > 1 || children.iter().any(Self::requires_independence_assumption)
            }
            Self::Or(children) => {
                if children.len() > 1 && Self::is_single_attribute_disjunction(children) {
                    return false;
                }
                children.len() > 1 || children.iter().any(Self::requires_independence_assumption)
            }
            Self::Not(child) => child.requires_independence_assumption(),
        }
    }

    /// Whether every child is an `Eq` or `In` on one and the same attribute —
    /// the mutually-exclusive case the estimator sums exactly rather than
    /// folding under independence.
    fn is_single_attribute_disjunction(children: &[Self]) -> bool {
        let mut shared: Option<&str> = None;
        for child in children {
            let name = match child {
                Self::Eq { attr, .. } | Self::In { attr, .. } => attr.as_str(),
                _ => return false,
            };
            match shared {
                Some(existing) if existing != name => return false,
                Some(_) => {}
                None => shared = Some(name),
            }
        }
        shared.is_some()
    }

    // ── Validation ───────────────────────────────────────────────────────

    /// Check that the AST is well-formed.
    ///
    /// A malformed predicate is rejected up front rather than silently
    /// matching nothing, because "my filter returns no results" is a far worse
    /// diagnostic than "your range has `lower > upper`".
    ///
    /// # Errors
    ///
    /// Returns [`FilteredSearchError::InvalidPredicate`] when:
    ///
    /// - an attribute name is empty;
    /// - a [`FilterBound`] carries a NaN;
    /// - a `Range`'s lower bound exceeds its upper bound, or the two coincide
    ///   while at least one of them is exclusive (an empty interval);
    /// - an `In` set contains duplicate values (which would double-count in the
    ///   selectivity estimator and signals a caller mistake).
    pub fn validate(&self) -> Result<(), FilteredSearchError> {
        let malformed = |reason: String| FilteredSearchError::InvalidPredicate { reason };

        match self {
            Self::Eq { attr, .. } | Self::Ne { attr, .. } | Self::Exists { attr } => {
                if attr.is_empty() {
                    return Err(malformed("attribute name must not be empty".to_string()));
                }
            }

            Self::In { attr, values } => {
                if attr.is_empty() {
                    return Err(malformed("attribute name must not be empty".to_string()));
                }
                for (position, value) in values.iter().enumerate() {
                    if values[..position].contains(value) {
                        return Err(malformed(format!(
                            "`In` set for attribute `{attr}` contains a duplicate {} value at \
                             position {position}",
                            value.kind_name()
                        )));
                    }
                }
            }

            Self::Range { attr, lower, upper } => {
                if attr.is_empty() {
                    return Err(malformed("attribute name must not be empty".to_string()));
                }
                for (label, bound) in [("lower", lower), ("upper", upper)] {
                    if let Some(position) = bound.value()
                        && position.is_nan()
                    {
                        return Err(malformed(format!(
                            "{label} bound of range on attribute `{attr}` is NaN"
                        )));
                    }
                }
                if let (Some(low), Some(high)) = (lower.value(), upper.value()) {
                    let degenerate_open = (low - high).abs() == 0.0
                        && (matches!(lower, FilterBound::Exclusive(_))
                            || matches!(upper, FilterBound::Exclusive(_)));
                    if low > high || degenerate_open {
                        return Err(malformed(format!(
                            "range on attribute `{attr}` is empty: [{low}, {high}] with \
                             lower={lower:?}, upper={upper:?}"
                        )));
                    }
                }
            }

            Self::And(children) | Self::Or(children) => {
                for child in children {
                    child.validate()?;
                }
            }

            Self::Not(child) => child.validate()?,
        }
        Ok(())
    }
}
