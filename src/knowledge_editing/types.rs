//! Value types, configuration, and the error taxonomy for [`crate::knowledge_editing`].

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::linalg::KnowledgeEditLinalgError;

/// Default relative ridge added to the empirical second moment `E[k k^T]`.
///
/// It is *relative*: the absolute ridge is this factor times the mean diagonal of the
/// empirical second moment, so rescaling every preserved key leaves the resulting edit
/// unchanged (see [`EditableMemory::from_preserved_keys`](super::EditableMemory::from_preserved_keys)).
/// `1e-6` is small enough to leave the key geometry untouched and large enough to keep the
/// Cholesky factorization comfortably inside the positive-definite cone even when the
/// preserved keys span a strict subspace.
pub const DEFAULT_COVARIANCE_RIDGE: f64 = 1e-6;

/// Default deferral radius `epsilon` of the edit codebook.
pub const DEFAULT_DEFERRAL_RADIUS: f64 = 0.1;

/// Default tolerance on the post-condition `W' k* = v*`.
///
/// The rank-1 edit satisfies `W' k* = v*` *exactly* in real arithmetic. In `f64` the
/// achieved residual is bounded by the backward error of the triangular solves that apply
/// `C^-1`, which is `O(d) * kappa(C) * f64::EPSILON` relative to `||v*||`. For a memory width
/// of a few hundred and a ridge-conditioned `C`, that lands far below `1e-9`; a residual
/// *above* `1e-9` therefore means the edit did not actually take, and the write is rolled
/// back rather than reported as a success.
pub const DEFAULT_POSTCONDITION_TOLERANCE: f64 = 1e-9;

/// Two edit keys closer than this in Euclidean distance are the **same** key: a second insert
/// at that key is an explicit *revision* of the existing entry, not a new entry that would
/// silently shadow it.
pub const IDENTICAL_KEY_TOLERANCE: f64 = 1e-12;

/// Two edit values closer than this in Euclidean distance assert the **same** fact. The
/// codebook does not push their deferral balls apart, because a query landing in the overlap
/// is served the same answer either way.
pub const IDENTICAL_VALUE_TOLERANCE: f64 = 1e-12;

/// Identifier of a single edit, unique within one [`EditCodebook`](super::EditCodebook).
///
/// Ids are handed out in insertion order and are **never reused**, including after a
/// [`retract`](super::EditCodebook::retract): an id that once named an edit never names a
/// different one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EditId(pub u64);

impl fmt::Display for EditId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edit#{}", self.0)
    }
}

/// A key into the linear memory — the `k*` of the rank-1 edit, and the query vector the
/// codebook measures its deferral radius against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditKey {
    values: Vec<f64>,
}

impl EditKey {
    /// Wrap a coordinate vector as an edit key.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::EmptyVector`] if `values` is empty, or
    /// [`KnowledgeEditError::NonFinite`] if any coordinate is a `NaN` or an infinity — a
    /// non-finite key would contaminate the memory matrix irrecoverably, so it is rejected at
    /// the boundary rather than deep inside a solve.
    pub fn new(values: Vec<f64>) -> Result<Self, KnowledgeEditError> {
        if values.is_empty() {
            return Err(KnowledgeEditError::EmptyVector { what: "edit key" });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(KnowledgeEditError::NonFinite { what: "edit key" });
        }
        Ok(Self { values })
    }

    /// The coordinates of the key.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// The key width `d`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// A value stored in the linear memory — the `v*` of the rank-1 edit, and what the codebook
/// serves when a query falls inside an edit's deferral radius.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditValue {
    values: Vec<f64>,
}

impl EditValue {
    /// Wrap a coordinate vector as an edit value.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::EmptyVector`] if `values` is empty, or
    /// [`KnowledgeEditError::NonFinite`] if any coordinate is a `NaN` or an infinity.
    pub fn new(values: Vec<f64>) -> Result<Self, KnowledgeEditError> {
        if values.is_empty() {
            return Err(KnowledgeEditError::EmptyVector { what: "edit value" });
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(KnowledgeEditError::NonFinite { what: "edit value" });
        }
        Ok(Self { values })
    }

    /// The coordinates of the value.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// The value width `m`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

/// How a sequence of edits is folded into the memory matrix.
///
/// The two strategies differ in *what they solve against*, and the difference is exactly the
/// well-known failure mode of sequential model editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum EditStrategy {
    /// `MEMIT`-style. Every accumulated constraint is re-solved **jointly against the base
    /// weights `W_0`**, so the cumulative delta is the minimum-`C`-norm matrix satisfying
    /// *all* of them at once. Every edited key maps to its value exactly, no matter how many
    /// edits precede it. Costs `O(E^3 + E^2 d + E d^2)` per call for `E` accumulated edits.
    ///
    /// This is the default because it is the only one of the two that keeps the module's
    /// stated post-condition true after more than one edit.
    #[default]
    Joint,
    /// `ROME`-style, applied one at a time. Each edit is solved against the **current**
    /// weights and written on top of them. Costs `O(d^2 + m d)` per edit, but the `e`-th edit
    /// perturbs every key — including the keys of edits `1..e`, which are *not* constraints of
    /// its optimization problem. Earlier edits therefore decay, and
    /// [`EditResult::prior_edit_residual`] measures by how much.
    ///
    /// Two edit keys that are `C^-1`-orthogonal do not interfere, so the decay is not
    /// inevitable — it is a function of how correlated the edits are. The module measures it
    /// rather than assuming it away.
    Sequential,
}

/// Where an edit was materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EditScope {
    /// Baked into the parametric memory at the named site — *and* recorded in the codebook,
    /// which every edit always is.
    Parametric {
        /// Index of the memory site that was written.
        site: usize,
    },
    /// Held in the codebook only. The parametric write was **declined**, because the exact
    /// collateral cost `||Delta||_C` of taking it would have exceeded
    /// [`EditConfig::max_collateral_drift`]. The edit is not lost — it is served from the
    /// codebook, exactly, with no drift at all. This is the deferral memory earning its keep.
    CodebookOnly,
}

impl EditScope {
    /// The site written, or `None` for a codebook-only edit.
    #[must_use]
    pub fn site(&self) -> Option<usize> {
        match *self {
            Self::Parametric { site } => Some(site),
            Self::CodebookOnly => None,
        }
    }

    /// Whether the edit was baked into a memory matrix.
    #[must_use]
    pub fn is_parametric(&self) -> bool {
        matches!(self, Self::Parametric { .. })
    }
}

/// What the codebook decided about one query key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EditVerdict {
    /// The query fell inside an edit's deferral radius and is served the edited value
    /// verbatim.
    Served {
        /// Which edit claimed the query.
        edit_id: EditId,
        /// The value that edit asserts.
        value: EditValue,
        /// Distance from the query to that edit's key. Always `<= radius`.
        distance: f64,
        /// The claiming edit's *current* deferral radius, which may be narrower than the
        /// radius it was inserted with if a later, conflicting edit landed nearby.
        radius: f64,
    },
    /// No edit claims this query: the system defers to the base model.
    Deferred {
        /// The closest edit, if the codebook is non-empty — reported for diagnostics, *not*
        /// served.
        nearest: Option<EditId>,
        /// Distance to that closest edit. Strictly greater than its radius, or it would have
        /// been served.
        nearest_distance: Option<f64>,
    },
}

impl EditVerdict {
    /// Whether an edit claimed the query.
    #[must_use]
    pub fn is_served(&self) -> bool {
        matches!(self, Self::Served { .. })
    }

    /// Whether the query fell through to the base model.
    #[must_use]
    pub fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred { .. })
    }

    /// The edited value, if one was served.
    #[must_use]
    pub fn served_value(&self) -> Option<&EditValue> {
        match self {
            Self::Served { value, .. } => Some(value),
            Self::Deferred { .. } => None,
        }
    }

    /// The id of the edit that served the query, if any.
    #[must_use]
    pub fn served_id(&self) -> Option<EditId> {
        match *self {
            Self::Served { edit_id, .. } => Some(edit_id),
            Self::Deferred { .. } => None,
        }
    }
}

/// One entry of the edit codebook: a durable, non-evicting key → value assertion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditRecord {
    /// Stable identifier, unique and never reused.
    pub id: EditId,
    /// The key this edit fires on.
    pub key: EditKey,
    /// The value it asserts.
    pub value: EditValue,
    /// The **current** deferral radius. Starts at [`initial_radius`](Self::initial_radius) and
    /// only ever shrinks — when a later edit with a *different* value lands close enough that
    /// their balls would overlap, both are pushed back to half the distance between their keys
    /// so that neither can answer for the other. Shrinking a radius never costs an edit its own
    /// key, which sits at distance zero from itself.
    pub radius: f64,
    /// The radius this edit was inserted with, retained so that a shrink is visible rather
    /// than silent.
    pub initial_radius: f64,
    /// Optional caller-supplied label, e.g. the fact being asserted.
    pub label: Option<String>,
    /// How many times this entry's value has been explicitly revised by a later insert at the
    /// same key. A revision is an *overwrite the caller asked for*, not an eviction: the entry,
    /// its id, and its key all survive.
    pub revisions: usize,
}

/// Counters describing the pressure a codebook has been under.
///
/// These exist so a caller can *demonstrate*, rather than assert, that pressure did not cost
/// it an edit: `inserts - retractions == len()` after any number of lookups, revisions, and
/// radius shrinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CodebookStats {
    /// Inserts that created a new entry.
    pub inserts: u64,
    /// Inserts at an existing key that revised its value in place.
    pub revisions: u64,
    /// Explicit, caller-requested retractions — the only way an entry ever leaves.
    pub retractions: u64,
    /// Lookups performed.
    pub lookups: u64,
    /// Lookups that an edit claimed.
    pub served: u64,
    /// Lookups that fell through to the base model.
    pub deferred: u64,
    /// Individual radius reductions performed to keep conflicting edits' balls disjoint.
    pub radius_shrinks: u64,
}

/// Configuration for a [`KnowledgeEditor`](super::KnowledgeEditor).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditConfig {
    /// Key width `d` of every memory site.
    pub key_dim: usize,
    /// Value width `m` of every memory site.
    pub value_dim: usize,
    /// Deferral radius `epsilon` given to a new codebook entry, unless the request overrides
    /// it with [`EditRequest::with_radius`].
    pub deferral_radius: f64,
    /// Largest `||W' k* - v*||` accepted as "the edit took". A write whose measured
    /// post-condition residual exceeds this is **rolled back** and reported as
    /// [`KnowledgeEditError::PostconditionViolated`] — the editor never returns a memory it
    /// has not verified. See [`DEFAULT_POSTCONDITION_TOLERANCE`] for why the default is what
    /// it is.
    pub postcondition_tolerance: f64,
    /// Budget on the exact collateral cost `||W' - W_0||_C` of the *cumulative* parametric
    /// delta, which upper-bounds the root-mean-square drift the edits inflict on the preserved
    /// keys.
    ///
    /// `None` — the default — always takes the parametric write. `Some(budget)` makes the
    /// editor decline any parametric write that would push the cumulative cost past `budget`,
    /// routing that edit to the codebook instead, where it is served exactly and drifts
    /// nothing. Admission is greedy in request order.
    pub max_collateral_drift: Option<f64>,
    /// How accumulated edits are folded into the memory. See [`EditStrategy`].
    pub strategy: EditStrategy,
}

impl Default for EditConfig {
    fn default() -> Self {
        Self {
            key_dim: 0,
            value_dim: 0,
            deferral_radius: DEFAULT_DEFERRAL_RADIUS,
            postcondition_tolerance: DEFAULT_POSTCONDITION_TOLERANCE,
            max_collateral_drift: None,
            strategy: EditStrategy::default(),
        }
    }
}

impl EditConfig {
    /// A configuration for a memory that maps `key_dim`-vectors to `value_dim`-vectors.
    #[must_use]
    pub fn new(key_dim: usize, value_dim: usize) -> Self {
        Self {
            key_dim,
            value_dim,
            ..Self::default()
        }
    }

    /// Set the default deferral radius `epsilon`.
    #[must_use]
    pub fn with_deferral_radius(mut self, radius: f64) -> Self {
        self.deferral_radius = radius;
        self
    }

    /// Set the multi-edit strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: EditStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the collateral-drift budget. See [`EditConfig::max_collateral_drift`].
    #[must_use]
    pub fn with_max_collateral_drift(mut self, budget: f64) -> Self {
        self.max_collateral_drift = Some(budget);
        self
    }

    /// Set the post-condition tolerance.
    #[must_use]
    pub fn with_postcondition_tolerance(mut self, tolerance: f64) -> Self {
        self.postcondition_tolerance = tolerance;
        self
    }

    /// Check the configuration is usable.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::InvalidConfig`] if a dimension is zero, or a threshold is
    /// negative, non-finite, or (for the tolerance) not strictly positive.
    pub fn validate(&self) -> Result<(), KnowledgeEditError> {
        if self.key_dim == 0 {
            return Err(KnowledgeEditError::InvalidConfig(
                "key_dim must be at least 1".to_string(),
            ));
        }
        if self.value_dim == 0 {
            return Err(KnowledgeEditError::InvalidConfig(
                "value_dim must be at least 1".to_string(),
            ));
        }
        if !self.deferral_radius.is_finite() || self.deferral_radius < 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "deferral_radius must be finite and non-negative, got {}",
                self.deferral_radius
            )));
        }
        if !self.postcondition_tolerance.is_finite() || self.postcondition_tolerance <= 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "postcondition_tolerance must be finite and positive, got {}",
                self.postcondition_tolerance
            )));
        }
        if let Some(budget) = self.max_collateral_drift
            && (!budget.is_finite() || budget < 0.0)
        {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "max_collateral_drift must be finite and non-negative, got {budget}"
            )));
        }
        Ok(())
    }
}

/// A request to make the memory answer `value` for `key`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditRequest {
    /// The key to edit.
    pub key: EditKey,
    /// The value it must return.
    pub value: EditValue,
    /// Optional human-readable label, carried through to the [`EditRecord`].
    pub label: Option<String>,
    /// Pin the edit to a named memory site. `None` lets the editor **locate** the cheapest
    /// site itself — see [`KnowledgeEditor::locate`](super::KnowledgeEditor::locate).
    pub site: Option<usize>,
    /// Override the codebook's default deferral radius for this edit.
    pub radius: Option<f64>,
}

impl EditRequest {
    /// A request with no label, no pinned site, and the codebook's default radius.
    #[must_use]
    pub fn new(key: EditKey, value: EditValue) -> Self {
        Self {
            key,
            value,
            label: None,
            site: None,
            radius: None,
        }
    }

    /// Attach a label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Pin the edit to a memory site, skipping the locate step.
    #[must_use]
    pub fn with_site(mut self, site: usize) -> Self {
        self.site = Some(site);
        self
    }

    /// Override the deferral radius for this edit.
    #[must_use]
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = Some(radius);
        self
    }
}

/// Everything that is known about an edit after it has been applied and **verified**.
///
/// Every field is a measurement taken from the memory as it now stands, not a prediction the
/// solve made about itself — except [`predicted_rms_drift`](Self::predicted_rms_drift), which
/// is labelled as such precisely so it can be compared against
/// [`measured_rms_drift`](Self::measured_rms_drift).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditResult {
    /// The edit's codebook id.
    pub edit_id: EditId,
    /// Where the edit was materialized.
    pub scope: EditScope,
    /// The site the edit was located to, whether or not the parametric write was taken.
    pub site: usize,
    /// That site's label.
    pub site_label: String,
    /// Measured `||W' k* - v*||` after the write. Zero-ish by construction for a parametric
    /// edit; for a [`EditScope::CodebookOnly`] edit this is the residual the memory *still*
    /// has, since the memory was deliberately not touched — the codebook is what serves the
    /// edit.
    pub postcondition_residual: f64,
    /// Measured `max_j ||W' k_j - v_j||` over the parametric edits applied to this site
    /// *before* this one. `0.0` when this is the first.
    ///
    /// Under [`EditStrategy::Joint`] this stays at machine precision forever: the earlier
    /// constraints are re-solved along with the new one. Under [`EditStrategy::Sequential`] it
    /// grows, and watching it grow is the point.
    pub prior_edit_residual: f64,
    /// The exact `C`-weighted norm `||W' - W_0||_C` of the **cumulative** parametric delta,
    /// recomputed from the committed weights rather than read back from the solve. This is the
    /// upper bound on the root-mean-square collateral drift.
    pub weighted_delta_norm: f64,
    /// `sqrt(||Delta||_C^2 - ridge * ||Delta||_F^2)` — the root-mean-square drift the preserved
    /// keys *must* be suffering, derived analytically from the identity documented on
    /// [`weighted_frobenius_norm_sq_from_cholesky`](super::linalg::weighted_frobenius_norm_sq_from_cholesky).
    pub predicted_rms_drift: f64,
    /// The root-mean-square of `||(W' - W_0) k_i||` over the preserved keys the site retained,
    /// measured directly. `None` when the site was built from a caller-supplied covariance and
    /// so has no key set to measure against.
    pub measured_rms_drift: Option<f64>,
    /// `max_i ||(W' - W_0) k_i||` over the retained preserved keys.
    pub measured_max_drift: Option<f64>,
    /// Ridge the Gram-matrix Cholesky needed. `0.0` means the edit keys were comfortably
    /// linearly independent; anything else means they were not, and the edit is worth a second
    /// look even though its post-condition was verified.
    pub cholesky_jitter: f64,
    /// Number of parametric constraints now folded into this site — the rank of the cumulative
    /// delta.
    pub rank: usize,
    /// The deferral radius the codebook entry actually ended up with, **after** any shrink
    /// forced by a nearby conflicting edit.
    pub deferral_radius: f64,
}

/// What the system answers for one query key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditRead {
    /// The answer. Either an edited value served verbatim from the codebook, or the memory's
    /// own read `W' k`.
    pub value: Vec<f64>,
    /// Which of those two it was, and why.
    pub verdict: EditVerdict,
    /// The site consulted when the query was deferred to the parametric memory. `None` when
    /// the codebook served the query and the memory was never consulted.
    pub site: Option<usize>,
}

/// Errors raised by [`crate::knowledge_editing`].
#[derive(Debug, Error, Clone, PartialEq)]
pub enum KnowledgeEditError {
    /// The configuration was not usable.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// A key, value, or matrix did not have the expected shape.
    #[error("dimension mismatch: {what} has {actual} elements, expected {expected}")]
    DimensionMismatch {
        /// Which argument was malformed.
        what: &'static str,
        /// The length required.
        expected: usize,
        /// The length supplied.
        actual: usize,
    },
    /// A key or value contained a `NaN` or an infinity.
    #[error("non-finite value in {what}")]
    NonFinite {
        /// Which argument was non-finite.
        what: &'static str,
    },
    /// A key or value had no coordinates at all.
    #[error("{what} must have at least one coordinate")]
    EmptyVector {
        /// Which argument was empty.
        what: &'static str,
    },
    /// A dense kernel failed. See [`KnowledgeEditLinalgError`].
    #[error("linear algebra failure: {0}")]
    Linalg(#[from] KnowledgeEditLinalgError),
    /// A [`KnowledgeEditor`](super::KnowledgeEditor) was built with no memory sites, so there
    /// is nowhere to locate an edit to.
    #[error("no memory site is registered")]
    NoSites,
    /// A request named a site that does not exist.
    #[error("memory site {site} is out of range ({available} registered)")]
    UnknownSite {
        /// The site index requested.
        site: usize,
        /// How many sites exist.
        available: usize,
    },
    /// A retraction named an edit the codebook has never issued, or has already retracted.
    #[error("unknown edit id {0}")]
    UnknownEdit(EditId),
    /// The edit key is the zero vector (or has underflowed to it), so `k^T C^-1 k == 0`. No
    /// rank-1 update of `W` can change `W k` when `k == 0`, because `W k == 0` for every `W`.
    /// This is a genuine impossibility, not a numerical hiccup: `C^-1` is positive definite, so
    /// the denominator is strictly positive for *every* non-zero key.
    #[error(
        "degenerate edit key: k^T C^-1 k = {denominator}, which for a positive-definite C^-1 \
         means the key is the zero vector"
    )]
    DegenerateEditKey {
        /// The computed denominator.
        denominator: f64,
    },
    /// The memory did not satisfy `W' k* = v*` after the write, so the write was **rolled
    /// back**. The memory is exactly as it was before the call.
    #[error(
        "post-condition W' k* = v* violated: residual {residual} exceeds tolerance {tolerance}; \
         the write has been rolled back and the memory is unchanged"
    )]
    PostconditionViolated {
        /// The largest measured `||W' k - v||` over the constraints that had to hold.
        residual: f64,
        /// The tolerance it had to clear.
        tolerance: f64,
    },
    /// Reading or writing the codebook's `JSON` form failed.
    ///
    /// The cause is flattened to a `String` because `std::io::Error` is neither `Clone` nor
    /// `PartialEq`, and every other error in this enum is both.
    #[error("codebook persistence failure: {0}")]
    Persistence(String),
}
