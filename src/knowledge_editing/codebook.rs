//! [`EditCodebook`] — a persistent, **non-evicting** discrete edit memory, consulted at
//! inference inside a deferral radius.
//!
//! # What it is
//!
//! A `GRACE`-style codebook (Hartvigsen et al., 2023, *Aging with `GRACE`: Lifelong Model
//! Editing with Discrete Key-Value Adaptors*) is a list of `(key, value, radius)` entries. At
//! inference, a query key is compared against every entry; if it falls inside some entry's
//! radius the edited value is served verbatim, and otherwise the system **defers** to the base
//! model. It is the discrete, non-parametric half of knowledge editing: nothing is approximated,
//! nothing decays, and an edit costs the base model's behaviour exactly nothing outside its own
//! ball. The `SERAC` architecture (Mitchell et al., 2022) is the same shape — a scope classifier
//! deciding whether a query is "about" an edit, and a counterfactual model answering it if so.
//!
//! # The invariant this type exists to hold
//!
//! > **Every edit ever inserted is still *served* for its own key, forever, until it is
//! > explicitly retracted.**
//!
//! Not "still stored" — *served*. That is a much stronger statement, and it survives arbitrary
//! pressure: any number of lookups, any number of later edits, and any number of radius shrinks.
//! It is guaranteed structurally, not defensively: an entry's own key sits at distance `0` from
//! itself and every other entry is at some strictly positive distance, so on its own key an entry
//! always wins the nearest-entry tie-break, whatever its radius has shrunk to.
//!
//! There is no capacity, no `TTL`, and no eviction policy, because those things exist to *lose*
//! data and losing data here means a model silently reverting to a fact it was told to forget.
//! [`EditCodebook::retract`] is the only way an entry ever leaves, it must be asked for by id,
//! and it returns the record it removed.
//!
//! # Radius splitting, and why it does not violate the invariant
//!
//! Two edits with *different* values whose balls overlap would be ambiguous: a query in the
//! overlap could be answered either way. `GRACE` resolves this by shrinking, not by dropping —
//! when a new key lands close to an existing entry that asserts a different value, both radii are
//! pushed back to half the distance between the keys, so the balls become disjoint. After the
//! shrink the following holds for every pair of entries with different values:
//!
//! ```text
//! radius_i + radius_j <= distance(key_i, key_j)
//! ```
//!
//! and it keeps holding as later inserts shrink radii further, because shrinking only ever
//! *decreases* the left-hand side. Entries that assert the **same** value are never split apart:
//! a query in their overlap gets the same answer either way, so there is nothing to disambiguate.
//!
//! Shrinking narrows an edit's *generalization* — it will claim fewer paraphrases of its key —
//! but it can never cost the edit its own key. That is the trade `GRACE` makes, and it is the
//! right one: an edit that generalizes less is degraded, an edit that is evicted is *gone*.
//!
//! # Contrast with `semantic_cache`
//!
//! Structurally, an `LRU` semantic cache is also a threshold-gated key → value lookup on an
//! embedding. It is not this. A cache **memoizes**: its entries are recomputable, so evicting one
//! costs latency and nothing else, and evicting under pressure is the entire point of a capacity
//! bound. An edit is not recomputable. It is a statement about what the system now believes, and
//! evicting it does not cost latency — it silently restores the belief the edit was made to
//! replace. That asymmetry is why this type has no capacity parameter to get wrong.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::linalg::l2_distance;
use super::types::{
    CodebookStats, EditId, EditKey, EditRecord, EditValue, EditVerdict, IDENTICAL_KEY_TOLERANCE,
    IDENTICAL_VALUE_TOLERANCE, KnowledgeEditError,
};

/// A persistent, non-evicting discrete edit memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditCodebook {
    key_dim: usize,
    value_dim: usize,
    default_radius: f64,
    entries: Vec<EditRecord>,
    next_id: u64,
    stats: CodebookStats,
}

impl EditCodebook {
    /// A codebook for `key_dim`-vectors → `value_dim`-vectors, with a default deferral radius.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::InvalidConfig`] if a dimension is zero, or the radius is negative or
    /// not finite.
    pub fn new(
        key_dim: usize,
        value_dim: usize,
        default_radius: f64,
    ) -> Result<Self, KnowledgeEditError> {
        if key_dim == 0 || value_dim == 0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "codebook dimensions must be at least 1, got key_dim={key_dim} value_dim={value_dim}"
            )));
        }
        if !default_radius.is_finite() || default_radius < 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "deferral radius must be finite and non-negative, got {default_radius}"
            )));
        }
        Ok(Self {
            key_dim,
            value_dim,
            default_radius,
            entries: Vec::new(),
            next_id: 0,
            stats: CodebookStats::default(),
        })
    }

    /// Record an edit at the codebook's default radius.
    ///
    /// # Errors
    ///
    /// As [`Self::insert_with_radius`].
    pub fn insert(
        &mut self,
        key: EditKey,
        value: EditValue,
        label: Option<String>,
    ) -> Result<EditId, KnowledgeEditError> {
        let radius = self.default_radius;
        self.insert_with_radius(key, value, label, radius)
    }

    /// Record an edit at an explicit radius.
    ///
    /// If an entry already exists at this key (within [`IDENTICAL_KEY_TOLERANCE`]) the insert is
    /// a **revision**: the existing entry's value is overwritten in place, its
    /// [`revisions`](EditRecord::revisions) counter is bumped, and its id is returned. That is an
    /// overwrite the caller explicitly asked for by re-asserting the same key — the entry, its
    /// id, and its key all survive, so the non-eviction invariant is untouched. (Handling it this
    /// way is also what makes the invariant *possible*: two entries at the same key with
    /// different values could not both be served, and one of them would be silently dead.)
    ///
    /// Otherwise a new entry is created, and its radius — together with those of any existing
    /// entries asserting a *different* value whose balls it would overlap — is shrunk until the
    /// balls are disjoint. See the [module documentation](self).
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::DimensionMismatch`] if the key or value width is wrong.
    /// * [`KnowledgeEditError::InvalidConfig`] if `radius` is negative or not finite.
    pub fn insert_with_radius(
        &mut self,
        key: EditKey,
        value: EditValue,
        label: Option<String>,
        radius: f64,
    ) -> Result<EditId, KnowledgeEditError> {
        self.check_key(key.as_slice())?;
        if value.dim() != self.value_dim {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "codebook edit value",
                expected: self.value_dim,
                actual: value.dim(),
            });
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(KnowledgeEditError::InvalidConfig(format!(
                "deferral radius must be finite and non-negative, got {radius}"
            )));
        }

        // Revision path: an entry already stands at this key.
        for entry in &mut self.entries {
            let distance = l2_distance(entry.key.as_slice(), key.as_slice())?;
            if distance <= IDENTICAL_KEY_TOLERANCE {
                entry.value = value;
                entry.revisions += 1;
                if let Some(label) = label {
                    entry.label = Some(label);
                }
                self.stats.revisions += 1;
                return Ok(entry.id);
            }
        }

        // Read-only pass: distances to every entry that asserts a *different* value. Done before
        // any mutation so that an error cannot leave the codebook half-shrunk.
        let mut distances: Vec<Option<f64>> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if values_agree(&entry.value, &value) {
                distances.push(None);
            } else {
                distances.push(Some(l2_distance(entry.key.as_slice(), key.as_slice())?));
            }
        }

        // Write pass: push overlapping balls apart. Processing entries in order is sound — a
        // later shrink of `radius` only *decreases* it, so a pair already made disjoint stays
        // disjoint.
        let mut radius = radius;
        for (entry, distance) in self.entries.iter_mut().zip(&distances) {
            let Some(distance) = *distance else { continue };
            if entry.radius + radius > distance {
                let half = 0.5 * distance;
                if radius > half {
                    radius = half;
                    self.stats.radius_shrinks += 1;
                }
                if entry.radius > half {
                    entry.radius = half;
                    self.stats.radius_shrinks += 1;
                }
            }
        }

        let id = EditId(self.next_id);
        self.next_id += 1;
        self.entries.push(EditRecord {
            id,
            key,
            value,
            radius,
            initial_radius: radius,
            label,
            revisions: 0,
        });
        self.stats.inserts += 1;
        Ok(id)
    }

    /// Consult the codebook for a query key.
    ///
    /// Among the entries whose ball contains the query, the **nearest** wins; ties go to the
    /// earlier insert, which makes the answer deterministic. Because an entry's own key is at
    /// distance `0` from itself and every other entry is strictly further away, an entry always
    /// wins for its own key — which is the non-eviction invariant, in the only form that matters.
    ///
    /// The boundary is **inclusive**: a query at distance exactly `radius` is served.
    ///
    /// Takes `&mut self` in order to count the lookup. Note that
    /// `semantic_cache`'s `LRU` also needs `&mut self` to read — but it needs it in order
    /// to *reorder and evict*. This one only needs it to keep score.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::DimensionMismatch`] if the query width is wrong;
    /// [`KnowledgeEditError::NonFinite`] if it contains a `NaN` or an infinity.
    pub fn lookup(&mut self, query: &[f64]) -> Result<EditVerdict, KnowledgeEditError> {
        self.check_key(query)?;
        self.stats.lookups += 1;

        let mut claimed: Option<(usize, f64)> = None;
        let mut nearest: Option<(usize, f64)> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let distance = l2_distance(entry.key.as_slice(), query)?;
            if nearest.is_none_or(|(_, best)| distance < best) {
                nearest = Some((index, distance));
            }
            if distance <= entry.radius && claimed.is_none_or(|(_, best)| distance < best) {
                claimed = Some((index, distance));
            }
        }

        if let Some((index, distance)) = claimed {
            self.stats.served += 1;
            let entry = &self.entries[index];
            Ok(EditVerdict::Served {
                edit_id: entry.id,
                value: entry.value.clone(),
                distance,
                radius: entry.radius,
            })
        } else {
            self.stats.deferred += 1;
            Ok(EditVerdict::Deferred {
                nearest: nearest.map(|(index, _)| self.entries[index].id),
                nearest_distance: nearest.map(|(_, distance)| distance),
            })
        }
    }

    /// Remove an edit **explicitly**, returning the record. The only path by which an entry ever
    /// leaves a codebook.
    ///
    /// Radii that were shrunk to make room for this edit are deliberately *not* re-grown: a
    /// retraction must never silently widen some *other* edit's scope over keys it was never
    /// asked about.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::UnknownEdit`] if no entry has that id — including if it was already
    /// retracted. Ids are never reused, so this is unambiguous.
    pub fn retract(&mut self, id: EditId) -> Result<EditRecord, KnowledgeEditError> {
        let position = self
            .entries
            .iter()
            .position(|entry| entry.id == id)
            .ok_or(KnowledgeEditError::UnknownEdit(id))?;
        self.stats.retractions += 1;
        Ok(self.entries.remove(position))
    }

    /// The record with this id.
    #[must_use]
    pub fn get(&self, id: EditId) -> Option<&EditRecord> {
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// Every record, in insertion order.
    #[must_use]
    pub fn records(&self) -> &[EditRecord] {
        &self.entries
    }

    /// How many edits are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Key width.
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    /// Value width.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    /// The radius a new entry gets unless the caller overrides it.
    #[must_use]
    pub fn default_radius(&self) -> f64 {
        self.default_radius
    }

    /// Counters describing the pressure this codebook has been under.
    ///
    /// `stats().inserts - stats().retractions == len()` after *any* sequence of operations. That
    /// equation is the non-eviction guarantee, arithmetically.
    #[must_use]
    pub fn stats(&self) -> CodebookStats {
        self.stats
    }

    /// Serialize the codebook to `JSON`.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Persistence`] if serialization fails.
    pub fn to_json(&self) -> Result<String, KnowledgeEditError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| KnowledgeEditError::Persistence(error.to_string()))
    }

    /// Restore a codebook from its `JSON` form.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Persistence`] if the text is not a codebook.
    pub fn from_json(text: &str) -> Result<Self, KnowledgeEditError> {
        serde_json::from_str(text)
            .map_err(|error| KnowledgeEditError::Persistence(error.to_string()))
    }

    /// Write the codebook to a file. Edits outlive the process that made them, which is what
    /// "persistent" has to mean for a memory whose entries are not recomputable.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Persistence`] if serialization or the write fails.
    pub fn save_json(&self, path: &Path) -> Result<(), KnowledgeEditError> {
        let text = self.to_json()?;
        fs::write(path, text).map_err(|error| {
            KnowledgeEditError::Persistence(format!("writing {}: {error}", path.display()))
        })
    }

    /// Read a codebook back from a file.
    ///
    /// The round trip preserves every id, every radius (including shrunk ones), and every
    /// revision count, so a reloaded codebook serves exactly what the saved one served.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Persistence`] if the read or the parse fails.
    pub fn load_json(path: &Path) -> Result<Self, KnowledgeEditError> {
        let text = fs::read_to_string(path).map_err(|error| {
            KnowledgeEditError::Persistence(format!("reading {}: {error}", path.display()))
        })?;
        Self::from_json(&text)
    }

    fn check_key(&self, key: &[f64]) -> Result<(), KnowledgeEditError> {
        if key.len() != self.key_dim {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "codebook key",
                expected: self.key_dim,
                actual: key.len(),
            });
        }
        if key.iter().any(|v| !v.is_finite()) {
            return Err(KnowledgeEditError::NonFinite {
                what: "codebook key",
            });
        }
        Ok(())
    }
}

/// Whether two edits assert the same value, and therefore need not be split apart.
fn values_agree(left: &EditValue, right: &EditValue) -> bool {
    if left.dim() != right.dim() {
        return false;
    }
    l2_distance(left.as_slice(), right.as_slice()).is_ok_and(|d| d <= IDENTICAL_VALUE_TOLERANCE)
}
