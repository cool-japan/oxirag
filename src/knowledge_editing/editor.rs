//! [`KnowledgeEditor`] — locate a site, decide whether the parametric write is worth its
//! collateral cost, take it, **verify it**, and record the edit somewhere it can never be lost.

use serde::{Deserialize, Serialize};

use super::codebook::EditCodebook;
use super::linalg::l2_norm;
use super::memory::EditableMemory;
use super::rank_one::{EditBatch, RankOneEdit};
use super::types::{
    EditConfig, EditId, EditKey, EditRead, EditRecord, EditRequest, EditResult, EditScope,
    EditStrategy, EditValue, KnowledgeEditError,
};

/// A located, radius-resolved edit request, ready to be tried.
#[derive(Debug, Clone)]
struct EditPlan {
    site: usize,
    key: EditKey,
    value: EditValue,
    label: Option<String>,
    radius: f64,
}

/// The locate-then-edit engine.
///
/// # What one `apply` actually does
///
/// 1. **Locate.** Unless the request pins a site, every site is trial-solved and the one at which
///    this edit would leave the smallest cumulative `||W' - W_0||_C` wins. That criterion is not
///    a heuristic — it is the exact collateral cost of the edit, and it is small exactly where
///    the edit key is *rare* under the preserved key distribution and the memory is *already
///    close* to the target value.
/// 2. **Admit, or defer.** If [`EditConfig::max_collateral_drift`] is set and the parametric
///    write would push the cumulative cost past it, the write is **declined**. The edit is not
///    dropped — it goes to the codebook, where it is served exactly and drifts nothing. This is
///    the deferral memory doing real work, not decoration.
/// 3. **Write.** Either the `MEMIT` joint re-solve of every accumulated constraint against `W_0`,
///    or the `ROME` rank-1 update of the current weights — see [`EditStrategy`].
/// 4. **Verify.** The post-condition `||W' k* - v*||` is *measured* against the trial weights. If
///    it exceeds [`EditConfig::postcondition_tolerance`] the write is discarded and the call
///    fails with [`KnowledgeEditError::PostconditionViolated`]. **Nothing is committed until the
///    verification passes**, so a failed edit leaves the memory bit-for-bit as it was.
/// 5. **Record.** The edit enters the codebook, and every number in the returned [`EditResult`]
///    is read back out of the committed memory rather than reported by the solve about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeEditor {
    config: EditConfig,
    sites: Vec<EditableMemory>,
    codebook: EditCodebook,
    batches: Vec<EditBatch>,
    batch_ids: Vec<Vec<EditId>>,
    history: Vec<EditResult>,
}

impl KnowledgeEditor {
    /// Build an editor over one or more memory sites.
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::InvalidConfig`] if the configuration is unusable.
    /// * [`KnowledgeEditError::NoSites`] if `sites` is empty.
    /// * [`KnowledgeEditError::DimensionMismatch`] if a site's shape disagrees with the
    ///   configured key and value widths.
    pub fn new(config: EditConfig, sites: Vec<EditableMemory>) -> Result<Self, KnowledgeEditError> {
        config.validate()?;
        if sites.is_empty() {
            return Err(KnowledgeEditError::NoSites);
        }
        for site in &sites {
            if site.cols() != config.key_dim {
                return Err(KnowledgeEditError::DimensionMismatch {
                    what: "memory site key width",
                    expected: config.key_dim,
                    actual: site.cols(),
                });
            }
            if site.rows() != config.value_dim {
                return Err(KnowledgeEditError::DimensionMismatch {
                    what: "memory site value width",
                    expected: config.value_dim,
                    actual: site.rows(),
                });
            }
        }
        let codebook = EditCodebook::new(config.key_dim, config.value_dim, config.deferral_radius)?;
        let count = sites.len();
        Ok(Self {
            config,
            sites,
            codebook,
            batches: vec![EditBatch::new(); count],
            batch_ids: vec![Vec::new(); count],
            history: Vec::new(),
        })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &EditConfig {
        &self.config
    }

    /// Every memory site.
    #[must_use]
    pub fn sites(&self) -> &[EditableMemory] {
        &self.sites
    }

    /// One memory site.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::UnknownSite`] if the index is out of range.
    pub fn site(&self, index: usize) -> Result<&EditableMemory, KnowledgeEditError> {
        self.sites
            .get(index)
            .ok_or(KnowledgeEditError::UnknownSite {
                site: index,
                available: self.sites.len(),
            })
    }

    /// The non-evicting edit codebook.
    #[must_use]
    pub fn codebook(&self) -> &EditCodebook {
        &self.codebook
    }

    /// The parametric constraints currently folded into a site.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::UnknownSite`] if the index is out of range.
    pub fn constraints(&self, site: usize) -> Result<&EditBatch, KnowledgeEditError> {
        self.batches
            .get(site)
            .ok_or(KnowledgeEditError::UnknownSite {
                site,
                available: self.sites.len(),
            })
    }

    /// Every edit applied so far, in order.
    #[must_use]
    pub fn history(&self) -> &[EditResult] {
        &self.history
    }

    /// The site at which this edit would leave the smallest cumulative `||W' - W_0||_C`.
    ///
    /// The trial solve is performed at every site and nothing is mutated.
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::NoSites`] if there are none.
    /// * [`KnowledgeEditError::DegenerateEditKey`] if *every* site rejects the key — which, since
    ///   the denominator `k^T C^-1 k` is strictly positive for every non-zero key, means the key
    ///   is zero.
    /// * [`KnowledgeEditError::DimensionMismatch`] / [`KnowledgeEditError::Linalg`] on a
    ///   malformed key or value.
    pub fn locate(&self, key: &EditKey, value: &EditValue) -> Result<usize, KnowledgeEditError> {
        if self.sites.is_empty() {
            return Err(KnowledgeEditError::NoSites);
        }
        let mut best: Option<(usize, f64)> = None;
        let mut degenerate: Option<KnowledgeEditError> = None;

        for index in 0..self.sites.len() {
            let cost = match self.config.strategy {
                EditStrategy::Joint => {
                    let mut candidate = self.batches[index].clone();
                    match candidate.push(key.clone(), value.clone()) {
                        Ok(()) => self.materialize_joint(index, &candidate).map(|(_, c, _)| c),
                        Err(error) => Err(error),
                    }
                }
                EditStrategy::Sequential => self
                    .materialize_sequential(index, self.site(index)?.weights(), key, value)
                    .map(|(_, cost)| cost),
            };
            match cost {
                Ok(cost) => {
                    if best.is_none_or(|(_, incumbent)| cost < incumbent) {
                        best = Some((index, cost));
                    }
                }
                Err(error @ KnowledgeEditError::DegenerateEditKey { .. }) => {
                    degenerate = Some(error);
                }
                Err(other) => return Err(other),
            }
        }

        match (best, degenerate) {
            (Some((index, _)), _) => Ok(index),
            (None, Some(error)) => Err(error),
            (None, None) => Err(KnowledgeEditError::NoSites),
        }
    }

    /// Apply one edit.
    ///
    /// # Errors
    ///
    /// As [`Self::apply_batch`].
    ///
    pub fn apply(&mut self, request: &EditRequest) -> Result<EditResult, KnowledgeEditError> {
        let mut results = self.apply_batch(std::slice::from_ref(request))?;
        match results.pop() {
            Some(result) => Ok(result),
            None => Err(KnowledgeEditError::InvalidConfig(
                "apply_batch returned no result for a single request".to_string(),
            )),
        }
    }

    /// Apply many edits at once.
    ///
    /// Under [`EditStrategy::Joint`] every request in the batch, *and* every edit already applied
    /// to the same site, becomes a constraint of one joint solve — so they all hold exactly, and
    /// the cumulative delta is the minimum-`C`-norm matrix that makes them hold. That is the
    /// `MEMIT` construction, and it is the only reason a *sequence* of edits does not slowly
    /// destroy the edits that came before it.
    ///
    /// Nothing is committed until every enforced post-condition has been verified against trial
    /// weights: an error leaves the editor exactly as it was.
    ///
    /// # Errors
    ///
    /// * [`KnowledgeEditError::DimensionMismatch`] if a key or value has the wrong width.
    /// * [`KnowledgeEditError::UnknownSite`] if a request pins a site that does not exist.
    /// * [`KnowledgeEditError::DegenerateEditKey`] if an edit key is the zero vector.
    /// * [`KnowledgeEditError::PostconditionViolated`] if a write that was supposed to install
    ///   `W' k = v` did not — in which case **the write is rolled back**.
    /// * [`KnowledgeEditError::Linalg`] if a solve overflows or the Gram matrix of the edit keys
    ///   is so degenerate it will not factor at all.
    pub fn apply_batch(
        &mut self,
        requests: &[EditRequest],
    ) -> Result<Vec<EditResult>, KnowledgeEditError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if self.sites.is_empty() {
            return Err(KnowledgeEditError::NoSites);
        }

        // ── Phase 1 — validate and locate. Nothing is mutated.
        let mut plans: Vec<EditPlan> = Vec::with_capacity(requests.len());
        for request in requests {
            self.check_request(request)?;
            let site = match request.site {
                Some(site) => {
                    self.site(site)?;
                    site
                }
                None => self.locate(&request.key, &request.value)?,
            };
            let radius = match request.radius {
                Some(radius) => radius,
                None => self.config.deferral_radius,
            };
            if !radius.is_finite() || radius < 0.0 {
                return Err(KnowledgeEditError::InvalidConfig(format!(
                    "deferral radius must be finite and non-negative, got {radius}"
                )));
            }
            plans.push(EditPlan {
                site,
                key: request.key.clone(),
                value: request.value.clone(),
                label: request.label.clone(),
                radius,
            });
        }

        // ── Phase 2 — per-site trial write. Still nothing mutated.
        let site_count = self.sites.len();
        let mut trial_batches = self.batches.clone();
        let mut trial_weights: Vec<Option<Vec<f64>>> = vec![None; site_count];
        let mut trial_jitter = vec![0.0; site_count];
        let mut accepted_by_site: Vec<Vec<usize>> = vec![Vec::new(); site_count];
        let mut parametric = vec![false; plans.len()];

        for site in 0..site_count {
            let group: Vec<usize> = plans
                .iter()
                .enumerate()
                .filter(|(_, plan)| plan.site == site)
                .map(|(index, _)| index)
                .collect();
            if group.is_empty() {
                continue;
            }

            let (batch, weights, jitter) = match self.config.strategy {
                EditStrategy::Joint => self.trial_joint(
                    site,
                    &trial_batches[site],
                    &plans,
                    &group,
                    &mut accepted_by_site[site],
                )?,
                EditStrategy::Sequential => self.trial_sequential(
                    site,
                    &trial_batches[site],
                    &plans,
                    &group,
                    &mut accepted_by_site[site],
                )?,
            };
            for &index in &accepted_by_site[site] {
                parametric[index] = true;
            }
            trial_batches[site] = batch;
            trial_weights[site] = Some(weights);
            trial_jitter[site] = jitter;
        }

        // ── Phase 3 — verify against the trial weights. A failure here returns without having
        //    touched a single byte of the editor's state.
        for site in 0..site_count {
            if accepted_by_site[site].is_empty() {
                continue;
            }
            let Some(weights) = trial_weights[site].as_ref() else {
                continue;
            };
            let batch = &trial_batches[site];
            // `Joint` promises every constraint holds. `Sequential` promises only the last one it
            // wrote: each rank-1 update is solved against the weights of the moment, so a later
            // edit in the same call perturbs an earlier one. That is the documented failure mode
            // of sequential editing, it is *measured* in `EditResult::prior_edit_residual`, and
            // pretending to enforce it here would be a lie.
            let enforce_from = match self.config.strategy {
                EditStrategy::Joint => 0,
                EditStrategy::Sequential => batch.len().saturating_sub(1),
            };
            self.verify_site(site, batch, weights, enforce_from)?;
        }

        // ── Phase 4 — commit. The codebook first, so that every edit is durably recorded before
        //    any weight moves; then the weights.
        let ids = self.commit(
            &plans,
            &mut trial_batches,
            &mut trial_weights,
            &accepted_by_site,
        )?;

        // ── Phase 5 — measure. Every number below is read back out of the committed memory.
        let mut results = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            results.push(self.build_result(index, plan, &ids, &parametric, &trial_jitter)?);
        }

        self.history.extend(results.iter().cloned());
        Ok(results)
    }

    /// Phase 4 of [`Self::apply_batch`]: durably record every edit in the codebook, then install
    /// the accepted trial weights. Returns the codebook id of each plan, in plan order.
    fn commit(
        &mut self,
        plans: &[EditPlan],
        trial_batches: &mut [EditBatch],
        trial_weights: &mut [Option<Vec<f64>>],
        accepted_by_site: &[Vec<usize>],
    ) -> Result<Vec<EditId>, KnowledgeEditError> {
        let mut ids: Vec<EditId> = Vec::with_capacity(plans.len());
        for plan in plans {
            let id = self.codebook.insert_with_radius(
                plan.key.clone(),
                plan.value.clone(),
                plan.label.clone(),
                plan.radius,
            )?;
            ids.push(id);
        }
        for site in 0..self.sites.len() {
            if accepted_by_site[site].is_empty() {
                continue;
            }
            if let Some(weights) = trial_weights[site].take() {
                self.sites[site].set_weights(weights);
            }
            self.batches[site] = trial_batches[site].clone();
            for &index in &accepted_by_site[site] {
                self.batch_ids[site].push(ids[index]);
            }
        }
        Ok(ids)
    }

    /// Phase 5 of [`Self::apply_batch`]: read every reported number back out of the committed
    /// memory for one plan, rather than trusting whatever the solve claimed about itself.
    fn build_result(
        &self,
        index: usize,
        plan: &EditPlan,
        ids: &[EditId],
        parametric: &[bool],
        trial_jitter: &[f64],
    ) -> Result<EditResult, KnowledgeEditError> {
        let site = plan.site;
        let memory = self.site(site)?;
        let read = memory.read(plan.key.as_slice())?;
        let postcondition_residual = residual_norm(&read, plan.value.as_slice());

        let batch = &self.batches[site];
        let position = if parametric[index] {
            self.batch_ids[site]
                .iter()
                .position(|id| *id == ids[index])
                .unwrap_or(batch.len())
        } else {
            batch.len()
        };
        let mut prior_edit_residual: f64 = 0.0;
        for (key, value) in batch.keys().iter().zip(batch.values()).take(position) {
            let read = memory.read(key.as_slice())?;
            prior_edit_residual = prior_edit_residual.max(residual_norm(&read, value.as_slice()));
        }

        let weighted_delta_norm = memory.cumulative_weighted_norm()?;
        let frobenius = memory.cumulative_frobenius_norm();
        let predicted_rms_drift = (weighted_delta_norm * weighted_delta_norm
            - memory.ridge() * frobenius * frobenius)
            .max(0.0)
            .sqrt();
        let drift = memory.drift_report()?;

        Ok(EditResult {
            edit_id: ids[index],
            scope: if parametric[index] {
                EditScope::Parametric { site }
            } else {
                EditScope::CodebookOnly
            },
            site,
            site_label: memory.label().to_string(),
            postcondition_residual,
            prior_edit_residual,
            weighted_delta_norm,
            predicted_rms_drift,
            measured_rms_drift: drift.map(|(rms, _)| rms),
            measured_max_drift: drift.map(|(_, maximum)| maximum),
            cholesky_jitter: trial_jitter[site],
            rank: batch.len(),
            deferral_radius: self
                .codebook
                .get(ids[index])
                .map_or(plan.radius, |record| record.radius),
        })
    }

    /// Answer a query key: the codebook first, the parametric memory at `site` if no edit claims
    /// it.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::UnknownSite`] if the site does not exist;
    /// [`KnowledgeEditError::DimensionMismatch`] or [`KnowledgeEditError::NonFinite`] on a
    /// malformed query.
    pub fn read(&mut self, site: usize, key: &[f64]) -> Result<EditRead, KnowledgeEditError> {
        self.site(site)?;
        let verdict = self.codebook.lookup(key)?;
        let served = verdict
            .served_value()
            .map(|value| value.as_slice().to_vec());
        if let Some(value) = served {
            Ok(EditRead {
                value,
                verdict,
                site: None,
            })
        } else {
            let value = self.site(site)?.read(key)?;
            Ok(EditRead {
                value,
                verdict,
                site: Some(site),
            })
        }
    }

    /// Measure `||W' k - v||` for every parametric constraint, at every site.
    ///
    /// An auditor: it recomputes, from the memory as it now stands, whether each edit is *still*
    /// installed. Under [`EditStrategy::Joint`] every residual stays at machine precision no
    /// matter how many edits follow. Under [`EditStrategy::Sequential`] they grow, and this is
    /// how you watch them grow.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::Linalg`] if a readout overflows.
    pub fn verify(&self) -> Result<Vec<(EditId, f64)>, KnowledgeEditError> {
        let mut residuals = Vec::new();
        for (site, batch) in self.batches.iter().enumerate() {
            let memory = self.site(site)?;
            for (position, (key, value)) in batch.keys().iter().zip(batch.values()).enumerate() {
                let read = memory.read(key.as_slice())?;
                let id = match self.batch_ids[site].get(position) {
                    Some(id) => *id,
                    None => continue,
                };
                residuals.push((id, residual_norm(&read, value.as_slice())));
            }
        }
        Ok(residuals)
    }

    /// Retract an edit **explicitly**: remove it from the codebook, drop it as a parametric
    /// constraint, and rebuild the memory from `W_0` without it.
    ///
    /// This is the only way an edit ever leaves the system, and it is nothing like an eviction:
    /// it is asked for by id, it rebuilds the memory so the retracted fact really is gone rather
    /// than merely unserved, and it hands the caller back the record it removed.
    ///
    /// # Errors
    ///
    /// [`KnowledgeEditError::UnknownEdit`] if no such edit exists;
    /// [`KnowledgeEditError::PostconditionViolated`] if the rebuilt memory no longer satisfies
    /// the *surviving* constraints, in which case nothing is changed.
    pub fn retract(&mut self, id: EditId) -> Result<EditRecord, KnowledgeEditError> {
        if self.codebook.get(id).is_none() {
            return Err(KnowledgeEditError::UnknownEdit(id));
        }

        let mut located: Option<(usize, usize)> = None;
        for (site, ids) in self.batch_ids.iter().enumerate() {
            if let Some(position) = ids.iter().position(|candidate| *candidate == id) {
                located = Some((site, position));
                break;
            }
        }

        if let Some((site, position)) = located {
            let mut batch = self.batches[site].clone();
            batch.remove(position)?;
            let (weights, _) = self.replay(site, &batch)?;
            let enforce_from = match self.config.strategy {
                EditStrategy::Joint => 0,
                EditStrategy::Sequential => batch.len().saturating_sub(1),
            };
            if !batch.is_empty() {
                self.verify_site(site, &batch, &weights, enforce_from)?;
            }
            self.sites[site].set_weights(weights);
            self.batches[site] = batch;
            self.batch_ids[site].remove(position);
        }

        self.codebook.retract(id)
    }

    /// Solve `batch` jointly against a site's base weights, returning the trial weights, their
    /// exact cumulative cost `||W - W_0||_C`, and the Gram-matrix jitter.
    fn materialize_joint(
        &self,
        site: usize,
        batch: &EditBatch,
    ) -> Result<(Vec<f64>, f64, f64), KnowledgeEditError> {
        let memory = self.site(site)?;
        let multi = batch.solve(memory)?;
        let mut weights = memory.base_weights().to_vec();
        multi.apply(&mut weights, memory.rows(), memory.cols())?;
        let cost = self.cost_of(site, &weights)?;
        Ok((weights, cost, multi.jitter()))
    }

    /// Apply one rank-1 edit on top of `weights`, returning the trial weights and their exact
    /// cumulative cost `||W - W_0||_C`.
    fn materialize_sequential(
        &self,
        site: usize,
        weights: &[f64],
        key: &EditKey,
        value: &EditValue,
    ) -> Result<(Vec<f64>, f64), KnowledgeEditError> {
        let memory = self.site(site)?;
        let edit = RankOneEdit::solve_against(memory, weights, key, value)?;
        let mut trial = weights.to_vec();
        edit.apply(&mut trial, memory.rows(), memory.cols())?;
        let cost = self.cost_of(site, &trial)?;
        Ok((trial, cost))
    }

    /// `||W - W_0||_C` for arbitrary weights at a site, recomputed from the weights themselves
    /// rather than read back from whatever solve produced them.
    fn cost_of(&self, site: usize, weights: &[f64]) -> Result<f64, KnowledgeEditError> {
        let memory = self.site(site)?;
        let delta: Vec<f64> = weights
            .iter()
            .zip(memory.base_weights())
            .map(|(w, b)| w - b)
            .collect();
        Ok(memory.weighted_norm_of(&delta)?)
    }

    /// Rebuild a site's weights from `W_0` under the configured strategy.
    fn replay(
        &self,
        site: usize,
        batch: &EditBatch,
    ) -> Result<(Vec<f64>, f64), KnowledgeEditError> {
        match self.config.strategy {
            EditStrategy::Joint => {
                let (weights, _, jitter) = self.materialize_joint(site, batch)?;
                Ok((weights, jitter))
            }
            EditStrategy::Sequential => {
                let memory = self.site(site)?;
                let mut weights = memory.base_weights().to_vec();
                for (key, value) in batch.keys().iter().zip(batch.values()) {
                    let edit = RankOneEdit::solve_against(memory, &weights, key, value)?;
                    edit.apply(&mut weights, memory.rows(), memory.cols())?;
                }
                Ok((weights, 0.0))
            }
        }
    }

    /// Greedy parametric admission under [`EditStrategy::Joint`].
    fn trial_joint(
        &self,
        site: usize,
        current: &EditBatch,
        plans: &[EditPlan],
        group: &[usize],
        accepted: &mut Vec<usize>,
    ) -> Result<(EditBatch, Vec<f64>, f64), KnowledgeEditError> {
        let mut batch = current.clone();
        match self.config.max_collateral_drift {
            None => {
                for &index in group {
                    batch.push(plans[index].key.clone(), plans[index].value.clone())?;
                    accepted.push(index);
                }
            }
            Some(budget) => {
                for &index in group {
                    let mut candidate = batch.clone();
                    candidate.push(plans[index].key.clone(), plans[index].value.clone())?;
                    let (_, cost, _) = self.materialize_joint(site, &candidate)?;
                    if cost > budget {
                        continue;
                    }
                    batch = candidate;
                    accepted.push(index);
                }
            }
        }
        if accepted.is_empty() {
            return Ok((batch, self.site(site)?.weights().to_vec(), 0.0));
        }
        let (weights, _, jitter) = self.materialize_joint(site, &batch)?;
        Ok((batch, weights, jitter))
    }

    /// Greedy parametric admission under [`EditStrategy::Sequential`].
    fn trial_sequential(
        &self,
        site: usize,
        current: &EditBatch,
        plans: &[EditPlan],
        group: &[usize],
        accepted: &mut Vec<usize>,
    ) -> Result<(EditBatch, Vec<f64>, f64), KnowledgeEditError> {
        let mut batch = current.clone();
        let mut weights = self.site(site)?.weights().to_vec();
        for &index in group {
            let plan = &plans[index];
            let (trial, cost) =
                self.materialize_sequential(site, &weights, &plan.key, &plan.value)?;
            if self
                .config
                .max_collateral_drift
                .is_some_and(|budget| cost > budget)
            {
                continue;
            }
            weights = trial;
            batch.push(plan.key.clone(), plan.value.clone())?;
            accepted.push(index);
        }
        Ok((batch, weights, 0.0))
    }

    /// Assert that constraints `enforce_from..` of `batch` hold against `weights`.
    fn verify_site(
        &self,
        site: usize,
        batch: &EditBatch,
        weights: &[f64],
        enforce_from: usize,
    ) -> Result<(), KnowledgeEditError> {
        let memory = self.site(site)?;
        for (key, value) in batch.keys().iter().zip(batch.values()).skip(enforce_from) {
            let read = memory.read_with(weights, key.as_slice())?;
            let residual = residual_norm(&read, value.as_slice());
            if !residual.is_finite() || residual > self.config.postcondition_tolerance {
                return Err(KnowledgeEditError::PostconditionViolated {
                    residual,
                    tolerance: self.config.postcondition_tolerance,
                });
            }
        }
        Ok(())
    }

    fn check_request(&self, request: &EditRequest) -> Result<(), KnowledgeEditError> {
        if request.key.dim() != self.config.key_dim {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "edit key",
                expected: self.config.key_dim,
                actual: request.key.dim(),
            });
        }
        if request.value.dim() != self.config.value_dim {
            return Err(KnowledgeEditError::DimensionMismatch {
                what: "edit value",
                expected: self.config.value_dim,
                actual: request.value.dim(),
            });
        }
        Ok(())
    }
}

/// `|| read - target ||`.
fn residual_norm(read: &[f64], target: &[f64]) -> f64 {
    let difference: Vec<f64> = read
        .iter()
        .zip(target)
        .map(|(left, right)| left - right)
        .collect();
    l2_norm(&difference)
}
