//! A deterministic click-log simulator: draw clicks from *known* ground-truth
//! parameters, then check that a click model can invert them.
//!
//! This is not a toy. Parameter recovery on synthetic data is the **only**
//! honest correctness test for an EM derivation: a real click log has no ground
//! truth to compare against, so a subtly wrong E-step produces plausible,
//! confidently-wrong numbers and nothing anywhere complains. Simulate from a
//! known `γ` and `α`, run EM, and demand the true values back — that is a test
//! that a broken derivation cannot pass.
//!
//! The simulator also records the ground-truth **examination** indicator on every
//! session ([`ClickSession::examinations`]), which no real log has and which is
//! exactly what makes the fully doubly-robust branch of
//! [`crate::click_model::DoublyRobustEstimator`] testable.
//!
//! # Randomness
//!
//! There is no `rand` dependency here. [`ClickSplitMix64`] is a hand-rolled
//! `SplitMix64` generator — the same finalising mix that seeds `rand`'s own
//! `SmallRng` — so every simulated log is a pure function of its seed and
//! reproduces byte-for-byte on every platform.

use super::types::{ClickLog, ClickModelError, ClickModelResult, ClickSession};

// ── ClickSplitMix64 ──────────────────────────────────────────────────────────

/// A deterministic `SplitMix64` pseudo-random generator.
///
/// ```text
/// z  = (state += 0x9E3779B97F4A7C15)
/// z  = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
/// z  = (z ^ (z >> 27)) * 0x94D049BB133111EB
/// z ^= z >> 31
/// ```
///
/// Fast, statistically sound for simulation, seeded by one `u64`, and — the
/// point — entirely reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickSplitMix64 {
    state: u64,
}

impl ClickSplitMix64 {
    /// A generator seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 raw bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform draw from `[0, 1)`, using the top 53 bits — exactly the width of
    /// an `f64` mantissa, so the conversion below is **exact**, not lossy.
    pub fn next_f64(&mut self) -> f64 {
        /// `2^53`, written as a literal so no integer→float cast is needed for it.
        const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
        // `>> 11` keeps the top 53 bits, and 53 is exactly the width of an `f64`
        // mantissa — so the cast below is *exact*, not lossy, whatever
        // `cast_precision_loss` assumes about `u64` in general.
        #[allow(clippy::cast_precision_loss)]
        let mantissa = (self.next_u64() >> 11) as f64;
        mantissa / TWO_POW_53
    }

    /// A Bernoulli draw with success probability `probability` (values outside
    /// `[0, 1]` are clamped).
    pub fn bernoulli(&mut self, probability: f64) -> bool {
        self.next_f64() < probability.clamp(0.0, 1.0)
    }

    /// A uniform integer in `[0, bound)`; `0` when `bound == 0`.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let value = (self.next_u64() % (bound as u64)) as usize;
        value
    }

    /// An in-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for index in (1..slice.len()).rev() {
            let target = self.below(index + 1);
            slice.swap(index, target);
        }
    }
}

// ── ClickSimulationModel ─────────────────────────────────────────────────────

/// Which generative click process to draw from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickSimulationModel {
    /// The position-based model: `C = Bernoulli(γ_r) ∧ Bernoulli(α_d)`, each
    /// position independent of the others. Uses
    /// [`ClickLogSimulator::with_examination`] and
    /// [`ClickLogSimulator::with_attractiveness`].
    PositionBased,
    /// The cascade model: scan top-down, click the first attractive document,
    /// stop. Uses [`ClickLogSimulator::with_attractiveness`] only — examination
    /// is a consequence of the ranking, not a parameter.
    Cascade,
    /// The dynamic Bayesian network: click on attractiveness, leave on
    /// satisfaction, continue with persistence. Uses
    /// [`ClickLogSimulator::with_attractiveness`],
    /// [`ClickLogSimulator::with_satisfaction`] and
    /// [`ClickLogSimulator::with_persistence`].
    Dbn,
}

// ── ClickLogSimulator ────────────────────────────────────────────────────────

/// Draws click logs from known ground-truth click-model parameters.
///
/// ```
/// # #[cfg(feature = "click-model")]
/// # {
/// use oxirag::click_model::{ClickLogSimulator, ClickSimulationModel};
///
/// let log = ClickLogSimulator::new(vec!["a".to_owned(), "b".to_owned()])
///     .with_attractiveness(vec![0.9, 0.1])
///     .with_examination(vec![1.0, 0.5])
///     .with_slate_size(2)
///     .with_seed(11)
///     .simulate(ClickSimulationModel::PositionBased, 1_000)
///     .expect("valid parameters");
///
/// assert_eq!(log.len(), 1_000);
/// // The simulator knows the truth, so it records it.
/// assert!(log.sessions()[0].examinations.is_some());
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ClickLogSimulator {
    doc_ids: Vec<String>,
    attractiveness: Vec<f64>,
    satisfaction: Vec<f64>,
    examination: Vec<f64>,
    persistence: f64,
    slate_size: usize,
    query_count: usize,
    seed: u64,
}

impl ClickLogSimulator {
    /// A simulator over `doc_ids`, with every parameter at a neutral default
    /// (attractiveness `0.5`, satisfaction `0.5`, persistence `0.9`, a flat
    /// examination curve of `1.0`) and a slate showing every document.
    #[must_use]
    pub fn new(doc_ids: Vec<String>) -> Self {
        let count = doc_ids.len();
        Self {
            attractiveness: vec![0.5; count],
            satisfaction: vec![0.5; count],
            examination: vec![1.0; count],
            persistence: 0.9,
            slate_size: count,
            query_count: 1,
            doc_ids,
            seed: 0x5EED,
        }
    }

    /// Ground-truth attractiveness `α_d`, one per document, in the order the
    /// document ids were supplied.
    #[must_use]
    pub fn with_attractiveness(mut self, attractiveness: Vec<f64>) -> Self {
        self.attractiveness = attractiveness;
        self
    }

    /// Ground-truth satisfaction `σ_d`, one per document (DBN only).
    #[must_use]
    pub fn with_satisfaction(mut self, satisfaction: Vec<f64>) -> Self {
        self.satisfaction = satisfaction;
        self
    }

    /// Ground-truth examination `γ_r`, one per **rank** (PBM only). Must cover at
    /// least [`ClickLogSimulator::with_slate_size`] ranks.
    #[must_use]
    pub fn with_examination(mut self, examination: Vec<f64>) -> Self {
        self.examination = examination;
        self
    }

    /// Ground-truth global persistence `γ` (DBN only).
    #[must_use]
    pub fn with_persistence(mut self, persistence: f64) -> Self {
        self.persistence = persistence;
        self
    }

    /// How many documents each simulated session shows. Sessions draw a uniform
    /// random subset of that size, in a uniform random order — which is what
    /// makes every `(rank, document)` cell observable and the PBM identifiable.
    #[must_use]
    pub fn with_slate_size(mut self, slate_size: usize) -> Self {
        self.slate_size = slate_size;
        self
    }

    /// How many distinct query ids to cycle through. Purely cosmetic for the
    /// click models (which parameterise attractiveness per document), but it
    /// exercises the per-query layer of
    /// [`crate::click_model::TableClickPolicy`].
    #[must_use]
    pub fn with_query_count(mut self, query_count: usize) -> Self {
        self.query_count = query_count.max(1);
        self
    }

    /// The PRNG seed. The whole simulation is a pure function of it.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The ground-truth examination curve, as a caller would hand it to
    /// [`crate::click_model::PropensityEstimates::new`] — the "perfect
    /// propensity model" against which a fitted one can be judged.
    #[must_use]
    pub fn true_examination(&self) -> &[f64] {
        &self.examination
    }

    /// The ground-truth attractiveness vector.
    #[must_use]
    pub fn true_attractiveness(&self) -> &[f64] {
        &self.attractiveness
    }

    /// The ground-truth satisfaction vector.
    #[must_use]
    pub fn true_satisfaction(&self) -> &[f64] {
        &self.satisfaction
    }

    fn validate(&self, model: ClickSimulationModel) -> ClickModelResult<()> {
        let invalid = |reason: String| ClickModelError::InvalidConfig { reason };
        if self.doc_ids.is_empty() {
            return Err(invalid("simulator needs at least one document".to_owned()));
        }
        if self.slate_size == 0 || self.slate_size > self.doc_ids.len() {
            return Err(invalid(format!(
                "slate_size must lie in 1..={}, got {}",
                self.doc_ids.len(),
                self.slate_size
            )));
        }
        if self.attractiveness.len() != self.doc_ids.len() {
            return Err(invalid(format!(
                "attractiveness has {} entries but there are {} documents",
                self.attractiveness.len(),
                self.doc_ids.len()
            )));
        }
        if matches!(model, ClickSimulationModel::Dbn)
            && self.satisfaction.len() != self.doc_ids.len()
        {
            return Err(invalid(format!(
                "satisfaction has {} entries but there are {} documents",
                self.satisfaction.len(),
                self.doc_ids.len()
            )));
        }
        if matches!(model, ClickSimulationModel::PositionBased)
            && self.examination.len() < self.slate_size
        {
            return Err(invalid(format!(
                "examination curve covers {} ranks but the slate is {} deep",
                self.examination.len(),
                self.slate_size
            )));
        }
        Ok(())
    }

    /// Simulate `session_count` sessions, each showing a uniformly random slate
    /// of [`ClickLogSimulator::with_slate_size`] documents in uniformly random
    /// order.
    ///
    /// The random ordering is what makes the resulting log **identifiable**:
    /// every document is seen at every rank, so the bipartite `rank ↔ document`
    /// graph is connected and the PBM's one free scale is pinned by the `γ₀ = 1`
    /// anchor. A log whose logging policy always shows the same document at the
    /// same rank cannot identify propensities *at all* — see
    /// [`crate::click_model::PbmIdentifiability`], and see
    /// [`ClickLogSimulator::simulate_rankings`] for the realistic middle ground.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::InvalidConfig`] if the parameter vectors do not match
    /// the document set, or if the examination curve is shallower than the slate.
    pub fn simulate(
        &self,
        model: ClickSimulationModel,
        session_count: usize,
    ) -> ClickModelResult<ClickLog> {
        self.validate(model)?;
        let mut rng = ClickSplitMix64::new(self.seed);
        let mut sessions = Vec::with_capacity(session_count);
        let mut pool: Vec<usize> = (0..self.doc_ids.len()).collect();

        for session_index in 0..session_count {
            rng.shuffle(&mut pool);
            let slate: Vec<usize> = pool[..self.slate_size].to_vec();
            sessions.push(self.simulate_session(model, session_index, &slate, &mut rng));
        }
        ClickLog::from_sessions(sessions)
    }

    /// Simulate one session per supplied ranking, in order.
    ///
    /// This is how you build a log with a *realistic* logging policy: mostly one
    /// fixed ranking (the production ordering, which is where the position bias
    /// comes from) plus a slice of randomised traffic (the swap intervention that
    /// makes propensities estimable at all). Real unbiased-LTR deployments do
    /// exactly this, and for exactly this reason.
    ///
    /// # Errors
    ///
    /// [`ClickModelError::InvalidConfig`] as for [`ClickLogSimulator::simulate`],
    /// and [`ClickModelError::UnknownDocument`] if a ranking names a document the
    /// simulator does not know.
    pub fn simulate_rankings(
        &self,
        model: ClickSimulationModel,
        rankings: &[Vec<String>],
    ) -> ClickModelResult<ClickLog> {
        self.validate(model)?;
        let mut rng = ClickSplitMix64::new(self.seed);
        let mut sessions = Vec::with_capacity(rankings.len());

        for (session_index, ranking) in rankings.iter().enumerate() {
            let slate: Vec<usize> = ranking
                .iter()
                .map(|doc_id| {
                    self.doc_ids
                        .iter()
                        .position(|known| known == doc_id)
                        .ok_or_else(|| ClickModelError::UnknownDocument {
                            doc_id: doc_id.clone(),
                        })
                })
                .collect::<ClickModelResult<Vec<usize>>>()?;
            if matches!(model, ClickSimulationModel::PositionBased)
                && slate.len() > self.examination.len()
            {
                return Err(ClickModelError::InvalidConfig {
                    reason: format!(
                        "examination curve covers {} ranks but ranking {session_index} is {} deep",
                        self.examination.len(),
                        slate.len()
                    ),
                });
            }
            sessions.push(self.simulate_session(model, session_index, &slate, &mut rng));
        }
        ClickLog::from_sessions(sessions)
    }

    fn simulate_session(
        &self,
        model: ClickSimulationModel,
        session_index: usize,
        slate: &[usize],
        rng: &mut ClickSplitMix64,
    ) -> ClickSession {
        let ranked_doc_ids: Vec<String> =
            slate.iter().map(|&doc| self.doc_ids[doc].clone()).collect();
        let query_id = format!("q{}", session_index % self.query_count);
        let (clicks, examinations) = match model {
            ClickSimulationModel::PositionBased => self.simulate_position_based(slate, rng),
            ClickSimulationModel::Cascade => self.simulate_cascade(slate, rng),
            ClickSimulationModel::Dbn => self.simulate_dbn(slate, rng),
        };
        ClickSession::new(query_id, ranked_doc_ids, clicks).with_examinations(examinations)
    }

    /// `E ~ Bern(γ_r)`, `A ~ Bern(α_d)`, `C = E ∧ A` — every position drawn
    /// independently of every other.
    fn simulate_position_based(
        &self,
        slate: &[usize],
        rng: &mut ClickSplitMix64,
    ) -> (Vec<bool>, Vec<bool>) {
        let mut clicks = Vec::with_capacity(slate.len());
        let mut examinations = Vec::with_capacity(slate.len());
        for (rank, &doc) in slate.iter().enumerate() {
            let examined = rng.bernoulli(self.examination[rank]);
            let attractive = rng.bernoulli(self.attractiveness[doc]);
            examinations.push(examined);
            clicks.push(examined && attractive);
        }
        (clicks, examinations)
    }

    /// Scan top-down; the first attractive document is clicked and the session
    /// ends there. Everything below the click is unexamined *by construction*.
    fn simulate_cascade(
        &self,
        slate: &[usize],
        rng: &mut ClickSplitMix64,
    ) -> (Vec<bool>, Vec<bool>) {
        let mut clicks = vec![false; slate.len()];
        let mut examinations = vec![false; slate.len()];
        for (rank, &doc) in slate.iter().enumerate() {
            examinations[rank] = true;
            if rng.bernoulli(self.attractiveness[doc]) {
                clicks[rank] = true;
                break;
            }
        }
        (clicks, examinations)
    }

    /// Attractiveness decides the click, satisfaction decides whether the user
    /// leaves, persistence decides whether an unsatisfied user carries on.
    fn simulate_dbn(&self, slate: &[usize], rng: &mut ClickSplitMix64) -> (Vec<bool>, Vec<bool>) {
        let mut clicks = vec![false; slate.len()];
        let mut examinations = vec![false; slate.len()];
        for (rank, &doc) in slate.iter().enumerate() {
            examinations[rank] = true;
            let clicked = rng.bernoulli(self.attractiveness[doc]);
            clicks[rank] = clicked;
            if clicked && rng.bernoulli(self.satisfaction[doc]) {
                break; // satisfied: the search is over.
            }
            if !rng.bernoulli(self.persistence) {
                break; // gave up.
            }
        }
        (clicks, examinations)
    }
}
