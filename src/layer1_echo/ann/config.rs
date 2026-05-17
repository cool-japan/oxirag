//! Configuration for the ANN (HNSW) index.

use serde::{Deserialize, Serialize};

use crate::layer1_echo::traits::SimilarityMetric;

/// Configuration for the ANN index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnConfig {
    /// Max connections per node per layer (M parameter).
    pub m: usize,
    /// Max connections for layer 0.
    pub m_max: usize,
    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search.
    pub ef_search: usize,
    /// Level multiplier (typically 1/ln(M)).
    pub ml: f64,
    /// The distance metric to use.
    pub distance_metric: SimilarityMetric,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / 16.0_f64.ln(),
            distance_metric: SimilarityMetric::Cosine,
        }
    }
}

impl AnnConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the M parameter (max connections per node per layer).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set the `M_max` parameter (max connections for layer 0).
    #[must_use]
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Set the `ef_construction` parameter.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the `ef_search` parameter.
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_distance_metric(mut self, metric: SimilarityMetric) -> Self {
        self.distance_metric = metric;
        self
    }
}
