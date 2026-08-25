//! Query expansion and reformulation strategies.

pub mod composite;
pub mod ngram;
pub mod prf;
pub mod reformulator;
pub mod stem;
pub mod synonym;
pub mod types;

pub use composite::CompositeExpander;
pub use ngram::NGramExpander;
pub use prf::PseudoRelevanceFeedback;
pub use reformulator::QueryReformulator;
pub use stem::StemExpander;
pub use synonym::SynonymExpander;
pub use types::{ExpandedQuery, ExpansionConfig, ExpansionMethod, QueryExpander};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
