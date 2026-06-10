//! Corrective RAG (CRAG): retrieval-quality grading and corrective re-retrieval loop.
//!
//! Based on Yan et al. 2024 "Corrective Retrieval Augmented Generation".
//!
//! # Architecture
//!
//! | Component | Responsibility |
//! |-----------|----------------|
//! | [`RetrievalGrader`] | Grade each retrieved document's relevance to the query |
//! | [`HeuristicRetrievalGrader`] | Lexical-Jaccard heuristic grader |
//! | [`MockRetrievalGrader`] | Scripted test double |
//! | [`KnowledgeRefiner`] | Decompose → filter → recompose knowledge strips |
//! | [`QueryRefiner`] | Rewrite query for re-retrieval when docs are Incorrect |
//! | [`CorrectiveRagEngine`] | Orchestrates the CRAG loop (grade → action → maybe re-retrieve) |
//!
//! # Quick start
//!
//! ```rust,ignore
//! # #[cfg(feature = "corrective-rag")] {
//! use oxirag::prelude::*;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let engine = CorrectiveRagEngine::new(
//!     MockRetrievalGrader::new(0.9),
//!     CragConfig::default(),
//! );
//! # }
//! # }
//! ```

pub mod engine;
pub mod grader;
pub mod strip;
#[cfg(test)]
mod tests;
pub mod types;

pub use engine::CorrectiveRagEngine;
pub use grader::{HeuristicRetrievalGrader, MockRetrievalGrader, RetrievalGrader};
pub use strip::{KnowledgeRefiner, QueryRefiner};
pub use types::{
    CorrectiveAction, CorrectiveRagError, CragConfig, CragOutput, GradedDocument, KnowledgeStrip,
    RetrievalGrade,
};
