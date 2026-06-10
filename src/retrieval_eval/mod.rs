//! Retrieval evaluation — IR ranking metrics: nDCG, MRR, MAP, Precision@k, Recall@k.
pub mod metrics;
#[cfg(test)]
mod tests;
pub mod types;
pub use metrics::{
    average_precision, dcg_at_k, f1_at_k, hit_rate_at_k, mrr, ndcg_at_k, precision_at_k,
    recall_at_k, reciprocal_rank,
};
pub use types::{
    AggregateScores, Qrels, RelevanceJudgment, RetrievalEvalConfig, RetrievalEvalError,
    RetrievalEvaluator, RetrievalScores,
};
